from __future__ import annotations

import hashlib
import json

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.hydraulic_quality import record_run_hydraulic_quality


def _seed_run(conn, run_id: int, continuity):
    inp = f"inp-{run_id}".encode()
    conn.execute(
        """INSERT INTO networks (network_id, network_sha256, name, source_filename,
                   inp_bytes, flow_units, created_at_utc)
           VALUES (?, ?, ?, 'network.inp', ?, 'LPS', '2026-10-26T00:00:00Z')""",
        (run_id, hashlib.sha256(inp).hexdigest(), f"network-{run_id}", inp),
    )
    conn.execute(
        """INSERT INTO scenarios (scenario_id, network_id, scenario_key,
                   scenario_kind, factor_mult, shape_id, duracion_horas,
                   tiempo_al_pico_h, config_json, config_sha256)
           VALUES (?, ?, 'base', 'factor', 1.0, 'base', 1, 0.5, '{}', ?)""",
        (run_id, run_id, "a" * 64),
    )
    conn.execute(
        """INSERT INTO runs (run_id, network_id, scenario_id, status,
                   config_sha256, continuity_error_pct, node_count)
           VALUES (?, ?, ?, 'COMPLETE', ?, ?, 1)""",
        (run_id, run_id, run_id, "b" * 64, continuity),
    )


@pytest.fixture
def quality_db(tmp_path):
    conn = connect_managed_database(tmp_path / "quality.sqlite3")
    apply_migrations(conn)
    _seed_run(conn, 1, 2.5)
    _seed_run(conn, 2, None)
    yield conn
    conn.close()


def test_records_eligible_quality_with_canonical_hashes(quality_db):
    record_run_hydraulic_quality(
        quality_db,
        run_id=1,
        continuity_error_pct=2.5,
        max_continuity_error_pct=5.0,
        continuity_limit_rationale="Owner-approved provisional limit; hydraulic justification pending.",
        finite_coverage_ok=True,
        drainage_state="drained",
        initial_condition={"policy_id": "source_inp", "state_sha256": "c" * 64},
        flooding_report_manifest={
            "parser": "strict_text_v1",
            "parser_version": "strict_text_v1",
            "report_sha256": "d" * 64,
        },
    )

    row = quality_db.execute(
        "SELECT policy_sha256, policy_json, eligible, reason_codes_json, manifest_json "
        "FROM run_hydraulic_quality WHERE run_id=1"
    ).fetchone()
    manifest = json.loads(row[4])
    assert row[0] == hashlib.sha256(row[1].encode("utf-8")).hexdigest()
    assert json.loads(row[1])["max_continuity_error_pct"] == 5.0
    assert "justification pending" in json.loads(row[1])["continuity_limit_rationale"]
    assert row[2] == 1
    assert row[3] == "[]"
    assert manifest["flooding_report"]["report_sha256"] == "d" * 64


def test_missing_continuity_is_recorded_unverified_and_ineligible(quality_db):
    result = record_run_hydraulic_quality(
        quality_db,
        run_id=2,
        continuity_error_pct=None,
        max_continuity_error_pct=5.0,
        continuity_limit_rationale="Owner-approved provisional limit; hydraulic justification pending.",
        finite_coverage_ok=False,
        drainage_state="unverified",
        initial_condition={"policy_id": "source_inp"},
        flooding_report_manifest=None,
    )

    assert result["eligible"] is False
    assert "continuity_error_missing" in result["reason_codes"]
    assert "finite_coverage_failed" in result["reason_codes"]
    stored = quality_db.execute(
        "SELECT continuity_error_pct, drainage_state, eligible "
        "FROM run_hydraulic_quality WHERE run_id=2"
    ).fetchone()
    assert tuple(stored) == (None, "unverified", 0)


def test_report_parser_failure_is_persisted_as_an_ineligibility_reason(quality_db):
    result = record_run_hydraulic_quality(
        quality_db,
        run_id=1,
        continuity_error_pct=2.5,
        max_continuity_error_pct=5.0,
        continuity_limit_rationale="Owner-approved provisional limit; hydraulic justification pending.",
        finite_coverage_ok=True,
        drainage_state="drained",
        initial_condition={"policy_id": "source_inp", "state_sha256": "c" * 64},
        flooding_report_manifest={"failure_code": "invalid_flooding_summary"},
    )

    assert result["eligible"] is False
    assert result["reason_codes"] == ["flooding_report_invalid"]


@pytest.mark.parametrize("limit", [-1, float("nan"), float("inf")])
def test_quality_policy_rejects_invalid_continuity_limit(quality_db, limit):
    with pytest.raises(ValueError, match="finite and non-negative"):
        record_run_hydraulic_quality(
            quality_db,
            run_id=1,
            continuity_error_pct=2.5,
            max_continuity_error_pct=limit,
            continuity_limit_rationale="Test policy.",
            finite_coverage_ok=True,
            drainage_state="drained",
            initial_condition={"policy_id": "source_inp"},
            flooding_report_manifest=None,
        )
