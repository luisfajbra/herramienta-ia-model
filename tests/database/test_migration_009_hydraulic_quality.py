import hashlib
import sqlite3

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations


def _seed_run(conn, run_id, *, continuity=1.25, pinned=False):
    inp = f"network-{run_id}".encode()
    conn.execute(
        """INSERT INTO networks (network_id, network_sha256, name,
                   source_filename, inp_bytes, flow_units, created_at_utc)
           VALUES (?, ?, ?, 'net.inp', ?, 'LPS', '2026-10-26T00:00:00Z')""",
        (run_id, hashlib.sha256(inp).hexdigest(), f"net-{run_id}", inp),
    )
    conn.execute(
        """INSERT INTO scenarios (scenario_id, network_id, scenario_key,
                   scenario_kind, factor_mult, shape_id, duracion_horas,
                   tiempo_al_pico_h, config_json, config_sha256)
           VALUES (?, ?, 'base', 'factor', 1.0, 'shape', 1, 0.5, '{}', ?)""",
        (run_id, run_id, "a" * 64),
    )
    conn.execute(
        """INSERT INTO runs (run_id, network_id, scenario_id, status,
                   config_sha256, continuity_error_pct, node_count)
           VALUES (?, ?, ?, 'COMPLETE', ?, ?, 1)""",
        (run_id, run_id, run_id, "b" * 64, continuity),
    )
    if pinned:
        conn.execute(
            """INSERT INTO training_runs (
                training_run_id, target, feature_contract_id,
                feature_contract_sha256, query_sql, query_params_json,
                included_run_ids_json, grouping_strategy, fold_count,
                random_seed, primary_metric, tie_breakers_json,
                python_version, library_versions_json, status,
                training_query_contract_id, training_query_contract_sha256
            ) VALUES (?, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', ?,
                      'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}',
                      'PENDING', 'training_samples_v17', ?)""",
            (run_id, "c" * 64, f"[{run_id}]", "d" * 64),
        )
        conn.execute(
            "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (?, ?)",
            (run_id, run_id),
        )


def _quality(conn, run_id, *, eligible=1):
    continuity = conn.execute(
        "SELECT continuity_error_pct FROM runs WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    policy = '{"max_continuity_error_pct":5,"continuity_limit_rationale":"test policy"}'
    initial = '{"policy_id":"test_initial","state_sha256":"' + "a" * 64 + '"}'
    manifest = (
        '{"eligible":true,"reason_codes":[],"continuity_error_pct":'
        + str(continuity)
        + ',"finite_coverage_ok":true,"drainage_state":"drained",'
        '"flooding_report":{"parser":"test","parser_version":"1",'
        '"report_sha256":"' + "b" * 64 + '"}}'
    )
    conn.execute(
        """INSERT INTO run_hydraulic_quality (
            run_id, policy_version, policy_json, policy_sha256,
            continuity_error_pct, finite_coverage_ok, drainage_state,
            initial_condition_json, eligible, reason_codes_json,
            manifest_json, created_at_utc
        ) VALUES (?, 'quality-v1', ?, ?, ?, 1, 'drained', ?, ?, '[]',
                  ?, '2026-10-26T00:00:00Z')""",
        (run_id, policy, "e" * 64, continuity, initial, eligible, manifest),
    )


def test_migration_009_keeps_old_pinned_run_quality_unknown(tmp_path):
    conn = connect_managed_database(tmp_path / "quality.sqlite3")
    apply_migrations(conn)
    _seed_run(conn, 1, continuity=None, pinned=True)
    conn.commit()

    assert conn.execute(
        "SELECT COUNT(*) FROM run_hydraulic_quality"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT continuity_error_pct FROM runs WHERE run_id=1"
    ).fetchone()[0] is None
    with pytest.raises(sqlite3.IntegrityError, match="pinned"):
        _quality(conn, 1)
    conn.rollback()
    conn.close()


def test_hydraulic_quality_manifest_is_append_only_and_checks_continuity(tmp_path):
    conn = connect_managed_database(tmp_path / "quality.sqlite3")
    apply_migrations(conn)
    _seed_run(conn, 1)
    _quality(conn, 1)
    _seed_run(conn, 2)
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(
            "UPDATE run_hydraulic_quality SET eligible=0 WHERE run_id=1"
        )
    conn.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM run_hydraulic_quality WHERE run_id=1")
    conn.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO run_hydraulic_quality (
                run_id, policy_version, policy_json, policy_sha256,
                continuity_error_pct, finite_coverage_ok, drainage_state,
                initial_condition_json, eligible, reason_codes_json,
                manifest_json, created_at_utc
            ) VALUES (2, 'quality-v1', '{"max_continuity_error_pct":5}', ?, 0.5, 1, 'drained', '{}', 1,
                      '[]', '{}', '2026-10-26T00:00:00Z')""",
            ("f" * 64,),
        )
    conn.rollback()
    conn.close()


def test_database_rejects_eligible_quality_without_required_provenance(tmp_path):
    conn = connect_managed_database(tmp_path / "quality.sqlite3")
    apply_migrations(conn)
    _seed_run(conn, 1)

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO run_hydraulic_quality (
                run_id, policy_version, policy_json, policy_sha256,
                continuity_error_pct, finite_coverage_ok, drainage_state,
                initial_condition_json, eligible, reason_codes_json,
                manifest_json, created_at_utc
            ) VALUES (1, 'quality-v1', '{\"max_continuity_error_pct\":5}', ?,
                      1.25, 1, 'drained', '{}', 1, '[]', '{}',
                      '2026-10-26T00:00:00Z')""",
            ("e" * 64,),
        )
    conn.rollback()
    conn.close()


def test_null_continuity_can_be_recorded_as_ineligible_unverified_quality(tmp_path):
    conn = connect_managed_database(tmp_path / "quality.sqlite3")
    apply_migrations(conn)
    _seed_run(conn, 1, continuity=None)

    conn.execute(
        """INSERT INTO run_hydraulic_quality (
            run_id, policy_version, policy_json, policy_sha256,
            continuity_error_pct, finite_coverage_ok, drainage_state,
            initial_condition_json, eligible, reason_codes_json,
            manifest_json, created_at_utc
        ) VALUES (1, 'quality-v1', '{"max_continuity_error_pct":5}', ?,
                  NULL, 0, 'unverified', '{}', 0,
                  '["continuity_error_missing"]', '{}',
                  '2026-10-26T00:00:00Z')""",
        ("e" * 64,),
    )

    row = conn.execute(
        """SELECT continuity_error_pct, finite_coverage_ok, drainage_state,
                  eligible, reason_codes_json
           FROM run_hydraulic_quality WHERE run_id=1"""
    ).fetchone()
    assert tuple(row) == (
        None,
        0,
        "unverified",
        0,
        '["continuity_error_missing"]',
    )
    conn.close()
