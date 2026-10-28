import hashlib
import sqlite3

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.hydraulic_quality import record_run_hydraulic_quality
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.run_source_artifacts import (
    register_run_source_artifacts,
    verify_run_source_artifacts,
)


ROLES = ("effective_inp", "event_source", "swmm_rpt", "swmm_out")


def _seed_run(conn, run_id=1, *, report_sha256=None):
    inp = f"network inp {run_id}".encode()
    conn.execute(
        """INSERT INTO networks (network_id, network_sha256, name,
                   source_filename, inp_bytes, flow_units, created_at_utc)
           VALUES (?, ?, 'network', 'net.inp', ?, 'LPS', '2026-10-26T00:00:00Z')""",
        (run_id, hashlib.sha256(inp).hexdigest(), inp),
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
           VALUES (?, ?, ?, 'COMPLETE', ?, 0.5, 1)""",
        (run_id, run_id, run_id, "b" * 64),
    )
    report_sha256 = report_sha256 or "c" * 64
    record_run_hydraulic_quality(
        conn,
        run_id=run_id,
        continuity_error_pct=0.5,
        max_continuity_error_pct=1,
        continuity_limit_rationale="test fixture",
        finite_coverage_ok=True,
        drainage_state="drained",
        initial_condition={"policy_id": "dry", "state_sha256": "d" * 64},
        flooding_report_manifest={
            "parser": "test",
            "parser_version": "1",
            "report_sha256": report_sha256,
        },
    )


def _fixture(tmp_path):
    conn = connect_managed_database(tmp_path / "sources.sqlite3")
    apply_migrations(conn)
    paths = {}
    for role in ROLES:
        path = tmp_path / f"{role}.dat"
        path.write_bytes(f"fixture bytes for {role}".encode())
        paths[role] = path
    _seed_run(conn, report_sha256=hashlib.sha256(paths["swmm_rpt"].read_bytes()).hexdigest())
    return conn, paths


def test_register_and_verify_complete_source_artifacts(tmp_path):
    conn, paths = _fixture(tmp_path)

    records = register_run_source_artifacts(conn, 1, paths, "e" * 64)

    assert set(records) == set(ROLES)
    verified = verify_run_source_artifacts(conn, 1)
    assert set(verified) == set(ROLES)
    assert verified["swmm_rpt"]["sha256"] == hashlib.sha256(
        paths["swmm_rpt"].read_bytes()
    ).hexdigest()
    conn.close()


def test_registration_rejects_missing_role_without_partial_rows(tmp_path):
    conn, paths = _fixture(tmp_path)
    paths.pop("swmm_out")

    with pytest.raises(ValueError, match="role"):
        register_run_source_artifacts(conn, 1, paths, "e" * 64)
    assert conn.execute("SELECT COUNT(*) FROM run_source_artifacts").fetchone()[0] == 0
    conn.close()


def test_verification_rejects_changed_file(tmp_path):
    conn, paths = _fixture(tmp_path)
    register_run_source_artifacts(conn, 1, paths, "e" * 64)
    paths["event_source"].write_bytes(b"changed")

    with pytest.raises(ValueError, match="event_source"):
        verify_run_source_artifacts(conn, 1)
    conn.close()


def test_registration_rejects_rpt_hash_mismatch(tmp_path):
    conn, paths = _fixture(tmp_path)
    _seed_run(conn, 2, report_sha256="f" * 64)

    with pytest.raises(ValueError, match="RPT"):
        register_run_source_artifacts(conn, 2, paths, "e" * 64)
    assert conn.execute(
        "SELECT COUNT(*) FROM run_source_artifacts WHERE run_id=2"
    ).fetchone()[0] == 0
    conn.close()


def test_registration_rejects_bad_manifest_hash_and_duplicate_registration(tmp_path):
    conn, paths = _fixture(tmp_path)
    with pytest.raises(ValueError, match="manifest"):
        register_run_source_artifacts(conn, 1, paths, "E" * 64)
    register_run_source_artifacts(conn, 1, paths, "e" * 64)

    with pytest.raises(ValueError, match="already"):
        register_run_source_artifacts(conn, 1, paths, "e" * 64)
    conn.close()


@pytest.mark.parametrize("bad_role", ["unrecognized", "effective_inp"])
def test_registration_rejects_unknown_or_duplicate_roles(tmp_path, bad_role):
    conn, paths = _fixture(tmp_path)
    artifacts = list(paths.items())
    artifacts.append((bad_role, paths["effective_inp"]))

    with pytest.raises(ValueError, match="role"):
        register_run_source_artifacts(conn, 1, artifacts, "e" * 64)
    assert conn.execute("SELECT COUNT(*) FROM run_source_artifacts").fetchone()[0] == 0
    conn.close()


@pytest.mark.parametrize("kind", ["empty", "directory"])
def test_registration_rejects_empty_or_non_file_artifact(tmp_path, kind):
    conn, paths = _fixture(tmp_path)
    invalid_path = tmp_path / "invalid"
    if kind == "empty":
        invalid_path.write_bytes(b"")
    else:
        invalid_path.mkdir()
    paths["event_source"] = invalid_path

    with pytest.raises(ValueError, match="event_source"):
        register_run_source_artifacts(conn, 1, paths, "e" * 64)
    assert conn.execute("SELECT COUNT(*) FROM run_source_artifacts").fetchone()[0] == 0
    conn.close()


def test_source_artifact_rows_are_immutable(tmp_path):
    conn, paths = _fixture(tmp_path)
    conn.commit()
    register_run_source_artifacts(conn, 1, paths, "e" * 64)
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(
            "UPDATE run_source_artifacts SET byte_size=byte_size+1 WHERE run_id=1"
        )
    conn.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM run_source_artifacts WHERE run_id=1")
    conn.rollback()
    conn.close()
