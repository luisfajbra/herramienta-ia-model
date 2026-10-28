import hashlib
import json
import sqlite3

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.hydraulic_quality import record_run_hydraulic_quality
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.run_source_artifacts import (
    register_run_source_artifacts,
    verify_run_source_artifacts,
)


ROLES = (
    "source_network_inp",
    "effective_inp",
    "event_source",
    "swmm_rpt",
    "swmm_out",
)


def _seed_run(conn, run_id=1, *, report_sha256=None, network_hash=None):
    inp = f"network inp {run_id}".encode()
    conn.execute(
        """INSERT INTO networks (network_id, network_sha256, name,
                   source_filename, inp_bytes, flow_units, created_at_utc)
           VALUES (?, ?, 'network', 'net.inp', ?, 'LPS', '2026-10-26T00:00:00Z')""",
        (run_id, network_hash or hashlib.sha256(inp).hexdigest(), inp),
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


def _fixture(tmp_path, *, network_hash=None):
    conn = connect_managed_database(tmp_path / "sources.sqlite3")
    apply_migrations(conn)
    paths = {}
    for role in ROLES:
        path = tmp_path / f"{role}.dat"
        path.write_bytes(f"fixture bytes for {role}".encode())
        paths[role] = path
    network_inp = f"network inp 1".encode()
    paths["source_network_inp"].write_bytes(network_inp)
    _seed_run(
        conn,
        report_sha256=hashlib.sha256(paths["swmm_rpt"].read_bytes()).hexdigest(),
        network_hash=network_hash,
    )
    return conn, paths


def test_register_and_verify_complete_source_artifacts(tmp_path):
    conn, paths = _fixture(tmp_path)

    manifest = register_run_source_artifacts(conn, 1, paths)

    assert set(manifest["artifacts"]) == set(ROLES)
    assert manifest["network_id"] == 1
    assert manifest["scenario_id"] == 1
    payload = {
        "run_id": manifest["run_id"],
        "network_id": manifest["network_id"],
        "scenario_id": manifest["scenario_id"],
        "config_sha256": manifest["config_sha256"],
        "artifacts": [
            {
                key: manifest["artifacts"][role][key]
                for key in ("artifact_role", "resolved_path", "sha256", "byte_size")
            }
            for role in sorted(ROLES)
        ],
    }
    expected_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert manifest["source_manifest_sha256"] == expected_digest
    verified = verify_run_source_artifacts(conn, 1)
    assert verified == manifest
    assert verified["artifacts"]["swmm_rpt"]["sha256"] == hashlib.sha256(
        paths["swmm_rpt"].read_bytes()
    ).hexdigest()
    conn.close()


def test_registration_rejects_missing_role_without_partial_rows(tmp_path):
    conn, paths = _fixture(tmp_path)
    paths.pop("swmm_out")

    with pytest.raises(ValueError, match="role"):
        register_run_source_artifacts(conn, 1, paths)
    assert conn.execute("SELECT COUNT(*) FROM run_source_artifacts").fetchone()[0] == 0
    conn.close()


def test_verification_rejects_changed_file(tmp_path):
    conn, paths = _fixture(tmp_path)
    register_run_source_artifacts(conn, 1, paths)
    paths["event_source"].write_bytes(b"changed")

    with pytest.raises(ValueError, match="event_source"):
        verify_run_source_artifacts(conn, 1)
    conn.close()


def test_registration_rejects_rpt_hash_mismatch(tmp_path):
    conn, paths = _fixture(tmp_path)
    _seed_run(conn, 2, report_sha256="f" * 64)
    paths["source_network_inp"].write_bytes(b"network inp 2")

    with pytest.raises(ValueError, match="RPT"):
        register_run_source_artifacts(conn, 2, paths)
    assert conn.execute(
        "SELECT COUNT(*) FROM run_source_artifacts WHERE run_id=2"
    ).fetchone()[0] == 0
    conn.close()


def test_registration_returns_canonical_manifest_and_rejects_duplicate_registration(tmp_path):
    conn, paths = _fixture(tmp_path)
    manifest = register_run_source_artifacts(conn, 1, paths)

    with pytest.raises(ValueError, match="already"):
        register_run_source_artifacts(conn, 1, paths)
    assert manifest["source_manifest_sha256"]
    conn.close()


@pytest.mark.parametrize("bad_role", ["unrecognized", "effective_inp"])
def test_registration_rejects_unknown_or_duplicate_roles(tmp_path, bad_role):
    conn, paths = _fixture(tmp_path)
    artifacts = list(paths.items())
    artifacts.append((bad_role, paths["effective_inp"]))

    with pytest.raises(ValueError, match="role"):
        register_run_source_artifacts(conn, 1, artifacts)
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
        register_run_source_artifacts(conn, 1, paths)
    assert conn.execute("SELECT COUNT(*) FROM run_source_artifacts").fetchone()[0] == 0
    conn.close()


def test_source_artifact_rows_are_immutable(tmp_path):
    conn, paths = _fixture(tmp_path)
    conn.commit()
    register_run_source_artifacts(conn, 1, paths)
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(
            "UPDATE run_source_artifacts SET byte_size=byte_size+1 WHERE run_id=1"
        )
    conn.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM run_source_artifacts WHERE run_id=1")
    conn.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="cannot be deleted"):
        conn.execute("DELETE FROM runs WHERE run_id=1")
    conn.rollback()
    conn.close()


@pytest.mark.parametrize(
    ("column", "value", "verification_error"),
    [
        ("scenario_id", 2, "identity"),
        ("config_sha256", "f" * 64, "manifest"),
        ("status", "FAILED", "COMPLETE"),
    ],
)
def test_run_identity_config_and_status_are_guarded_and_verified(
    tmp_path, column, value, verification_error
):
    conn, paths = _fixture(tmp_path)
    conn.execute(
        """INSERT INTO scenarios (scenario_id, network_id, scenario_key,
                   scenario_kind, factor_mult, shape_id, duracion_horas,
                   tiempo_al_pico_h, config_json, config_sha256)
           VALUES (2, 1, 'other', 'factor', 1.0, 'other-shape', 1, 0.5,
                   '{}', ?)""",
        ("a" * 64,),
    )
    register_run_source_artifacts(conn, 1, paths)
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="identity/config/status"):
        conn.execute(f"UPDATE runs SET {column}=? WHERE run_id=1", (value,))
    conn.rollback()
    conn.execute("DROP TRIGGER runs_source_artifacts_identity_immutable")
    conn.execute(f"UPDATE runs SET {column}=? WHERE run_id=1", (value,))
    with pytest.raises(ValueError, match=verification_error):
        verify_run_source_artifacts(conn, 1)
    conn.close()


def test_verification_rejects_tampered_manifest_digest(tmp_path):
    conn, paths = _fixture(tmp_path)
    register_run_source_artifacts(conn, 1, paths)
    conn.commit()
    conn.execute("DROP TRIGGER run_source_artifacts_v11_no_update")
    conn.execute(
        "UPDATE run_source_artifacts SET source_manifest_sha256=? WHERE run_id=1 AND artifact_role='event_source'",
        ("f" * 64,),
    )

    with pytest.raises(ValueError, match="manifest"):
        verify_run_source_artifacts(conn, 1)
    conn.close()


def test_source_network_inp_must_match_database_bytes_and_hash(tmp_path):
    conn, paths = _fixture(tmp_path, network_hash="f" * 64)

    with pytest.raises(ValueError, match="network INP"):
        register_run_source_artifacts(conn, 1, paths)
    assert conn.execute("SELECT COUNT(*) FROM run_source_artifacts").fetchone()[0] == 0
    conn.close()


def test_source_network_inp_path_must_match_stored_network_inp(tmp_path):
    conn, paths = _fixture(tmp_path)
    paths["source_network_inp"].write_bytes(b"a different network inp")

    with pytest.raises(ValueError, match="source network INP file hash"):
        register_run_source_artifacts(conn, 1, paths)
    assert conn.execute("SELECT COUNT(*) FROM run_source_artifacts").fetchone()[0] == 0
    conn.close()


def test_verification_rejects_changed_source_network_inp(tmp_path):
    conn, paths = _fixture(tmp_path)
    register_run_source_artifacts(conn, 1, paths)
    paths["source_network_inp"].write_bytes(b"changed network")

    with pytest.raises(ValueError, match="source_network_inp"):
        verify_run_source_artifacts(conn, 1)
    conn.close()
