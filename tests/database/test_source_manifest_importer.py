import hashlib
import json

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.hydraulic_quality import record_run_hydraulic_quality
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.source_manifest_importer import (
    import_frozen_experiment_manifest,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, *, db_runs=1, batch_runs=1, eligible=True):
    root = tmp_path / "experiment"
    root.mkdir(parents=True)
    conn = connect_managed_database(tmp_path / "source.sqlite3")
    apply_migrations(conn)
    inp = root / "network.inp"
    inp.write_bytes(b"[TITLE]\nsynthetic source network\n")
    network_hash = _sha(inp)
    conn.execute(
        """INSERT INTO networks (network_id, network_sha256, name,
                   source_filename, inp_bytes, flow_units, created_at_utc)
           VALUES (1, ?, 'network', 'network.inp', ?, 'LPS', '2026-10-26T00:00:00Z')""",
        (network_hash, inp.read_bytes()),
    )
    for run_id in range(1, db_runs + 1):
        conn.execute(
            """INSERT INTO scenarios (scenario_id, network_id, scenario_key,
                       scenario_kind, factor_mult, shape_id, duracion_horas,
                       tiempo_al_pico_h, config_json, config_sha256)
               VALUES (?, ?, ?, 'factor', 1.0, 'shape_a', 1, 0.5, '{}', ?)""",
            (run_id, 1, "shape_a@1.0", hex(run_id)[2:] * 64),
        )
        conn.execute(
            """INSERT INTO runs (run_id, network_id, scenario_id, status,
                       config_sha256, continuity_error_pct, node_count)
               VALUES (?, ?, ?, 'COMPLETE', ?, 0.5, 1)""",
            (run_id, 1, run_id, "b" * 64),
        )
        rpt = root / "runs" / f"run-{run_id}.rpt"
        rpt.parent.mkdir(parents=True, exist_ok=True)
        rpt.write_bytes(f"report {run_id}".encode())
        record_run_hydraulic_quality(
            conn, run_id=run_id, continuity_error_pct=0.5,
            max_continuity_error_pct=5,
            continuity_limit_rationale="synthetic fixture",
            finite_coverage_ok=eligible, drainage_state="drained",
            initial_condition={"policy_id": "synthetic", "state_sha256": "d" * 64},
            flooding_report_manifest={"parser": "test", "parser_version": "1",
                                      "report_sha256": _sha(rpt)},
        )

    shape_csv = root / "events" / "shape_a.csv"
    shape_csv.parent.mkdir(parents=True, exist_ok=True)
    shape_csv.write_text("t,q\n0,0\n1,1\n", encoding="utf-8")
    records = []
    for run_id in range(1, batch_runs + 1):
        artifacts = {}
        for role, suffix in (("effective_inp", ".inp"), ("event_source", ".csv"),
                             ("swmm_rpt", ".rpt"), ("swmm_out", ".out")):
            path = root / "runs" / f"run-{run_id}{suffix}"
            if role == "swmm_rpt":
                path.write_bytes(f"report {run_id}".encode())
                artifacts[role] = {"path": path.relative_to(root).as_posix(), "sha256": _sha(path)}
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{role} {run_id}".encode())
                artifacts[role] = {"path": path.relative_to(root).as_posix(), "sha256": _sha(path)}
        rpt = artifacts["swmm_rpt"]
        records.append({
            "shape_id": "shape_a", "factor": 1.0,
            "scenario_key": "shape_a@1.0", "scenario_id": "synthetic-run-key",
            "rpt": rpt["path"], "rpt_sha256": rpt["sha256"],
            "eligible_by_software_qa": True, "reason_codes": [],
            "flooding_report": {"report_sha256": rpt["sha256"]},
            "event_series_sha256": artifacts["event_source"]["sha256"],
            "artifacts": artifacts,
        })
    protocol = {
        "protocol_id": "synthetic-frozen-v1", "status": "frozen_before_run",
        "input_inp": "network.inp", "input_inp_sha256": network_hash,
        "shape_csvs": {"shape_a": {"path": "events/shape_a.csv", "sha256": _sha(shape_csv)}},
        "expected_scenarios": batch_runs,
    }
    protocol_path = root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    batch = {"protocol_sha256": _sha(protocol_path), "status": "complete", "results": records}
    (root / "batch_progress.json").write_text(json.dumps(batch, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return conn, root


def test_import_maps_by_network_hash_shape_factor_and_writes_immutable_manifest(tmp_path):
    conn, root = _fixture(tmp_path)

    manifest = import_frozen_experiment_manifest(conn, root)

    assert manifest["protocol_sha256"] == _sha(root / "protocol.json")
    assert manifest["runs"][0]["run_id"] == 1
    assert manifest["runs"][0]["scenario_key"] == "shape_a@1.0"
    assert manifest["link_table_sha256"]
    out = root / manifest["manifest_path"]
    assert out.is_file()
    with pytest.raises(FileExistsError):
        import_frozen_experiment_manifest(conn, root)
    conn.close()


def test_import_rejects_missing_exact_database_mapping(tmp_path):
    conn, root = _fixture(tmp_path, db_runs=0)

    with pytest.raises(ValueError, match="no database run"):
        import_frozen_experiment_manifest(conn, root)
    assert conn.execute("SELECT COUNT(*) FROM run_source_artifacts").fetchone()[0] == 0
    conn.close()


def test_import_rejects_ambiguous_exact_database_mapping(tmp_path):
    conn, root = _fixture(tmp_path, db_runs=2)

    with pytest.raises(ValueError, match="ambiguous"):
        import_frozen_experiment_manifest(conn, root)
    assert conn.execute("SELECT COUNT(*) FROM run_source_artifacts").fetchone()[0] == 0
    conn.close()


def test_import_rejects_noneligible_quality_and_hash_mismatch(tmp_path):
    conn, root = _fixture(tmp_path, eligible=False)
    with pytest.raises(ValueError, match="ineligible"):
        import_frozen_experiment_manifest(conn, root)
    conn.close()

    conn, root = _fixture(tmp_path / "hash")
    batch_path = root / "batch_progress.json"
    batch = json.loads(batch_path.read_text())
    batch["results"][0]["artifacts"]["event_source"]["sha256"] = "0" * 64
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        import_frozen_experiment_manifest(conn, root)
    assert conn.execute("SELECT COUNT(*) FROM run_source_artifacts").fetchone()[0] == 0
    conn.close()


def test_import_rejects_duplicate_record_and_scenario_key_mismatch(tmp_path):
    conn, root = _fixture(tmp_path)
    batch_path = root / "batch_progress.json"
    batch = json.loads(batch_path.read_text())
    batch["results"].append(batch["results"][0])
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["expected_scenarios"] = 2
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    batch["protocol_sha256"] = _sha(protocol_path)
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate batch record"):
        import_frozen_experiment_manifest(conn, root)
    conn.close()

    conn, root = _fixture(tmp_path / "scenario-key")
    batch_path = root / "batch_progress.json"
    batch = json.loads(batch_path.read_text())
    batch["results"][0]["scenario_key"] = "different@1.0"
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    with pytest.raises(ValueError, match="scenario_key mismatch"):
        import_frozen_experiment_manifest(conn, root)
    conn.close()


def test_import_rejects_protocol_seal_and_missing_shape_source_path(tmp_path):
    conn, root = _fixture(tmp_path)
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["shape_csvs"]["shape_a"].pop("path")
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol seal"):
        import_frozen_experiment_manifest(conn, root)
    batch_path = root / "batch_progress.json"
    batch = json.loads(batch_path.read_text())
    batch["protocol_sha256"] = _sha(protocol_path)
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen shape source"):
        import_frozen_experiment_manifest(conn, root)
    conn.close()


def test_manifest_write_failure_removes_partial_output_and_rolls_back_links(tmp_path, monkeypatch):
    from pathlib import Path

    conn, root = _fixture(tmp_path)
    output = root / "experiment_source_manifest.json"
    real_open = Path.open

    class FailedWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, _data):
            raise OSError("synthetic disk write failure")

    def fail_manifest_write(path, *args, **kwargs):
        stream = real_open(path, *args, **kwargs)
        if path == output and args and args[0] == "x":
            return FailedWriter(stream)
        return stream

    monkeypatch.setattr(Path, "open", fail_manifest_write)
    with pytest.raises(OSError, match="synthetic disk write failure"):
        import_frozen_experiment_manifest(conn, root)
    assert not output.exists()
    assert conn.execute("SELECT COUNT(*) FROM run_source_artifacts").fetchone()[0] == 0
    conn.close()
