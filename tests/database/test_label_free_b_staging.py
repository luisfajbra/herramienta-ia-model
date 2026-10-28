import hashlib
import json

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, *, event_nodes=("J1",), continuity=0.5):
    from tests.validation.test_network_admission import _FIXTURE

    root = tmp_path / "frozen-b"
    root.mkdir(parents=True)
    inp = root / "network.inp"
    inp.write_text(_FIXTURE, encoding="utf-8")
    # The event file contains the complete declared inflow-node set and actual
    # L/s values; its identity and hash are frozen in the manifest below.
    event = root / "event.csv"
    event.write_text(
        "node_id,time,value_lps\n" + "".join(
            f"{node},0:00,0\n{node},1:00,1\n" for node in event_nodes
        ), encoding="utf-8",
    )
    records = []
    for index, role in enumerate(("effective_inp", "event_source", "swmm_rpt", "swmm_out")):
        source = {"effective_inp": inp, "event_source": event}.get(role)
        if source is None:
            source = root / f"run-1{'.rpt' if role == 'swmm_rpt' else '.out'}"
            source.write_bytes(role.encode())
        records.append({"path": source.name, "sha256": _sha(source)})
    role_names = ("effective_inp", "event_source", "swmm_rpt", "swmm_out")
    artifacts = {role: item for role, item in zip(role_names, records)}
    artifacts["source_network_inp"] = {"path": inp.name, "sha256": _sha(inp)}
    report = artifacts["swmm_rpt"]["sha256"]
    scenario = {
        "scenario_key": "pulse__f1.00", "shape_id": "pulse", "factor_mult": 1.0,
        "duracion_horas": 1.0, "tiempo_al_pico_h": 1.0,
        "hydraulic_quality": {
            "continuity_error_pct": continuity, "max_continuity_error_pct": 5.0,
            "continuity_limit_rationale": "provisional synthetic fixture",
            "finite_coverage_ok": True, "drainage_state": "drained",
            "initial_condition": {"policy_id": "fixture", "state_sha256": "a" * 64},
            "flooding_report": {"parser": "fixture", "parser_version": "1", "report_sha256": report},
        },
        "artifacts": artifacts,
    }
    protocol = {
        "schema_version": "b_transfer_protocol_v1", "status": "frozen_before_run",
        "network": {"path": inp.name, "sha256": _sha(inp)},
        "expected_scenario_keys": ["pulse__f1.00"],
        "scenarios": [{key: scenario[key] for key in (
            "scenario_key", "shape_id", "factor_mult", "duracion_horas", "tiempo_al_pico_h"
        )}],
    }
    protocol_path = root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    path = root / "frozen_b_manifest.json"
    batch = {
        "schema_version": "frozen_b_transfer_manifest_v1", "status": "complete",
        "protocol_path": protocol_path.name, "protocol_sha256": _sha(protocol_path),
        "runs": [{key: scenario[key] for key in ("scenario_key", "hydraulic_quality", "artifacts")}],
    }
    path.write_text(json.dumps(batch, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    db = connect_managed_database(tmp_path / "b.sqlite3")
    apply_migrations(db)
    return db, root, batch


def test_frozen_b_staging_allocates_run_and_features_before_prediction_without_targets(tmp_path):
    from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free

    conn, root, _ = _fixture(tmp_path)
    staged = stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21")

    assert staged["run_ids"] == [1]
    assert staged["keys"].columns.tolist() == [
        "run_id", "network_id", "scenario_id", "scenario_key", "scenario_kind",
        "node_id", "factor_mult", "shape_id",
    ]
    assert len(staged["keys"]) == len(staged["features"]) > 0
    assert list(staged["features"].columns) == staged["feature_columns"]
    assert conn.execute("SELECT status FROM runs WHERE run_id=1").fetchone()[0] == "COMPLETE"
    assert conn.execute("SELECT eligible FROM run_hydraulic_quality WHERE run_id=1").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM run_source_artifacts WHERE run_id=1").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM node_features_v4_21 WHERE run_id=1").fetchone()[0] == len(staged["keys"])
    assert conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] == 0
    conn.close()


def test_frozen_b_staging_rejects_source_hash_change_and_target_fields(tmp_path):
    from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free

    conn, root, manifest = _fixture(tmp_path)
    # Target-bearing records cannot enter the feature/prediction path.
    manifest["runs"][0]["hydraulic_quality"]["vol_inundacion_m3"] = [0.0]
    (root / "frozen_b_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="target|label"):
        stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21")
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    conn.close()


def test_frozen_b_staging_rejects_event_hash_change_and_incomplete_node_coverage(tmp_path):
    from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free

    conn, root, _ = _fixture(tmp_path)
    (root / "event.csv").write_text("node_id,time,value_lps\nJ1,0:00,0\nJ1,1:00,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21")
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    conn.close()

    conn, root, manifest = _fixture(tmp_path / "coverage")
    (root / "event.csv").write_text("node_id,time,value_lps\nJ2,0:00,0\nJ2,1:00,1\n", encoding="utf-8")
    manifest["runs"][0]["artifacts"]["event_source"]["sha256"] = _sha(root / "event.csv")
    manifest["runs"][0]["artifacts"]["event_source"]["path"] = "event.csv"
    (root / "frozen_b_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="node coverage|Incorrect nodes"):
        stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21")
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    conn.close()


def test_ineligible_run_is_staged_and_reported_without_features(tmp_path):
    from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free

    conn, root, _ = _fixture(tmp_path, continuity=8.0)
    staged = stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21")

    assert staged["run_ids"] == [1]
    assert staged["keys"].empty and staged["features"].empty
    assert staged["manifest"]["run_order"][0]["eligible"] is False
    assert "continuity_limit_exceeded" in staged["manifest"]["run_order"][0]["reason_codes"]
    assert conn.execute("SELECT status FROM runs WHERE run_id=1").fetchone()[0] == "COMPLETE"
    assert conn.execute("SELECT eligible FROM run_hydraulic_quality WHERE run_id=1").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM run_source_artifacts WHERE run_id=1").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM node_features_v4_21 WHERE run_id=1").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] == 0
    conn.close()


def test_frozen_event_timing_must_match_protocol_before_database_writes(tmp_path):
    from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free

    conn, root, batch = _fixture(tmp_path)
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["scenarios"][0]["duracion_horas"] = 2.0
    protocol_path.write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
    batch["protocol_sha256"] = _sha(protocol_path)
    (root / "frozen_b_manifest.json").write_text(json.dumps(batch), encoding="utf-8")

    with pytest.raises(ValueError, match="duration/peak time disagrees"):
        stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21")
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    conn.close()


def test_prediction_boundary_reverifies_files_and_never_accepts_labels(tmp_path, monkeypatch):
    from swmm_resilience.database.label_free_b_staging import (
        predict_staged_transfer, stage_frozen_b_label_free,
    )
    import swmm_resilience.ml.bench.frozen_transfer as transfer

    conn, root, _ = _fixture(tmp_path)
    staged = stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21")
    (root / "event.csv").write_text("node_id,time,value_lps\nJ1,0:00,0\nJ1,1:00,2\n", encoding="utf-8")
    called = []
    monkeypatch.setattr(transfer, "predict_transfer", lambda *args: called.append(args))

    with pytest.raises(ValueError, match="SHA-256"):
        predict_staged_transfer(tmp_path / "transfer", conn, staged)
    assert called == []
    assert conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] == 0
    conn.close()


def test_prediction_rejects_database_scenario_metadata_changed_after_staging(tmp_path, monkeypatch):
    from swmm_resilience.database.label_free_b_staging import (
        predict_staged_transfer, stage_frozen_b_label_free,
    )
    import swmm_resilience.ml.bench.frozen_transfer as transfer

    conn, root, _ = _fixture(tmp_path)
    staged = stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21")
    conn.execute("UPDATE scenarios SET factor_mult=2.0 WHERE scenario_id=1")
    conn.commit()
    called = []
    monkeypatch.setattr(transfer, "predict_transfer", lambda *args: called.append(args))

    with pytest.raises(ValueError, match="scenario metadata"):
        predict_staged_transfer(tmp_path / "transfer", conn, staged)

    assert called == []
    assert conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] == 0
    conn.close()


def test_prediction_boundary_builds_exact_frozen_transfer_key_schema(tmp_path, monkeypatch):
    from swmm_resilience.database.label_free_b_staging import (
        predict_staged_transfer, stage_frozen_b_label_free,
    )
    from swmm_resilience.ml.bench.schemas import KEY_COLUMNS
    import swmm_resilience.ml.bench.frozen_transfer as transfer

    conn, root, _ = _fixture(tmp_path)
    staged = stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21")
    calls = []

    def fake_predict(directory, keys, features, runs):
        assert tuple(keys.columns) == KEY_COLUMNS
        assert len(keys) == len(features) == 2
        assert tuple(features.columns) == tuple(staged["feature_columns"])
        assert all(set(run) == {"network_id", "run_id", "event_series_sha256", "hydraulic_quality"} for run in runs)
        assert not set(keys.columns).intersection({"inunda", "vol_inundacion_m3"})
        calls.append((keys.copy(), features.copy(), runs))

    monkeypatch.setattr(transfer, "predict_transfer", fake_predict)
    predict_staged_transfer(tmp_path / "transfer", conn, staged)

    assert len(calls) == 1
    assert list(calls[0][0]["node_id"]) == ["J1", "J2"]
    assert conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] == 0
    conn.close()


def test_run_ids_follow_protocol_order_not_batch_record_order(tmp_path):
    from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free

    conn, root, batch = _fixture(tmp_path)
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    first_definition = protocol["scenarios"][0]
    second_definition = {**first_definition, "scenario_key": "pulse__f1.5", "factor_mult": 1.5}
    protocol["expected_scenario_keys"] = ["pulse__f1.5", "pulse__f1.00"]
    protocol["scenarios"] = [second_definition, first_definition]
    protocol_path.write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
    batch["protocol_sha256"] = _sha(protocol_path)
    first_result = batch["runs"][0]
    second_rpt = root / "run-2.rpt"
    second_rpt.write_bytes(b"second synthetic report")
    second_result = json.loads(json.dumps(first_result))
    second_result["scenario_key"] = "pulse__f1.5"
    second_result["artifacts"]["swmm_rpt"] = {"path": second_rpt.name, "sha256": _sha(second_rpt)}
    second_result["hydraulic_quality"]["flooding_report"]["report_sha256"] = _sha(second_rpt)
    batch["runs"] = [first_result, second_result]  # intentionally opposite the frozen order
    (root / "frozen_b_manifest.json").write_text(json.dumps(batch), encoding="utf-8")

    staged = stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21")

    assert staged["run_ids"] == [1, 2]
    assert [entry["scenario_key"] for entry in staged["manifest"]["run_order"]] == ["pulse__f1.5", "pulse__f1.00"]
    assert [tuple(row) for row in conn.execute("SELECT run_id,scenario_id FROM runs ORDER BY run_id")] == [(1, 1), (2, 2)]
    conn.close()


def test_staged_registration_reconstructs_plan_from_sqlite_then_predicts(tmp_path, monkeypatch):
    import swmm_resilience.database.label_free_b_staging as staging
    from swmm_resilience.ml.bench import frozen_transfer

    conn, root, _ = _fixture(tmp_path)
    staged = staging.stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21")
    calls = {}
    monkeypatch.setattr(
        "swmm_resilience.ml.bench.train.load_candidate_pair",
        lambda *args: (None, None, {"feature_contract_id": "tabular_v4_21"}),
    )
    def fake_register(*args):
        calls["register"] = args
        return tmp_path / "registered"
    monkeypatch.setattr(frozen_transfer, "register_transfer", fake_register)
    # The wrapper must pass the exact registered directory to its inference boundary.
    def fake_predict(directory, db, result):
        calls["predict"] = (directory, db, result)
    monkeypatch.setattr(staging, "predict_staged_transfer", fake_predict)

    directory = staging.register_and_predict_staged_transfer(
        tmp_path / "candidate", {"feature_contract_version": "tabular_v4_21"},
        [24, 108, 344], tmp_path / "transfer-output", conn, staged,
    )

    assert directory == tmp_path / "registered"
    planned = calls["register"][3]
    assert set(planned[0]) == {
        "network_id", "run_id", "event_series_sha256", "event_series_components",
        "hydraulic_policy", "hydraulic_policy_sha256", "expected_node_ids",
    }
    assert planned[0]["expected_node_ids"] == ["J1", "J2"]
    assert planned[0]["hydraulic_policy_sha256"]
    assert calls["predict"] == (directory, conn, staged)
    conn.close()
