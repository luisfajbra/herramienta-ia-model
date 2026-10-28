import hashlib
import json

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, *, event_nodes=("J1",), continuity=0.5, two_scenarios=False):
    from tests.validation.test_network_admission import _FIXTURE

    root = tmp_path / "frozen-b"
    root.mkdir(parents=True)
    inp = root / "network.inp"
    source_inp = tmp_path / "data" / "networks" / "network.inp"
    source_inp.parent.mkdir(parents=True)
    inp.write_text(_FIXTURE, encoding="utf-8")
    source_inp.write_text(_FIXTURE, encoding="utf-8")
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
            source.write_text(f"CONTINUITY={continuity}" if role == "swmm_rpt" else "synthetic out", encoding="utf-8")
        records.append({"path": source.name, "sha256": _sha(source)})
    role_names = ("effective_inp", "event_source", "swmm_rpt", "swmm_out")
    artifacts = {role: item for role, item in zip(role_names, records)}
    artifacts["source_network_inp"] = {"path": str(source_inp), "sha256": _sha(source_inp)}
    report = artifacts["swmm_rpt"]["sha256"]
    from swmm_resilience.simulation.initial_conditions import build_initial_condition_manifest
    frozen_initial = build_initial_condition_manifest(source_inp)
    scenario = {
        "scenario_key": "pulse__f1.00", "shape_id": "pulse", "factor_mult": 1.0,
        "duracion_horas": 1.0, "tiempo_al_pico_h": 1.0,
        "hydraulic_quality": {
            "continuity_error_pct": continuity, "max_continuity_error_pct": 5.0,
            "continuity_limit_rationale": "provisional synthetic fixture",
            "finite_coverage_ok": True, "drainage_state": "drained",
            "initial_condition": {"policy_id": frozen_initial["policy_id"],
                                  "state_sha256": frozen_initial["state_sha256"]},
            "flooding_report": {"schema_version": "flooding_report_v1", "parser": "fixture",
                                "parser_version": "1", "report_sha256": report},
        },
        "artifacts": artifacts,
    }
    first_definition = {key: scenario[key] for key in (
        "scenario_key", "shape_id", "factor_mult", "duracion_horas", "tiempo_al_pico_h"
    )}
    second_definition = {**first_definition, "scenario_key": "pulse__f1.5", "factor_mult": 1.5}
    definitions = [second_definition, first_definition] if two_scenarios else [first_definition]
    protocol = {
        "schema_version": "b_transfer_protocol_v1", "status": "frozen_before_run",
        # A descriptive network name is permitted alongside the source INP
        # path/hash; it must not be mistaken for a third artifact field.
        "network": {"name": "synthetic external network", "path": str(source_inp), "sha256": _sha(source_inp)},
        "expected_scenario_keys": [item["scenario_key"] for item in definitions],
        "scenarios": definitions,
        "hydraulic_policy": {
            "max_continuity_error_pct": 5.0, "continuity_limit_rationale": "provisional synthetic fixture",
            "drain_down_hours": 6.0, "tail_window_hours": 1.0, "min_report_steps": 2,
            "inflow_tolerance_lps": 0.01, "flooding_tolerance_lps": 0.01,
            "storage_change_tolerance_m3": 0.01, "residual_flow_tolerance_lps": 0.01,
        },
    }
    from swmm_resilience.database.b_transfer_protocol import freeze_b_transfer_protocol
    anchor_path = tmp_path / "protocol-anchor.sqlite3"
    receipt = freeze_b_transfer_protocol(root, protocol, anchor_db_path=anchor_path,
        git_commit="a" * 40, execution_context={"runner": "synthetic-test", "swmm_version": "fixture"})
    protocol_path = root / "protocol.json"
    path = root / "frozen_b_manifest.json"
    batch = {
        "schema_version": "frozen_b_transfer_manifest_v1", "status": "complete",
        "protocol_path": protocol_path.name, "protocol_sha256": _sha(protocol_path),
        "protocol_anchor": receipt,
        "runs": [{key: scenario[key] for key in ("scenario_key", "hydraulic_quality", "artifacts")}],
    }
    if two_scenarios:
        second_rpt = root / "run-2.rpt"
        second_out = root / "run-2.out"
        second_rpt.write_text(f"CONTINUITY={continuity}", encoding="utf-8")
        second_out.write_text("synthetic out 2", encoding="utf-8")
        second = json.loads(json.dumps(batch["runs"][0]))
        second["scenario_key"] = "pulse__f1.5"
        second["artifacts"]["swmm_rpt"] = {"path": second_rpt.name, "sha256": _sha(second_rpt)}
        second["artifacts"]["swmm_out"] = {"path": second_out.name, "sha256": _sha(second_out)}
        second["hydraulic_quality"]["flooding_report"]["report_sha256"] = _sha(second_rpt)
        batch["runs"].append(second)
    path.write_text(json.dumps(batch, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    db = connect_managed_database(tmp_path / "b.sqlite3")
    apply_migrations(db)
    return db, root, batch, anchor_path


@pytest.fixture(autouse=True)
def _synthetic_swmm_qa(monkeypatch):
    import pandas as pd
    import swmm_resilience.database.label_free_b_staging as staging

    def continuity(path):
        return float(path.read_text(encoding="utf-8").split("=", 1)[1])

    def report(path):
        return pd.DataFrame({"node_id": ["J1"], "flooding_volume_m3": [0.0]}), {
            "schema_version": "flooding_report_v1", "parser": "fixture", "parser_version": "1",
            "report_sha256": _sha(path),
        }

    def labels(path, node_ids):
        return _fixture_labels(path, node_ids)

    def observables(_path, drain_down_hours, tail_window_hours):
        return {
            "n_report_steps": 2, "observed_window_hours": tail_window_hours, "has_link_data": True,
            "node_total_inflow_lps": {"J1": [0.0, 0.0], "J2": [0.0, 0.0]},
            "node_flooding_lps": {"J1": [0.0, 0.0], "J2": [0.0, 0.0]},
            "node_volume_m3": {"J1": [0.0, 0.0], "J2": [0.0, 0.0]},
            "link_flow_lps": {"C1": [0.0, 0.0]},
        }

    def drainage(_observables, **kwargs):
        return "drained", {"policy_version": "tail_window_v1", "window_hours": kwargs["window_hours"]}

    monkeypatch.setattr(staging, "read_continuity_error", continuity, raising=False)
    monkeypatch.setattr(staging, "read_flooding_volumes_with_manifest", report, raising=False)
    monkeypatch.setattr(staging, "extract_labels", labels, raising=False)
    monkeypatch.setattr(staging, "read_tail_window_observables", observables, raising=False)
    monkeypatch.setattr(staging, "classify_drainage_state", drainage, raising=False)


def test_frozen_b_staging_allocates_run_and_features_before_prediction_without_targets(tmp_path):
    from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free

    conn, root, _, anchor = _fixture(tmp_path)
    staged = stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21", protocol_anchor_db_path=anchor)

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
    source_link = conn.execute(
        "SELECT resolved_path FROM run_source_artifacts WHERE run_id=1 AND artifact_role='source_network_inp'"
    ).fetchone()[0]
    assert source_link == str(tmp_path / "data" / "networks" / "network.inp")
    assert conn.execute("SELECT COUNT(*) FROM node_features_v4_21 WHERE run_id=1").fetchone()[0] == len(staged["keys"])
    assert conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] == 0
    conn.close()


def test_frozen_b_staging_rejects_source_hash_change_and_target_fields(tmp_path):
    from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free

    conn, root, manifest, anchor = _fixture(tmp_path)
    # Target-bearing records cannot enter the feature/prediction path.
    manifest["runs"][0]["hydraulic_quality"]["vol_inundacion_m3"] = [0.0]
    (root / "frozen_b_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="target|label"):
        stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21", protocol_anchor_db_path=anchor)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    conn.close()


def test_frozen_b_staging_rejects_event_hash_change_and_incomplete_node_coverage(tmp_path):
    from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free

    conn, root, _, anchor = _fixture(tmp_path)
    (root / "event.csv").write_text("node_id,time,value_lps\nJ1,0:00,0\nJ1,1:00,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21", protocol_anchor_db_path=anchor)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    conn.close()

    conn, root, manifest, anchor = _fixture(tmp_path / "coverage")
    (root / "event.csv").write_text("node_id,time,value_lps\nJ2,0:00,0\nJ2,1:00,1\n", encoding="utf-8")
    manifest["runs"][0]["artifacts"]["event_source"]["sha256"] = _sha(root / "event.csv")
    manifest["runs"][0]["artifacts"]["event_source"]["path"] = "event.csv"
    (root / "frozen_b_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="node coverage|Incorrect nodes"):
        stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21", protocol_anchor_db_path=anchor)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    conn.close()


def test_ineligible_run_is_staged_and_reported_without_features(tmp_path):
    from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free

    conn, root, _, anchor = _fixture(tmp_path, continuity=8.0)
    staged = stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21", protocol_anchor_db_path=anchor)

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


def test_editing_protocol_and_rehashing_batch_cannot_replace_pre_run_anchor(tmp_path):
    from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free

    conn, root, batch, anchor = _fixture(tmp_path)
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["scenarios"][0]["duracion_horas"] = 2.0
    protocol_path.write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
    batch["protocol_sha256"] = _sha(protocol_path)
    (root / "frozen_b_manifest.json").write_text(json.dumps(batch), encoding="utf-8")

    with pytest.raises(ValueError, match="pre-run|anchor|receipt"):
        stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21", protocol_anchor_db_path=anchor)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    conn.close()


def test_prediction_boundary_reverifies_files_and_never_accepts_labels(tmp_path, monkeypatch):
    from swmm_resilience.database.label_free_b_staging import (
        predict_staged_transfer, stage_frozen_b_label_free,
    )
    import swmm_resilience.ml.bench.frozen_transfer as transfer

    conn, root, _, anchor = _fixture(tmp_path)
    staged = stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21", protocol_anchor_db_path=anchor)
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

    conn, root, _, anchor = _fixture(tmp_path)
    staged = stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21", protocol_anchor_db_path=anchor)
    conn.execute("UPDATE scenarios SET factor_mult=2.0 WHERE scenario_id=1")
    conn.commit()
    called = []
    monkeypatch.setattr(transfer, "predict_transfer", lambda *args: called.append(args))

    with pytest.raises(ValueError, match="scenario metadata"):
        predict_staged_transfer(tmp_path / "transfer", conn, staged)

    assert called == []
    assert conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] == 0
    conn.close()


def test_staging_rejects_caller_continuity_that_disagrees_with_rpt(tmp_path, monkeypatch):
    import swmm_resilience.database.label_free_b_staging as staging

    conn, root, _, anchor = _fixture(tmp_path, continuity=0.5)
    monkeypatch.setattr(staging, "read_continuity_error", lambda _path: 0.5, raising=False)
    monkeypatch.setattr(staging, "read_flooding_volumes_with_manifest", lambda path: (
        __import__("pandas").DataFrame({"node_id": ["J1"], "flooding_volume_m3": [0.0]}),
        {"schema_version": "flooding_report_v1", "parser": "fixture", "parser_version": "1",
         "report_sha256": _sha(path)},
    ), raising=False)
    monkeypatch.setattr(staging, "extract_labels", lambda path, node_ids: _fixture_labels(path, node_ids), raising=False)
    manifest = json.loads((root / "frozen_b_manifest.json").read_text(encoding="utf-8"))
    manifest["runs"][0]["hydraulic_quality"]["continuity_error_pct"] = 0.0
    (root / "frozen_b_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="continuity_error_pct.*RPT|RPT.*continuity_error_pct"):
        staging.stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21", protocol_anchor_db_path=anchor)

    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    conn.close()


@pytest.mark.parametrize(("field", "value"), [
    ("finite_coverage_ok", False),
    ("drainage_state", "not_drained"),
    ("initial_condition", {"policy_id": "caller", "state_sha256": "0" * 64}),
])
def test_staging_rejects_caller_out_or_inp_qa_that_disagrees_with_artifacts(tmp_path, field, value):
    from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free

    conn, root, manifest, anchor = _fixture(tmp_path)
    manifest["runs"][0]["hydraulic_quality"][field] = value
    (root / "frozen_b_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="declaration.*differs"):
        stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21", protocol_anchor_db_path=anchor)

    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    conn.close()


def _fixture_labels(path, node_ids):
    import pandas as pd

    frame = pd.DataFrame({"node_id": [str(node) for node in node_ids],
                          "vol_inundacion_m3": [0.0] * len(node_ids), "inunda": [0] * len(node_ids)})
    frame.attrs["flooding_report_manifest"] = {
        "schema_version": "flooding_report_v1", "parser": "fixture", "parser_version": "1",
        "report_sha256": _sha(path),
    }
    return frame


def test_prediction_boundary_builds_exact_frozen_transfer_key_schema(tmp_path, monkeypatch):
    from swmm_resilience.database.label_free_b_staging import (
        predict_staged_transfer, stage_frozen_b_label_free,
    )
    from swmm_resilience.ml.bench.schemas import KEY_COLUMNS
    import swmm_resilience.ml.bench.frozen_transfer as transfer

    conn, root, _, anchor = _fixture(tmp_path)
    staged = stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21", protocol_anchor_db_path=anchor)
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


def test_run_ids_follow_frozen_protocol_order_not_batch_record_order(tmp_path):
    from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free

    conn, root, batch, anchor = _fixture(tmp_path, two_scenarios=True)
    batch["runs"].reverse()  # intentionally opposite the order frozen before simulation
    (root / "frozen_b_manifest.json").write_text(json.dumps(batch), encoding="utf-8")

    staged = stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21", protocol_anchor_db_path=anchor)

    assert staged["run_ids"] == [1, 2]
    assert [entry["scenario_key"] for entry in staged["manifest"]["run_order"]] == ["pulse__f1.5", "pulse__f1.00"]
    assert [tuple(row) for row in conn.execute("SELECT run_id,scenario_id FROM runs ORDER BY run_id")] == [(1, 1), (2, 2)]
    conn.close()


def test_staged_registration_reconstructs_plan_from_sqlite_then_predicts(tmp_path, monkeypatch):
    import swmm_resilience.database.label_free_b_staging as staging
    from swmm_resilience.ml.bench import frozen_transfer

    conn, root, _, anchor = _fixture(tmp_path)
    staged = staging.stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21", protocol_anchor_db_path=anchor)
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
