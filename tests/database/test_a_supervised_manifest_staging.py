import hashlib
import json
from pathlib import Path
import tempfile
import uuid

import pandas as pd
import pytest
import yaml


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def tmp_path():
    path = Path(tempfile.gettempdir()) / "codex-a-supervised-stage" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def _prepare_a(tmp_path, monkeypatch):
    from tests.database.test_label_free_b_staging import _fixture
    from swmm_resilience.database import label_free_b_staging as bstage
    from swmm_resilience.extraction import labels
    from swmm_resilience.simulation.initial_conditions import build_initial_condition_manifest

    conn, root, manifest, anchor = _fixture(tmp_path, two_scenarios=True)
    from swmm_resilience.database.b_transfer_protocol import freeze_b_transfer_protocol
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["network"]["name"] = "A"
    protocol.pop("initial_condition", None)
    protocol_path.unlink()
    anchor = tmp_path / "protocol-anchor-a.sqlite3"
    receipt = freeze_b_transfer_protocol(
        root, protocol, anchor_db_path=anchor, git_commit="a" * 40,
        execution_context={"runner": "synthetic-test", "network_role": "A", "swmm_version": "fixture"},
    )
    manifest["schema_version"] = "frozen_a_training_manifest_v1"
    manifest["protocol_sha256"] = _sha(protocol_path)
    manifest["protocol_anchor"] = receipt
    rpt_parser = labels.read_node_flooding_summary

    def parse_report(path):
        text = path.read_text(encoding="utf-8")
        flooded = float(next(line.split("=", 1)[1] for line in text.splitlines() if line.startswith("VOLUME=")))
        return pd.DataFrame({"node_id": ["J1"], "flooding_volume_m3": [flooded]})

    monkeypatch.setattr(labels, "read_node_flooding_summary", parse_report)
    run_records = manifest["runs"]
    for index, record in enumerate(run_records):
        rpt = root / record["artifacts"]["swmm_rpt"]["path"]
        volume = 1.5 if index == 0 else 0.5
        continuity = 0.5 if index == 0 else 10.0
        rpt.write_text(
            f"CONTINUITY={continuity}\nNode Flooding Summary\n----\nJ1 0 0 {volume / 1000:.9f}\nOutfall Loading Summary\nVOLUME={volume}\n",
            encoding="utf-8",
        )
        record["artifacts"]["swmm_rpt"]["sha256"] = _sha(rpt)
        report = labels.read_flooding_volumes_with_manifest(rpt)[1]
        record["hydraulic_quality"]["continuity_error_pct"] = continuity
        record["hydraulic_quality"]["flooding_report"] = report

    source_initial = build_initial_condition_manifest(root / "network.inp")

    def derive(*, source_inp, effective_inp, rpt_path, **kwargs):
        report_manifest = labels.read_flooding_volumes_with_manifest(rpt_path)[1]
        continuity = float(rpt_path.read_text(encoding="utf-8").split("CONTINUITY=", 1)[1].splitlines()[0])
        effective_initial = build_initial_condition_manifest(effective_inp)
        return {
            "continuity_error_pct": continuity,
            "finite_coverage_ok": True,
            "drainage_state": "drained",
            "drainage_manifest": {"state": "drained"},
            "initial_condition": {
                "policy_id": source_initial["policy_id"],
                "state_sha256": source_initial["state_sha256"],
                "source_inp_sha256": source_initial["source_inp_sha256"],
                "effective_inp_sha256": effective_initial["source_inp_sha256"],
            },
            "flooding_report": report_manifest,
            "finite_coverage_diagnostics": {},
        }

    monkeypatch.setattr(bstage, "_derive_hydraulic_quality", derive)
    (root / "frozen_a_training_manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    from swmm_resilience.database import a_supervised_manifest_staging as astage
    monkeypatch.setattr(astage, "_approved_a_scenario_keys", lambda _config: protocol["expected_scenario_keys"])
    config = tmp_path / "config.yaml"
    raw_config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    raw_config["network"]["inp_path"] = protocol["network"]["path"]
    config.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    return conn, root, manifest, anchor, config


def test_supervised_stage_persists_targets_only_for_qa_eligible_runs(tmp_path, monkeypatch):
    from swmm_resilience.database import a_supervised_manifest_staging as staging
    from swmm_resilience.database.a_supervised_manifest_staging import stage_frozen_a_supervised

    conn, root, _manifest, anchor, config = _prepare_a(tmp_path, monkeypatch)
    publication_in_transaction = []
    write_manifest = staging._write_manifest_exclusive

    def write_manifest_in_transaction(*args, **kwargs):
        publication_in_transaction.append(conn.in_transaction)
        return write_manifest(*args, **kwargs)

    monkeypatch.setattr(staging, "_write_manifest_exclusive", write_manifest_in_transaction)
    result = stage_frozen_a_supervised(conn, root, protocol_anchor_db_path=anchor, config_path=config)

    assert result["run_ids"] == [1, 2]
    assert result["eligible_run_ids"] == [2]
    assert result["excluded_runs"] == [{"run_id": 1, "scenario_key": "pulse__f1.5",
                                         "reason_codes": ["continuity_limit_exceeded"]}]
    assert conn.execute("SELECT COUNT(*) FROM node_features_v4_21 WHERE run_id=1").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM node_features_v4_21 WHERE run_id=2").fetchone()[0] == 2
    labels = conn.execute(
        "SELECT n.node_id,r.inunda,r.vol_inundacion_m3 FROM node_results r JOIN nodes n USING(node_pk) WHERE r.run_id=2 ORDER BY n.node_id"
    ).fetchall()
    assert [(row[0], row[1], row[2]) for row in labels] == [("J1", 1, 1.5), ("J2", 0, 0.0)]
    assert conn.execute("SELECT COUNT(*) FROM node_results WHERE run_id=1").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM run_source_artifacts WHERE run_id=1").fetchone()[0] == 5
    assert [row[0] for row in conn.execute("SELECT DISTINCT scenario_kind FROM scenarios")] == ["frozen_a_training"]
    sealed = json.loads((root / "a_training_source_manifest.json").read_text(encoding="utf-8"))
    assert sealed["eligible_run_ids"] == [2]
    assert sealed["excluded_runs"][0]["run_id"] == 1
    assert sealed["excluded_runs"][0]["reason_codes"] == ["continuity_limit_exceeded"]
    assert sealed["node_results_rows"] == 2
    assert publication_in_transaction == [True]
    conn.close()


@pytest.mark.parametrize("mutation", ["event", "report", "expected_keys"])
def test_supervised_stage_rejects_mutated_inputs_before_persisting(tmp_path, monkeypatch, mutation):
    from swmm_resilience.database.a_supervised_manifest_staging import stage_frozen_a_supervised

    conn, root, manifest, anchor, config = _prepare_a(tmp_path, monkeypatch)
    if mutation == "event":
        (root / manifest["runs"][0]["artifacts"]["event_source"]["path"]).write_text(
            "node_id,time,value_lps\nJ1,0:00,0\nJ1,1:00,2\n", encoding="utf-8")
    elif mutation == "report":
        path = root / manifest["runs"][0]["artifacts"]["swmm_rpt"]["path"]
        path.write_text(path.read_text(encoding="utf-8") + "MUTATED\n", encoding="utf-8")
    else:
        manifest["runs"].pop()
        (root / "frozen_a_training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        stage_frozen_a_supervised(conn, root, protocol_anchor_db_path=anchor, config_path=config)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] == 0
    assert not (root / "a_training_source_manifest.json").exists()
    conn.close()


def test_final_a_manifest_is_absent_when_label_extraction_fails_after_source_stage(tmp_path, monkeypatch):
    from swmm_resilience.database import a_supervised_manifest_staging as staging
    from swmm_resilience.database.a_supervised_manifest_staging import stage_frozen_a_supervised

    conn, root, _manifest, anchor, config = _prepare_a(tmp_path, monkeypatch)
    monkeypatch.setattr(staging, "extract_labels", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad report")))
    with pytest.raises(ValueError, match="bad report"):
        stage_frozen_a_supervised(conn, root, protocol_anchor_db_path=anchor, config_path=config)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] == 0
    assert not (root / "a_training_source_manifest.json").exists()
    conn.close()


def test_source_only_database_is_rejected_by_training_loader(tmp_path, monkeypatch):
    from swmm_resilience.database import a_supervised_manifest_staging as staging
    from swmm_resilience.database.a_supervised_manifest_staging import stage_frozen_a_supervised
    from swmm_resilience.database.training_queries import load_training_samples
    from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V21, TABULAR_V4_21

    conn, root, _manifest, anchor, config = _prepare_a(tmp_path, monkeypatch)
    monkeypatch.setattr(staging, "extract_labels", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("simulated interruption")))
    with pytest.raises(ValueError, match="simulated interruption"):
        stage_frozen_a_supervised(conn, root, protocol_anchor_db_path=anchor, config_path=config)

    with pytest.raises(ValueError, match="feature/result row count or cardinality mismatch"):
        load_training_samples(
            conn,
            run_ids=[2],
            view_name="training_samples_v4_21",
            feature_columns=FEATURE_COLUMNS_V21,
            feature_contract=TABULAR_V4_21,
            node_features_table="node_features_v4_21",
        )
    assert not (root / "a_training_source_manifest.json").exists()
    conn.close()


def test_failure_to_publish_final_manifest_rolls_back_all_targets(tmp_path, monkeypatch):
    from swmm_resilience.database import a_supervised_manifest_staging as staging
    from swmm_resilience.database.a_supervised_manifest_staging import stage_frozen_a_supervised

    conn, root, _manifest, anchor, config = _prepare_a(tmp_path, monkeypatch)
    monkeypatch.setattr(staging, "_write_manifest_exclusive", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("manifest write failed")))
    with pytest.raises(OSError, match="manifest write failed"):
        stage_frozen_a_supervised(conn, root, protocol_anchor_db_path=anchor, config_path=config)

    assert conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] == 0
    assert not (root / "a_training_source_manifest.json").exists()
    conn.close()


def test_supervised_stage_requires_an_exclusive_fresh_sqlite_database(tmp_path, monkeypatch):
    from swmm_resilience.database.a_supervised_manifest_staging import stage_frozen_a_supervised

    conn, root, _manifest, anchor, config = _prepare_a(tmp_path, monkeypatch)
    conn.execute("CREATE TABLE stale_stage_marker (value TEXT)")
    conn.execute("INSERT INTO stale_stage_marker VALUES ('old data')")
    conn.commit()
    with pytest.raises(ValueError, match="fresh database"):
        stage_frozen_a_supervised(conn, root, protocol_anchor_db_path=anchor, config_path=config)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert not (root / "a_training_source_manifest.json").exists()
    conn.close()


def test_supervised_stage_rejects_network_that_differs_from_configured_a_inp(tmp_path, monkeypatch):
    from swmm_resilience.database.a_supervised_manifest_staging import stage_frozen_a_supervised

    conn, root, _manifest, anchor, config = _prepare_a(tmp_path, monkeypatch)
    raw_config = yaml.safe_load(config.read_text(encoding="utf-8"))
    different_inp = tmp_path / "other-network.inp"
    different_inp.write_text("[TITLE]\nOTHER NETWORK\n", encoding="utf-8")
    raw_config["network"]["inp_path"] = str(different_inp)
    config.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="path/hash differs from config.network.inp_path"):
        stage_frozen_a_supervised(conn, root, protocol_anchor_db_path=anchor, config_path=config)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] == 0
    conn.close()


def test_b_default_stager_rejects_a_manifest_and_keeps_no_database_rows(tmp_path, monkeypatch):
    from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free

    conn, root, _manifest, anchor, _config = _prepare_a(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unsupported or non-frozen B run manifest"):
        stage_frozen_b_label_free(
            conn, root, feature_contract_id="tabular_v4_21",
            protocol_anchor_db_path=anchor, manifest_name="frozen_a_training_manifest.json",
        )
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] == 0
    conn.close()


def test_approved_a_allowlist_is_1000_keys_in_frozen_order():
    from swmm_resilience.config import load_config
    from swmm_resilience.database.a_supervised_manifest_staging import _approved_a_scenario_keys

    keys = _approved_a_scenario_keys(load_config("config.yaml"))
    factors = [f"{0.2 * index:.3f}" for index in range(1, 26)]
    shape_ids = sorted(path.stem for path in load_config("config.yaml").simulation.hydrograph_shapes_dir.glob("*.csv"))
    expected = [f"A:base:f{factor}" for factor in factors]
    expected.extend(f"A:{shape_id}:f{factor}" for shape_id in shape_ids for factor in factors)
    assert len(keys) == 1000
    assert keys == expected
