import hashlib
import json

import pandas as pd
import pytest

KEYS = ["network_id", "run_id", "node_id"]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_directory(tmp_path):
    inputs = pd.DataFrame({"network_id": [1], "run_id": [7], "node_id": ["a"], "shape_id": ["s"],
                           "factor_mult": [1.0], "volumen_tributario_integrado_m3": [50.0],
                           "volumen_local_integrado_m3": [5.0]})
    truth = pd.DataFrame({"network_id": [1], "run_id": [7], "node_id": ["a"], "inunda": [1],
                          "vol_inundacion_m3": [10.0]})
    inputs.to_parquet(tmp_path / "holdout_inputs.parquet", index=False)
    truth.to_parquet(tmp_path / "truth.parquet", index=False)
    models = {}
    for family in ("xgboost", "mlp"):
        path = tmp_path / f"{family}_predictions.parquet"
        pd.DataFrame({"network_id": [1], "run_id": [7], "node_id": ["a"], "shape_id": ["s"],
                      "factor_mult": [1.0], "y_prob_clf": [0.9], "y_pred_clf": [1],
                      "volume_routed_m3": [12.0]}).to_parquet(path, index=False)
        models[family] = {"predictions_sha256": _sha(path)}
    (tmp_path / "prediction_seal.json").write_text(json.dumps(
        {"inputs_sha256": _sha(tmp_path / "holdout_inputs.parquet"), "models": models}), encoding="utf-8")
    (tmp_path / "scores.json").write_text(json.dumps(
        {"seal_sha256": _sha(tmp_path / "prediction_seal.json"),
         "truth_sha256": _sha(tmp_path / "truth.parquet")}), encoding="utf-8")
    (tmp_path / "status.json").write_text(json.dumps(
        {"status": "holdout_scored", "scores_sha256": _sha(tmp_path / "scores.json")}), encoding="utf-8")
    return tmp_path


def test_verify_sealed_evaluation_accepts_intact_files(tmp_path):
    from scripts.auditar_reserva_conjunta import verify_sealed_evaluation
    sealed = verify_sealed_evaluation(sealed_directory(tmp_path))
    assert set(sealed) == {"status", "scores", "seal"}


@pytest.mark.parametrize("name, message", [
    ("scores.json", "scores"), ("truth.parquet", "truth"),
    ("holdout_inputs.parquet", "inputs"), ("mlp_predictions.parquet", "mlp predictions"),
    ("prediction_seal.json", "prediction seal"), ("xgboost_predictions.parquet", "xgboost predictions")])
def test_verify_sealed_evaluation_detects_tampering(tmp_path, name, message):
    from scripts.auditar_reserva_conjunta import verify_sealed_evaluation
    directory = sealed_directory(tmp_path)
    with (directory / name).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match=message):
        verify_sealed_evaluation(directory)


def test_assemble_family_frame_joins_one_to_one_and_rejects_partial_coverage(tmp_path):
    from scripts.auditar_reserva_conjunta import assemble_family_frame
    directory = sealed_directory(tmp_path)
    inputs = pd.read_parquet(directory / "holdout_inputs.parquet")
    truth = pd.read_parquet(directory / "truth.parquet")
    predictions = pd.read_parquet(directory / "xgboost_predictions.parquet")
    frame = assemble_family_frame(inputs, truth, predictions)
    assert frame.loc[0, "volume_routed_m3"] == 12.0 and frame.loc[0, "vol_inundacion_m3"] == 10.0
    assert frame.loc[0, "shape_id"] == "s"
    extra = pd.concat([truth, truth.assign(node_id="b")], ignore_index=True)
    with pytest.raises(ValueError, match="same pairs"):
        assemble_family_frame(inputs, extra, predictions)


def _partition(tmp_path, holdout):
    path = tmp_path / "partition.json"
    path.write_text(json.dumps({"run_ids": {"development": [1, 2], "excluded": [], "holdout": holdout}}),
                    encoding="utf-8")
    return path, _sha(path)


def test_verify_partition_matches_seal_accepts_matching_partition(tmp_path):
    from scripts.auditar_reserva_conjunta import verify_partition_matches_seal
    path, digest = _partition(tmp_path, [9, 7, 8])
    seal = {"partition_sha256": digest, "holdout_run_ids": [7, 8, 9]}
    verify_partition_matches_seal(path, digest, seal)


def test_verify_partition_matches_seal_rejects_other_hash(tmp_path):
    from scripts.auditar_reserva_conjunta import verify_partition_matches_seal
    path, digest = _partition(tmp_path, [7, 8])
    seal = {"partition_sha256": "0" * 64, "holdout_run_ids": [7, 8]}
    with pytest.raises(ValueError, match="partition does not match the sealed evaluation"):
        verify_partition_matches_seal(path, digest, seal)


def test_verify_partition_matches_seal_rejects_other_holdout(tmp_path):
    from scripts.auditar_reserva_conjunta import verify_partition_matches_seal
    path, digest = _partition(tmp_path, [7, 8, 10])
    seal = {"partition_sha256": digest, "holdout_run_ids": [7, 8, 9]}
    with pytest.raises(ValueError, match="partition does not match the sealed evaluation"):
        verify_partition_matches_seal(path, digest, seal)


def _family_frame():
    rows = []
    for run, factor, truth, predicted in [(7, 1.0, 10.0, 12.0), (8, 2.0, 20.0, 18.0)]:
        rows.append({"network_id": 1, "run_id": run, "node_id": "a", "shape_id": "s", "factor_mult": factor,
                     "vol_inundacion_m3": truth, "volume_routed_m3": predicted,
                     "volumen_tributario_integrado_m3": 100.0, "inunda": 1,
                     "y_pred_clf": 1, "y_prob_clf": 0.9})
    return pd.DataFrame(rows)


def test_audit_family_rejects_official_macro_that_does_not_reproduce():
    from scripts.auditar_reserva_conjunta import audit_family
    frame = _family_frame()
    continuity = pd.Series({7: 0.1, 8: 0.2})
    with pytest.raises(ValueError, match="reproduce"):
        audit_family(frame, official_macro=999.0, continuity=continuity)
