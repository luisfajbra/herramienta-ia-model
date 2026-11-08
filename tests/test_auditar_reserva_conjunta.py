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
    ("holdout_inputs.parquet", "inputs"), ("mlp_predictions.parquet", "mlp predictions")])
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
