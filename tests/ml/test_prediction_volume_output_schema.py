from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from swmm_resilience.config import ML_TARGET_CLASSIFICATION, ML_TARGET_REGRESSION
from swmm_resilience.ml import predict_from_inp, predict_tabular


class _RegressionPipeline:
    def __init__(self, values):
        self._values = values

    def predict(self, rows):
        return self._values[: len(rows)]


class _ClassificationPipeline:
    def predict(self, rows):
        return pd.Series([1] * len(rows))

    def predict_proba(self, rows):
        return pd.DataFrame({0: [0.2] * len(rows), 1: [0.8] * len(rows)}).to_numpy()


class _FailingPipeline:
    def predict(self, rows):
        raise AssertionError("stale regression artifacts must be rejected before prediction")


def _artifact(task, target, pipeline):
    return SimpleNamespace(
        task=task,
        model_name=f"{task}-model",
        target=target,
        feature_columns=["inflow_multiplier", "base_inflow_lps"],
        artifact_path=Path(f"{task}.joblib"),
        pipeline=pipeline,
        metadata={},
    )


def _patch_artifact_loader(monkeypatch, module, *, regression_target=ML_TARGET_REGRESSION):
    def load_saved_model_artifact(task, artifacts_dir, model_name=None):
        if task == "regression":
            pipeline = (
                _FailingPipeline()
                if regression_target != ML_TARGET_REGRESSION
                else _RegressionPipeline([4.25, 7.5])
            )
            return _artifact("regression", regression_target, pipeline)
        return _artifact(
            "classification",
            ML_TARGET_CLASSIFICATION,
            _ClassificationPipeline(),
        )

    monkeypatch.setattr(module.train, "load_saved_model_artifact", load_saved_model_artifact)


def test_csv_prediction_output_uses_total_flood_volume_column(monkeypatch):
    monkeypatch.setattr(
        predict_tabular.train,
        "load_dataset",
        lambda path: pd.DataFrame(
            {
                "node_id": ["J1"],
                "inflow_multiplier": [1.0],
                "base_inflow_lps": [2.0],
            }
        ),
    )
    monkeypatch.setattr(
        predict_tabular.train,
        "default_artifact_dir",
        lambda path: Path("artifacts"),
    )
    _patch_artifact_loader(monkeypatch, predict_tabular)

    result = predict_tabular.predict_steady_flows([1.5], dataset_csv=Path("dataset.csv"))

    assert "predicted_total_flood_volume_m3" in result.predictions.columns
    assert "predicted_peak_flooding_lps" not in result.predictions.columns
    assert result.predictions["predicted_total_flood_volume_m3"].tolist() == [4.25]


def test_stale_regression_artifact_target_raises_before_prediction(monkeypatch):
    monkeypatch.setattr(
        predict_tabular.train,
        "load_dataset",
        lambda path: pd.DataFrame(
            {
                "node_id": ["J1"],
                "inflow_multiplier": [1.0],
                "base_inflow_lps": [2.0],
            }
        ),
    )
    monkeypatch.setattr(
        predict_tabular.train,
        "default_artifact_dir",
        lambda path: Path("artifacts"),
    )
    _patch_artifact_loader(monkeypatch, predict_tabular, regression_target="peak_flooding_lps")

    with pytest.raises(ValueError, match="peak_flooding_lps.*total_flood_volume_m3.*retrain"):
        predict_tabular.predict_steady_flows([1.5], dataset_csv=Path("dataset.csv"))


def test_inp_prediction_output_uses_total_flood_volume_column(monkeypatch):
    monkeypatch.setattr(
        predict_from_inp,
        "_prediction_rows_from_inp",
        lambda inp_file, inflow_multipliers, target_nodes: (
            pd.DataFrame(
                {
                    "node_id": ["J1"],
                    "network_hash": ["abc123"],
                    "network_file": ["network.inp"],
                    "base_inflow_lps": [2.0],
                    "inflow_multiplier": [float(inflow_multipliers[0])],
                }
            ),
            "abc123",
        ),
    )
    _patch_artifact_loader(monkeypatch, predict_from_inp)

    result = predict_from_inp.predict_steady_flows_from_inp([1.5], inp_file=Path("network.inp"))

    assert "predicted_total_flood_volume_m3" in result.predictions.columns
    assert "predicted_peak_flooding_lps" not in result.predictions.columns
    assert result.predictions["predicted_total_flood_volume_m3"].tolist() == [4.25]
