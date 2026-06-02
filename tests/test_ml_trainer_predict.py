from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from swmm_resilience.ml import predict
from swmm_resilience.ml import trainer


def test_train_models_rejects_dataset_without_flooded_rows(tmp_path, tiny_config_factory, trainer_training_df):
    df = trainer_training_df.copy()
    df["inunda"] = 0
    df["vol_inundacion_m3"] = 0.0

    with pytest.raises(ValueError, match="No hay filas inundadas"):
        trainer.train_models(df, tiny_config_factory(), tmp_path / "models")


def test_train_models_writes_artifacts(tmp_path, tiny_config_factory, trainer_training_df):
    trainer.train_models(trainer_training_df, tiny_config_factory(), tmp_path / "models")

    assert (tmp_path / "models" / "classifier.joblib").exists()
    assert (tmp_path / "models" / "regressor.joblib").exists()
    assert (tmp_path / "models" / "training_inp_hash.txt").exists()


class FakeClassifier:
    def predict(self, X):
        assert list(X.columns) == trainer.FEATURE_COLS
        return pd.Series([0, 1])


class FakeRegressor:
    def predict(self, X):
        assert list(X.columns) == trainer.FEATURE_COLS
        return [33.0]


def test_predict_network_uses_dataframe_features(monkeypatch, tmp_path, tiny_config_factory):
    cfg = tiny_config_factory()
    cfg.simulation = SimpleNamespace(factor_min=0.5, factor_max=2.0)
    models = tmp_path / "models"
    models.mkdir()
    joblib.dump(FakeClassifier(), models / "classifier.joblib")
    joblib.dump(FakeRegressor(), models / "regressor.joblib")
    (models / "training_inp_hash.txt").write_text(predict._md5(cfg.network.inp_path), encoding="utf-8")

    static = pd.DataFrame(
        {
            "node_id": ["J1", "J2"],
            "coord_x": [0.0, 1.0],
            "coord_y": [0.0, 1.0],
            **{col: [1.0, 2.0] for col in trainer.FEATURE_COLS if col not in {"factor_mult", "q_pico_nodo", "q_pico_acum_escalado"}},
        }
    )

    def fake_extract_static_features(path):
        return static[["node_id", "coord_x", "coord_y", "elev_fondo", "prof_max"]].copy()

    def fake_compute_topology_features(static_df, inp_path):
        return static.copy()

    def fake_compute_dynamic_features(full_df, factor):
        return pd.DataFrame(
            {
                "node_id": ["J1", "J2"],
                "factor_mult": [factor, factor],
                "q_pico_nodo": [factor, factor * 2],
                "q_pico_acum_escalado": [factor, factor * 2],
            }
        )

    monkeypatch.setattr(predict, "extract_static_features", fake_extract_static_features)
    monkeypatch.setattr(predict, "compute_topology_features", fake_compute_topology_features)
    monkeypatch.setattr(predict, "compute_dynamic_features", fake_compute_dynamic_features)

    result = predict.predict_network(1.0, cfg, models)

    assert result.loc[result["node_id"] == "J1", "inunda_pred"].iloc[0] == 0
    assert result.loc[result["node_id"] == "J1", "vol_pred_m3"].iloc[0] == 0.0
    assert result.loc[result["node_id"] == "J2", "inunda_pred"].iloc[0] == 1
    assert result.loc[result["node_id"] == "J2", "vol_pred_m3"].iloc[0] == pytest.approx(33.0)
