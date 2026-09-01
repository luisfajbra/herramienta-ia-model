from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml import predict
from swmm_resilience.ml import trainer
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17, FeatureContractError


def test_train_models_rejects_dataset_without_flooded_rows(tmp_path, tiny_config_factory, trainer_training_df):
    df = trainer_training_df.copy()
    df["inunda"] = 0
    df["vol_inundacion_m3"] = 0.0

    with pytest.raises(ValueError, match="No hay filas inundadas"):
        trainer.train_models(df, tiny_config_factory(), tmp_path / "models")


def test_train_models_validates_contract_before_model_fit(
    monkeypatch, tmp_path, tiny_config_factory, trainer_training_df
):
    class FailOnFit:
        def __init__(self, model_name):
            self.model_name = model_name

        def fit(self, X, y):
            raise AssertionError(f"{self.model_name} fit ran before contract validation")

    monkeypatch.setattr(
        trainer,
        "make_classifier",
        lambda config, scale_pos_weight: FailOnFit("classifier"),
    )
    monkeypatch.setattr(
        trainer,
        "make_regressor",
        lambda config: FailOnFit("regressor"),
    )
    invalid = trainer_training_df.copy()
    invalid.loc[0, "duracion_horas"] = float("nan")

    with pytest.raises(FeatureContractError, match="duracion_horas"):
        trainer.train_models(invalid, tiny_config_factory(), tmp_path / "models")


def test_train_models_writes_artifacts(tmp_path, tiny_config_factory, trainer_training_df):
    trainer.train_models(trainer_training_df, tiny_config_factory(), tmp_path / "models")

    assert (tmp_path / "models" / "classifier.joblib").exists()
    assert (tmp_path / "models" / "regressor.joblib").exists()
    assert (tmp_path / "models" / "training_inp_hash.txt").exists()


class RecordingRegressor:
    def __init__(self):
        self.fit_y = None
        self.fit_X = None

    def fit(self, X, y):
        self.fit_X = X
        self.fit_y = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


def test_train_models_fits_regressor_in_log_space(monkeypatch, tmp_path, tiny_config_factory, trainer_training_df):
    recorded = RecordingRegressor()
    monkeypatch.setattr(trainer, "make_regressor", lambda config: recorded)

    trainer.train_models(trainer_training_df, tiny_config_factory(), tmp_path / "models")

    flooded = trainer_training_df[trainer_training_df["inunda"] == 1]
    np.testing.assert_allclose(recorded.fit_y, np.log1p(flooded["vol_inundacion_m3"].to_numpy()))


def test_train_models_fits_regressor_on_contract_validated_frame(
    monkeypatch, tmp_path, tiny_config_factory, trainer_training_df
):
    """The regressor must see the same contract-validated, ordered, float64
    feature frame as the classifier — not a raw column slice."""
    recorded = RecordingRegressor()
    monkeypatch.setattr(trainer, "make_regressor", lambda config: recorded)

    df = trainer_training_df.copy()
    # A feature column arrives as integer dtype; the contract coerces to float.
    df["n_nodos_aguas_arriba"] = df["n_nodos_aguas_arriba"].astype(int)

    trainer.train_models(df, tiny_config_factory(), tmp_path / "models")

    X = recorded.fit_X
    assert list(X.columns) == list(FEATURE_COLUMNS_V17)
    assert (X.dtypes == float).all()


class FakeClassifier:
    def predict(self, X):
        assert list(X.columns) == trainer.FEATURE_COLS
        return pd.Series([0, 1])


class FakeRegressor:
    def predict(self, X):
        assert list(X.columns) == trainer.FEATURE_COLS
        return np.log1p([33.0])


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
