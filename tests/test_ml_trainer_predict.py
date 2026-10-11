from types import SimpleNamespace

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml import predict
from swmm_resilience.ml import trainer
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17, FeatureContractError


def _write_promotion_record(models_dir):
    """promotion.json coherente con los joblib del fixture (los crea si faltan)."""
    import hashlib
    import json

    record = {}
    for name, field in (("classifier", "classifier_sha256"), ("regressor", "regressor_sha256")):
        path = models_dir / f"{name}.joblib"
        if not path.exists():
            path.write_bytes(f"stub-{name}".encode())
        record[field] = hashlib.sha256(path.read_bytes()).hexdigest()
    (models_dir / "promotion.json").write_text(json.dumps(record), encoding="utf-8")


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


class _StubModelWithFeatureCount:
    """Stands in for a fitted sklearn Pipeline -- the guard only reads
    n_features_in_, so this doesn't need to be a real fitted model."""

    def __init__(self, n_features_in_):
        self.n_features_in_ = n_features_in_


def test_train_models_refuses_to_overwrite_a_mismatched_contract_model(
    tmp_path, tiny_config_factory, trainer_training_df
):
    """Permanent regression test for CRITICAL-2's redesigned guard: it must
    derive truth from the already-tracked classifier.joblib/regressor.joblib
    themselves (n_features_in_), not from the gitignored promotion.json."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(_StubModelWithFeatureCount(21), models_dir / "classifier.joblib")
    joblib.dump(_StubModelWithFeatureCount(21), models_dir / "regressor.joblib")
    clf_bytes_before = (models_dir / "classifier.joblib").read_bytes()
    reg_bytes_before = (models_dir / "regressor.joblib").read_bytes()

    with pytest.raises(RuntimeError, match=r"21.*17|17.*21"):
        trainer.train_models(trainer_training_df, tiny_config_factory(), models_dir)

    # The guard must fire before any write -- the existing (mismatched)
    # artifacts must be left byte-for-byte untouched.
    assert (models_dir / "classifier.joblib").read_bytes() == clf_bytes_before
    assert (models_dir / "regressor.joblib").read_bytes() == reg_bytes_before


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
    n_features_in_ = len(trainer.FEATURE_COLS)

    def predict(self, X):
        assert list(X.columns) == trainer.FEATURE_COLS
        return pd.Series([0, 1])


class FakeRegressor:
    n_features_in_ = len(trainer.FEATURE_COLS)

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
    _write_promotion_record(models)
    (models / "training_inp_hash.txt").write_text(predict._md5(cfg.network.inp_path), encoding="utf-8")

    # Columns predict_network computes dynamically (compute_dynamic_features +
    # compute_tributary_dynamic_features) must NOT be pre-filled here as if
    # they were static, or the merge in predict_network collides with the
    # real computed columns (pandas suffixes duplicates _x/_y and FEATURE_COLS
    # lookup raises KeyError) -- same reasoning as test_scenario_predict.py's
    # _DYNAMIC_FEATURE_COLS.
    _dynamic_cols = {
        "factor_mult", "q_pico_nodo", "q_pico_acum_escalado",
        "duracion_horas", "tiempo_al_pico_h",
        "volumen_local_integrado_m3", "volumen_tributario_integrado_m3",
        "duracion_efectiva_tributario_h", "tiempo_primer_pico_tributario_h",
        "tiempo_sobre_capacidad_h", "integral_exceso_capacidad_m3",
    }
    static = pd.DataFrame(
        {
            "node_id": ["J1", "J2"],
            "coord_x": [0.0, 1.0],
            "coord_y": [0.0, 1.0],
            **{col: [1.0, 2.0] for col in trainer.FEATURE_COLS if col not in _dynamic_cols},
        }
    )

    def fake_extract_static_features(path):
        return static[["node_id", "coord_x", "coord_y", "elev_fondo", "prof_max"]].copy()

    def fake_compute_topology_features(static_df, inp_path):
        return static.copy()

    def fake_compute_dynamic_features(
        full_df, factor, duracion_horas=0.0, tiempo_al_pico_h=0.0
    ):
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
    monkeypatch.setattr(predict, "base_shape_stats", lambda inp: (3.0, 0.7))
    # Tributary features: no real .inp/graph in this unit test, so stub the
    # two lookups predict_network needs to reconstruct a node_series for
    # compute_tributary_dynamic_features. An empty node_series/graph still
    # exercises the real merge path (unlike a full FEATURE_COLS monkeypatch).
    monkeypatch.setattr(predict, "load_inp", lambda p: {})
    monkeypatch.setattr(predict, "build_network_graph", lambda inp: (nx.DiGraph(), set()))

    result = predict.predict_network(1.0, cfg, models)

    assert result.loc[result["node_id"] == "J1", "inunda_pred"].iloc[0] == 0
    assert result.loc[result["node_id"] == "J1", "vol_pred_m3"].iloc[0] == 0.0
    assert result.loc[result["node_id"] == "J2", "inunda_pred"].iloc[0] == 1
    assert result.loc[result["node_id"] == "J2", "vol_pred_m3"].iloc[0] == pytest.approx(33.0)


def test_predict_network_rejects_a_model_with_the_wrong_feature_count(monkeypatch, tmp_path, tiny_config_factory):
    """Permanent regression test for predict.py's n_features_in_ guard."""
    cfg = tiny_config_factory()
    cfg.simulation = SimpleNamespace(factor_min=0.5, factor_max=2.0)
    models = tmp_path / "models"
    models.mkdir()
    _write_promotion_record(models)
    (models / "training_inp_hash.txt").write_text(predict._md5(cfg.network.inp_path), encoding="utf-8")

    wrong_count = len(predict.FEATURE_COLS) - 1
    mismatched_clf = _StubModelWithFeatureCount(wrong_count)
    mismatched_reg = _StubModelWithFeatureCount(wrong_count)
    monkeypatch.setattr(
        predict.joblib,
        "load",
        lambda path: mismatched_clf if "classifier" in str(path) else mismatched_reg,
    )

    with pytest.raises(RuntimeError, match="features"):
        predict.predict_network(1.0, cfg, models)
