import json

import joblib
import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.train import load_candidate, train_candidate
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17

TINY = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3, "subsample": 1.0}


@pytest.fixture
def prepared(sql_training_db, tmp_path):
    return prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )


def test_artifacts_are_written(prepared, tmp_path):
    artifacts = train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")

    assert artifacts.classifier_path.exists()
    assert artifacts.regressor_path.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.classifier_path.parent.name == "xgboost"


def test_metadata_records_the_prep_id_and_feature_order(prepared, tmp_path):
    artifacts = train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    assert metadata["prep_id"] == prepared.prep_id
    assert metadata["family"] == "xgboost"
    assert metadata["ordered_features"] == list(FEATURE_COLUMNS_V17)
    assert metadata["target_transform"] == {"regressor": "log1p", "inverse": "expm1"}
    assert len(metadata["classifier_sha256"]) == 64
    assert len(metadata["regressor_sha256"]) == 64
    assert "library_versions" in metadata
    assert metadata["preprocessing"]["imputer"] == "median"


def test_the_saved_classifier_predicts_on_the_contract_features(prepared, tmp_path):
    artifacts = train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")
    classifier = joblib.load(artifacts.classifier_path)

    predictions = classifier.predict(prepared.X.head(3))
    assert len(predictions) == 3
    assert set(np.unique(predictions)) <= {0, 1}


def test_the_regressor_is_trained_only_on_flooded_rows(prepared, tmp_path):
    """Igual que trainer.py: el regresor solo ve inunda == 1."""
    artifacts = train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["regressor_training_rows"] == int(prepared.y_clf.sum())


def test_training_with_no_flooded_rows_raises_clearly(prepared, tmp_path):
    import pandas as pd
    from swmm_resilience.ml.bench.schemas import PreparedDataset

    dry = PreparedDataset(
        prep_id=prepared.prep_id,
        keys=prepared.keys,
        X=prepared.X,
        y_clf=pd.Series([0] * len(prepared.y_clf), name="inunda"),
        y_reg=prepared.y_reg,
        folds=prepared.folds,
        manifest=prepared.manifest,
        quality=prepared.quality,
    )
    with pytest.raises(ValueError, match="ninguna fila inundada"):
        train_candidate(dry, "xgboost", TINY, TINY, tmp_path / "candidates")


def test_load_candidate_roundtrips(prepared, tmp_path):
    written = train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")
    loaded = load_candidate(written.classifier_path.parent)

    assert loaded.family == "xgboost"
    assert loaded.metadata["prep_id"] == prepared.prep_id


def test_classifier_and_regressor_fit_calls_see_the_right_rows(prepared, tmp_path, monkeypatch):
    """Espia lo que realmente llega a Pipeline.fit(), no solo la metadata.

    metadata["regressor_training_rows"] se calcula por separado de la llamada
    real a fit(); un mutante que entrene con todas las filas (o sin log1p) la
    deja intacta. Se parchea Pipeline.fit a nivel de clase (como en el test
    de fuga de datos de evaluate.py) en vez de envolver el objeto devuelto
    por build_regressor/build_classifier: envolverlo en una clase local rompe
    joblib.dump (no es picklable). train_candidate hace exactamente dos
    llamadas de alto nivel a Pipeline.fit, en orden: clasificador primero
    (sobre TODAS las filas), regresor despues (solo inunda == 1, log1p).
    """
    from sklearn.pipeline import Pipeline

    recorded_calls = []
    original_fit = Pipeline.fit

    def spy_fit(self, X, y=None, **kwargs):
        recorded_calls.append((X, y))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", spy_fit)

    train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")

    assert len(recorded_calls) == 2, "se esperaban dos Pipeline.fit: clasificador y regresor"
    clf_X, clf_y = recorded_calls[0]
    reg_X, reg_y = recorded_calls[1]

    # El clasificador debe entrenar sobre TODAS las filas, sin subconjuntos.
    assert len(clf_X) == len(prepared.X)
    pd.testing.assert_frame_equal(
        clf_X.reset_index(drop=True), prepared.X.reset_index(drop=True)
    )
    assert np.asarray(clf_y) == pytest.approx(prepared.y_clf.to_numpy())

    flooded_mask = prepared.y_clf.to_numpy() == 1
    flooded_positions = np.flatnonzero(flooded_mask)

    assert len(reg_X) == int(prepared.y_clf.sum())
    expected_X = prepared.X.iloc[flooded_positions].reset_index(drop=True)
    pd.testing.assert_frame_equal(reg_X.reset_index(drop=True), expected_X)

    expected_y = np.log1p(prepared.y_reg.to_numpy()[flooded_positions])
    assert np.asarray(reg_y) == pytest.approx(expected_y)


def test_scale_pos_weight_is_the_whole_dataset_ratio(prepared, tmp_path, monkeypatch):
    """A diferencia de evaluate.py (fold-local), aqui es correcto usar el dataset entero."""
    from swmm_resilience.ml.bench.models import xgboost_family

    recorded = []
    real_build_classifier = xgboost_family.build_classifier

    def spy_build_classifier(params, scale_pos_weight):
        recorded.append(scale_pos_weight)
        return real_build_classifier(params, scale_pos_weight)

    monkeypatch.setattr(xgboost_family, "build_classifier", spy_build_classifier)

    train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")

    y_clf = prepared.y_clf.to_numpy()
    n_negative = int((y_clf == 0).sum())
    n_positive = int((y_clf == 1).sum())
    expected = n_negative / n_positive

    assert len(recorded) == 1
    assert recorded[0] == pytest.approx(expected)
