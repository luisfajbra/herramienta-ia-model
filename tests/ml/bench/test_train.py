import json
from dataclasses import replace
from pathlib import Path
import shutil
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.train import load_candidate, train_candidate
from swmm_resilience.ml.contracts import (
    FEATURE_COLUMNS_V17,
    FEATURE_COLUMNS_V21,
    TABULAR_V4_21,
)
from tests.ml.bench.prepared_factory import make_prepared

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
    assert metadata["preprocessing"]["classifier"]["imputer"] == "median"
    assert metadata["preprocessing"]["regressor"]["imputer"] == "median"


@pytest.fixture
def prepared_v21():
    base = make_prepared(
        [
            {"shape_id": "shape_a", "node_id": 1, "factor_mult": 1.0,
             "vol": 0.0, "duracion_horas": 1.0, "tiempo_al_pico_h": 0.5},
            {"shape_id": "shape_b", "node_id": 1, "factor_mult": 2.0,
             "vol": 2.0, "duracion_horas": 1.0, "tiempo_al_pico_h": 0.5},
        ],
        ("LOSO",),
    )
    X = pd.DataFrame(0.0, index=base.X.index, columns=FEATURE_COLUMNS_V21)
    manifest = {
        **base.manifest,
        "contract_id": TABULAR_V4_21.contract_id,
        "feature_contract_sha256": TABULAR_V4_21.descriptor_sha256,
        "feature_columns": list(FEATURE_COLUMNS_V21),
    }
    return replace(base, X=X, manifest=manifest)


def test_v4_21_candidate_records_its_actual_contract_and_feature_order(prepared_v21):
    output_dir = Path("outputs") / f"contract-metadata-test-{uuid4().hex}"
    try:
        artifacts = train_candidate(
            prepared_v21, "xgboost", TINY, TINY, output_dir
        )
        metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
        classifier = joblib.load(artifacts.classifier_path)
        regressor = joblib.load(artifacts.regressor_path)

        assert classifier.n_features_in_ == 21
        assert regressor.n_features_in_ == 21
        assert metadata["feature_contract_id"] == TABULAR_V4_21.contract_id
        assert metadata["feature_contract_sha256"] == TABULAR_V4_21.descriptor_sha256
        assert metadata["ordered_features"] == list(FEATURE_COLUMNS_V21)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_v17_candidate_keeps_its_complete_legacy_contract():
    prepared_v17 = make_prepared(
        [
            {"shape_id": "shape_a", "node_id": 1, "factor_mult": 1.0,
             "vol": 0.0, "duracion_horas": 1.0, "tiempo_al_pico_h": 0.5},
            {"shape_id": "shape_b", "node_id": 1, "factor_mult": 2.0,
             "vol": 2.0, "duracion_horas": 1.0, "tiempo_al_pico_h": 0.5},
        ],
        ("LOSO",),
    )
    output_dir = Path("outputs") / f"contract-metadata-v17-{uuid4().hex}"
    try:
        artifacts = train_candidate(
            prepared_v17, "xgboost", TINY, TINY, output_dir
        )
        metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

        assert metadata["feature_contract_id"] == "tabular_v3_17"
        assert metadata["ordered_features"] == list(FEATURE_COLUMNS_V17)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_prepared_dataset_rejects_v4_21_manifest_with_wrong_contract_hash(prepared_v21):
    with pytest.raises(ValueError, match="feature_contract_sha256"):
        replace(
            prepared_v21,
            manifest={**prepared_v21.manifest, "feature_contract_sha256": "0" * 64},
        )


def test_prepared_dataset_rejects_contractless_v17_manifest():
    prepared_v17 = make_prepared(
        [
            {"shape_id": "shape_a", "node_id": 1, "factor_mult": 1.0,
             "vol": 0.0, "duracion_horas": 1.0, "tiempo_al_pico_h": 0.5},
            {"shape_id": "shape_b", "node_id": 1, "factor_mult": 2.0,
             "vol": 2.0, "duracion_horas": 1.0, "tiempo_al_pico_h": 0.5},
        ],
        ("LOSO",),
    )
    manifest = {"prep_id": prepared_v17.prep_id, "protocols": ["LOSO"]}

    with pytest.raises(ValueError, match="contract metadata"):
        replace(prepared_v17, manifest=manifest)


def test_prepared_dataset_rejects_v4_21_manifest_with_unknown_contract(prepared_v21):
    with pytest.raises(ValueError, match="Unknown feature contract"):
        replace(prepared_v21, manifest={**prepared_v21.manifest, "contract_id": "v99"})


def test_prepared_dataset_does_not_infer_v17_for_v21_features(prepared_v21):
    manifest = dict(prepared_v21.manifest)
    manifest.pop("contract_id")

    with pytest.raises(ValueError, match="contract_id"):
        replace(prepared_v21, manifest=manifest)


def test_prepared_dataset_rejects_feature_order_different_from_contract(prepared_v21):
    manifest = {
        **prepared_v21.manifest,
        "feature_columns": list(reversed(FEATURE_COLUMNS_V21)),
    }
    with pytest.raises(ValueError, match="feature_columns.*contract order"):
        replace(prepared_v21, manifest=manifest)


def test_training_rejects_mutated_feature_order_before_fit(prepared_v21, monkeypatch):
    prepared_v21.X.columns = list(reversed(FEATURE_COLUMNS_V21))

    def unexpected_fit(*args, **kwargs):
        pytest.fail("candidate fitting must not start with a mismatched feature order")

    from swmm_resilience.ml.bench.models import xgboost_family

    monkeypatch.setattr(xgboost_family.Pipeline, "fit", unexpected_fit)
    with pytest.raises(ValueError, match="X.columns.*contract order"):
        train_candidate(
            prepared_v21,
            "xgboost",
            TINY,
            TINY,
            Path("outputs") / f"contract-order-test-{uuid4().hex}",
        )


def test_svm_metadata_records_each_tasks_kernel_mapping(prepared, tmp_path):
    classifier_params = {
        "gamma": 0.05,
        "n_components": 4,
        "alpha": 1e-4,
        "max_iter": 2000,
        "tol": 1e-3,
    }
    regressor_params = {
        "gamma": 0.2,
        "n_components": 6,
        "alpha": 1e-4,
        "epsilon": 0.1,
        "max_iter": 2000,
        "tol": 1e-3,
    }

    artifacts = train_candidate(
        prepared,
        "svm",
        classifier_params,
        regressor_params,
        tmp_path / "candidates",
    )
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    assert metadata["preprocessing"]["classifier"]["kernel_approx"] == {
        "method": "Nystroem",
        "kernel": "rbf",
        "gamma": 0.05,
        "n_components": 4,
    }
    assert metadata["preprocessing"]["regressor"]["kernel_approx"] == {
        "method": "Nystroem",
        "kernel": "rbf",
        "gamma": 0.2,
        "n_components": 6,
    }


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
    with pytest.raises(ValueError, match="no flooded row"):
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
