import json

import joblib
import numpy as np
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
