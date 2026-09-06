import hashlib
import json

import pytest

from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.promote import promote_candidate
from swmm_resilience.ml.bench.train import train_candidate

TINY = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3, "subsample": 1.0}


@pytest.fixture
def trained(sql_training_db, tmp_path):
    prepared = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )
    artifacts = train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")
    return prepared, artifacts


def test_promote_copies_the_artifacts_to_the_classic_paths(trained, tmp_path):
    """Los siete consumidores aguas abajo leen exactamente estos dos nombres."""
    _, artifacts = trained
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")
    models_dir = tmp_path / "outputs" / "models"

    promote_candidate(artifacts.classifier_path.parent, models_dir, inp_path)

    assert (models_dir / "classifier.joblib").exists()
    assert (models_dir / "regressor.joblib").exists()
    hash_path = models_dir / "training_inp_hash.txt"
    assert hash_path.exists()

    stored = hash_path.read_text(encoding="utf-8").strip()
    expected_md5 = hashlib.md5(inp_path.read_bytes()).hexdigest()
    assert len(stored) == 32, (
        "training_inp_hash.txt debe contener un digest MD5 (32 caracteres hex); "
        f"got {len(stored)} caracteres -- ml/predict.py y el resto de consumidores "
        "comparan esto contra un MD5 recien calculado del .inp."
    )
    assert stored == expected_md5


def test_promotion_record_names_the_family_and_prep_id(trained, tmp_path):
    prepared, artifacts = trained
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")

    record = promote_candidate(
        artifacts.classifier_path.parent, tmp_path / "models", inp_path
    )

    assert record["family"] == "xgboost"
    assert record["prep_id"] == prepared.prep_id
    assert len(record["classifier_sha256"]) == 64


def test_the_promoted_classifier_loads_and_predicts(trained, tmp_path):
    import joblib

    prepared, artifacts = trained
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")
    models_dir = tmp_path / "models"

    promote_candidate(artifacts.classifier_path.parent, models_dir, inp_path)
    classifier = joblib.load(models_dir / "classifier.joblib")

    assert len(classifier.predict(prepared.X.head(2))) == 2


def test_promoting_a_directory_without_metadata_fails_clearly(tmp_path):
    empty = tmp_path / "vacio"
    empty.mkdir()
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="metadata.json"):
        promote_candidate(empty, tmp_path / "models", inp_path)


def test_promotion_record_is_written_next_to_the_models(trained, tmp_path):
    _, artifacts = trained
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\n", encoding="utf-8")
    models_dir = tmp_path / "models"

    promote_candidate(artifacts.classifier_path.parent, models_dir, inp_path)

    record = json.loads((models_dir / "promotion.json").read_text(encoding="utf-8"))
    assert record["family"] == "xgboost"


def test_bench_flags_are_registered_in_the_cli():
    """La Task 15 extrae build_parser() de main.py; sin eso este test falla."""
    import main

    assert hasattr(main, "build_parser"), (
        "main.py debe exponer build_parser() para que el parser sea testeable"
    )
    known = {action.dest for action in main.build_parser()._actions}
    assert {"bench_prepare", "bench_train", "bench_evaluate", "bench_promote", "bench"} <= known
    assert {"models", "prep_id"} <= known
