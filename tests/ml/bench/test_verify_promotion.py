import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from swmm_resilience.ml.bench.promote import (
    ModelProvenanceError,
    assert_promotion_consistent,
    model_provenance,
    verify_promotion,
)
from swmm_resilience.ml.contracts import (
    FEATURE_COLUMNS_V17,
    FEATURE_COLUMNS_V21,
    TABULAR_V3_17,
    TABULAR_V4_21,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _models_dir(tmp_path, *, classifier=b"clf", regressor=b"reg", record=True, tamper=False):
    columns = list(FEATURE_COLUMNS_V21)
    X = pd.DataFrame(np.zeros((4, len(columns))), columns=columns)
    classifier_model = DummyClassifier(strategy="most_frequent").fit(X, [0, 1, 0, 1])
    regressor_model = DummyRegressor().fit(X, [1.0, 2.0, 3.0, 4.0])
    joblib.dump(classifier_model, tmp_path / "classifier.joblib")
    joblib.dump(regressor_model, tmp_path / "regressor.joblib")
    classifier_bytes = (tmp_path / "classifier.joblib").read_bytes()
    regressor_bytes = (tmp_path / "regressor.joblib").read_bytes()
    if record:
        payload = {
            "family": "xgboost",
            "prep_id": "abc",
            "feature_contract_id": "tabular_v4_21",
            "feature_contract_sha256": TABULAR_V4_21.descriptor_sha256,
            "ordered_features": list(FEATURE_COLUMNS_V21),
            "training_inp_md5": "network-md5",
            "classifier_sha256": _sha(classifier_bytes),
            "regressor_sha256": _sha(regressor_bytes),
        }
        (tmp_path / "promotion.json").write_text(json.dumps(payload), encoding="utf-8")
        (tmp_path / "training_inp_hash.txt").write_text("network-md5", encoding="utf-8")
    if tamper:
        (tmp_path / "classifier.joblib").write_bytes(b"otro modelo")
    return tmp_path


def test_matching_hashes_are_verified(tmp_path):
    result = verify_promotion(_models_dir(tmp_path))
    assert result["status"] == "verified" and result["ok"] is True
    assert result["discrepancies"] == []


def test_promotion_copies_and_verifies_complete_source_manifest(tmp_path, sql_training_db):
    from swmm_resilience.ml.bench.preprocess import prepare_dataset
    from swmm_resilience.ml.bench.train import train_candidate
    from swmm_resilience.ml.bench.promote import promote_candidate
    prepared = prepare_dataset(sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.,
                               output_dir=tmp_path / "prepared")
    params = {"n_estimators": 2, "max_depth": 1}
    candidate = train_candidate(prepared, "xgboost", params, params, tmp_path / "candidates")
    inp = tmp_path / "toy.inp"
    inp.write_text("synthetic network", encoding="utf-8")
    target = tmp_path / "active"
    record = promote_candidate(candidate.metadata_path.parent, target, inp)
    assert record["source_manifest"] == prepared.manifest
    assert verify_promotion(target)["ok"]
    (target / "source_manifest.json").write_text("{}", encoding="utf-8")
    assert verify_promotion(target)["status"] == "mismatch"


def test_a_replaced_model_is_reported_with_both_hashes(tmp_path):
    models = _models_dir(tmp_path, tamper=True)
    record = json.loads((models / "promotion.json").read_text(encoding="utf-8"))
    result = verify_promotion(models)

    assert result["status"] == "mismatch" and result["ok"] is False
    discrepancy = result["discrepancies"][0]
    assert discrepancy["artifact"] == "classifier.joblib"
    assert discrepancy["expected"] == record["classifier_sha256"]
    assert discrepancy["actual"] == _sha(b"otro modelo")


def test_a_missing_artifact_is_a_discrepancy(tmp_path):
    models = _models_dir(tmp_path)
    (models / "regressor.joblib").unlink()
    result = verify_promotion(models)
    assert result["status"] == "mismatch"
    assert result["discrepancies"][0]["actual"] is None


def test_a_changed_network_hash_is_reported(tmp_path):
    models = _models_dir(tmp_path)
    (models / "training_inp_hash.txt").write_text("tampered-network", encoding="utf-8")

    with pytest.raises(ModelProvenanceError, match="training_inp_hash.txt"):
        assert_promotion_consistent(models)


def test_v4_promotion_requires_a_bound_training_network_hash(tmp_path):
    models = _models_dir(tmp_path)
    record_path = models / "promotion.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.pop("training_inp_md5")
    record_path.write_text(json.dumps(record), encoding="utf-8")
    (models / "training_inp_hash.txt").unlink()

    result = verify_promotion(models)
    assert result["status"] == "mismatch"
    assert any(
        item["expected"] == "non-empty training_inp_md5 for the promoted network"
        for item in result["discrepancies"]
    )
    with pytest.raises(ModelProvenanceError, match="training_inp_md5"):
        assert_promotion_consistent(models)


def test_v4_promotion_rejects_an_empty_training_network_hash(tmp_path):
    models = _models_dir(tmp_path)
    record_path = models / "promotion.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["training_inp_md5"] = ""
    record_path.write_text(json.dumps(record), encoding="utf-8")
    (models / "training_inp_hash.txt").write_text("", encoding="utf-8")

    result = verify_promotion(models)
    assert result["status"] == "mismatch"
    assert result["discrepancies"][0]["actual"] == ""


def test_no_promotion_record_is_reported_not_raised(tmp_path):
    result = verify_promotion(_models_dir(tmp_path, record=False))
    assert result["status"] == "no_record" and result["ok"] is None


def test_assert_raises_on_mismatch_naming_the_hashes(tmp_path):
    with pytest.raises(ModelProvenanceError, match="classifier.joblib"):
        assert_promotion_consistent(_models_dir(tmp_path, tamper=True))


def test_assert_warns_but_continues_without_a_record(tmp_path):
    with pytest.warns(UserWarning, match="promotion.json"):
        result = assert_promotion_consistent(_models_dir(tmp_path, record=False))
    assert result["status"] == "no_record"


def test_assert_returns_silently_when_consistent(tmp_path):
    assert assert_promotion_consistent(_models_dir(tmp_path))["status"] == "verified"


def test_v4_promotion_requires_descriptor_hash(tmp_path):
    models = _models_dir(tmp_path)
    record_path = models / "promotion.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.pop("feature_contract_sha256")
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ModelProvenanceError, match="feature_contract_sha256"):
        assert_promotion_consistent(models)


def test_promotion_rejects_an_unknown_model_family(tmp_path):
    models = _models_dir(tmp_path)
    record_path = models / "promotion.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["family"] = "unregistered_family"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ModelProvenanceError, match="family"):
        assert_promotion_consistent(models)


def test_malformed_promotion_json_raises_provenance_error(tmp_path):
    models = _models_dir(tmp_path)
    (models / "promotion.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(ModelProvenanceError, match="promotion.json"):
        assert_promotion_consistent(models)


def test_documented_legacy_v17_promotion_remains_valid(tmp_path):
    columns = list(FEATURE_COLUMNS_V17)
    X = pd.DataFrame(np.zeros((4, len(columns))), columns=columns)
    classifier_model = DummyClassifier(strategy="most_frequent").fit(X, [0, 1, 0, 1])
    regressor_model = DummyRegressor().fit(X, [1.0, 2.0, 3.0, 4.0])
    joblib.dump(classifier_model, tmp_path / "classifier.joblib")
    joblib.dump(regressor_model, tmp_path / "regressor.joblib")
    classifier = (tmp_path / "classifier.joblib").read_bytes()
    regressor = (tmp_path / "regressor.joblib").read_bytes()
    record = {
        "family": "xgboost_legacy",
        "feature_contract_id": TABULAR_V3_17.contract_id,
        "ordered_features": list(FEATURE_COLUMNS_V17),
        "classifier_sha256": _sha(classifier),
        "regressor_sha256": _sha(regressor),
        "source": "train_models (legacy, --only-ml / default pipeline)",
    }
    (tmp_path / "promotion.json").write_text(json.dumps(record), encoding="utf-8")

    assert assert_promotion_consistent(tmp_path)["status"] == "verified"


def test_promote_candidate_writes_a_fully_verifiable_pair(tmp_path):
    from swmm_resilience.ml.bench.promote import promote_candidate

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    columns = list(FEATURE_COLUMNS_V21)
    X = pd.DataFrame(np.zeros((4, len(columns))), columns=columns)
    joblib.dump(
        DummyClassifier(strategy="most_frequent").fit(X, [0, 1, 0, 1]),
        candidate / "classifier.joblib",
    )
    joblib.dump(
        DummyRegressor().fit(X, [1.0, 2.0, 3.0, 4.0]),
        candidate / "regressor.joblib",
    )
    record = {
        "family": "xgboost",
        "prep_id": "prep-promote",
        "feature_contract_id": TABULAR_V4_21.contract_id,
        "feature_contract_sha256": TABULAR_V4_21.descriptor_sha256,
        "ordered_features": columns,
        "classifier_sha256": _sha((candidate / "classifier.joblib").read_bytes()),
        "regressor_sha256": _sha((candidate / "regressor.joblib").read_bytes()),
    }
    (candidate / "metadata.json").write_text(json.dumps(record), encoding="utf-8")
    inp = tmp_path / "network.inp"
    inp.write_text("[TITLE]\nnetwork\n", encoding="utf-8")
    models = tmp_path / "models"

    promoted = promote_candidate(candidate, models, inp)
    assert promoted["training_inp_md5"] == hashlib.md5(inp.read_bytes()).hexdigest()
    assert assert_promotion_consistent(models)["status"] == "verified"
    assert promote_candidate(candidate, models, inp)["prep_id"] == "prep-promote"


def test_model_provenance_summarises_the_active_model(tmp_path):
    models = _models_dir(tmp_path)
    info = model_provenance(models)
    record = json.loads((models / "promotion.json").read_text(encoding="utf-8"))
    assert info["family"] == "xgboost"
    assert info["feature_contract_id"] == "tabular_v4_21"
    assert info["classifier_sha256"] == record["classifier_sha256"]
    assert info["prep_id"] == "abc"


def test_model_provenance_without_a_record_is_marked_unknown(tmp_path):
    info = model_provenance(_models_dir(tmp_path, record=False))
    assert info["family"] == "unknown"


def test_predict_network_refuses_models_that_do_not_match_promotion_json(tmp_path):
    from types import SimpleNamespace
    from swmm_resilience.ml.predict import predict_network

    models = _models_dir(tmp_path, tamper=True)
    with pytest.raises(ModelProvenanceError):
        predict_network(1.0, SimpleNamespace(), models)


def test_importing_predict_does_not_load_torch():
    # promote.py arrastra torch (bench.train -> registry); predict.py debe
    # importarlo de forma perezosa para no cargar torch en todo consumidor.
    code = "import sys, swmm_resilience.ml.predict; assert 'torch' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def _artifact_dir(tmp_path):
    """Layout de ``ml/train.py``: manifest.json + <task>_<modelo>.joblib, sin promotion.json."""
    clf = b"classification-bytes"
    reg = b"regression-bytes"
    (tmp_path / "classification_Logistic.joblib").write_bytes(clf)
    (tmp_path / "regression_Ridge.joblib").write_bytes(reg)
    (tmp_path / "unused_other.txt").write_text("x", encoding="utf-8")
    manifest = json.dumps(
        {"latest": {"classification": "Logistic", "regression": "Ridge"}}
    ).encode("utf-8")
    (tmp_path / "manifest.json").write_bytes(manifest)
    return tmp_path, {
        "manifest.json": _sha(manifest),
        "classification_Logistic.joblib": _sha(clf),
        "regression_Ridge.joblib": _sha(reg),
    }


def test_provenance_without_a_record_hashes_the_artifact_files(tmp_path):
    models, expected = _artifact_dir(tmp_path)
    info = model_provenance(models)
    assert info["family"] == "unknown"
    assert info["artifact_sha256"] == expected


def test_provenance_of_a_missing_or_empty_dir_has_no_artifact_hashes(tmp_path):
    assert model_provenance(tmp_path / "nope")["artifact_sha256"] == {}
    assert model_provenance(tmp_path)["artifact_sha256"] == {}


def test_provenance_with_a_record_keeps_its_original_keys(tmp_path):
    info = model_provenance(_models_dir(tmp_path))
    assert "artifact_sha256" not in info


def test_predict_network_rejects_missing_model_provenance(tmp_path):
    from types import SimpleNamespace
    from swmm_resilience.ml.predict import predict_network

    models = _models_dir(tmp_path, record=False)
    # Pasa la verificación (solo avisa) y falla después, al cargar los joblib falsos.
    with pytest.raises(ModelProvenanceError, match="cannot be verified"):
        predict_network(1.0, SimpleNamespace(), models)
