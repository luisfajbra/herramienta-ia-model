import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from swmm_resilience.ml.bench.promote import (
    ModelProvenanceError,
    assert_promotion_consistent,
    model_provenance,
    verify_promotion,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _models_dir(tmp_path, *, classifier=b"clf", regressor=b"reg", record=True, tamper=False):
    (tmp_path / "classifier.joblib").write_bytes(classifier)
    (tmp_path / "regressor.joblib").write_bytes(regressor)
    if record:
        payload = {
            "family": "xgboost",
            "prep_id": "abc",
            "feature_contract_id": "tabular_v4_21",
            "classifier_sha256": _sha(b"clf"),
            "regressor_sha256": _sha(b"reg"),
        }
        (tmp_path / "promotion.json").write_text(json.dumps(payload), encoding="utf-8")
    if tamper:
        (tmp_path / "classifier.joblib").write_bytes(b"otro modelo")
    return tmp_path


def test_matching_hashes_are_verified(tmp_path):
    result = verify_promotion(_models_dir(tmp_path))
    assert result["status"] == "verified" and result["ok"] is True
    assert result["discrepancies"] == []


def test_a_replaced_model_is_reported_with_both_hashes(tmp_path):
    result = verify_promotion(_models_dir(tmp_path, tamper=True))

    assert result["status"] == "mismatch" and result["ok"] is False
    discrepancy = result["discrepancies"][0]
    assert discrepancy["artifact"] == "classifier.joblib"
    assert discrepancy["expected"] == _sha(b"clf")
    assert discrepancy["actual"] == _sha(b"otro modelo")


def test_a_missing_artifact_is_a_discrepancy(tmp_path):
    models = _models_dir(tmp_path)
    (models / "regressor.joblib").unlink()
    result = verify_promotion(models)
    assert result["status"] == "mismatch"
    assert result["discrepancies"][0]["actual"] is None


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


def test_model_provenance_summarises_the_active_model(tmp_path):
    info = model_provenance(_models_dir(tmp_path))
    assert info["family"] == "xgboost"
    assert info["feature_contract_id"] == "tabular_v4_21"
    assert info["classifier_sha256"] == _sha(b"clf")
    assert info["prep_id"] == "abc"


def test_model_provenance_without_a_record_is_marked_unknown(tmp_path):
    info = model_provenance(_models_dir(tmp_path, record=False))
    assert info["family"] == "desconocida"


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
