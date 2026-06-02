import json
import math

import pandas as pd

from swmm_resilience.ml.evaluator import _nse, evaluate_models
from swmm_resilience.ml.trainer import FEATURE_COLS


def evaluation_df():
    rows = []
    for factor in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for node_idx in range(4):
            row = {col: float(node_idx + 1) for col in FEATURE_COLS}
            row["factor_mult"] = factor
            row["q_pico_nodo"] = factor * (node_idx + 1)
            row["q_pico_acum_escalado"] = factor * (node_idx + 2)
            row["inunda"] = 1 if factor >= 2.0 and node_idx >= 2 else 0
            row["vol_inundacion_m3"] = factor * 10.0 if row["inunda"] else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_nse_zero_variance_contract():
    assert _nse(pd.Series([5.0, 5.0]), pd.Series([5.0, 5.0])) == 1.0
    assert _nse(pd.Series([5.0, 5.0]), pd.Series([4.0, 6.0])) == 0.0


def test_evaluate_models_writes_expected_json_files(tmp_path, tiny_config_factory):
    cfg = tiny_config_factory(algorithm="random_forest")
    cfg.evaluation = type("Eval", (), {"methods": ["LOSO", "GroupKFold5"], "stratify_by_factor": True})()

    results = evaluate_models(evaluation_df(), cfg, tmp_path / "metrics")

    assert "LOSO" in results
    assert "GroupKFold5" in results
    assert (tmp_path / "metrics" / "metrics_classifier.json").exists()
    assert (tmp_path / "metrics" / "metrics_regressor.json").exists()
    assert (tmp_path / "metrics" / "metrics_endtoend.json").exists()
    assert (tmp_path / "metrics" / "metrics_by_factor.json").exists()

    reg_metrics = json.loads((tmp_path / "metrics" / "metrics_regressor.json").read_text(encoding="utf-8"))
    assert "nse" in reg_metrics
    assert math.isfinite(reg_metrics["nse"])
    assert "log_nse" in reg_metrics
    assert math.isfinite(reg_metrics["log_nse"])
    assert reg_metrics["log_nse"] > -10
