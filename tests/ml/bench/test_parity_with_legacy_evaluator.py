"""Paridad entre el banco nuevo y ml/evaluator.py.

DESECHABLE: se borra junto al stack B en el Plan 3. Existe para autorizar ese
borrado con evidencia, no para vivir en la suite.

Si falla, NO subas la tolerancia. Investiga.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from swmm_resilience.database.training_queries import load_training_frame
from swmm_resilience.ml.bench.evaluate import evaluate_candidate
from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.evaluator import evaluate_models

PARAMS = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1, "subsample": 1.0}
RTOL = 1e-6


def _legacy_config():
    return SimpleNamespace(
        ml=SimpleNamespace(
            classifier=SimpleNamespace(
                algorithm="xgboost", n_estimators=20, max_depth=3,
                learning_rate=0.1, subsample=1.0, scale_pos_weight="auto",
            ),
            regressor=SimpleNamespace(
                algorithm="xgboost", n_estimators=20, max_depth=3,
                learning_rate=0.1, subsample=1.0,
            ),
            use_scaler=False,
        ),
        evaluation=SimpleNamespace(methods=["LOSO"], stratify_by_factor=True),
    )


def _assert_metrics_match(legacy: dict, bench: dict, level: str) -> None:
    assert set(legacy) == set(bench), f"{level}: claves distintas"
    for key, legacy_value in legacy.items():
        bench_value = bench[key]
        if np.isnan(legacy_value):
            assert np.isnan(bench_value), f"{level}.{key}: legacy NaN, banco {bench_value}"
        else:
            assert bench_value == pytest.approx(legacy_value, rel=RTOL), (
                f"{level}.{key}: legacy={legacy_value} banco={bench_value}"
            )


def test_xgboost_candidate_reproduces_the_legacy_loso_metrics(sql_training_db, tmp_path):
    frame = load_training_frame(sql_training_db)
    legacy = evaluate_models(frame, _legacy_config(), tmp_path / "legacy_metrics")["LOSO"]

    prepared = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )
    _, bench = evaluate_candidate(prepared, "xgboost", PARAMS, PARAMS)

    _assert_metrics_match(legacy["classifier"], bench["LOSO"]["classifier"], "classifier")
    _assert_metrics_match(
        legacy["regressor_oracle"], bench["LOSO"]["regressor_oracle"], "regressor_oracle"
    )
    _assert_metrics_match(legacy["end_to_end"], bench["LOSO"]["end_to_end"], "end_to_end")


def test_the_factor_breakdown_also_matches(sql_training_db, tmp_path):
    frame = load_training_frame(sql_training_db)
    legacy = evaluate_models(frame, _legacy_config(), tmp_path / "legacy_metrics")["LOSO"]

    prepared = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )
    _, bench = evaluate_candidate(prepared, "xgboost", PARAMS, PARAMS)

    assert set(legacy["by_factor"]) == set(bench["LOSO"]["by_factor"])
    for factor, legacy_metrics in legacy["by_factor"].items():
        _assert_metrics_match(
            legacy_metrics, bench["LOSO"]["by_factor"][factor], f"by_factor[{factor}]"
        )
