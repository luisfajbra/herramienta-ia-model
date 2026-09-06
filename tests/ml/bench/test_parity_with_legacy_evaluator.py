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

# Las cuatro claves de primer nivel que ambos lados deben producir. Se afirma
# como conjunto para que una clave anadida o retirada en el legacy no pase
# inadvertida: los tests leen las sub-dicts por nombre y, sin esta asercion,
# un cambio de superficie seria invisible para la puerta.
TOP_LEVEL_KEYS = {"classifier", "regressor_oracle", "end_to_end", "by_factor"}


def _legacy_config(method: str = "LOSO"):
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
        evaluation=SimpleNamespace(methods=[method], stratify_by_factor=True),
    )


def _run_both_sides(db_path, tmp_path, method: str):
    """Corre el stack legado y el banco sobre la misma base, con el mismo protocolo."""
    frame = load_training_frame(db_path)
    legacy = evaluate_models(frame, _legacy_config(method), tmp_path / "legacy_metrics")[method]

    prepared = prepare_dataset(
        db_path, protocols=(method,), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )
    _, bench = evaluate_candidate(prepared, "xgboost", PARAMS, PARAMS)
    return legacy, bench[method]


def _assert_top_level_shape(legacy: dict, bench: dict) -> None:
    """Las claves de primer nivel coinciden y ninguna sub-dict viene vacia.

    Sin esto la puerta podria 'pasar' comparando diccionarios vacios contra
    diccionarios vacios: tanto _mean_metrics([]) como
    _pooled_regressor_oracle_metrics([]) devuelven {} por diseno.
    """
    assert set(legacy) == TOP_LEVEL_KEYS, (
        f"claves de primer nivel del legacy: {sorted(legacy)}"
    )
    assert set(bench) == TOP_LEVEL_KEYS, (
        f"claves de primer nivel del banco: {sorted(bench)}"
    )
    for level in sorted(TOP_LEVEL_KEYS):
        assert legacy[level], (
            f"legacy[{level!r}] esta vacio: la evaluacion legada no produjo nada "
            "y la paridad no significaria nada."
        )
        assert bench[level], (
            f"banco[{level!r}] esta vacio: la evaluacion del banco no produjo nada "
            "y la paridad no significaria nada."
        )


def _assert_metrics_match(legacy: dict, bench: dict, level: str) -> None:
    assert legacy, f"{level}: dict de metricas legacy vacio, la comparacion seria vacua"
    assert bench, f"{level}: dict de metricas del banco vacio, la comparacion seria vacua"
    assert set(legacy) == set(bench), f"{level}: claves distintas"
    for key, legacy_value in legacy.items():
        bench_value = bench[key]
        if np.isnan(legacy_value):
            assert np.isnan(bench_value), f"{level}.{key}: legacy NaN, banco {bench_value}"
        else:
            assert bench_value == pytest.approx(legacy_value, rel=RTOL), (
                f"{level}.{key}: legacy={legacy_value} banco={bench_value}"
            )


def _assert_by_factor_matches(legacy: dict, bench: dict) -> None:
    assert legacy["by_factor"], "legacy['by_factor'] vacio: no hay desglose que comparar"
    assert set(legacy["by_factor"]) == set(bench["by_factor"])
    for factor, legacy_metrics in legacy["by_factor"].items():
        _assert_metrics_match(
            legacy_metrics, bench["by_factor"][factor], f"by_factor[{factor}]"
        )


def test_xgboost_candidate_reproduces_the_legacy_loso_metrics(sql_training_db, tmp_path):
    legacy, bench = _run_both_sides(sql_training_db, tmp_path, "LOSO")

    _assert_top_level_shape(legacy, bench)
    _assert_metrics_match(legacy["classifier"], bench["classifier"], "classifier")
    _assert_metrics_match(
        legacy["regressor_oracle"], bench["regressor_oracle"], "regressor_oracle"
    )
    _assert_metrics_match(legacy["end_to_end"], bench["end_to_end"], "end_to_end")


def test_the_factor_breakdown_also_matches(sql_training_db, tmp_path):
    legacy, bench = _run_both_sides(sql_training_db, tmp_path, "LOSO")

    _assert_top_level_shape(legacy, bench)
    _assert_by_factor_matches(legacy, bench)


def test_groupkfold5_parity_on_a_six_factor_dataset(sql_training_db_six_factors, tmp_path):
    """GroupKFold5 es el otro protocolo que reporta la tesis; tambien debe cuadrar.

    El fixture sql_training_db tiene 3 grupos de factor_mult y GroupKFold5
    necesita 5, por eso este test usa el fixture de 6 factores.
    """
    legacy, bench = _run_both_sides(sql_training_db_six_factors, tmp_path, "GroupKFold5")

    _assert_top_level_shape(legacy, bench)
    _assert_metrics_match(legacy["classifier"], bench["classifier"], "classifier")
    _assert_metrics_match(
        legacy["regressor_oracle"], bench["regressor_oracle"], "regressor_oracle"
    )
    _assert_metrics_match(legacy["end_to_end"], bench["end_to_end"], "end_to_end")
    _assert_by_factor_matches(legacy, bench)
