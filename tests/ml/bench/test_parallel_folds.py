"""Paralelismo opcional por folds en evaluate_candidate y run_study.

El valor por defecto (n_jobs=1) conserva el camino secuencial de siempre; con
n_jobs > 1 cada fold corre en un proceso aparte (joblib/loky) y el resultado
debe ser idéntico, fold a fold, al secuencial.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.bench.evaluate import evaluate_candidate
from swmm_resilience.ml.bench.tune import SEARCH_SPACES, run_study, save_tune_result
from tests.ml.bench.prepared_factory import make_prepared

# n_jobs=1 de XGBoost: el resultado no es reproducible entre n_jobs distintos
# (hilos), así que la comparación secuencial/paralelo fija 1 hilo por modelo.
TINY = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3, "subsample": 1.0, "n_jobs": 1}


def _rows(n_shapes: int = 6, factors=(1.0, 2.0, 3.0, 4.0), nodes=("N0", "N1")):
    """Cuadrícula forma × factor × nodo. Hay filas inundadas y no inundadas en
    cualquier combinación de entrenamiento de LeaveOneShapeOut y de LOSO."""
    rows = []
    for index in range(n_shapes):
        for factor in factors:
            for node_index, node in enumerate(nodes):
                rows.append(
                    dict(
                        shape_id=f"S{index}",
                        node_id=node,
                        factor_mult=factor,
                        vol=(
                            0.0
                            if factor == 1.0 or (node_index == 1 and factor == 2.0)
                            else 3.0 * factor * (1 + index)
                        ),
                        duracion_horas=1.0 + index,
                        tiempo_al_pico_h=0.5 * (1 + index),
                    )
                )
    return rows


def _with_signal(prepared):
    """Da a los árboles algo que aprender (factor y nodo) en dos features."""
    columns = list(prepared.X.columns)
    prepared.X[columns[0]] = prepared.keys["factor_mult"].to_numpy(dtype=float)
    prepared.X[columns[1]] = (prepared.keys["node_id"] == "N1").to_numpy(dtype=float)
    return prepared


def _same(first: dict, second: dict) -> bool:
    """Igualdad de métricas tolerante a NaN (NaN == NaN dentro de un dict)."""
    return json.dumps(first, sort_keys=True, default=float) == json.dumps(
        second, sort_keys=True, default=float
    )


@pytest.fixture(scope="module")
def prepared():
    return _with_signal(make_prepared(_rows(), ["LeaveOneShapeOut", "LOSO"]))


def test_protocols_restricts_the_folds_that_are_fitted_and_scored(prepared):
    only, metrics = evaluate_candidate(
        prepared, "xgboost", TINY, TINY, protocols=["LeaveOneShapeOut"]
    )
    full, full_metrics = evaluate_candidate(prepared, "xgboost", TINY, TINY)

    assert set(only["protocol"]) == {"LeaveOneShapeOut"}
    assert set(metrics) == {"LeaveOneShapeOut"}
    assert set(full["protocol"]) == {"LeaveOneShapeOut", "LOSO"}
    subset = full[full["protocol"] == "LeaveOneShapeOut"].reset_index(drop=True)
    pd.testing.assert_frame_equal(only, subset)
    assert _same(metrics["LeaveOneShapeOut"], full_metrics["LeaveOneShapeOut"])


def test_n_jobs_two_gives_the_same_oof_as_sequential(prepared):
    sequential, seq_metrics = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    parallel, par_metrics = evaluate_candidate(prepared, "xgboost", TINY, TINY, n_jobs=2)

    pd.testing.assert_frame_equal(sequential, parallel)
    assert _same(seq_metrics, par_metrics)


def test_n_jobs_restores_the_thread_environment(prepared, monkeypatch):
    import os

    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    evaluate_candidate(prepared, "xgboost", TINY, TINY, protocols=["LOSO"], n_jobs=2)

    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert "MKL_NUM_THREADS" not in os.environ


def test_the_thread_limiter_sets_and_restores_the_variables(monkeypatch):
    import os

    from swmm_resilience.ml.bench.evaluate import single_thread_environment

    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    with single_thread_environment():
        for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
        ):
            assert os.environ[name] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert "OPENBLAS_NUM_THREADS" not in os.environ


def test_run_study_with_parallel_inner_folds_gives_the_same_trial_values():
    prepared = _with_signal(make_prepared(_rows(n_shapes=6), ["LeaveOneShapeOut"]))
    fixed = {"classifier": {"n_estimators": 5, "n_jobs": 1}, "regressor": {"n_estimators": 5, "n_jobs": 1}}

    sequential = run_study(prepared, "xgboost", n_trials=2, n_splits=5, fixed_params=fixed)
    parallel = run_study(
        prepared, "xgboost", n_trials=2, n_splits=5, fold_n_jobs=2, fixed_params=fixed
    )

    assert [t["value"] for t in sequential.trials] == [t["value"] for t in parallel.trials]
    assert sequential.best_value == parallel.best_value
    assert sequential.fold_n_jobs == 1
    assert parallel.fold_n_jobs == 2


def test_fixed_params_override_the_sampled_values_and_are_recorded(tmp_path):
    prepared = _with_signal(make_prepared(_rows(n_shapes=6), ["LeaveOneShapeOut"]))
    fixed = {"classifier": {"n_estimators": 7, "n_jobs": 1}, "regressor": {"n_estimators": 9, "n_jobs": 1}}

    result = run_study(prepared, "xgboost", n_trials=2, n_splits=5, fixed_params=fixed)

    for trial in result.trials:
        assert trial["classifier_params"]["n_estimators"] == 7
        assert trial["regressor_params"]["n_estimators"] == 9
        assert trial["classifier_params"]["n_jobs"] == 1
        # lo no fijado sigue muestreado
        assert set(SEARCH_SPACES["xgboost"]["classifier"]) <= set(trial["classifier_params"])
    assert result.best_classifier_params["n_estimators"] == 7
    assert result.fixed_params == fixed

    payload = json.loads(save_tune_result(result, tmp_path).read_text(encoding="utf-8"))
    assert payload["fixed_params"] == fixed
    assert payload["fold_n_jobs"] == 1


def test_defaults_record_empty_fixed_params_and_one_job(tmp_path):
    prepared = _with_signal(make_prepared(_rows(n_shapes=6), ["LeaveOneShapeOut"]))
    result = run_study(prepared, "xgboost", n_trials=1, n_splits=5)

    assert result.fixed_params == {}
    assert result.fold_n_jobs == 1
    assert result.status == "estimacion_exploratoria_condicionada"
    payload = json.loads(save_tune_result(result, tmp_path).read_text(encoding="utf-8"))
    assert payload["fixed_params"] == {}
    assert payload["fold_n_jobs"] == 1


def test_importing_tune_and_evaluate_does_not_import_torch():
    code = (
        "import sys\n"
        "import swmm_resilience.ml.bench.tune\n"
        "import swmm_resilience.ml.bench.evaluate\n"
        "sys.exit(1 if 'torch' in sys.modules else 0)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_mlp_folds_run_in_worker_processes_with_one_torch_thread(prepared):
    """Los hijos importan torch por su cuenta (el padre no lo importó en el módulo)."""
    tiny_mlp = {"hidden_sizes": [8], "epochs": 2, "batch_size": 64}
    oof, metrics = evaluate_candidate(
        prepared, "mlp", tiny_mlp, tiny_mlp, protocols=["LOSO"], n_jobs=2
    )

    assert set(oof["protocol"]) == {"LOSO"}
    assert len(oof) == len(prepared.folds.query("protocol == 'LOSO' and split == 'test'"))
    assert oof["y_prob_clf"].between(0.0, 1.0).all()
