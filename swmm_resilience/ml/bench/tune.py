"""Búsqueda automática de hiperparámetros (Optuna), aislada del banco.

Este módulo NUNCA se invoca desde --bench: solo desde --bench-tune,
explícito, sobre una sola familia. Los folds que genera son internos y
efímeros -- GroupKFold(5) agrupado por shape_id sobre todo el
PreparedDataset -- y nunca se declaran como protocolo del banco: no tocan
PROTOCOL_GROUP_COLUMNS, group_column_for() ni el manifiesto persistido de
PreparedDataset (ver spec docs/superpowers/specs/2026-09-29-optuna-tuning-
familia-ganadora-design.md §4.2).

Al no reservar un holdout de formas separado, todo resultado de un estudio
se etiqueta "estimacion_exploratoria_condicionada" (spec §12 del diseño de
cierre de auditoría): no debe presentarse como confirmatorio.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version as _package_version
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from ...config import ML_RANDOM_STATE
from .evaluate import _fit_predict_fold, fit_predict_folds_parallel
from .metrics import end_to_end_metrics
from .schemas import PreparedDataset


def build_tuning_folds(
    keys: pd.DataFrame, n_splits: int = 5, seed: int = ML_RANDOM_STATE
) -> pd.DataFrame:
    """GroupKFold(n_splits) agrupado por shape_id sobre todo `keys`.

    Devuelve un frame con columnas ('fold_id', 'split', 'sample_idx'), sin
    'protocol': estos folds son efímeros, por-trial, y nunca se escriben en
    el manifiesto de PreparedDataset.
    """
    groups = keys["shape_id"]
    distinct = int(groups.nunique())
    if distinct < n_splits:
        raise ValueError(
            f"GroupKFold({n_splits}) para tuning necesita al menos {n_splits} "
            f"formas (shape_id) distintas; el dataset tiene {distinct}."
        )
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x_dummy = np.zeros((len(groups), 1))
    records: list[dict] = []
    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(x_dummy, None, groups.values)
    ):
        for split_name, indices in (("train", train_idx), ("test", test_idx)):
            for sample_idx in sorted(int(value) for value in indices):
                records.append(
                    {"fold_id": fold_id, "split": split_name, "sample_idx": sample_idx}
                )
    return pd.DataFrame.from_records(records, columns=["fold_id", "split", "sample_idx"])


# Notación por parámetro: ("int", low, high) | ("float", low, high) |
# ("float_log", low, high) muestreado log-uniforme | ("categorical", [...]).
# Rangos centrados en los defaults actuales de config.yaml (ver spec §6).
SEARCH_SPACES: dict[str, dict[str, dict[str, tuple]]] = {
    "xgboost": {
        "classifier": {
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float_log", 1e-3, 0.3),
            "n_estimators": ("int", 50, 500),
            "subsample": ("float", 0.5, 1.0),
        },
        "regressor": {
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float_log", 1e-3, 0.3),
            "n_estimators": ("int", 50, 500),
            "subsample": ("float", 0.5, 1.0),
        },
    },
    "random_forest": {
        "classifier": {"max_depth": ("int", 3, 15), "n_estimators": ("int", 50, 500)},
        "regressor": {"max_depth": ("int", 3, 15), "n_estimators": ("int", 50, 500)},
    },
    "linear": {
        "classifier": {"C": ("float_log", 1e-3, 100.0)},
        "regressor": {"alpha": ("float_log", 1e-3, 100.0)},
    },
    "svm": {
        "classifier": {
            "gamma": ("float_log", 1e-3, 1.0),
            "n_components": ("int", 100, 400),
            "alpha": ("float_log", 1e-6, 1e-2),
        },
        "regressor": {
            "gamma": ("float_log", 1e-3, 1.0),
            "n_components": ("int", 100, 400),
            "alpha": ("float_log", 1e-6, 1e-2),
            "epsilon": ("float", 0.01, 1.0),
        },
    },
    "mlp": {
        "classifier": {
            "hidden_sizes": ("categorical", [(32,), (64, 32), (128, 64), (64, 32, 16)]),
            "learning_rate": ("float_log", 1e-4, 1e-2),
            "dropout": ("float", 0.0, 0.5),
        },
        "regressor": {
            "hidden_sizes": ("categorical", [(32,), (64, 32), (128, 64), (64, 32, 16)]),
            "learning_rate": ("float_log", 1e-4, 1e-2),
            "dropout": ("float", 0.0, 0.5),
        },
    },
}


def _sample_params(trial, space: dict, *, prefix: str) -> dict:
    """Muestrea un dict de hiperparámetros desde `space` usando `trial`.

    `prefix` evita colisiones de nombre entre clasificador y regresor
    dentro del mismo trial (p.ej. "xgboost_classifier_max_depth" vs
    "xgboost_regressor_max_depth"): Optuna identifica cada valor muestreado
    por su nombre completo dentro del estudio.
    """
    params: dict = {}
    for name, spec in space.items():
        kind = spec[0]
        full_name = f"{prefix}{name}"
        if kind == "int":
            params[name] = trial.suggest_int(full_name, spec[1], spec[2])
        elif kind == "float":
            params[name] = trial.suggest_float(full_name, spec[1], spec[2])
        elif kind == "float_log":
            params[name] = trial.suggest_float(full_name, spec[1], spec[2], log=True)
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(full_name, spec[1])
        else:
            raise ValueError(f"Tipo de espacio desconocido: {kind!r}")
    return params


_STATUS_EXPLORATORY = "estimacion_exploratoria_condicionada"
_STATUS_REASON_EXPLORATORY = (
    "No se reservó un holdout de formas separado ni se aplicó selección de "
    "familia anidada (spec §12 del diseño de cierre de auditoría, sección "
    "'Optimización posterior'); este resultado no debe presentarse como "
    "confirmatorio."
)


def _merge_fixed(sampled: dict, fixed: dict | None, task: str) -> dict:
    """Parámetros efectivos: lo muestreado, pisado por lo fijado para ``task``."""
    return {**sampled, **((fixed or {}).get(task) or {})}


def _objective(
    trial,
    prepared,
    family_name: str,
    tuning_folds: pd.DataFrame,
    fold_n_jobs: int = 1,
    fixed_params: dict | None = None,
) -> float:
    classifier_params = _merge_fixed(
        _sample_params(
            trial, SEARCH_SPACES[family_name]["classifier"], prefix=f"{family_name}_classifier_"
        ),
        fixed_params,
        "classifier",
    )
    regressor_params = _merge_fixed(
        _sample_params(
            trial, SEARCH_SPACES[family_name]["regressor"], prefix=f"{family_name}_regressor_"
        ),
        fixed_params,
        "regressor",
    )
    trial.set_user_attr("classifier_params", classifier_params)
    trial.set_user_attr("regressor_params", regressor_params)

    y_clf_all = prepared.y_clf.to_numpy()
    y_reg_all = prepared.y_reg.to_numpy()

    splits = []
    for _, chunk in tuning_folds.groupby("fold_id"):
        train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"].to_numpy()
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"].to_numpy()
        splits.append((train_idx, test_idx))

    if fold_n_jobs > 1:
        predictions = fit_predict_folds_parallel(
            prepared, family_name, classifier_params, regressor_params, splits, fold_n_jobs
        )
    else:
        predictions = [
            _fit_predict_fold(
                prepared, family_name, classifier_params, regressor_params, train_idx, test_idx
            )
            for train_idx, test_idx in splits
        ]

    fold_values: list[float] = []
    for (_, test_idx), (yc_pred, _, yr_pred) in zip(splits, predictions):
        routed = np.where(yc_pred == 1, yr_pred, 0.0)
        fold_values.append(
            end_to_end_metrics(
                y_reg_all[test_idx], routed, y_clf_all[test_idx], yc_pred,
            )["rmse_vol_todos_nodos"]
        )
    return float(np.mean(fold_values))


@dataclass(frozen=True)
class TuneResult:
    family: str
    prep_id: str
    inner_cv: dict
    objective_metric: str
    direction: str
    n_trials: int
    seed: int
    optuna_version: str
    started_at_utc: str
    finished_at_utc: str
    duration_s: float
    trials: list = field(default_factory=list)
    best_classifier_params: dict = field(default_factory=dict)
    best_regressor_params: dict = field(default_factory=dict)
    best_value: float = float("nan")
    status: str = _STATUS_EXPLORATORY
    status_reason: str = _STATUS_REASON_EXPLORATORY
    fixed_params: dict = field(default_factory=dict)
    fold_n_jobs: int = 1


def run_study(
    prepared: PreparedDataset,
    family_name: str,
    n_trials: int = 50,
    n_splits: int = 5,
    seed: int = ML_RANDOM_STATE,
    fold_n_jobs: int = 1,
    fixed_params: dict | None = None,
) -> TuneResult:
    """Corre un estudio Optuna sobre `family_name`, inner GroupKFold(n_splits) por shape_id.

    Etiqueta el resultado como estimación exploratoria: no hay holdout de
    formas separado del que usa esta búsqueda (spec §12).

    ``fold_n_jobs`` > 1 evalúa los folds internos de cada trial en procesos
    aparte (el valor del trial es la misma media). ``fixed_params``
    (``{"classifier": {...}, "regressor": {...}}``) se fusiona SOBRE lo
    muestreado en cada trial, para fijar ajustes que no se buscan (épocas del
    MLP, hilos de XGBoost).
    """
    if family_name not in SEARCH_SPACES:
        raise ValueError(
            f"Familia sin espacio de búsqueda declarado: {family_name!r}. "
            f"Disponibles: {', '.join(sorted(SEARCH_SPACES))}"
        )

    tuning_folds = build_tuning_folds(prepared.keys, n_splits=n_splits, seed=seed)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    started = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    study.optimize(
        lambda trial: _objective(
            trial, prepared, family_name, tuning_folds, fold_n_jobs, fixed_params
        ),
        n_trials=n_trials,
        catch=(Exception,),
    )
    duration_s = time.perf_counter() - start_perf
    finished = datetime.now(timezone.utc)

    trials = [
        {
            "number": t.number,
            "classifier_params": t.user_attrs.get("classifier_params"),
            "regressor_params": t.user_attrs.get("regressor_params"),
            "value": t.value,
            "duration_s": t.duration.total_seconds() if t.duration else None,
        }
        for t in study.trials
    ]

    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise ValueError(
            f"Los {n_trials} trials de Optuna para {family_name!r} fallaron todos; "
            "no hay ningún resultado válido que reportar."
        )

    return TuneResult(
        family=family_name,
        prep_id=prepared.prep_id,
        inner_cv={"method": "GroupKFold", "n_splits": n_splits, "group_column": "shape_id"},
        objective_metric="end_to_end.rmse_vol_todos_nodos",
        direction="minimize",
        n_trials=n_trials,
        seed=seed,
        optuna_version=_package_version("optuna"),
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        duration_s=duration_s,
        trials=trials,
        best_classifier_params=study.best_trial.user_attrs["classifier_params"],
        best_regressor_params=study.best_trial.user_attrs["regressor_params"],
        best_value=float(study.best_value),
        fixed_params=json.loads(json.dumps(fixed_params or {})),
        fold_n_jobs=int(fold_n_jobs),
    )


def save_tune_result(result: TuneResult, output_dir: Path | str) -> Path:
    """Escribe `result` en <output_dir>/<family>/<prep_id>_<timestamp>/study.json."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(output_dir) / result.family / f"{result.prep_id}_{timestamp}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "study.json"
    payload = {
        "family": result.family,
        "prep_id": result.prep_id,
        "inner_cv": result.inner_cv,
        "objective_metric": result.objective_metric,
        "direction": result.direction,
        "n_trials": result.n_trials,
        "seed": result.seed,
        "optuna_version": result.optuna_version,
        "started_at_utc": result.started_at_utc,
        "finished_at_utc": result.finished_at_utc,
        "duration_s": result.duration_s,
        "trials": result.trials,
        "best_classifier_params": result.best_classifier_params,
        "best_regressor_params": result.best_regressor_params,
        "best_value": result.best_value,
        "status": result.status,
        "status_reason": result.status_reason,
        "fixed_params": result.fixed_params,
        "fold_n_jobs": result.fold_n_jobs,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
