"""Etapa 3 del banco, en dos capas.

score_predictions() es la capa núcleo: NO sabe de modelos. Recibe
predicciones out-of-fold ya calculadas, verifica su proveniencia contra el
PreparedDataset, y devuelve las métricas de los tres niveles. Cualquier
modelo —tabular, temporal, escrito en otro script o dentro de seis meses—
entra por aquí y su número cae en la misma tabla. Es el mecanismo que
garantiza "mismas entradas, mismos benchmarks" (spec §13.1).

evaluate_candidate() es la capa de conveniencia que produce esas
predicciones para las familias tabulares del banco (Task 9).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .metrics import (
    classifier_metrics,
    confusion_counts,
    end_to_end_metrics,
    end_to_end_pooled_metrics,
    fold_classifier_metrics,
    mean_metrics,
    macro_network_rmse,
    pooled_classifier_metrics,
    pooled_regressor_metrics,
    shape_breakdown_entry,
    summarize_fold_metrics,
)
from .schemas import OOF_COLUMNS, PreparedDataset


@dataclass(frozen=True)
class _FoldTrainingView:
    """Subset materializado que deja al tuning acceso solo al outer train."""

    prep_id: str
    keys: pd.DataFrame
    X: pd.DataFrame
    y_clf: pd.Series
    y_reg: pd.Series
    run_ids: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class NestedNetworkFold:
    fold_id: int
    held_out_network_id: object
    outer_train_sample_idx: tuple[int, ...]
    outer_test_sample_idx: tuple[int, ...]
    outer_train_run_ids: tuple[tuple[str, str], ...]
    outer_test_run_ids: tuple[tuple[str, str], ...]
    inner_run_ids: tuple[tuple[str, str], ...]
    inner_prep_id: str
    optuna_study_id: str
    selected_classifier_params: dict
    selected_regressor_params: dict
    objective: str
    inner_best_value: float
    seed: int
    trials: tuple[dict, ...]
    oof_row_keys: tuple[dict, ...]


@dataclass(frozen=True)
class NestedNetworkResult:
    family: str
    prep_id: str
    objective: str
    volume_band_edges_m3: tuple[float, ...]
    oof: pd.DataFrame
    folds: tuple[NestedNetworkFold, ...]
    metrics: dict


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_study(*args, **kwargs):
    """Carga tardía el módulo de tuning para evitar el ciclo de imports."""
    from .tune import run_study as _run_study

    return _run_study(*args, **kwargs)


def _network_run_ids(keys: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    pairs = {(str(row.network_id), str(row.run_id)) for row in keys.itertuples()}
    return tuple(sorted(pairs))


def _fold_training_view(prepared: PreparedDataset, train_idx):
    keys = prepared.keys.iloc[train_idx].reset_index(drop=True).copy()
    X = prepared.X.iloc[train_idx].reset_index(drop=True).copy()
    y_clf = prepared.y_clf.iloc[train_idx].reset_index(drop=True).copy()
    y_reg = prepared.y_reg.iloc[train_idx].reset_index(drop=True).copy()
    fingerprint_payload = {
        "contract_id": prepared.manifest.get("contract_id"),
        "feature_contract_sha256": prepared.manifest.get("feature_contract_sha256"),
        "flood_threshold_m3": prepared.manifest.get("flood_threshold_m3"),
        "fingerprint_version": "outer_train_view_v1",
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    )
    for frame in (keys, X, y_clf.to_frame(), y_reg.to_frame()):
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    prep_id = digest.hexdigest()[:16]
    return _FoldTrainingView(
        prep_id=prep_id,
        keys=keys,
        X=X,
        y_clf=y_clf,
        y_reg=y_reg,
        run_ids=_network_run_ids(keys),
    )


class ProvenanceMismatchError(ValueError):
    """Las predicciones no corresponden a este PreparedDataset."""


def _validate_oof(
    prepared: PreparedDataset, oof: pd.DataFrame, protocols=None
) -> None:
    missing = [column for column in OOF_COLUMNS if column not in oof.columns]
    if missing:
        raise ValueError(f"The OOF frame is missing columns: {missing}")

    duplicated = oof.duplicated(subset=["protocol", "fold_id", "sample_idx"]).sum()
    if duplicated:
        raise ValueError(
            f"The OOF frame has {duplicated} duplicated prediction(s) for the same "
            "(protocol, fold_id, sample_idx)."
        )

    expected = prepared.folds[prepared.folds["split"] == "test"]
    if protocols is not None:
        expected = expected[expected["protocol"].isin(list(protocols))]
    expected_keys = set(
        map(tuple, expected[["protocol", "fold_id", "sample_idx"]].itertuples(index=False))
    )
    got_keys = set(
        map(tuple, oof[["protocol", "fold_id", "sample_idx"]].itertuples(index=False))
    )
    if expected_keys != got_keys:
        raise ValueError(
            "The OOF frame coverage does not match the test rows of the folds: "
            f"missing {len(expected_keys - got_keys)}, extra {len(got_keys - expected_keys)}."
        )


def score_predictions(
    prepared: PreparedDataset,
    oof: pd.DataFrame,
    provenance: dict,
    protocols=None,
) -> dict:
    """Puntúa predicciones out-of-fold contra ``prepared``.

    ``provenance`` debe traer el ``prep_id`` del dataset con el que se
    generaron. Sin esa verificación se podrían comparar números de datasets
    distintos sin notarlo, que es el modo de fallo silencioso que este
    diseño existe para cerrar.

    ``protocols`` restringe la cobertura exigida al frame OOF (una referencia
    sin aprendizaje puede aplicar solo a algunos protocolos). Por defecto se
    exigen todos los del PreparedDataset.

    Por protocolo devuelve:
    - ``classifier``: métricas AGREGADAS sobre todas las filas de test (cifra
      principal) y ``confusion_matrix`` agregada.
    - ``classifier_per_fold``: dispersión entre folds, con NaN donde no está definido.
    - ``classifier_legacy_mean_zero_fill``: la media por fold con ``zero_division=0``
      (definición anterior, conservada para citar el antes y el después).
    - ``regressor_oracle``, ``end_to_end``, ``by_factor``: sin cambios.
    - ``end_to_end_pooled`` y ``by_shape``: agregado de nivel 3 y desglose por forma.
    """
    if provenance.get("prep_id") != prepared.prep_id:
        raise ProvenanceMismatchError(
            f"prep_id of the predictions ({provenance.get('prep_id')!r}) does not match "
            f"the dataset one ({prepared.prep_id!r}). They are not comparable."
        )
    _validate_oof(prepared, oof, protocols)

    y_clf = prepared.y_clf.to_numpy()
    y_reg = prepared.y_reg.to_numpy()
    factors = prepared.keys["factor_mult"].to_numpy()
    shapes = prepared.keys["shape_id"].astype(str).to_numpy()

    results: dict = {}
    for protocol, protocol_oof in oof.groupby("protocol"):
        classifier_folds, fold_stats, e2e_folds = [], [], []
        oracle_true, oracle_pred = [], []
        pool_true, pool_pred, pool_prob = [], [], []
        pool_vol_true, pool_vol_routed = [], []
        by_factor: dict[str, list] = {}
        by_shape_parts: dict[str, dict[str, list]] = {}

        for _, fold_oof in protocol_oof.groupby("fold_id"):
            idx = fold_oof["sample_idx"].to_numpy()
            yc_true, yr_true = y_clf[idx], y_reg[idx]
            yc_pred = fold_oof["y_pred_clf"].to_numpy()
            yc_prob = fold_oof["y_prob_clf"].to_numpy()
            yr_pred = fold_oof["y_pred_reg"].to_numpy()

            classifier_folds.append(classifier_metrics(yc_true, yc_pred, yc_prob))
            fold_stats.append(fold_classifier_metrics(yc_true, yc_pred))
            pool_true.append(yc_true)
            pool_pred.append(yc_pred)
            pool_prob.append(yc_prob)

            flooded = yc_true == 1
            if flooded.sum():
                oracle_true.append(yr_true[flooded])
                oracle_pred.append(yr_pred[flooded])

            routed = np.where(yc_pred == 1, yr_pred, 0.0)
            e2e_folds.append(end_to_end_metrics(yr_true, routed, yc_true, yc_pred))
            pool_vol_true.append(yr_true)
            pool_vol_routed.append(routed)

            for factor in np.unique(factors[idx]):
                mask = factors[idx] == factor
                key = f"{factor:.2f}"
                by_factor.setdefault(key, []).append(
                    {
                        "f1": classifier_metrics(
                            yc_true[mask], yc_pred[mask], yc_prob[mask]
                        )["f1"],
                        "rmse_vol": end_to_end_metrics(
                            yr_true[mask], routed[mask], yc_true[mask], yc_pred[mask]
                        )["rmse_vol_todos_nodos"],
                    }
                )

            for shape in np.unique(shapes[idx]):
                mask = shapes[idx] == shape
                parts = by_shape_parts.setdefault(
                    str(shape),
                    {"yc_true": [], "yc_pred": [], "vol_true": [], "vol_routed": []},
                )
                parts["yc_true"].append(yc_true[mask])
                parts["yc_pred"].append(yc_pred[mask])
                parts["vol_true"].append(yr_true[mask])
                parts["vol_routed"].append(routed[mask])

        all_true = np.concatenate(pool_true)
        all_pred = np.concatenate(pool_pred)
        results[protocol] = {
            "classifier": pooled_classifier_metrics(pool_true, pool_pred, pool_prob),
            "confusion_matrix": confusion_counts(all_true, all_pred),
            "classifier_per_fold": summarize_fold_metrics(fold_stats),
            "classifier_legacy_mean_zero_fill": mean_metrics(classifier_folds),
            "regressor_oracle": pooled_regressor_metrics(oracle_true, oracle_pred),
            "end_to_end": mean_metrics(e2e_folds),
            "end_to_end_pooled": end_to_end_pooled_metrics(pool_vol_true, pool_vol_routed),
            "by_factor": {key: mean_metrics(value) for key, value in by_factor.items()},
            "by_shape": {
                shape: shape_breakdown_entry(
                    np.concatenate(parts["yc_true"]),
                    np.concatenate(parts["yc_pred"]),
                    np.concatenate(parts["vol_true"]),
                    np.concatenate(parts["vol_routed"]),
                )
                for shape, parts in by_shape_parts.items()
            },
        }
    return results


def _fit_predict_fold(
    prepared: PreparedDataset,
    family_name: str,
    classifier_params: dict,
    regressor_params: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ajusta clasificador+regresor SOLO con train_idx, predice sobre test_idx.

    Única función que decide fit/predict de un fold: evaluate_candidate()
    (folds exteriores de comparación) y tune.py (folds interiores de
    búsqueda de hiperparámetros) la comparten, así que la garantía de que
    ningún test toca .fit() vive en un solo lugar.

    Devuelve (y_pred_clf, y_prob_clf, y_pred_reg).
    """
    from .registry import get_family

    family = get_family(family_name)
    X = prepared.X
    y_clf = prepared.y_clf.to_numpy()
    y_reg = prepared.y_reg.to_numpy()

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    yc_train = y_clf[train_idx]
    yr_train = y_reg[train_idx]

    n_negative, n_positive = int((yc_train == 0).sum()), int((yc_train == 1).sum())
    scale_pos_weight = n_negative / n_positive if n_positive else 1.0

    classifier = family.build_classifier(classifier_params, scale_pos_weight)
    classifier.fit(X_train, yc_train)
    yc_pred = classifier.predict(X_test)
    yc_prob = classifier.predict_proba(X_test)[:, 1]

    flooded_train = yc_train == 1
    if flooded_train.sum():
        regressor = family.build_regressor(regressor_params)
        regressor.fit(X_train.iloc[flooded_train], np.log1p(yr_train[flooded_train]))
        yr_pred = np.clip(np.expm1(regressor.predict(X_test)), a_min=0.0, a_max=None)
    else:
        # Divergencia deliberada de _run_cv: allí, si el fold no tiene filas
        # inundadas de entrenamiento, el regresor nunca se ajusta pero el
        # nivel end-to-end igual puede invocar reg.predict(...), lo que
        # lanzaría NotFittedError (bug latente no manejado en el código
        # legado). Aquí predecimos volumen cero explícitamente en su
        # lugar. Con LOSO sobre los ~25 factores reales cada fold de
        # entrenamiento tiene filas inundadas, así que esta rama no
        # afecta al test de paridad.
        yr_pred = np.zeros(len(test_idx), dtype=float)

    return yc_pred.astype(int), yc_prob.astype(float), yr_pred.astype(float)


# Variables de entorno que acotan los hilos de BLAS/OpenMP; en el paralelismo
# por folds cada proceso debe usar un solo hilo (varios procesos × varios hilos
# saturaría la máquina y, en el MLP, hasta empeora el tiempo).
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextmanager
def single_thread_environment():
    """Fija a "1" las variables de hilos mientras dura el bloque y las restaura.

    Se aplica en el proceso padre ANTES de crear el pool de joblib, para que los
    procesos hijos nazcan heredando el límite.
    """
    previous = {name: os.environ.get(name) for name in _THREAD_ENV_VARS}
    try:
        for name in _THREAD_ENV_VARS:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class _FoldData(NamedTuple):
    """Lo mínimo que ``_fit_predict_fold`` necesita de un PreparedDataset.

    Viaja a los procesos hijos en vez del PreparedDataset completo: ``folds``
    (decenas de millones de filas con LeaveOneShapeOut) no se usa para ajustar
    y pesaría en cada tarea.
    """

    X: pd.DataFrame
    y_clf: pd.Series
    y_reg: pd.Series


def _fit_predict_fold_worker(
    data: _FoldData,
    family_name: str,
    classifier_params: dict,
    regressor_params: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
):
    """Tarea de un fold en un proceso hijo (nivel de módulo: joblib la serializa por nombre)."""
    from threadpoolctl import threadpool_limits

    from .registry import get_family

    get_family(family_name)  # importa el módulo de la familia (torch, si es el MLP)
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(1)
    with threadpool_limits(limits=1):
        return _fit_predict_fold(
            data, family_name, classifier_params, regressor_params, train_idx, test_idx
        )


def fit_predict_folds_parallel(
    prepared: PreparedDataset,
    family_name: str,
    classifier_params: dict,
    regressor_params: dict,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_jobs: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Ajusta y predice cada ``(train_idx, test_idx)`` en procesos aparte.

    Devuelve los resultados en el mismo orden que ``splits``, de modo que el
    resultado es idéntico al del bucle secuencial para familias deterministas.
    """
    from joblib import Parallel, delayed

    data = _FoldData(prepared.X, prepared.y_clf, prepared.y_reg)
    workers = max(1, min(int(n_jobs), len(splits)))
    with single_thread_environment():
        return Parallel(n_jobs=workers, backend="loky")(
            delayed(_fit_predict_fold_worker)(
                data, family_name, classifier_params, regressor_params, train_idx, test_idx
            )
            for train_idx, test_idx in splits
        )


def evaluate_candidate(
    prepared: PreparedDataset,
    family_name: str,
    classifier_params: dict,
    regressor_params: dict,
    *,
    protocols=None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Entrena y predice fold a fold, y puntúa el resultado.

    Es una función pura de los hiperparámetros: exactamente la forma que
    necesita un ``objective`` de Optuna (spec §13.2).

    El Pipeline de la familia lleva dentro la imputación y el escalado, así
    que ``fit`` sobre el train del fold los ajusta SOLO con esas filas. Ahí
    está la garantía anti-fuga en ejecución.

    ``protocols`` (iterable o None = todos) restringe los folds que se ajustan
    y puntúan. ``n_jobs`` > 1 corre los folds en procesos aparte (un hilo cada
    uno); los folds se concatenan siempre en el orden (protocolo, fold), así
    que el OOF no depende de ``n_jobs``.
    """
    folds = prepared.folds
    score_protocols = None
    if protocols is not None:
        score_protocols = list(protocols)
        folds = folds[folds["protocol"].isin(score_protocols)]

    groups = []
    for (protocol, fold_id), chunk in folds.groupby(["protocol", "fold_id"]):
        train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"].to_numpy()
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"].to_numpy()
        groups.append((protocol, fold_id, train_idx, test_idx))

    if n_jobs > 1:
        predictions = fit_predict_folds_parallel(
            prepared,
            family_name,
            classifier_params,
            regressor_params,
            [(train_idx, test_idx) for _, _, train_idx, test_idx in groups],
            n_jobs,
        )
    else:
        predictions = [
            _fit_predict_fold(
                prepared, family_name, classifier_params, regressor_params, train_idx, test_idx
            )
            for _, _, train_idx, test_idx in groups
        ]

    records: list[pd.DataFrame] = []
    for (protocol, fold_id, _, test_idx), (yc_pred, yc_prob, yr_pred) in zip(
        groups, predictions
    ):
        records.append(
            pd.DataFrame(
                {
                    "sample_idx": test_idx,
                    "protocol": protocol,
                    "fold_id": fold_id,
                    "y_pred_clf": yc_pred.astype(int),
                    "y_prob_clf": yc_prob.astype(float),
                    "y_pred_reg": yr_pred.astype(float),
                },
                columns=list(OOF_COLUMNS),
            )
        )

    oof = pd.concat(records, ignore_index=True)
    metrics = score_predictions(
        prepared, oof, {"prep_id": prepared.prep_id}, protocols=score_protocols
    )
    return oof, metrics


def _nested_reporting_entry(
    y_clf_true,
    y_clf_pred,
    y_clf_prob,
    volume_true,
    volume_pred,
    volume_band_edges_m3,
    flood_threshold_m3,
) -> dict:
    y_clf_true = np.asarray(y_clf_true, dtype=int)
    y_clf_pred = np.asarray(y_clf_pred, dtype=int)
    y_clf_prob = np.asarray(y_clf_prob, dtype=float)
    volume_true = np.asarray(volume_true, dtype=float)
    volume_pred = np.asarray(volume_pred, dtype=float)
    counts = confusion_counts(y_clf_true, y_clf_pred)
    classifier = fold_classifier_metrics(y_clf_true, y_clf_pred)
    pr_auc = (
        float(average_precision_score(y_clf_true, y_clf_prob))
        if np.unique(y_clf_true).size == 2
        else None
    )
    classifier = {
        key: (None if np.isnan(value) else float(value))
        for key, value in classifier.items()
    }
    real_total = float(volume_true.sum())
    predicted_total = float(volume_pred.sum())
    band_rows = []
    starts = (float(flood_threshold_m3), *volume_band_edges_m3[:-1])
    for band_index, (lower, upper) in enumerate(
        zip(starts, volume_band_edges_m3)
    ):
        if band_index == 0:
            band_mask = (volume_true >= lower) & (volume_true <= upper)
            label = f"[{lower:g}, {upper:g}]"
        else:
            band_mask = (volume_true > lower) & (volume_true <= upper)
            label = f"({lower:g}, {upper:g}]"
        actual_positive = band_mask & (y_clf_true == 1)
        false_negative = actual_positive & (y_clf_pred == 0)
        positive_count = int(actual_positive.sum())
        band_rows.append(
            {
                "band_m3": label,
                "lower_m3_inclusive": float(lower),
                "upper_m3_inclusive": float(upper),
                "n_samples": int(band_mask.sum()),
                "n_actual_flooded": positive_count,
                "false_negatives": int(false_negative.sum()),
                "recall": (
                    float((actual_positive & (y_clf_pred == 1)).sum() / positive_count)
                    if positive_count
                    else None
                ),
            }
        )
    lower = volume_band_edges_m3[-1]
    band_mask = volume_true > lower
    actual_positive = band_mask & (y_clf_true == 1)
    false_negative = actual_positive & (y_clf_pred == 0)
    positive_count = int(actual_positive.sum())
    band_rows.append(
        {
            "band_m3": f"> {lower:g}",
            "lower_m3_exclusive": float(lower),
            "upper_m3_inclusive": None,
            "n_samples": int(band_mask.sum()),
            "n_actual_flooded": positive_count,
            "false_negatives": int(false_negative.sum()),
            "recall": (
                float((actual_positive & (y_clf_pred == 1)).sum() / positive_count)
                if positive_count
                else None
            ),
        }
    )
    return {
        "n_samples": int(len(y_clf_true)),
        "n_actual_flooded": int(y_clf_true.sum()),
        "confusion_matrix": counts,
        "precision": classifier["precision"],
        "recall": classifier["recall"],
        "f1": classifier["f1"],
        "pr_auc": pr_auc,
        "rmse_m3": float(np.sqrt(np.mean((volume_true - volume_pred) ** 2))),
        "mae_m3": float(np.mean(np.abs(volume_true - volume_pred))),
        "volume_bias_m3": predicted_total - real_total,
        "volume_bias_pct": (
            100.0 * (predicted_total - real_total) / real_total
            if real_total > 0
            else None
        ),
        "false_negatives_by_volume_band": band_rows,
    }


def nested_network_event_metrics(
    prepared: PreparedDataset,
    oof: pd.DataFrame,
    *,
    volume_band_edges_m3: Sequence[float],
) -> dict:
    """Report network and hydrograph-shape OOF metrics using fixed volume bands.

    ``shape_id`` identifies the hydrograph shape represented in the prepared
    data. It is a reporting stratum, not a claim that event-family leakage has
    been controlled; event-family holdout remains a separate protocol.
    """
    from ...config import validate_volume_band_edges

    edges = validate_volume_band_edges(volume_band_edges_m3, required=True)
    flood_threshold_m3 = float(prepared.manifest["flood_threshold_m3"])
    if edges[0] < flood_threshold_m3:
        raise ValueError(
            "The first volume_band_edges_m3 value must be at least the configured "
            "flood_threshold_m3."
        )
    _validate_oof(prepared, oof, protocols=("LeaveOneNetworkOut",))
    predictions = oof.set_index("sample_idx").sort_index()
    keys = prepared.keys.reset_index(drop=True)
    y_clf_true = prepared.y_clf.to_numpy(dtype=int)
    volume_true = prepared.y_reg.to_numpy(dtype=float)
    sample_indices = np.arange(len(keys))
    y_clf_pred = predictions.loc[sample_indices, "y_pred_clf"].to_numpy(dtype=int)
    y_clf_prob = predictions.loc[sample_indices, "y_prob_clf"].to_numpy(dtype=float)
    volume_pred_raw = predictions.loc[sample_indices, "y_pred_reg"].to_numpy(
        dtype=float
    )
    volume_pred = np.where(y_clf_pred == 1, volume_pred_raw, 0.0)

    def entry(sample_idx):
        sample_idx = np.asarray(sample_idx, dtype=int)
        return _nested_reporting_entry(
            y_clf_true[sample_idx],
            y_clf_pred[sample_idx],
            y_clf_prob[sample_idx],
            volume_true[sample_idx],
            volume_pred[sample_idx],
            edges,
            flood_threshold_m3,
        )

    by_network = {}
    for network_id, group in keys.groupby("network_id", sort=True):
        by_network[str(network_id)] = entry(group.index.to_numpy())

    by_network_event = []
    for (network_id, shape_id), group in keys.groupby(
        ["network_id", "shape_id"], sort=True, dropna=False
    ):
        by_network_event.append(
            {
                "network_id": str(network_id),
                "shape_id": str(shape_id),
                **entry(group.index.to_numpy()),
            }
        )
    return {
        "event_unit": "shape_id (hydrograph shape; not event-family holdout)",
        "pr_auc_method": "average_precision_score",
        "flood_threshold_m3": flood_threshold_m3,
        "volume_band_edges_m3": list(edges),
        "by_network": by_network,
        "by_network_event": by_network_event,
    }


def evaluate_nested_networks(
    prepared: PreparedDataset,
    family: str,
    inner_protocols: Sequence[str],
    objective: str,
    *,
    volume_band_edges_m3: Sequence[float],
    n_trials: int = 50,
    n_splits: int = 5,
    seed: int = 42,
    fold_n_jobs: int = 1,
    fixed_params: dict | None = None,
) -> NestedNetworkResult:
    """Ajusta hiperparámetros dentro de cada outer train y evalúa la red retenida.

    Por ahora, la partición interna ``LeaveOneShapeOut`` deja exactamente una
    forma de hidrograma fuera por fold dentro de la vista del outer train.
    El argumento ``inner_protocols`` declara esa decisión en la procedencia.
    ``run_study`` recibe una vista que no contiene filas, targets ni features
    de la red externa.
    """
    from ...config import validate_tuning_objective
    from ...config import validate_volume_band_edges

    if tuple(inner_protocols) != ("LeaveOneShapeOut",):
        raise ValueError(
            "Nested network evaluation currently supports only "
            "inner_protocols=('LeaveOneShapeOut',)."
        )
    validate_tuning_objective(("LeaveOneNetworkOut",), objective)
    volume_band_edges_m3 = validate_volume_band_edges(
        volume_band_edges_m3, required=True
    )
    outer_folds = prepared.folds[
        (prepared.folds["protocol"] == "LeaveOneNetworkOut")
    ]
    if outer_folds.empty:
        raise ValueError("PreparedDataset has no LeaveOneNetworkOut folds.")

    oof_parts: list[pd.DataFrame] = []
    fold_results: list[NestedNetworkFold] = []
    per_network_true: dict[str, list[np.ndarray]] = {}
    per_network_pred: dict[str, list[np.ndarray]] = {}
    for fold_id, chunk in outer_folds.groupby("fold_id", sort=True):
        train_idx = np.sort(chunk.loc[chunk["split"] == "train", "sample_idx"].to_numpy())
        test_idx = np.sort(chunk.loc[chunk["split"] == "test", "sample_idx"].to_numpy())
        if train_idx.size == 0 or test_idx.size == 0:
            raise ValueError(f"Outer fold {fold_id} must have train and test rows.")
        train_networks = set(prepared.keys.iloc[train_idx]["network_id"])
        test_networks = set(prepared.keys.iloc[test_idx]["network_id"])
        if len(test_networks) != 1 or train_networks.intersection(test_networks):
            raise ValueError(
                f"Outer fold {fold_id} must hold out exactly one unseen network."
            )

        view = _fold_training_view(prepared, train_idx)
        if not view.keys["network_id"].isin(train_networks).all():
            raise RuntimeError("The inner training view contains a non-training network.")
        inner_seed = int(seed) + int(fold_id)
        tuned = run_study(
            view,
            family,
            n_trials=n_trials,
            n_splits=n_splits,
            seed=inner_seed,
            fold_n_jobs=fold_n_jobs,
            fixed_params=fixed_params,
            objective=objective,
            inner_protocol=inner_protocols[0],
        )
        yc_pred, yc_prob, yr_pred = _fit_predict_fold(
            prepared,
            family,
            tuned.best_classifier_params,
            tuned.best_regressor_params,
            train_idx,
            test_idx,
        )
        oof_parts.append(
            pd.DataFrame(
                {
                    "sample_idx": test_idx,
                    "protocol": "LeaveOneNetworkOut",
                    "fold_id": int(fold_id),
                    "y_pred_clf": yc_pred.astype(int),
                    "y_prob_clf": yc_prob.astype(float),
                    "y_pred_reg": yr_pred.astype(float),
                },
                columns=list(OOF_COLUMNS),
            )
        )

        for network_id in sorted(test_networks, key=str):
            mask = prepared.keys.iloc[test_idx]["network_id"].to_numpy() == network_id
            y_true = prepared.y_reg.iloc[test_idx].to_numpy(dtype=float)[mask]
            y_routed = np.where(yc_pred[mask] == 1, yr_pred[mask], 0.0)
            per_network_true.setdefault(str(network_id), []).append(y_true)
            per_network_pred.setdefault(str(network_id), []).append(y_routed)
        key_records = prepared.keys.iloc[test_idx]
        row_keys = tuple(
            {
                "sample_idx": int(sample_idx),
                "network_id": str(row.network_id),
                "run_id": str(row.run_id),
                "scenario_id": str(row.scenario_id),
                "node_id": str(row.node_id),
            }
            for sample_idx, row in zip(test_idx, key_records.itertuples())
        )
        fold_results.append(
            NestedNetworkFold(
                fold_id=int(fold_id),
                held_out_network_id=next(iter(test_networks)),
                outer_train_sample_idx=tuple(int(value) for value in train_idx),
                outer_test_sample_idx=tuple(int(value) for value in test_idx),
                outer_train_run_ids=_network_run_ids(prepared.keys.iloc[train_idx]),
                outer_test_run_ids=_network_run_ids(prepared.keys.iloc[test_idx]),
                inner_run_ids=view.run_ids,
                inner_prep_id=view.prep_id,
                optuna_study_id=str(tuned.study_id),
                selected_classifier_params=dict(tuned.best_classifier_params),
                selected_regressor_params=dict(tuned.best_regressor_params),
                objective=objective,
                inner_best_value=float(tuned.best_value),
                seed=inner_seed,
                trials=tuple(tuned.trials),
                oof_row_keys=row_keys,
            )
        )

    oof = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["fold_id", "sample_idx"], kind="stable"
    ).reset_index(drop=True)
    benchmark_metrics = score_predictions(
        prepared,
        oof,
        {"prep_id": prepared.prep_id},
        protocols=("LeaveOneNetworkOut",),
    )
    true_by_network = {
        key: np.concatenate(parts) for key, parts in per_network_true.items()
    }
    pred_by_network = {
        key: np.concatenate(parts) for key, parts in per_network_pred.items()
    }
    pooled_true = np.concatenate(list(true_by_network.values()))
    pooled_pred = np.concatenate(list(pred_by_network.values()))
    per_network_rmse = {
        key: float(np.sqrt(np.mean((true_by_network[key] - pred_by_network[key]) ** 2)))
        for key in true_by_network
    }
    metrics = {
        "objective": objective,
        "macro_network_rmse": macro_network_rmse(true_by_network, pred_by_network),
        "pooled_rmse": float(np.sqrt(np.mean((pooled_true - pooled_pred) ** 2))),
        "per_network_rmse": per_network_rmse,
        "benchmark": benchmark_metrics,
        "network_and_event": nested_network_event_metrics(
            prepared, oof, volume_band_edges_m3=volume_band_edges_m3
        ),
    }
    return NestedNetworkResult(
        family=family,
        prep_id=prepared.prep_id,
        objective=objective,
        volume_band_edges_m3=volume_band_edges_m3,
        oof=oof,
        folds=tuple(fold_results),
        metrics=metrics,
    )


def evaluate_nested_networks_from_config(
    prepared: PreparedDataset,
    family: str,
    inner_protocols: Sequence[str],
    config,
    *,
    seed: int = 42,
    fold_n_jobs: int = 1,
    fixed_params: dict | None = None,
) -> NestedNetworkResult:
    """Runs nested evaluation using the objective and tuning budget in Config.

    This is the application entry point for a loaded project config; it avoids
    copying the selection objective into a separate call-site argument.
    """
    bench = getattr(config, "bench", None)
    if bench is None:
        raise ValueError("Nested network evaluation requires a bench config block.")
    tuning = bench.tuning
    if tuning.objective is None:
        raise ValueError(
            "Nested network evaluation requires bench.tuning.objective."
        )
    return evaluate_nested_networks(
        prepared,
        family,
        inner_protocols,
        objective=tuning.objective,
        n_trials=tuning.n_trials,
        n_splits=tuning.n_splits,
        seed=seed,
        fold_n_jobs=fold_n_jobs,
        fixed_params=fixed_params,
        volume_band_edges_m3=bench.reporting.volume_band_edges_m3,
    )


def save_nested_network_result(
    result: NestedNetworkResult, output_dir: Path | str
) -> Path:
    """Persists an immutable nested run under a timestamped experiment ID."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reporting_config = {
        "event_unit": "shape_id (hydrograph shape; not event-family holdout)",
        "flood_threshold_m3": float(
            result.metrics["network_and_event"]["flood_threshold_m3"]
        ),
        "volume_band_edges_m3": list(result.volume_band_edges_m3),
    }
    reporting_config_sha256 = hashlib.sha256(
        json.dumps(reporting_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    experiment_id = (
        f"{timestamp}-{result.prep_id[:12]}-{reporting_config_sha256[:8]}"
    )
    experiment_dir = Path(output_dir) / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "family": result.family,
        "source_prep_id": result.prep_id,
        "outer_protocol": "LeaveOneNetworkOut",
        "inner_protocols": ["LeaveOneShapeOut"],
        "objective": result.objective,
        "network_aggregation": "equal_weight_per_network",
        "reporting_config": reporting_config,
        "reporting_config_sha256": reporting_config_sha256,
        "metrics": result.metrics,
        "folds": [asdict(fold) for fold in result.folds],
    }
    (experiment_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    result.oof.to_parquet(experiment_dir / "oof.parquet", index=False)
    return experiment_dir
