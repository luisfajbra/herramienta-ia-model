"""Referencias sin aprendizaje para el banco (hallazgo M-01).

Una referencia produce un frame OOF con el mismo contrato que
``evaluate_candidate`` y se puntúa con ``score_predictions``. Sirve para
saber cuánto de un resultado se debe a aprender y cuánto a que la tarea es
fácil: LOSO y GroupKFold5 retienen factores de intensidad, y una tabla de
consulta que interpola entre factores vecinos ya rinde muy bien.

- ``baseline_interpolacion_factor``: para una fila (forma, nodo, factor)
  retenida, interpola linealmente el volumen de la misma forma y nodo entre
  los factores vecinos del entrenamiento (en los extremos, el vecino más
  cercano). Aplica a los protocolos agrupados por ``factor_mult``.
- ``baseline_forma_mas_cercana``: para una fila (forma retenida, nodo,
  factor), toma el volumen del mismo nodo y factor en la forma de
  entrenamiento más cercana en los descriptores de forma estandarizados.
  Aplica a los protocolos agrupados por ``shape_id``. Los descriptores salen
  de ``shape_descriptors`` (tabla indexada por ``shape_id``) o, si no se
  pasa, de ``duracion_horas`` y ``tiempo_al_pico_h`` en ``prepared.X``
  (contrato tabular_v3_17); contratos sin esas columnas, como tabular_v4_21,
  exigen ``shape_descriptors`` y, sin ella, la referencia no aplica.

La etiqueta predicha es ``volumen_predicho >= flood_threshold_m3`` y la
"probabilidad" es esa misma etiqueta (0/1): el AUC-ROC de una referencia sin
probabilidades reales no es interpretable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import score_predictions
from .schemas import OOF_COLUMNS, PreparedDataset, group_column_for

INTERPOLACION_FACTOR = "baseline_interpolacion_factor"
FORMA_MAS_CERCANA = "baseline_forma_mas_cercana"
BASELINES = (INTERPOLACION_FACTOR, FORMA_MAS_CERCANA)

_SHAPE_DESCRIPTORS = ("duracion_horas", "tiempo_al_pico_h")
_FACTOR_DECIMALS = 6


class BaselineNotApplicable(ValueError):
    """La referencia no aplica a ningún protocolo del PreparedDataset."""


def _applicable_protocols(prepared: PreparedDataset, name: str) -> list[str]:
    wanted = "factor_mult" if name == INTERPOLACION_FACTOR else "shape_id"
    protocols = sorted(prepared.folds["protocol"].unique())
    return [protocol for protocol in protocols if group_column_for(protocol) == wanted]


def _node_ids(prepared: PreparedDataset) -> np.ndarray:
    return prepared.keys["node_id"].astype(str).to_numpy()


def _shape_ids(prepared: PreparedDataset) -> np.ndarray:
    return prepared.keys["shape_id"].astype(str).to_numpy()


def _factors(prepared: PreparedDataset) -> np.ndarray:
    return prepared.keys["factor_mult"].to_numpy(dtype=float).round(_FACTOR_DECIMALS)


def _interpolacion_factor_fold(
    prepared: PreparedDataset, train_idx: np.ndarray, test_idx: np.ndarray
) -> np.ndarray:
    series = np.char.add(np.char.add(_shape_ids(prepared), "|"), _node_ids(prepared))
    factors = _factors(prepared)
    volumes = prepared.y_reg.to_numpy(dtype=float)

    train_factors = np.unique(factors[train_idx])
    table = pd.DataFrame(
        {"series": series[train_idx], "factor": factors[train_idx], "vol": volumes[train_idx]}
    )
    wide = table.pivot(index="series", columns="factor", values="vol").reindex(
        columns=train_factors
    )
    if wide.isna().any().any():
        raise ValueError(
            "baseline_interpolacion_factor: hay series (forma, nodo) sin todos los "
            "factores de entrenamiento; no se puede interpolar."
        )
    row_of = pd.Series(np.arange(len(wide)), index=wide.index)
    rows = row_of.reindex(series[test_idx])
    if rows.isna().any():
        raise ValueError(
            "baseline_interpolacion_factor: una serie (forma, nodo) del test no está en el train."
        )
    rows = rows.to_numpy(dtype=int)
    matrix = wide.to_numpy(dtype=float)

    f_test = factors[test_idx]
    position = np.searchsorted(train_factors, f_test)
    low = np.clip(position - 1, 0, len(train_factors) - 1)
    high = np.clip(position, 0, len(train_factors) - 1)
    f_low, f_high = train_factors[low], train_factors[high]
    span = f_high - f_low
    weight = np.divide(f_test - f_low, span, out=np.zeros_like(f_test), where=span > 0)
    return matrix[rows, low] * (1.0 - weight) + matrix[rows, high] * weight


def _resolve_shape_descriptors(
    prepared: PreparedDataset, shape_descriptors: pd.DataFrame | None
) -> pd.DataFrame:
    """Tabla (forma -> descriptores numéricos) para la referencia de forma más cercana."""
    shapes = _shape_ids(prepared)
    if shape_descriptors is not None:
        table = shape_descriptors.copy()
        table.index = table.index.astype(str)
        missing = sorted(set(shapes) - set(table.index))
        if missing:
            raise ValueError(
                "shape_descriptors no cubre todas las formas del dataset; faltan: "
                f"{missing}"
            )
        return table.astype(float)
    if not all(column in prepared.X.columns for column in _SHAPE_DESCRIPTORS):
        raise BaselineNotApplicable(
            f"{FORMA_MAS_CERCANA} no aplica: el contrato de features de este "
            f"PreparedDataset no incluye los descriptores de forma {_SHAPE_DESCRIPTORS} "
            "y no se pasó shape_descriptors."
        )
    descriptors = pd.DataFrame(
        {"shape": shapes, **{c: prepared.X[c].to_numpy(dtype=float) for c in _SHAPE_DESCRIPTORS}}
    )
    grouped = descriptors.groupby("shape")[list(_SHAPE_DESCRIPTORS)]
    if (grouped.nunique() > 1).any().any():
        raise ValueError(
            "baseline_forma_mas_cercana: las features de forma no son constantes dentro de una forma."
        )
    return grouped.first()


def _forma_mas_cercana_fold(
    prepared: PreparedDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    table: pd.DataFrame,
) -> np.ndarray:
    shapes = _shape_ids(prepared)
    nodes = _node_ids(prepared)
    factors = _factors(prepared)
    volumes = prepared.y_reg.to_numpy(dtype=float)

    train_shapes = sorted(set(shapes[train_idx]))
    test_shapes = sorted(set(shapes[test_idx]))
    train_table = table.loc[train_shapes]
    mean = train_table.mean()
    std = train_table.std(ddof=0).replace(0.0, 1.0)
    z_train = (train_table - mean) / std

    nearest = {}
    for shape in test_shapes:
        z_shape = (table.loc[shape] - mean) / std
        distance = np.sqrt(((z_train - z_shape) ** 2).sum(axis=1))
        nearest[shape] = distance.idxmin()

    index = pd.MultiIndex.from_arrays(
        [shapes[train_idx], nodes[train_idx], factors[train_idx]]
    )
    lookup_table = pd.Series(volumes[train_idx], index=index)
    if lookup_table.index.has_duplicates:
        raise ValueError(
            "baseline_forma_mas_cercana: hay (forma, nodo, factor) duplicados en el train."
        )
    wanted = pd.MultiIndex.from_arrays(
        [[nearest[shape] for shape in shapes[test_idx]], nodes[test_idx], factors[test_idx]]
    )
    predicted = lookup_table.reindex(wanted)
    if predicted.isna().any():
        raise ValueError(
            "baseline_forma_mas_cercana: la forma vecina no tiene todos los (nodo, factor) del test."
        )
    return predicted.to_numpy(dtype=float)


def baseline_oof(
    prepared: PreparedDataset,
    name: str,
    flood_threshold_m3: float,
    shape_descriptors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Frame OOF de la referencia ``name`` para los protocolos donde aplica.

    ``shape_descriptors`` (índice = shape_id) solo lo usa la referencia de forma
    más cercana; ver el docstring del módulo.
    """
    if name not in BASELINES:
        raise ValueError(f"Referencia desconocida: {name!r}. Opciones: {BASELINES}")
    protocols = _applicable_protocols(prepared, name)
    if not protocols:
        available = ", ".join(sorted(prepared.folds["protocol"].unique()))
        raise BaselineNotApplicable(
            f"{name} no aplica a ningún protocolo de este PreparedDataset ({available})."
        )
    if name == INTERPOLACION_FACTOR:
        predict_fold = _interpolacion_factor_fold
    else:
        table = _resolve_shape_descriptors(prepared, shape_descriptors)

        def predict_fold(prepared, train_idx, test_idx):
            return _forma_mas_cercana_fold(prepared, train_idx, test_idx, table)

    records = []
    for (protocol, fold_id), chunk in prepared.folds.groupby(["protocol", "fold_id"]):
        if protocol not in protocols:
            continue
        train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"].to_numpy()
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"].to_numpy()
        volume = predict_fold(prepared, train_idx, test_idx)
        flooded = (volume >= flood_threshold_m3).astype(int)
        records.append(
            pd.DataFrame(
                {
                    "sample_idx": test_idx,
                    "protocol": protocol,
                    "fold_id": fold_id,
                    "y_pred_clf": flooded,
                    "y_prob_clf": flooded.astype(float),
                    "y_pred_reg": volume.astype(float),
                },
                columns=list(OOF_COLUMNS),
            )
        )
    return pd.concat(records, ignore_index=True)


def evaluate_baseline(
    prepared: PreparedDataset,
    name: str,
    flood_threshold_m3: float,
    shape_descriptors: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Produce el OOF de la referencia y lo puntúa como a cualquier candidato."""
    oof = baseline_oof(prepared, name, flood_threshold_m3, shape_descriptors)
    protocols = tuple(sorted(oof["protocol"].unique()))
    metrics = score_predictions(
        prepared, oof, {"prep_id": prepared.prep_id}, protocols=protocols
    )
    return oof, metrics
