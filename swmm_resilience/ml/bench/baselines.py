"""Referencias sin aprendizaje para el banco (hallazgo M-01).

Una referencia produce un frame OOF con el mismo contrato que
``evaluate_candidate`` y se puntúa con ``score_predictions``. Sirve para
saber cuánto de un resultado se debe a aprender y cuánto a que la tarea es
fácil: LOSO y GroupKFold5 retienen factores de intensidad, y una tabla de
consulta que interpola entre factores vecinos ya rinde muy bien.

- ``baseline_interpolacion_factor``: para una fila (forma, nodo, factor)
  retenida, interpola linealmente el volumen de la misma forma y nodo entre
  el factor de entrenamiento inferior y el superior más cercanos. Aplica a
  los protocolos agrupados por ``factor_mult``.
- ``baseline_forma_mas_cercana``: para una fila (forma retenida, nodo,
  factor), toma el volumen del mismo nodo y factor en la forma de
  entrenamiento más cercana en los descriptores de forma estandarizados.
  Aplica a los protocolos agrupados por ``shape_id``. Los descriptores salen
  de ``shape_descriptors`` (tabla indexada por ``shape_id``) o, si no se
  pasa, de ``duracion_horas`` y ``tiempo_al_pico_h`` en ``prepared.X``
  (contrato tabular_v3_17); contratos sin esas columnas, como tabular_v4_21,
  exigen ``shape_descriptors`` y, sin ella, la referencia no aplica.

Mallas incompletas. El filtro de continuidad puede retirar celdas (forma,
factor) del dataset evaluado, así que la malla (forma, nodo, factor) no es
completa y las referencias se puntúan sobre exactamente las filas que ven los
modelos. Reglas de respaldo (solo usan filas de entrenamiento del fold):

- Interpolación: cada serie (forma|nodo) usa SUS PROPIOS factores de
  entrenamiento disponibles. Interpola entre el inferior y el superior más
  cercanos de esa serie; si solo existe un lado, usa el vecino disponible
  (igual que en los extremos). Falla (ValueError, nombrando la serie) solo si
  la serie no tiene ningún valor de entrenamiento en el fold.
- Forma más cercana: las formas de entrenamiento se ordenan por distancia
  (desempate estable por identificador de forma) y cada fila usa la más
  cercana que TENGA la celda (nodo, factor); recorre el ranking hacia abajo.
  Falla solo si ninguna forma de entrenamiento tiene esa celda. Las
  estadísticas de estandarización son solo de las formas de entrenamiento.

Diagnóstico de respaldo: el frame que devuelve ``baseline_oof`` lleva
``attrs["n_rows"]`` (filas OOF) y ``attrs["fallback_rows"]`` (filas que
usaron una fuente no primaria). Definición, interpolación: filas cuyo par de
factores de entrenamiento de la serie difiere del par que habría usado con la
malla completa del fold (los factores de entrenamiento del fold inmediatamente
inferior y superior a ``f``; en los extremos de la malla del fold solo hay un
lado y eso NO cuenta como respaldo). Forma más cercana: filas que usaron una
forma distinta de la más cercana.

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


def _reject_duplicate_cells(cells: pd.DataFrame, columns: list[str], where: str) -> None:
    duplicated = cells.duplicated(columns, keep=False)
    if duplicated.any():
        examples = [
            tuple(row)
            for row in cells.loc[duplicated, columns].drop_duplicates().head(3).to_numpy()
        ]
        raise ValueError(
            f"{where}: hay celdas duplicadas en el train {columns}; ejemplos: {examples}."
        )


def _reject_nan_volumes(volumes: np.ndarray, where: str) -> None:
    if np.isnan(volumes).any():
        raise ValueError(f"{where}: hay volúmenes NaN en las filas de entrenamiento.")


def _interpolacion_factor_fold(
    prepared: PreparedDataset, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[np.ndarray, int]:
    where = INTERPOLACION_FACTOR
    series = np.char.add(np.char.add(_shape_ids(prepared), "|"), _node_ids(prepared))
    factors = _factors(prepared)
    volumes = prepared.y_reg.to_numpy(dtype=float)

    _reject_nan_volumes(volumes[train_idx], where)
    train_factors = np.unique(factors[train_idx])
    n_factors = len(train_factors)
    table = pd.DataFrame(
        {"series": series[train_idx], "factor": factors[train_idx], "vol": volumes[train_idx]}
    )
    _reject_duplicate_cells(table, ["series", "factor"], where)
    wide = table.pivot(index="series", columns="factor", values="vol").reindex(
        columns=train_factors
    )
    matrix = wide.to_numpy(dtype=float)
    available = ~np.isnan(matrix)

    row_of = pd.Series(np.arange(len(wide)), index=wide.index)
    rows = row_of.reindex(series[test_idx])
    if rows.isna().any():
        missing = sorted(set(series[test_idx][rows.isna().to_numpy()]))
        raise ValueError(
            f"{where}: las series {missing[:5]} del test no tienen ningún valor de "
            "entrenamiento en este fold."
        )
    rows = rows.to_numpy(dtype=int)

    # Posición (por serie) del último/primer factor disponible a la izquierda/derecha;
    # -1 y n_factors significan "no existe".
    positions = np.arange(n_factors)
    last_at_or_before = np.maximum.accumulate(np.where(available, positions, -1), axis=1)
    first_at_or_after = np.minimum.accumulate(
        np.where(available, positions, n_factors)[:, ::-1], axis=1
    )[:, ::-1]

    f_test = factors[test_idx]
    position = np.searchsorted(train_factors, f_test)      # primer factor de train >= f
    below = position - 1                                    # último factor de train < f
    expected_low = np.where(below >= 0, below, -1)
    expected_high = np.where(position < n_factors, position, n_factors)
    low = np.where(below >= 0, last_at_or_before[rows, np.clip(below, 0, None)], -1)
    high = np.where(
        position < n_factors,
        first_at_or_after[rows, np.clip(position, None, n_factors - 1)],
        n_factors,
    )

    has_low, has_high = low >= 0, high < n_factors
    empty = ~(has_low | has_high)
    if empty.any():
        names = sorted(set(series[test_idx][empty]))
        raise ValueError(
            f"{where}: las series {names[:5]} no tienen ningún valor de entrenamiento "
            "en este fold; no se puede interpolar."
        )

    low_c = np.clip(low, 0, n_factors - 1)
    high_c = np.clip(high, 0, n_factors - 1)
    f_low, f_high = train_factors[low_c], train_factors[high_c]
    v_low, v_high = matrix[rows, low_c], matrix[rows, high_c]
    both = has_low & has_high
    span = np.where(both, f_high - f_low, 1.0)
    weight = np.where(both, (f_test - f_low) / span, 0.0)
    interpolated = v_low * (1.0 - weight) + v_high * weight
    predicted = np.where(both, interpolated, np.where(has_low, v_low, v_high))

    fallback = int(((low != expected_low) | (high != expected_high)).sum())
    return predicted, fallback


def _reject_nan(has_nan: pd.DataFrame, where: str) -> None:
    """``has_nan``: booleanos (forma x descriptor). Falla nombrando formas y columnas."""
    if has_nan.to_numpy().any():
        shapes = [str(shape) for shape in has_nan.index[has_nan.any(axis=1)]]
        columns = [str(col) for col in has_nan.columns[has_nan.any(axis=0)]]
        raise ValueError(
            f"{where} tiene NaN en los descriptores de forma; formas: {shapes}; "
            f"columnas: {columns}."
        )


def _resolve_shape_descriptors(
    prepared: PreparedDataset, shape_descriptors: pd.DataFrame | None
) -> pd.DataFrame:
    """Tabla (forma -> descriptores numéricos) para la referencia de forma más cercana."""
    shapes = _shape_ids(prepared)
    if shape_descriptors is not None:
        table = shape_descriptors.copy()
        table.index = table.index.astype(str)
        if table.index.has_duplicates:
            duplicated = sorted(set(table.index[table.index.duplicated()]))
            raise ValueError(
                f"shape_descriptors tiene formas duplicadas en el índice: {duplicated}."
            )
        if table.shape[1] == 0:
            raise ValueError("shape_descriptors no tiene ninguna columna (sin columnas).")
        missing = sorted(set(shapes) - set(table.index))
        if missing:
            raise ValueError(
                "shape_descriptors no cubre todas las formas del dataset; faltan: "
                f"{missing}"
            )
        table = table.astype(float)
        _reject_nan(table.loc[sorted(set(shapes))].isna(), "shape_descriptors")
        return table
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
    _reject_nan(
        descriptors[list(_SHAPE_DESCRIPTORS)].isna().groupby(descriptors["shape"]).any(),
        "prepared.X",
    )
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
) -> tuple[np.ndarray, int]:
    where = FORMA_MAS_CERCANA
    shapes = _shape_ids(prepared)
    nodes = _node_ids(prepared)
    factors = _factors(prepared)
    volumes = prepared.y_reg.to_numpy(dtype=float)

    _reject_nan_volumes(volumes[train_idx], where)
    train_shapes = sorted(set(shapes[train_idx]))
    train_table = table.loc[train_shapes]
    mean = train_table.mean()
    std = train_table.std(ddof=0).replace(0.0, 1.0)
    z_train = (train_table - mean) / std

    # Ranking de formas de entrenamiento por distancia, estable por id de forma.
    test_shapes, inverse = np.unique(shapes[test_idx], return_inverse=True)
    ranking = []
    for shape in test_shapes:
        z_shape = (table.loc[shape] - mean) / std
        distance = np.sqrt(((z_train - z_shape) ** 2).sum(axis=1))
        ranking.append(distance.sort_values(kind="stable").index.to_numpy())
    ranking = np.array(ranking, dtype=object)

    cells = pd.DataFrame(
        {"shape": shapes[train_idx], "node": nodes[train_idx], "factor": factors[train_idx]}
    )
    _reject_duplicate_cells(cells, ["shape", "node", "factor"], where)
    lookup = pd.Series(volumes[train_idx], index=pd.MultiIndex.from_frame(cells))

    test_nodes, test_factors = nodes[test_idx], factors[test_idx]
    predicted = np.full(len(test_idx), np.nan)
    used_rank = np.full(len(test_idx), -1)
    for rank in range(len(train_shapes)):
        pending = np.flatnonzero(np.isnan(predicted))
        if pending.size == 0:
            break
        wanted = pd.MultiIndex.from_arrays(
            [ranking[inverse[pending], rank], test_nodes[pending], test_factors[pending]]
        )
        found = lookup.reindex(wanted).to_numpy(dtype=float)
        hit = ~np.isnan(found)
        predicted[pending[hit]] = found[hit]
        used_rank[pending[hit]] = rank

    pending = np.flatnonzero(np.isnan(predicted))
    if pending.size:
        examples = sorted(
            {
                (str(shapes[test_idx][i]), str(test_nodes[i]), float(test_factors[i]))
                for i in pending
            }
        )[:3]
        raise ValueError(
            f"{where}: ninguna forma de entrenamiento tiene la celda (nodo, factor) "
            f"requerida; ejemplos (forma, nodo, factor): {examples}."
        )
    return predicted, int((used_rank > 0).sum())


def baseline_oof(
    prepared: PreparedDataset,
    name: str,
    flood_threshold_m3: float,
    shape_descriptors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Frame OOF de la referencia ``name`` para los protocolos donde aplica.

    ``shape_descriptors`` (índice = shape_id) solo lo usa la referencia de forma
    más cercana; ver el docstring del módulo. El frame lleva
    ``attrs["n_rows"]`` y ``attrs["fallback_rows"]`` (definición en el docstring).
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
    fallback_rows = 0
    for (protocol, fold_id), chunk in prepared.folds.groupby(["protocol", "fold_id"]):
        if protocol not in protocols:
            continue
        train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"].to_numpy()
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"].to_numpy()
        volume, fold_fallback = predict_fold(prepared, train_idx, test_idx)
        fallback_rows += fold_fallback
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
    oof = pd.concat(records, ignore_index=True)
    oof.attrs["n_rows"] = int(len(oof))
    oof.attrs["fallback_rows"] = int(fallback_rows)
    return oof


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
