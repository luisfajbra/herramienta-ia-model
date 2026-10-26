"""Fábrica mínima de PreparedDataset para pruebas del banco (sin SQL)."""

from __future__ import annotations

import pandas as pd

from swmm_resilience.ml.bench.folds import build_all_folds
from swmm_resilience.ml.bench.schemas import KEY_COLUMNS, PreparedDataset
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17, TABULAR_V3_17

PREP_ID = "0123456789abcdef"


def make_prepared(rows: list[dict], protocols, *, threshold: float = 1.0) -> PreparedDataset:
    """Construye un PreparedDataset a partir de filas ya expandidas.

    Cada fila: ``shape_id, node_id, factor_mult, vol, duracion_horas, tiempo_al_pico_h``.
    ``inunda`` = ``vol >= threshold``. Las demás features valen 0.0.
    """
    frame = pd.DataFrame(rows).reset_index(drop=True)
    pairs = sorted(set(zip(frame["shape_id"], frame["factor_mult"])))
    scenario_id = {pair: index + 1 for index, pair in enumerate(pairs)}
    ids = [scenario_id[(s, f)] for s, f in zip(frame["shape_id"], frame["factor_mult"])]
    keys = pd.DataFrame(
        {
            "run_id": ids,
            "network_id": 1,
            "scenario_id": ids,
            "scenario_key": [
                f"{s}@{f}" for s, f in zip(frame["shape_id"], frame["factor_mult"])
            ],
            "scenario_kind": "base",
            "node_id": frame["node_id"],
            "factor_mult": frame["factor_mult"].astype(float),
            "shape_id": frame["shape_id"],
        },
        columns=list(KEY_COLUMNS),
    )
    X = pd.DataFrame(0.0, index=range(len(frame)), columns=list(FEATURE_COLUMNS_V17))
    X["duracion_horas"] = frame["duracion_horas"].to_numpy(dtype=float)
    X["tiempo_al_pico_h"] = frame["tiempo_al_pico_h"].to_numpy(dtype=float)
    volume = frame["vol"].to_numpy(dtype=float)
    folds = build_all_folds(keys, list(protocols))
    return PreparedDataset(
        prep_id=PREP_ID,
        keys=keys,
        X=X,
        y_clf=pd.Series((volume >= threshold).astype(int), name="inunda"),
        y_reg=pd.Series(volume, name="vol_inundacion_m3"),
        folds=folds,
        manifest={
            "prep_id": PREP_ID,
            "protocols": list(protocols),
            "contract_id": TABULAR_V3_17.contract_id,
            "feature_contract_sha256": TABULAR_V3_17.descriptor_sha256,
            "feature_columns": list(FEATURE_COLUMNS_V17),
        },
        quality={},
    )


def perfect_oof(prepared: PreparedDataset) -> pd.DataFrame:
    """Predicciones fuera de fold perfectas para todos los folds de test."""
    tests = prepared.folds[prepared.folds["split"] == "test"]
    idx = tests["sample_idx"].to_numpy()
    return pd.DataFrame(
        {
            "sample_idx": idx,
            "protocol": tests["protocol"].to_numpy(),
            "fold_id": tests["fold_id"].to_numpy(),
            "y_pred_clf": prepared.y_clf.iloc[idx].to_numpy(),
            "y_prob_clf": prepared.y_clf.iloc[idx].to_numpy().astype(float),
            "y_pred_reg": prepared.y_reg.iloc[idx].to_numpy(),
        }
    )
