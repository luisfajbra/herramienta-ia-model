import hashlib
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from ..config import Config
from ..extraction.static_features import extract_static_features
from ..extraction.topology import build_network_graph, compute_topology_features
from ..extraction.dynamic_features import (
    compute_dynamic_features, compute_tributary_dynamic_features,
)
from ..simulation.hydrograph_shapes import apply_shape, get_shape_stats
from ..simulation.swmm_api_io import load_inp
from .contracts import TABULAR_V4_21
from .trainer import FEATURE_COLS


def _base_shape_series(inp_path: Path) -> list[tuple[float, float]]:
    """Return the .inp's own hydrograph series (raw L/s values, not normalized).

    The first non-empty TIMESERIES defined in the .inp -- same lookup rule
    used (until now) only by base_shape_stats, and now also by
    _normalized_base_shape() below.
    """
    inp = load_inp(inp_path)
    if "TIMESERIES" not in inp:
        return []
    return next((list(ts.data) for ts in inp["TIMESERIES"].values() if ts.data), [])


def base_shape_stats(inp_path: Path) -> tuple[float, float]:
    """Return (duracion_horas, tiempo_al_pico_h) of the .inp's own hydrograph.

    The training pipeline labels every base-shape row with these two values,
    so inference must reproduce them. Their default of 0.0 is a value that
    appears in no training row at all, and the regressor leans on both, so
    omitting them silently pushed predictions out of the trained domain.
    """
    return get_shape_stats(_base_shape_series(inp_path))


def _normalized_base_shape(inp_path: Path) -> list[tuple[float, float]]:
    """Normalize the .inp's own hydrograph to q_norm peak == 1.0.

    Feeds apply_shape() so predict_network can reconstruct a per-node
    node_series (base_inflow_lps * factor * q_norm) and compute the six
    tabular_v4_21 tributary features with the SAME function training uses
    (compute_tributary_dynamic_features) -- spec §6's "mismo validador y
    función para entrenamiento e inferencia", now also honoured by the
    scalar-factor prediction path (ScenarioPredictor already did this for
    arbitrary hydrograph CSVs). Empty/zero-peak series return [] -- callers
    build an empty node_series from that, same as a node with no own inflow.
    """
    series = _base_shape_series(inp_path)
    if not series:
        return []
    peak = max(v for _, v in series)
    if peak <= 0:
        return []
    return [(t, v / peak) for t, v in series]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict_network(factor: float, config: Config, models_dir: Path) -> pd.DataFrame:
    """Predict flooding for all junction nodes at a given factor without running SWMM.

    Validates that the current .inp matches the one used at training time (MD5 hash).
    Returns DataFrame: node_id, inunda_pred, vol_pred_m3, coord_x, coord_y
    """
    # Import perezoso: promote arrastra bench.train -> registry -> torch.
    from .bench.promote import assert_promotion_consistent

    assert_promotion_consistent(models_dir)
    clf = joblib.load(models_dir / "classifier.joblib")
    reg = joblib.load(models_dir / "regressor.joblib")

    if clf.n_features_in_ != len(FEATURE_COLS) or reg.n_features_in_ != len(FEATURE_COLS):
        raise RuntimeError(
            f"El modelo en '{models_dir}' espera {clf.n_features_in_}/{reg.n_features_in_} "
            f"features (clf/reg) pero FEATURE_COLS tiene {len(FEATURE_COLS)}. "
            "El modelo activo no corresponde al contrato de features actual -- "
            "vuelve a promoverlo (--bench-promote o el script de migración correspondiente)."
        )

    stored_hash = (models_dir / "training_inp_hash.txt").read_text().strip()
    if _md5(config.network.inp_path) != stored_hash:
        raise ValueError(
            f"El .inp en '{config.network.inp_path}' ha cambiado desde el entrenamiento. "
            "Re-entrena el modelo o usa el .inp original."
        )

    if not (config.simulation.factor_min <= factor <= config.simulation.factor_max):
        print(
            f"ADVERTENCIA: factor={factor} fuera del rango de entrenamiento "
            f"[{config.simulation.factor_min}, {config.simulation.factor_max}] — extrapolación no validada"
        )

    static_df = extract_static_features(config.network.inp_path)
    full_df = compute_topology_features(static_df, config.network.inp_path)
    duracion_horas, tiempo_al_pico_h = base_shape_stats(config.network.inp_path)
    dynamic_df = compute_dynamic_features(
        full_df,
        factor,
        duracion_horas=duracion_horas,
        tiempo_al_pico_h=tiempo_al_pico_h,
    )

    node_ids = full_df["node_id"].astype(str)
    normalized_shape = _normalized_base_shape(config.network.inp_path)
    base_inflows = dict(zip(node_ids, full_df["base_inflow_lps"]))
    node_series = apply_shape(normalized_shape, base_inflows, factor) if normalized_shape else {}
    graph, _ = build_network_graph(load_inp(config.network.inp_path))
    capacity_by_node = dict(zip(node_ids, full_df["upstream_capacity_lps"]))
    tributary_df = compute_tributary_dynamic_features(
        list(node_ids), node_series, graph, capacity_by_node,
    )

    merged = (
        full_df.merge(dynamic_df, on="node_id", how="left")
        .merge(tributary_df, on="node_id", how="left")
    )

    X = TABULAR_V4_21.validate_frame(merged[FEATURE_COLS])
    inunda_pred = clf.predict(X)
    vol_pred = np.zeros(len(X))
    flood_mask = inunda_pred == 1
    if flood_mask.sum() > 0:
        vol_pred[flood_mask] = np.expm1(reg.predict(X.loc[flood_mask]))
        vol_pred = np.clip(vol_pred, a_min=0.0, a_max=None)

    merged["inunda_pred"] = inunda_pred
    merged["vol_pred_m3"] = vol_pred
    return merged[["node_id", "inunda_pred", "vol_pred_m3", "coord_x", "coord_y"]]
