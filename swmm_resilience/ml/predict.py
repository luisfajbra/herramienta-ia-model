import hashlib
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from ..config import Config
from ..extraction.static_features import extract_static_features
from ..extraction.topology import compute_topology_features
from ..extraction.dynamic_features import compute_dynamic_features
from ..simulation.hydrograph_shapes import get_shape_stats
from ..simulation.swmm_api_io import load_inp
from .trainer import FEATURE_COLS


def base_shape_stats(inp_path: Path) -> tuple[float, float]:
    """Return (duracion_horas, tiempo_al_pico_h) of the .inp's own hydrograph.

    The training pipeline labels every base-shape row with these two values,
    so inference must reproduce them. Their default of 0.0 is a value that
    appears in no training row at all, and the regressor leans on both, so
    omitting them silently pushed predictions out of the trained domain.
    """
    inp = load_inp(inp_path)
    if "TIMESERIES" not in inp:
        return 0.0, 0.0
    series = next((list(ts.data) for ts in inp["TIMESERIES"].values() if ts.data), [])
    return get_shape_stats(series)


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
    clf = joblib.load(models_dir / "classifier.joblib")
    reg = joblib.load(models_dir / "regressor.joblib")

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
    merged = full_df.merge(dynamic_df, on="node_id", how="left")

    X = merged[FEATURE_COLS]
    inunda_pred = clf.predict(X)
    vol_pred = np.zeros(len(X))
    flood_mask = inunda_pred == 1
    if flood_mask.sum() > 0:
        vol_pred[flood_mask] = np.expm1(reg.predict(X.loc[flood_mask]))
        vol_pred = np.clip(vol_pred, a_min=0.0, a_max=None)

    merged["inunda_pred"] = inunda_pred
    merged["vol_pred_m3"] = vol_pred
    return merged[["node_id", "inunda_pred", "vol_pred_m3", "coord_x", "coord_y"]]
