from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from ..ml.predict import predict_network


def compute_flood_volume_curve(
    df_swmm: pd.DataFrame,
    factors: list[float],
    config,
    models_dir: Optional[Path],
    _predict_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """Compute total flooded volume per factor for SWMM data and ML predictions.

    vol_total = sum of vol_inundacion_m3 (or vol_pred_m3) across all nodes.

    ML predictions are computed if:
      - _predict_fn is explicitly provided (used for testing), OR
      - config and models_dir are both non-None (production path, uses predict_network)
    Pass _predict_fn=None with config=None for SWMM-only (vol_total_ml = NaN).
    """
    use_ml = _predict_fn is not None or (config is not None and models_dir is not None)
    actual_predict = _predict_fn if _predict_fn is not None else predict_network

    rows = []
    for factor in factors:
        df_f = df_swmm[abs(df_swmm["factor_mult"] - factor) < 1e-6]
        vol_swmm = float(df_f["vol_inundacion_m3"].sum())

        if use_ml:
            pred = actual_predict(factor, config, models_dir)
            vol_ml = float(pred["vol_pred_m3"].sum())
        else:
            vol_ml = float("nan")

        rows.append({"factor": factor, "vol_total_swmm": vol_swmm, "vol_total_ml": vol_ml})

    return pd.DataFrame(rows)
