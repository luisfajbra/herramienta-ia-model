from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from ..ml.predict import predict_network


def compute_resilience_curve(
    df_swmm: pd.DataFrame,
    factors: list[float],
    config,
    models_dir: Optional[Path],
    _predict_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """Compute resilience per factor for SWMM data and ML predictions.

    resilience = non-flooded nodes / total nodes (range 0-1).

    ML predictions are computed if:
      - _predict_fn is explicitly provided (used for testing), OR
      - config and models_dir are both non-None (production path, uses predict_network)
    Pass _predict_fn=None with config=None to compute SWMM-only (resilience_ml = NaN).
    """
    use_ml = _predict_fn is not None or (config is not None and models_dir is not None)
    actual_predict = _predict_fn if _predict_fn is not None else predict_network

    rows = []
    for factor in factors:
        df_f = df_swmm[abs(df_swmm["factor_mult"] - factor) < 1e-6]
        n_total = len(df_f)
        res_swmm = float((df_f["inunda"] == 0).sum()) / n_total if n_total > 0 else float("nan")

        if use_ml:
            pred = actual_predict(factor, config, models_dir)
            n_pred = len(pred)
            res_ml = float((pred["inunda_pred"] == 0).sum()) / n_pred if n_pred > 0 else float("nan")
        else:
            res_ml = float("nan")

        rows.append({"factor": factor, "resilience_swmm": res_swmm, "resilience_ml": res_ml})

    return pd.DataFrame(rows)
