"""
Scenario predictor: given a HydrographScenario, predict per-node flood results
using pre-trained classifier and regressor artifacts (joblib format).

ScenarioPredictor loads the models and computes static + topology features
once at construction; predict() only computes dynamic features and runs
inference, so per-scenario timing reflects true surrogate cost.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..extraction.dynamic_features import compute_scenario_dynamic_features
from ..extraction.static_features import extract_static_features
from ..extraction.topology import build_network_graph, compute_topology_features
from ..ml.trainer import FEATURE_COLS
from ..simulation.swmm_api_io import load_inp
from ..validation.hydrograph_csv import HydrographScenario


def _peak_inflow_lps(series: list[tuple[float, float]]) -> float:
    """Return the maximum flow value (lps) from a node's time series."""
    if not series:
        return 0.0
    return max(v for _, v in series)


class ScenarioPredictor:
    """Reusable predictor: load models and static features once, then call
    predict()/predict_timed() per scenario.

    Predictions cover every junction in the .inp network, not only the nodes
    present in the scenario CSV: junctions without a direct inflow get
    q_pico_nodo = 0 and accumulate upstream scenario peaks, consistent with
    how the training dataset was built.
    """

    def __init__(
        self,
        clf_path: Path,
        reg_path: Path,
        inp_path: Path,
        factor_range: tuple[float, float] | None = None,
    ) -> None:
        t0 = time.perf_counter()
        self.clf = joblib.load(clf_path)
        self.reg = joblib.load(reg_path)
        self.model_load_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        static_df = extract_static_features(inp_path)
        self.full_df = compute_topology_features(static_df, inp_path)
        self.graph, _ = build_network_graph(load_inp(inp_path))
        self.static_features_s = time.perf_counter() - t0

        self.inp_path = Path(inp_path)
        self.factor_range = factor_range
        self.node_ids: list[str] = self.full_df["node_id"].astype(str).tolist()

    # -- helpers -----------------------------------------------------------

    def _extrapolation_flags(self, peak_map: dict[str, float]) -> pd.Series:
        """True where the node's scenario peak falls outside the per-node
        range seen in training (base_inflow x [factor_min, factor_max]).

        Nodes with base inflow 0 are flagged only if they receive a direct
        scenario peak > 0 (training never saw q_pico_nodo > 0 for them).
        """
        if self.factor_range is None:
            return pd.Series(False, index=self.full_df.index)

        fmin, fmax = self.factor_range
        flags = []
        for _, row in self.full_df.iterrows():
            base = float(row.get("base_inflow_lps", 0.0) or 0.0)
            peak = float(peak_map.get(str(row["node_id"]), 0.0))
            if base > 0.0:
                ratio = peak / base
                flags.append(ratio < fmin or ratio > fmax)
            else:
                flags.append(peak > 0.0)
        return pd.Series(flags, index=self.full_df.index)

    def _probabilities(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.clf, "predict_proba"):
            return np.asarray(self.clf.predict_proba(X))[:, 1]
        return np.asarray(self.clf.predict(X), dtype=float)

    # -- public API --------------------------------------------------------

    def predict_timed(
        self, scenario: HydrographScenario
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Predict and return (result_df, timings).

        result_df columns: node_id, inunda_pred, prob_inunda, vol_pred_m3,
        extrapolated. timings keys: t_features_s, t_inference_s.
        """
        t0 = time.perf_counter()
        peak_map = {
            str(nid): _peak_inflow_lps(series)
            for nid, series in scenario.node_series.items()
        }
        dynamic_df = compute_scenario_dynamic_features(
            self.full_df, peak_map, self.graph
        )
        static_base = self.full_df.drop(
            columns=[
                c
                for c in ("factor_mult", "q_pico_nodo", "q_pico_acum_escalado")
                if c in self.full_df.columns
            ]
        )
        merged = static_base.merge(dynamic_df, on="node_id", how="left")
        X = merged[FEATURE_COLS]
        t_features = time.perf_counter() - t0

        t0 = time.perf_counter()
        inunda_pred = np.asarray(self.clf.predict(X)).astype(int)
        prob = self._probabilities(X)
        vol_pred = np.zeros(len(X))
        flood_mask = inunda_pred == 1
        if flood_mask.sum() > 0:
            vol_pred[flood_mask] = np.expm1(self.reg.predict(X.loc[flood_mask]))
        vol_pred = np.clip(vol_pred, a_min=0.0, a_max=None)
        t_inference = time.perf_counter() - t0

        result = pd.DataFrame(
            {
                "node_id": merged["node_id"].astype(str),
                "inunda_pred": pd.Series(inunda_pred, dtype=int),
                "prob_inunda": prob,
                "vol_pred_m3": vol_pred,
                "extrapolated": self._extrapolation_flags(peak_map).to_numpy(),
            }
        ).reset_index(drop=True)

        return result, {"t_features_s": t_features, "t_inference_s": t_inference}

    def predict(self, scenario: HydrographScenario) -> pd.DataFrame:
        result, _ = self.predict_timed(scenario)
        return result


def predict_scenario(
    scenario: HydrographScenario,
    clf_path: Path,
    reg_path: Path,
    flood_threshold_m3: float,
    inp_path: Path,
    factor_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """One-shot convenience wrapper around ScenarioPredictor.

    flood_threshold_m3 is informational: inunda_pred is decided by the
    classifier and vol_pred_m3 is reported as-is (documented reconciliation
    rule). Prefer ScenarioPredictor for batch use so models load once.
    """
    predictor = ScenarioPredictor(
        clf_path=clf_path,
        reg_path=reg_path,
        inp_path=inp_path,
        factor_range=factor_range,
    )
    return predictor.predict(scenario)
