"""Batch coordinator for hydrograph CSV validation against SWMM simulations.

Responsibilities:
1. Validate the base .inp (flow units, ponding, training hash).
2. Glob a directory for *.csv hydrograph files and load each scenario.
3. Write a scenario-specific SWMM .inp file (with drain-down buffer).
4. Run SWMM and parse flooding truth via the shared label module
   (all junctions, not only inflow nodes).
5. Run ML prediction with a ScenarioPredictor (models loaded once).
6. Build per-scenario comparison DataFrames, plots and metrics.
7. Persist comparison_summary.csv, scenario_totals.csv, timings.csv and
   metrics_per_scenario.csv; return an aggregate summary dict.
"""

from __future__ import annotations

import hashlib
import re
import time
import warnings
from pathlib import Path

import pandas as pd

from ..analysis.model_comparison import (
    build_comparison_df,
    compute_classification_metrics,
    compute_conditional_volume_metrics,
    compute_per_node_r2,
    compute_pr_auc,
    compute_volume_metrics,
)
from ..extraction.labels import extract_labels
from ..ml.scenario_predict import ScenarioPredictor
from ..simulation.swmm_api_io import get_node_timeseries_map, load_inp
from ..simulation.timeseries_scenario import write_scenario_inp
from ..validation.hydrograph_csv import HydrographScenario, load_scenario
from ..visualization.hydrograph import plot_scenario_hydrograph
from ..visualization.model_comparison import (
    plot_node_profiles,
    plot_parity_aggregated,
    plot_parity_nodes,
    plot_scenario_flood_maps,
    plot_totals_comparison,
)

CONTINUITY_ERROR_WARN_PCT = 5.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expected_nodes_from_inp(base_inp_path: Path) -> set[str]:
    """Return nodes with FLOW timeseries references in the base SWMM file."""
    node_timeseries = get_node_timeseries_map(load_inp(base_inp_path))
    if not node_timeseries:
        raise ValueError(
            f"No FLOW timeseries references were found in [INFLOWS]: {base_inp_path}"
        )
    return set(node_timeseries)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_option(inp, key: str) -> str | None:
    """Tolerant accessor for OPTIONS values across swmm_api versions."""
    try:
        if "OPTIONS" not in inp:
            return None
        opts = inp["OPTIONS"]
        value = opts.get(key) if hasattr(opts, "get") else opts[key]
    except Exception:
        return None
    return None if value is None else str(value).strip()


def _validate_base_inp(
    base_inp_path: Path,
    clf_path: Path,
    allow_inp_mismatch: bool,
) -> None:
    """Validity guards: abort on wrong units or training-network mismatch.

    - FLOW_UNITS must be LPS: the CSV contract is value_lps and the .rpt
      volume conversion (10^6 L -> m³) assumes SI units. Any other unit
      system would silently corrupt every number downstream.
    - ALLOW_PONDING changes the .rpt column semantics and volume accounting:
      warn so results are interpreted accordingly.
    - The base .inp must be the network the models were trained on (MD5 in
      training_inp_hash.txt next to the classifier); otherwise the
      comparison measures noise.
    """
    inp = load_inp(base_inp_path)

    flow_units = _get_option(inp, "FLOW_UNITS")
    if flow_units is not None and flow_units.upper() != "LPS":
        raise ValueError(
            f"FLOW_UNITS='{flow_units}' en el .inp base; el flujo de validación "
            "requiere LPS (contrato value_lps y conversión 10^6 L → m³)."
        )

    ponding = _get_option(inp, "ALLOW_PONDING")
    if ponding is not None and ponding.upper() in ("YES", "TRUE", "1"):
        warnings.warn(
            "ALLOW_PONDING está activado en el .inp base: el 'Node Flooding "
            "Summary' agrega la columna de volumen estancado y cambia la "
            "contabilidad de volúmenes. Interpretar resultados con cautela.",
            stacklevel=2,
        )

    hash_file = Path(clf_path).parent / "training_inp_hash.txt"
    if not hash_file.exists():
        warnings.warn(
            f"No se encontró {hash_file}; no se puede verificar que el .inp "
            "base coincida con la red de entrenamiento.",
            stacklevel=2,
        )
        return
    stored = hash_file.read_text().strip()
    current = _md5(Path(base_inp_path))
    if stored and current != stored:
        msg = (
            "El .inp base no coincide con la red usada para entrenar los "
            f"modelos (hash {current} != {stored}). La validación mediría una "
            "red distinta a la del modelo."
        )
        if allow_inp_mismatch:
            warnings.warn(msg + " Continuando por --allow-inp-mismatch.", stacklevel=2)
        else:
            raise ValueError(msg + " Usa --allow-inp-mismatch para forzar.")


def _make_predictor(
    clf_path: Path,
    reg_path: Path,
    inp_path: Path,
    factor_range: tuple[float, float] | None,
) -> ScenarioPredictor:
    """Factory seam (patchable in tests)."""
    return ScenarioPredictor(
        clf_path=clf_path,
        reg_path=reg_path,
        inp_path=inp_path,
        factor_range=factor_range,
    )


def _run_swmm(scenario_inp_path: Path) -> Path:
    """Run a SWMM simulation on *scenario_inp_path* and return the .rpt path.

    Uses ``pyswmm.Simulation`` directly — the .inp is already scenario-specific
    and does not need further scaling.
    """
    from pyswmm import Simulation

    with Simulation(str(scenario_inp_path)) as sim:
        for _ in sim:
            pass

    rpt_path = scenario_inp_path.with_suffix(".rpt")
    if not rpt_path.exists():
        raise FileNotFoundError(
            f"SWMM did not generate the expected .rpt file: {rpt_path}"
        )
    return rpt_path


_CONTINUITY_RE = re.compile(
    r"Continuity Error \(%\)\s*\.*\s*(-?\d+(?:\.\d+)?)"
)


def _read_continuity_error(rpt_path: Path) -> float | None:
    """Worst (max |value|) continuity error (%) reported in the .rpt.

    A routing continuity error above ~5 % means SWMM itself does not trust
    its mass balance, so the 'truth' side of the comparison is unreliable.
    """
    try:
        text = Path(rpt_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    values = [float(v) for v in _CONTINUITY_RE.findall(text)]
    if not values:
        return None
    return max(values, key=abs)


def _build_swmm_df(
    all_node_ids: list[str],
    rpt_path: Path,
    flood_threshold_m3: float,
) -> pd.DataFrame:
    """SWMM truth over ALL junctions via the shared label module.

    Nodes absent from the 'Node Flooding Summary' get zero volume. Returns
    DataFrame: node_id, inunda_swmm, vol_swmm_m3.
    """
    labels_df = extract_labels(rpt_path, all_node_ids, flood_threshold_m3)
    return labels_df.rename(
        columns={"vol_inundacion_m3": "vol_swmm_m3", "inunda": "inunda_swmm"}
    )[["node_id", "inunda_swmm", "vol_swmm_m3"]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_batch_validation(
    csv_dir: Path,
    base_inp_path: Path,
    clf_path: Path,
    reg_path: Path,
    flood_threshold_m3: float,
    out_dir: Path,
    expected_nodes: set[str] | None = None,
    drain_down_hours: float = 6.0,
    allow_inp_mismatch: bool = False,
    factor_range: tuple[float, float] | None = None,
) -> dict:
    """Run batch validation of ML predictions against SWMM simulations.

    Parameters
    ----------
    csv_dir:
        Directory containing ``*.csv`` hydrograph scenario files.
    base_inp_path:
        Path to the base (unmodified) SWMM ``.inp`` network file.
    clf_path / reg_path:
        Paths to the joblib-serialised classifier / regressor.
    flood_threshold_m3:
        Minimum flood volume (m³) for a node to count as flooded (applied via
        the shared label rule, vol >= threshold).
    out_dir:
        Root output directory. Sub-directories ``inp/`` and ``plots/`` are
        created automatically.
    expected_nodes:
        Optional set of node IDs that every CSV must contain (defaults to the
        FLOW-inflow nodes of the base .inp).
    drain_down_hours:
        Zero-inflow buffer appended after the last CSV point so the network
        drains before END_TIME (avoids truncated flood volumes).
    allow_inp_mismatch:
        Continue (with a warning) when the base .inp hash does not match the
        training hash stored next to the classifier.
    factor_range:
        (factor_min, factor_max) of the training sweep; used to flag
        per-node extrapolation. None disables flagging.

    Returns
    -------
    dict with keys:
        - ``n_scenarios`` (int)
        - ``classification`` (pooled, incl. csi)
        - ``volume`` (pooled) and ``volume_flooded_only`` (conditional)
        - ``pr_auc`` (float | None)
        - ``per_node_r2`` (dict[str, float | None])
        - ``per_scenario`` (list[dict])
        - ``timings`` (list[dict])
        - ``summary_csv_path``, ``scenario_totals_csv_path``,
          ``timings_csv_path``, ``metrics_per_scenario_csv_path`` (Path)
    """
    csv_dir = Path(csv_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp_dir = out_dir / "inp"
    inp_dir.mkdir(parents=True, exist_ok=True)

    _validate_base_inp(base_inp_path, clf_path, allow_inp_mismatch)

    csv_files = sorted(csv_dir.glob("*.csv"))
    if expected_nodes is None:
        expected_nodes = _expected_nodes_from_inp(base_inp_path)

    predictor = _make_predictor(clf_path, reg_path, base_inp_path, factor_range)
    all_node_ids = list(predictor.node_ids)

    all_comp_dfs: list[pd.DataFrame] = []
    all_records: list[dict] = []
    pooled_truth: list[int] = []
    pooled_probs: list[float] = []
    per_scenario: list[dict] = []
    timing_rows: list[dict] = []

    for csv_path in csv_files:
        # Step a: load scenario
        scenario = load_scenario(csv_path, expected_nodes)

        # Step b: write scenario-specific .inp (with drain-down buffer)
        t0 = time.perf_counter()
        scenario_inp_path = write_scenario_inp(
            base_inp_path, scenario, inp_dir, drain_down_hours=drain_down_hours
        )
        t_write_inp = time.perf_counter() - t0

        # Step c: run SWMM
        t0 = time.perf_counter()
        rpt_path = _run_swmm(scenario_inp_path)
        t_swmm = time.perf_counter() - t0

        # Step d: SWMM truth over all junctions + continuity check
        t0 = time.perf_counter()
        swmm_df = _build_swmm_df(all_node_ids, rpt_path, flood_threshold_m3)
        t_parse_rpt = time.perf_counter() - t0

        continuity_pct = _read_continuity_error(rpt_path)
        if continuity_pct is not None and abs(continuity_pct) > CONTINUITY_ERROR_WARN_PCT:
            warnings.warn(
                f"[{scenario.scenario_id}] Error de continuidad SWMM "
                f"{continuity_pct:.2f}% (> {CONTINUITY_ERROR_WARN_PCT}%): la "
                "referencia de validación es poco confiable para este escenario.",
                stacklevel=2,
            )

        # Step e: ML prediction (models/static features already loaded)
        pred_full_df, pred_timings = predictor.predict_timed(scenario)
        pred_df = pred_full_df[["node_id", "inunda_pred", "vol_pred_m3"]]

        # Step f: comparison DataFrame
        comp_df = build_comparison_df(swmm_df, pred_df, scenario.scenario_id)
        all_comp_dfs.append(comp_df)

        # Step g: plots — each scenario gets its own subdirectory
        scenario_dir = out_dir / scenario.scenario_id
        flood_maps_dir = scenario_dir / "flood_maps"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        flood_maps_dir.mkdir(parents=True, exist_ok=True)

        plot_parity_nodes(comp_df, scenario_dir, scenario.scenario_id)
        plot_parity_aggregated(comp_df, scenario_dir, scenario.scenario_id)
        plot_node_profiles(comp_df, scenario_dir, scenario.scenario_id)
        plot_scenario_flood_maps(
            comp_df,
            scenario_inp_path,
            flood_maps_dir,
            scenario.scenario_id,
            t_swmm_s=t_swmm,
            t_ml_s=pred_timings["t_features_s"] + pred_timings["t_inference_s"],
        )
        plot_scenario_hydrograph(
            scenario,
            scenario_dir / f"hydrograph_{scenario.scenario_id}.png",
        )

        # Step h: per-scenario metrics, totals, timings, pooled PR-AUC inputs
        merged_prob = swmm_df.merge(
            pred_full_df[["node_id", "prob_inunda"]], on="node_id", how="inner"
        )
        pooled_truth.extend(merged_prob["inunda_swmm"].astype(int).tolist())
        pooled_probs.extend(merged_prob["prob_inunda"].astype(float).tolist())

        scen_class = compute_classification_metrics(
            comp_df["inunda_swmm"], comp_df["inunda_pred"]
        )
        scen_cond = compute_conditional_volume_metrics(
            comp_df["vol_swmm_m3"], comp_df["vol_pred_m3"], comp_df["inunda_swmm"]
        )
        vol_total_swmm = float(comp_df["vol_swmm_m3"].sum())
        vol_total_pred = float(comp_df["vol_pred_m3"].sum())
        n_extrapolated = int(pred_full_df["extrapolated"].sum())
        per_scenario.append(
            {
                "scenario_id": scenario.scenario_id,
                **scen_class,
                **scen_cond,
                "vol_total_swmm_m3": vol_total_swmm,
                "vol_total_pred_m3": vol_total_pred,
                "error_m3": vol_total_pred - vol_total_swmm,
                "error_pct": (
                    (vol_total_pred - vol_total_swmm) / vol_total_swmm * 100.0
                    if vol_total_swmm > 0
                    else None
                ),
                "n_extrapolated": n_extrapolated,
                "continuity_error_pct": continuity_pct,
            }
        )

        t_ml = pred_timings["t_features_s"] + pred_timings["t_inference_s"]
        timing_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "t_write_inp_s": t_write_inp,
                "t_swmm_s": t_swmm,
                "t_parse_rpt_s": t_parse_rpt,
                "t_features_s": pred_timings["t_features_s"],
                "t_inference_s": pred_timings["t_inference_s"],
                "speedup": (t_swmm / t_ml) if t_ml > 0 else None,
                "t_model_load_s": predictor.model_load_s,
                "t_static_features_s": predictor.static_features_s,
                "device": "cpu",
            }
        )

        # Collect records for per-node R²
        for _, row in comp_df.iterrows():
            all_records.append(
                {
                    "node_id": str(row["node_id"]),
                    "vol_swmm_m3": float(row["vol_swmm_m3"]),
                    "vol_pred_m3": float(row["vol_pred_m3"]),
                }
            )

    n_scenarios = len(csv_files)

    # Aggregate and compute metrics
    if all_comp_dfs:
        full_df = pd.concat(all_comp_dfs, ignore_index=True)
        classification_metrics = compute_classification_metrics(
            full_df["inunda_swmm"], full_df["inunda_pred"]
        )
        volume_metrics = compute_volume_metrics(
            full_df["vol_swmm_m3"], full_df["vol_pred_m3"]
        )
        volume_flooded_only = compute_conditional_volume_metrics(
            full_df["vol_swmm_m3"], full_df["vol_pred_m3"], full_df["inunda_swmm"]
        )
        per_node_r2 = compute_per_node_r2(all_records)
        pr_auc = compute_pr_auc(pooled_truth, pooled_probs)
    else:
        # No scenarios: return empty/zero metrics
        full_df = pd.DataFrame(
            columns=[
                "scenario_id", "node_id",
                "inunda_swmm", "inunda_pred", "clasificacion_correcta",
                "vol_swmm_m3", "vol_pred_m3", "error_m3", "abs_error_m3",
            ]
        )
        classification_metrics = compute_classification_metrics([], [])
        # compute_volume_metrics requires at least one sample; return zeros
        volume_metrics = {
            "mae_m3": 0.0,
            "rmse_m3": 0.0,
            "vol_total_swmm_m3": 0.0,
            "vol_total_pred_m3": 0.0,
            "error_abs_total_m3": 0.0,
            "error_pct_total": None,
        }
        volume_flooded_only = {
            "mae_flooded_m3": None,
            "rmse_flooded_m3": None,
            "n_flooded": 0,
        }
        per_node_r2 = {}
        pr_auc = None

    summary_csv_path = out_dir / "comparison_summary.csv"
    full_df.to_csv(summary_csv_path, index=False)

    totals_cols = [
        "scenario_id", "vol_total_swmm_m3", "vol_total_pred_m3",
        "error_m3", "error_pct", "n_extrapolated",
    ]
    totals_df = (
        pd.DataFrame(per_scenario)[totals_cols]
        if per_scenario
        else pd.DataFrame(columns=totals_cols)
    )
    scenario_totals_csv_path = out_dir / "scenario_totals.csv"
    totals_df.to_csv(scenario_totals_csv_path, index=False)
    plot_totals_comparison(totals_df, out_dir / "totals_comparison.png")

    timings_df = pd.DataFrame(timing_rows)
    timings_csv_path = out_dir / "timings.csv"
    timings_df.to_csv(timings_csv_path, index=False)

    metrics_per_scenario_csv_path = out_dir / "metrics_per_scenario.csv"
    pd.DataFrame(per_scenario).to_csv(metrics_per_scenario_csv_path, index=False)

    return {
        "n_scenarios": n_scenarios,
        "classification": classification_metrics,
        "volume": volume_metrics,
        "volume_flooded_only": volume_flooded_only,
        "pr_auc": pr_auc,
        "per_node_r2": per_node_r2,
        "per_scenario": per_scenario,
        "timings": timing_rows,
        "summary_csv_path": summary_csv_path,
        "scenario_totals_csv_path": scenario_totals_csv_path,
        "timings_csv_path": timings_csv_path,
        "metrics_per_scenario_csv_path": metrics_per_scenario_csv_path,
    }
