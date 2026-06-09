"""Batch coordinator for hydrograph CSV validation against SWMM simulations.

Responsibilities:
1. Glob a directory for *.csv hydrograph files.
2. Load each as a HydrographScenario.
3. Write a scenario-specific SWMM .inp file.
4. Run SWMM simulation and parse flooding results from the .rpt.
5. Run ML prediction for each scenario.
6. Build a per-scenario comparison DataFrame.
7. Generate plots (parity nodes, parity aggregated, node profiles).
8. Aggregate metrics across all scenarios.
9. Save a summary CSV and return a summary dict.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from ..analysis.model_comparison import (
    build_comparison_df,
    compute_classification_metrics,
    compute_per_node_r2,
    compute_volume_metrics,
)
from ..ml.scenario_predict import predict_scenario
from ..simulation.timeseries_scenario import write_scenario_inp
from ..validation.hydrograph_csv import HydrographScenario, load_scenario
from ..visualization.model_comparison import (
    plot_node_profiles,
    plot_parity_aggregated,
    plot_parity_nodes,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


def _parse_node_flooding(rpt_path: Path, flood_threshold_m3: float) -> pd.DataFrame:
    """Parse the 'Node Flooding Summary' section of a SWMM .rpt file.

    Returns a DataFrame with columns:
        node_id (str), vol_swmm_m3 (float), inunda_swmm (int 0/1)

    Volume in the .rpt is reported as 10^6 litres (Megalitres).
    Conversion: 1 Megaliter = 1 000 m³.

    Attempts to use ``swmm_api`` first; falls back to a pure-text parser when
    swmm_api is unavailable or returns an empty table.

    All nodes listed in the summary have vol_swmm_m3 > 0 (SWMM only lists nodes
    that actually flooded).  Nodes absent from the summary have zero flooding.
    """
    # --- Try swmm_api first -------------------------------------------------
    try:
        from ..simulation.swmm_api_io import read_node_flooding_summary

        rpt_df = read_node_flooding_summary(rpt_path)
        if rpt_df is not None and not rpt_df.empty and "flooding_volume_m3" in rpt_df.columns:
            rows = []
            for _, row in rpt_df.iterrows():
                vol = float(row["flooding_volume_m3"]) if pd.notna(row["flooding_volume_m3"]) else 0.0
                rows.append(
                    {
                        "node_id": str(row["node_id"]),
                        "vol_swmm_m3": vol,
                        "inunda_swmm": int(vol >= flood_threshold_m3),
                    }
                )
            if rows:
                return pd.DataFrame(rows)
    except Exception as exc:
        warnings.warn(
            f"swmm_api RPT read failed ({exc}); falling back to text parser.",
            stacklevel=2,
        )

    # --- Text-based fallback parser -----------------------------------------
    return _parse_node_flooding_text(rpt_path, flood_threshold_m3)


def _parse_node_flooding_text(rpt_path: Path, flood_threshold_m3: float) -> pd.DataFrame:
    """Pure-text fallback parser for the 'Node Flooding Summary' section.

    The section header line contains the words 'Node Flooding Summary'.
    Data lines start after two header rows and the dashed separator.
    Each data line has the form:
        <node_id>  <hours_flooded>  <rate_lps>  <volume_megalitres>

    Columns are whitespace-delimited; we pick the last numeric token as the
    volume (position −1) to be robust against varying column widths.
    """
    try:
        text = rpt_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return pd.DataFrame(columns=["node_id", "vol_swmm_m3", "inunda_swmm"])

    rows: list[dict] = []
    in_section = False
    past_header = False  # True once we pass the dashed separator line

    for line in text.splitlines():
        stripped = line.strip()

        # Detect section start
        if not in_section:
            if "Node Flooding Summary" in line:
                in_section = True
                past_header = False
            continue

        # Skip blank lines within the section
        if not stripped:
            continue

        # Blank line after dashed separator may signal end of section
        # We'll stop at the next all-caps section header or end of file.
        if in_section and past_header and stripped.startswith("*") :
            break
        if in_section and past_header and stripped and stripped[0].isalpha() and stripped.isupper():
            break

        # Dashed separator marks the end of header rows
        if stripped.startswith("---"):
            past_header = True
            continue

        if not past_header:
            # Still in header rows above the dashes
            continue

        # Data line: node_id followed by numeric columns
        parts = stripped.split()
        if len(parts) < 4:
            continue
        try:
            vol_megalitres = float(parts[-1])
        except ValueError:
            continue

        node_id = parts[0]
        vol_m3 = vol_megalitres * 1000.0  # 1 Megaliter = 1000 m³
        rows.append(
            {
                "node_id": node_id,
                "vol_swmm_m3": vol_m3,
                "inunda_swmm": int(vol_m3 >= flood_threshold_m3),
            }
        )

    return pd.DataFrame(rows, columns=["node_id", "vol_swmm_m3", "inunda_swmm"])


def _build_swmm_df(
    scenario: HydrographScenario,
    rpt_path: Path,
    flood_threshold_m3: float,
) -> pd.DataFrame:
    """Build the SWMM result DataFrame for all nodes in the scenario.

    Nodes absent from the 'Node Flooding Summary' (not flooded) are added with
    zero volume and inunda_swmm = 0.

    Returns a DataFrame with columns: node_id, inunda_swmm, vol_swmm_m3.
    """
    flooded_df = _parse_node_flooding(rpt_path, flood_threshold_m3)
    flooded_lookup: dict[str, dict] = {}
    for _, row in flooded_df.iterrows():
        flooded_lookup[str(row["node_id"])] = {
            "vol_swmm_m3": float(row["vol_swmm_m3"]),
            "inunda_swmm": int(row["inunda_swmm"]),
        }

    all_nodes = list(scenario.node_series.keys())
    records = []
    for node_id in all_nodes:
        if node_id in flooded_lookup:
            rec = flooded_lookup[node_id]
            records.append(
                {
                    "node_id": node_id,
                    "inunda_swmm": rec["inunda_swmm"],
                    "vol_swmm_m3": rec["vol_swmm_m3"],
                }
            )
        else:
            records.append(
                {
                    "node_id": node_id,
                    "inunda_swmm": 0,
                    "vol_swmm_m3": 0.0,
                }
            )

    return pd.DataFrame(records)


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
) -> dict:
    """Run batch validation of ML predictions against SWMM simulations.

    Parameters
    ----------
    csv_dir:
        Directory containing ``*.csv`` hydrograph scenario files.
    base_inp_path:
        Path to the base (unmodified) SWMM ``.inp`` network file.
    clf_path:
        Path to the joblib-serialised classifier.
    reg_path:
        Path to the joblib-serialised regressor.
    flood_threshold_m3:
        Minimum flood volume (m³) to consider a node flooded.
    out_dir:
        Root output directory. Sub-directories ``inp/`` and ``plots/`` are
        created automatically. ``comparison_summary.csv`` is written here.
    expected_nodes:
        Optional set of node IDs that every CSV must contain.

    Returns
    -------
    dict with keys:
        - ``n_scenarios`` (int)
        - ``classification`` (dict from ``compute_classification_metrics``)
        - ``volume`` (dict from ``compute_volume_metrics``)
        - ``per_node_r2`` (dict[str, float | None])
        - ``summary_csv_path`` (Path)
    """
    csv_dir = Path(csv_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp_dir = out_dir / "inp"
    plots_dir = out_dir / "plots"
    inp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))

    all_comp_dfs: list[pd.DataFrame] = []
    all_records: list[dict] = []

    for csv_path in csv_files:
        # Step a: load scenario
        scenario = load_scenario(csv_path, expected_nodes)

        # Step b: write scenario-specific .inp
        scenario_inp_path = write_scenario_inp(base_inp_path, scenario, inp_dir)

        # Step c: run SWMM
        rpt_path = _run_swmm(scenario_inp_path)

        # Step d: build SWMM result DataFrame (with zero-fill for non-flooded)
        swmm_df = _build_swmm_df(scenario, rpt_path, flood_threshold_m3)

        # Step e: ML prediction
        pred_df = predict_scenario(
            scenario=scenario,
            clf_path=clf_path,
            reg_path=reg_path,
            flood_threshold_m3=flood_threshold_m3,
            inp_path=scenario_inp_path,
        )

        # Step f: comparison DataFrame
        comp_df = build_comparison_df(swmm_df, pred_df, scenario.scenario_id)
        all_comp_dfs.append(comp_df)

        # Step g: plots
        plot_parity_nodes(comp_df, plots_dir, scenario.scenario_id)
        plot_parity_aggregated(comp_df, plots_dir, scenario.scenario_id)
        plot_node_profiles(comp_df, plots_dir, scenario.scenario_id)

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
        per_node_r2 = compute_per_node_r2(all_records)
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
        per_node_r2 = {}

    summary_csv_path = out_dir / "comparison_summary.csv"
    full_df.to_csv(summary_csv_path, index=False)

    return {
        "n_scenarios": n_scenarios,
        "classification": classification_metrics,
        "volume": volume_metrics,
        "per_node_r2": per_node_r2,
        "summary_csv_path": summary_csv_path,
    }
