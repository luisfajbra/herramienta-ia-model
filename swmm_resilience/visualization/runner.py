"""
Plot runner — generates all network and flood-map figures.

Usage
-----
    python -m swmm_resilience.visualization.runner [--source swmm|ml|both]
                                                   [--network-dir PATH]
                                                   [--skip-existing]

For each network directory it finds a .inp file, generates:
  - network_map.png (once per network, topology only)
  - flood_map_qx{mult:.2f}_swmm.png  (one per completed SWMM run)
  - flood_map_qx{mult:.2f}_ml.png    (one per unique inflow_multiplier, via ML)

Output structure
----------------
  <network_dir>/results/plots/network_map.png
  <network_dir>/results/plots/flood_map_qx1.00_swmm.png
  <network_dir>/ml/results/flood_map_qx1.00_ml.png
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from ..config import DEFAULT_DB_FILE, DEFAULT_MODEL_ARTIFACTS_DIR, NETWORKS_DIR
from .flood_map import plot_flood_map
from .loaders import load_all_ml, load_all_swmm_runs
from .network_map import plot_network


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_inp(network_dir: Path) -> Path | None:
    """Return the first .inp file found directly inside network_dir, or None."""
    candidates = sorted(network_dir.glob("*.inp"))
    return candidates[0] if candidates else None


def _swmm_plots_dir(network_dir: Path) -> Path:
    return network_dir / "results" / "plots"


def _ml_plots_dir(network_dir: Path) -> Path:
    return network_dir / "ml" / "results"


def _global_vmax(db_path: Path) -> float:
    """Return the maximum flooding_volume_m3 across ALL completed runs."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT MAX(flooding_volume_m3) FROM node_results"
        )
        result = cur.fetchone()[0]
    finally:
        conn.close()
    return float(result) if result is not None else 1.0


# ── main routine ──────────────────────────────────────────────────────────────

def run(
    network_dir: Path,
    db_path: Path = DEFAULT_DB_FILE,
    artifacts_dir: Path = DEFAULT_MODEL_ARTIFACTS_DIR,
    source: str = "both",
    skip_existing: bool = True,
) -> None:
    """
    Generate all plots for a given network directory.

    Parameters
    ----------
    network_dir   : Folder containing the .inp file and results/.
    db_path       : SQLite database with SWMM run results.
    artifacts_dir : Trained ML model artifacts directory.
    source        : 'swmm', 'ml', or 'both'.
    skip_existing : If True, skip PNG files that already exist.
    """
    inp = _find_inp(network_dir)
    if inp is None:
        print(f"[SKIP] No .inp found in {network_dir}", file=sys.stderr)
        return

    network_name = network_dir.name
    vmax = _global_vmax(db_path)
    generated = 0

    # ── Figure 1: topology (once per network) ─────────────────────────────────
    net_out = _swmm_plots_dir(network_dir) / "network_map.png"
    if skip_existing and net_out.exists():
        print(f"[skip] {net_out.name}")
    else:
        plot_network(
            inp_path=inp,
            output_path=net_out,
            title=f"Topología de red — {network_name}",
        )
        print(f"[OK]   {net_out}")
        generated += 1

    # ── Figure 2a: SWMM flood maps ─────────────────────────────────────────────
    if source in ("swmm", "both"):
        swmm_runs = load_all_swmm_runs(db_path)
        for run_id, mult, node_data in swmm_runs:
            out = _swmm_plots_dir(network_dir) / f"flood_map_qx{mult:.2f}_swmm.png"
            if skip_existing and out.exists():
                print(f"[skip] {out.name}")
                continue
            title = f"Mapa de inundación — {network_name}\nFactor de caudal: Qx = {mult:.2f} (SWMM)"
            plot_flood_map(
                node_data=node_data,
                inp_path=inp,
                output_path=out,
                title=title,
                vmax_global=vmax,
            )
            print(f"[OK]   {out.name}")
            generated += 1

    # ── Figure 2b: ML flood maps ───────────────────────────────────────────────
    if source in ("ml", "both"):
        # Use the same multipliers as the SWMM runs so figures are comparable
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT DISTINCT inflow_multiplier FROM runs WHERE status='completed' ORDER BY inflow_multiplier"
            ).fetchall()
        finally:
            conn.close()
        multipliers = [float(r[0]) for r in rows]

        if multipliers:
            ml_runs = load_all_ml(inp, multipliers, artifacts_dir)
            for mult, node_data in ml_runs:
                out = _ml_plots_dir(network_dir) / f"flood_map_qx{mult:.2f}_ml.png"
                if skip_existing and out.exists():
                    print(f"[skip] {out.name}")
                    continue
                title = f"Mapa de inundación — {network_name}\nFactor de caudal: Qx = {mult:.2f} (ML)"
                plot_flood_map(
                    node_data=node_data,
                    inp_path=inp,
                    output_path=out,
                    title=title,
                    vmax_global=vmax,
                )
                print(f"[OK]   {out.name}")
                generated += 1

    print(f"\n{generated} figura(s) generada(s) para {network_name}.")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate SWMM network and flood-map figures.")
    parser.add_argument(
        "--network-dir",
        type=Path,
        default=None,
        help="Path to the network folder (default: all folders under data/networks/).",
    )
    parser.add_argument(
        "--source",
        choices=["swmm", "ml", "both"],
        default="both",
        help="Which flood maps to generate (default: both).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_FILE,
        help="Path to the SQLite database.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_MODEL_ARTIFACTS_DIR,
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Regenerate figures even if they already exist.",
    )
    args = parser.parse_args()

    if args.network_dir:
        network_dirs = [args.network_dir]
    else:
        network_dirs = [p for p in NETWORKS_DIR.iterdir() if p.is_dir()]

    for nd in sorted(network_dirs):
        print(f"\n=== {nd.name} ===")
        run(
            network_dir=nd,
            db_path=args.db,
            artifacts_dir=args.artifacts_dir,
            source=args.source,
            skip_existing=not args.no_skip,
        )


if __name__ == "__main__":
    _cli()
