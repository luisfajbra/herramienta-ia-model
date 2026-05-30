"""
Reset utility — clears simulation results, plots, datasets, and ML artifacts.

Usage
-----
    python -m swmm_resilience.reset --all
    python -m swmm_resilience.reset --db --plots
    python -m swmm_resilience.reset --db --plots --dataset --artifacts --temporal

Categories
----------
  --db         Corridas y resultados en la base de datos
  --plots      Imágenes generadas (mapas PNG de red e inundación)
  --dataset    Archivos dataset_ml.csv
  --artifacts  Artefactos ML (.joblib, manifest.json, CSVs de métricas)
  --temporal   Series temporales (archivos Parquet)
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Callable

from .config import DEFAULT_DB_FILE, NETWORKS_DIR


RESET_CATEGORIES: dict[str, str] = {
    "db":        "Corridas y resultados en la base de datos",
    "plots":     "Imágenes generadas (mapas PNG)",
    "dataset":   "Archivos dataset_ml.csv",
    "artifacts": "Artefactos ML (.joblib, manifest.json, CSVs de métricas)",
    "temporal":  "Series temporales (archivos Parquet)",
}

_DB_TABLES_IN_ORDER = [
    "temporal_artifacts",
    "node_results",
    "link_results",
    "run_inputs",
    "run_summary",
    "network_nodes",
    "network_links",
    "runs",
]


def _log(msg: str, callback: Callable[[str], None] | None) -> None:
    print(msg)
    if callback:
        callback(msg + "\n")


def reset_db(
    db_path: Path = DEFAULT_DB_FILE,
    callback: Callable[[str], None] | None = None,
) -> int:
    """Delete all simulation data from the database. Returns total rows deleted."""
    if not db_path.exists():
        _log(f"  BD no encontrada: {db_path}", callback)
        return 0

    total = 0
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        for table in _DB_TABLES_IN_ORDER:
            try:
                cur = conn.execute(f"DELETE FROM {table}")
                n = cur.rowcount
                total += n
                _log(f"  {table}: {n} fila(s) eliminada(s)", callback)
            except sqlite3.OperationalError:
                pass
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()
    finally:
        conn.close()
    return total


def reset_plots(
    networks_dir: Path = NETWORKS_DIR,
    callback: Callable[[str], None] | None = None,
) -> int:
    """Delete all generated PNG files. Returns count deleted."""
    count = 0
    for net_dir in sorted(networks_dir.iterdir()):
        if not net_dir.is_dir():
            continue
        search_dirs = [
            net_dir / "results" / "plots",
            net_dir / "ml" / "results",
        ]
        for folder in search_dirs:
            if not folder.exists():
                continue
            for png in sorted(folder.glob("*.png")):
                png.unlink()
                _log(f"  Eliminado: {png.relative_to(networks_dir)}", callback)
                count += 1
    return count


def reset_dataset(
    networks_dir: Path = NETWORKS_DIR,
    callback: Callable[[str], None] | None = None,
) -> int:
    """Delete all dataset_ml.csv files. Returns count deleted."""
    count = 0
    for net_dir in sorted(networks_dir.iterdir()):
        if not net_dir.is_dir():
            continue
        csv = net_dir / "results" / "dataset_ml.csv"
        if csv.exists():
            csv.unlink()
            _log(f"  Eliminado: {csv.relative_to(networks_dir)}", callback)
            count += 1
    return count


def reset_artifacts(
    networks_dir: Path = NETWORKS_DIR,
    callback: Callable[[str], None] | None = None,
) -> int:
    """Delete all ML artifact files (.joblib, manifest.json, metric CSVs/XLSX, .pt). Returns count deleted."""
    count = 0
    suffixes = {".joblib", ".json", ".csv", ".xlsx", ".pt"}
    for net_dir in sorted(networks_dir.iterdir()):
        if not net_dir.is_dir():
            continue
        artifact_dirs = [
            net_dir / "results" / "model_artifacts",
            net_dir / "results" / "temporal" / "model_artifacts",
        ]
        for artifacts_dir in artifact_dirs:
            if not artifacts_dir.exists():
                continue
            for f in sorted(artifacts_dir.iterdir()):
                if f.is_file() and f.suffix in suffixes:
                    f.unlink()
                    _log(f"  Eliminado: {f.relative_to(networks_dir)}", callback)
                    count += 1
    return count


def reset_temporal(
    networks_dir: Path = NETWORKS_DIR,
    callback: Callable[[str], None] | None = None,
) -> int:
    """Delete all Parquet temporal files. Returns count deleted."""
    count = 0
    for net_dir in sorted(networks_dir.iterdir()):
        if not net_dir.is_dir():
            continue
        temporal_dir = net_dir / "results" / "temporal" / "node_timeseries"
        if not temporal_dir.exists():
            continue
        for parquet in sorted(temporal_dir.glob("*.parquet")):
            parquet.unlink()
            _log(f"  Eliminado: {parquet.relative_to(networks_dir)}", callback)
            count += 1
    return count


def reset(
    db: bool = False,
    plots: bool = False,
    dataset: bool = False,
    artifacts: bool = False,
    temporal: bool = False,
    db_path: Path = DEFAULT_DB_FILE,
    networks_dir: Path = NETWORKS_DIR,
    callback: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Run selected reset operations. Returns a summary dict with counts per category."""
    if not any([db, plots, dataset, artifacts, temporal]):
        _log("Nada seleccionado para limpiar.", callback)
        return {}

    results: dict[str, int] = {}

    if db:
        _log("\n[BD] Limpiando corridas y resultados...", callback)
        results["db"] = reset_db(db_path, callback)

    if plots:
        _log("\n[Plots] Eliminando imágenes...", callback)
        results["plots"] = reset_plots(networks_dir, callback)

    if dataset:
        _log("\n[Dataset] Eliminando dataset_ml.csv...", callback)
        results["dataset"] = reset_dataset(networks_dir, callback)

    if artifacts:
        _log("\n[Artefactos] Eliminando modelos ML...", callback)
        results["artifacts"] = reset_artifacts(networks_dir, callback)

    if temporal:
        _log("\n[Temporal] Eliminando archivos Parquet...", callback)
        results["temporal"] = reset_temporal(networks_dir, callback)

    total = sum(results.values())
    _log(f"\nLimpieza completada — {total} elemento(s) eliminado(s).", callback)
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Limpia resultados, plots, datasets y artefactos ML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  --{k:10s}  {v}" for k, v in RESET_CATEGORIES.items()),
    )
    parser.add_argument("--all",       action="store_true", help="Limpiar todo.")
    parser.add_argument("--db",        action="store_true", help=RESET_CATEGORIES["db"])
    parser.add_argument("--plots",     action="store_true", help=RESET_CATEGORIES["plots"])
    parser.add_argument("--dataset",   action="store_true", help=RESET_CATEGORIES["dataset"])
    parser.add_argument("--artifacts", action="store_true", help=RESET_CATEGORIES["artifacts"])
    parser.add_argument("--temporal",  action="store_true", help=RESET_CATEGORIES["temporal"])
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    do_all = args.all
    print("ADVERTENCIA: Esta operación es irreversible.")
    confirm = input("¿Confirmar? (s/N): ").strip().lower()
    if confirm != "s":
        print("Cancelado.")
        sys.exit(0)

    reset(
        db=do_all or args.db,
        plots=do_all or args.plots,
        dataset=do_all or args.dataset,
        artifacts=do_all or args.artifacts,
        temporal=do_all or args.temporal,
    )


if __name__ == "__main__":
    _main()
