"""
Main orchestration for the SWMM resilience pipeline.
"""

import os
from pathlib import Path

from pyswmm import Links, Nodes, Simulation

from .analysis.dataset import export_ml_dataset
from .config import (
    DEFAULT_DB_FILE,
    DEFAULT_DELTA_INFLOWS_M3PS,
    DEFAULT_INP_FILE,
    DEFAULT_NETWORK_DIR,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_SCENARIO_TYPE,
    DEFAULT_SPATIAL_PATTERN,
    LEGACY_INP_FILE,
    NETWORKS_DIR,
)
from .database.repository import (
    connect_db,
    export_run_summary,
    save_results,
    save_static_topology,
    update_run_status,
    verify_run_saved,
)
from .simulation.runner import extract_static_topology, run_simulation
from .utils import file_hash, new_id


def resolve_inp_file(inp_file=None) -> Path:
    """Resolve the network path, keeping backward compatibility with the legacy location."""
    if inp_file is not None:
        return Path(inp_file)
    if DEFAULT_INP_FILE.exists():
        return DEFAULT_INP_FILE
    return LEGACY_INP_FILE


def ensure_directories():
    """Create expected data directories."""
    NETWORKS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_NETWORK_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment(
    inp_file=None,
    db_file=None,
    output_csv=None,
    delta_inflows_m3ps=None,
    scenario_type: str = DEFAULT_SCENARIO_TYPE,
    spatial_pattern: str = DEFAULT_SPATIAL_PATTERN,
    reset_db: bool = True,
):
    """Run the full simulation-to-database-to-dataset pipeline."""
    ensure_directories()

    inp_path = resolve_inp_file(inp_file)
    db_path = Path(db_file) if db_file is not None else DEFAULT_DB_FILE
    csv_path = Path(output_csv) if output_csv is not None else DEFAULT_OUTPUT_CSV
    deltas = delta_inflows_m3ps if delta_inflows_m3ps is not None else DEFAULT_DELTA_INFLOWS_M3PS

    if not inp_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo .inp: {inp_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if reset_db and db_path.exists():
        os.remove(db_path)
        print(f"  [info] BD anterior eliminada: {db_path}")

    conn = connect_db(str(db_path))
    network_hash = file_hash(str(inp_path))

    print(f"\n{'=' * 70}")
    print("  SWMM Resilience ML - Generador de Base de Datos")
    print(f"  Red        : {inp_path}")
    print(f"  Hash       : {network_hash[:12]}...")
    print(f"  Corridas   : {len(deltas)}")
    print(f"  DB Salida  : {db_path}")
    print(f"{'=' * 70}\n")

    print("  [1/2] Extrayendo topologia estatica...")
    topology = extract_static_topology(str(inp_path), network_hash, Nodes, Links, Simulation)
    save_static_topology(conn, topology)
    link_static = topology["link_static"]

    print(f"\n  [2/2] Corriendo {len(deltas)} simulaciones...\n")
    for index, delta in enumerate(deltas, start=1):
        run_id = new_id()
        print(f"[{index:>2}/{len(deltas)}] run={run_id[:8]}  Dq={delta:.4f} m3/s/nodo")

        conn.execute(
            """
            INSERT INTO runs
              (run_id, network_file, network_hash, scenario_type,
               spatial_pattern, delta_inflow_m3ps, executed_at, status)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'), 'running')
            """,
            (run_id, str(inp_path), network_hash, scenario_type, spatial_pattern, delta),
        )
        conn.commit()

        try:
            results = run_simulation(str(inp_path), delta, link_static, Nodes, Links, Simulation)
            save_results(conn, run_id, results)
            update_run_status(conn, run_id, "completed")
            verify_run_saved(conn, run_id)

            summary = results["summary"]
            print(
                f"    ok Inundados : {summary['total_flooded_nodes']}/{summary['total_nodes']} "
                f"({summary['pct_flooded_nodes']:.1f}%)"
            )
            print(f"    ok Vol total : {summary['total_flooding_volume_m3']:.2f} m3")
            print(f"    ok Resiliencia: {summary['resilience_index']:.3f}")
            if summary["time_to_first_flood_min"]:
                print(f"    ok 1er flood : {summary['time_to_first_flood_min']:.1f} min")
            print()
        except Exception:
            update_run_status(conn, run_id, "failed")
            raise

    conn.close()
    print(f"{'=' * 70}")
    print(f"  Listo. Base de datos generada: {db_path}")
    print(f"{'=' * 70}\n")

    export_run_summary(str(db_path))
    export_ml_dataset(str(db_path), str(csv_path))

    return {"db_file": db_path, "csv_file": csv_path}


if __name__ == "__main__":
    run_experiment()
