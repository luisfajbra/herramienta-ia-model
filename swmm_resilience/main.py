"""
Main orchestration for the SWMM resilience pipeline.
"""

import os
from pathlib import Path

from pyswmm import Links, Nodes, Simulation

from .analysis.dataset import export_ml_dataset
from .config import (
    DEFAULT_DB_FILE,
    DEFAULT_DELTA_INFLOWS_LPS,
    DEFAULT_HYDROGRAPH_FILE,
    DEFAULT_INP_FILE,
    DEFAULT_NETWORK_DIR,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_SCENARIO_TYPE,
    DEFAULT_SPATIAL_PATTERN,
    DEFAULT_TARGET_NODES,
    HYDROGRAPHS_DIR,
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
from .simulation.runner import extract_static_topology, load_hydrograph, run_simulation
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
    HYDROGRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_NETWORK_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_target_nodes(target_nodes):
    if target_nodes is None:
        return None
    if isinstance(target_nodes, str):
        return [target_nodes]
    return [str(node_id) for node_id in target_nodes]


def _normalize_hydrograph_multipliers(hydrograph_multipliers):
    if hydrograph_multipliers is None:
        return [1.0]
    multipliers = [float(value) for value in hydrograph_multipliers]
    if not multipliers:
        raise ValueError("Debes indicar al menos un multiplicador de hidrograma.")
    if any(value <= 0 for value in multipliers):
        raise ValueError("Los multiplicadores de hidrograma deben ser mayores que cero.")
    return multipliers


def run_experiment(
    inp_file=None,
    db_file=None,
    output_csv=None,
    delta_inflows_lps=None,
    hydrograph_file=DEFAULT_HYDROGRAPH_FILE,
    hydrograph_multipliers=None,
    target_nodes=DEFAULT_TARGET_NODES,
    scenario_type: str = DEFAULT_SCENARIO_TYPE,
    spatial_pattern: str = DEFAULT_SPATIAL_PATTERN,
    reset_db: bool = True,
):
    """Run the full simulation-to-database-to-dataset pipeline."""
    ensure_directories()

    inp_path = resolve_inp_file(inp_file)
    db_path = Path(db_file) if db_file is not None else DEFAULT_DB_FILE
    csv_path = Path(output_csv) if output_csv is not None else DEFAULT_OUTPUT_CSV
    target_nodes = _normalize_target_nodes(target_nodes)
    hydrograph = load_hydrograph(hydrograph_file) if hydrograph_file else None
    hydrograph_peak_lps = max((point[1] for point in hydrograph), default=None) if hydrograph else None
    if hydrograph is not None:
        multipliers = _normalize_hydrograph_multipliers(hydrograph_multipliers)
        run_plan = [(hydrograph_peak_lps * multiplier, multiplier) for multiplier in multipliers]
        deltas = [delta for delta, _multiplier in run_plan]
        scenario_type = "hydrograph_inflow"
        spatial_pattern = "all_nodes" if target_nodes is None else "selected_nodes"
    else:
        deltas = delta_inflows_lps if delta_inflows_lps is not None else DEFAULT_DELTA_INFLOWS_LPS
        run_plan = [(delta, 1.0) for delta in deltas]

    if not inp_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo .inp: {inp_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if reset_db and db_path.exists():
        try:
            os.remove(db_path)
        except PermissionError as exc:
            raise PermissionError(
                "No se pudo reiniciar la base de datos porque esta abierta en otra ventana "
                f"o proceso: {db_path}. Cierra el visor SQLite/DB Browser u otra app que la "
                "este usando, o desmarca 'Reiniciar base de datos antes de correr'."
            ) from exc
        print(f"  [info] BD anterior eliminada: {db_path}")

    conn = connect_db(str(db_path))
    network_hash = file_hash(str(inp_path))

    print(f"\n{'=' * 70}")
    print("  SWMM Resilience ML - Generador de Base de Datos")
    print(f"  Red        : {inp_path}")
    print(f"  Hash       : {network_hash[:12]}...")
    print(f"  Corridas   : {len(deltas)}")
    if hydrograph is not None:
        print(f"  Hidrograma : {Path(hydrograph_file)}")
        print(f"  Pico       : {hydrograph_peak_lps:.4f} L/s")
        print(f"  Escalas    : {', '.join(f'{multiplier:g}x' for _delta, multiplier in run_plan)}")
        print(f"  Nodos      : {'todos' if target_nodes is None else ', '.join(target_nodes)}")
    print(f"  DB Salida  : {db_path}")
    print(f"{'=' * 70}\n")

    print("  [1/2] Extrayendo topologia estatica...")
    topology = extract_static_topology(str(inp_path), network_hash, Nodes, Links, Simulation)
    save_static_topology(conn, topology)
    link_static = topology["link_static"]

    print(f"\n  [2/2] Corriendo {len(deltas)} simulaciones...\n")
    for index, (delta, hydrograph_multiplier) in enumerate(run_plan, start=1):
        run_id = new_id()
        scale_text = f"  escala={hydrograph_multiplier:g}x" if hydrograph is not None else ""
        print(f"[{index:>2}/{len(deltas)}] run={run_id[:8]}  Dq={delta:.4f} L/s/nodo{scale_text}")

        conn.execute(
            """
            INSERT INTO runs
              (run_id, network_file, network_hash, scenario_type,
               spatial_pattern, delta_inflow_lps, executed_at, status)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'), 'running')
            """,
            (run_id, str(inp_path), network_hash, scenario_type, spatial_pattern, delta),
        )
        conn.commit()

        try:
            results = run_simulation(
                str(inp_path),
                delta,
                link_static,
                Nodes,
                Links,
                Simulation,
                hydrograph=hydrograph,
                hydrograph_multiplier=hydrograph_multiplier,
                target_nodes=target_nodes,
            )
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
