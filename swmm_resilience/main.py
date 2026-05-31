"""
Main orchestration for the SWMM resilience pipeline.
"""

import os
from pathlib import Path

from .analysis.dataset import export_ml_dataset
from .analysis.eda import run_dataset_review
from .config import (
    DEFAULT_DATASET_REVIEW_DIR,
    DEFAULT_DB_FILE,
    DEFAULT_INFLOW_MULTIPLIERS,
    DEFAULT_INP_FILE,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_SCENARIO_TYPE,
    DEFAULT_SPATIAL_PATTERN,
    DEFAULT_TARGET_NODES,
    LEGACY_INP_FILE,
    NETWORKS_DIR,
    SCENARIO_MODE_STEADY,
    SCENARIO_MODE_TIMESERIES,
    SCENARIO_MODE_TO_TYPE,
    SUPPORTED_SCENARIO_MODES,
    TRAINING_DIR,
)
from .database.repository import (
    connect_db,
    export_run_summary,
    save_results,
    save_static_topology,
    update_run_status,
    verify_run_saved,
)
from .database.queries import register_temporal_artifact
from .ml.temporal.dataset import save_node_timeseries_parquet
from .simulation.runner import extract_static_topology, run_simulation
from .utils import file_hash, new_id, normalize_inflow_multipliers


def _load_pyswmm():
    """Import PySWMM lazily to avoid OpenMP conflicts during app startup."""
    from pyswmm import Links, Nodes, Simulation

    return Nodes, Links, Simulation


def resolve_inp_file(inp_file=None) -> Path:
    """Resolve the network path, keeping backward compatibility with the legacy location."""
    if inp_file is not None:
        return Path(inp_file)
    if DEFAULT_INP_FILE.exists():
        return DEFAULT_INP_FILE
    return LEGACY_INP_FILE


def network_results_dir(inp_path: Path) -> Path:
    """Return the results folder associated with the selected network file."""
    return inp_path.parent / "results"


def network_node_timeseries_dir(inp_path: Path) -> Path:
    """Return the folder for per-run node time series artifacts."""
    return network_results_dir(inp_path) / "temporal" / "node_timeseries"


def network_display_name(inp_path: Path) -> str:
    """Return the display/storage name for one network file."""
    return inp_path.name


def infer_scenario_mode(inp_path: Path) -> str:
    """Best-effort scenario-mode guess from the selected file name."""
    file_name = inp_path.name.lower()
    if "steady" in file_name:
        return SCENARIO_MODE_STEADY
    return SCENARIO_MODE_TIMESERIES


def ensure_directories(inp_path: Path):
    """Create expected data directories."""
    NETWORKS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    inp_path.parent.mkdir(parents=True, exist_ok=True)
    network_results_dir(inp_path).mkdir(parents=True, exist_ok=True)
    network_node_timeseries_dir(inp_path).mkdir(parents=True, exist_ok=True)


def _normalize_target_nodes(target_nodes):
    if target_nodes is None:
        return None
    if isinstance(target_nodes, str):
        return [target_nodes]
    return [str(node_id) for node_id in target_nodes]


def _normalize_inflow_multipliers(inflow_multipliers):
    if inflow_multipliers is None:
        return list(DEFAULT_INFLOW_MULTIPLIERS)
    return normalize_inflow_multipliers(
        inflow_multipliers,
        minimum=1.0,
        label="Los factores multiplicadores de caudal",
    )


def normalize_scenario_mode(scenario_mode, inp_path: Path) -> str:
    """Normalize user-facing scenario labels into one supported scaling mode."""
    if scenario_mode is None:
        return infer_scenario_mode(inp_path)

    normalized = str(scenario_mode).strip().lower()
    aliases = {
        "timeseries": SCENARIO_MODE_TIMESERIES,
        "time_series": SCENARIO_MODE_TIMESERIES,
        "time-series": SCENARIO_MODE_TIMESERIES,
        "hidrograma": SCENARIO_MODE_TIMESERIES,
        "hidrograma_interno": SCENARIO_MODE_TIMESERIES,
        "hydrograph": SCENARIO_MODE_TIMESERIES,
        "steady": SCENARIO_MODE_STEADY,
        "steady_flow": SCENARIO_MODE_STEADY,
        "steady-flow": SCENARIO_MODE_STEADY,
        "baseline": SCENARIO_MODE_STEADY,
        "baseline_inflow": SCENARIO_MODE_STEADY,
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in SUPPORTED_SCENARIO_MODES:
        supported = ", ".join(SUPPORTED_SCENARIO_MODES)
        raise ValueError(
            f"Modo de escenario invalido: '{scenario_mode}'. Usa uno de: {supported}."
        )
    return resolved


def resolve_scenario_type(scenario_mode: str, scenario_type: str | None) -> str:
    """Resolve the stored scenario label for the run metadata."""
    if scenario_type is not None:
        return str(scenario_type)
    return SCENARIO_MODE_TO_TYPE.get(scenario_mode, DEFAULT_SCENARIO_TYPE)


def run_experiment(
    inp_file=None,
    db_file=None,
    output_csv=None,
    inflow_multipliers=None,
    delta_inflows_lps=None,
    target_nodes=DEFAULT_TARGET_NODES,
    scenario_mode: str | None = None,
    scenario_type: str | None = None,
    spatial_pattern: str = DEFAULT_SPATIAL_PATTERN,
    reset_db: bool = False,
):
    """Run the full simulation-to-database-to-dataset pipeline."""
    if delta_inflows_lps is not None:
        raise ValueError(
            "delta_inflows_lps esta deshabilitado temporalmente porque antes se "
            "interpretaba como multiplicadores de caudal. Usa inflow_multipliers=[...] "
            "para escenarios Qx."
        )

    inp_path = resolve_inp_file(inp_file)
    ensure_directories(inp_path)

    db_path = Path(db_file) if db_file is not None else DEFAULT_DB_FILE
    csv_path = (
        Path(output_csv)
        if output_csv is not None
        else network_results_dir(inp_path) / DEFAULT_OUTPUT_CSV.name
    )
    scenario_mode = normalize_scenario_mode(scenario_mode, inp_path)
    scenario_type = resolve_scenario_type(scenario_mode, scenario_type)
    target_nodes = _normalize_target_nodes(target_nodes)
    multipliers = _normalize_inflow_multipliers(inflow_multipliers)
    run_plan = multipliers

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
    print(f"  Red        : {network_display_name(inp_path)}")
    print(f"  Hash       : {network_hash[:12]}...")
    print(f"  Escenario  : {scenario_mode} ({scenario_type})")
    print(f"  Modo BD    : {'reiniciar y reemplazar' if reset_db else 'agregar a existente (append)'}")
    print(f"  Corridas   : {len(run_plan)}")
    print(f"  Factores   : {', '.join(f'{multiplier:g}' for multiplier in run_plan)}")
    print(f"  Nodos      : {'todos' if target_nodes is None else ', '.join(target_nodes)}")
    print(f"  DB Salida  : {db_path}")
    print(f"{'=' * 70}\n")

    Nodes, Links, Simulation = _load_pyswmm()

    print("  [1/2] Extrayendo topologia estatica...")
    topology = extract_static_topology(str(inp_path), network_hash, Nodes, Links, Simulation)
    save_static_topology(conn, topology)
    link_static = topology["link_static"]
    node_inflow_profiles = topology.get("node_inflow_profiles", {})

    print(f"\n  [2/2] Corriendo {len(run_plan)} simulaciones...\n")
    for index, scenario_multiplier in enumerate(run_plan, start=1):
        run_id = new_id()
        inflow_multiplier = scenario_multiplier
        print(
            f"[{index:>2}/{len(run_plan)}] run={run_id[:8]}  factor={inflow_multiplier:.4f}x"
        )

        conn.execute(
            """
            INSERT INTO runs
              (run_id, network_file, network_hash, scenario_type,
               spatial_pattern, delta_inflow_lps, inflow_multiplier,
               input_source, executed_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'running')
            """,
            (
                run_id,
                network_display_name(inp_path),
                network_hash,
                scenario_type,
                spatial_pattern,
                inflow_multiplier,
                inflow_multiplier,
                "hydrograph" if scenario_mode == SCENARIO_MODE_TIMESERIES else "steady",
            ),
        )
        conn.commit()

        try:
            results = run_simulation(
                str(inp_path),
                inflow_multiplier,
                link_static,
                Nodes,
                Links,
                Simulation,
                target_nodes=target_nodes,
                node_inflow_profiles=node_inflow_profiles,
                scenario_mode=scenario_mode,
                run_id=run_id,
                network_hash=network_hash,
            )
            save_results(conn, run_id, results)
            node_timeseries_path = (
                network_node_timeseries_dir(inp_path) / f"run_{run_id}.parquet"
            )
            ts_records = results["node_timeseries_records"]
            save_node_timeseries_parquet(ts_records, node_timeseries_path)

            if ts_records:
                _node_ids = {r["node_id"] for r in ts_records}
                _node_count = len(_node_ids)
                _step_count = len(ts_records) // _node_count if _node_count else 0
            else:
                _node_count = 0
                _step_count = 0

            register_temporal_artifact(
                conn,
                run_id=run_id,
                network_hash=network_hash,
                parquet_path=node_timeseries_path,
                node_count=_node_count,
                step_count=_step_count,
            )
            update_run_status(conn, run_id, "completed")
            verify_run_saved(conn, run_id)

            summary = results["summary"]
            print(
                f"    ok Fallaron  : {summary['failed_nodes_count']}/{summary['total_nodes']} "
                f"({summary['pct_flooded_nodes']:.1f}%)"
            )
            print(f"    ok Pico flood: {summary['total_peak_flooding_lps']:.2f} lps (suma nodos)")
            print(f"    ok Vol flood : {summary['total_flood_volume_m3']:.3f} m3 (suma nodos)")
            print(f"    ok Resiliencia: {summary['resilience_index']:.3f}")
            print(f"    ok Temporal  : {node_timeseries_path}")
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
    run_dataset_review(str(csv_path), output_dir=DEFAULT_DATASET_REVIEW_DIR)
    print(
        "  [info] Entrenamiento ML: si elegiste append, se entrenara con todas las corridas "
        "acumuladas en esta base. Si elegiste reiniciar, se entrenara solo con las corridas nuevas."
    )

    return {"db_file": db_path, "csv_file": csv_path}


if __name__ == "__main__":
    run_experiment()
