import warnings
from pathlib import Path

from .runner import run_simulation_simple as run_simulation
from ..config import Config
from ..validation.drainage_state import (
    DRAINAGE_STATE_POLICY_VERSION, classify_drainage_state, read_tail_window_observables,
    save_drainage_state_manifest,
)


def run_batch(config: Config, run_dir: Path) -> list:
    """Run SWMM for every factor in config. Returns list of (factor, rpt_path)."""
    factors = config.factors()
    results = []
    for i, factor in enumerate(factors, 1):
        print(f"  [{i:>2}/{len(factors)}] factor={factor:.2f}", end=" ... ", flush=True)
        rpt_path = run_simulation(config.network.inp_path, factor, run_dir)
        print("OK")
        results.append((factor, rpt_path))
    return results


def _prepare_closed_scenario_inp(
    config,
    shape_id: str,
    factor: float,
    shape: "list[tuple[float, float]]",
    time_grid: "list[float]",
    base_inflows: "dict[str, float]",
    run_dir: Path,
    drain_down_hours_override: float | None = None,
) -> "tuple[Path, Path]":
    """Construye el escenario de (shape_id, factor), lo cierra, guarda el
    manifiesto y escribe el .inp. Devuelve (scenario_inp_path, rpt_path).

    Se extrajo de run_batch_shapes para que el cierre del aporte sea
    comprobable sin necesidad de simular con pyswmm de verdad.

    drain_down_hours_override, si se da, reemplaza
    config.validation.drain_down_hours -- lo usa el bucle de reintento por
    extension de run_batch_shapes para alargar la simulacion sin tocar la
    config global.
    """
    from .hydrograph_shapes import apply_shape
    from .inflow_closure import apply_closure_policy, save_closure_manifest
    from .timeseries_scenario import write_scenario_inp
    from ..validation.hydrograph_csv import HydrographScenario

    drain_down_hours = (
        drain_down_hours_override
        if drain_down_hours_override is not None
        else config.validation.drain_down_hours
    )

    node_series = apply_shape(shape, base_inflows, factor)
    scenario = HydrographScenario(
        scenario_id=f"{shape_id}_f{factor:.3f}",
        node_series=node_series,
        time_grid_hours=time_grid,
        last_time_hours=shape[-1][0],
    )
    scenario, closure_manifest = apply_closure_policy(
        scenario, config.validation.inflow_closure_tolerance_lps
    )
    closure_manifest["mode"] = "close"
    if closure_manifest["changed_nodes"]:
        warnings.warn(
            f"[{scenario.scenario_id}] Aporte cerrado a cero en los nodos: "
            f"{closure_manifest['changed_nodes']}.",
            stacklevel=2,
        )
    shape_run_dir = run_dir / shape_id
    save_closure_manifest(closure_manifest, shape_run_dir, scenario.scenario_id)
    scenario_inp = write_scenario_inp(
        config.network.inp_path, scenario, shape_run_dir,
        drain_down_hours=drain_down_hours,
    )
    return scenario_inp, scenario_inp.with_suffix(".rpt")


def run_batch_shapes(
    config: Config,
    shapes: "dict[str, list[tuple[float, float]]]",
    base_inflows: "dict[str, float]",
    run_dir: Path,
) -> "list[tuple[str, float, Path]]":
    """Run SWMM for every (shape, factor) combination.

    shapes: {shape_id: [(time_h, q_norm), ...]}
    base_inflows: {node_id: base_inflow_lps}
    Returns list of (shape_id, factor, rpt_path).
    """
    from pyswmm import Simulation as _Simulation

    factors = config.factors()
    total = len(shapes) * len(factors)
    results = []
    i = 0

    for shape_id, shape in shapes.items():
        time_grid = [t for t, _ in shape]

        for factor in factors:
            i += 1
            print(
                f"  [{i:>3}/{total}] shape={shape_id:<20} factor={factor:.2f}",
                end=" ... ", flush=True,
            )

            max_retries = config.validation.drainage_extension_max_retries
            increment = config.validation.drainage_extension_increment_hours
            extension_log = []
            attempt = 0
            state = "unverified"
            drainage_manifest = {}
            scenario_inp = None
            rpt_path = None

            while True:
                current_drain_down_hours = (
                    config.validation.drain_down_hours
                    if attempt == 0
                    else config.validation.drain_down_hours + attempt * increment
                )
                scenario_inp, rpt_path = _prepare_closed_scenario_inp(
                    config, shape_id, factor, shape, time_grid, base_inflows,
                    run_dir,
                    drain_down_hours_override=(
                        None if attempt == 0 else current_drain_down_hours
                    ),
                )
                with _Simulation(str(scenario_inp)) as sim:
                    for _ in sim:
                        pass
                if not rpt_path.exists():
                    raise FileNotFoundError(f"SWMM no generó .rpt: {rpt_path}")
                out_path = scenario_inp.with_suffix(".out")
                scenario_id = scenario_inp.stem
                window_hours = min(
                    config.validation.drainage_tail_window_hours,
                    current_drain_down_hours,
                )
                try:
                    observables = read_tail_window_observables(
                        out_path, current_drain_down_hours,
                        config.validation.drainage_tail_window_hours,
                    )
                    state, drainage_manifest = classify_drainage_state(
                        observables,
                        inflow_tolerance_lps=config.validation.drainage_inflow_tolerance_lps,
                        flooding_tolerance_lps=config.validation.drainage_flooding_tolerance_lps,
                        storage_change_tolerance_m3=config.validation.drainage_storage_change_tolerance_m3,
                        residual_flow_tolerance_lps=config.validation.drainage_inflow_tolerance_lps,
                        window_hours=window_hours,
                    )
                except Exception as exc:
                    # Un .out ilegible/truncado/faltante no debe abortar el resto
                    # del lote -- se degrada a "unverified" (spec Sec.7) y se
                    # sigue con el resto de escenarios ya simulados.
                    state = "unverified"
                    drainage_manifest = {
                        "policy_version": DRAINAGE_STATE_POLICY_VERSION,
                        "state": "unverified",
                        "failed_checks": ["out_read_error"],
                        "error": repr(exc),
                        "n_report_steps": None,
                        "has_link_data": None,
                        "window_hours": window_hours,
                        "min_report_steps": 2,
                        "max_inflow_lps": None,
                        "max_flooding_lps": None,
                        "max_storage_change_m3": None,
                        "max_residual_flow_lps": None,
                        "tolerances": {
                            "inflow_tolerance_lps": config.validation.drainage_inflow_tolerance_lps,
                            "flooding_tolerance_lps": config.validation.drainage_flooding_tolerance_lps,
                            "storage_change_tolerance_m3": (
                                config.validation.drainage_storage_change_tolerance_m3
                            ),
                            "residual_flow_tolerance_lps": config.validation.drainage_inflow_tolerance_lps,
                        },
                    }
                    warnings.warn(
                        f"[{scenario_id}] No se pudo leer/clasificar el .out: {exc!r} — "
                        f"estado marcado unverified.",
                        stacklevel=2,
                    )

                extension_log.append({
                    "attempt": attempt,
                    "drain_down_hours": current_drain_down_hours,
                    "state": state,
                })

                if state == "drained" or attempt >= max_retries:
                    break
                attempt += 1

            drainage_manifest["extension_log"] = extension_log
            save_drainage_state_manifest(
                drainage_manifest, scenario_inp.parent, scenario_inp.stem,
            )
            results.append((shape_id, factor, rpt_path))
            print(f"OK [{state}]")
            if state != "drained":
                warnings.warn(
                    f"[{scenario_id}] Estado de drenaje: {state} "
                    f"({drainage_manifest.get('failed_checks')}) tras "
                    f"{len(extension_log)} intento(s).",
                    stacklevel=2,
                )

    return results
