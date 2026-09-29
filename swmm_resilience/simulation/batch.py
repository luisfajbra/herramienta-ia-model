from pathlib import Path

from .runner import run_simulation_simple as run_simulation
from ..config import Config


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
) -> "tuple[Path, Path]":
    """Construye el escenario de (shape_id, factor), lo cierra, guarda el
    manifiesto y escribe el .inp. Devuelve (scenario_inp_path, rpt_path).

    Se extrajo de run_batch_shapes para que el cierre del aporte sea
    comprobable sin necesidad de simular con pyswmm de verdad.
    """
    from .hydrograph_shapes import apply_shape
    from .inflow_closure import apply_closure_policy, save_closure_manifest
    from .timeseries_scenario import write_scenario_inp
    from ..validation.hydrograph_csv import HydrographScenario

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
    shape_run_dir = run_dir / shape_id
    save_closure_manifest(closure_manifest, shape_run_dir, scenario.scenario_id)
    scenario_inp = write_scenario_inp(
        config.network.inp_path, scenario, shape_run_dir,
        drain_down_hours=config.validation.drain_down_hours,
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
            scenario_inp, rpt_path = _prepare_closed_scenario_inp(
                config, shape_id, factor, shape, time_grid, base_inflows, run_dir,
            )
            with _Simulation(str(scenario_inp)) as sim:
                for _ in sim:
                    pass
            if not rpt_path.exists():
                raise FileNotFoundError(f"SWMM no generó .rpt: {rpt_path}")
            results.append((shape_id, factor, rpt_path))
            print("OK")

    return results
