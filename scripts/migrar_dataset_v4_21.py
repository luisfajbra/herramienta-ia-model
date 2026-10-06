"""Regenera dataset_final.csv con las 6 features tributarias del contrato
tabular_v4_21 (Fase B2), sin re-simular SWMM -- las features tributarias
dependen de la forma del hidrograma y la topologia, no de la salida de
SWMM, asi que se recalculan sobre las simulaciones ya corridas.

Nunca sobreescribe dataset_final.csv -- escribe un archivo nuevo con
identidad propia, tal como exige el spec (Sec.6, "el nuevo dataset tendra
identificador y hashes propios").

Uso:
    venv/Scripts/python.exe scripts/migrar_dataset_v4_21.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from swmm_resilience.config import load_config
from swmm_resilience.extraction.dynamic_features import compute_tributary_dynamic_features
from swmm_resilience.extraction.topology import build_network_graph
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V21
from swmm_resilience.simulation.hydrograph_shapes import apply_shape, load_all_shapes
from swmm_resilience.simulation.inflow_closure import apply_closure_policy
from swmm_resilience.simulation.swmm_api_io import get_node_timeseries_map, load_inp
from swmm_resilience.validation.hydrograph_csv import HydrographScenario

SOURCE = Path("data/training/dataset_final.csv")
OUTPUT = Path("data/training/dataset_final_v4_21.csv")
LEGACY_ONLY_COLUMNS = ("duracion_horas", "tiempo_al_pico_h")
TRIBUTARY_COLUMNS = (
    "volumen_local_integrado_m3",
    "volumen_tributario_integrado_m3",
    "duracion_efectiva_tributario_h",
    "tiempo_primer_pico_tributario_h",
    "tiempo_sobre_capacidad_h",
    "integral_exceso_capacidad_m3",
)


def base_node_series(inp, factor: float) -> dict[str, list[tuple[float, float]]]:
    """node_series de la forma 'base': la hidrografia ya embebida en el
    .inp, escalada por `factor` -- la misma multiplicacion que usa
    run_batch internamente (swmm_api_io._scale_target_timeseries:
    scaled_value = flow_value * multiplier, sin mfactor ni baseline)."""
    node_to_ts = get_node_timeseries_map(inp)
    series: dict[str, list[tuple[float, float]]] = {}
    for node_id, ts_name in node_to_ts.items():
        ts_obj = inp["TIMESERIES"].get(ts_name) if "TIMESERIES" in inp else None
        if ts_obj is None:
            continue
        series[node_id] = [(t, v * factor) for t, v in ts_obj.data]
    return series


def main() -> None:
    config = load_config("config.yaml")
    df = pd.read_csv(SOURCE)
    print(f"Dataset fuente: {len(df)} filas, {df['shape_id'].nunique()} formas (incl. 'base')")

    graph, _ = build_network_graph(load_inp(config.network.inp_path))
    inp_for_base = load_inp(config.network.inp_path)
    shapes = load_all_shapes(config.simulation.hydrograph_shapes_dir)

    base_inflows = df.drop_duplicates("node_id").set_index("node_id")["base_inflow_lps"].to_dict()
    capacity_by_node = df.drop_duplicates("node_id").set_index("node_id")["upstream_capacity_lps"].to_dict()

    out_frames = []
    groups = list(df.groupby(["shape_id", "factor_mult"]))
    for done, ((shape_id, factor), group) in enumerate(groups, 1):
        if shape_id == "base":
            node_series = base_node_series(inp_for_base, factor)
        else:
            shape = shapes[shape_id]
            raw_series = apply_shape(shape, base_inflows, factor)
            time_grid = [t for t, _ in shape]
            effective_scenario, _ = apply_closure_policy(
                HydrographScenario(
                    scenario_id=f"{shape_id}_f{factor:.3f}",
                    node_series=raw_series,
                    time_grid_hours=time_grid,
                    last_time_hours=shape[-1][0],
                ),
                config.validation.inflow_closure_tolerance_lps,
            )
            node_series = effective_scenario.node_series

        trib_df = compute_tributary_dynamic_features(
            group["node_id"].tolist(), node_series, graph, capacity_by_node,
        )
        trib_df = trib_df.set_index("node_id").loc[group["node_id"]].reset_index(drop=True)
        trib_df.index = group.index
        out_frames.append(trib_df)
        if done % 100 == 0 or done == len(groups):
            print(f"  features tributarias: {done}/{len(groups)} escenarios")

    tributary = pd.concat(out_frames).sort_index()
    merged = pd.concat([df, tributary], axis=1)
    merged = merged.drop(columns=list(LEGACY_ONLY_COLUMNS))

    expected = set(FEATURE_COLUMNS_V21)
    missing = expected - set(merged.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas del contrato tabular_v4_21: {sorted(missing)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False)
    print(f"Guardado: {OUTPUT} ({merged.shape[0]} filas x {merged.shape[1]} cols)")


if __name__ == "__main__":
    main()
