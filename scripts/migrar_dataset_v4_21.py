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
import hashlib
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from swmm_resilience.config import load_config
from swmm_resilience.extraction.dynamic_features import compute_tributary_dynamic_features
from swmm_resilience.extraction.topology import build_network_graph
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V21, TABULAR_V4_21
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_manifest(
    *,
    source_path: Path,
    output_path: Path,
    inp_path: Path,
    hydrograph_shapes_dir: Path,
    row_count: int,
) -> dict:
    """Describe the inputs that produced a v4_21 CSV and hash its contents."""
    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    inp_path = Path(inp_path).expanduser().resolve()
    shapes_dir = Path(hydrograph_shapes_dir).expanduser().resolve()
    if not source_path.is_file() or not output_path.is_file() or not inp_path.is_file():
        raise FileNotFoundError("Source CSV, output CSV and network INP must all exist")
    if not shapes_dir.is_dir():
        raise FileNotFoundError(f"Hydrograph shape directory does not exist: {shapes_dir}")
    shape_sources = {}
    for shape_path in sorted(shapes_dir.glob("*.csv")):
        shape_sources[shape_path.stem] = {
            "path": str(shape_path.resolve()),
            "sha256": _sha256_file(shape_path),
        }
    return {
        "format_version": 1,
        "feature_contract_id": "tabular_v4_21",
        "feature_contract_sha256": TABULAR_V4_21.descriptor_sha256,
        "ordered_features": list(FEATURE_COLUMNS_V21),
        "source_dataset": {"path": str(source_path), "sha256": _sha256_file(source_path)},
        "network_inp": {"path": str(inp_path), "sha256": _sha256_file(inp_path)},
        "hydrograph_shapes_dir": str(shapes_dir),
        "hydrograph_shapes": shape_sources,
        "output_dataset": {
            "path": str(output_path),
            "sha256": _sha256_file(output_path),
            "rows": int(row_count),
        },
    }


def validate_dataset_manifest(
    manifest_path: Path,
    dataset_path: Path,
    inp_path: Path,
    source_path: Path,
    hydrograph_shapes_dir: Path,
) -> dict:
    """Reject a stale dataset or one produced from another source network."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    dataset_path = Path(dataset_path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read v4_21 source manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"v4_21 source manifest must be a JSON object: {manifest_path}")
    if manifest.get("feature_contract_id") != "tabular_v4_21":
        raise ValueError("v4_21 source manifest declares the wrong feature contract")
    expected_contract_hash = TABULAR_V4_21.descriptor_sha256
    if manifest.get("feature_contract_sha256") != expected_contract_hash:
        raise ValueError("v4_21 source manifest has the wrong feature contract hash")
    if manifest.get("ordered_features") != list(FEATURE_COLUMNS_V21):
        raise ValueError("v4_21 source manifest has the wrong ordered feature columns")
    for field, path in (
        ("output_dataset", dataset_path),
        ("network_inp", Path(inp_path).expanduser().resolve()),
        ("source_dataset", Path(source_path).expanduser().resolve()),
    ):
        record = manifest.get(field)
        if not isinstance(record, dict) or record.get("path") != str(path):
            raise ValueError(f"v4_21 source manifest {field} path does not match {path}")
        if not path.is_file() or record.get("sha256") != _sha256_file(path):
            raise ValueError(f"v4_21 source manifest {field} SHA-256 does not match {path}")
    shapes_dir = Path(hydrograph_shapes_dir).expanduser().resolve()
    if not shapes_dir.is_dir():
        raise ValueError(f"Hydrograph shape directory does not exist: {shapes_dir}")
    if manifest.get("hydrograph_shapes_dir") != str(shapes_dir):
        raise ValueError(
            f"v4_21 source manifest hydrograph_shapes_dir does not match {shapes_dir}"
        )
    shape_records = manifest.get("hydrograph_shapes")
    if not isinstance(shape_records, dict):
        raise ValueError("v4_21 source manifest is missing hydrograph shape hashes")
    current_shapes = {path.stem: path for path in shapes_dir.glob("*.csv")}
    if set(shape_records) != set(current_shapes):
        raise ValueError("Hydrograph shape set differs from the v4_21 source manifest")
    for shape_id, path in current_shapes.items():
        record = shape_records[shape_id]
        if (
            not isinstance(record, dict)
            or record.get("path") != str(path.resolve())
            or record.get("sha256") != _sha256_file(path)
        ):
            raise ValueError(f"Hydrograph shape SHA-256 does not match: {path}")
    return manifest


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
    print(f"Source dataset: {len(df)} rows, {df['shape_id'].nunique()} shapes (incl. 'base')")

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
            print(f"  tributary features: {done}/{len(groups)} scenarios")

    tributary = pd.concat(out_frames).sort_index()
    merged = pd.concat([df, tributary], axis=1)
    merged = merged.drop(columns=list(LEGACY_ONLY_COLUMNS))

    expected = set(FEATURE_COLUMNS_V21)
    missing = expected - set(merged.columns)
    if missing:
        raise RuntimeError(f"Missing columns of the tabular_v4_21 contract: {sorted(missing)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False)
    manifest = build_dataset_manifest(
        source_path=SOURCE,
        output_path=OUTPUT,
        inp_path=config.network.inp_path,
        hydrograph_shapes_dir=config.simulation.hydrograph_shapes_dir,
        row_count=len(merged),
    )
    manifest_path = OUTPUT.with_suffix(OUTPUT.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved source manifest: {manifest_path}")
    print(f"Saved: {OUTPUT} ({merged.shape[0]} rows x {merged.shape[1]} cols)")


if __name__ == "__main__":
    main()
