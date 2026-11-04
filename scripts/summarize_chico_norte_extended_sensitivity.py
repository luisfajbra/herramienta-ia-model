"""Verify and compare the ten documented exploratory Chico Norte cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.review_chico_norte_hydraulics import parse_routing_diagnostics, sha256
from scripts.run_chico_norte_extended_sensitivity import FACTORS, SHAPES
from swmm_resilience.extraction.labels import extract_labels

VARIANTS = ("reference_5s", "routing_1s", "zero_initflow_5s")
LABELS = ("Referencia: máximo 5 s", "Máximo 1 s", "InitFlow cero: máximo 5 s")
COLORS = ("#23435f", "#e77e24", "#278279")


def summarize(experiments: list[Path], output: Path) -> None:
    expected = {f"B:{shape}:f{factor:.3f}": (label, factor)
                for label, shape in SHAPES for factor in FACTORS}
    cases = {}
    for root in experiments:
        design = json.loads((root / "design.json").read_text(encoding="utf-8"))
        items = json.loads((root / "results.json").read_text(encoding="utf-8"))
        for source in design["source_cases"]:
            key = source["scenario_key"]
            if key not in expected:
                continue
            if key in cases or not source["eligible"]:
                raise ValueError(f"duplicate or ineligible case: {key}")
            for artifact in source["artifacts"].values():
                if sha256(Path(artifact["path"])) != artifact["sha256"]:
                    raise ValueError(f"source artifact changed: {artifact['path']}")
            if sha256(Path(source["manifest"])) != source["manifest_sha256"]:
                raise ValueError(f"source manifest changed: {key}")
            matching = [r for r in items if r["case"] == source["name"]]
            records = {r["variant"]: r for r in matching}
            if len(matching) != len(VARIANTS) or set(records) != set(VARIANTS):
                raise ValueError(f"incomplete case: {key}")
            for variant, record in records.items():
                folder = root / source["name"] / variant
                for name, digest in record["artifacts"].items():
                    if sha256(folder / name) != digest:
                        raise ValueError(f"modified output: {folder / name}")
            cases[key] = (root, source, records)
    if set(cases) != set(expected):
        raise ValueError(f"missing expected cases: {sorted(set(expected) - set(cases))}")

    rows, changed_classes = [], []
    for key, (shape, factor) in expected.items():
        root, source, records = cases[key]
        reference = None
        for variant in VARIANTS:
            record = records[variant]
            folder = root / source["name"] / variant
            nodes = pd.read_csv(folder / "node_flood_volumes.csv", dtype={"node_id": str}).set_index("node_id")
            if nodes.index.has_duplicates or nodes.empty:
                raise ValueError(f"invalid node inventory: {key}/{variant}")
            volumes = nodes["vol_inundacion_m3"].to_numpy(dtype=float)
            if (not np.isfinite(volumes).all() or (volumes < 0).any()
                    or not np.array_equal(nodes["inunda"].to_numpy(dtype=int), (volumes >= 1).astype(int))
                    or not np.isclose(volumes.sum(), record["flood_volume_m3_rpt"])
                    or int((volumes >= 1).sum()) != record["flooded_nodes_ge_1m3"]):
                raise ValueError(f"inconsistent node volumes or labels: {key}/{variant}")
            if reference is None:
                reference = nodes
                original = extract_labels(Path(source["artifacts"]["swmm_rpt"]["path"]), nodes.index.tolist()).set_index("node_id")
                original = original.reindex(nodes.index)
                # CSV parsing can add machine-epsilon noise; RPT values have 1 m³ resolution.
                if original.isna().any().any() or not np.allclose(nodes["vol_inundacion_m3"], original["vol_inundacion_m3"], rtol=0, atol=1e-6):
                    raise ValueError(f"baseline differs from frozen volumes: {key}")
                diag = parse_routing_diagnostics(Path(source["artifacts"]["swmm_rpt"]["path"]).read_text(encoding="utf-8", errors="replace"))
                if any(record[k] != diag[k] for k in ("continuity_error_pct", "steps_not_converging_pct", "top_fii_link", "top_fii")):
                    raise ValueError(f"baseline differs from frozen diagnostics: {key}")
            if set(nodes.index) != set(reference.index):
                raise ValueError(f"node inventories differ: {key}/{variant}")
            nodes = nodes.reindex(reference.index)
            delta = nodes["vol_inundacion_m3"] - reference["vol_inundacion_m3"]
            delta = delta.mask(delta.abs() <= 1e-6, 0.0)
            for node_id in nodes.index[nodes["inunda"] != reference["inunda"]]:
                changed_classes.append({"hidrograma": shape, "factor": factor, "variante": variant,
                                        "node_id": node_id,
                                        "volumen_referencia_m3": float(reference.loc[node_id, "vol_inundacion_m3"]),
                                        "volumen_variante_m3": float(nodes.loc[node_id, "vol_inundacion_m3"])})
            rows.append({"hidrograma": shape, "factor": factor, "variante": variant,
                         "continuidad_pct": record["continuity_error_pct"],
                         "pasos_sin_convergencia_pct": record["steps_not_converging_pct"],
                         "fii_max": record["top_fii"], "fii_conducto": record["top_fii_link"],
                         "volumen_total_m3": record["flood_volume_m3_rpt"],
                         "nodos_inundados_ge_1m3": record["flooded_nodes_ge_1m3"],
                         "diferencia_volumen_total_m3": float(delta.sum()),
                         "nodos_con_volumen_distinto": int((delta != 0).sum()),
                         "max_diferencia_nodo_m3": float(delta.abs().max()),
                         "nodos_cambian_clase": int((nodes["inunda"] != reference["inunda"]).sum()),
                         "scenario_key": key})
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "comparacion_diez_escenarios.csv", index=False)
    pd.DataFrame(changed_classes, columns=["hidrograma", "factor", "variante", "node_id",
                                        "volumen_referencia_m3", "volumen_variante_m3"]).to_csv(
        output / "cambios_clase_por_nodo.csv", index=False)
    for shape in ("pulsos", "meseta"):
        for column, title, unit in (
            ("volumen_total_m3", "Volumen de inundación", "Volumen nodal total (m³)"),
            ("pasos_sin_convergencia_pct", "Pasos sin convergencia", "Pasos sin convergencia (%)"),
            ("continuidad_pct", "Error de continuidad", "Error de continuidad (%)"),
        ):
            fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
            for variant, label, color in zip(VARIANTS, LABELS, COLORS):
                subset = frame[(frame.hidrograma == shape) & (frame.variante == variant)].sort_values("factor")
                ax.plot(subset.factor, subset[column], marker="o", label=label, color=color)
            ax.set(title=f"Chico Norte · {title} · {shape}", xlabel="Factor del hidrograma", ylabel=unit)
            ax.grid(alpha=0.22)
            ax.legend(frameon=False, fontsize=8)
            fig.savefig(output / f"{shape}_{column}.png", dpi=170, facecolor="white")
            plt.close(fig)
    manifest = {"source_experiments": [str(p.resolve()) for p in experiments],
                "source_design_and_results_sha256": {str(p / name): sha256(p / name) for p in experiments for name in ("design.json", "results.json")},
                "files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
                "baseline_volume_absolute_tolerance_m3": 1e-6,
                "warning": "Exploratorio, no prueba ciega ni validación física."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Verified {len(cases)} cases and {len(rows)} variants; wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("experiments", nargs="+", type=Path)
    args = parser.parse_args()
    summarize(args.experiments, args.output)
