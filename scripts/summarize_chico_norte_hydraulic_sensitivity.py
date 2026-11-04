"""Verify and summarize the isolated Chico Norte hydraulic sensitivity runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.review_chico_norte_hydraulics import parse_routing_diagnostics, sha256
from swmm_resilience.extraction.labels import extract_labels
from swmm_resilience.simulation.swmm_api_io import read_out_timeseries

VARIANTS = ("reference_5s", "routing_1s", "zero_initflow_5s")
NAMES = {"reference_5s": "Inicial fuente · paso máx. 5 s",
         "routing_1s": "Inicial fuente · paso máx. 1 s",
         "zero_initflow_5s": "InitFlow cero · paso máx. 5 s"}
COLORS = {"reference_5s": "#233F5C", "routing_1s": "#E87525", "zero_initflow_5s": "#2E8B83"}
OBJECTS = (("link", "T507", "flow", "Caudal (L/s)"),
           ("link", "T508", "flow", "Caudal (L/s)"),
           ("link", "T463", "flow", "Caudal (L/s)"),
           ("link", "T92", "flow", "Caudal (L/s)"),
           ("node", "523C", "depth", "Profundidad (m)"),
           ("node", "396I", "depth", "Profundidad (m)"))


def summarize(experiment: Path, evidence: Path) -> dict:
    design_path, result_path = experiment / "design.json", experiment / "results.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    results = json.loads(result_path.read_text(encoding="utf-8"))
    if len(results) != len(design["source_cases"]) * len(VARIANTS):
        raise ValueError("sensitivity experiment is incomplete")
    by_key = {(x["case"], x["variant"]): x for x in results}
    if len(by_key) != len(results):
        raise ValueError("duplicate case/variant")
    evidence.mkdir(parents=True, exist_ok=True)
    resources = evidence / "artefactos_sensibilidad"
    figures = evidence / "graficas_sensibilidad"
    resources.mkdir(exist_ok=False)
    figures.mkdir(exist_ok=False)
    shutil.copy2(design_path, resources / "design.json")
    shutil.copy2(result_path, resources / "results.json")
    row_metrics, series_metrics, source_checks = [], [], []
    for source in design["source_cases"]:
        case = source["name"]
        for role in ("effective_inp", "event_source", "swmm_rpt", "swmm_out"):
            rec = source["artifacts"][role]
            if sha256(Path(rec["path"])) != rec["sha256"]:
                raise ValueError(f"frozen source {role} changed")
        frames, labels = {}, {}
        for variant in VARIANTS:
            item = by_key[(case, variant)]
            folder = experiment / case / variant
            for filename, expected in item["artifacts"].items():
                if sha256(folder / filename) != expected:
                    raise ValueError(f"artifact changed: {folder / filename}")
            prefix = f"{case}_{variant}"
            for name, suffix in (("scenario.inp", ".inp"), ("scenario.rpt", ".rpt"),
                                 ("node_flood_volumes.csv", "_volumen_nodos.csv")):
                shutil.copy2(folder / name, resources / (prefix + suffix))
            labels[variant] = pd.read_csv(folder / "node_flood_volumes.csv", dtype={"node_id": str}).set_index("node_id")
            if labels[variant].index.has_duplicates:
                raise ValueError(f"duplicate node results in {prefix}")
            out = read_out_timeseries(folder / "scenario.out")
            series = {}
            for kind, name, variable, _unit in OBJECTS:
                frame = out.get_part(kind, name)
                series[f"{name}_{variable}"] = pd.Series(frame[variable].to_numpy(dtype=float), index=pd.to_datetime(frame.index))
            data = pd.DataFrame(series)
            if data.empty or not np.isfinite(data.to_numpy()).all() or data.index.has_duplicates:
                raise ValueError(f"invalid timeseries in {prefix}")
            data.index.name = "timestamp_swmm"
            data.to_csv(resources / (prefix + "_series_1min.csv"))
            frames[variant] = data
            if hasattr(out, "close"):
                out.close()

        reference_labels = labels["reference_5s"]
        ids = reference_labels.index.tolist()
        original = extract_labels(Path(source["artifacts"]["swmm_rpt"]["path"]), ids).set_index("node_id")
        orig_diag = parse_routing_diagnostics(Path(source["artifacts"]["swmm_rpt"]["path"]).read_text(encoding="utf-8", errors="replace"))
        ref_diag = by_key[(case, "reference_5s")]
        volume_match = np.allclose(reference_labels["vol_inundacion_m3"], original.loc[ids, "vol_inundacion_m3"], rtol=0, atol=1e-6)
        diag_match = all(ref_diag[key] == orig_diag[key] for key in ("continuity_error_pct", "steps_not_converging_pct", "top_fii_link", "top_fii"))
        source_checks.append({"case": case, "node_volumes_match": bool(volume_match), "key_diagnostics_match": bool(diag_match)})
        if not volume_match or not diag_match:
            raise ValueError(f"one-minute reference did not reproduce frozen case: {case}")
        ref_frame = frames["reference_5s"]
        for variant in VARIANTS:
            entry = by_key[(case, variant)]
            lab = labels[variant].reindex(ids)
            if lab.isna().any().any():
                raise ValueError(f"node inventory differs: {case}/{variant}")
            delta = lab["vol_inundacion_m3"] - reference_labels["vol_inundacion_m3"]
            row_metrics.append({"caso": case, "variante": variant,
                                "continuidad_pct": entry["continuity_error_pct"],
                                "pasos_sin_convergencia_pct": entry["steps_not_converging_pct"],
                                "fii_max": entry["top_fii"], "fii_conducto": entry["top_fii_link"],
                                "volumen_total_m3": entry["flood_volume_m3_rpt"],
                                "nodos_inundados_ge_1m3": entry["flooded_nodes_ge_1m3"],
                                "diferencia_volumen_total_m3": float(delta.sum()),
                                "nodos_con_volumen_distinto": int((delta.abs() > 1e-6).sum()),
                                "max_diferencia_nodo_m3": float(delta.abs().max()),
                                "rmse_diferencia_nodal_m3": float(np.sqrt(np.mean(np.square(delta))))})
            if not frames[variant].index.equals(ref_frame.index):
                raise ValueError(f"one-minute report timestamps differ: {case}/{variant}")
            for kind, name, variable, unit in OBJECTS:
                column = f"{name}_{variable}"
                difference = frames[variant][column] - ref_frame[column]
                series_metrics.append({"caso": case, "variante": variant, "elemento": name,
                                       "variable": variable, "unidad": unit,
                                       "maximo": float(frames[variant][column].max()),
                                       "minimo": float(frames[variant][column].min()),
                                       "mayor_diferencia_absoluta": float(difference.abs().max()),
                                       "instante_mayor_diferencia": str(difference.abs().idxmax()),
                                       "rmse_diferencia": float(np.sqrt(np.mean(np.square(difference))))})

        for kind, name, variable, unit in OBJECTS:
            column = f"{name}_{variable}"
            fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
            for variant in VARIANTS:
                frame = frames[variant]
                hours = (frame.index - frame.index[0]).total_seconds() / 3600
                ax.plot(hours, frame[column], label=NAMES[variant], color=COLORS[variant], linewidth=1.05)
            ax.set_title(f"Chico Norte · {case.replace('_', ' ')} · {name}")
            ax.set_xlabel("Horas desde el primer registro (1 min)")
            ax.set_ylabel(unit)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, frameon=False)
            fig.savefig(figures / f"{case}_{name}_{variable}.png", dpi=170, facecolor="white")
            plt.close(fig)
    pd.DataFrame(row_metrics).to_csv(evidence / "sensibilidad_resumen.csv", index=False)
    pd.DataFrame(series_metrics).to_csv(evidence / "sensibilidad_series_resumen.csv", index=False)
    files = sorted(p for p in evidence.rglob("*") if p.is_file() and p.name != "manifiesto_sensibilidad.json")
    manifest = {"experiment": str(experiment), "source_checks": source_checks,
                "sampling": "SWMM OUT at exact one-minute reporting intervals, with no time-stride forcing",
                "rpt_volume_resolution_m3": 1,
                "files_sha256": {p.relative_to(evidence).as_posix(): sha256(p) for p in files}}
    (evidence / "manifiesto_sensibilidad.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    result = summarize(args.experiment.resolve(strict=True), args.evidence.resolve())
    print(f"Verified {len(result['source_checks'])} baseline cases; wrote {len(result['files_sha256'])} evidence files.")


if __name__ == "__main__":
    main()
