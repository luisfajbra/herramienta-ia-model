"""Verify and plot native-step windows from the Chico Norte sensitivity study."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.review_chico_norte_hydraulics import sha256
from scripts.summarize_chico_norte_hydraulic_sensitivity import COLORS, NAMES, VARIANTS

WINDOWS = {"inicio": (0, 900), "ventana_critica": (40500, 45000)}
VARIABLES = {"link_T507_flow_lps": ("T507", "Caudal (L/s)"),
             "link_T508_flow_lps": ("T508", "Caudal (L/s)"),
             "link_T463_flow_lps": ("T463", "Caudal (L/s)"),
             "link_T92_flow_lps": ("T92", "Caudal (L/s)"),
             "node_523C_depth_m": ("523C", "Profundidad (m)"),
             "node_396I_depth_m": ("396I", "Profundidad (m)")}


def compare_on_common_seconds(t1, y1, t2, y2) -> dict:
    """Compare linear interpolations only where both native series exist."""
    t1, y1, t2, y2 = (np.asarray(x, dtype=float) for x in (t1, y1, t2, y2))
    for times, values in ((t1, y1), (t2, y2)):
        if len(times) < 2 or len(times) != len(values) or not np.isfinite(times).all() or not np.isfinite(values).all():
            raise ValueError("invalid native time series")
        if not (np.diff(times) > 0).all():
            raise ValueError("native timestamps must increase strictly")
    start, end = math.ceil(max(t1[0], t2[0])), math.floor(min(t1[-1], t2[-1]))
    if end < start:
        raise ValueError("native series have no common whole-second interval")
    grid = np.arange(start, end + 1, dtype=float)
    difference = np.interp(grid, t2, y2) - np.interp(grid, t1, y1)
    peak_index = int(np.argmax(np.abs(difference)))
    return {"start_s": start, "end_s": end, "n_comparison_seconds": len(grid),
            "max_abs_difference": float(abs(difference[peak_index])),
            "time_of_max_difference_s": float(grid[peak_index]),
            "rmse_difference": float(np.sqrt(np.mean(difference ** 2)))}


def summarize(experiment: Path, evidence: Path) -> None:
    design = json.loads((experiment / "design.json").read_text(encoding="utf-8"))
    results = json.loads((experiment / "results.json").read_text(encoding="utf-8"))
    if len(results) != len(design["sources"]) or not all(x["reproduced_reference_diagnostics_and_node_volumes"] for x in results):
        raise ValueError("native-step experiment incomplete or did not reproduce controls")
    evidence.mkdir(parents=True, exist_ok=False)
    raw = evidence / "datos"
    graphs = evidence / "graficas"
    raw.mkdir()
    graphs.mkdir()
    shutil.copy2(experiment / "design.json", raw / "design.json")
    shutil.copy2(experiment / "results.json", raw / "results.json")
    tables = {}
    for item in results:
        case, variant = item["case"], item["variant"]
        folder = experiment / case / variant
        for name, expected in item["artifacts"].items():
            if sha256(folder / name) != expected:
                raise ValueError(f"changed native-step artifact: {folder / name}")
        prefix = f"{case}_{variant}"
        shutil.copy2(folder / "native_windows.csv.gz", raw / (prefix + ".csv.gz"))
        shutil.copy2(folder / "scenario.rpt", raw / (prefix + ".rpt"))
        frame = pd.read_csv(folder / "native_windows.csv.gz")
        if len(frame) != item["sample_count"] or not np.isfinite(frame["elapsed_seconds"]).all():
            raise ValueError("native time series is incomplete")
        tables[(case, variant)] = frame
    rows = []
    for case in dict.fromkeys(x["case"] for x in results):
        for window_name, (start, end) in WINDOWS.items():
            filtered = {variant: tables[(case, variant)].loc[
                tables[(case, variant)]["elapsed_seconds"].between(start, end)] for variant in VARIANTS}
            for column, (element, ylabel) in VARIABLES.items():
                baseline = filtered["reference_5s"]
                for variant in VARIANTS[1:]:
                    candidate = filtered[variant]
                    comparison = compare_on_common_seconds(baseline.elapsed_seconds, baseline[column],
                                                           candidate.elapsed_seconds, candidate[column])
                    rows.append({"caso": case, "ventana": window_name, "elemento": element,
                                 "variable": column, "variante": variant, **comparison})
                fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
                for variant in VARIANTS:
                    data = filtered[variant]
                    ax.plot(data.elapsed_seconds / 3600, data[column], color=COLORS[variant],
                            linewidth=0.95, label=NAMES[variant])
                ax.set_title(f"Chico Norte · {case.replace('_', ' ')} · {element}")
                ax.set_xlabel("Horas desde el inicio de simulación")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.25)
                ax.legend(frameon=False, fontsize=8)
                fig.savefig(graphs / f"{case}_{window_name}_{element}.png", dpi=170, facecolor="white")
                plt.close(fig)
    pd.DataFrame(rows).to_csv(evidence / "comparaciones_interpoladas_1s.csv", index=False)
    files = sorted(p for p in evidence.rglob("*") if p.is_file())
    manifest = {"experiment": str(experiment),
                "method": "all native routing steps in the declared windows; comparisons linearly interpolated to common whole seconds without extrapolation",
                "files_sha256": {p.relative_to(evidence).as_posix(): sha256(p) for p in files}}
    (evidence / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(results)} native-step reruns; wrote {len(rows)} interpolated comparisons and {len(list(graphs.glob('*.png')))} plots.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    summarize(args.experiment.resolve(strict=True), args.evidence.resolve())


if __name__ == "__main__":
    main()
