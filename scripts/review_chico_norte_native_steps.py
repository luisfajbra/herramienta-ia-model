"""Capture native SWMM steps in two fixed windows without forcing its timestep."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from pyswmm import Links, Nodes, Simulation

from scripts.review_chico_norte_hydraulics import ROOT, parse_routing_diagnostics, sha256
from swmm_resilience.extraction.labels import extract_labels

WINDOWS_SECONDS = ((0, 900), (40500, 45000))
VARIANTS = ("reference_5s", "routing_1s", "zero_initflow_5s")


def within_windows(elapsed_seconds: float) -> bool:
    return any(start <= elapsed_seconds <= end for start, end in WINDOWS_SECONDS)


def main() -> None:
    source = ROOT / "outputs/experiments/chico_norte_hydraulic_sensitivity_20261103T130312Z"
    design_source = json.loads((source / "design.json").read_text(encoding="utf-8"))
    prior = json.loads((source / "results.json").read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = ROOT / "outputs/experiments" / f"chico_norte_native_windows_{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    design = {"created_utc": stamp, "source_experiment": str(source),
              "source_design_sha256": sha256(source / "design.json"),
              "source_results_sha256": sha256(source / "results.json"),
              "windows_seconds_since_simulation_start": WINDOWS_SECONDS,
              "method": "capture elapsed decimal days returned by native swmm_step, converted to seconds; no stride or input change",
              "sources": []}
    for case in design_source["source_cases"]:
        for variant in VARIANTS:
            item = next(x for x in prior if x["case"] == case["name"] and x["variant"] == variant)
            folder = source / case["name"] / variant
            for name in ("scenario.inp", "scenario.rpt", "node_flood_volumes.csv"):
                if sha256(folder / name) != item["artifacts"][name]:
                    raise ValueError(f"source hash mismatch: {folder / name}")
            design["sources"].append({"case": case["name"], "variant": variant,
                                      "inp": str(folder / "scenario.inp"), "inp_sha256": item["artifacts"]["scenario.inp"],
                                      "rpt_sha256": item["artifacts"]["scenario.rpt"],
                                      "node_volumes_sha256": item["artifacts"]["node_flood_volumes.csv"]})
    (target / "design.json").write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    results = []
    for spec in design["sources"]:
        case, variant = spec["case"], spec["variant"]
        destination = target / case / variant
        destination.mkdir(parents=True, exist_ok=False)
        inp = destination / "scenario.inp"
        shutil.copyfile(spec["inp"], inp)
        if sha256(inp) != spec["inp_sha256"]:
            raise ValueError("copied INP hash mismatch")
        csv_path = destination / "native_windows.csv.gz"
        print(f"START {case} {variant}", flush=True)
        began = time.monotonic()
        count = 0
        with gzip.open(csv_path, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestamp", "elapsed_seconds", "node_523C_depth_m", "node_396I_depth_m",
                             "link_T507_flow_lps", "link_T508_flow_lps", "link_T463_flow_lps", "link_T92_flow_lps"])
            with Simulation(str(inp)) as sim:
                nodes, links = Nodes(sim), Links(sim)
                n1, n2 = nodes["523C"], nodes["396I"]
                l1, l2, l3, l4 = links["T507"], links["T508"], links["T463"], links["T92"]
                start = sim.start_time
                sim.start()
                while True:
                    elapsed_days = sim._model.swmm_step()
                    if elapsed_days <= 0:
                        break
                    elapsed = elapsed_days * 86400.0
                    if within_windows(elapsed):
                        timestamp = start + timedelta(seconds=elapsed)
                        writer.writerow([timestamp.isoformat(), elapsed, n1.depth, n2.depth,
                                         l1.flow, l2.flow, l3.flow, l4.flow])
                        count += 1
        rpt = inp.with_suffix(".rpt")
        diag = parse_routing_diagnostics(rpt.read_text(encoding="utf-8", errors="replace"))
        original = next(x for x in prior if x["case"] == case and x["variant"] == variant)
        keys = ("continuity_error_pct", "steps_not_converging_pct", "top_fii", "top_fii_link")
        if any(diag[key] != original[key] for key in keys):
            raise ValueError(f"native-step capture changed routing diagnostics: {case}/{variant}")
        initial_labels = pd.read_csv(source / case / variant / "node_flood_volumes.csv", dtype={"node_id": str})
        repeated_labels = extract_labels(rpt, initial_labels["node_id"].tolist())
        if not np.allclose(initial_labels["vol_inundacion_m3"], repeated_labels["vol_inundacion_m3"], rtol=0, atol=1e-6):
            raise ValueError(f"native-step capture changed node volumes: {case}/{variant}")
        if not count:
            raise ValueError("native-step windows contain no observations")
        results.append({"case": case, "variant": variant, "sample_count": count,
                        "elapsed_seconds": round(time.monotonic()-began, 2),
                        "reproduced_reference_diagnostics_and_node_volumes": True,
                        "artifacts": {p.name: sha256(p) for p in (inp, rpt, inp.with_suffix('.out'), csv_path)}})
        (target / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"DONE {case} {variant}: {count} native steps, {results[-1]['elapsed_seconds']}s", flush=True)
    print(f"OUTPUT {target}", flush=True)


if __name__ == "__main__":
    main()
