"""Reproduce a small, isolated hydraulic sensitivity study in Chico Norte.

The experiment copies authenticated effective INPs; it never changes frozen
inputs, scores, models or databases. Report interval is one minute in every
variant, while the routing maximum or conduit initial flows vary separately.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("historico_no_convergencia", ROOT / "outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z/B",
     "B:shape_quasi_uniform_block:f1.400"),
    ("meseta_nueva_factor_4_3", ROOT / "outputs/experiments/chico_norte_novel_hydrograph_eval_20261102/run_20261103T034108Z",
     "B:novel_20261102_sustained_late_surge:f4.300"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_sensitivity_inp(text: str, *, routing_seconds: int, zero_initial: bool) -> tuple[str, int]:
    if routing_seconds not in (1, 5):
        raise ValueError("routing_seconds must be 1 or 5")
    section = ""
    found = {"REPORT_STEP": 0, "ROUTING_STEP": 0}
    changed_initial = 0
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.upper()
        elif section == "[OPTIONS]" and stripped and not stripped.startswith(";"):
            key = stripped.split()[0].upper()
            if key in found:
                found[key] += 1
                line = f"{key} " + ("00:01:00" if key == "REPORT_STEP" else f"00:00:{routing_seconds:02d}")
        elif section == "[CONDUITS]" and stripped and not stripped.startswith(";") and zero_initial:
            body, sep, comment = line.partition(";")
            columns = body.split()
            if len(columns) < 8:
                raise ValueError(f"invalid conduit row: {line}")
            float(columns[7])
            columns[7] = "0"
            line = " ".join(columns) + (" ;" + comment if sep else "")
            changed_initial += 1
        out.append(line)
    if any(count != 1 for count in found.values()):
        raise ValueError(f"expected one of each option; found {found}")
    if zero_initial and not changed_initial:
        raise ValueError("no conduit initial flows were changed")
    return "\n".join(out) + "\n", changed_initial


def parse_routing_diagnostics(text: str) -> dict:
    def value(pattern: str, cast=float):
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match is None:
            raise ValueError(f"missing SWMM routing diagnostic: {pattern}")
        return cast(match.group(1))

    fii = re.search(r"Highest Flow Instability Indexes\s*\*+\s*Link\s+(\S+)\s+\((\d+)\)", text)
    node = re.search(r"Most Frequent Nonconverging Nodes\s*\*+\s*Node\s+(\S+)\s+\(([\d.]+)%\)", text)
    return {
        "continuity_error_pct": value(r"Continuity Error \(%\)\s*\.*\s*(-?[\d.]+)"),
        "steps_not_converging_pct": value(r"% of Steps Not Converging\s*:\s*([\d.]+)"),
        "routing_min_s": value(r"Minimum Time Step\s*:\s*([\d.]+)"),
        "routing_average_s": value(r"Average Time Step\s*:\s*([\d.]+)"),
        "routing_max_s": value(r"Maximum Time Step\s*:\s*([\d.]+)"),
        "top_fii_link": fii.group(1) if fii else None,
        "top_fii": int(fii.group(2)) if fii else None,
        "top_nonconverging_node": node.group(1) if node else None,
        "top_nonconverging_node_pct": float(node.group(2)) if node else None,
    }


def main() -> None:
    import pandas as pd
    from pyswmm import Links, Nodes, Simulation
    from swmm_resilience.extraction.labels import extract_labels
    from swmm_resilience.extraction.static_features import extract_static_features

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "outputs/experiments" / f"chico_norte_hydraulic_sensitivity_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    design = {"created_utc": stamp, "sampling_seconds_target": 60,
              "sampling_method": "first native SWMM routing step at or after each minute; no stride",
              "source_cases": [],
              "variants": {"reference_5s": {"routing_max_s": 5, "zero_initflow": False},
                           "routing_1s": {"routing_max_s": 1, "zero_initflow": False},
                           "zero_initflow_5s": {"routing_max_s": 5, "zero_initflow": True}},
              "warning": "Exploratory synthetic hydraulic sensitivity, not model training or network admission."}
    for label, root, key in SOURCES:
        manifest = root / "experiment_source_manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        matches = [x for x in data["run_order"] if x["scenario_key"] == key]
        if len(matches) != 1:
            raise ValueError(f"scenario not unique: {key}")
        item = matches[0]
        records = {}
        for role in ("effective_inp", "event_source", "swmm_rpt", "swmm_out"):
            artifact = item["artifacts"][role]
            path = root / artifact["path"]
            if sha256(path) != artifact["sha256"]:
                raise ValueError(f"source {role} hash mismatch: {path}")
            records[role] = {"path": str(path), "sha256": artifact["sha256"]}
        design["source_cases"].append({"name": label, "scenario_key": key, "eligible": item["eligible"],
                                       "manifest": str(manifest), "manifest_sha256": sha256(manifest),
                                       "artifacts": records})
    (output / "design.json").write_text(json.dumps(design, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    results = []
    for case in design["source_cases"]:
        source = Path(case["artifacts"]["effective_inp"]["path"])
        source_text = source.read_text(encoding="utf-8")
        node_ids = extract_static_features(source)["node_id"].astype(str).tolist()
        for variant, cfg in design["variants"].items():
            target = output / case["name"] / variant
            target.mkdir(parents=True, exist_ok=False)
            inp = target / "scenario.inp"
            text, changed = make_sensitivity_inp(source_text, routing_seconds=cfg["routing_max_s"], zero_initial=cfg["zero_initflow"])
            if cfg["zero_initflow"] and changed != 522:
                raise ValueError(f"expected 522 conduits, found {changed}")
            inp.write_text(text, encoding="utf-8")
            started = time.monotonic()
            print(f"START {case['name']} {variant}", flush=True)
            with (target / "selected_timeseries.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time", "node_523C_depth_m", "node_523C_head_m", "node_523C_flooding_lps",
                                 "node_396I_depth_m", "node_396I_head_m", "node_396I_flooding_lps",
                                 "link_T507_flow_lps", "link_T507_depth_m", "link_T508_flow_lps", "link_T508_depth_m"])
                with Simulation(str(inp)) as sim:
                    nodes, links = Nodes(sim), Links(sim)
                    n1, n2, l1, l2 = nodes["523C"], nodes["396I"], links["T507"], links["T508"]
                    next_sample = sim.start_time + timedelta(seconds=60)
                    for _ in sim:
                        if sim.current_time < next_sample:
                            continue
                        writer.writerow([sim.current_time.isoformat(), n1.depth, n1.head, n1.flooding,
                                         n2.depth, n2.head, n2.flooding, l1.flow, l1.depth, l2.flow, l2.depth])
                        while next_sample <= sim.current_time:
                            next_sample += timedelta(seconds=60)
            rpt = inp.with_suffix(".rpt")
            out = inp.with_suffix(".out")
            if not rpt.is_file() or not out.is_file() or out.stat().st_size == 0:
                raise RuntimeError(f"SWMM did not create report/output: {inp}")
            diagnostics = parse_routing_diagnostics(rpt.read_text(encoding="utf-8", errors="replace"))
            labels = extract_labels(rpt, node_ids)
            labels.to_csv(target / "node_flood_volumes.csv", index=False)
            samples = pd.read_csv(target / "selected_timeseries.csv")
            if samples.empty:
                raise ValueError("no hydraulic time-series samples")
            row = {"case": case["name"], "scenario_key": case["scenario_key"], "variant": variant,
                   "elapsed_seconds": round(time.monotonic() - started, 2), "changed_initial_flows": changed,
                   **diagnostics, "flood_volume_m3_rpt": float(labels["vol_inundacion_m3"].sum()),
                   "flooded_nodes_ge_1m3": int(labels["inunda"].sum()), "samples": len(samples),
                   "artifacts": {p.name: sha256(p) for p in (inp, rpt, out, target / "selected_timeseries.csv", target / "node_flood_volumes.csv")}}
            results.append(row)
            (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"DONE {case['name']} {variant}: {diagnostics['steps_not_converging_pct']}% nonconvergent, {row['elapsed_seconds']}s", flush=True)
    print(f"OUTPUT {output}", flush=True)


if __name__ == "__main__":
    main()
