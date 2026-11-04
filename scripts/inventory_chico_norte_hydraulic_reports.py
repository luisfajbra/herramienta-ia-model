"""Rebuild the report/endpoint inventory for the Chico Norte hydraulic review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd

from scripts.review_chico_norte_hydraulics import ROOT, parse_routing_diagnostics, sha256
from swmm_resilience.database.label_free_b_staging import _derive_hydraulic_quality, _inp_object_ids

SOURCES = (
    ("transferencia_20261030", ROOT / "outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z/B"),
    ("hidrogramas_nuevos", ROOT / "outputs/experiments/chico_norte_novel_hydrograph_eval_20261102/run_20261103T034108Z"),
)


def parse_flow_balance_m3(text: str) -> dict[str, float]:
    """Read SWMM LPS routing balance: the second column is 10^6 litres."""
    if "Flow Routing Continuity" not in text or "Time-Step Critical Elements" not in text:
        raise ValueError("missing routing balance boundaries")
    section = text.split("Flow Routing Continuity", 1)[1].split("Time-Step Critical Elements", 1)[0]
    if "10^6 ltr" not in section:
        raise ValueError("unsupported routing balance volume units")
    names = {"External Inflow": "aporte_externo_m3", "External Outflow": "salida_externa_m3",
             "Flooding Loss": "perdida_inundacion_m3", "Initial Stored Volume": "volumen_inicial_m3",
             "Final Stored Volume": "volumen_final_m3"}
    result = {}
    for label, key in names.items():
        match = re.search(rf"^\s*{re.escape(label)}\s*\.+\s*(-?[\d.]+)\s+(-?[\d.]+)", section, re.MULTILINE)
        if match is None:
            raise ValueError(f"missing routing balance row: {label}")
        result[key] = float(match.group(2)) * 1000.0
    return result


def compare_inputs(first: str, second: str, allowed: dict) -> dict[str, int]:
    """Reject any changed INP data field outside the declared sensitivity."""
    left, right = first.splitlines(), second.splitlines()
    if len(left) != len(right):
        raise ValueError("unexpected change in number of INP lines")
    section, counts = "", {}
    for number, (a, b) in enumerate(zip(left, right), start=1):
        x, y = a.split(";", 1)[0].strip(), b.split(";", 1)[0].strip()
        if x.startswith("[") and x.endswith("]"):
            section = x.upper()
        before, after = x.split(), y.split()
        if before == after:
            continue
        if (section == "[OPTIONS]" and before and before[0] in allowed.get("options", ())
                and len(before) == len(after) == 2 and before[0] == after[0]):
            key = before[0]
        elif (section == "[CONDUITS]" and allowed.get("init") and len(before) == len(after)
              and len(before) >= 8 and before[:7] == after[:7] and before[8:] == after[8:]
              and float(after[7]) == 0):
            key = "InitFlow"
        else:
            raise ValueError(f"unexpected change on line {number} in {section}")
        counts[key] = counts.get(key, 0) + 1
    return counts


def inventory(output: Path, sensitivity: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    rows, endpoints = [], []
    historical = None
    for name, root in SOURCES:
        data = json.loads((root / "experiment_source_manifest.json").read_text(encoding="utf-8"))
        if name == "transferencia_20261030":
            historical = data
        for item in data["run_order"]:
            report = item["artifacts"]["swmm_rpt"]
            path = root / report["path"]
            if sha256(path) != report["sha256"]:
                raise ValueError(f"RPT hash mismatch: {path}")
            diagnostic = parse_routing_diagnostics(path.read_text(encoding="utf-8", errors="replace"))
            rows.append({"lote": name, "scenario_key": item["scenario_key"], "run_id": item["run_id"],
                         "eligible": item["eligible"], "rpt_sha256": report["sha256"], **diagnostic})
            if name == "hidrogramas_nuevos":
                event = item["artifacts"]["event_source"]
                event_path = root / event["path"]
                if sha256(event_path) != event["sha256"]:
                    raise ValueError(f"event hash mismatch: {event_path}")
                frame = pd.read_csv(event_path)
                tail = frame.groupby("node_id", sort=False).tail(1)
                endpoints.append({"scenario_key": item["scenario_key"], "nodes": len(tail),
                                  "nodes_final_nonzero": int((tail["value_lps"].abs() > 0.01).sum()),
                                  "max_final_lps": float(tail["value_lps"].abs().max())})
    if historical is None:
        raise ValueError("missing historical manifest")
    pd.DataFrame(rows).to_csv(output / "diagnosticos_reportes_congelados.csv", index=False)
    pd.DataFrame(endpoints).to_csv(output / "cierre_eventos_nuevos.csv", index=False)

    root = SOURCES[0][1]
    exploratory = ROOT / "outputs/experiments/chico_norte_exploratory_a_to_b_20261026"
    qa = pd.read_csv(exploratory / "scenario_qa.csv")
    closure = []
    for item in historical["run_order"]:
        if item["hydraulic_quality"]["drainage_state"] != "not_drained":
            continue
        event = item["artifacts"]["event_source"]
        path = root / event["path"]
        if sha256(path) != event["sha256"]:
            raise ValueError(f"historical event hash mismatch: {path}")
        data = pd.read_csv(path)
        tail = data.groupby("node_id", sort=False).tail(1)
        shape, factor = item["scenario"]["shape_id"], item["scenario"]["factor_mult"]
        matched = qa[(qa.shape_id == shape) & ((qa.factor - factor).abs() < 1e-8)]
        if len(matched) != 1:
            raise ValueError(f"exploratory scenario not unique: {item['scenario_key']}")
        old = matched.iloc[0]
        if sha256(exploratory / Path(old.rpt)) != old.rpt_sha256:
            raise ValueError(f"exploratory RPT hash mismatch: {old.rpt}")
        closure.append({"scenario_key": item["scenario_key"], "shape": shape, "factor": factor,
                        "nodes": len(tail), "nodes_end_above_0_01_lps": int((tail.value_lps.abs() > 0.01).sum()),
                        "max_final_lps": float(tail.value_lps.abs().max()), "frozen_eligible": item["eligible"],
                        "frozen_continuity_pct": item["hydraulic_quality"]["continuity_error_pct"],
                        "frozen_drainage": item["hydraulic_quality"]["drainage_state"],
                        "exploratory_closure_nodes": int(old.closure_changed_node_count),
                        "exploratory_drainage": old.drainage_state,
                        "exploratory_eligible": bool(old.eligible_by_software_qa),
                        "exploratory_continuity_pct": float(old.continuity_error_pct),
                        "exploratory_rpt_sha256": old.rpt_sha256})
    pd.DataFrame(closure).to_csv(output / "cierre_lote_historico.csv", index=False)

    design = json.loads((sensitivity / "design.json").read_text(encoding="utf-8"))
    results = json.loads((sensitivity / "results.json").read_text(encoding="utf-8"))
    checks, balances = [], []
    for item in results:
        source = next(x for x in design["source_cases"] if x["name"] == item["case"])
        source_root = Path(source["manifest"]).parent
        policy = json.loads((source_root / "protocol.json").read_text(encoding="utf-8"))["hydraulic_policy"]
        progress = json.loads((source_root / "batch_progress.json").read_text(encoding="utf-8"))
        hours = progress["attempt_history"][source["scenario_key"]][-1]["drain_down_hours"]
        inp = sensitivity / item["case"] / item["variant"] / "scenario.inp"
        for role, path in (("scenario.inp", inp), ("scenario.rpt", inp.with_suffix(".rpt")),
                           ("scenario.out", inp.with_suffix(".out"))):
            if sha256(path) != item["artifacts"][role]:
                raise ValueError(f"sensitivity artifact hash mismatch: {path}")
        nodes, links = _inp_object_ids(inp)
        # Treat each copied INP as its own source for this diagnostic. In the
        # zero-InitFlow variant this deliberately does not certify equivalence
        # with the original network state or admit the run for training.
        q = _derive_hydraulic_quality(source_inp=inp, effective_inp=inp, rpt_path=inp.with_suffix(".rpt"),
                                      out_path=inp.with_suffix(".out"), expected_node_ids=nodes,
                                      expected_link_ids=links, policy=policy, drain_down_hours=hours)
        checks.append({"caso": item["case"], "variante": item["variant"], "horas_drenaje": hours,
                       "continuidad_pct": q["continuity_error_pct"], "cobertura_finita": q["finite_coverage_ok"],
                       "estado_drenaje": q["drainage_state"], "fallos_drenaje": "|".join(q["drainage_manifest"]["failed_checks"]),
                       "max_residual_lps": q["drainage_manifest"]["max_residual_flow_lps"],
                       "max_flooding_lps": q["drainage_manifest"]["max_flooding_lps"],
                       "max_storage_change_m3": q["drainage_manifest"]["max_storage_change_m3"]})
        balances.append({"caso": item["case"], "variante": item["variant"],
                         **parse_flow_balance_m3(inp.with_suffix(".rpt").read_text(encoding="utf-8", errors="replace"))})
    (output / "sensibilidad").mkdir()
    pd.DataFrame(checks).to_csv(output / "sensibilidad/qa_sensibilidad.csv", index=False)
    pd.DataFrame(balances).to_csv(output / "sensibilidad/balance_sensibilidad.csv", index=False)
    changes = []
    for source in design["source_cases"]:
        case = source["name"]
        original = Path(source["artifacts"]["effective_inp"]["path"])
        ref = sensitivity / case / "reference_5s/scenario.inp"
        comparisons = (
            ("reference_vs_original", original, ref, {"options": ["REPORT_STEP", "ROUTING_STEP"]}, {"REPORT_STEP": 1, "ROUTING_STEP": 1}),
            ("routing_1s_vs_reference", ref, sensitivity / case / "routing_1s/scenario.inp", {"options": ["ROUTING_STEP"]}, {"ROUTING_STEP": 1}),
            ("zero_initflow_vs_reference", ref, sensitivity / case / "zero_initflow_5s/scenario.inp", {"init": True}, {"InitFlow": 522}),
        )
        for name, left, right, allowed, expected in comparisons:
            actual = compare_inputs(left.read_text(encoding="utf-8"), right.read_text(encoding="utf-8"), allowed)
            if actual != expected:
                raise ValueError(f"unexpected field count in {case}/{name}: {actual}")
            changes.append({"caso": case, "comparacion": name,
                            "cambios_verificados": json.dumps(actual, sort_keys=True), "cambios_no_autorizados": 0})
    pd.DataFrame(changes).to_csv(output / "sensibilidad/cambios_entrada_verificados.csv", index=False)
    print(f"Verified {len(rows)} RPTs, {len(endpoints)} new event files, {len(closure)} historical closures and {len(checks)} sensitivity QA records.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("sensitivity", type=Path)
    args = parser.parse_args()
    inventory(args.output.resolve(), args.sensitivity.resolve(strict=True))


if __name__ == "__main__":
    main()
