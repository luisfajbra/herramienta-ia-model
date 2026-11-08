"""Post hoc audit of the sealed joint holdout; never refits, reselects or rescores officially."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.prepare_joint_dataset import development_run_ids, verify_partition_hash
from swmm_resilience.ml.bench.evaluate import _nested_reporting_entry
from swmm_resilience.ml.bench.holdout_audit import (
    continuity_subset, leave_one_shape_out_macro, macro_network_rmse, rmse_by_shape_network)
from swmm_resilience.ml.bench.partition_coverage import inflow_coverage, run_inflow_volumes
from swmm_resilience.ml.bench.physical_checks import (
    cap_at_tributary_volume, node_monotonicity_violations, total_monotonicity_violations,
    tributary_bound_violations)

KEYS = ["network_id", "run_id", "node_id"]
FAMILIES = ("xgboost", "mlp")
BANDS = (24.0, 108.0, 344.0)
FLOOD_THRESHOLD_M3 = 1.0
CONTINUITY_LIMITS_PCT = (2.0, 1.0)


def sha256(path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def verify_sealed_evaluation(directory: Path) -> dict:
    directory = Path(directory)
    status = _read_json(directory / "status.json")
    if status.get("status") != "holdout_scored" or sha256(directory / "scores.json") != status.get("scores_sha256"):
        raise ValueError("sealed scores changed")
    scores = _read_json(directory / "scores.json")
    if sha256(directory / "prediction_seal.json") != scores.get("seal_sha256"):
        raise ValueError("sealed prediction seal changed")
    if sha256(directory / "truth.parquet") != scores.get("truth_sha256"):
        raise ValueError("sealed truth changed")
    seal = _read_json(directory / "prediction_seal.json")
    if sha256(directory / "holdout_inputs.parquet") != seal.get("inputs_sha256"):
        raise ValueError("sealed inputs changed")
    for family in FAMILIES:
        if sha256(directory / f"{family}_predictions.parquet") != seal["models"][family]["predictions_sha256"]:
            raise ValueError(f"sealed {family} predictions changed")
    return {"status": status, "scores": scores, "seal": seal}


def verify_partition_matches_seal(partition_path: Path, partition_sha256: str, seal: dict) -> None:
    """The audited partition must be the one the sealed evaluation was run against."""
    if partition_sha256 != seal.get("partition_sha256"):
        raise ValueError("partition does not match the sealed evaluation")
    data = _read_json(Path(partition_path))
    partition_holdout = {int(run) for run in data["run_ids"]["holdout"]}
    if partition_holdout != {int(run) for run in seal["holdout_run_ids"]}:
        raise ValueError("partition does not match the sealed evaluation")


def assemble_family_frame(inputs: pd.DataFrame, truth: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if not (len(inputs) == len(truth) == len(predictions)):
        raise ValueError("sealed inputs, predictions and truth do not cover the same pairs")
    frame = (predictions.loc[:, KEYS + ["y_prob_clf", "y_pred_clf", "volume_routed_m3"]]
             .merge(inputs.loc[:, KEYS + ["shape_id", "factor_mult", "volumen_tributario_integrado_m3"]],
                    on=KEYS, how="inner", validate="one_to_one")
             .merge(truth.loc[:, KEYS + ["inunda", "vol_inundacion_m3"]],
                    on=KEYS, how="inner", validate="one_to_one"))
    if len(frame) != len(predictions):
        raise ValueError("sealed inputs, predictions and truth do not cover the same pairs")
    return frame


def _metrics_by_network(frame: pd.DataFrame) -> dict:
    return {str(network): _nested_reporting_entry(group.inunda, group.y_pred_clf, group.y_prob_clf,
                                                  group.vol_inundacion_m3, group.volume_routed_m3,
                                                  BANDS, FLOOD_THRESHOLD_M3)
            for network, group in frame.groupby("network_id", sort=True)}


def audit_family(frame: pd.DataFrame, official_macro: float, continuity: pd.Series) -> dict:
    recomputed = macro_network_rmse(frame)
    if not np.isclose(recomputed, official_macro, rtol=1e-9, atol=1e-9):
        raise ValueError("macro RMSE does not reproduce the sealed score")
    violations = tributary_bound_violations(frame)
    capped = frame.assign(volume_routed_m3=cap_at_tributary_volume(
        frame["volume_routed_m3"], frame["volumen_tributario_integrado_m3"]))
    sensitivity = {}
    for limit in CONTINUITY_LIMITS_PCT:
        subset = continuity_subset(frame, continuity, limit)
        sensitivity[f"max_abs_{limit:g}_pct"] = {
            "runs": int(subset["run_id"].nunique()), "pairs": int(len(subset)),
            "macro_network_rmse_m3": macro_network_rmse(subset), "by_network": _metrics_by_network(subset)}
    return {"macro_network_rmse_m3": recomputed,
            "by_shape_network": _records(rmse_by_shape_network(frame)),
            "leave_one_shape_out": _records(leave_one_shape_out_macro(frame)),
            "continuity_sensitivity": sensitivity,
            "tributary_bound": {"tolerance_m3": 1.0, "violations": int(len(violations)),
                                "max_ratio": float(violations["ratio"].max()) if len(violations) else None,
                                "capped_macro_network_rmse_m3": macro_network_rmse(capped)},
            "total_monotonicity_violations": _records(total_monotonicity_violations(frame, "volume_routed_m3")),
            "node_monotonicity": node_monotonicity_violations(frame, "volume_routed_m3")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("evaluation", "package", "partition", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--partition-sha256", required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    for frozen in (args.evaluation.resolve(), args.package.resolve()):
        if output == frozen or frozen in output.parents:
            raise ValueError("output must be outside sealed inputs")
    sealed = verify_sealed_evaluation(args.evaluation)
    verify_partition_hash(args.partition, args.partition_sha256)
    verify_partition_matches_seal(args.partition, args.partition_sha256, sealed["seal"])
    database = args.package / "joint.sqlite3"
    if sha256(database) != sealed["seal"]["source_database_sha256"]:
        raise ValueError("joint database differs from the sealed evaluation")
    development = development_run_ids(args.package, args.partition)
    holdout = [int(run) for run in sealed["seal"]["holdout_run_ids"]]
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as connection:
        marks = ",".join("?" for _ in holdout)
        continuity = pd.read_sql_query(
            f"SELECT run_id, continuity_error_pct FROM runs WHERE run_id IN ({marks})",
            connection, params=holdout).set_index("run_id")["continuity_error_pct"]
        dev_marks = ",".join("?" for _ in development)
        dev_features = pd.read_sql_query(
            "SELECT network_id, run_id, shape_id, factor_mult, volumen_local_integrado_m3 "
            f"FROM training_samples_v4_21 WHERE run_id IN ({dev_marks})", connection, params=development)
    inputs = pd.read_parquet(args.evaluation / "holdout_inputs.parquet")
    truth = pd.read_parquet(args.evaluation / "truth.parquet")
    report = {"schema": "joint_holdout_post_hoc_audit_v1",
              "status": "post_hoc_audit_not_model_selection",
              "scope": "descriptive sensitivity of sealed predictions; official metrics stay in scores.json",
              "evaluation_scores_sha256": sealed["status"]["scores_sha256"],
              "source_database_sha256": sealed["seal"]["source_database_sha256"],
              "partition_sha256": args.partition_sha256, "models": {}}
    for family in FAMILIES:
        predictions = pd.read_parquet(args.evaluation / f"{family}_predictions.parquet")
        frame = assemble_family_frame(inputs, truth, predictions)
        official = float(sealed["scores"]["models"][family]["macro_network_rmse_m3"])
        report["models"][family] = audit_family(frame, official, continuity)
    volumes = pd.concat([run_inflow_volumes(dev_features), run_inflow_volumes(inputs)], ignore_index=True)
    report["inflow_coverage"] = _records(inflow_coverage(volumes, development, holdout))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "audit.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output / "audit.json")}))


if __name__ == "__main__":
    main()
