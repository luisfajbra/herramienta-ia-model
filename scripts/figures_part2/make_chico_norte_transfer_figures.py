"""Generate audited parity plots and flood maps for frozen Chico Norte scores.

The script reads the sealed transfer predictions and manifest-linked SWMM RPTs.
Labels exist only in memory while plots and aggregate provenance are rendered;
no row-level target file is written and SQLite is opened read-only.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from swmm_resilience.extraction.labels import read_flooding_volumes_with_manifest  # noqa: E402
from swmm_resilience.ml.bench.frozen_transfer import _hash as canonical_hash  # noqa: E402
from swmm_resilience.ml.bench.frozen_transfer import _sha256  # noqa: E402
from swmm_resilience.visualization._inp_parser import (  # noqa: E402
    parse_conduits,
    parse_coordinates,
)


SOURCE_ROOT = (
    REPO_ROOT / "outputs" / "experiments" / "frozen_pair_rerun_20261028"
    / "20261028T194035Z" / "B"
)
SOURCE_MANIFEST = SOURCE_ROOT / "experiment_source_manifest.json"
FROZEN_BATCH_MANIFEST = SOURCE_ROOT / "frozen_b_manifest.json"
STAGING_DB = SOURCE_ROOT / "B_verified_label_free_v1.sqlite3"
TRANSFER_ROOTS = {
    "xgboost": REPO_ROOT / "outputs" / "bench" / "multinetwork"
    / "20261030T005341Z-813323e3ef75" / "A_to_B_frozen",
    "mlp": REPO_ROOT / "outputs" / "bench" / "multinetwork"
    / "20261030T011406Z-813323e3ef75" / "A_to_B_frozen",
}
OUTPUT_DIR = REPO_ROOT / "docs" / "figures_chico_norte_transferencia"
MODEL_NAMES = {"xgboost": "XGBoost", "mlp": "MLP (ensemble, 5 semillas)"}
MODEL_COLORS = {"xgboost": "#2878B5", "mlp": "#D97925"}
FLOOD_THRESHOLD_M3 = 1.0

KEY_COLUMNS = [
    "run_id",
    "network_id",
    "scenario_id",
    "scenario_key",
    "scenario_kind",
    "node_id",
    "factor_mult",
    "shape_id",
]
PREDICTION_COLUMNS = [
    *KEY_COLUMNS,
    "y_prob_clf",
    "y_pred_clf",
    "volume_conditional_m3",
    "volume_routed_m3",
]
MODEL_ORDER = ("xgboost", "mlp")
REPRESENTATIVE_QUANTILES = (("bajo", 0.10), ("mediano", 0.50), ("alto", 0.90))


def verify_sha256(path: str | Path, expected: str, role: str) -> str:
    """Hash a file and fail closed when its expected provenance is absent or wrong."""
    file_path = Path(path)
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(char not in "0123456789abcdef" for char in expected.lower())
    ):
        raise ValueError(f"{role} has no valid expected SHA-256")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected.lower():
        raise ValueError(f"{role} SHA-256 mismatch: {file_path}")
    return actual


def select_representative_scenarios(totals: pd.DataFrame) -> list[dict]:
    """Choose distinct eligible positive-volume runs nearest to p10/p50/p90."""
    required = {"scenario_key", "total_swmm_m3", "eligible"}
    missing = required - set(totals.columns)
    if missing:
        raise ValueError(f"Scenario totals are missing columns: {sorted(missing)}")
    frame = totals.loc[:, ["scenario_key", "total_swmm_m3", "eligible"]].copy()
    frame["scenario_key"] = frame["scenario_key"].astype(str)
    frame["total_swmm_m3"] = pd.to_numeric(frame["total_swmm_m3"], errors="coerce")
    if (
        frame["scenario_key"].duplicated().any()
        or not np.isfinite(frame["total_swmm_m3"].to_numpy(dtype=float)).all()
        or (frame["total_swmm_m3"] < 0).any()
        or not frame["eligible"].isin([True, False]).all()
    ):
        raise ValueError("Scenario totals contain duplicate keys or invalid values")
    eligible = frame.loc[
        frame["eligible"].eq(True)
        & frame["total_swmm_m3"].ge(FLOOD_THRESHOLD_M3)
    ].copy()
    if len(eligible) < len(REPRESENTATIVE_QUANTILES):
        raise ValueError("At least three distinct positive eligible runs are required")

    volumes = eligible["total_swmm_m3"].to_numpy(dtype=float)
    selected: list[dict] = []
    used: set[str] = set()
    for label, quantile in REPRESENTATIVE_QUANTILES:
        target = float(np.quantile(volumes, quantile, method="linear"))
        candidates = eligible.assign(
            distance=(eligible["total_swmm_m3"] - target).abs()
        ).sort_values(["distance", "scenario_key"], kind="mergesort")
        choice = next(row for row in candidates.itertuples(index=False) if row.scenario_key not in used)
        used.add(choice.scenario_key)
        selected.append(
            {
                "label": label,
                "quantile": quantile,
                "quantile_target_m3": target,
                "scenario_key": choice.scenario_key,
                "total_swmm_m3": float(choice.total_swmm_m3),
            }
        )
    return selected


def validate_prediction_frame(
    predictions: pd.DataFrame,
    inputs: pd.DataFrame,
    *,
    classification_threshold: float,
) -> None:
    """Validate sealed prediction schema, exact feature-key coverage, and cascade."""
    if predictions.columns.tolist() != PREDICTION_COLUMNS:
        raise ValueError("Sealed prediction file has an unsupported schema")
    if inputs.columns.tolist()[: len(KEY_COLUMNS)] != KEY_COLUMNS:
        raise ValueError("Sealed input file has an unsupported key schema")
    key_dtype = {
        column: "string"
        for column in ("scenario_key", "scenario_kind", "node_id", "shape_id")
    }
    prediction_keys = predictions.loc[:, KEY_COLUMNS].copy()
    input_keys = inputs.loc[:, KEY_COLUMNS].copy()
    for frame in (prediction_keys, input_keys):
        for column, dtype in key_dtype.items():
            frame[column] = frame[column].astype(dtype)
        for column in ("run_id", "network_id", "scenario_id"):
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
        frame["factor_mult"] = pd.to_numeric(frame["factor_mult"], errors="raise").astype(float)
    sort_keys = ["run_id", "node_id"]
    prediction_keys = prediction_keys.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    input_keys = input_keys.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    if prediction_keys.duplicated(sort_keys).any() or not prediction_keys.equals(input_keys):
        raise ValueError("Sealed predictions and inputs do not have exact unique run-node coverage")

    probability = predictions["y_prob_clf"].to_numpy(dtype=float)
    predicted_class = predictions["y_pred_clf"].to_numpy(dtype=float)
    conditional = predictions["volume_conditional_m3"].to_numpy(dtype=float)
    routed = predictions["volume_routed_m3"].to_numpy(dtype=float)
    if (
        not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
        or not np.isin(predicted_class, [0, 1]).all()
        or not np.isfinite(conditional).all()
        or (conditional < 0).any()
        or not np.isfinite(routed).all()
        or (routed < 0).any()
        or not np.array_equal(
            predicted_class.astype(int),
            (probability >= float(classification_threshold)).astype(int),
        )
        or not np.allclose(
            routed,
            np.where(predicted_class == 1, conditional, 0.0),
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise ValueError("Sealed predictions violate the registered cascade or contain invalid values")


def validate_run_eligibility(
    run_order: list[dict], run_status: list[dict], score: dict
) -> set[int]:
    """Require the source manifest, sealed admission, and score to name same runs."""
    source_by_id = {int(item["run_id"]): item for item in run_order}
    status_by_id = {int(item["run_id"]): item for item in run_status}
    if len(source_by_id) != len(run_order) or len(status_by_id) != len(run_status):
        raise ValueError("Source manifest or prediction seal has duplicate run IDs")
    if set(source_by_id) != set(status_by_id):
        raise ValueError("Source manifest and prediction seal run IDs differ")
    for run_id, source in source_by_id.items():
        status = status_by_id[run_id]
        if (
            status.get("eligible") is not (source.get("eligible") is True)
            or status.get("reason_codes", []) != source.get("reason_codes", [])
        ):
            raise ValueError(f"Sealed QA decision differs from source manifest for run {run_id}")
    eligible_ids = {run_id for run_id, item in source_by_id.items() if item.get("eligible") is True}
    score_ids = [int(run_id) for run_id in score.get("eligible_run_ids", [])]
    label_ids = [int(item["run_id"]) for item in score.get("label_sources", [])]
    if (
        len(score_ids) != len(set(score_ids))
        or len(label_ids) != len(set(label_ids))
        or set(score_ids) != eligible_ids
        or set(label_ids) != eligible_ids
    ):
        raise ValueError("Verified score eligible run IDs differ from the source manifest")
    return eligible_ids


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read verified JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def validate_registration_schema(registration: dict) -> None:
    """Validate the preregistration schema; its usage marker is label-blind."""
    if registration.get("schema") != "frozen_transfer_v1":
        raise ValueError("Registration has an unsupported schema")


def expected_nodes_by_run(registration_runs: list[dict], eligible_ids: set[int]) -> dict[int, set[str]]:
    """Return the registered node inventory for every eligible run exactly once."""
    by_id: dict[int, set[str]] = {}
    for item in registration_runs:
        run_id = int(item["run_id"])
        nodes = item.get("expected_node_ids")
        if (
            run_id in by_id
            or not isinstance(nodes, list)
            or not nodes
            or any(not isinstance(node, str) or not node.strip() for node in nodes)
            or len(set(nodes)) != len(nodes)
        ):
            raise ValueError("Registration has duplicate or invalid expected-node inventory")
        if run_id in eligible_ids:
            by_id[run_id] = set(nodes)
    if set(by_id) != set(eligible_ids):
        raise ValueError("Registration expected-node inventory omits an eligible run")
    return by_id


def _resolve_artifact(root: Path, item: dict) -> Path:
    path = Path(item["path"])
    return path.resolve(strict=True) if path.is_absolute() else (root / path).resolve(strict=True)


def _assert_database_remains_label_free(path: Path) -> None:
    uri = f"file:{path.resolve(strict=True).as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        row = conn.execute("SELECT COUNT(*) FROM node_results").fetchone()
    if row is None or int(row[0]) != 0:
        raise ValueError("Staging SQLite contains node_results; expected a label-free database")


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("scenario_key", "scenario_kind", "node_id", "shape_id"):
        result[column] = result[column].astype(str)
    for column in ("run_id", "network_id", "scenario_id"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    result["factor_mult"] = pd.to_numeric(result["factor_mult"], errors="raise").astype(float)
    return result


def load_verified_transfer(
    model_dirs: dict[str, Path], source_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Authenticate sealed transfer sources and reconstruct labels in memory."""
    source_root = Path(source_root).resolve(strict=True)
    source_manifest_path = source_root / "experiment_source_manifest.json"
    frozen_batch_path = source_root / "frozen_b_manifest.json"
    source_manifest = _read_json(source_manifest_path)
    frozen_batch = _read_json(frozen_batch_path)
    source_manifest_sha = verify_sha256(
        source_manifest_path, _sha256(source_manifest_path), "B source manifest"
    )
    if source_manifest.get("node_results_rows") != 0:
        raise ValueError("B source manifest does not attest a label-free SQLite staging")
    if source_manifest.get("input_manifest_path") != frozen_batch_path.name:
        raise ValueError("B source manifest points to an unexpected frozen batch manifest")
    verify_sha256(
        frozen_batch_path,
        source_manifest.get("input_manifest_sha256", ""),
        "frozen batch manifest",
    )
    protocol_path = source_root / source_manifest["protocol_path"]
    verify_sha256(protocol_path, source_manifest.get("protocol_sha256", ""), "frozen protocol")
    staging_db = source_root / "B_verified_label_free_v1.sqlite3"
    _assert_database_remains_label_free(staging_db)

    if frozen_batch.get("schema_version") != "frozen_b_transfer_manifest_v1" or frozen_batch.get("status") != "complete":
        raise ValueError("Frozen Chico Norte run manifest is incomplete or has an unsupported schema")
    source_runs = source_manifest.get("run_order")
    batch_runs = frozen_batch.get("runs")
    if not isinstance(source_runs, list) or not isinstance(batch_runs, list):
        raise ValueError("Frozen source and batch manifests must contain run lists")
    source_by_key = {str(item["scenario_key"]): item for item in source_runs}
    batch_by_key = {str(item["scenario_key"]): item for item in batch_runs}
    if len(source_by_key) != len(source_runs) or set(source_by_key) != set(batch_by_key):
        raise ValueError("Frozen source and batch manifests do not cover the same scenarios")

    model_payloads: dict[str, dict] = {}
    for model_name in MODEL_ORDER:
        transfer_dir = Path(model_dirs[model_name]).resolve(strict=True)
        registration_path = transfer_dir / "registration.json"
        registration_seal_path = transfer_dir / "registration_seal.json"
        seal_path = transfer_dir / "prediction_seal.json"
        predictions_path = transfer_dir / "predictions.csv"
        inputs_path = transfer_dir / "inputs.csv"
        score_path = transfer_dir / "verified_scores.json"
        registration = _read_json(registration_path)
        registration_seal = _read_json(registration_seal_path)
        seal = _read_json(seal_path)
        score = _read_json(score_path)
        artifact_name = f"{model_name}-verified_scores.json"
        preserved_score = (
            REPO_ROOT / "docs" / "superpowers" / "reports" / "artifacts"
            / "2026-10-30-chico-norte" / artifact_name
        )
        verify_sha256(score_path, _sha256(preserved_score), f"{model_name} preserved score JSON")
        registration_sha = verify_sha256(
            registration_path, registration_seal.get("sha256", ""), f"{model_name} registration"
        )
        if seal.get("registration_sha256") != registration_sha:
            raise ValueError(f"{model_name} prediction seal is detached from registration")
        verify_sha256(predictions_path, seal.get("predictions_sha256", ""), f"{model_name} predictions")
        verify_sha256(inputs_path, seal.get("inputs_sha256", ""), f"{model_name} inputs")
        if (
            score.get("registration_sha256") != seal.get("registration_sha256")
            or score.get("prediction_sha256") != seal.get("predictions_sha256")
            or score.get("input_sha256") != seal.get("inputs_sha256")
            or score.get("b_source_manifest_path") != source_manifest_path.name
            or score.get("b_source_manifest_sha256") != source_manifest_sha
            or score.get("labels_sha256") is None
        ):
            raise ValueError(f"{model_name} verified score is detached from its seals or B source manifest")
        if seal.get("rejections_sha256") is not None:
            verify_sha256(
                transfer_dir / "rejections.json",
                seal["rejections_sha256"],
                f"{model_name} rejection report",
            )
        validate_registration_schema(registration)
        candidate_dir = Path(registration["candidate_dir"]).resolve(strict=True)
        for filename, field in (
            ("metadata.json", "candidate_metadata_sha256"),
            ("classifier.joblib", "classifier_sha256"),
            ("regressor.joblib", "regressor_sha256"),
        ):
            verify_sha256(candidate_dir / filename, registration.get(field, ""), f"{model_name} candidate {filename}")
        if (
            registration.get("b_inp_sha256") != source_manifest.get("network_sha256")
            or registration.get("b_inp_sha256") != score.get("b_inp_sha256")
        ):
            raise ValueError(f"{model_name} registration and scored Chico Norte INP hashes differ")
        verify_sha256(
            Path(registration["b_inp_path"]),
            registration["b_inp_sha256"],
            "Chico Norte network INP",
        )

        predictions = pd.read_csv(
            predictions_path,
            dtype={"scenario_key": str, "scenario_kind": str, "node_id": str, "shape_id": str},
            keep_default_na=False,
        )
        inputs = pd.read_csv(
            inputs_path,
            dtype={"scenario_key": str, "scenario_kind": str, "node_id": str, "shape_id": str},
        )
        validate_prediction_frame(
            predictions,
            inputs,
            classification_threshold=float(registration["classification_threshold"]),
        )
        model_payloads[model_name] = {
            "directory": transfer_dir,
            "registration": registration,
            "seal": seal,
            "score": score,
            "predictions": _normalize_keys(predictions),
            "inputs": _normalize_keys(inputs),
        }

    base_model = model_payloads[MODEL_ORDER[0]]
    base_registration = base_model["registration"]
    base_keys = base_model["predictions"].loc[:, KEY_COLUMNS].sort_values(
        ["run_id", "node_id"], kind="mergesort"
    ).reset_index(drop=True)
    for model_name, payload in model_payloads.items():
        reg = payload["registration"]
        score = payload["score"]
        if (
            score.get("usage") != "verified_staged_sources"
            or score.get("flood_threshold_m3") != FLOOD_THRESHOLD_M3
            or reg.get("b_inp_sha256") != base_registration.get("b_inp_sha256")
            or score.get("labels_sha256") != base_model["score"].get("labels_sha256")
        ):
            raise ValueError("The XGBoost and MLP scores do not share the frozen Chico Norte label set")
        keys = payload["predictions"].loc[:, KEY_COLUMNS].sort_values(
            ["run_id", "node_id"], kind="mergesort"
        ).reset_index(drop=True)
        if not base_keys.equals(keys):
            raise ValueError(f"{model_name} predictions do not have the same run-node keys as XGBoost")
        status = payload["seal"].get("run_status")
        if not isinstance(status, list):
            raise ValueError(f"{model_name} prediction seal has no run-status list")
        eligible_ids = validate_run_eligibility(source_runs, status, score)
        predicted_ids = set(payload["predictions"]["run_id"].unique().astype(int))
        if predicted_ids != eligible_ids:
            raise ValueError(f"{model_name} predictions contain missing or excluded run IDs")
        expected_node_ids = expected_nodes_by_run(reg.get("b_runs", []), eligible_ids)
        for run_id, group in payload["predictions"].groupby("run_id", sort=False):
            if set(group["node_id"]) != expected_node_ids[int(run_id)] or len(group) != len(expected_node_ids[int(run_id)]):
                raise ValueError(f"{model_name} prediction node coverage differs for run {run_id}")

    eligible_ids = set(model_payloads[MODEL_ORDER[0]]["score"]["eligible_run_ids"])
    labels_by_run = {
        int(item["run_id"]): item
        for item in model_payloads[MODEL_ORDER[0]]["score"]["label_sources"]
    }
    predictions_xgb = model_payloads["xgboost"]["predictions"]
    network_ids = predictions_xgb.groupby("run_id")["network_id"].nunique()
    if (network_ids != 1).any():
        raise ValueError("A Chico Norte run maps to multiple local network IDs")
    network_by_run = predictions_xgb.groupby("run_id")["network_id"].first().astype(int).to_dict()
    truth_rows: list[pd.DataFrame] = []
    scenario_rows: list[dict] = []
    for entry in source_runs:
        run_id = int(entry["run_id"])
        if run_id not in eligible_ids:
            continue
        if int(entry["artifacts"]["swmm_rpt"]["byte_size"]) <= 0:
            raise ValueError(f"Eligible RPT has invalid byte size for run {run_id}")
        rpt_record = entry["artifacts"]["swmm_rpt"]
        rpt_path = _resolve_artifact(source_root, rpt_record)
        verify_sha256(rpt_path, rpt_record["sha256"], f"SWMM RPT run {run_id}")
        if run_id not in labels_by_run:
            raise ValueError(f"Verified score has no RPT label source for run {run_id}")
        label_source = labels_by_run[run_id]
        if (
            label_source.get("scenario_key") != entry["scenario_key"]
            or label_source.get("swmm_rpt_sha256") != rpt_record["sha256"]
            or entry["hydraulic_quality"]["flooding_report"]["report_sha256"] != rpt_record["sha256"]
        ):
            raise ValueError(f"Score RPT provenance differs from staged QA for run {run_id}")
        run_predictions = predictions_xgb.loc[predictions_xgb["run_id"].eq(run_id)]
        node_ids = run_predictions["node_id"].astype(str).tolist()
        if len(node_ids) != len(set(node_ids)):
            raise ValueError(f"Prediction keys contain duplicate nodes for run {run_id}")
        raw, report_manifest = read_flooding_volumes_with_manifest(rpt_path)
        if (
            report_manifest.get("report_sha256") != rpt_record["sha256"]
            or not set(raw["node_id"].astype(str)).issubset(set(node_ids))
        ):
            raise ValueError(f"RPT labels are not valid for the frozen node inventory in run {run_id}")
        volume_by_node = dict(zip(raw["node_id"].astype(str), raw["flooding_volume_m3"].astype(float)))
        labels = pd.DataFrame({"node_id": node_ids})
        labels["vol_inundacion_m3"] = labels["node_id"].map(volume_by_node).fillna(0.0).astype(float)
        labels["inunda"] = (labels["vol_inundacion_m3"] >= FLOOD_THRESHOLD_M3).astype(int)
        labels["network_id"] = network_by_run[run_id]
        labels["run_id"] = run_id
        labels = labels[["network_id", "run_id", "node_id", "inunda", "vol_inundacion_m3"]]
        if int(label_source.get("label_rows", -1)) != len(labels):
            raise ValueError(f"RPT label count differs from frozen score for run {run_id}")
        truth_rows.append(labels)
        scenario_rows.append(
            {
                "run_id": run_id,
                "scenario_key": entry["scenario_key"],
                "scenario_kind": entry["scenario"]["scenario_kind"],
                "shape_id": entry["scenario"]["shape_id"],
                "factor_mult": float(entry["scenario"]["factor_mult"]),
                "eligible": True,
                "total_swmm_m3": float(labels["vol_inundacion_m3"].sum()),
                "n_swmm_flooded": int(labels["inunda"].sum()),
            }
        )

    truth = pd.concat(truth_rows, ignore_index=True)
    if truth.duplicated(["network_id", "run_id", "node_id"]).any():
        raise ValueError("Reconstructed SWMM labels contain duplicate run-node keys")
    labels_hash = canonical_hash(
        truth.sort_values(["network_id", "run_id", "node_id"])
        .to_dict(orient="records")
    )
    if labels_hash != model_payloads["xgboost"]["score"]["labels_sha256"]:
        raise ValueError("Reconstructed SWMM labels differ from the frozen scoring label hash")

    combined_predictions = predictions_xgb.merge(
        model_payloads["mlp"]["predictions"],
        on=KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
        suffixes=("_xgb", "_mlp"),
        indicator=True,
    )
    if not combined_predictions["_merge"].eq("both").all():
        raise ValueError("XGBoost and MLP predictions do not cover identical run-node keys")
    combined_predictions = combined_predictions.drop(columns="_merge").merge(
        truth,
        on=["network_id", "run_id", "node_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not combined_predictions["_merge"].eq("both").all():
        raise ValueError("Reconstructed SWMM labels do not exactly cover sealed predictions")
    combined_predictions = combined_predictions.drop(columns="_merge")
    for model_name in MODEL_ORDER:
        validate_recomputed_score(
            combined_predictions,
            model_payloads[model_name]["score"]["overall"],
            model_name,
        )
    if len(combined_predictions) != 410_814 or len(eligible_ids) != 787:
        raise ValueError("Frozen Chico Norte evaluation row or eligible-run count has changed")
    scenario_totals = pd.DataFrame(scenario_rows).sort_values("scenario_key").reset_index(drop=True)
    provenance = {
        "source_manifest_sha256": source_manifest_sha,
        "frozen_batch_manifest_sha256": verify_sha256(
            frozen_batch_path, source_manifest["input_manifest_sha256"], "frozen batch manifest"
        ),
        "network_sha256": source_manifest["network_sha256"],
        "labels_sha256": labels_hash,
        "threshold_m3": FLOOD_THRESHOLD_M3,
        "eligible_runs": len(eligible_ids),
        "excluded_runs": len(source_runs) - len(eligible_ids),
        "rows": len(combined_predictions),
        "models": {
            name: {
                "registration_sha256": payload["score"]["registration_sha256"],
                "prediction_sha256": payload["score"]["prediction_sha256"],
                "input_sha256": payload["score"]["input_sha256"],
                "score_sha256": _sha256(payload["directory"] / "verified_scores.json"),
                "overall": payload["score"]["overall"],
            }
            for name, payload in model_payloads.items()
        },
    }
    return combined_predictions, scenario_totals, provenance


def _routed_column(model_name: str) -> str:
    return f"volume_routed_m3_{'xgb' if model_name == 'xgboost' else 'mlp'}"


def _class_column(model_name: str) -> str:
    return f"y_pred_clf_{'xgb' if model_name == 'xgboost' else 'mlp'}"


def _probability_column(model_name: str) -> str:
    return f"y_prob_clf_{'xgb' if model_name == 'xgboost' else 'mlp'}"


def _figure_metrics(frame: pd.DataFrame, model_name: str) -> dict:
    actual = frame["vol_inundacion_m3"].to_numpy(dtype=float)
    actual_class = actual >= FLOOD_THRESHOLD_M3
    predicted = frame[_routed_column(model_name)].to_numpy(dtype=float)
    predicted_class = frame[_class_column(model_name)].to_numpy(dtype=int).astype(bool)
    error = predicted - actual
    tp = int((actual_class & predicted_class).sum())
    fp = int((~actual_class & predicted_class).sum())
    fn = int((actual_class & ~predicted_class).sum())
    tn = int((~actual_class & ~predicted_class).sum())
    return {
        "n": int(len(frame)),
        "n_actual_flooded": int(actual_class.sum()),
        "n_predicted_flooded": int(predicted_class.sum()),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "rmse_m3": float(np.sqrt(np.mean(error**2))),
        "mae_m3": float(np.mean(np.abs(error))),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "total_swmm_m3": float(actual.sum()),
        "total_predicted_m3": float(predicted.sum()),
        "volume_bias_m3": float(predicted.sum() - actual.sum()),
    }


def validate_recomputed_score(frame: pd.DataFrame, frozen_score: dict, model_name: str) -> None:
    """Require aggregate figure metrics to reproduce the saved external score."""
    actual = _figure_metrics(frame, model_name)
    for key in ("n", "n_actual_flooded", "confusion_matrix"):
        score_key = "n_samples" if key == "n" else key
        if key == "n_actual_flooded":
            score_key = "n_actual_flooded"
        if actual[key] != frozen_score.get(score_key):
            raise ValueError(f"Recomputed figure metrics do not reproduce frozen score field {score_key}")
    for key in (
        "precision",
        "recall",
        "rmse_m3",
        "mae_m3",
        "volume_bias_m3",
    ):
        if not np.isclose(
            actual[key], float(frozen_score.get(key, np.nan)), rtol=1e-12, atol=1e-8
        ):
            raise ValueError(f"Recomputed figure metrics do not reproduce frozen score field {key}")


def _volume_ticks(max_volume: float) -> tuple[list[float], list[str]]:
    candidates = [0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0]
    values = [value for value in candidates if value <= max_volume]
    return [float(np.log10(1.0 + value)) for value in values], [f"{value:g}" for value in values]


def make_parity_figure(frame: pd.DataFrame, output_path: Path) -> dict:
    """Render pooled and actual-flooded node parity with zero-safe log axes."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    actual = frame["vol_inundacion_m3"].to_numpy(dtype=float)
    wet_frame = frame.loc[frame["inunda"].eq(1)]
    maximum = max(
        float(actual.max()),
        *(float(frame[_routed_column(name)].max()) for name in MODEL_ORDER),
    )
    transformed_max = float(np.log10(1.0 + maximum))
    ticks, ticklabels = _volume_ticks(maximum)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2), constrained_layout=True)
    subsets = (("Todos los pares corrida–nodo", frame), ("Nodos inundados por SWMM (≥1 m³)", wet_frame))
    plotted_metrics = {}
    for row, (subset_label, subset) in enumerate(subsets):
        for col, model_name in enumerate(MODEL_ORDER):
            ax = axes[row, col]
            target = subset["vol_inundacion_m3"].to_numpy(dtype=float)
            pred = subset[_routed_column(model_name)].to_numpy(dtype=float)
            x = np.log10(1.0 + target)
            y = np.log10(1.0 + pred)
            hb = ax.hexbin(
                x,
                y,
                gridsize=75,
                mincnt=1,
                bins="log",
                cmap="viridis",
                linewidths=0,
            )
            ax.plot([0, transformed_max], [0, transformed_max], "--", color="#bd2d28", linewidth=1.2, label="1:1")
            ax.set_xlim(0, transformed_max * 1.02)
            ax.set_ylim(0, transformed_max * 1.02)
            ax.set_xticks(ticks, ticklabels)
            ax.set_yticks(ticks, ticklabels)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(color="#d9dfe5", linewidth=0.5, alpha=0.7)
            ax.set_title(f"{MODEL_NAMES[model_name]} · {subset_label}")
            ax.set_xlabel("Volumen SWMM observado por nodo (m³)")
            ax.set_ylabel("Volumen predicho enrutado por nodo (m³)")
            current_metrics = _figure_metrics(subset, model_name)
            plotted_metrics[f"{model_name}_{'all' if row == 0 else 'swmm_flooded'}"] = current_metrics
            ax.text(
                0.025,
                0.97,
                f"n = {current_metrics['n']:,}\nRMSE = {current_metrics['rmse_m3']:.1f} m³\nMAE = {current_metrics['mae_m3']:.1f} m³",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd3da", "alpha": 0.93},
            )
            fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03, label="Densidad de pares (escala log)")
    fig.suptitle(
        "Chico Norte · paridad de volumen nodal\n"
        "Ejes: log₁₀(1 + volumen) para conservar ceros y mostrar el rango completo",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plotted_metrics


def _scenario_metrics(case: pd.DataFrame, model_name: str) -> dict:
    metrics = _figure_metrics(case, model_name)
    actual = case["inunda"].to_numpy(dtype=int).astype(bool)
    predicted = case[_class_column(model_name)].to_numpy(dtype=int).astype(bool)
    metrics["false_positives"] = int((~actual & predicted).sum())
    metrics["false_negatives"] = int((actual & ~predicted).sum())
    metrics["relative_total_volume_bias_pct"] = (
        float(metrics["volume_bias_m3"] / metrics["total_swmm_m3"] * 100.0)
        if metrics["total_swmm_m3"] > 0
        else None
    )
    return metrics


def make_flood_map(
    frame: pd.DataFrame,
    scenario: dict,
    run_id: int,
    inp_path: Path,
    output_path: Path,
    shared_max_volume: float,
) -> dict:
    """Render one SWMM/XGBoost/MLP comparison on frozen INP node coordinates."""
    from matplotlib.lines import Line2D

    case = frame.loc[frame["run_id"].eq(int(run_id))].copy()
    if len(case) != 522 or case["node_id"].duplicated().any():
        raise ValueError(f"Selected scenario {scenario['scenario_key']} lacks 522 unique node predictions")
    coordinates = parse_coordinates(inp_path)
    conduits = parse_conduits(inp_path)
    if not set(case["node_id"]).issubset(coordinates):
        raise ValueError(f"INP coordinates do not cover all nodes in {scenario['scenario_key']}")
    max_value = max(float(shared_max_volume), 1.0)
    norm = mcolors.PowerNorm(gamma=0.45, vmin=0.0, vmax=max_value)
    cmap = plt.get_cmap("YlOrRd")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 8.4), constrained_layout=True)
    model_stats: dict[str, dict] = {}
    node_xy = {node: coordinates[node] for node in case["node_id"]}
    for ax in axes:
        for _link_id, start_node, end_node in conduits:
            if start_node in coordinates and end_node in coordinates:
                x0, y0 = coordinates[start_node]
                x1, y1 = coordinates[end_node]
                ax.plot([x0, x1], [y0, y1], color="#9ba7b2", linewidth=0.48, alpha=0.43, zorder=1)
        ax.scatter(
            [xy[0] for xy in node_xy.values()],
            [xy[1] for xy in node_xy.values()],
            s=10,
            color="#dce2e7",
            linewidths=0,
            zorder=2,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()

    actual_volume = case["vol_inundacion_m3"].to_numpy(dtype=float)
    actual_flooded = actual_volume >= FLOOD_THRESHOLD_M3
    axes[0].scatter(
        case.loc[actual_flooded, "node_id"].map(lambda node: coordinates[node]).map(lambda xy: xy[0]),
        case.loc[actual_flooded, "node_id"].map(lambda node: coordinates[node]).map(lambda xy: xy[1]),
        c=actual_volume[actual_flooded],
        s=16.0 + 85.0 * np.sqrt(actual_volume[actual_flooded] / max_value),
        cmap=cmap,
        norm=norm,
        edgecolors="#4b5560",
        linewidths=0.35,
        zorder=3,
    )
    axes[0].set_title(
        f"SWMM\nTotal: {actual_volume.sum():,.0f} m³ · nodos inundados: {int(actual_flooded.sum())}",
        fontsize=10,
    )

    for ax, model_name in zip(axes[1:], MODEL_ORDER):
        pred_volume = case[_routed_column(model_name)].to_numpy(dtype=float)
        pred_flooded = case[_class_column(model_name)].to_numpy(dtype=int).astype(bool)
        false_positive = ~actual_flooded & pred_flooded
        false_negative = actual_flooded & ~pred_flooded
        positions = case["node_id"].map(lambda node: coordinates[node])
        ax.scatter(
            positions.loc[pred_flooded].map(lambda xy: xy[0]),
            positions.loc[pred_flooded].map(lambda xy: xy[1]),
            c=pred_volume[pred_flooded],
            s=16.0 + 85.0 * np.sqrt(pred_volume[pred_flooded] / max_value),
            cmap=cmap,
            norm=norm,
            edgecolors=MODEL_COLORS[model_name],
            linewidths=0.65,
            zorder=3,
        )
        if false_positive.any():
            fp_pos = positions.loc[false_positive]
            ax.scatter(
                fp_pos.map(lambda xy: xy[0]),
                fp_pos.map(lambda xy: xy[1]),
                marker="o",
                s=53,
                facecolors="none",
                edgecolors="#8d40a8",
                linewidths=1.2,
                zorder=5,
            )
        if false_negative.any():
            fn_pos = positions.loc[false_negative]
            ax.scatter(
                fn_pos.map(lambda xy: xy[0]),
                fn_pos.map(lambda xy: xy[1]),
                marker="x",
                s=34,
                c="#b3202a",
                linewidths=1.0,
                zorder=6,
            )
        stats = _scenario_metrics(case, model_name)
        model_stats[model_name] = stats
        ax.set_title(
            f"{MODEL_NAMES[model_name]}\nTotal: {stats['total_predicted_m3']:,.0f} m³ · RMSE nodo: {stats['rmse_m3']:.1f} m³\n"
            f"Inundados: {stats['n_predicted_flooded']} · FP: {stats['false_positives']} · FN: {stats['false_negatives']}",
            fontsize=9.5,
        )

    from matplotlib.cm import ScalarMappable

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=list(axes), shrink=0.72, pad=0.02, label="Volumen total nodal (m³; color con escala raíz)")
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#b7c0c8", markeredgecolor="#59636c", label="Nodo inundado según fuente del panel", markersize=7),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#8d40a8", label="Falso positivo del modelo", markersize=8),
            Line2D([0], [0], marker="x", color="#b3202a", label="Falso negativo del modelo", markersize=7),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        f"Chico Norte · escenario {scenario['label']} · {scenario['scenario_key']}\n"
        f"Cuantil p{int(scenario['quantile'] * 100)} del volumen SWMM positivo: {scenario['total_swmm_m3']:,.0f} m³",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.004,
        "Coordenadas y conducciones del INP; esquema espacial, sin sistema de referencia geográfica verificado. Umbral de inundación: 1 m³.",
        ha="center",
        fontsize=8,
        color="#4f5963",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "label": scenario["label"],
        "run_id": int(run_id),
        "scenario_key": scenario["scenario_key"],
        "quantile": scenario["quantile"],
        "quantile_target_m3": scenario["quantile_target_m3"],
        "total_swmm_m3": float(actual_volume.sum()),
        "n_swmm_flooded": int(actual_flooded.sum()),
        "models": model_stats,
    }


def generate_figures(
    frame: pd.DataFrame,
    scenario_totals: pd.DataFrame,
    provenance: dict,
    output_dir: Path,
) -> dict:
    """Write plots and aggregate provenance without exporting node-level targets."""
    output_dir = Path(output_dir)
    expected_outputs = [
        output_dir / "paridad_volumen_chico_norte.png",
        *(output_dir / f"mapa_inundacion_{label}_chico_norte.png" for label, _ in REPRESENTATIVE_QUANTILES),
        output_dir / "figure_manifest.json",
    ]
    existing = [path.name for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing figure artifacts: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    parity_path = output_dir / "paridad_volumen_chico_norte.png"
    parity_metrics = make_parity_figure(frame, parity_path)
    selected = select_representative_scenarios(scenario_totals)
    selected_runs = scenario_totals.set_index("scenario_key")["run_id"].to_dict()
    inp_path = Path(TRANSFER_ROOTS["xgboost"] / "registration.json")
    registration = _read_json(inp_path)
    inp_path = Path(registration["b_inp_path"]).resolve(strict=True)
    shared_max = max(
        float(frame["vol_inundacion_m3"].max()),
        *(float(frame[_routed_column(name)].max()) for name in MODEL_ORDER),
    )
    map_summaries = []
    for scenario in selected:
        run_id = int(selected_runs[scenario["scenario_key"]])
        map_path = output_dir / f"mapa_inundacion_{scenario['label']}_chico_norte.png"
        map_summaries.append(
            make_flood_map(frame, scenario, run_id, inp_path, map_path, shared_max)
        )

    artifact_paths = [parity_path, *(output_dir / f"mapa_inundacion_{item['label']}_chico_norte.png" for item in selected)]
    manifest = {
        "schema_version": "chico_norte_transfer_figures_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "frozen_batch_manifest_sha256": provenance["frozen_batch_manifest_sha256"],
        "network_sha256": provenance["network_sha256"],
        "labels_sha256": provenance["labels_sha256"],
        "flood_threshold_m3": FLOOD_THRESHOLD_M3,
        "eligible_runs": provenance["eligible_runs"],
        "excluded_runs": provenance["excluded_runs"],
        "run_node_rows": provenance["rows"],
        "label_persistence": "transient_from_hash_verified_rpt; no_node_level_target_file_written",
        "scenario_selection": {
            "population": "eligible_runs_with_total_swmm_flooding_ge_1_m3",
            "quantile_method": "linear",
            "selection": "nearest_total_to_p10_p50_p90; distinct_runs; ties_by_scenario_key_ascending",
            "cases": map_summaries,
        },
        "parity": {
            "figure": parity_path.name,
            "x_and_y_transform": "log10(1 + volume_m3)",
            "density": "hexbin_log_count",
            "metrics": parity_metrics,
        },
        "score_provenance": provenance["models"],
        "plot_software": {
            package: importlib.metadata.version(package)
            for package in ("matplotlib", "numpy", "pandas", "swmm-api")
        },
        "artifacts": [
            {"path": path.name, "sha256": _sha256(path), "byte_size": path.stat().st_size}
            for path in artifact_paths
        ],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    frame, scenario_totals, provenance = load_verified_transfer(TRANSFER_ROOTS, SOURCE_ROOT)
    manifest = generate_figures(frame, scenario_totals, provenance, OUTPUT_DIR)
    print(f"Generated figures in {OUTPUT_DIR}")
    print(f"Verified rows: {provenance['rows']}; eligible runs: {provenance['eligible_runs']}; label SHA-256: {provenance['labels_sha256']}")
    for item in manifest["scenario_selection"]["cases"]:
        print(
            f"{item['scenario_key']}: total SWMM = {item['total_swmm_m3']:.1f} m3; "
            f"RMSE XGBoost = {item['models']['xgboost']['rmse_m3']:.2f}; "
            f"RMSE MLP = {item['models']['mlp']['rmse_m3']:.2f}"
        )


if __name__ == "__main__":
    main()
