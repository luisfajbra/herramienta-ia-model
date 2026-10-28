"""Etapa 2 del banco: ajuste sobre todo el dataset y escritura de artefactos.

Intencionalmente idéntico a lo que hace ml/trainer.py: clasificador sobre
todas las filas, regresor sobre inunda == 1 con objetivo log1p. Esa identidad
es la condición para que el test de paridad pueda pasar.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as _package_version
from pathlib import Path

import joblib
import numpy as np

from ...config import ML_RANDOM_STATE
from ..contracts import validate_feature_contract_manifest
from ..model_provenance import (
    ModelProvenanceError,
    validate_estimators,
    validate_pair_files,
)
from .registry import get_family
from .schemas import PreparedDataset

MLP_SEEDS = tuple(ML_RANDOM_STATE + offset for offset in range(5))


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _snapshot_source_files(source_networks: list, directory: Path) -> list[dict]:
    """Snapshot every file named by network/event/dataset source manifests."""
    references = {}

    def collect(value):
        if isinstance(value, dict):
            for path_key, hash_key in (("inp_path", "network_sha256"),
                                       ("hydrograph_path", "hydrograph_sha256"),
                                       ("path", "sha256")):
                if value.get(path_key):
                    path = Path(value[path_key]).expanduser().resolve()
                    expected = value.get(hash_key)
                    if path in references and references[path] != expected:
                        raise ModelProvenanceError(f"Conflicting source hashes for {path}")
                    references[path] = expected
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(source_networks)
    snapshots = []
    for path, expected in sorted(references.items(), key=lambda item: str(item[0])):
        if not path.is_file() or _sha256(path) != expected:
            raise ModelProvenanceError(f"Source file SHA-256 does not match manifest: {path}")
        snapshot_name = f"source-{expected}.bin"
        snapshot = directory / snapshot_name
        if not snapshot.exists():
            shutil.copy2(path, snapshot)
        if _sha256(snapshot) != expected:
            raise ModelProvenanceError(f"Source snapshot SHA-256 changed while copying {path}")
        snapshots.append({"source_path": str(path), "snapshot_file": snapshot_name, "sha256": expected})
    return snapshots


def _persist_provenance(directory, metadata, prepared):
    metadata["manifest_version"] = "paired_experiment_v1"
    metadata["source_manifest"] = prepared.manifest
    metadata["source_files"] = _snapshot_source_files(prepared.manifest["source_networks"], directory)
    metadata["target_threshold_m3"] = prepared.manifest.get("flood_threshold_m3")
    metadata["quality_policy"] = {
        "version": sorted({
            (run.get("hydraulic_quality") or {}).get("policy_version", "unknown")
            for network in prepared.manifest["source_networks"] for run in network["runs"]
        }) or ["unknown"],
        "max_continuity_error_pct": prepared.manifest.get("max_continuity_error_pct"),
        "per_run": [
            {"run_id": run["run_id"], "quality": run.get("hydraulic_quality")}
            for network in prepared.manifest["source_networks"] for run in network["runs"]
        ],
    }
    metadata["swmm_version"] = [
        {"run_id": run["run_id"], "version": run.get("swmm_version") or "unknown"}
        for network in prepared.manifest["source_networks"] for run in network["runs"]
    ]
    metadata["split_protocol"] = {
        "outer": list(prepared.manifest["protocols"]),
        "inner": list(metadata.pop("inner_protocols", [])),
        "artifact_training_scope": "all_prepared_rows (not an outer-fold model)",
    }
    normalized = json.loads(json.dumps(metadata))
    metadata.clear()
    metadata.update(normalized)
    documents = {
        "source_manifest.json": prepared.manifest,
        "splits.json": prepared.folds.to_dict(orient="records"),
        "quality_report.json": prepared.quality,
    }
    for filename, document in documents.items():
        _write_json(directory / filename, document)
    metadata["source_manifest_sha256"] = _sha256(directory / "source_manifest.json")
    metadata["split_sha256"] = _sha256(directory / "splits.json")
    metadata["quality_report_sha256"] = _sha256(directory / "quality_report.json")
    _write_json(directory / "experiment.json", dict(metadata))
    filenames = [*documents, "experiment.json", *(source["snapshot_file"] for source in metadata["source_files"])]
    metadata["provenance_files_sha256"] = {name: _sha256(directory / name) for name in filenames}
    _write_json(directory / "metadata.json", metadata)


@dataclass(frozen=True)
class CandidateArtifacts:
    family: str
    classifier_path: Path
    regressor_path: Path
    metadata_path: Path
    metadata: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _library_versions() -> dict:
    return {
        name: _package_version(name)
        for name in ("pandas", "numpy", "scikit-learn", "xgboost", "torch")
    }


def train_candidate(
    prepared: PreparedDataset,
    family_name: str,
    classifier_params: dict,
    regressor_params: dict,
    output_dir: Path | str,
    *,
    random_seed: int | None = None,
    random_seeds: tuple[int, ...] | None = None,
    objective: str | None = None,
    inner_protocols: tuple[str, ...] = (),
    trial_metadata: dict | None = None,
    immutable: bool = False,
) -> CandidateArtifacts:
    """Fit one full-data pair at the repeatable legacy family path.

    ``objective=None`` records configured parameters without claiming model
    selection. Callers with actual tuning provenance may pass its objective
    and trial metadata. The explicit multi-seed API opts into immutable paths.
    """
    family = get_family(family_name)
    feature_contract = validate_feature_contract_manifest(
        prepared.manifest,
        tuple(prepared.X.columns),
    )
    actual_seed = ML_RANDOM_STATE if random_seed is None else int(random_seed)
    if family_name != "mlp" and actual_seed != ML_RANDOM_STATE:
        raise ValueError("Only MLP supports explicit candidate seeds")
    if family_name == "mlp":
        classifier_params = {**classifier_params, "random_state": actual_seed}
        regressor_params = {**regressor_params, "random_state": actual_seed}
    directory = Path(output_dir) / family_name
    if immutable and family_name == "mlp":
        directory = directory / f"seed-{actual_seed}"
    directory.mkdir(parents=True, exist_ok=not immutable)

    X = prepared.X
    y_clf = prepared.y_clf.to_numpy()
    y_reg = prepared.y_reg.to_numpy()

    n_negative, n_positive = int((y_clf == 0).sum()), int((y_clf == 1).sum())
    if n_positive == 0:
        # Deliberate divergence from trainer.train_models: this raises before
        # fitting anything, instead of fitting the classifier first and only
        # raising once the regressor's flooded subset turns out empty. Failing
        # fast avoids burning a full classifier fit before aborting; both
        # paths raise ValueError, so no caller can tell the difference.
        raise ValueError(
            "The dataset has no flooded row (inunda == 1); "
            "the regressor cannot be trained."
        )
    scale_pos_weight = n_negative / n_positive

    classifier = family.build_classifier(classifier_params, scale_pos_weight)
    classifier.fit(X, y_clf)

    flooded = y_clf == 1
    regressor = family.build_regressor(regressor_params)
    regressor.fit(X.iloc[flooded], np.log1p(y_reg[flooded]))

    classifier_path = directory / "classifier.joblib"
    regressor_path = directory / "regressor.joblib"
    joblib.dump(classifier, classifier_path)
    joblib.dump(regressor, regressor_path)

    metadata = {
        "family": family_name,
        "prep_id": prepared.prep_id,
        "classifier_params": classifier_params,
        "regressor_params": regressor_params,
        "preprocessing": {
            "classifier": family.preprocessing_descriptor(classifier_params),
            "regressor": family.preprocessing_descriptor(regressor_params),
        },
        "ordered_features": list(feature_contract.feature_names),
        "feature_contract_id": feature_contract.contract_id,
        "feature_contract_sha256": feature_contract.descriptor_sha256,
        "source_network_ids": list(prepared.manifest["source_network_ids"]),
        "source_networks": prepared.manifest["source_networks"],
        "excluded_run_ids": list(prepared.manifest["excluded_run_ids"]),
        "excluded_runs": prepared.manifest["excluded_runs"],
        "source_snapshot_sha256": prepared.manifest["source_snapshot_sha256"],
        "target_transform": {"regressor": "log1p", "inverse": "expm1"},
        "scale_pos_weight": float(scale_pos_weight),
        "regressor_training_rows": n_positive,
        "random_seed": actual_seed,
        "random_seeds": list(random_seeds or (actual_seed,)),
        "objective": objective,
        "inner_protocols": list(inner_protocols),
        "trial_metadata": trial_metadata or {"selection": "configured_parameters", "trials": []},
        "classifier_sha256": _sha256(classifier_path),
        "regressor_sha256": _sha256(regressor_path),
        "python_version": platform.python_version(),
        "library_versions": _library_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = directory / "metadata.json"
    _persist_provenance(directory, metadata, prepared)

    return CandidateArtifacts(
        family=family_name,
        classifier_path=classifier_path,
        regressor_path=regressor_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def train_seed_candidates(prepared, family_name, classifier_params, regressor_params, output_dir, **kwargs):
    """Persist all five predeclared MLP seeds, without selecting on test results."""
    seeds = MLP_SEEDS if family_name == "mlp" else (ML_RANDOM_STATE,)
    for seed in seeds:
        directory = Path(output_dir) / family_name
        if family_name == "mlp":
            directory = directory / f"seed-{seed}"
        if directory.exists():
            raise FileExistsError(f"Candidate is immutable: {directory}")
    return tuple(train_candidate(prepared, family_name, classifier_params, regressor_params,
                                 output_dir, random_seed=seed, random_seeds=seeds, immutable=True, **kwargs)
                 for seed in seeds)


def load_candidate_pair(classifier_dir, regressor_dir):
    from ..model_provenance import load_verified_model_pair
    return load_verified_model_pair(Path(classifier_dir) / "classifier.joblib",
                                    Path(regressor_dir) / "regressor.joblib")


def train_mlp_ensemble(prepared, classifier_params, regressor_params, output_dir, **kwargs):
    """Fit five immutable seed pairs and persist their frozen mean ensemble."""
    from .models.mlp_family import SeedEnsembleClassifier, SeedEnsembleRegressor

    directory = Path(output_dir) / "mlp" / "ensemble"
    if directory.exists():
        raise FileExistsError(f"Candidate is immutable: {directory}")
    members = train_seed_candidates(prepared, "mlp", classifier_params, regressor_params, output_dir, **kwargs)
    directory.mkdir(parents=True, exist_ok=False)
    classifier_path = directory / "classifier.joblib"
    regressor_path = directory / "regressor.joblib"
    joblib.dump(SeedEnsembleClassifier([joblib.load(c.classifier_path) for c in members]), classifier_path)
    joblib.dump(SeedEnsembleRegressor([joblib.load(c.regressor_path) for c in members]), regressor_path)
    metadata = {key: value for key, value in members[0].metadata.items()
                if key not in {"provenance_files_sha256", "manifest_version", "split_protocol"}}
    metadata.update({
        "random_seed": None,
        "classifier_params": classifier_params,
        "regressor_params": regressor_params,
        "classifier_sha256": _sha256(classifier_path),
        "regressor_sha256": _sha256(regressor_path),
        "inner_protocols": list(kwargs.get("inner_protocols", ())),
        "ensemble": {"probability": "arithmetic_mean", "conditional_volume": "mean_after_expm1_m3",
                     "conditional_volume_min_m3_per_seed": 0.0,
                     "classification_threshold": 0.5, "dry_routed_volume_m3": 0.0},
        "members": [{"random_seed": c.metadata["random_seed"],
                     "classifier_sha256": c.metadata["classifier_sha256"],
                     "regressor_sha256": c.metadata["regressor_sha256"],
                     "provenance_files_sha256": c.metadata["provenance_files_sha256"]}
                    for c in members],
    })
    _persist_provenance(directory, metadata, prepared)
    return CandidateArtifacts("mlp", classifier_path, regressor_path,
                              directory / "metadata.json", metadata), members


def load_candidate(directory: Path | str) -> CandidateArtifacts:
    """Load a candidate only after verifying its paired artifact provenance."""
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"There is no metadata.json in {directory}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelProvenanceError(f"Could not read candidate metadata in {directory}: {exc}") from exc
    classifier_path = directory / "classifier.joblib"
    regressor_path = directory / "regressor.joblib"
    validate_pair_files(metadata, classifier_path, regressor_path)
    try:
        get_family(metadata["family"])
    except (KeyError, ValueError) as exc:
        raise ModelProvenanceError(
            f"Candidate metadata declares an unknown family: {metadata.get('family')!r}"
        ) from exc
    try:
        classifier = joblib.load(classifier_path)
        regressor = joblib.load(regressor_path)
    except Exception as exc:
        raise ModelProvenanceError(f"Could not load candidate model pair in {directory}: {exc}") from exc
    validate_estimators(metadata, classifier, regressor)
    return CandidateArtifacts(
        family=metadata["family"],
        classifier_path=classifier_path,
        regressor_path=regressor_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )
