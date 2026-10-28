"""Shared validation for classifier/regressor model pairs and their provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .contracts import TABULAR_V3_17, validate_feature_contract_manifest


class ModelProvenanceError(RuntimeError):
    """A model artifact does not match its declared training provenance."""


def _is_documented_legacy_v17(record: dict) -> bool:
    return (
        record.get("feature_contract_id") == TABULAR_V3_17.contract_id
        and str(record.get("source", "")).startswith("train_models (legacy")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_metadata(
    record: dict,
    *,
    require_prep_id: bool,
    allow_documented_legacy: bool = True,
) -> tuple[str, str, tuple[str, ...], str | None]:
    """Validate contract and pair identity fields without loading estimators."""
    if not isinstance(record, dict):
        raise ModelProvenanceError("Model metadata must be a JSON object")
    contract_id = record.get("feature_contract_id")
    if not isinstance(contract_id, str) or not contract_id:
        raise ModelProvenanceError("Model metadata is missing feature_contract_id")
    ordered = record.get("ordered_features")
    if not isinstance(ordered, list) or not all(
        isinstance(name, str) for name in ordered
    ):
        raise ModelProvenanceError("Model metadata is missing ordered_features")

    # The legacy trainer wrote complete v17 names and a documented source, but
    # predates descriptor hashes and prep_id. Never extend this exception to v4_21.
    documented_v17 = allow_documented_legacy and _is_documented_legacy_v17(record)
    manifest = {
        "contract_id": contract_id,
        "feature_columns": ordered,
        "feature_contract_sha256": record.get("feature_contract_sha256"),
    }
    if documented_v17 and manifest["feature_contract_sha256"] is None:
        manifest["feature_contract_sha256"] = TABULAR_V3_17.descriptor_sha256
    try:
        contract = validate_feature_contract_manifest(manifest, tuple(ordered))
    except (TypeError, ValueError) as exc:
        raise ModelProvenanceError(f"Invalid feature contract metadata: {exc}") from exc

    family = record.get("family")
    if not isinstance(family, str) or not family:
        raise ModelProvenanceError("Model metadata is missing family")
    known_families = {
        "xgboost",
        "random_forest",
        "linear",
        "svm",
        "mlp",
        "xgboost_legacy",
        "random_forest_legacy",
    }
    if family not in known_families:
        raise ModelProvenanceError(f"Model metadata declares an unknown family: {family!r}")
    prep_id = record.get("prep_id")
    if require_prep_id and (not isinstance(prep_id, str) or not prep_id):
        raise ModelProvenanceError("Model metadata is missing prep_id")
    if prep_id is not None and (not isinstance(prep_id, str) or not prep_id):
        raise ModelProvenanceError("Model metadata has an invalid prep_id")
    return contract.contract_id, family, tuple(ordered), prep_id


def validate_estimators(record: dict, classifier, regressor) -> None:
    """Ensure both estimators accept exactly the declared ordered contract."""
    contract_id, _, ordered, _ = validate_model_metadata(
        record, require_prep_id=not _is_documented_legacy_v17(record)
    )
    for role, estimator in (("classifier", classifier), ("regressor", regressor)):
        count = getattr(estimator, "n_features_in_", None)
        if count != len(ordered):
            raise ModelProvenanceError(
                f"{role} has {count} features but {contract_id} declares {len(ordered)}"
            )
        names = getattr(estimator, "feature_names_in_", None)
        if names is not None and tuple(str(name) for name in names) != ordered:
            raise ModelProvenanceError(
                f"{role} feature_names_in_ do not match ordered_features"
            )


def validate_pair_files(record: dict, classifier_path: Path, regressor_path: Path) -> None:
    """Check artifact presence and both SHA-256 values against one metadata record."""
    validate_model_metadata(record, require_prep_id=not _is_documented_legacy_v17(record))
    validate_experiment_files(record, classifier_path.parent)
    for path, key in (
        (classifier_path, "classifier_sha256"),
        (regressor_path, "regressor_sha256"),
    ):
        if not path.is_file():
            raise ModelProvenanceError(f"Model artifact is missing: {path}")
        expected = record.get(key)
        actual = sha256_file(path)
        if not isinstance(expected, str) or actual != expected:
            raise ModelProvenanceError(
                f"{path.name} SHA-256 does not match model metadata"
            )


def validate_experiment_files(record: dict, directory: Path) -> None:
    """Verify every persisted provenance document and its shared pair identity."""
    if record.get("manifest_version") != "paired_experiment_v1":
        return  # Existing explicit legacy records remain readable.
    hashes = record.get("provenance_files_sha256")
    required = {"experiment.json", "source_manifest.json", "splits.json", "quality_report.json"}
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise ModelProvenanceError("Incomplete experiment provenance_files_sha256")
    for source in record.get("source_files", []):
        if hashes.get(source.get("snapshot_file")) != source.get("sha256"):
            raise ModelProvenanceError("Incomplete source snapshot SHA-256 record")
    for name, expected in hashes.items():
        if Path(name).name != name:
            raise ModelProvenanceError("Invalid provenance filename")
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ModelProvenanceError(f"{name} SHA-256 does not match experiment metadata")
    experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    for field, value in experiment.items():
        if record.get(field) != value:
            raise ModelProvenanceError(f"Experiment manifest differs in {field}")


def validate_training_network_record(record: dict, directory: Path) -> None:
    """Bind the recorded input-network hash to the sidecar used by inference."""
    expected = record.get("training_inp_md5")
    if expected is None:
        if not _is_documented_legacy_v17(record):
            raise ModelProvenanceError(
                "promotion.json is missing training_inp_md5; the training network "
                "cannot be verified"
            )
        return
    if not isinstance(expected, str) or not expected:
        raise ModelProvenanceError("promotion.json has an invalid training_inp_md5")
    path = directory / "training_inp_hash.txt"
    if not path.is_file():
        raise ModelProvenanceError(f"Network provenance file is missing: {path}")
    actual = path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise ModelProvenanceError("training_inp_hash.txt does not match promotion.json")


def _record_for(directory: Path) -> tuple[dict | None, Path | None]:
    for filename in ("promotion.json", "metadata.json"):
        path = directory / filename
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8")), path
            except (OSError, json.JSONDecodeError) as exc:
                raise ModelProvenanceError(f"Could not read {path}: {exc}") from exc
    return None, None


def _validate_single_artifact(record: dict, path: Path, role: str) -> None:
    validate_model_metadata(record, require_prep_id=not _is_documented_legacy_v17(record))
    if not path.is_file():
        raise ModelProvenanceError(f"Model artifact is missing: {path}")
    expected = record.get(f"{role}_sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ModelProvenanceError(f"{path.name} SHA-256 does not match model metadata")


def load_verified_model_pair(classifier_path: Path | str, regressor_path: Path | str):
    """Load a pair from one candidate/promotion record, or a matched pair of records.

    Inference requires provenance. A feature count alone cannot establish which
    contract or preparation produced a model. One-sided or inconsistent
    provenance is always rejected; it is never treated as a legacy pair.
    """
    import joblib

    classifier_path, regressor_path = Path(classifier_path), Path(regressor_path)
    if classifier_path.parent.resolve() == regressor_path.parent.resolve():
        record, record_path = _record_for(classifier_path.parent)
        if record is None:
            raise ModelProvenanceError(
                f"No candidate metadata or promotion.json in {classifier_path.parent}; "
                "the model pair cannot be verified for inference."
            )
        else:
            validate_pair_files(record, classifier_path, regressor_path)
            if record_path is not None and record_path.name == "promotion.json":
                validate_training_network_record(record, classifier_path.parent)
    else:
        classifier_record, classifier_record_path = _record_for(classifier_path.parent)
        regressor_record, regressor_record_path = _record_for(regressor_path.parent)
        if classifier_record is None and regressor_record is None:
            raise ModelProvenanceError(
                "Classifier and regressor are in different directories and neither "
                "has provenance metadata; the pair cannot be verified."
            )
        elif classifier_record is None or regressor_record is None:
            raise ModelProvenanceError("Classifier and regressor do not both have provenance metadata")
        else:
            classifier_identity = validate_model_metadata(
                classifier_record, require_prep_id=not _is_documented_legacy_v17(classifier_record)
            )
            regressor_identity = validate_model_metadata(
                regressor_record, require_prep_id=not _is_documented_legacy_v17(regressor_record)
            )
            if classifier_identity != regressor_identity:
                raise ModelProvenanceError(
                    "Classifier and regressor provenance differ in contract, family, "
                    "ordered features, or prep_id"
                )
            if any(
                classifier_record.get(field) != regressor_record.get(field)
                for field in (
                    "classifier_sha256",
                    "regressor_sha256",
                    "training_inp_md5",
                )
            ):
                raise ModelProvenanceError(
                    "Classifier and regressor metadata do not describe the same artifact pair"
                )
            for field in ("source_snapshot_sha256", "source_networks", "random_seed",
                          "random_seeds", "provenance_files_sha256"):
                if classifier_record.get(field) != regressor_record.get(field):
                    raise ModelProvenanceError(f"Classifier and regressor provenance differ in {field}")
            validate_experiment_files(classifier_record, classifier_path.parent)
            validate_experiment_files(regressor_record, regressor_path.parent)
            _validate_single_artifact(classifier_record, classifier_path, "classifier")
            _validate_single_artifact(regressor_record, regressor_path, "regressor")
            if str(classifier_record_path).endswith("promotion.json"):
                validate_training_network_record(
                    classifier_record, classifier_path.parent
                )
            if str(regressor_record_path).endswith("promotion.json"):
                validate_training_network_record(
                    regressor_record, regressor_path.parent
                )
            record = classifier_record
    try:
        classifier = joblib.load(classifier_path)
        regressor = joblib.load(regressor_path)
    except Exception as exc:
        if record is None:
            raise
        raise ModelProvenanceError(f"Could not load classifier/regressor pair: {exc}") from exc
    if record is not None:
        validate_estimators(record, classifier, regressor)
    return classifier, regressor, record
