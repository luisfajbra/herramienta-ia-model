"""Promoción del candidato ganador a las rutas que lee el resto del pipeline.

outputs/models/{classifier,regressor}.joblib + training_inp_hash.txt es el
formato que consumen --predict, --simulate, los mapas, --evaluate-hydrographs,
ml/predict.py, ml/scenario_predict.py, ml/feature_analysis.py y
ml/feature_importance.py. Mantenerlo idéntico es lo que permite retirar el
legacy sin tocar ninguno de ellos.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .train import _sha256, load_candidate


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelProvenanceError(RuntimeError):
    """Los modelos de ``models_dir`` no coinciden con lo declarado en promotion.json."""


_PROMOTION_ARTIFACTS = (
    ("classifier.joblib", "classifier_sha256"),
    ("regressor.joblib", "regressor_sha256"),
)


def verify_promotion(models_dir: Path | str) -> dict:
    """Compara los SHA-256 de los modelos con los de ``promotion.json``.

    ``status``: ``verified`` (coinciden), ``mismatch`` (alguno difiere o falta) o
    ``no_record`` (no hay ``promotion.json``; no se puede afirmar ni negar nada).
    """
    target = Path(models_dir)
    record_path = target / "promotion.json"
    if not record_path.exists():
        return {"status": "no_record", "ok": None, "discrepancies": []}
    record = json.loads(record_path.read_text(encoding="utf-8"))
    discrepancies = []
    for artifact, field in _PROMOTION_ARTIFACTS:
        path = target / artifact
        actual = _sha256(path) if path.exists() else None
        expected = record.get(field)
        if actual != expected:
            discrepancies.append({"artifact": artifact, "expected": expected, "actual": actual})
    status = "mismatch" if discrepancies else "verified"
    return {"status": status, "ok": not discrepancies, "discrepancies": discrepancies}


def assert_promotion_consistent(models_dir: Path | str) -> dict:
    """Falla si los modelos no son los declarados; avisa si no hay registro."""
    result = verify_promotion(models_dir)
    if result["status"] == "mismatch":
        detail = "; ".join(
            f"{d['artifact']}: expected {d['expected']}, actual {d['actual']}"
            for d in result["discrepancies"]
        )
        raise ModelProvenanceError(
            f"The models in {models_dir} do not match promotion.json ({detail}). "
            "The results cannot be attributed to the promoted candidate. Promote "
            "again or regenerate the models before predicting."
        )
    if result["status"] == "no_record":
        warnings.warn(
            f"There is no promotion.json in {models_dir}: the model provenance cannot be verified.",
            UserWarning,
            stacklevel=2,
        )
    return result


def _artifact_files_sha256(models_dir: Path) -> dict:
    """SHA-256 de los archivos que definen el modelo de un directorio sin promotion.json.

    Cubre los dos formatos: el de ``ml/train.py`` (``manifest.json`` + los
    ``<task>_<modelo>.joblib`` que su ``latest`` referencia) y el clásico
    (``classifier.joblib`` / ``regressor.joblib``). Sin ninguno de los dos,
    todos los ``*.joblib`` del directorio. No es recursivo.
    """
    if not models_dir.is_dir():
        return {}
    names: set[str] = set()
    manifest_path = models_dir / "manifest.json"
    if manifest_path.exists():
        names.add("manifest.json")
        latest = json.loads(manifest_path.read_text(encoding="utf-8")).get("latest", {})
        for task, model_name in latest.items():
            names.add(f"{task}_{str(model_name).replace(' ', '_')}.joblib")
    names.update(artifact for artifact, _ in _PROMOTION_ARTIFACTS)
    names = {name for name in names if (models_dir / name).is_file()}
    if not any(name.endswith(".joblib") for name in names):
        names.update(path.name for path in models_dir.glob("*.joblib"))
    return {name: _sha256(models_dir / name) for name in sorted(names)}


def model_provenance(models_dir: Path | str) -> dict:
    """Resumen de procedencia del modelo activo, para estamparlo junto a las figuras.

    Sin ``promotion.json`` la familia es ``unknown`` y ``artifact_sha256``
    registra los SHA-256 de los archivos del modelo realmente presentes.
    """
    record_path = Path(models_dir) / "promotion.json"
    if not record_path.exists():
        return {
            "family": "unknown", "feature_contract_id": None,
            "classifier_sha256": None, "regressor_sha256": None, "prep_id": None,
            "artifact_sha256": _artifact_files_sha256(Path(models_dir)),
        }
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return {
        "family": record.get("family"),
        "feature_contract_id": record.get("feature_contract_id"),
        "classifier_sha256": record.get("classifier_sha256"),
        "regressor_sha256": record.get("regressor_sha256"),
        "prep_id": record.get("prep_id"),
    }


def promote_candidate(
    candidate_dir: Path | str,
    models_dir: Path | str,
    inp_path: Path | str,
) -> dict:
    """Copia los artefactos del candidato a ``models_dir`` en el formato clásico."""
    candidate = load_candidate(candidate_dir)
    target = Path(models_dir)
    target.mkdir(parents=True, exist_ok=True)

    existing_clf_path = target / "classifier.joblib"
    existing_reg_path = target / "regressor.joblib"
    if existing_clf_path.exists() and existing_reg_path.exists():
        try:
            existing_clf = joblib.load(existing_clf_path)
            existing_reg = joblib.load(existing_reg_path)
        except Exception as load_error:
            raise RuntimeError(
                f"The existing model in {target} could not be read to verify its "
                f"feature contract ({load_error}). Manually check the contents "
                f"of {target} before continuing -- promote_candidate() does not blindly "
                "overwrite a directory it cannot inspect."
            ) from load_error
        candidate_n_features = len(candidate.metadata.get("ordered_features") or [])
        if candidate_n_features and (
            existing_clf.n_features_in_ != candidate_n_features
            or existing_reg.n_features_in_ != candidate_n_features
        ):
            raise RuntimeError(
                f"The active model in {target} has "
                f"{existing_clf.n_features_in_}/{existing_reg.n_features_in_} features "
                f"(clf/reg), but the candidate in {candidate_dir} has "
                f"{candidate_n_features} ({candidate.metadata.get('feature_contract_id', 'unknown contract')}). "
                "promote_candidate() cannot silently overwrite an active model "
                "of another contract. If you really want to replace it, move or "
                "delete classifier.joblib/regressor.joblib in "
                f"{target} manually first (you can recover them with "
                "`git checkout -- outputs/models/` if you make a mistake)."
            )

    shutil.copy2(candidate.classifier_path, target / "classifier.joblib")
    shutil.copy2(candidate.regressor_path, target / "regressor.joblib")
    (target / "training_inp_hash.txt").write_text(_md5(Path(inp_path)))

    record = {
        "family": candidate.family,
        "prep_id": candidate.metadata["prep_id"],
        "feature_contract_id": candidate.metadata.get("feature_contract_id", "tabular_v3_17"),
        "ordered_features": candidate.metadata.get("ordered_features"),
        "classifier_sha256": candidate.metadata["classifier_sha256"],
        "regressor_sha256": candidate.metadata["regressor_sha256"],
        "source_dir": str(Path(candidate_dir).resolve()),
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target / "promotion.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return record
