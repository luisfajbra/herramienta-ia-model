"""Etapa 2 del banco: ajuste sobre todo el dataset y escritura de artefactos.

Intencionalmente idéntico a lo que hace ml/trainer.py: clasificador sobre
todas las filas, regresor sobre inunda == 1 con objetivo log1p. Esa identidad
es la condición para que el test de paridad pueda pasar.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as _package_version
from pathlib import Path

import joblib
import numpy as np

from ...config import ML_RANDOM_STATE
from ..contracts import FEATURE_COLUMNS_V17, TABULAR_V3_17
from .registry import get_family
from .schemas import PreparedDataset


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
        for name in ("pandas", "numpy", "scikit-learn", "xgboost")
    }


def train_candidate(
    prepared: PreparedDataset,
    family_name: str,
    classifier_params: dict,
    regressor_params: dict,
    output_dir: Path | str,
) -> CandidateArtifacts:
    """Ajusta la cascada sobre todo el dataset y persiste los artefactos."""
    family = get_family(family_name)
    directory = Path(output_dir) / family_name
    directory.mkdir(parents=True, exist_ok=True)

    X = prepared.X
    y_clf = prepared.y_clf.to_numpy()
    y_reg = prepared.y_reg.to_numpy()

    n_negative, n_positive = int((y_clf == 0).sum()), int((y_clf == 1).sum())
    if n_positive == 0:
        raise ValueError(
            "El dataset no tiene ninguna fila inundada (inunda == 1); "
            "el regresor no se puede entrenar."
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
        "preprocessing": family.preprocessing_descriptor(classifier_params),
        "ordered_features": list(FEATURE_COLUMNS_V17),
        "feature_contract_id": TABULAR_V3_17.contract_id,
        "feature_contract_sha256": TABULAR_V3_17.descriptor_sha256,
        "target_transform": {"regressor": "log1p", "inverse": "expm1"},
        "scale_pos_weight": float(scale_pos_weight),
        "regressor_training_rows": n_positive,
        "random_seed": ML_RANDOM_STATE,
        "classifier_sha256": _sha256(classifier_path),
        "regressor_sha256": _sha256(regressor_path),
        "python_version": platform.python_version(),
        "library_versions": _library_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = directory / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return CandidateArtifacts(
        family=family_name,
        classifier_path=classifier_path,
        regressor_path=regressor_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def load_candidate(directory: Path | str) -> CandidateArtifacts:
    """Lee los artefactos de un candidato ya entrenado."""
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No hay metadata.json en {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return CandidateArtifacts(
        family=metadata["family"],
        classifier_path=directory / "classifier.joblib",
        regressor_path=directory / "regressor.joblib",
        metadata_path=metadata_path,
        metadata=metadata,
    )
