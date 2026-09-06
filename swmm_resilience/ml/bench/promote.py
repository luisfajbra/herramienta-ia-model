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
from datetime import datetime, timezone
from pathlib import Path

from .train import load_candidate


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote_candidate(
    candidate_dir: Path | str,
    models_dir: Path | str,
    inp_path: Path | str,
) -> dict:
    """Copia los artefactos del candidato a ``models_dir`` en el formato clásico."""
    candidate = load_candidate(candidate_dir)
    target = Path(models_dir)
    target.mkdir(parents=True, exist_ok=True)

    shutil.copy2(candidate.classifier_path, target / "classifier.joblib")
    shutil.copy2(candidate.regressor_path, target / "regressor.joblib")
    (target / "training_inp_hash.txt").write_text(_md5(Path(inp_path)))

    record = {
        "family": candidate.family,
        "prep_id": candidate.metadata["prep_id"],
        "classifier_sha256": candidate.metadata["classifier_sha256"],
        "regressor_sha256": candidate.metadata["regressor_sha256"],
        "source_dir": str(Path(candidate_dir).resolve()),
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target / "promotion.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return record
