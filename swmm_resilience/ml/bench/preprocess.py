"""Etapa 1 del banco: SQL -> PreparedDataset.

NO-OBJETIVOS de esta etapa, y son deliberados: no imputa, no escala, no
aplica PCA, no elimina filas con nulos permitidos por el contrato, y no
transforma el target. Todo eso pertenece al Pipeline de cada familia, que se
ajusta dentro de cada fold. Si algo de eso aparece en este archivo, es un bug
— y tests/ml/bench/test_stage1_imports.py lo detecta.

Tampoco re-deriva ``inunda``: ese target ya viene persistido en la vista
training_samples_v17, y recalcularlo contra el umbral podría discrepar de lo
que la base afirma. La coherencia se verifica y se reporta (quality.py), no
se corrige en silencio.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ...database.training_queries import load_training_frame
from ..contracts import FEATURE_COLUMNS_V17, TABULAR_V3_17
from .folds import build_all_folds
from .quality import build_quality_report
from .schemas import KEY_COLUMNS, PROTOCOLS, PreparedDataset

SORT_KEY = ("run_id", "node_id")


def _library_versions() -> dict:
    return {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": _package_version("scikit-learn"),
    }


def compute_prep_id(descriptor: dict) -> str:
    """Hash reproducible de la definición del dataset.

    Se calcula sólo sobre la *definición* (qué runs, qué columnas, qué umbral,
    qué protocolos), no sobre los valores, para que dos corridas sobre los
    mismos datos den el mismo id.
    """
    payload = json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def prepare_dataset(
    db_path: Path | str,
    *,
    protocols: Sequence[str] = PROTOCOLS,
    flood_threshold_m3: float,
    output_dir: Path | str,
) -> PreparedDataset:
    """Lee el frame de SQL y materializa un PreparedDataset en ``output_dir``."""
    frame = load_training_frame(db_path)
    frame = frame.sort_values(list(SORT_KEY)).reset_index(drop=True)

    keys = frame.loc[:, list(KEY_COLUMNS)].copy()
    X = TABULAR_V3_17.validate_frame(frame.loc[:, list(FEATURE_COLUMNS_V17)])
    y_clf = frame["inunda"].astype(int).rename("inunda")
    y_reg = frame["vol_inundacion_m3"].astype(float).rename("vol_inundacion_m3")

    quality = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3)
    folds = build_all_folds(keys["factor_mult"], tuple(protocols))

    descriptor = {
        "contract_id": TABULAR_V3_17.contract_id,
        "feature_contract_sha256": TABULAR_V3_17.descriptor_sha256,
        "run_ids": sorted(int(value) for value in keys["run_id"].unique()),
        "feature_columns": list(FEATURE_COLUMNS_V17),
        "flood_threshold_m3": float(flood_threshold_m3),
        "sort_key": list(SORT_KEY),
        "protocols": list(protocols),
    }
    prep_id = compute_prep_id(descriptor)

    manifest = dict(
        descriptor,
        prep_id=prep_id,
        db_path=str(db_path),
        n_rows=int(len(frame)),
        python_version=platform.python_version(),
        library_versions=_library_versions(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    prepared = PreparedDataset(
        prep_id=prep_id,
        keys=keys,
        X=X,
        y_clf=y_clf,
        y_reg=y_reg,
        folds=folds,
        manifest=manifest,
        quality=quality,
    )
    prepared.save(output_dir)
    return prepared


def resolve_prepared(
    output_dir: Path | str, prep_id: str | None = None
) -> PreparedDataset:
    """Carga un PreparedDataset por id, o el más reciente si no se da uno."""
    root = Path(output_dir)
    if prep_id is not None:
        directory = root / prep_id
        if not directory.exists():
            raise FileNotFoundError(
                f"No existe el PreparedDataset {prep_id} en {root}. "
                "Corre `python main.py --bench-prepare` primero."
            )
        return PreparedDataset.load(directory)

    candidates = [path for path in root.glob("*") if (path / "manifest.json").exists()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(
            f"No hay ningún PreparedDataset en {root}. "
            "Corre `python main.py --bench-prepare` primero."
        )
    newest = max(candidates, key=lambda path: (path / "manifest.json").stat().st_mtime)
    return PreparedDataset.load(newest)
