"""Estructuras de datos compartidas por las tres etapas del banco.

Este módulo no ejecuta lógica de negocio: sólo define la forma de los datos
que viajan entre etapas y cómo se persisten.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..contracts import FEATURE_COLUMNS_V17

KEY_COLUMNS = (
    "run_id",
    "network_id",
    "scenario_id",
    "scenario_key",
    "scenario_kind",
    "node_id",
    "factor_mult",
    "shape_id",
)

FOLD_COLUMNS = ("protocol", "fold_id", "sample_idx", "split")

OOF_COLUMNS = (
    "sample_idx",
    "protocol",
    "fold_id",
    "y_pred_clf",
    "y_prob_clf",
    "y_pred_reg",
)

PROTOCOLS = ("LOSO", "GroupKFold5")

_COMPONENT_FILES = {
    "keys": "keys.parquet",
    "features": "features.parquet",
    "targets": "targets.parquet",
    "folds": "folds.parquet",
    "manifest": "manifest.json",
    "quality": "quality_report.json",
}


@dataclass(frozen=True)
class PreparedDataset:
    """Salida de la etapa 1: datos listos para entrenar, sin transformar.

    Deliberadamente NO contiene nada imputado ni escalado. Esas operaciones
    pertenecen al Pipeline de cada familia, que se ajusta dentro del fold.
    """

    prep_id: str
    keys: pd.DataFrame
    X: pd.DataFrame
    y_clf: pd.Series
    y_reg: pd.Series
    folds: pd.DataFrame
    manifest: dict
    quality: dict

    def __post_init__(self) -> None:
        if tuple(self.X.columns) != FEATURE_COLUMNS_V17:
            raise ValueError(
                "Las features no están en el orden del contrato "
                f"{FEATURE_COLUMNS_V17}; recibido {tuple(self.X.columns)}"
            )
        if tuple(self.keys.columns) != KEY_COLUMNS:
            raise ValueError(
                f"keys debe tener las columnas {KEY_COLUMNS}; "
                f"recibido {tuple(self.keys.columns)}"
            )
        if tuple(self.folds.columns) != FOLD_COLUMNS:
            raise ValueError(
                f"folds debe tener las columnas {FOLD_COLUMNS}; "
                f"recibido {tuple(self.folds.columns)}"
            )

    def save(self, output_dir: Path | str) -> Path:
        """Escribe el dataset en ``output_dir/<prep_id>/`` y devuelve esa ruta."""
        directory = Path(output_dir) / self.prep_id
        directory.mkdir(parents=True, exist_ok=True)
        self.keys.to_parquet(directory / _COMPONENT_FILES["keys"], index=False)
        self.X.to_parquet(directory / _COMPONENT_FILES["features"], index=False)
        pd.DataFrame({self.y_clf.name: self.y_clf, self.y_reg.name: self.y_reg}).to_parquet(
            directory / _COMPONENT_FILES["targets"], index=False
        )
        self.folds.to_parquet(directory / _COMPONENT_FILES["folds"], index=False)
        (directory / _COMPONENT_FILES["manifest"]).write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (directory / _COMPONENT_FILES["quality"]).write_text(
            json.dumps(self.quality, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> PreparedDataset:
        """Lee un dataset escrito por :meth:`save`."""
        directory = Path(directory)
        for filename in _COMPONENT_FILES.values():
            if not (directory / filename).exists():
                raise FileNotFoundError(
                    f"Falta {filename} en {directory}; el PreparedDataset está incompleto"
                )
        targets = pd.read_parquet(directory / _COMPONENT_FILES["targets"])
        manifest = json.loads(
            (directory / _COMPONENT_FILES["manifest"]).read_text(encoding="utf-8")
        )
        return cls(
            prep_id=manifest["prep_id"],
            keys=pd.read_parquet(directory / _COMPONENT_FILES["keys"]),
            X=pd.read_parquet(directory / _COMPONENT_FILES["features"]),
            y_clf=targets["inunda"],
            y_reg=targets["vol_inundacion_m3"],
            folds=pd.read_parquet(directory / _COMPONENT_FILES["folds"]),
            manifest=manifest,
            quality=json.loads(
                (directory / _COMPONENT_FILES["quality"]).read_text(encoding="utf-8")
            ),
        )
