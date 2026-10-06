"""Estructuras de datos compartidas por las tres etapas del banco.

Este módulo no ejecuta lógica de negocio: sólo define la forma de los datos
que viajan entre etapas y cómo se persisten.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import pandas as pd

from ..contracts import FEATURE_COLUMNS_V17, TARGET_DEFINITIONS_V17

_TARGET_NAME_BY_TASK = {item.task: item.name for item in TARGET_DEFINITIONS_V17}
_CLF_TARGET_NAME = _TARGET_NAME_BY_TASK["classification"]
_REG_TARGET_NAME = _TARGET_NAME_BY_TASK["regression"]

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

FOLD_COLUMNS = (
    "protocol",
    "fold_id",
    "sample_idx",
    "split",
    "group_column",
    "held_out_groups",
)

OOF_COLUMNS = (
    "sample_idx",
    "protocol",
    "fold_id",
    "y_pred_clf",
    "y_prob_clf",
    "y_pred_reg",
)

PROTOCOLS = ("LOSO", "GroupKFold5", "LeaveOneShapeOut")
TRAINING_FRAME_FINGERPRINT_VERSION = "training_frame_canonical_v1"

PROTOCOL_GROUP_COLUMNS = {
    "LOSO": "factor_mult",
    "GroupKFold5": "factor_mult",
    "LeaveOneShapeOut": "shape_id",
}

PREP_ID_DESCRIPTOR_FIELDS = (
    "contract_id",
    "feature_contract_sha256",
    "run_ids",
    "n_rows",
    "feature_columns",
    "flood_threshold_m3",
    "sort_key",
    "protocols",
    "protocol_definitions",
    "source_capture_method",
    "training_frame_fingerprint_version",
    "training_frame_sha256",
)

_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_REGENERATE_MESSAGE = (
    "Corre `python main.py --bench-prepare` para regenerarlo."
)


def compute_prep_id(descriptor: dict) -> str:
    """Calcula la identidad determinista de un descriptor de preparación."""
    payload = json.dumps(
        descriptor,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _prep_descriptor_from_manifest(manifest: dict) -> dict:
    missing = [field for field in PREP_ID_DESCRIPTOR_FIELDS if field not in manifest]
    if missing:
        raise ValueError(
            "El descriptor del PreparedDataset está incompleto; faltan "
            f"{', '.join(missing)}. {_REGENERATE_MESSAGE}"
        )
    source_capture_method = manifest["source_capture_method"]
    if source_capture_method != "sqlite_backup_readonly_v1":
        raise ValueError(
            "source_capture_method no es compatible con el protocolo actual. "
            f"{_REGENERATE_MESSAGE}"
        )
    source_hash = manifest["training_frame_sha256"]
    if not isinstance(source_hash, str) or _SHA256_PATTERN.fullmatch(source_hash) is None:
        raise ValueError(
            "training_frame_sha256 debe ser un SHA-256 hexadecimal de 64 caracteres. "
            f"{_REGENERATE_MESSAGE}"
        )
    if manifest["training_frame_fingerprint_version"] != TRAINING_FRAME_FINGERPRINT_VERSION:
        raise ValueError(
            "training_frame_fingerprint_version no es compatible con el protocolo actual. "
            f"{_REGENERATE_MESSAGE}"
        )
    return {field: manifest[field] for field in PREP_ID_DESCRIPTOR_FIELDS}


def _validate_manifest_identity(manifest: dict) -> None:
    snapshot_hash = manifest.get("source_snapshot_sha256")
    if not isinstance(snapshot_hash, str) or _SHA256_PATTERN.fullmatch(snapshot_hash) is None:
        raise ValueError(
            "source_snapshot_sha256 debe ser un SHA-256 hexadecimal de 64 caracteres. "
            f"{_REGENERATE_MESSAGE}"
        )
    descriptor = _prep_descriptor_from_manifest(manifest)
    expected_prep_id = compute_prep_id(descriptor)
    if manifest.get("prep_id") != expected_prep_id:
        raise ValueError(
            "El prep_id persistido no coincide con el descriptor del "
            f"PreparedDataset. {_REGENERATE_MESSAGE}"
        )


def group_column_for(protocol: str) -> str:
    try:
        return PROTOCOL_GROUP_COLUMNS[protocol]
    except KeyError as exc:
        raise ValueError(
            f"Protocolo desconocido: {protocol!r}. Opciones: {PROTOCOLS}"
        ) from exc


def _validate_manifest_protocol_definitions(manifest: dict) -> None:
    protocols = manifest.get("protocols")
    definitions = manifest.get("protocol_definitions")
    try:
        if not isinstance(protocols, list) or not all(
            isinstance(protocol, str) for protocol in protocols
        ):
            raise ValueError
        expected = {
            protocol: {"group_column": group_column_for(protocol)}
            for protocol in protocols
        }
    except ValueError as exc:
        raise ValueError(
            "La semántica de protocolos del descriptor del PreparedDataset es "
            f"inválida. {_REGENERATE_MESSAGE}"
        ) from exc
    if definitions != expected:
        raise ValueError(
            "La semántica de protocolos del descriptor del PreparedDataset falta "
            f"o no coincide con el catálogo actual. {_REGENERATE_MESSAGE}"
        )


def _validate_fold_provenance(
    keys: pd.DataFrame, folds: pd.DataFrame, manifest: dict
) -> None:
    manifest_protocols = manifest.get("protocols")
    if (
        not isinstance(manifest_protocols, list)
        or not all(isinstance(protocol, str) for protocol in manifest_protocols)
        or len(manifest_protocols) != len(set(manifest_protocols))
    ):
        raise ValueError(
            "Los protocolos del manifiesto son inválidos. "
            f"{_REGENERATE_MESSAGE}"
        )

    required_values = (
        "protocol",
        "fold_id",
        "sample_idx",
        "split",
        "group_column",
        "held_out_groups",
    )
    if folds.loc[:, list(required_values)].isna().any().any():
        raise ValueError(
            "La procedencia de folds contiene valores nulos. "
            f"{_REGENERATE_MESSAGE}"
        )

    fold_protocols = set(folds["protocol"])
    if fold_protocols != set(manifest_protocols):
        raise ValueError(
            "Los protocolos de los folds no coinciden con los protocolos del "
            f"manifiesto. {_REGENERATE_MESSAGE}"
        )

    invalid_splits = set(folds["split"]) - {"train", "test"}
    if invalid_splits:
        raise ValueError(
            f"La procedencia de folds contiene split inválido. {_REGENERATE_MESSAGE}"
        )

    fold_ids: list[int] = []
    for fold_id in folds["fold_id"]:
        if isinstance(fold_id, bool) or not isinstance(fold_id, Integral):
            raise ValueError(
                f"folds.fold_id contiene valores inválidos. {_REGENERATE_MESSAGE}"
            )
        normalized_fold_id = int(fold_id)
        if normalized_fold_id < 0:
            raise ValueError(
                f"folds.fold_id no puede ser negativo. {_REGENERATE_MESSAGE}"
            )
        fold_ids.append(normalized_fold_id)

    sample_indices: list[int] = []
    for sample_idx in folds["sample_idx"]:
        if isinstance(sample_idx, bool) or not isinstance(sample_idx, Integral):
            raise ValueError(
                f"folds.sample_idx contiene valores inválidos. {_REGENERATE_MESSAGE}"
            )
        positional_idx = int(sample_idx)
        if positional_idx < 0 or positional_idx >= len(keys):
            raise ValueError(
                f"folds.sample_idx está fuera de rango. {_REGENERATE_MESSAGE}"
            )
        sample_indices.append(positional_idx)

    folds_with_positions = folds.assign(
        _fold_id=fold_ids, _sample_position=sample_indices
    )
    all_positions = set(range(len(keys)))
    test_positions_by_protocol: dict[str, list[int]] = {
        protocol: [] for protocol in manifest_protocols
    }
    fold_ids_by_protocol: dict[str, set[int]] = {
        protocol: set() for protocol in manifest_protocols
    }
    held_group_counts_by_protocol: dict[str, list[int]] = {
        protocol: [] for protocol in manifest_protocols
    }
    for (protocol, _fold_id), fold_rows in folds_with_positions.groupby(
        ["protocol", "_fold_id"], sort=False
    ):
        try:
            expected_group_column = group_column_for(protocol)
        except ValueError as exc:
            raise ValueError(
                f"Los folds contienen un protocolo inválido. {_REGENERATE_MESSAGE}"
            ) from exc
        if not fold_rows["group_column"].eq(expected_group_column).all():
            raise ValueError(
                f"La procedencia de folds para {protocol!r} debe declarar "
                f"group_column={expected_group_column!r}. {_REGENERATE_MESSAGE}"
            )
        fold_positions = fold_rows["_sample_position"].tolist()
        if (
            len(fold_positions) != len(keys)
            or len(set(fold_positions)) != len(fold_positions)
            or set(fold_positions) != all_positions
        ):
            raise ValueError(
                "Cada fold debe ser una partición completa y sin duplicados de "
                f"todas las muestras. {_REGENERATE_MESSAGE}"
            )
        test_positions = fold_rows.loc[
            fold_rows["split"] == "test", "_sample_position"
        ].tolist()
        if not test_positions:
            raise ValueError(
                "Cada fold debe contener al menos una fila test. "
                f"{_REGENERATE_MESSAGE}"
            )
        all_group_values = keys[expected_group_column]
        if all_group_values.isna().any():
            raise ValueError(
                "Los grupos usados por la procedencia contienen nulos. "
                f"{_REGENERATE_MESSAGE}"
            )
        test_group_values = keys.iloc[test_positions][expected_group_column]
        expected_groups = sorted({str(value) for value in test_group_values})
        if not expected_groups:
            raise ValueError(
                "held_out_groups no puede representar una lista vacía. "
                f"{_REGENERATE_MESSAGE}"
            )
        expected_encoded_groups = json.dumps(
            expected_groups, ensure_ascii=False
        )
        if not fold_rows["held_out_groups"].eq(expected_encoded_groups).all():
            raise ValueError(
                "held_out_groups no coincide con los grupos reconstruidos desde "
                f"las filas test. {_REGENERATE_MESSAGE}"
            )
        train_positions = fold_rows.loc[
            fold_rows["split"] == "train", "_sample_position"
        ].tolist()
        train_groups = {
            str(value)
            for value in keys.iloc[train_positions][expected_group_column]
        }
        if train_groups.intersection(expected_groups):
            raise ValueError(
                "Los grupos de test no pueden aparecer en train dentro del mismo "
                f"fold. {_REGENERATE_MESSAGE}"
            )
        expected_test_positions = {
            position
            for position, value in enumerate(all_group_values)
            if str(value) in expected_groups
        }
        if set(test_positions) != expected_test_positions:
            raise ValueError(
                "Todas las filas de cada grupo retenido deben pertenecer a test. "
                f"{_REGENERATE_MESSAGE}"
            )
        test_positions_by_protocol[protocol].extend(test_positions)
        fold_ids_by_protocol[protocol].add(_fold_id)
        held_group_counts_by_protocol[protocol].append(len(expected_groups))

    for protocol in manifest_protocols:
        tested_positions = test_positions_by_protocol[protocol]
        if (
            len(tested_positions) != len(keys)
            or len(set(tested_positions)) != len(tested_positions)
            or set(tested_positions) != all_positions
        ):
            raise ValueError(
                "Cada muestra debe aparecer en test exactamente una vez por "
                f"protocolo. {_REGENERATE_MESSAGE}"
            )
        group_column = group_column_for(protocol)
        distinct_groups = int(keys[group_column].nunique())
        if protocol in ("LOSO", "LeaveOneShapeOut"):
            if any(
                group_count != 1
                for group_count in held_group_counts_by_protocol[protocol]
            ):
                raise ValueError(
                    f"{protocol} debe retener exactamente un grupo por fold. "
                    f"{_REGENERATE_MESSAGE}"
                )
            if len(fold_ids_by_protocol[protocol]) != distinct_groups:
                raise ValueError(
                    f"{protocol} debe tener un fold por cada grupo distinto. "
                    f"{_REGENERATE_MESSAGE}"
                )
        elif protocol == "GroupKFold5" and fold_ids_by_protocol[protocol] != set(
            range(5)
        ):
            raise ValueError(
                "GroupKFold5 debe tener exactamente los fold_id 0..4. "
                f"{_REGENERATE_MESSAGE}"
            )

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
        expected_feature_columns = tuple(
            self.manifest.get("feature_columns", FEATURE_COLUMNS_V17)
        )
        if tuple(self.X.columns) != expected_feature_columns:
            raise ValueError(
                "Las features no están en el orden del contrato "
                f"{expected_feature_columns}; recibido {tuple(self.X.columns)}"
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
        if self.y_clf.name != _CLF_TARGET_NAME:
            raise ValueError(
                f"y_clf debe llamarse {_CLF_TARGET_NAME!r}; "
                f"recibido {self.y_clf.name!r}"
            )
        if self.y_reg.name != _REG_TARGET_NAME:
            raise ValueError(
                f"y_reg debe llamarse {_REG_TARGET_NAME!r}; "
                f"recibido {self.y_reg.name!r}"
            )
        n = len(self.keys)
        for label, component in (
            ("keys", self.keys),
            ("X", self.X),
            ("y_clf", self.y_clf),
            ("y_reg", self.y_reg),
        ):
            index = component.index
            if not (
                isinstance(index, pd.RangeIndex)
                and index.start == 0
                and index.step == 1
                and len(index) == n
            ):
                raise ValueError(
                    f"{label} debe tener un RangeIndex por defecto (0..{n - 1}, "
                    "compartido entre keys/X/y_clf/y_reg) para que "
                    "folds.sample_idx siga siendo posicionalmente válido; "
                    f"recibido index={index!r}"
                )
        _validate_fold_provenance(self.keys, self.folds, self.manifest)

    def save(self, output_dir: Path | str) -> Path:
        """Escribe el dataset en ``output_dir/<prep_id>/`` y devuelve esa ruta."""
        _validate_manifest_protocol_definitions(self.manifest)
        _validate_manifest_identity(self.manifest)
        if self.prep_id != self.manifest.get("prep_id"):
            raise ValueError(
                "El prep_id del PreparedDataset no coincide con el prep_id del "
                f"manifiesto. {_REGENERATE_MESSAGE}"
            )
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
        _validate_manifest_protocol_definitions(manifest)
        _validate_manifest_identity(manifest)
        return cls(
            prep_id=manifest["prep_id"],
            keys=pd.read_parquet(directory / _COMPONENT_FILES["keys"]),
            X=pd.read_parquet(directory / _COMPONENT_FILES["features"]),
            y_clf=targets[_CLF_TARGET_NAME],
            y_reg=targets[_REG_TARGET_NAME],
            folds=pd.read_parquet(directory / _COMPONENT_FILES["folds"]),
            manifest=manifest,
            quality=json.loads(
                (directory / _COMPONENT_FILES["quality"]).read_text(encoding="utf-8")
            ),
        )
