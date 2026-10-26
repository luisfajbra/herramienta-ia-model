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
import math
import platform
import sqlite3
import sys
from contextlib import contextmanager
from datetime import date, datetime, timezone
from importlib.metadata import version as _package_version
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import numpy as np
import pandas as pd

from ...database.training_queries import load_training_frame
from ..contracts import FEATURE_COLUMNS_V17, TABULAR_V3_17
from .folds import build_all_folds
from .quality import build_quality_report
from .schemas import (
    KEY_COLUMNS,
    PROTOCOLS,
    PreparedDataset,
    TRAINING_FRAME_FINGERPRINT_VERSION,
    compute_prep_id,
    group_column_for,
)

SORT_KEY = ("run_id", "node_id")


def _safe_package_version(name: str) -> str:
    """Version lookup for provenance metadata, not for anything load-bearing.

    importlib.metadata.version() raises PackageNotFoundError if the
    distribution is installed under a different name than the import name
    (or is not resolvable as a distribution at all). That must not abort the
    whole prepare stage over a string destined for manifest.json -- so a
    failed lookup degrades to "unknown" for that entry instead of
    propagating.
    """
    try:
        return _package_version(name)
    except Exception:
        return "unknown"


def _library_versions() -> dict:
    return {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": _safe_package_version("scikit-learn"),
    }


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_scalar(value: object) -> dict[str, object]:
    """Encode one supported pandas scalar without relying on ``repr``."""
    if value is None or value is pd.NA or value is pd.NaT:
        return {"type": "null"}
    if isinstance(value, np.datetime64) and np.isnat(value):
        return {"type": "null"}
    if isinstance(value, (bool, np.bool_)):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, (Integral, np.integer)):
        return {"type": "int", "value": int(value)}
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            encoded = "nan"
        elif math.isinf(number):
            encoded = "inf" if number > 0 else "-inf"
        else:
            encoded = number.hex()
        return {"type": "float", "value": encoded}
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64, date)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return {"type": "null"}
        return {"type": "datetime", "value": timestamp.isoformat()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise TypeError(
        "Unsupported training-frame value type for canonical fingerprint: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _training_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash the ordered frame using the versioned, dtype-aware serialization."""
    payload = {
        "version": TRAINING_FRAME_FINGERPRINT_VERSION,
        "columns": [
            {
                "name": column,
                "dtype": str(series.dtype),
                "values": [_canonical_scalar(value) for value in series.tolist()],
            }
            for column, series in frame.items()
        ],
        "row_count": len(frame),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


@contextmanager
def _readonly_sqlite_snapshot(db_path: Path | str):
    """Yield a SQLite backup captured from ``db_path`` through a read-only URI."""
    source_connection = None
    snapshot_connection = None
    with TemporaryDirectory(prefix="training-snapshot-") as temporary_dir:
        snapshot_path = Path(temporary_dir) / "source.sqlite3"
        try:
            source_uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
            source_connection = sqlite3.connect(source_uri, uri=True)
            snapshot_connection = sqlite3.connect(snapshot_path)
            source_connection.backup(snapshot_connection)
            snapshot_connection.commit()
        except Exception as exc:
            raise RuntimeError(
                f"Could not capture {db_path!s} through a read-only SQLite backup"
            ) from exc
        finally:
            if snapshot_connection is not None:
                snapshot_connection.close()
            if source_connection is not None:
                source_connection.close()
        yield snapshot_path


def protocol_definitions(protocols: Sequence[str]) -> dict[str, dict[str, str]]:
    return {
        protocol: {"group_column": group_column_for(protocol)}
        for protocol in protocols
    }


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _source_provenance(
    db_path: Path | str,
    *,
    included_run_ids: Sequence[int],
    run_ids: Sequence[int] | None,
    network_ids: Sequence[int] | None,
    max_continuity_error_pct: float | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Collect source hashes and excluded runs from the same read-only snapshot."""
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        clauses = ["run.status='COMPLETE'"]
        params: list[int] = []
        if run_ids is not None:
            if not run_ids:
                raise ValueError("run_ids cannot be empty")
            clauses.append(f"run.run_id IN ({','.join('?' for _ in run_ids)})")
            params.extend(run_ids)
        elif network_ids is not None:
            if not network_ids:
                raise ValueError("network_ids cannot be empty")
            clauses.append(f"run.network_id IN ({','.join('?' for _ in network_ids)})")
            params.extend(network_ids)
        rows = conn.execute(
            """SELECT run.run_id, run.network_id, run.continuity_error_pct,
                      run.config_sha256 AS run_config_sha256,
                      scenario.scenario_id, scenario.shape_id, scenario.factor_mult,
                      scenario.config_json, scenario.config_sha256 AS scenario_config_sha256,
                      network.name, network.network_sha256
               FROM runs AS run
               JOIN scenarios AS scenario
                 ON scenario.scenario_id=run.scenario_id
                AND scenario.network_id=run.network_id
               JOIN networks AS network ON network.network_id=run.network_id
               WHERE """ + " AND ".join(clauses) + " ORDER BY run.network_id, run.run_id",
            params,
        ).fetchall()
    finally:
        conn.close()

    included = set(int(value) for value in included_run_ids)
    excluded = []
    networks: dict[int, dict[str, object]] = {}
    require_complete_provenance = (
        (network_ids is not None and len(network_ids) > 1)
        or len({int(row["network_id"]) for row in rows}) > 1
    )
    for row in rows:
        config = json.loads(row["config_json"])
        valid_config_hashes = {
            hashlib.sha256(row["config_json"].encode("utf-8")).hexdigest(),
            _canonical_json_sha256(config),
        }
        if row["scenario_config_sha256"] not in valid_config_hashes:
            raise ValueError(
                f"Scenario source manifest hash is invalid for scenario_id={row['scenario_id']}"
            )
        if row["run_config_sha256"] != row["scenario_config_sha256"]:
            raise ValueError(
                f"Run/scenario configuration hash mismatch for run_id={row['run_id']}"
            )
        source = config.get("source_manifest", {})
        if require_complete_provenance:
            if not isinstance(source, dict) or not source:
                raise ValueError(
                    f"Source manifest is missing for multi-network run_id={row['run_id']}"
                )
            training_dataset = source.get("training_dataset")
            v4_dataset = source.get("v4_21_training_dataset")
            event_components = source.get("event_series_components")
            event_sha = source.get("event_series_sha256")
            if not source.get("network_inp_path") or not _is_sha256(
                source.get("network_inp_sha256")
            ):
                raise ValueError(
                    f"Network INP provenance is incomplete for run_id={row['run_id']}"
                )
            if not isinstance(training_dataset, dict) or not _is_sha256(
                training_dataset.get("sha256")
            ):
                raise ValueError(
                    f"Training dataset provenance is incomplete for run_id={row['run_id']}"
                )
            if not isinstance(event_components, dict) or not event_sha:
                raise ValueError(
                    f"Event-series provenance is incomplete for run_id={row['run_id']}"
                )
            if (
                event_components.get("network_inp_sha256") != row["network_sha256"]
                or event_components.get("shape_id") != row["shape_id"]
            ):
                raise ValueError(
                    f"Event-series identity does not match network/scenario for run_id={row['run_id']}"
                )
            try:
                event_factor = float(event_components["factor_mult"])
                scenario_factor = float(row["factor_mult"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Event-series factor is missing or invalid for run_id={row['run_id']}"
                ) from exc
            if event_factor != scenario_factor:
                raise ValueError(
                    f"Event-series factor does not match run_id={row['run_id']}"
                )
            if row["shape_id"] != "base" and not _is_sha256(
                event_components.get("hydrograph_sha256")
            ):
                raise ValueError(
                    f"Hydrograph source hash is missing for run_id={row['run_id']}"
                )
            if (
                not isinstance(v4_dataset, dict)
                or not _is_sha256(v4_dataset.get("sha256"))
                or not isinstance(v4_dataset.get("manifest"), dict)
                or not _is_sha256(v4_dataset["manifest"].get("sha256"))
            ):
                raise ValueError(
                    f"v4_21 dataset provenance is incomplete for run_id={row['run_id']}"
                )
        if source.get("network_id") not in (None, row["network_id"]):
            raise ValueError(
                f"Network source manifest identity mismatch for run_id={row['run_id']}"
            )
        inp_sha = source.get("network_inp_sha256")
        if inp_sha is not None and inp_sha != row["network_sha256"]:
            raise ValueError(
                f"Network source manifest hash does not match network_id={row['network_id']}"
            )
        event_components = source.get("event_series_components")
        event_sha = source.get("event_series_sha256")
        if event_components is not None and event_sha != _canonical_json_sha256(event_components):
            raise ValueError(
                f"Event source manifest hash is invalid for run_id={row['run_id']}"
            )
        network_id = int(row["network_id"])
        network = networks.setdefault(
            network_id,
            {
                "network_id": network_id,
                "name": row["name"],
                "network_sha256": row["network_sha256"],
                "inp_path": source.get("network_inp_path"),
                "runs": [],
            },
        )
        network["runs"].append(
            {
                "run_id": int(row["run_id"]),
                "scenario_id": int(row["scenario_id"]),
                "shape_id": row["shape_id"],
                "factor_mult": row["factor_mult"],
                "event_series_sha256": source.get("event_series_sha256"),
                "event_series_components": event_components,
                "v17_dataset": source.get("training_dataset"),
                "v4_21_dataset": source.get("v4_21_training_dataset"),
            }
        )
        if int(row["run_id"]) not in included:
            continuity = row["continuity_error_pct"]
            reason = (
                "continuity_error_exceeds_threshold"
                if max_continuity_error_pct is not None
                and continuity is not None
                and abs(float(continuity)) > max_continuity_error_pct
                else "not_in_training_frame"
            )
            excluded.append(
                {
                    "run_id": int(row["run_id"]),
                    "network_id": network_id,
                    "continuity_error_pct": continuity,
                    "reason": reason,
                }
            )
    return list(networks.values()), excluded


def prepare_dataset(
    db_path: Path | str,
    *,
    protocols: Sequence[str] = PROTOCOLS,
    flood_threshold_m3: float,
    output_dir: Path | str,
    max_continuity_error_pct: float | None = None,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS_V17,
    feature_contract=TABULAR_V3_17,
    view_name: str = "training_samples_v17",
    node_features_table: str = "node_features",
    run_ids: Sequence[int] | None = None,
    network_ids: Sequence[int] | None = None,
) -> PreparedDataset:
    """Lee el frame de SQL y materializa un PreparedDataset en ``output_dir``.

    ``max_continuity_error_pct`` (normalmente
    ``config.validation.max_continuity_error_pct``) excluye del banco los runs
    cuyo |continuity_error_pct| supera el umbral; ``None`` no excluye nada.
    Un ``continuity_error_pct`` NULL nunca excluye (misma regla que SQL).

    ``feature_columns``/``feature_contract``/``view_name``/``node_features_table``
    seleccionan el contrato de entrenamiento (default: tabular_v3_17, el
    legado). Pasar FEATURE_COLUMNS_V21/TABULAR_V4_21/"training_samples_v4_21"/
    "node_features_v4_21" para entrenar sobre el contrato tributario nuevo.
    """
    protocols = tuple(protocols)
    with _readonly_sqlite_snapshot(db_path) as snapshot_path:
        source_snapshot_sha256 = _sha256_file(snapshot_path)
        frame = load_training_frame(
            snapshot_path,
            max_continuity_error_pct=max_continuity_error_pct,
            view_name=view_name,
            feature_columns=feature_columns,
            feature_contract=feature_contract,
            node_features_table=node_features_table,
            run_ids=list(run_ids) if run_ids is not None else None,
            network_ids=list(network_ids) if network_ids is not None else None,
        )
        source_networks, excluded_runs = _source_provenance(
            snapshot_path,
            included_run_ids=frame["run_id"].unique(),
            run_ids=run_ids,
            network_ids=network_ids,
            max_continuity_error_pct=max_continuity_error_pct,
        )
    frame = frame.sort_values(list(SORT_KEY)).reset_index(drop=True)
    training_frame_sha256 = _training_frame_sha256(frame)

    keys = frame.loc[:, list(KEY_COLUMNS)].copy()
    X = feature_contract.validate_frame(frame.loc[:, list(feature_columns)])
    y_clf = frame["inunda"].astype(int).rename("inunda")
    y_reg = frame["vol_inundacion_m3"].astype(float).rename("vol_inundacion_m3")

    nullable_columns = frozenset(
        item.name for item in feature_contract.definitions if item.nullable
    )
    quality = build_quality_report(
        keys,
        X,
        y_clf,
        y_reg,
        flood_threshold_m3,
        feature_columns=feature_columns,
        nullable_columns=nullable_columns,
    )
    folds = build_all_folds(keys, protocols)
    definitions = protocol_definitions(protocols)

    descriptor = {
        "contract_id": feature_contract.contract_id,
        "feature_contract_sha256": feature_contract.descriptor_sha256,
        "run_ids": sorted(int(value) for value in keys["run_id"].unique()),
        "source_network_ids": sorted(int(value) for value in keys["network_id"].unique()),
        "source_networks": source_networks,
        "excluded_runs": excluded_runs,
        "n_rows": int(len(frame)),
        "feature_columns": list(feature_columns),
        "flood_threshold_m3": float(flood_threshold_m3),
        "sort_key": list(SORT_KEY),
        "protocols": list(protocols),
        "protocol_definitions": definitions,
        "source_capture_method": "sqlite_backup_readonly_v1",
        "training_frame_fingerprint_version": TRAINING_FRAME_FINGERPRINT_VERSION,
        "training_frame_sha256": training_frame_sha256,
    }
    prep_id = compute_prep_id(descriptor)

    manifest = dict(
        descriptor,
        # Fuera del descriptor (y por tanto del prep_id) a proposito: el efecto
        # del umbral sobre los datos ya queda capturado por
        # training_frame_sha256, que si entra al prep_id. Aqui queda solo como
        # procedencia de que umbral se aplico.
        max_continuity_error_pct=max_continuity_error_pct,
        source_snapshot_sha256=source_snapshot_sha256,
        excluded_run_ids=sorted(item["run_id"] for item in excluded_runs),
        prep_id=prep_id,
        db_path=str(db_path),
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


def _validate_requested_protocols(
    prepared: PreparedDataset, protocols: Sequence[str] | None
) -> PreparedDataset:
    if protocols is None:
        return prepared
    requested = protocol_definitions(tuple(protocols))
    available = prepared.manifest["protocol_definitions"]
    incompatible = [
        protocol
        for protocol, definition in requested.items()
        if available.get(protocol) != definition
    ]
    if incompatible:
        available_protocols = ", ".join(sorted(available)) or "(none)"
        raise ValueError(
            f"The PreparedDataset {prepared.prep_id} does not cover the semantics of the "
            f"protocol(s) {', '.join(incompatible)} requested by config.yaml "
            "(bench.protocols). Protocols available with the current semantics: "
            f"{available_protocols}. Run `python main.py --bench-prepare` again "
            "to regenerate it with the current protocols."
        )
    return prepared


def resolve_prepared(
    output_dir: Path | str,
    prep_id: str | None = None,
    protocols: Sequence[str] | None = None,
) -> PreparedDataset:
    """Carga un PreparedDataset por id, o el más reciente si no se da uno."""
    root = Path(output_dir)
    if prep_id is not None:
        directory = root / prep_id
        if not directory.exists():
            raise FileNotFoundError(
                f"The PreparedDataset {prep_id} does not exist in {root}. "
                "Run `python main.py --bench-prepare` first."
            )
        return _validate_requested_protocols(
            PreparedDataset.load(directory), protocols
        )

    candidates = [path for path in root.glob("*") if (path / "manifest.json").exists()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(
            f"There is no PreparedDataset in {root}. "
            "Run `python main.py --bench-prepare` first."
        )
    newest = max(candidates, key=lambda path: (path / "manifest.json").stat().st_mtime)
    return _validate_requested_protocols(PreparedDataset.load(newest), protocols)
