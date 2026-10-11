"""Export de métricas y ranking a archivos, para lectura humana y para la tesis.

La fuente de verdad de estos números es SQL (Plan 2); estos archivos son la
copia cómoda de leer, no el registro autoritativo.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .schemas import PROTOCOL_INTERPRETATION, PreparedDataset


def _json_safe(obj):
    """Reemplaza NaN/Infinity por ``None`` recursivamente.

    JSON estricto (RFC 8259) no admite ``NaN``/``Infinity``; el módulo
    ``json`` de Python los serializa igual como tokens desnudos salvo que
    se le pida lo contrario. Sin este saneo, un ``auc_roc`` NaN (folds LOSO
    con una sola clase, algo que ocurre en corridas reales con factores de
    intensidad bajos donde ningún nodo se inunda) o un ``primary_value``
    NaN producirían un archivo que *parece* JSON válido pero que lectores
    estrictos -- como las funciones ``json1`` de SQLite, que leen estos
    archivos en un plan posterior -- rechazan. No se pierde información:
    en el ranking, ``valid``/``invalid_reason`` ya documentan el caso, y en
    las métricas ``null`` significa lo mismo que ``NaN``: "no definido
    aquí".
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    return obj


def write_metrics_files(
    metrics_by_name: dict[str, dict], output_dir: Path | str, prefix: str = "metrics_"
) -> dict[str, Path]:
    """Escribe un JSON estricto por nombre (familia o referencia)."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, metrics in metrics_by_name.items():
        path = directory / f"{prefix}{name}.json"
        path.write_text(
            json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        written[f"{prefix}{name}"] = path
    return written


def write_oof(oof: pd.DataFrame, root: Path | str, prep_id: str, name: str) -> Path:
    """Guarda las predicciones fuera de fold en ``<root>/<prep_id>/<name>.parquet``."""
    directory = Path(root) / prep_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.parquet"
    oof.to_parquet(path, index=False)
    return path


def write_provenance(
    output_dir: Path | str,
    prepared: PreparedDataset,
    families: dict[str, dict],
    baselines: list[str],
) -> Path:
    """``provenance.json``: qué dataset, qué hiperparámetros y qué mide cada protocolo."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "prep_id": prepared.prep_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_interpretation": dict(PROTOCOL_INTERPRETATION),
        "families": families,
        "baselines": list(baselines),
    }
    path = directory / "provenance.json"
    path.write_text(
        json.dumps(_json_safe(record), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path


def write_reports(
    metrics_by_family: dict[str, dict],
    ranking: pd.DataFrame,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Escribe un JSON por familia más el ranking en JSON y CSV.

    ``rank_candidates`` sigue devolviendo NaN tal cual en el DataFrame en
    memoria -- solo la representación en disco cambia NaN por ``null``.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = write_metrics_files(metrics_by_family, directory)

    ranking_json = directory / "ranking.json"
    ranking_json.write_text(
        json.dumps(
            _json_safe(ranking.to_dict(orient="records")),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    ranking_csv = directory / "ranking.csv"
    ranking.to_csv(ranking_csv, index=False)

    written["ranking_json"] = ranking_json
    written["ranking_csv"] = ranking_csv
    return written
