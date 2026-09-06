"""Export de métricas y ranking a archivos, para lectura humana y para la tesis.

La fuente de verdad de estos números es SQL (Plan 2); estos archivos son la
copia cómoda de leer, no el registro autoritativo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_reports(
    metrics_by_family: dict[str, dict],
    ranking: pd.DataFrame,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Escribe un JSON por familia más el ranking en JSON y CSV."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for family, metrics in metrics_by_family.items():
        path = directory / f"metrics_{family}.json"
        path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written[f"metrics_{family}"] = path

    ranking_json = directory / "ranking.json"
    ranking_json.write_text(
        json.dumps(ranking.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    ranking_csv = directory / "ranking.csv"
    ranking.to_csv(ranking_csv, index=False)

    written["ranking_json"] = ranking_json
    written["ranking_csv"] = ranking_csv
    return written
