"""Orden de candidatos por métrica primaria y desempates.

Un candidato cuya métrica primaria sea NaN o falte no se descarta en
silencio: queda en el ranking, marcado inválido y con su motivo, en el
último lugar. Esa información es parte de la evidencia, no ruido.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

_DIRECTIONS = ("minimize", "maximize")


@dataclass(frozen=True)
class RankingCriterion:
    primary_metric: str
    primary_direction: str
    tie_breakers: Sequence[tuple[str, str]] = ()

    def __post_init__(self) -> None:
        for direction in [self.primary_direction] + [d for _, d in self.tie_breakers]:
            if direction not in _DIRECTIONS:
                raise ValueError(
                    f"Dirección desconocida: {direction!r}. Opciones: minimize, maximize"
                )


def resolve_metric(metrics: dict, dotted_path: str) -> float | None:
    """Lee ``'end_to_end.rmse_vol_todos_nodos'`` de un dict anidado."""
    node = metrics
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, (int, float)) else None


def _sort_value(value: float | None, direction: str) -> float:
    if value is None or math.isnan(value):
        return math.inf
    return -value if direction == "maximize" else value


def rank_candidates(
    metrics_by_family: dict[str, dict],
    criterion: RankingCriterion,
    protocol: str,
) -> pd.DataFrame:
    """Ordena las familias por la métrica primaria y sus desempates."""
    rows = []
    for family, all_metrics in metrics_by_family.items():
        # Every row gets the full column set up front, regardless of which
        # branch below fires. Otherwise a run where every candidate is
        # missing the protocol never touches the tie-breaker keys, and with
        # no valid row to union against, pandas silently drops those
        # columns from the frame.
        row = {
            "family": family,
            "primary_value": None,
            "valid": 0,
            "invalid_reason": None,
        }
        for path, _ in criterion.tie_breakers:
            row[path] = None

        protocol_metrics = all_metrics.get(protocol)
        if protocol_metrics is None:
            row["invalid_reason"] = f"el candidato no tiene resultados para {protocol}"
            rows.append(row)
            continue

        primary = resolve_metric(protocol_metrics, criterion.primary_metric)
        if primary is None:
            row["invalid_reason"] = f"falta la métrica primaria {criterion.primary_metric}"
        elif math.isnan(primary):
            row["invalid_reason"] = f"la métrica primaria {criterion.primary_metric} es NaN"
        else:
            row["valid"] = 1

        row["primary_value"] = primary
        for path, _ in criterion.tie_breakers:
            row[path] = resolve_metric(protocol_metrics, path)
        rows.append(row)

    def sort_key(row: dict) -> tuple:
        keys = [1 - row["valid"], _sort_value(row["primary_value"], criterion.primary_direction)]
        for path, direction in criterion.tie_breakers:
            keys.append(_sort_value(row.get(path), direction))
        keys.append(row["family"])
        return tuple(keys)

    ordered = sorted(rows, key=sort_key)
    frame = pd.DataFrame(ordered)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    frame.insert(1, "protocol", protocol)
    return frame
