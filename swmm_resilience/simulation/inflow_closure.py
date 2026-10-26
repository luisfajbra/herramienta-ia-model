"""Política de cierre del aporte: separa fin-de-aporte de fin-de-simulación.

El fin del aporte (cuándo deja de entrar agua a la red) y el fin de la
simulación (cuánto tiempo se deja drenar después) son conceptos distintos.
Antes de este módulo, un aporte generado o subido por el usuario que
terminaba en un valor positivo se "cerraba" implícitamente: write_scenario_inp
insertaba un único punto en cero al final del colchón de drenaje, así que
SWMM interpolaba linealmente TODO ese colchón desde el último valor positivo
hasta cero -- el drenaje terminaba cerrando el aporte en vez de solo drenar
la red ya cerrada (spec §6 del diseño de cierre de auditoría).

Este módulo decide, antes de escribir ningún .inp, si cada serie de nodo ya
termina en (casi) cero. Si no, añade UN punto de cierre en el siguiente paso
de malla -- un cierre lineal rápido, versionado y explícito -- y deja que el
colchón de drenaje empiece ya en cero. Nunca sustituye ese colchón: solo dejar
de necesitarlo para cerrar el aporte.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..validation.hydrograph_csv import HydrographScenario

INFLOW_CLOSURE_POLICY_VERSION = "linear_v1"


def close_series_to_zero(
    series: list[tuple[float, float]], tolerance_lps: float,
) -> tuple[list[tuple[float, float]], bool]:
    """Cierra ``series`` a cero si su último valor supera ``tolerance_lps``.

    Si ``|series[-1][1]| <= tolerance_lps`` la serie ya está cerrada: se
    devuelve intacta y ``changed=False``. En otro caso se añade un único
    punto nuevo en ``series[-1][0] + paso``, valor 0.0, donde ``paso`` es el
    intervalo entre los dos últimos puntos de ``series``. Nunca trunca,
    reordena ni modifica los puntos existentes.
    """
    if len(series) < 2:
        raise ValueError(
            "The series needs at least 2 points to infer the grid step"
        )
    last_t, last_v = series[-1]
    if abs(last_v) <= tolerance_lps:
        return list(series), False
    step = last_t - series[-2][0]
    if step <= 0:
        raise ValueError(f"Non-positive grid step between the last two points: {step}")
    return list(series) + [(last_t + step, 0.0)], True


def apply_closure_policy(
    scenario: HydrographScenario, tolerance_lps: float,
) -> tuple[HydrographScenario, dict]:
    """Aplica ``close_series_to_zero`` a cada nodo de ``scenario``.

    Devuelve ``(escenario_efectivo, manifest)``. Si ningún nodo necesitó
    cierre, ``escenario_efectivo is scenario`` (mismo objeto, sin copiar) y
    el manifest documenta que no hubo cambios. ``scenario`` nunca se muta.
    """
    changed_nodes: list[str] = []
    new_node_series: dict[str, list[tuple[float, float]]] = {}
    new_last_time = scenario.last_time_hours

    for node_id, series in scenario.node_series.items():
        closed, changed = close_series_to_zero(series, tolerance_lps)
        new_node_series[node_id] = closed
        if changed:
            changed_nodes.append(node_id)
            new_last_time = max(new_last_time, closed[-1][0])

    manifest = {
        "policy_version": INFLOW_CLOSURE_POLICY_VERSION,
        "tolerance_lps": tolerance_lps,
        "source_scenario_id": scenario.scenario_id,
        "changed_nodes": sorted(changed_nodes),
        "original_last_time_hours": scenario.last_time_hours,
        "effective_last_time_hours": new_last_time,
    }

    if not changed_nodes:
        return scenario, manifest

    new_time_grid = sorted(set(scenario.time_grid_hours) | {new_last_time})
    closed_scenario = HydrographScenario(
        scenario_id=scenario.scenario_id,
        node_series=new_node_series,
        time_grid_hours=new_time_grid,
        last_time_hours=new_last_time,
    )
    return closed_scenario, manifest


def enforce_strict_zero_endpoint(scenario: HydrographScenario, tolerance_lps: float) -> None:
    """Rechaza ``scenario`` si algún nodo no termina en (casi) cero.

    Modo estricto para CSV de usuario: nunca modifica la serie. No se
    infiere ningún cierre; el usuario corrige el CSV o pide explícitamente
    el modo de cierre (``apply_closure_policy``).
    """
    bad = sorted(
        node_id for node_id, series in scenario.node_series.items()
        if abs(series[-1][1]) > tolerance_lps
    )
    if bad:
        raise ValueError(
            f"The scenario '{scenario.scenario_id}' has a non-zero final flow "
            f"(> {tolerance_lps} L/s) at nodes: {bad}. Strict mode (default): fix "
            "the CSV so that it ends at zero, or set "
            "validation.csv_inflow_closure_mode: \"close\" to close it automatically."
        )


def save_closure_manifest(manifest: dict, out_dir: Path | str, scenario_id: str) -> Path:
    """Escribe ``manifest`` en ``<out_dir>/<scenario_id>.closure_manifest.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scenario_id}.closure_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
