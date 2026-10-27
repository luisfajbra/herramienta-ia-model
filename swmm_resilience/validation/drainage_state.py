"""Clasificacion del estado de drenaje a partir del .out binario de SWMM.

Antes de este modulo, el pipeline asumia que una simulacion habia drenado
solo por haber corrido END_TIME + drain_down_hours, o por profundidad cero
aislada en un unico paso -- ninguna de las dos demuestra drenaje real (spec
Sec.7 del diseno de cierre de auditoria). Este modulo lee observables reales
del .out binario de SWMM (nunca antes leido en este proyecto -- todo el resto
del pipeline usa el .rpt de texto) en la ultima ventana de la simulacion, con
al menos dos reportes: aporte total, inundacion y volumen por nodo, y flujo
por enlace.

El modulo cubre las tres etapas del flujo de estado de drenaje:

- `read_tail_window_observables()` lee esas series crudas de la ventana final
  del .out.
- `classify_drainage_state()` aplica la regla operativa del spec sobre esas
  series (ausencia de aporte e inundacion sobre tolerancia, cambio de
  almacenamiento estabilizado, flujo residual dentro de tolerancia) y
  devuelve el veredicto junto con un diagnostico auditable.
- `save_drainage_state_manifest()` persiste ese diagnostico como sidecar
  `<scenario_id>.drainage_state.json` junto al escenario.

El flujo residual es una simplificacion "basal cero": se compara el flujo de
enlaces contra cero, no contra una corrida basal real, porque este plan no
incluye una simulacion de referencia. Las cuatro tolerancias (aporte,
inundacion, cambio de almacenamiento, flujo residual) son valores por
defecto pendientes de calibracion piloto, no constantes derivadas del
comportamiento real de una red -- ver Global Constraints del plan que
introdujo este modulo.
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path


def _part_to_dict(part) -> dict:
    """Convierte el resultado de SwmmOutput.get_part() a {id: [valores]}.

    get_part() devuelve un pandas.DataFrame cuando hay mas de un objeto
    (columna) seleccionado, pero una pandas.Series cuando solo hay uno --
    documentado en su propio help(). Normalizamos ambos casos y forzamos
    las claves a str para que coincidan con como el resto del proyecto
    identifica nodos/enlaces (ids como strings, no lo que swmm_api use
    internamente para las etiquetas de columna).
    """
    if part is None:
        return {}
    if hasattr(part, "columns"):
        return {str(col): part[col].tolist() for col in part.columns}
    # Series: un unico objeto seleccionado. part.name NO es el id limpio --
    # SwmmOutput._to_pandas() devuelve df.iloc[:, 0] *antes* de separar la
    # etiqueta interna de columna, asi que part.name es el label crudo
    # "kind;label;variable" unido con SwmmOutput._COL_SEP (p.ej.
    # "node;J1;total_inflow"), no "J1" -- verificado contra swmm_api 0.4.73
    # forzando una seleccion de un solo objeto (get_part(label=...)) e
    # inspeccionando .name directamente, y confirmado leyendo
    # SwmmOutput._to_pandas() en swmm_api/output_file/out.py (linea ~369:
    # `if df.columns.size == 1: return df.iloc[:, 0]` ocurre antes del split
    # por _COL_SEP que arma el MultiIndex para el caso multi-columna).
    from swmm_api import SwmmOutput

    raw_label = str(part.name)
    segments = raw_label.split(SwmmOutput._COL_SEP)
    object_id = segments[1] if len(segments) >= 2 else raw_label
    return {object_id: part.tolist()}


def read_tail_window_observables(
    out_path: Path | str,
    drain_down_hours: float,
    tail_window_hours: float = 1.0,
) -> dict:
    """Lee nodos y enlaces en la ventana final de `out_path`.

    La ventana es `min(tail_window_hours, drain_down_hours)` horas antes del
    final de la simulacion. Devuelve un dict con `n_report_steps`,
    `has_link_data`, y las series por objeto de aporte/inundacion/volumen
    (nodos) y flujo (enlaces).
    """
    from swmm_api import SwmmOutput

    window_hours = min(tail_window_hours, drain_down_hours)

    with SwmmOutput(str(out_path)) as out:
        end_time = out.index[-1]
        start_time = end_time - datetime.timedelta(hours=window_hours)
        # Si la ventana pedida es mas larga que la simulacion completa,
        # start_time cae antes del primer paso reportado. get_part() hace un
        # seek() absoluto sobre el binario con ese offset y revienta con
        # OSError: [Errno 22] Invalid argument si es negativo -- se ajusta al
        # primer paso disponible en vez de dejar que la libreria falle.
        start_time = max(start_time, out.index[0])
        observed_window_hours = max(
            0.0, (end_time - start_time).total_seconds() / 3600.0
        )

        node_inflow = out.get_part(kind="node", variable="total_inflow", start=start_time, end=end_time)
        node_flooding = out.get_part(kind="node", variable="flooding", start=start_time, end=end_time)
        node_volume = out.get_part(kind="node", variable="volume", start=start_time, end=end_time)
        try:
            link_flow = out.get_part(kind="link", variable="flow", start=start_time, end=end_time)
            has_link_data = link_flow is not None and not link_flow.empty
        except Exception:
            link_flow = None
            has_link_data = False

    n_report_steps = len(node_inflow.index) if node_inflow is not None else 0

    return {
        "n_report_steps": n_report_steps,
        "observed_window_hours": observed_window_hours,
        "has_link_data": has_link_data,
        "node_total_inflow_lps": _part_to_dict(node_inflow),
        "node_flooding_lps": _part_to_dict(node_flooding),
        "node_volume_m3": _part_to_dict(node_volume),
        "link_flow_lps": _part_to_dict(link_flow) if has_link_data else {},
    }


DRAINAGE_STATE_POLICY_VERSION = "tail_window_v1"


def _max_abs_value(series_by_id: dict) -> float | None:
    """Maximo absoluto entre todos los valores de todas las series de `series_by_id`."""
    if not isinstance(series_by_id, dict):
        return None
    values = [
        abs(float(value))
        for series in series_by_id.values()
        for value in series
        if _is_finite_number(value)
    ]
    return max(values) if values else None


def _max_abs_step_delta(series_by_id: dict) -> float | None:
    """Maxima diferencia absoluta entre pasos consecutivos, entre todas las series."""
    if not isinstance(series_by_id, dict):
        return None
    deltas = []
    for series in series_by_id.values():
        for i in range(len(series) - 1):
            if _is_finite_number(series[i]) and _is_finite_number(series[i + 1]):
                deltas.append(abs(float(series[i + 1]) - float(series[i])))
    return max(deltas) if deltas else None


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def classify_drainage_state(
    observables: dict,
    inflow_tolerance_lps: float,
    flooding_tolerance_lps: float,
    storage_change_tolerance_m3: float,
    residual_flow_tolerance_lps: float,
    window_hours: float,
    min_report_steps: int = 2,
) -> tuple[str, dict]:
    """Aplica la regla operativa del spec Sec.7 sobre `observables`.

    "drained" exige: suficientes reportes, datos de enlace presentes,
    aporte e inundacion bajo tolerancia en toda la ventana, cambio de
    almacenamiento estabilizado entre pasos consecutivos, y flujo residual
    de nodos y enlaces bajo tolerancia (basal = cero, ver Global
    Constraints de este plan). Cualquier dato insuficiente -> "unverified",
    nunca "drained" ni "not_drained" por omision.

    `window_hours` es la ventana efectiva ya resuelta por el llamador (no se
    recalcula aqui); se registra en el diagnostico devuelto para que el
    manifiesto sea auditable.
    """
    failed_checks: list[str] = []
    node_series = (
        observables.get("node_total_inflow_lps", {}),
        observables.get("node_flooding_lps", {}),
        observables.get("node_volume_m3", {}),
    )
    link_series = observables.get("link_flow_lps", {})

    if any(
        not _is_finite_number(tolerance) or float(tolerance) < 0
        for tolerance in (
            inflow_tolerance_lps,
            flooding_tolerance_lps,
            storage_change_tolerance_m3,
            residual_flow_tolerance_lps,
        )
    ):
        failed_checks.append("invalid_tolerance_policy")
    if not _is_finite_number(window_hours) or window_hours <= 0:
        failed_checks.append("invalid_requested_window")
    if (
        observables.get("observed_window_hours") is not None
        and (
            not _is_finite_number(observables["observed_window_hours"])
            or observables["observed_window_hours"] + 1e-9 < window_hours
        )
    ):
        failed_checks.append("insufficient_observed_window")
    if len({frozenset(series) for series in node_series if isinstance(series, dict)}) != 1:
        failed_checks.append("incomplete_node_variable_coverage")
    if any(
        not isinstance(series_dict, dict)
        or any(
            not isinstance(values, (list, tuple))
            or any(not _is_finite_number(value) for value in values)
            for values in series_dict.values()
        )
        for series_dict in (*node_series, link_series)
    ):
        failed_checks.append("non_finite_observables")

    if observables.get("n_report_steps", 0) < min_report_steps:
        failed_checks.append("insufficient_report_steps")
    if not observables.get("has_link_data") or not link_series:
        failed_checks.append("missing_link_data")
    if (
        not node_series[0]
        or not node_series[1]
        or not node_series[2]
    ):
        failed_checks.append("missing_node_data")
    if any(
        len(series) < min_report_steps
        for series_dict in (*node_series, link_series)
        for series in series_dict.values()
    ):
        failed_checks.append("insufficient_series_length")
    if any(
        len(series) != observables.get("n_report_steps")
        for series_dict in (*node_series, link_series)
        for series in series_dict.values()
    ):
        failed_checks.append("incomplete_report_series")

    if failed_checks:
        state = "unverified"
    else:
        if any(
            abs(v) > inflow_tolerance_lps
            for series in observables["node_total_inflow_lps"].values()
            for v in series
        ):
            failed_checks.append("inflow")
        if any(
            abs(v) > flooding_tolerance_lps
            for series in observables["node_flooding_lps"].values()
            for v in series
        ):
            failed_checks.append("flooding")
        if any(
            abs(series[i + 1] - series[i]) > storage_change_tolerance_m3
            for series in observables["node_volume_m3"].values()
            for i in range(len(series) - 1)
        ):
            failed_checks.append("storage_change")
        if any(
            abs(v) > residual_flow_tolerance_lps
            for series in observables["link_flow_lps"].values()
            for v in series
        ):
            failed_checks.append("residual_flow")

        state = "drained" if not failed_checks else "not_drained"

    diagnostics = {
        "policy_version": DRAINAGE_STATE_POLICY_VERSION,
        "state": state,
        "failed_checks": failed_checks,
        "n_report_steps": observables.get("n_report_steps"),
        "observed_window_hours": observables.get("observed_window_hours"),
        "has_link_data": observables.get("has_link_data"),
        "window_hours": window_hours,
        "min_report_steps": min_report_steps,
        "max_inflow_lps": _max_abs_value(observables.get("node_total_inflow_lps", {})),
        "max_flooding_lps": _max_abs_value(observables.get("node_flooding_lps", {})),
        "max_storage_change_m3": _max_abs_step_delta(observables.get("node_volume_m3", {})),
        "max_residual_flow_lps": _max_abs_value(observables.get("link_flow_lps", {})),
        "tolerances": {
            "inflow_tolerance_lps": inflow_tolerance_lps,
            "flooding_tolerance_lps": flooding_tolerance_lps,
            "storage_change_tolerance_m3": storage_change_tolerance_m3,
            "residual_flow_tolerance_lps": residual_flow_tolerance_lps,
        },
    }
    return state, diagnostics


def save_drainage_state_manifest(manifest: dict, out_dir: Path | str, scenario_id: str) -> Path:
    """Escribe `manifest` en `<out_dir>/<scenario_id>.drainage_state.json`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scenario_id}.drainage_state.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
