"""Lectura de la ventana final del .out y clasificacion del estado de drenaje.

Antes de este modulo, el pipeline asumia que una simulacion habia drenado
solo por haber corrido END_TIME + drain_down_hours, o por profundidad cero
aislada en un unico paso -- ninguna de las dos demuestra drenaje real (spec
Sec.7 del diseno de cierre de auditoria). Este modulo lee observables reales
del .out binario de SWMM (nunca antes leido en este proyecto -- todo el resto
del pipeline usa el .rpt de texto) en la ultima ventana de la simulacion, con
al menos dos reportes, y aplica la regla operativa del spec: ausencia de
aporte e inundacion sobre tolerancia, cambio de almacenamiento estabilizado,
y flujo residual de nodos y enlaces dentro de tolerancia.
"""

from __future__ import annotations

import datetime
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
    # Series: un unico objeto seleccionado; part.name es su id.
    return {str(part.name): part.tolist()}


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
        "has_link_data": has_link_data,
        "node_total_inflow_lps": _part_to_dict(node_inflow),
        "node_flooding_lps": _part_to_dict(node_flooding),
        "node_volume_m3": _part_to_dict(node_volume),
        "link_flow_lps": _part_to_dict(link_flow) if has_link_data else {},
    }
