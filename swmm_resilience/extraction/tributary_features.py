"""Q_tributario y las seis features de forma tributaria (spec §6, contrato nuevo).

Q_tributario(t) para un nodo es la suma de su propia serie de aporte y la de
todos sus ancestros únicos en la red, sobre una malla temporal común. Se
calcula por interpolación lineal explícita -- nunca sustituyendo la suma
temporal por una suma de picos, que es justamente lo que ya hace
q_pico_acum_escalado (contrato legado, se conserva como proxy documentado).

Este módulo es puro: no sabe de SWMM, de la base de datos ni de la política
de cierre de fase B1. Recibe la serie ya efectiva (post-cierre) de cada
nodo y el grafo de la red, y no hace I/O.
"""

from __future__ import annotations

import networkx as nx

_Point = tuple[float, float]
_Series = list[_Point]


def _interp_at(series: _Series, t: float) -> float:
    """Interpolación lineal de `series` (ordenada por t) en el instante `t`.

    Fuera del rango de `series` se sostiene el valor del extremo más
    cercano (clamp), consistente con cómo write_scenario_inp trata el
    tramo de drenaje: una serie nunca "empieza" antes de su primer punto
    ni "sigue cambiando" después del último.
    """
    if t <= series[0][0]:
        return series[0][1]
    if t >= series[-1][0]:
        return series[-1][1]
    for (t0, v0), (t1, v1) in zip(series, series[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return v0
            frac = (t - t0) / (t1 - t0)
            return v0 + (v1 - v0) * frac
    return series[-1][1]


def compute_q_tributario(
    node_id: str,
    node_series: dict[str, _Series],
    graph: "nx.DiGraph",
) -> _Series:
    """Suma la serie de `node_id` y la de sus ancestros únicos, malla común.

    Un nodo sin aporte propio (base_inflow_lps == 0) nunca aparece en
    node_series -- se ignora, no contribuye. Un nodo ausente del grafo
    (o sin ancestros) devuelve solo su propia serie.
    """
    ancestors = set(nx.ancestors(graph, node_id)) if graph.has_node(node_id) else set()
    contributing = sorted((ancestors | {node_id}) & set(node_series))
    if not contributing:
        return []

    grid = sorted({t for nid in contributing for t, _ in node_series[nid]})
    return [
        (t, sum(_interp_at(node_series[nid], t) for nid in contributing))
        for t in grid
    ]


def _trapezoid_lps_h(series: _Series) -> float:
    """Integral trapezoidal de `series` (t en horas, q en L/s), en L/s*h."""
    return sum(
        (t1 - t0) * (v0 + v1) / 2.0
        for (t0, v0), (t1, v1) in zip(series, series[1:])
    )


def _segment_duration_above(t0: float, v0: float, t1: float, v1: float, threshold: float) -> float:
    """Duracion exacta donde v(t) > threshold sobre el segmento lineal [t0,t1]."""
    e0, e1 = v0 - threshold, v1 - threshold
    if e0 <= 0 and e1 <= 0:
        return 0.0
    if e0 > 0 and e1 > 0:
        return t1 - t0
    t_cross = t0 + (t1 - t0) * (e0 / (e0 - e1))
    return (t_cross - t0) if e0 > 0 else (t1 - t_cross)


def _segment_excess_integral(t0: float, v0: float, t1: float, v1: float, cap: float) -> float:
    """Integral exacta de max(v(t)-cap, 0) sobre el segmento lineal [t0,t1], en L/s*h."""
    e0, e1 = v0 - cap, v1 - cap
    if e0 <= 0 and e1 <= 0:
        return 0.0
    if e0 >= 0 and e1 >= 0:
        return (t1 - t0) * (e0 + e1) / 2.0
    t_cross = t0 + (t1 - t0) * (e0 / (e0 - e1))
    if e0 > 0:
        return (t_cross - t0) * e0 / 2.0
    return (t1 - t_cross) * e1 / 2.0


def volumen_local_integrado_m3(series: _Series) -> float:
    """Integral de la serie propia del nodo, en m3 (spec §6)."""
    return _trapezoid_lps_h(series) * 3.6


def volumen_tributario_integrado_m3(q_tributario: _Series) -> float:
    """Integral de Q_tributario, en m3."""
    return _trapezoid_lps_h(q_tributario) * 3.6


def duracion_efectiva_tributario_h(q_tributario: _Series) -> float:
    """Duracion en que Q_tributario supera el 1% de su propio pico."""
    if not q_tributario:
        return 0.0
    peak = max(v for _, v in q_tributario)
    if peak <= 0:
        return 0.0
    threshold = 0.01 * peak
    return sum(
        _segment_duration_above(t0, v0, t1, v1, threshold)
        for (t0, v0), (t1, v1) in zip(q_tributario, q_tributario[1:])
    )


def tiempo_primer_pico_tributario_h(q_tributario: _Series) -> float:
    """Instante del primer maximo de Q_tributario; cero para una serie nula."""
    if not q_tributario:
        return 0.0
    peak = max(v for _, v in q_tributario)
    if peak <= 0:
        return 0.0
    return next(t for t, v in q_tributario if v == peak)


def tiempo_sobre_capacidad_h(q_tributario: _Series, capacity_lps: float | None) -> float | None:
    """Duracion en que Q_tributario supera upstream_capacity_lps.

    None si la capacidad falta o es <= 0 (spec §6: sin divisiones artificiales).
    """
    if capacity_lps is None or capacity_lps <= 0:
        return None
    return sum(
        _segment_duration_above(t0, v0, t1, v1, capacity_lps)
        for (t0, v0), (t1, v1) in zip(q_tributario, q_tributario[1:])
    )


def integral_exceso_capacidad_m3(q_tributario: _Series, capacity_lps: float | None) -> float | None:
    """Integral de max(Q_tributario - upstream_capacity_lps, 0), en m3.

    None si la capacidad falta o es <= 0.
    """
    if capacity_lps is None or capacity_lps <= 0:
        return None
    total = sum(
        _segment_excess_integral(t0, v0, t1, v1, capacity_lps)
        for (t0, v0), (t1, v1) in zip(q_tributario, q_tributario[1:])
    )
    return total * 3.6


def compute_node_tributary_features(
    node_id: str,
    node_series: dict[str, _Series],
    graph: "nx.DiGraph",
    capacity_lps: float | None,
) -> dict:
    """Agrupa las seis features tributarias de un nodo."""
    own_series = node_series.get(node_id, [(0.0, 0.0), (0.0, 0.0)])
    q_trib = compute_q_tributario(node_id, node_series, graph)
    return {
        "volumen_local_integrado_m3": volumen_local_integrado_m3(own_series),
        "volumen_tributario_integrado_m3": volumen_tributario_integrado_m3(q_trib),
        "duracion_efectiva_tributario_h": duracion_efectiva_tributario_h(q_trib),
        "tiempo_primer_pico_tributario_h": tiempo_primer_pico_tributario_h(q_trib),
        "tiempo_sobre_capacidad_h": tiempo_sobre_capacidad_h(q_trib, capacity_lps),
        "integral_exceso_capacidad_m3": integral_exceso_capacidad_m3(q_trib, capacity_lps),
    }
