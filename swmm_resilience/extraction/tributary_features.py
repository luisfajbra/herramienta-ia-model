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
