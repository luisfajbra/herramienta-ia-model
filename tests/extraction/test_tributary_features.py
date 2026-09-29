import networkx as nx
import pytest

from swmm_resilience.extraction.tributary_features import compute_q_tributario


def _linear_graph():
    """upstream -> mid -> downstream (upstream y mid son ancestros de downstream)."""
    g = nx.DiGraph()
    g.add_edge("upstream", "mid")
    g.add_edge("mid", "downstream")
    return g


def test_q_tributario_sums_node_and_ancestors_on_shared_grid():
    graph = _linear_graph()
    node_series = {
        "upstream": [(0.0, 1.0), (1.0, 3.0)],
        "mid": [(0.0, 2.0), (1.0, 4.0)],
        "downstream": [(0.0, 0.5), (1.0, 0.5)],
    }
    q = compute_q_tributario("downstream", node_series, graph)
    assert q == [(0.0, 3.5), (1.0, 7.5)]  # 1.0+2.0+0.5, 3.0+4.0+0.5


def test_q_tributario_interpolates_a_series_with_a_different_grid():
    graph = _linear_graph()
    node_series = {
        "upstream": [(0.0, 0.0), (2.0, 4.0)],  # solo 2 puntos, malla distinta
        "mid": [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)],
        "downstream": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
    }
    q = compute_q_tributario("downstream", node_series, graph)
    # malla union: {0.0, 1.0, 2.0}. upstream en t=1.0 se interpola: 0.0 + (4.0-0.0)*(1.0/2.0) = 2.0
    assert q == [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)]


def test_q_tributario_ignores_a_node_absent_from_node_series():
    """Un nodo sin aporte propio (base_inflow == 0) nunca entra a node_series;
    no debe romper la suma de sus descendientes."""
    graph = _linear_graph()
    node_series = {
        "mid": [(0.0, 2.0), (1.0, 4.0)],
        "downstream": [(0.0, 0.5), (1.0, 0.5)],
    }  # "upstream" ausente
    q = compute_q_tributario("downstream", node_series, graph)
    assert q == [(0.0, 2.5), (1.0, 4.5)]


def test_q_tributario_of_an_isolated_node_is_just_its_own_series():
    graph = nx.DiGraph()
    graph.add_node("solo")
    node_series = {"solo": [(0.0, 1.0), (1.0, 2.0)]}
    assert compute_q_tributario("solo", node_series, graph) == [(0.0, 1.0), (1.0, 2.0)]


def test_q_tributario_of_a_node_not_in_the_graph_falls_back_to_its_own_series():
    graph = nx.DiGraph()
    node_series = {"orphan": [(0.0, 1.0), (1.0, 2.0)]}
    assert compute_q_tributario("orphan", node_series, graph) == [(0.0, 1.0), (1.0, 2.0)]
