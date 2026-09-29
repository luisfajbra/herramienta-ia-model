import networkx as nx
import pytest

from swmm_resilience.extraction.tributary_features import (
    compute_node_tributary_features,
    compute_q_tributario,
    duracion_efectiva_tributario_h,
    integral_exceso_capacidad_m3,
    tiempo_primer_pico_tributario_h,
    tiempo_sobre_capacidad_h,
    volumen_local_integrado_m3,
    volumen_tributario_integrado_m3,
)


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


def test_volumen_integrado_uses_trapezoidal_rule_times_3_6():
    # trapecio: (0,0)-(1,10) -> area = 0.5*1*10 = 5.0 L/s*h -> *3.6 = 18.0 m3
    assert volumen_local_integrado_m3([(0.0, 0.0), (1.0, 10.0)]) == pytest.approx(18.0)


def test_volumen_tributario_integrado_matches_volumen_local_for_a_single_series():
    series = [(0.0, 0.0), (1.0, 10.0), (2.0, 0.0)]
    assert volumen_tributario_integrado_m3(series) == pytest.approx(
        volumen_local_integrado_m3(series)
    )


def test_duracion_efectiva_counts_only_time_above_one_percent_of_peak():
    # pico = 10.0, 1% = 0.1. Serie sube de 0 a 10 en [0,1], baja de 10 a 0 en [1,2].
    # cruza 0.1 en t=0.01 (subida) y en t=1.99 (bajada) -> duracion ~= 1.98h
    series = [(0.0, 0.0), (1.0, 10.0), (2.0, 0.0)]
    assert duracion_efectiva_tributario_h(series) == pytest.approx(1.98, abs=1e-6)


def test_duracion_efectiva_is_zero_for_a_null_series():
    assert duracion_efectiva_tributario_h([(0.0, 0.0), (1.0, 0.0)]) == 0.0


def test_tiempo_primer_pico_returns_the_time_of_the_first_maximum():
    # dos picos iguales a 5.0, en t=1.0 y t=3.0 -> el PRIMERO es t=1.0
    series = [(0.0, 0.0), (1.0, 5.0), (2.0, 0.0), (3.0, 5.0), (4.0, 0.0)]
    assert tiempo_primer_pico_tributario_h(series) == 1.0


def test_tiempo_primer_pico_is_zero_for_a_null_series():
    assert tiempo_primer_pico_tributario_h([(0.0, 0.0), (1.0, 0.0)]) == 0.0


def test_tiempo_sobre_capacidad_counts_time_above_the_exact_threshold():
    # capacidad = 5.0. Serie sube 0->10 en [0,1]: cruza 5.0 en t=0.5.
    series = [(0.0, 0.0), (1.0, 10.0)]
    assert tiempo_sobre_capacidad_h(series, capacity_lps=5.0) == pytest.approx(0.5)


def test_tiempo_sobre_capacidad_is_none_when_capacity_is_missing_or_non_positive():
    series = [(0.0, 0.0), (1.0, 10.0)]
    assert tiempo_sobre_capacidad_h(series, capacity_lps=None) is None
    assert tiempo_sobre_capacidad_h(series, capacity_lps=0.0) is None
    assert tiempo_sobre_capacidad_h(series, capacity_lps=-1.0) is None


def test_integral_exceso_capacidad_uses_exact_segment_crossings():
    # capacidad = 5.0. Serie sube 0->10 en [0,1] (cruza en t=0.5), luego
    # 10->10 en [1,2] (constante). Exceso: triangulo [0.5,1] area=0.5*0.5*5=1.25
    # mas rectangulo [1,2] area=1.0*5=5.0 -> total 6.25 L/s*h -> *3.6 = 22.5 m3
    series = [(0.0, 0.0), (1.0, 10.0), (2.0, 10.0)]
    assert integral_exceso_capacidad_m3(series, capacity_lps=5.0) == pytest.approx(22.5)


def test_integral_exceso_capacidad_is_zero_when_never_exceeded():
    series = [(0.0, 0.0), (1.0, 3.0)]
    assert integral_exceso_capacidad_m3(series, capacity_lps=5.0) == pytest.approx(0.0)


def test_integral_exceso_capacidad_is_none_when_capacity_is_missing_or_non_positive():
    series = [(0.0, 0.0), (1.0, 10.0)]
    assert integral_exceso_capacidad_m3(series, capacity_lps=None) is None
    assert integral_exceso_capacidad_m3(series, capacity_lps=0.0) is None


def test_compute_node_tributary_features_returns_all_six_keys():
    graph = _linear_graph()
    node_series = {
        "upstream": [(0.0, 1.0), (1.0, 3.0)],
        "mid": [(0.0, 2.0), (1.0, 4.0)],
        "downstream": [(0.0, 0.5), (1.0, 0.5)],
    }
    result = compute_node_tributary_features("downstream", node_series, graph, capacity_lps=6.0)
    assert set(result) == {
        "volumen_local_integrado_m3", "volumen_tributario_integrado_m3",
        "duracion_efectiva_tributario_h", "tiempo_primer_pico_tributario_h",
        "tiempo_sobre_capacidad_h", "integral_exceso_capacidad_m3",
    }


def test_compute_node_tributary_features_nulls_only_the_capacity_dependent_pair():
    graph = _linear_graph()
    node_series = {"downstream": [(0.0, 0.0), (1.0, 10.0)]}
    result = compute_node_tributary_features(
        "downstream", node_series, graph, capacity_lps=None,
    )
    assert result["tiempo_sobre_capacidad_h"] is None
    assert result["integral_exceso_capacidad_m3"] is None
    assert result["volumen_local_integrado_m3"] is not None
    assert result["volumen_tributario_integrado_m3"] is not None
    assert result["duracion_efectiva_tributario_h"] is not None
    assert result["tiempo_primer_pico_tributario_h"] is not None
