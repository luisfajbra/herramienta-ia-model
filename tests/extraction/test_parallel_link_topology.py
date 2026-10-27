from __future__ import annotations

import pandas as pd
import pytest

from swmm_resilience.extraction.topology import build_network_graph, compute_topology_features
from swmm_resilience.simulation.swmm_api_io import load_inp
from swmm_resilience.utils import circular_full_flow_lps


_PARALLEL_NETWORK = """\
[OPTIONS]
FLOW_UNITS LPS
FLOW_ROUTING DYNWAVE
[JUNCTIONS]
UP 10 2 0 0 0
DN 0 2 0 0 0
[OUTFALLS]
OUT -1 FREE NO
[CONDUITS]
PIPE_A UP DN 100 0.013 0 0
PIPE_B UP DN 150 0.014 0 0
PIPE_OUT DN OUT 50 0.013 0 0
[XSECTIONS]
PIPE_A CIRCULAR 0.4 0 0 0
PIPE_B CIRCULAR 0.6 0 0 0
PIPE_OUT CIRCULAR 0.5 0 0 0
"""


def test_parallel_conduit_ids_and_capacities_are_preserved(tmp_path):
    inp_path = tmp_path / "parallel.inp"
    inp_path.write_text(_PARALLEL_NETWORK, encoding="utf-8")
    inp = load_inp(inp_path)

    graph, outfalls = build_network_graph(inp)
    edge_records = list(graph.edges(keys=True, data=True))
    static = pd.DataFrame(
        {
            "node_id": ["UP", "DN", "OUT"],
            "elev_fondo": [10.0, 0.0, -1.0],
            "base_inflow_lps": [2.0, 3.0, 0.0],
        }
    )
    result = compute_topology_features(static, inp_path).set_index("node_id")

    assert graph.is_multigraph()
    assert {edge[2] for edge in edge_records if edge[0] == "UP" and edge[1] == "DN"} == {
        "PIPE_A",
        "PIPE_B",
    }
    assert outfalls == {"OUT"}
    assert result.loc["DN", "n_nodos_aguas_arriba"] == 1
    assert result.loc["DN", "q_pico_acum_base"] == 5.0
    expected = circular_full_flow_lps(0.4, 0.1, 0.013) + circular_full_flow_lps(
        0.6, 10.0 / 150.0, 0.014
    )
    assert result.loc["DN", "upstream_capacity_lps"] == pytest.approx(expected)
