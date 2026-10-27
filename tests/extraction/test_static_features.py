from __future__ import annotations

import pytest

from swmm_resilience.extraction.static_features import extract_static_features


_INFLOW_INP = """\
[OPTIONS]
FLOW_UNITS LPS
FLOW_ROUTING DYNWAVE
[JUNCTIONS]
J1 10 2 0 0 0
[OUTFALLS]
O1 -1 FREE NO
[CONDUITS]
C1 J1 O1 100 0.013 0 0 0 0
[XSECTIONS]
C1 CIRCULAR 0.4 0 0 0 1
[INFLOWS]
J1 FLOW TS FLOW 1 2 5
[TIMESERIES]
TS 0:00 10
TS 0:10 12
"""


def test_static_features_use_effective_peak_flow_not_raw_series_peak(tmp_path):
    inp_path = tmp_path / "effective_inflow.inp"
    inp_path.write_text(_INFLOW_INP, encoding="utf-8")

    features = extract_static_features(inp_path).set_index("node_id")

    assert features.loc["J1", "base_inflow_lps"] == pytest.approx(29.0)
