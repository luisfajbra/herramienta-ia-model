from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.migrar_dataset_v4_21 import base_node_series


class _FakeTs:
    def __init__(self, data):
        self.data = list(data)


class _FakeInflow:
    def __init__(self, ts_name):
        self.time_series = ts_name


class _FakeInp:
    def __init__(self):
        self._s = {
            "INFLOWS": {("N1", "FLOW"): _FakeInflow("N1_ts")},
            "TIMESERIES": {"N1_ts": _FakeTs([(0.0, 0.0), (1.0, 2.0), (2.0, 0.0)])},
        }

    def __contains__(self, key):
        return key in self._s

    def __getitem__(self, key):
        return self._s[key]


def test_base_node_series_scales_embedded_timeseries_by_factor():
    inp = _FakeInp()
    result = base_node_series(inp, factor=2.5)
    assert result == {"N1": [(0.0, 0.0), (1.0, 5.0), (2.0, 0.0)]}


def test_base_node_series_returns_empty_series_for_nodes_without_inflow():
    inp = _FakeInp()
    result = base_node_series(inp, factor=1.0)
    assert "N2" not in result
