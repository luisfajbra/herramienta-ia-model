from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.migrar_dataset_v4_21 import base_node_series
from scripts.migrar_dataset_v4_21 import build_dataset_manifest, validate_dataset_manifest
import json
import pytest


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


def test_v4_manifest_binds_dataset_network_and_hydrograph_sources(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "v4.csv"
    inp = tmp_path / "network.inp"
    shapes = tmp_path / "shapes"
    shapes.mkdir()
    shape = shapes / "storm.csv"
    source.write_text("source", encoding="utf-8")
    output.write_text("output", encoding="utf-8")
    inp.write_text("network", encoding="utf-8")
    shape.write_text("0,0\\n1,1", encoding="utf-8")
    manifest = build_dataset_manifest(
        source_path=source,
        output_path=output,
        inp_path=inp,
        hydrograph_shapes_dir=shapes,
        row_count=1,
    )
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    verified = validate_dataset_manifest(manifest_path, output, inp, source, shapes)

    assert verified["hydrograph_shapes"]["storm"]["sha256"] == manifest[
        "hydrograph_shapes"
    ]["storm"]["sha256"]


def test_v4_manifest_rejects_dataset_from_a_different_network(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "v4.csv"
    inp = tmp_path / "network.inp"
    shapes = tmp_path / "shapes"
    shapes.mkdir()
    source.write_text("source", encoding="utf-8")
    output.write_text("output", encoding="utf-8")
    inp.write_text("network A", encoding="utf-8")
    manifest = build_dataset_manifest(
        source_path=source,
        output_path=output,
        inp_path=inp,
        hydrograph_shapes_dir=shapes,
        row_count=1,
    )
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    inp.write_text("network B", encoding="utf-8")

    with pytest.raises(ValueError, match="network_inp SHA-256"):
        validate_dataset_manifest(manifest_path, output, inp, source, shapes)
