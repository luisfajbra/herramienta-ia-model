import matplotlib
import pandas as pd

from swmm_resilience.visualization import flood_map
from swmm_resilience.visualization.loaders import _standardize


def test_standardize_keeps_predicted_volume_separate_from_peak_flow():
    rows = pd.DataFrame(
        {
            "node_id": ["J1"],
            "total_flood_volume_m3": [2.5],
            "flooded": [1],
        }
    )

    out = _standardize(rows, source="ml", inflow_multiplier=1.5)

    assert out.loc[0, "total_flood_volume_m3"] == 2.5
    assert out.loc[0, "peak_flooding_lps"] == 0.0
    assert out.loc[0, "source"] == "ml"
    assert out.loc[0, "inflow_multiplier"] == 1.5


def test_standardize_does_not_force_volume_for_legacy_swmm_peak_data():
    rows = pd.DataFrame(
        {
            "node_id": ["J1"],
            "peak_flooding_lps": [12.0],
            "total_flood_volume_m3": [0.0],
            "flooded": [1],
        }
    )

    out = _standardize(rows, source="swmm", inflow_multiplier=1.5)

    assert out.attrs.get("preferred_flood_metric") is None
    assert out.loc[0, "peak_flooding_lps"] == 12.0
    assert out.loc[0, "total_flood_volume_m3"] == 0.0


def test_metric_choice_respects_preferred_volume_even_when_zero(monkeypatch, tmp_path):
    node_data = pd.DataFrame(
        {
            "node_id": ["J1"],
            "total_flood_volume_m3": [0.0],
            "peak_flooding_lps": [0.0],
            "flooded": [0],
        }
    )
    node_data.attrs["preferred_flood_metric"] = "total_flood_volume_m3"
    labels = []

    monkeypatch.setattr(flood_map, "parse_coordinates", lambda path: {"J1": (0.0, 0.0)})
    monkeypatch.setattr(flood_map, "parse_conduits", lambda path: [])

    original_colorbar = flood_map.plt.Figure.colorbar

    def capture_colorbar(self, *args, **kwargs):
        cbar = original_colorbar(self, *args, **kwargs)
        original_set_label = cbar.set_label

        def set_label(label, *label_args, **label_kwargs):
            labels.append(label)
            return original_set_label(label, *label_args, **label_kwargs)

        cbar.set_label = set_label
        return cbar

    monkeypatch.setattr(flood_map.plt.Figure, "colorbar", capture_colorbar)

    flood_map.plot_flood_map(
        node_data=node_data,
        inp_path=tmp_path / "network.inp",
        output_path=tmp_path / "map.png",
        title="Mapa",
    )

    assert labels[-1] == "Total Flood Volume (m3)"


def test_plot_flood_map_draws_runtime_annotation(monkeypatch, tmp_path):
    node_data = pd.DataFrame(
        {
            "node_id": ["J1"],
            "total_flood_volume_m3": [2.5],
            "peak_flooding_lps": [0.0],
            "flooded": [1],
        }
    )
    node_data.attrs["preferred_flood_metric"] = "total_flood_volume_m3"

    monkeypatch.setattr(flood_map, "parse_coordinates", lambda path: {"J1": (0.0, 0.0)})
    monkeypatch.setattr(flood_map, "parse_conduits", lambda path: [])

    texts = []
    original_text = matplotlib.axes.Axes.text

    def spy_text(self, *args, **kwargs):
        if len(args) >= 3:
            texts.append(args[2])
        return original_text(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", spy_text)

    out = tmp_path / "map.png"
    flood_map.plot_flood_map(
        node_data=node_data,
        inp_path=tmp_path / "network.inp",
        output_path=out,
        title="Mapa",
        runtime_text="Tiempo de cómputo: 1.85 s",
    )

    assert "Tiempo de cómputo: 1.85 s" in texts
    assert out.exists() and out.stat().st_size > 0


def test_generate_flood_map_draws_runtime_annotation(monkeypatch, tmp_path):
    from types import SimpleNamespace

    fake_inp = {"COORDINATES": {"J1": SimpleNamespace(x=0.0, y=0.0)}}
    monkeypatch.setattr(flood_map, "load_inp", lambda p: fake_inp)
    vol_data = pd.DataFrame({"node_id": ["J1"], "vol_inundacion_m3": [2.5]})

    texts = []
    original_text = matplotlib.axes.Axes.text

    def spy_text(self, *args, **kwargs):
        if len(args) >= 3:
            texts.append(args[2])
        return original_text(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", spy_text)

    out = tmp_path / "factor_map.png"
    flood_map.generate_flood_map(
        tmp_path / "net.inp",
        vol_data,
        3.0,
        out,
        runtime_text="Tiempo de cómputo: 1.85 s",
    )

    assert "Tiempo de cómputo: 1.85 s" in texts
    assert out.exists() and out.stat().st_size > 0
