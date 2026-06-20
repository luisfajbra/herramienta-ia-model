import matplotlib.colors as mcolors
import pandas as pd

from swmm_resilience.visualization import model_comparison


def test_plot_flooded_swmm_node_profile_excludes_zero_swmm_nodes(
    monkeypatch, tmp_path
):
    df = pd.DataFrame(
        {
            "node_id": ["1C", "2C", "3C"],
            "vol_swmm_m3": [12.0, 0.0, 3.0],
            "vol_pred_m3": [10.0, 8.0, 5.0],
        }
    )
    figures = []
    original_subplots = model_comparison.plt.subplots

    def capture_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        figures.append(ax)
        return fig, ax

    monkeypatch.setattr(model_comparison.plt, "subplots", capture_subplots)

    path = model_comparison.plot_flooded_swmm_node_profile(
        df, tmp_path, 1.0
    )

    assert path == (
        tmp_path / "volume_by_node_flooded_swmm_factor_1.00.png"
    )
    assert path.exists() and path.stat().st_size > 0
    assert [tick.get_text() for tick in figures[0].get_xticklabels()] == [
        "1",
        "3",
    ]
    assert [bar.get_height() for bar in figures[0].containers[0]] == [12.0, 3.0]
    assert [bar.get_height() for bar in figures[0].containers[1]] == [10.0, 5.0]


def test_plot_factor_comparison_writes_profile_and_parity(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "node_id": ["1I", "2O"],
            "vol_swmm_m3": [12.0, 3.0],
            "vol_pred_m3": [10.0, 5.0],
        }
    )
    figures = []
    original_subplots = model_comparison.plt.subplots

    def capture_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        figures.append(ax)
        return fig, ax

    monkeypatch.setattr(model_comparison.plt, "subplots", capture_subplots)

    paths = model_comparison.plot_factor_comparison(df, tmp_path, 1.0)

    assert paths == (
        tmp_path / "volume_by_node_factor_1.00.png",
        tmp_path / "parity_factor_1.00.png",
    )
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert [
        container[0].get_facecolor() for container in figures[0].containers
    ] == [
        mcolors.to_rgba("#2176ae"),
        mcolors.to_rgba("#f28e2b"),
    ]
    assert [tick.get_text() for tick in figures[0].get_xticklabels()] == ["1", "2"]
    assert figures[0].get_xlabel() == "Node ID"
    assert figures[0].get_ylabel() == "Flood Volume (m³)"
    assert figures[1].get_xlabel() == "SWMM Flood Volume (m³)"
    assert figures[1].get_ylabel() == "XGBoost Flood Volume (m³)"
