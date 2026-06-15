from swmm_resilience.visualization import hydrograph
from swmm_resilience.validation.hydrograph_csv import HydrographScenario


def test_plot_hydrograph_selects_peak_node_and_writes_png(monkeypatch, tmp_path):
    fake_profiles = {
        "J1": {"timeseries": "J1", "points": [(0.0, 1.0), (30.0, 5.0), (60.0, 2.0)], "mfactor": 1.0, "baseline": 0.0},
        "J2": {"timeseries": "J2", "points": [(0.0, 2.0), (30.0, 10.0), (60.0, 3.0)], "mfactor": 1.0, "baseline": 0.0},
    }
    monkeypatch.setattr(hydrograph, "get_node_inflow_profiles", lambda inp: fake_profiles)
    monkeypatch.setattr(hydrograph, "load_inp", lambda path: {})

    titles = []
    original_subplots = hydrograph.plt.subplots

    def capturing_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        original_set_title = ax.set_title
        def spy_set_title(t, **kw):
            titles.append(t)
            return original_set_title(t, **kw)
        ax.set_title = spy_set_title
        return fig, ax

    monkeypatch.setattr(hydrograph.plt, "subplots", capturing_subplots)

    output = hydrograph.plot_hydrograph(tmp_path / "network.inp", tmp_path / "hydro.png")

    assert output == tmp_path / "hydro.png"
    assert (tmp_path / "hydro.png").exists()
    assert (tmp_path / "hydro.png").stat().st_size > 0
    assert any(t == "Inflow Hydrograph - Node J2 (Qx1)" for t in titles)


def test_plot_hydrograph_removes_numeric_io_suffix_from_visible_node(monkeypatch, tmp_path):
    fake_profiles = {
        "208O": {
            "timeseries": "208O",
            "points": [(0.0, 1.0), (30.0, 5.0)],
            "mfactor": 1.0,
            "baseline": 0.0,
        },
    }
    monkeypatch.setattr(hydrograph, "get_node_inflow_profiles", lambda inp: fake_profiles)
    monkeypatch.setattr(hydrograph, "load_inp", lambda path: {})

    titles = []
    original_subplots = hydrograph.plt.subplots

    def capturing_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        original_set_title = ax.set_title

        def spy_set_title(title, **title_kwargs):
            titles.append(title)
            return original_set_title(title, **title_kwargs)

        ax.set_title = spy_set_title
        return fig, ax

    monkeypatch.setattr(hydrograph.plt, "subplots", capturing_subplots)

    hydrograph.plot_hydrograph(tmp_path / "network.inp", tmp_path / "hydro.png")

    assert titles == ["Inflow Hydrograph - Node 208 (Qx1)"]


def test_plot_scenario_hydrograph_selects_peak_node_and_converts_time_to_minutes(
    monkeypatch, tmp_path
):
    scenario = HydrographScenario(
        scenario_id="storm_a",
        node_series={
            "75C": [(0.0, 0.0), (0.5, 20.0), (1.0, 0.0)],
            "87I": [(0.0, 0.0), (0.5, 40.0), (1.0, 0.0)],
        },
        time_grid_hours=[0.0, 0.5, 1.0],
        last_time_hours=1.0,
    )
    captured = {}
    original_subplots = hydrograph.plt.subplots

    def capturing_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        original_plot = ax.plot
        original_set_title = ax.set_title

        def spy_plot(x, y, *plot_args, **plot_kwargs):
            captured["times"] = list(x)
            captured["flows"] = list(y)
            return original_plot(x, y, *plot_args, **plot_kwargs)

        def spy_set_title(title, **title_kwargs):
            captured["title"] = title
            return original_set_title(title, **title_kwargs)

        ax.plot = spy_plot
        ax.set_title = spy_set_title
        return fig, ax

    monkeypatch.setattr(hydrograph.plt, "subplots", capturing_subplots)
    output_path = tmp_path / "hydrograph_storm_a.png"

    result = hydrograph.plot_scenario_hydrograph(scenario, output_path)

    assert result == output_path
    assert captured["times"] == [0.0, 30.0, 60.0]
    assert captured["flows"] == [0.0, 40.0, 0.0]
    assert captured["title"] == "Inflow Hydrograph - Node 87 (storm_a)"
    assert output_path.exists()
    assert output_path.stat().st_size > 0
