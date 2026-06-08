from swmm_resilience.visualization import hydrograph


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
    assert any("J2" in t for t in titles), f"Expected J2 in title, got: {titles}"
