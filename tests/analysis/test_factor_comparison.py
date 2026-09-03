from types import SimpleNamespace

import pandas as pd

from swmm_resilience.analysis import factor_comparison


def test_generate_factor_comparisons_processes_every_factor(
    monkeypatch, tmp_path
):
    frame = pd.DataFrame(
        {
            "node_id": ["1C", "2C", "1C", "2C"],
            "factor_mult": [1.0, 1.0, 2.0, 2.0],
            "vol_inundacion_m3": [1.0, 2.0, 3.0, 4.0],
            "inunda": [1, 1, 1, 1],
        }
    )
    config = SimpleNamespace()
    prediction_calls = []
    plot_calls = []
    flooded_plot_calls = []

    def fake_predict(factor, config_arg, models_dir):
        prediction_calls.append(factor)
        return pd.DataFrame(
            {
                "node_id": ["1C", "2C"],
                "inunda_pred": [1, 1],
                "vol_pred_m3": [factor * 10, factor * 20],
            }
        )

    def fake_plot(df, output_dir, factor):
        plot_calls.append((df.copy(), output_dir, factor))
        return (
            output_dir / f"volume_by_node_factor_{factor:.2f}.png",
            output_dir / f"parity_factor_{factor:.2f}.png",
        )

    def fake_flooded_plot(df, output_dir, factor):
        flooded_plot_calls.append((df.copy(), output_dir, factor))
        return (
            output_dir
            / f"volume_by_node_flooded_swmm_factor_{factor:.2f}.png"
        )

    monkeypatch.setattr(factor_comparison, "predict_network", fake_predict)
    monkeypatch.setattr(factor_comparison, "plot_factor_comparison", fake_plot)
    monkeypatch.setattr(
        factor_comparison,
        "plot_flooded_swmm_node_profile",
        fake_flooded_plot,
    )
    out_dir = tmp_path / "plots"

    paths = factor_comparison.generate_factor_comparisons(
        frame=frame,
        config=config,
        models_dir=tmp_path / "models",
        output_dir=out_dir,
    )

    assert prediction_calls == [1.0, 2.0]
    assert len(paths) == 6
    assert [call[2] for call in plot_calls] == [1.0, 2.0]
    assert [call[2] for call in flooded_plot_calls] == [1.0, 2.0]
    assert plot_calls[0][0]["vol_swmm_m3"].tolist() == [1.0, 2.0]
    assert plot_calls[0][0]["vol_pred_m3"].tolist() == [10.0, 20.0]
