import sys
from pathlib import Path

import pandas as pd

import main


def base_shape_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [1, 1, 2, 2],
            "node_id": ["N0", "N1", "N0", "N1"],
            "factor_mult": [1.0, 1.0, 2.0, 2.0],
            "shape_id": ["base", "base", "base", "base"],
            "inunda": [0, 1, 1, 1],
            "vol_inundacion_m3": [0.0, 5.0, 7.0, 9.0],
        }
    )


def test_resilience_curve_reads_the_frame_from_sql(monkeypatch):
    load_calls = []
    curve_calls = []

    def fake_load(db_path):
        load_calls.append(Path(db_path))
        return base_shape_frame()

    def fake_curve(df, factors, config, models_dir):
        curve_calls.append((df.copy(), list(factors), config))
        return pd.DataFrame(
            {
                "factor": list(factors),
                "resilience_swmm": [1.0] * len(factors),
                "resilience_ml": [1.0] * len(factors),
            }
        )

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(main, "compute_resilience_curve", fake_curve)
    monkeypatch.setattr(main, "plot_resilience_curve", lambda result, out: None)
    monkeypatch.setattr(sys, "argv", ["main.py", "--resilience-curve"])

    main.main()

    config = curve_calls[0][2]
    assert load_calls == [config.dataset.db_path]
    assert curve_calls[0][1] == [1.0, 2.0]


def test_flood_volume_curve_reads_the_frame_from_sql(monkeypatch):
    load_calls = []
    curve_calls = []

    def fake_load(db_path):
        load_calls.append(Path(db_path))
        return base_shape_frame()

    def fake_curve(df, factors, config, models_dir):
        curve_calls.append((df.copy(), list(factors), config))
        return pd.DataFrame(
            {
                "factor": list(factors),
                "vol_total_swmm": [1.0] * len(factors),
                "vol_total_ml": [1.0] * len(factors),
            }
        )

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(main, "compute_flood_volume_curve", fake_curve)
    monkeypatch.setattr(main, "plot_flood_volume_curve", lambda result, out: None)
    monkeypatch.setattr(main, "plot_flood_volume_combined", lambda result, out: None)
    monkeypatch.setattr(sys, "argv", ["main.py", "--flood-volume-curve"])

    main.main()

    config = curve_calls[0][2]
    assert load_calls == [config.dataset.db_path]
    assert curve_calls[0][1] == [1.0, 2.0]
