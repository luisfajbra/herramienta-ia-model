import sys
from pathlib import Path

import pandas as pd

import main


def test_factor_comparison_cli_generates_plots_in_metrics_directory(
    monkeypatch, capsys
):
    calls = []
    load_calls = []
    expected_paths = [
        Path("outputs/metrics/factor_comparison/volume_by_node_factor_1.00.png"),
        Path("outputs/metrics/factor_comparison/parity_factor_1.00.png"),
    ]
    frame = pd.DataFrame(
        {
            "node_id": ["1C"],
            "factor_mult": [1.0],
            "shape_id": ["base"],
            "vol_inundacion_m3": [1.0],
            "inunda": [1],
        }
    )

    def fake_load(db_path):
        load_calls.append(Path(db_path))
        return frame

    def fake_generate(frame, config, models_dir, output_dir):
        calls.append((frame, config, models_dir, output_dir))
        return expected_paths

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(main, "generate_factor_comparisons", fake_generate)
    monkeypatch.setattr(sys, "argv", ["main.py", "--factor-comparison"])

    main.main()

    assert len(calls) == 1
    frame_arg, config, models_dir, output_dir = calls[0]
    assert load_calls == [config.dataset.db_path]
    assert frame_arg.equals(frame)
    assert models_dir == main.MODELS_DIR
    assert output_dir == main.METRICS_DIR / "factor_comparison"
    output = capsys.readouterr().out
    assert all(str(path) in output for path in expected_paths)
