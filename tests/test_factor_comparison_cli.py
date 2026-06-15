import sys
from pathlib import Path

import main


def test_factor_comparison_cli_generates_plots_in_metrics_directory(
    monkeypatch, capsys
):
    calls = []
    expected_paths = [
        Path("outputs/metrics/factor_comparison/volume_by_node_factor_1.00.png"),
        Path("outputs/metrics/factor_comparison/parity_factor_1.00.png"),
    ]

    def fake_generate(dataset_path, config, models_dir, output_dir):
        calls.append((dataset_path, config, models_dir, output_dir))
        return expected_paths

    monkeypatch.setattr(main, "generate_factor_comparisons", fake_generate)
    monkeypatch.setattr(sys, "argv", ["main.py", "--factor-comparison"])

    main.main()

    assert len(calls) == 1
    dataset_path, config, models_dir, output_dir = calls[0]
    assert dataset_path == config.dataset.output_path
    assert models_dir == main.MODELS_DIR
    assert output_dir == main.METRICS_DIR / "factor_comparison"
    output = capsys.readouterr().out
    assert all(str(path) in output for path in expected_paths)
