"""main.py --evaluate-validation-shapes: normalized shapes -> absolute-flow CSVs.

The bridge this covers is the one --evaluate-hydrographs never had: a
directory of normalized (time_h, q_norm) profiles is expanded into the
per-node L/s scenario CSVs that run_batch_validation actually consumes.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

import main
from swmm_resilience.validation import hydrograph_batch


def _write_shape(dir_path: Path, name: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{name}.csv").write_text(
        "time_h,q_norm\n0.0000,0.0000\n1.0000,1.0000\n2.0000,0.0000\n",
        newline="\n",
    )


def _static_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node_id": ["N0", "N1", "N2"],
            "base_inflow_lps": [10.0, 4.0, 0.0],  # N2 has no inflow: excluded
        }
    )


@pytest.fixture
def captured_runs(monkeypatch):
    """Record every run_batch_validation call instead of running SWMM."""
    calls = []

    def fake_run(**kwargs):
        csv_dir = Path(kwargs["csv_dir"])
        calls.append(
            {
                "kwargs": kwargs,
                "csv_files": sorted(p.name for p in csv_dir.glob("*.csv")),
                "first_csv": pd.read_csv(sorted(csv_dir.glob("*.csv"))[0]),
            }
        )
        return {
            "n_scenarios": len(list(csv_dir.glob("*.csv"))),
            "classification": {"f1": 0.9},
            "volume": {"nse": 0.8, "error_pct_total": -3.0},
        }

    monkeypatch.setattr(hydrograph_batch, "run_batch_validation", fake_run)
    monkeypatch.setattr(main, "extract_static_features", lambda inp: _static_frame())
    return calls


def test_expands_normalized_shapes_into_absolute_flow_csvs(
    tmp_path, captured_runs, monkeypatch
):
    shapes_dir = tmp_path / "val_shapes"
    _write_shape(shapes_dir, "val_demo")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--evaluate-validation-shapes",
            str(shapes_dir),
            "--factors",
            "1.5,2.5",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    main.main()

    assert len(captured_runs) == 1
    run = captured_runs[0]
    assert run["csv_files"] == ["val_demo_f1.500.csv", "val_demo_f2.500.csv"]

    frame = run["first_csv"]
    assert sorted(frame.columns) == ["node_id", "time", "value_lps"]
    # Only nodes with base_inflow > 0 are written.
    assert sorted(frame["node_id"].unique()) == ["N0", "N1"]
    # q_norm 1.0 at the peak -> base_inflow * factor.
    peak = frame[(frame["node_id"] == "N0") & (frame["time"] == "1:00")]
    assert peak["value_lps"].item() == pytest.approx(10.0 * 1.5)


def test_runs_one_batch_per_shape(tmp_path, captured_runs, monkeypatch):
    shapes_dir = tmp_path / "val_shapes"
    _write_shape(shapes_dir, "val_a")
    _write_shape(shapes_dir, "val_b")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--evaluate-validation-shapes",
            str(shapes_dir),
            "--factors",
            "1.0",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    main.main()

    assert len(captured_runs) == 2
    out_dirs = sorted(Path(c["kwargs"]["out_dir"]).name for c in captured_runs)
    assert out_dirs == ["val_a", "val_b"]


def test_defaults_to_the_unseen_midpoint_factors(tmp_path, captured_runs, monkeypatch):
    shapes_dir = tmp_path / "val_shapes"
    _write_shape(shapes_dir, "val_demo")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--evaluate-validation-shapes",
            str(shapes_dir),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    main.main()

    written = captured_runs[0]["csv_files"]
    # Midpoints between the 25 training factors: 24 unseen values, none of
    # which is a training factor (0.2, 0.4, ... 5.0).
    assert len(written) == 24
    assert "val_demo_f0.300.csv" in written
    assert "val_demo_f1.100.csv" in written
    assert "val_demo_f0.200.csv" not in written


def test_never_reads_the_training_database(tmp_path, captured_runs, monkeypatch):
    """Validation must not depend on having run --persist-sql."""

    def explode(db_path):
        raise AssertionError("--evaluate-validation-shapes must not read SQL")

    shapes_dir = tmp_path / "val_shapes"
    _write_shape(shapes_dir, "val_demo")
    monkeypatch.setattr(main, "load_training_frame", explode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--evaluate-validation-shapes",
            str(shapes_dir),
            "--factors",
            "1.0",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    main.main()

    assert len(captured_runs) == 1


def test_rejects_a_shape_that_is_also_in_the_training_directory(
    tmp_path, captured_runs, monkeypatch, capsys
):
    shapes_dir = tmp_path / "val_shapes"
    # base_shape_rows / load_all_shapes key on the file stem, so reusing a
    # training stem means the shape is in-sample.
    _write_shape(shapes_dir, "shape_late_peak_4h")
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--evaluate-validation-shapes", str(shapes_dir)],
    )

    with pytest.raises(SystemExit) as exit_info:
        main.main()

    assert exit_info.value.code == 2
    assert "shape_late_peak_4h" in capsys.readouterr().err
    assert captured_runs == []


def test_rejects_a_missing_directory(tmp_path, captured_runs, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--evaluate-validation-shapes", str(tmp_path / "nope")],
    )

    with pytest.raises(SystemExit) as exit_info:
        main.main()

    assert exit_info.value.code == 2
    assert "no existe el directorio" in capsys.readouterr().err


def test_rejects_a_directory_with_no_shape_csvs(
    tmp_path, captured_runs, monkeypatch, capsys
):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--evaluate-validation-shapes", str(empty)]
    )

    with pytest.raises(SystemExit) as exit_info:
        main.main()

    assert exit_info.value.code == 2
    assert "no hay CSVs de forma" in capsys.readouterr().err
