"""Final-review fix (Finding 1): the 3 main.py call sites of
run_batch_validation that feed CODEBASE-GENERATED CSVs (via
write_shape_validation_csv, which writes the RAW/unclosed shape) must always
pass csv_inflow_closure_mode="close" -- never the config-driven value -- so a
config.yaml with the (default) "strict" mode does not make these CLI
workflows abort on an unclosed synthetic shape such as "base".

Only --evaluate-hydrographs (the genuine user-uploaded-CSV entry point) is
meant to respect validation.csv_inflow_closure_mode from config.yaml.

These are full main.main() behavioral tests: run_batch_validation is
monkeypatched (no real SWMM/pyswmm/joblib run), but the CLI wiring, config
loading (config.yaml, whose validation.csv_inflow_closure_mode is "strict")
and the shape-generation loop above each call site all run for real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

import main
from swmm_resilience.validation import hydrograph_batch


def _training_frame() -> pd.DataFrame:
    """Minimal frame satisfying load_training_frame's contract for the
    --evaluate-shapes / --evaluate-generalization branches: only
    node_id/base_inflow_lps is read out of it before the shape loop."""
    return pd.DataFrame(
        {
            "node_id": ["1C", "2C"],
            "base_inflow_lps": [2.0, 3.0],
        }
    )


@pytest.fixture
def captured_close_mode_calls(monkeypatch):
    """Record every run_batch_validation call's csv_inflow_closure_mode
    instead of actually running SWMM/joblib models."""
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {
            "n_scenarios": 1,
            "classification": {"f1": 0.9},
            "volume": {"nse": 0.8, "error_pct_total": -3.0},
            "volume_flooded_only": {},
            "pr_auc": None,
            "per_scenario": [],
            "timings": [],
            "summary_csv_path": "summary.csv",
            "scenario_totals_csv_path": "totals.csv",
            "timings_csv_path": "timings.csv",
            "metrics_per_scenario_csv_path": "metrics.csv",
        }

    monkeypatch.setattr(hydrograph_batch, "run_batch_validation", fake_run)
    return calls


def test_config_yaml_default_closure_mode_is_strict():
    """Sanity check the premise: config.yaml's real default is 'strict', so a
    behavioral test that loads it for real is actually exercising the bug
    Finding 1 describes, not a vacuous case."""
    config = main.load_config("config.yaml")
    assert config.validation.csv_inflow_closure_mode == "strict"


def test_evaluate_shapes_always_closes_generated_shapes(
    monkeypatch, captured_close_mode_calls
):
    monkeypatch.setattr(main, "load_training_frame", lambda db_path: _training_frame())
    monkeypatch.setattr(sys, "argv", ["main.py", "--evaluate-shapes"])

    main.main()

    assert captured_close_mode_calls, "run_batch_validation was never called"
    modes = {kwargs["csv_inflow_closure_mode"] for kwargs in captured_close_mode_calls}
    assert modes == {"close"}, (
        f"--evaluate-shapes must always pass csv_inflow_closure_mode='close' "
        f"for generated shapes (never the config-driven strict default): {modes}"
    )


def test_evaluate_generalization_always_closes_generated_shapes(
    monkeypatch, captured_close_mode_calls
):
    monkeypatch.setattr(main, "load_training_frame", lambda db_path: _training_frame())
    monkeypatch.setattr(sys, "argv", ["main.py", "--evaluate-generalization"])

    main.main()

    assert captured_close_mode_calls, "run_batch_validation was never called"
    modes = {kwargs["csv_inflow_closure_mode"] for kwargs in captured_close_mode_calls}
    assert modes == {"close"}, (
        f"--evaluate-generalization must always pass "
        f"csv_inflow_closure_mode='close' for generated shapes: {modes}"
    )


def test_evaluate_validation_shapes_always_closes_generated_shapes(
    monkeypatch, captured_close_mode_calls, tmp_path
):
    shapes_dir = tmp_path / "val_shapes"
    shapes_dir.mkdir()
    (shapes_dir / "val_demo.csv").write_text(
        # Deliberately does NOT end at q_norm=0: this is exactly the shape
        # of CSV write_shape_validation_csv produces (raw, unclosed).
        "time_h,q_norm\n0.0000,0.0000\n1.0000,1.0000\n2.0000,0.4000\n",
        newline="\n",
    )
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

    assert captured_close_mode_calls, "run_batch_validation was never called"
    modes = {kwargs["csv_inflow_closure_mode"] for kwargs in captured_close_mode_calls}
    assert modes == {"close"}, (
        f"--evaluate-validation-shapes must always pass "
        f"csv_inflow_closure_mode='close' for generated shapes: {modes}"
    )


def test_evaluate_hydrographs_still_respects_the_config_driven_mode(
    monkeypatch, captured_close_mode_calls, tmp_path
):
    """The 4th, genuine user-CSV call site must NOT be touched by Finding 1's
    fix: it should keep reading validation.csv_inflow_closure_mode from
    config.yaml (here: 'strict')."""
    csv_dir = tmp_path / "user_csvs"
    csv_dir.mkdir()
    base_inp = tmp_path / "base.inp"
    base_inp.write_text("dummy", encoding="utf-8")
    clf = tmp_path / "clf.joblib"
    reg = tmp_path / "reg.joblib"
    clf.touch()
    reg.touch()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--evaluate-hydrographs",
            str(csv_dir),
            "--base-inp",
            str(base_inp),
            "--clf-path",
            str(clf),
            "--reg-path",
            str(reg),
        ],
    )

    main.main()

    assert len(captured_close_mode_calls) == 1
    assert captured_close_mode_calls[0]["csv_inflow_closure_mode"] == "strict"
