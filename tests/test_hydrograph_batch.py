"""Tests for swmm_resilience.validation.hydrograph_batch.run_batch_validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from swmm_resilience.validation import hydrograph_batch
from swmm_resilience.validation.hydrograph_batch import run_batch_validation
from swmm_resilience.validation.hydrograph_csv import HydrographScenario


# ---------------------------------------------------------------------------
# Shared fake objects
# ---------------------------------------------------------------------------

_NODES = ["J1", "J2"]

_FAKE_SCENARIO = HydrographScenario(
    scenario_id="sc001",
    node_series={
        "J1": [(0.0, 0.0), (1.0, 50.0)],
        "J2": [(0.0, 0.0), (1.0, 80.0)],
    },
    time_grid_hours=[0.0, 1.0],
    last_time_hours=1.0,
)

_FAKE_PRED_DF = pd.DataFrame(
    {
        "node_id": ["J1", "J2"],
        "inunda_pred": [0, 1],
        "vol_pred_m3": [0.0, 5.0],
    }
)

_FAKE_SWMM_FLOOD_DF = pd.DataFrame(
    {
        "node_id": ["J1", "J2"],
        "vol_swmm_m3": [0.0, 4.0],
        "inunda_swmm": [0, 1],
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(tmp_path: Path, name: str = "sc001.csv") -> Path:
    """Write a minimal (but well-formed) hydrograph CSV."""
    rows = [
        {"node_id": "J1", "time": "0:00", "value_lps": 0.0},
        {"node_id": "J1", "time": "1:00", "value_lps": 50.0},
        {"node_id": "J2", "time": "0:00", "value_lps": 0.0},
        {"node_id": "J2", "time": "1:00", "value_lps": 80.0},
    ]
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _patch_all(monkeypatch, tmp_path: Path):
    """Monkeypatch every external dependency of run_batch_validation."""

    # load_scenario → fake scenario
    monkeypatch.setattr(
        hydrograph_batch,
        "load_scenario",
        lambda csv_path, expected_nodes: _FAKE_SCENARIO,
    )

    # write_scenario_inp → fake .inp path (file must exist for _run_swmm)
    fake_inp = tmp_path / "inp" / "sc001.inp"
    fake_inp.parent.mkdir(parents=True, exist_ok=True)
    fake_inp.touch()

    monkeypatch.setattr(
        hydrograph_batch,
        "write_scenario_inp",
        lambda base_inp_path, scenario, out_dir: fake_inp,
    )

    # _run_swmm → create a fake .rpt and return its path
    fake_rpt = fake_inp.with_suffix(".rpt")
    fake_rpt.touch()

    monkeypatch.setattr(
        hydrograph_batch,
        "_run_swmm",
        lambda scenario_inp_path: fake_rpt,
    )

    # _build_swmm_df → fake SWMM result DataFrame
    monkeypatch.setattr(
        hydrograph_batch,
        "_build_swmm_df",
        lambda scenario, rpt_path, flood_threshold_m3: _FAKE_SWMM_FLOOD_DF.copy(),
    )

    # predict_scenario → fake prediction DataFrame
    monkeypatch.setattr(
        hydrograph_batch,
        "predict_scenario",
        lambda scenario, clf_path, reg_path, flood_threshold_m3, inp_path: _FAKE_PRED_DF.copy(),
    )

    # Plot functions → no-ops (return empty list)
    monkeypatch.setattr(
        hydrograph_batch, "plot_parity_nodes",
        lambda df, out_dir, scenario_id: [],
    )
    monkeypatch.setattr(
        hydrograph_batch, "plot_parity_aggregated",
        lambda df, out_dir, scenario_id: [],
    )
    monkeypatch.setattr(
        hydrograph_batch, "plot_node_profiles",
        lambda df, out_dir, scenario_id: [],
    )

    return fake_inp, fake_rpt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunBatchValidationReturnsSummaryKeys:
    """run_batch_validation must return a dict with the required top-level keys."""

    def test_run_batch_validation_returns_summary_keys(self, monkeypatch, tmp_path):
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        _make_csv(csv_dir)

        _patch_all(monkeypatch, tmp_path)

        result = run_batch_validation(
            csv_dir=csv_dir,
            base_inp_path=tmp_path / "base.inp",
            clf_path=tmp_path / "clf.joblib",
            reg_path=tmp_path / "reg.joblib",
            flood_threshold_m3=1.0,
            out_dir=tmp_path / "out",
        )

        required_keys = {"n_scenarios", "classification", "volume", "per_node_r2", "summary_csv_path"}
        assert required_keys.issubset(set(result.keys())), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )


class TestRunBatchValidationNScenarios:
    """n_scenarios must equal the number of CSV files found in csv_dir."""

    def test_run_batch_validation_n_scenarios_one(self, monkeypatch, tmp_path):
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        _make_csv(csv_dir, "sc001.csv")

        _patch_all(monkeypatch, tmp_path)

        result = run_batch_validation(
            csv_dir=csv_dir,
            base_inp_path=tmp_path / "base.inp",
            clf_path=tmp_path / "clf.joblib",
            reg_path=tmp_path / "reg.joblib",
            flood_threshold_m3=1.0,
            out_dir=tmp_path / "out",
        )

        assert result["n_scenarios"] == 1

    def test_run_batch_validation_n_scenarios_three(self, monkeypatch, tmp_path):
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        for name in ("sc001.csv", "sc002.csv", "sc003.csv"):
            _make_csv(csv_dir, name)

        _patch_all(monkeypatch, tmp_path)

        result = run_batch_validation(
            csv_dir=csv_dir,
            base_inp_path=tmp_path / "base.inp",
            clf_path=tmp_path / "clf.joblib",
            reg_path=tmp_path / "reg.joblib",
            flood_threshold_m3=1.0,
            out_dir=tmp_path / "out",
        )

        assert result["n_scenarios"] == 3


class TestRunBatchValidationSavesSummaryCsv:
    """summary_csv_path must point to a file that exists after the run."""

    def test_run_batch_validation_saves_summary_csv(self, monkeypatch, tmp_path):
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        _make_csv(csv_dir)

        _patch_all(monkeypatch, tmp_path)

        out_dir = tmp_path / "out"
        result = run_batch_validation(
            csv_dir=csv_dir,
            base_inp_path=tmp_path / "base.inp",
            clf_path=tmp_path / "clf.joblib",
            reg_path=tmp_path / "reg.joblib",
            flood_threshold_m3=1.0,
            out_dir=out_dir,
        )

        csv_path = result["summary_csv_path"]
        assert isinstance(csv_path, Path), "summary_csv_path must be a Path"
        assert csv_path.exists(), f"summary CSV not found at {csv_path}"


class TestRunBatchValidationEmptyDir:
    """When csv_dir contains no CSV files, n_scenarios must be 0 and
    the returned metric dicts must still be present (but represent empty data)."""

    def test_run_batch_validation_empty_dir(self, monkeypatch, tmp_path):
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        # no .csv files written

        _patch_all(monkeypatch, tmp_path)

        result = run_batch_validation(
            csv_dir=csv_dir,
            base_inp_path=tmp_path / "base.inp",
            clf_path=tmp_path / "clf.joblib",
            reg_path=tmp_path / "reg.joblib",
            flood_threshold_m3=1.0,
            out_dir=tmp_path / "out",
        )

        assert result["n_scenarios"] == 0, "Empty dir should yield n_scenarios=0"
        assert isinstance(result["classification"], dict), "classification must be a dict"
        assert isinstance(result["volume"], dict), "volume must be a dict"
        assert isinstance(result["per_node_r2"], dict), "per_node_r2 must be a dict"
        assert len(result["per_node_r2"]) == 0, "per_node_r2 must be empty for empty dir"

        # summary CSV must still be written (empty frame)
        assert result["summary_csv_path"].exists(), "summary CSV must exist even for empty run"


class TestRunBatchValidationPerNodeR2Keys:
    """per_node_r2 keys must include the node IDs from the processed scenarios."""

    def test_run_batch_validation_per_node_r2_keys(self, monkeypatch, tmp_path):
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        # Two CSV files so each node appears twice → R² can be computed
        _make_csv(csv_dir, "sc001.csv")
        _make_csv(csv_dir, "sc002.csv")

        _patch_all(monkeypatch, tmp_path)

        result = run_batch_validation(
            csv_dir=csv_dir,
            base_inp_path=tmp_path / "base.inp",
            clf_path=tmp_path / "clf.joblib",
            reg_path=tmp_path / "reg.joblib",
            flood_threshold_m3=1.0,
            out_dir=tmp_path / "out",
        )

        per_node_r2 = result["per_node_r2"]
        assert isinstance(per_node_r2, dict), "per_node_r2 must be a dict"

        # Node IDs come from the fake comparison DataFrames built during the run
        expected_nodes = set(_NODES)
        actual_nodes = set(per_node_r2.keys())
        assert expected_nodes.issubset(actual_nodes), (
            f"per_node_r2 missing nodes {expected_nodes - actual_nodes}"
        )
