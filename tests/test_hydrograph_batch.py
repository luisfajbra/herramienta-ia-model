"""Tests for swmm_resilience.validation.hydrograph_batch.run_batch_validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from swmm_resilience.validation import hydrograph_batch
from swmm_resilience.validation.hydrograph_batch import run_batch_validation
from swmm_resilience.validation.hydrograph_csv import HydrographScenario
from swmm_resilience.visualization import model_comparison


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

_FAKE_PRED_FULL_DF = pd.DataFrame(
    {
        "node_id": ["J1", "J2"],
        "inunda_pred": [0, 1],
        "prob_inunda": [0.1, 0.9],
        "vol_pred_m3": [0.0, 5.0],
        "extrapolated": [False, True],
    }
)

_FAKE_SWMM_FLOOD_DF = pd.DataFrame(
    {
        "node_id": ["J1", "J2"],
        "inunda_swmm": [0, 1],
        "vol_swmm_m3": [0.0, 4.0],
    }
)


class _FakePredictor:
    node_ids = list(_NODES)
    model_load_s = 0.01
    static_features_s = 0.02

    def predict_timed(self, scenario):
        return _FAKE_PRED_FULL_DF.copy(), {
            "t_features_s": 0.001,
            "t_inference_s": 0.002,
        }


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
    monkeypatch.setattr(
        hydrograph_batch,
        "_validate_base_inp",
        lambda base_inp_path, clf_path, allow_inp_mismatch: None,
    )
    monkeypatch.setattr(
        hydrograph_batch,
        "_expected_nodes_from_inp",
        lambda base_inp_path: set(_NODES),
    )
    monkeypatch.setattr(
        hydrograph_batch,
        "_make_predictor",
        lambda clf_path, reg_path, inp_path, factor_range: _FakePredictor(),
    )

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
        lambda base_inp_path, scenario, out_dir, drain_down_hours=6.0: fake_inp,
    )

    # _run_swmm → create a fake .rpt and return its path
    fake_rpt = fake_inp.with_suffix(".rpt")
    fake_rpt.touch()

    monkeypatch.setattr(
        hydrograph_batch,
        "_run_swmm",
        lambda scenario_inp_path: fake_rpt,
    )

    # _build_swmm_df → fake SWMM result DataFrame (new node-id-list signature)
    monkeypatch.setattr(
        hydrograph_batch,
        "_build_swmm_df",
        lambda all_node_ids, rpt_path, flood_threshold_m3: _FAKE_SWMM_FLOOD_DF.copy(),
    )

    monkeypatch.setattr(
        hydrograph_batch,
        "_read_continuity_error",
        lambda rpt_path: 0.5,
    )

    # Plot functions → no-ops
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
    monkeypatch.setattr(
        hydrograph_batch,
        "plot_scenario_flood_maps",
        lambda df, inp_path, out_dir, scenario_id, **kwargs: (),
        raising=False,
    )
    monkeypatch.setattr(
        hydrograph_batch,
        "plot_scenario_hydrograph",
        lambda scenario, output_path: output_path,
        raising=False,
    )
    monkeypatch.setattr(
        hydrograph_batch,
        "plot_totals_comparison",
        lambda totals_df, out_path: out_path,
        raising=False,
    )

    return fake_inp, fake_rpt


def _run(tmp_path, csv_dir, **kwargs):
    return run_batch_validation(
        csv_dir=csv_dir,
        base_inp_path=tmp_path / "base.inp",
        clf_path=tmp_path / "clf.joblib",
        reg_path=tmp_path / "reg.joblib",
        flood_threshold_m3=1.0,
        out_dir=tmp_path / "out",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests: summary contract
# ---------------------------------------------------------------------------

class TestRunBatchValidationReturnsSummaryKeys:
    def test_run_batch_validation_returns_summary_keys(self, monkeypatch, tmp_path):
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        _make_csv(csv_dir)
        _patch_all(monkeypatch, tmp_path)

        result = _run(tmp_path, csv_dir)

        required_keys = {
            "n_scenarios", "classification", "volume", "volume_flooded_only",
            "pr_auc", "per_node_r2", "per_scenario", "timings",
            "summary_csv_path", "scenario_totals_csv_path",
            "timings_csv_path", "metrics_per_scenario_csv_path",
        }
        assert required_keys.issubset(set(result.keys())), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )


class TestRunBatchValidationNScenarios:
    def test_run_batch_validation_n_scenarios_one(self, monkeypatch, tmp_path):
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        _make_csv(csv_dir, "sc001.csv")
        _patch_all(monkeypatch, tmp_path)

        assert _run(tmp_path, csv_dir)["n_scenarios"] == 1

    def test_run_batch_validation_n_scenarios_three(self, monkeypatch, tmp_path):
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        for name in ("sc001.csv", "sc002.csv", "sc003.csv"):
            _make_csv(csv_dir, name)
        _patch_all(monkeypatch, tmp_path)

        assert _run(tmp_path, csv_dir)["n_scenarios"] == 3


def test_run_batch_validation_derives_expected_nodes_from_base_inp(
    monkeypatch, tmp_path
):
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    _make_csv(csv_dir)
    _patch_all(monkeypatch, tmp_path)

    captured = {}

    def capture_load_scenario(csv_path, expected_nodes):
        captured["expected_nodes"] = expected_nodes
        return _FAKE_SCENARIO

    monkeypatch.setattr(hydrograph_batch, "load_scenario", capture_load_scenario)

    _run(tmp_path, csv_dir)

    assert captured["expected_nodes"] == {"J1", "J2"}


# ---------------------------------------------------------------------------
# Tests: new outputs (totals, timings, per-scenario metrics)
# ---------------------------------------------------------------------------

def test_scenario_totals_csv_contents(monkeypatch, tmp_path):
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    _make_csv(csv_dir)
    _patch_all(monkeypatch, tmp_path)

    result = _run(tmp_path, csv_dir)

    totals = pd.read_csv(result["scenario_totals_csv_path"])
    assert list(totals.columns) == [
        "scenario_id", "vol_total_swmm_m3", "vol_total_pred_m3",
        "error_m3", "error_pct", "n_extrapolated",
    ]
    row = totals.iloc[0]
    assert row["scenario_id"] == "sc001"
    assert row["vol_total_swmm_m3"] == 4.0
    assert row["vol_total_pred_m3"] == 5.0
    assert row["error_m3"] == 1.0
    assert row["error_pct"] == 25.0
    assert row["n_extrapolated"] == 1


def test_timings_csv_contents(monkeypatch, tmp_path):
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    _make_csv(csv_dir)
    _patch_all(monkeypatch, tmp_path)

    result = _run(tmp_path, csv_dir)

    timings = pd.read_csv(result["timings_csv_path"])
    expected_cols = {
        "scenario_id", "t_write_inp_s", "t_swmm_s", "t_parse_rpt_s",
        "t_features_s", "t_inference_s", "speedup",
        "t_model_load_s", "t_static_features_s", "device",
    }
    assert expected_cols.issubset(set(timings.columns))
    row = timings.iloc[0]
    assert row["t_features_s"] == 0.001
    assert row["t_inference_s"] == 0.002
    assert row["speedup"] == pytest.approx(row["t_swmm_s"] / 0.003)
    assert row["device"] == "cpu"


def test_metrics_per_scenario_includes_csi_and_continuity(monkeypatch, tmp_path):
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    _make_csv(csv_dir)
    _patch_all(monkeypatch, tmp_path)

    result = _run(tmp_path, csv_dir)

    per_scen = pd.read_csv(result["metrics_per_scenario_csv_path"])
    assert {"csi", "continuity_error_pct", "mae_flooded_m3"}.issubset(per_scen.columns)
    # Fake data: TP=1, FP=0, FN=0 → CSI = 1.0
    assert per_scen.iloc[0]["csi"] == 1.0
    assert per_scen.iloc[0]["continuity_error_pct"] == 0.5


def test_pr_auc_computed_from_probabilities(monkeypatch, tmp_path):
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    _make_csv(csv_dir)
    _patch_all(monkeypatch, tmp_path)

    result = _run(tmp_path, csv_dir)

    # Perfect separation in the fake data (0.1 → 0, 0.9 → 1) → AP = 1.0
    assert result["pr_auc"] == 1.0


# ---------------------------------------------------------------------------
# Tests: guards
# ---------------------------------------------------------------------------

class _FakeOptionsInp:
    def __init__(self, options):
        self._options = options

    def __contains__(self, key):
        return key == "OPTIONS"

    def __getitem__(self, key):
        return self._options


def test_validate_base_inp_rejects_non_lps_units(monkeypatch, tmp_path):
    monkeypatch.setattr(
        hydrograph_batch, "load_inp",
        lambda path: _FakeOptionsInp({"FLOW_UNITS": "CFS"}),
    )
    with pytest.raises(ValueError, match="FLOW_UNITS"):
        hydrograph_batch._validate_base_inp(
            tmp_path / "base.inp", tmp_path / "clf.joblib", False
        )


def test_validate_base_inp_hash_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        hydrograph_batch, "load_inp",
        lambda path: _FakeOptionsInp({"FLOW_UNITS": "LPS"}),
    )
    base = tmp_path / "base.inp"
    base.write_text("network-v2")
    (tmp_path / "training_inp_hash.txt").write_text("0" * 32)

    with pytest.raises(ValueError, match="no coincide"):
        hydrograph_batch._validate_base_inp(base, tmp_path / "clf.joblib", False)

    # allow_inp_mismatch downgrades to a warning
    with pytest.warns(UserWarning, match="no coincide"):
        hydrograph_batch._validate_base_inp(base, tmp_path / "clf.joblib", True)


def test_validate_base_inp_warns_on_ponding(monkeypatch, tmp_path):
    monkeypatch.setattr(
        hydrograph_batch, "load_inp",
        lambda path: _FakeOptionsInp({"FLOW_UNITS": "LPS", "ALLOW_PONDING": "YES"}),
    )
    with pytest.warns(UserWarning, match="ALLOW_PONDING"):
        hydrograph_batch._validate_base_inp(
            tmp_path / "base.inp", tmp_path / "clf.joblib", False
        )


def test_read_continuity_error_parses_worst_value(tmp_path):
    rpt = tmp_path / "run.rpt"
    rpt.write_text(
        "\n".join(
            [
                "  Runoff Quantity Continuity",
                "  Continuity Error (%) .....        -0.05",
                "  Flow Routing Continuity",
                "  Continuity Error (%) .....        -7.31",
            ]
        ),
        encoding="utf-8",
    )
    assert hydrograph_batch._read_continuity_error(rpt) == -7.31


def test_build_swmm_df_covers_all_requested_nodes(monkeypatch, tmp_path):
    """Truth covers the full junction list, zero-filling non-flooded nodes."""
    from swmm_resilience.extraction import labels

    monkeypatch.setattr(
        labels,
        "read_node_flooding_summary",
        lambda path: pd.DataFrame(
            {"node_id": ["J2"], "flooding_volume_m3": [4.0]}
        ),
    )
    df = hydrograph_batch._build_swmm_df(["J1", "J2", "J3"], tmp_path / "x.rpt", 1.0)
    assert df["node_id"].tolist() == ["J1", "J2", "J3"]
    assert df["vol_swmm_m3"].tolist() == [0.0, 4.0, 0.0]
    assert df["inunda_swmm"].tolist() == [0, 1, 0]


# ---------------------------------------------------------------------------
# Tests: plots wiring (unchanged behavior)
# ---------------------------------------------------------------------------

def test_plot_scenario_flood_maps_uses_shared_scale_and_root_output(
    monkeypatch, tmp_path
):
    comp_df = pd.DataFrame(
        {
            "scenario_id": ["storm_a", "storm_a"],
            "node_id": ["1I", "2C"],
            "inunda_swmm": [1, 0],
            "inunda_pred": [0, 1],
            "vol_swmm_m3": [12.0, 0.0],
            "vol_pred_m3": [0.0, 5.0],
        }
    )
    calls = []

    def capture_plot_flood_map(**kwargs):
        calls.append(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(
        model_comparison,
        "plot_flood_map",
        capture_plot_flood_map,
        raising=False,
    )

    paths = model_comparison.plot_scenario_flood_maps(
        comp_df,
        tmp_path / "network.inp",
        tmp_path,
        "storm_a",
    )

    assert paths == (
        tmp_path / "flood_map_swmm_storm_a.png",
        tmp_path / "flood_map_ml_storm_a.png",
    )
    assert [call["vmax_global"] for call in calls] == [12.0, 12.0]
    assert calls[0]["node_data"]["total_flood_volume_m3"].tolist() == [12.0, 0.0]
    assert calls[1]["node_data"]["total_flood_volume_m3"].tolist() == [0.0, 5.0]
    assert calls[0]["title"] == "Flood Map - storm_a\nSWMM Simulation"
    assert calls[1]["title"] == "Flood Map - storm_a\nML Prediction"


def _capture_scenario_flood_map_calls(monkeypatch):
    comp_df = pd.DataFrame(
        {
            "scenario_id": ["storm_a", "storm_a"],
            "node_id": ["1I", "2C"],
            "inunda_swmm": [1, 0],
            "inunda_pred": [0, 1],
            "vol_swmm_m3": [12.0, 0.0],
            "vol_pred_m3": [0.0, 5.0],
        }
    )
    calls = []

    def capture_plot_flood_map(**kwargs):
        calls.append(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(
        model_comparison, "plot_flood_map", capture_plot_flood_map, raising=False
    )
    return comp_df, calls


def test_plot_scenario_flood_maps_stamps_runtime_when_times_given(
    monkeypatch, tmp_path
):
    comp_df, calls = _capture_scenario_flood_map_calls(monkeypatch)

    model_comparison.plot_scenario_flood_maps(
        comp_df, tmp_path / "network.inp", tmp_path, "storm_a",
        t_swmm_s=1.85, t_ml_s=0.024,
    )

    assert calls[0]["runtime_text"] == "Compute time: 1.85 s"
    assert calls[1]["runtime_text"] == "Compute time: 0.0240 s"


def test_plot_scenario_flood_maps_no_runtime_when_times_absent(
    monkeypatch, tmp_path
):
    comp_df, calls = _capture_scenario_flood_map_calls(monkeypatch)

    model_comparison.plot_scenario_flood_maps(
        comp_df, tmp_path / "network.inp", tmp_path, "storm_a",
    )

    assert calls[0]["runtime_text"] is None
    assert calls[1]["runtime_text"] is None


def test_run_batch_validation_generates_flood_maps_in_output_root(
    monkeypatch, tmp_path
):
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    _make_csv(csv_dir)
    fake_inp, _ = _patch_all(monkeypatch, tmp_path)
    calls = []

    def capture_maps(df, inp_path, out_dir, scenario_id, **kwargs):
        calls.append((df, inp_path, out_dir, scenario_id, kwargs))
        return ()

    monkeypatch.setattr(
        hydrograph_batch,
        "plot_scenario_flood_maps",
        capture_maps,
        raising=False,
    )
    out_dir = tmp_path / "out"

    _run(tmp_path, csv_dir)

    assert len(calls) == 1
    _, inp_path, map_out_dir, scenario_id, kwargs = calls[0]
    assert inp_path == fake_inp
    assert map_out_dir == out_dir / "sc001" / "flood_maps"
    assert scenario_id == "sc001"
    # Compute-time annotations: measured SWMM and ML times are forwarded
    assert isinstance(kwargs["t_swmm_s"], float)
    assert isinstance(kwargs["t_ml_s"], float)


def test_run_batch_validation_generates_critical_hydrograph_in_output_root(
    monkeypatch, tmp_path
):
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    _make_csv(csv_dir)
    _patch_all(monkeypatch, tmp_path)
    calls = []

    def capture_hydrograph(scenario, output_path):
        calls.append((scenario, output_path))
        return output_path

    monkeypatch.setattr(
        hydrograph_batch,
        "plot_scenario_hydrograph",
        capture_hydrograph,
        raising=False,
    )
    out_dir = tmp_path / "out"

    _run(tmp_path, csv_dir)

    assert calls == [
        (_FAKE_SCENARIO, out_dir / "sc001" / "hydrograph_sc001.png"),
    ]


class TestRunBatchValidationSavesSummaryCsv:
    def test_run_batch_validation_saves_summary_csv(self, monkeypatch, tmp_path):
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        _make_csv(csv_dir)
        _patch_all(monkeypatch, tmp_path)

        result = _run(tmp_path, csv_dir)

        csv_path = result["summary_csv_path"]
        assert isinstance(csv_path, Path)
        assert csv_path.exists()


class TestRunBatchValidationEmptyDir:
    def test_run_batch_validation_empty_dir(self, monkeypatch, tmp_path):
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        _patch_all(monkeypatch, tmp_path)

        result = _run(tmp_path, csv_dir)

        assert result["n_scenarios"] == 0
        assert isinstance(result["classification"], dict)
        assert isinstance(result["volume"], dict)
        assert isinstance(result["per_node_r2"], dict)
        assert len(result["per_node_r2"]) == 0
        assert result["pr_auc"] is None
        assert result["per_scenario"] == []

        # all CSVs must still be written (empty frames)
        assert result["summary_csv_path"].exists()
        assert result["scenario_totals_csv_path"].exists()
        assert result["timings_csv_path"].exists()


class TestRunBatchValidationPerNodeR2Keys:
    def test_run_batch_validation_per_node_r2_keys(self, monkeypatch, tmp_path):
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        _make_csv(csv_dir, "sc001.csv")
        _make_csv(csv_dir, "sc002.csv")
        _patch_all(monkeypatch, tmp_path)

        result = _run(tmp_path, csv_dir)

        per_node_r2 = result["per_node_r2"]
        assert isinstance(per_node_r2, dict)
        expected_nodes = set(_NODES)
        actual_nodes = set(per_node_r2.keys())
        assert expected_nodes.issubset(actual_nodes), (
            f"per_node_r2 missing nodes {expected_nodes - actual_nodes}"
        )
