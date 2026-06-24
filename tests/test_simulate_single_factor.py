"""Tests for --simulate --factor X: runs SWMM + ML for an arbitrary factor."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _run_simulate(factor: float, tmp_path: Path, monkeypatch):
    """Drive main() with --simulate --factor <factor> using monkeypatched deps."""
    fake_nodes = pd.DataFrame({"node_id": ["J1", "J2"]})
    fake_labels = pd.DataFrame(
        {"node_id": ["J1", "J2"], "vol_inundacion_m3": [10.0, 0.0], "inunda": [1, 0]}
    )
    fake_pred = pd.DataFrame(
        {"node_id": ["J1", "J2"], "vol_pred_m3": [9.0, 0.0], "inunda_pred": [1, 0]}
    )
    fake_rpt = tmp_path / "sim.rpt"
    fake_rpt.touch()

    calls = {}

    import main as main_module
    import swmm_resilience.visualization.flood_map as fm_mod

    fake_config = SimpleNamespace(
        network=SimpleNamespace(inp_path=tmp_path / "net.inp", name="TestNet"),
        dataset=SimpleNamespace(output_path=tmp_path / "ds.csv", flood_threshold_m3=1.0),
        visualization=SimpleNamespace(
            output_path=tmp_path / "maps",
            colormap="plasma",
            show_labels_top_n=5,
        ),
    )

    monkeypatch.setattr(main_module, "load_config", lambda _: fake_config)
    monkeypatch.setattr(main_module, "extract_static_features", lambda _: fake_nodes)
    monkeypatch.setattr(main_module, "run_simulation_simple", lambda *a, **kw: fake_rpt)
    monkeypatch.setattr(main_module, "extract_labels", lambda *a, **kw: fake_labels)
    monkeypatch.setattr(main_module, "predict_network", lambda *a, **kw: fake_pred)

    def capture_generate_flood_map(inp, vol_data, f, out, *a, **kw):
        calls[out.name] = {"factor": f, "runtime_text": kw.get("runtime_text")}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
        return out

    monkeypatch.setattr(main_module, "generate_flood_map", capture_generate_flood_map)

    monkeypatch.setattr(sys, "argv", ["main.py", "--simulate", "--factor", str(factor)])
    main_module.main()
    return calls


def test_simulate_generates_swmm_map(monkeypatch, tmp_path):
    calls = _run_simulate(0.3458, tmp_path, monkeypatch)
    swmm_key = f"flood_map_factor_0.35.png"
    assert swmm_key in calls, f"Expected SWMM map key in {list(calls)}"


def test_simulate_generates_pred_map(monkeypatch, tmp_path):
    calls = _run_simulate(0.3458, tmp_path, monkeypatch)
    pred_key = f"flood_map_pred_0.35.png"
    assert pred_key in calls, f"Expected pred map key in {list(calls)}"


def test_simulate_stamps_runtime_on_both_maps(monkeypatch, tmp_path):
    calls = _run_simulate(0.3458, tmp_path, monkeypatch)
    for key in [f"flood_map_factor_0.35.png", f"flood_map_pred_0.35.png"]:
        rt = calls[key]["runtime_text"]
        assert rt is not None and rt.startswith("Compute time:"), (
            f"{key} runtime_text={rt!r}"
        )


def test_simulate_requires_factor(monkeypatch, tmp_path):
    import main as main_module

    fake_config = SimpleNamespace(
        network=SimpleNamespace(inp_path=tmp_path / "net.inp", name="TestNet"),
        dataset=SimpleNamespace(output_path=tmp_path / "ds.csv", flood_threshold_m3=1.0),
        visualization=SimpleNamespace(
            output_path=tmp_path / "maps", colormap="plasma", show_labels_top_n=5
        ),
    )
    monkeypatch.setattr(main_module, "load_config", lambda _: fake_config)
    monkeypatch.setattr(sys, "argv", ["main.py", "--simulate"])

    with pytest.raises(SystemExit):
        main_module.main()
