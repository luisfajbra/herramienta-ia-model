"""Fase C2 / revision final (I4): un solo umbral de continuidad configurado.

``hydrograph_batch.CONTINUITY_ERROR_WARN_PCT`` seguia hardcodeado a 5.0 y se
usaba en el chequeo warn-only de ``run_batch_validation``, en paralelo a
``config.validation.max_continuity_error_pct`` (el umbral que excluye runs del
entrenamiento). Dos numeros para la misma regla del spec pueden divergir, asi
que ``run_batch_validation`` ahora recibe el umbral como parametro y los call
sites de main.py pasan el valor configurado.

Estos son tests de comportamiento sobre main.main(): run_batch_validation esta
monkeypatcheada (no corre SWMM/joblib), pero el parseo de CLI, la carga de
config y el cableado hasta el call site corren de verdad.
"""
from __future__ import annotations

import sys

import pandas as pd
import pytest

import main
from swmm_resilience.validation import hydrograph_batch

PINNED_THRESHOLD = 3.0  # distinto del default 5.0: la asercion no es vacua


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame({"node_id": ["1C", "2C"], "base_inflow_lps": [2.0, 3.0]})


@pytest.fixture
def captured_calls(monkeypatch):
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


@pytest.fixture
def pinned_config(monkeypatch):
    """config.yaml real, pero con max_continuity_error_pct = 3.0."""
    real_load_config = main.load_config

    def load_with_pinned_threshold(path):
        config = real_load_config(path)
        config.validation.max_continuity_error_pct = PINNED_THRESHOLD
        return config

    monkeypatch.setattr(main, "load_config", load_with_pinned_threshold)


def test_config_yaml_threshold_default_differs_from_the_pinned_value():
    """Premisa del test: 3.0 no es lo que config.yaml trae, asi que ver 3.0 en
    el call site solo puede venir del cableado nuevo."""
    config = main.load_config("config.yaml")
    assert config.validation.max_continuity_error_pct != PINNED_THRESHOLD


def test_evaluate_shapes_passes_the_configured_continuity_threshold(
    monkeypatch, captured_calls, pinned_config
):
    monkeypatch.setattr(main, "load_training_frame", lambda db_path: _training_frame())
    monkeypatch.setattr(sys, "argv", ["main.py", "--evaluate-shapes"])

    main.main()

    assert captured_calls, "run_batch_validation nunca fue llamada"
    thresholds = {kwargs.get("max_continuity_error_pct") for kwargs in captured_calls}
    assert thresholds == {PINNED_THRESHOLD}, thresholds


def test_evaluate_generalization_passes_the_configured_continuity_threshold(
    monkeypatch, captured_calls, pinned_config
):
    monkeypatch.setattr(main, "load_training_frame", lambda db_path: _training_frame())
    monkeypatch.setattr(sys, "argv", ["main.py", "--evaluate-generalization"])

    main.main()

    assert captured_calls, "run_batch_validation nunca fue llamada"
    thresholds = {kwargs.get("max_continuity_error_pct") for kwargs in captured_calls}
    assert thresholds == {PINNED_THRESHOLD}, thresholds


def test_evaluate_hydrographs_passes_the_configured_continuity_threshold(
    monkeypatch, captured_calls, pinned_config, tmp_path
):
    csv_dir = tmp_path / "scenarios"
    csv_dir.mkdir()
    (csv_dir / "s1.csv").write_text("time_hours,1C\n0,0\n1,0\n", encoding="utf-8")
    base_inp = tmp_path / "base.inp"
    base_inp.write_text("[TITLE]\nfixture\n", encoding="utf-8")
    clf = tmp_path / "classifier.joblib"
    clf.write_bytes(b"")
    reg = tmp_path / "regressor.joblib"
    reg.write_bytes(b"")

    monkeypatch.setattr(
        sys, "argv",
        [
            "main.py", "--evaluate-hydrographs", str(csv_dir),
            "--base-inp", str(base_inp),
            "--clf-path", str(clf), "--reg-path", str(reg),
            "--out-dir", str(tmp_path / "out"),
        ],
    )

    main.main()

    assert captured_calls, "run_batch_validation nunca fue llamada"
    assert captured_calls[0]["max_continuity_error_pct"] == PINNED_THRESHOLD


def test_evaluate_validation_shapes_passes_the_configured_continuity_threshold(
    monkeypatch, captured_calls, pinned_config, tmp_path
):
    shapes_dir = tmp_path / "val_shapes"
    shapes_dir.mkdir()
    (shapes_dir / "val_demo.csv").write_text(
        "time_h,q_norm\n0.0000,0.0000\n1.0000,1.0000\n2.0000,0.4000\n",
        newline="\n",
    )
    monkeypatch.setattr(
        sys, "argv",
        [
            "main.py", "--evaluate-validation-shapes", str(shapes_dir),
            "--factors", "1.0", "--out-dir", str(tmp_path / "out"),
        ],
    )

    main.main()

    assert captured_calls, "run_batch_validation nunca fue llamada"
    thresholds = {kwargs.get("max_continuity_error_pct") for kwargs in captured_calls}
    assert thresholds == {PINNED_THRESHOLD}, thresholds
