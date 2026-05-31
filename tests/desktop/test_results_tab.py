"""TDD tests for the Resultados tab in the desktop app."""

from __future__ import annotations

import tkinter as tk

from swmm_resilience.desktop.app import ResilienciaDesktopApp


def test_results_tab_exists():
    root = tk.Tk()
    root.withdraw()
    app = ResilienciaDesktopApp(root)
    try:
        tabs = [app.notebook.tab(i, "text") for i in range(app.notebook.index("end"))]
        assert "Resultados" in tabs
    finally:
        root.destroy()


def test_results_tree_has_three_folders():
    root = tk.Tk()
    root.withdraw()
    app = ResilienciaDesktopApp(root)
    try:
        children = app.results_tree.get_children()
        assert len(children) == 3
        texts = [app.results_tree.item(c, "text") for c in children]
        assert "SWMM" in texts
        assert "ML Tabular" in texts
        assert "Surrogado (CNN)" in texts
    finally:
        root.destroy()


def test_predict_surr_button_disabled_without_artifacts():
    """Button is disabled when surrogate_cnn_weights.pt does not exist."""
    root = tk.Tk()
    root.withdraw()
    app = ResilienciaDesktopApp(root)
    try:
        app.predict_inp_var.set("/nonexistent/path/model.inp")
        app._update_predict_button_state()
        assert str(app.surr_predict_btn.cget("state")) == "disabled"
    finally:
        root.destroy()


def test_predict_surrogate_validates_input():
    """Predict with invalid multiplier shows error (no crash)."""
    root = tk.Tk()
    root.withdraw()
    app = ResilienciaDesktopApp(root)
    try:
        app.surr_mult_var.set("not-a-number")
        app._predict_surrogate()
    finally:
        root.destroy()
