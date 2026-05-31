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
