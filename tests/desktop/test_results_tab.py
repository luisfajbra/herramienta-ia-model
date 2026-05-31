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

def test_refresh_tree_populates_children(tmp_path, monkeypatch):
    """Tree gets populated when map files exist in network directories."""
    root = tk.Tk()
    root.withdraw()
    app = ResilienciaDesktopApp(root)
    try:
        network_dir = tmp_path / "test_network"
        swmm_dir = network_dir / "results" / "plots"
        swmm_dir.mkdir(parents=True)
        (swmm_dir / "flood_map_qx1.50_swmm.png").write_bytes(b"dummy")
        inp_file = network_dir / "test_network.inp"
        inp_file.write_bytes(b"")
        app.predict_inp_var.set(str(inp_file))

        app._refresh_results_tree()
        swmm_children = app.results_tree.get_children(app._swmm_root)
        assert len(swmm_children) == 1
        assert app.results_tree.item(swmm_children[0], "text") == "flood_map_qx1.50"
    finally:
        root.destroy()


def test_display_image_shows_selected_map(tmp_path):
    """Selecting a tree item displays the image in the right panel."""
    root = tk.Tk()
    root.withdraw()
    app = ResilienciaDesktopApp(root)
    try:
        network_dir = tmp_path / "test_network"
        swmm_dir = network_dir / "results" / "plots"
        swmm_dir.mkdir(parents=True)
        inp_file = network_dir / "test_network.inp"
        inp_file.write_bytes(b"")

        from PIL import Image as PILImage
        img = PILImage.new("RGB", (10, 10), color="red")
        png_path = swmm_dir / "flood_map_qx2.00_swmm.png"
        img.save(png_path)

        app.predict_inp_var.set(str(inp_file))
        app._refresh_results_tree()

        child = app.results_tree.get_children(app._swmm_root)[0]
        app.results_tree.selection_set(child)
        app._on_tree_select(None)

        assert app.image_label.cget("image") != ""
        assert hasattr(app, "_current_photo")
    finally:
        root.destroy()


def test_refresh_tree_empty_when_no_dirs_exist():
    """Tree is empty when no map directories exist."""
    root = tk.Tk()
    root.withdraw()
    app = ResilienciaDesktopApp(root)
    try:
        app.predict_inp_var.set("/nonexistent/path/model.inp")
        app._refresh_results_tree()
        for root_key in (app._swmm_root, app._ml_root, app._surr_root):
            assert len(app.results_tree.get_children(root_key)) == 0
    finally:
        root.destroy()


def test_refresh_tree_shows_all_three_categories(tmp_path):
    """Tree populates children under each category folder."""
    root = tk.Tk()
    root.withdraw()
    app = ResilienciaDesktopApp(root)
    try:
        network_dir = tmp_path / "test_network"
        swmm_dir = network_dir / "results" / "plots"
        swmm_dir.mkdir(parents=True)
        ml_dir = network_dir / "ml" / "results"
        ml_dir.mkdir(parents=True)
        surr_dir = network_dir / "results" / "temporal" / "maps"
        surr_dir.mkdir(parents=True)
        inp_file = network_dir / "test_network.inp"
        inp_file.write_bytes(b"")

        (swmm_dir / "flood_map_qx1.00_swmm.png").write_bytes(b"d")
        (ml_dir / "flood_map_qx1.00_ml.png").write_bytes(b"d")
        (surr_dir / "surrogate_map_qx1.00.png").write_bytes(b"d")

        app.predict_inp_var.set(str(inp_file))
        app._refresh_results_tree()

        assert len(app.results_tree.get_children(app._swmm_root)) == 1
        assert len(app.results_tree.get_children(app._ml_root)) == 1
        assert len(app.results_tree.get_children(app._surr_root)) == 1
    finally:
        root.destroy()
