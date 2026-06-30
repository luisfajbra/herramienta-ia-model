from types import SimpleNamespace

import numpy as np

from swmm_resilience.ml import feature_importance
from swmm_resilience.ml.trainer import FEATURE_COLS


def test_plot_importance_uses_descriptive_english_labels(monkeypatch, tmp_path):
    model = SimpleNamespace(feature_importances_=np.arange(len(FEATURE_COLS)))
    pipeline = SimpleNamespace(named_steps={"model": model})
    captured = {}
    original_subplots = feature_importance.plt.subplots

    def capturing_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        original_barh = ax.barh

        def barh(labels, values, **bar_kwargs):
            captured["labels"] = list(labels)
            return original_barh(labels, values, **bar_kwargs)

        ax.barh = barh
        return fig, ax

    monkeypatch.setattr(feature_importance.plt, "subplots", capturing_subplots)

    feature_importance._plot_importance(
        pipeline,
        "Feature Importance - Classifier",
        tmp_path / "importance.png",
    )

    assert "Base Inflow" in captured["labels"]
    assert "base_inflow_lps" not in captured["labels"]


def test_plot_importance_hides_zero_importance_features(monkeypatch, tmp_path):
    importances = np.ones(len(FEATURE_COLS))
    importances[FEATURE_COLS.index("n_tuberias_out")] = 0.0
    model = SimpleNamespace(feature_importances_=importances)
    pipeline = SimpleNamespace(named_steps={"model": model})
    captured = {}
    original_subplots = feature_importance.plt.subplots

    def capturing_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        original_barh = ax.barh

        def barh(labels, values, **bar_kwargs):
            captured["labels"] = list(labels)
            captured["values"] = list(values)
            return original_barh(labels, values, **bar_kwargs)

        ax.barh = barh
        return fig, ax

    monkeypatch.setattr(feature_importance.plt, "subplots", capturing_subplots)

    feature_importance._plot_importance(
        pipeline,
        "Feature Importance - Classifier",
        tmp_path / "importance.png",
    )

    assert "Outlet Pipe Count" not in captured["labels"]
    assert all(value > 0.0 for value in captured["values"])
