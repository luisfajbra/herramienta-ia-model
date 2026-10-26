import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from swmm_resilience.visualization.flood_volume_curve import (
    plot_flood_volume_combined,
    plot_flood_volume_curve,
)
from swmm_resilience.visualization.labels import (
    IN_SAMPLE_BANNER,
    IN_SAMPLE_NOTE,
    add_in_sample_note,
    print_in_sample_banner,
)
from swmm_resilience.visualization.model_comparison import plot_factor_comparison
from swmm_resilience.visualization.resilience_curve import plot_resilience_curve


@pytest.fixture
def captured_figure_texts(monkeypatch):
    seen = []
    original = Figure.savefig

    def spy(self, *args, **kwargs):
        seen.append((args[0] if args else None, [t.get_text() for t in self.texts]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy)
    return seen


def _has_note(seen, name_part):
    return any(name_part in str(path) and IN_SAMPLE_NOTE in texts for path, texts in seen)


def test_add_in_sample_note_writes_the_note_into_the_figure():
    fig = Figure()
    add_in_sample_note(fig)
    assert IN_SAMPLE_NOTE in [t.get_text() for t in fig.texts]


def test_resilience_ml_figure_carries_the_in_sample_note(tmp_path, captured_figure_texts):
    df = pd.DataFrame({"factor": [1.0, 2.0], "resilience_swmm": [1.0, 0.5], "resilience_ml": [1.0, 0.6]})
    plot_resilience_curve(df, tmp_path)
    assert _has_note(captured_figure_texts, "resilience_ml")


def test_flood_volume_figures_carry_the_in_sample_note(tmp_path, captured_figure_texts):
    df = pd.DataFrame({"factor": [1.0, 2.0], "vol_total_swmm": [10.0, 20.0], "vol_total_ml": [11.0, 19.0]})
    plot_flood_volume_curve(df, tmp_path)
    plot_flood_volume_combined(df, tmp_path)
    assert _has_note(captured_figure_texts, "flood_volume_ml")
    assert _has_note(captured_figure_texts, "flood_volume_combined")


def test_factor_comparison_figures_carry_the_in_sample_note(tmp_path, captured_figure_texts):
    df = pd.DataFrame({"node_id": ["1I", "2O"], "vol_swmm_m3": [12.0, 3.0], "vol_pred_m3": [10.0, 5.0]})
    plot_factor_comparison(df, tmp_path, 1.0)
    assert _has_note(captured_figure_texts, "volume_by_node_factor")
    assert _has_note(captured_figure_texts, "parity_factor")


def test_print_in_sample_banner_emits_the_banner(capsys):
    print_in_sample_banner()
    assert IN_SAMPLE_BANNER in capsys.readouterr().out


def test_the_evaluate_shapes_branch_prints_the_in_sample_banner():
    source = (Path(__file__).resolve().parents[2] / "main.py").read_text(encoding="utf-8")
    branch = source.split("if args.evaluate_shapes:", 1)[1].split("load_all_shapes", 1)[0]
    assert "print_in_sample_banner()" in branch
