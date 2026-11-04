import pytest
from pptx import Presentation
import matplotlib.pyplot as plt
import numpy as np

from scripts.build_chico_norte_presentation_figures import (
    expand_map_scenarios,
    plot_flood_map,
    plot_hydrograph_series,
    select_map_scenarios,
)
from scripts.update_weekly_2026_11_02_deck import remove_inherited_comparison_map_slides


def test_hydrograph_series_uses_the_existing_line_and_fill_style():
    fig, ax = plt.subplots()
    try:
        plot_hydrograph_series(ax, np.array([0.0, 0.5, 1.0]),
                               np.array([0.0, 0.8, 0.2]))

        assert len(ax.lines) == 1
        assert len(ax.patches) == 0
        assert len(ax.collections) == 1
        assert ax.lines[0].get_color() == "#385F8B"
        assert ax.lines[0].get_linewidth() == 2.5
        assert ax.lines[0].get_marker() == "o"
        assert ax.get_xlabel() == "Tiempo desde el inicio del evento (h)"
        assert ax.get_ylabel() == "Caudal normalizado (q/qmáx)"
        assert ax.xaxis._major_tick_kw["gridOn"] is True
        assert ax.yaxis._major_tick_kw["gridOn"] is True
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
    finally:
        plt.close(fig)


def test_select_map_scenarios_returns_low_mid_high_for_each_hydrograph():
    available = [
        {"shape_id": shape, "factor_mult": factor,
         "scenario_key": f"{shape}:f{factor:.3f}"}
        for shape in ("pulses", "sustained")
        for factor in (0.7, 1.3, 2.7, 3.9, 4.3)
    ]

    selected = select_map_scenarios(available, factors=(0.7, 2.7, 4.3))

    assert [(row["shape_id"], row["factor_mult"]) for row in selected] == [
        ("pulses", 0.7), ("pulses", 2.7), ("pulses", 4.3),
        ("sustained", 0.7), ("sustained", 2.7), ("sustained", 4.3),
    ]


def test_select_map_scenarios_fails_if_a_requested_factor_is_missing():
    available = [
        {"shape_id": "pulses", "factor_mult": 0.7, "scenario_key": "pulses:f0.7"},
        {"shape_id": "sustained", "factor_mult": 0.7, "scenario_key": "sustained:f0.7"},
    ]

    with pytest.raises(ValueError, match="missing requested map scenario"):
        select_map_scenarios(available, factors=(0.7, 2.7, 4.3))


def test_each_selected_scenario_expands_to_one_map_per_model():
    scenarios = [
        {"shape_id": shape, "factor_mult": factor, "scenario_key": f"{shape}:{factor}"}
        for shape in ("pulses", "sustained")
        for factor in (0.7, 1.3, 2.7, 3.9, 4.3)
    ]

    records = expand_map_scenarios(scenarios, factors=(0.7, 2.7, 4.3))

    assert len(records) == 18
    assert [(record["scenario_key"], record["model_name"]) for record in records[:3]] == [
        ("pulses:0.7", "SWMM"), ("pulses:0.7", "MLP"), ("pulses:0.7", "XGBoost")
    ]
    assert {record["model_key"] for record in records} == {"swmm", "mlp", "xgboost"}


def test_standalone_flood_map_draws_network_and_volume_points_without_error_markers():
    coordinates = {"N1": (0.0, 0.0), "N2": (1.0, 0.0), "N3": (2.0, 1.0)}
    fig, ax = plt.subplots()
    try:
        mappable = plot_flood_map(
            ax, coordinates, [("C1", "N1", "N2"), ("C2", "N2", "N3")],
            ["N1", "N2", "N3"], np.array([0.0, 12.0, 85.0]),
            np.array([False, True, True]), "MLP",
        )

        assert len(fig.axes) == 1
        assert len(ax.lines) == 2
        assert len(ax.collections) == 1  # inundated nodes only
        assert mappable.get_cmap().name == "YlOrRd"
        assert mappable.norm.vmin == 0
        assert mappable.norm.vmax == 85
        assert mappable.get_edgecolors().size == 0 or not mappable.get_linewidths().any()
        assert not ax.axison
    finally:
        plt.close(fig)


def test_legacy_comparison_map_slides_are_removed_from_the_copied_deck():
    source = Presentation(
        "Presentacion_MLSMFP_30-10_avance_chico_norte_y_modelos_paridad_global_final.pptx"
    )

    removed = remove_inherited_comparison_map_slides(source)

    assert removed == [13, 14, 15]
    assert len(source.slides) == 14
    assert all("XGB FP" not in " ".join(shape.text for shape in slide.shapes
                                       if getattr(shape, "has_text_frame", False))
               for slide in source.slides)


@pytest.mark.parametrize("slide_number", [35, 36])
def test_parity_plot_does_not_overlap_its_explanation(slide_number):
    deck = Presentation(
        "Presentacion_MLSMFP_02-11_avance_chico_norte_mapas_separados_inundacion.pptx"
    )
    assert len(deck.slides) == 38
    assert all("XGB FP" not in " ".join(shape.text for shape in slide.shapes
                                       if getattr(shape, "has_text_frame", False))
               for slide in deck.slides)
    slide = deck.slides[slide_number - 1]
    picture = next(shape for shape in slide.shapes if shape.shape_type == 13)
    note = next(shape for shape in slide.shapes
                if getattr(shape, "has_text_frame", False)
                and "RMSE agrupado" in shape.text)

    assert picture.top + picture.height <= note.top


def test_map_slides_show_one_model_each_without_error_marker_explanations():
    deck = Presentation(
        "Presentacion_MLSMFP_02-11_avance_chico_norte_mapas_separados_inundacion.pptx"
    )
    expected_models = ["SWMM", "MLP", "XGBoost"] * 6

    for slide_number, model_name in zip(range(17, 35), expected_models):
        slide = deck.slides[slide_number - 1]
        text = " ".join(shape.text for shape in slide.shapes
                        if getattr(shape, "has_text_frame", False))
        pictures = [shape for shape in slide.shapes if shape.shape_type == 13]

        assert model_name in text
        assert len(pictures) == 1
        assert "falsa alarma" not in text.lower()
        assert "omitida" not in text.lower()
