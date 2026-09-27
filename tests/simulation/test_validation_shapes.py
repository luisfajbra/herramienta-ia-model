"""The out-of-sample validation catalog in data/hydrograph_validation/.

These shapes must never reach training: they are the only evidence that the
surrogate generalises to hydrographs it has not seen.
"""

from pathlib import Path

import pytest

from swmm_resilience.simulation.hydrograph_shapes import (
    get_shape_stats,
    load_all_shapes,
    load_shape,
)

ROOT = Path(__file__).parents[2]
TRAIN_DIR = ROOT / "data" / "hydrograph_shapes"
VAL_DIR = ROOT / "data" / "hydrograph_validation"

EXTRAPOLATION_SHAPES = {"val_extrap_ultrafast_20min", "val_extrap_marathon_20h"}


def _training_duration_range() -> tuple[float, float]:
    durations = [get_shape_stats(s)[0] for s in load_all_shapes(TRAIN_DIR).values()]
    return min(durations), max(durations)


def test_validation_catalog_has_seven_shapes():
    assert len(load_all_shapes(VAL_DIR)) == 7


@pytest.mark.parametrize(
    "csv_path",
    sorted(VAL_DIR.glob("*.csv")),
    ids=lambda path: path.stem,
)
def test_validation_shape_files_are_valid_normalized_profiles(csv_path):
    shape = load_shape(csv_path)
    times = [time_h for time_h, _ in shape]
    flows = [q_norm for _, q_norm in shape]

    assert times[0] == 0.0
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    assert all(0.0 <= flow <= 1.0 for flow in flows)
    assert max(flows) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "csv_path",
    sorted(VAL_DIR.glob("*.csv")),
    ids=lambda path: path.stem,
)
def test_validation_times_stay_distinct_at_minute_resolution(csv_path):
    """write_shape_validation_csv serialises H:MM, so minutes must not collide."""
    minutes = [round(time_h * 60) for time_h, _ in load_shape(csv_path)]

    assert all(later > earlier for earlier, later in zip(minutes, minutes[1:]))


def test_no_validation_shape_is_also_a_training_shape():
    assert not set(load_all_shapes(VAL_DIR)) & set(load_all_shapes(TRAIN_DIR))


def test_validation_descriptors_never_collide_with_training():
    training = {get_shape_stats(s) for s in load_all_shapes(TRAIN_DIR).values()}
    validation = {get_shape_stats(s) for s in load_all_shapes(VAL_DIR).values()}

    assert not training & validation


def test_validation_descriptors_are_unique_among_themselves():
    descriptors = [get_shape_stats(s) for s in load_all_shapes(VAL_DIR).values()]

    assert len(descriptors) == len(set(descriptors))


def test_five_validation_shapes_interpolate_inside_the_trained_range():
    dur_min, dur_max = _training_duration_range()
    shapes = load_all_shapes(VAL_DIR)

    inside = {
        name
        for name, shape in shapes.items()
        if dur_min <= get_shape_stats(shape)[0] <= dur_max
    }

    assert inside == set(shapes) - EXTRAPOLATION_SHAPES


def test_two_validation_shapes_extrapolate_outside_the_trained_range():
    dur_min, dur_max = _training_duration_range()
    shapes = load_all_shapes(VAL_DIR)

    below = get_shape_stats(shapes["val_extrap_ultrafast_20min"])[0]
    above = get_shape_stats(shapes["val_extrap_marathon_20h"])[0]

    assert below < dur_min
    assert above > dur_max
