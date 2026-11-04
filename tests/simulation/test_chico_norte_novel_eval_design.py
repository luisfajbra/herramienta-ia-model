import shutil
from uuid import uuid4

import pytest
from pathlib import Path

from scripts.run_chico_norte_novel_hydrograph_eval import (
    REPO_ROOT,
    _build_event_inputs,
    draw_unseen_factors,
    validate_normalized_shape,
)


def test_random_factor_draw_is_reproducible_and_off_training_grid():
    first = draw_unseen_factors(seed=20261102, count=5)
    second = draw_unseen_factors(seed=20261102, count=5)

    assert first == second == [0.7, 1.3, 2.7, 3.9, 4.3]
    assert all(0.2 <= factor <= 5.0 for factor in first)
    assert all(not (factor / 0.2).is_integer() for factor in first)


def test_new_shape_requires_zero_start_end_unit_peak_and_increasing_times():
    valid = [(0.0, 0.0), (0.5, 0.4), (1.0, 1.0), (1.5, 0.0)]
    validate_normalized_shape(valid)

    with pytest.raises(ValueError, match="start and end at zero"):
        validate_normalized_shape([(0.0, 0.1), (1.0, 1.0), (2.0, 0.0)])

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_normalized_shape([(0.0, 0.0), (1.0, 1.0), (1.0, 0.0)])


def test_event_builder_loads_config_and_materializes_closed_input():
    shapes = {"test_shape": [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]}
    shape_paths = {"test_shape": REPO_ROOT / "data/hydrograph_shapes/common_bogota.csv"}
    root = Path(__file__).parent / f".tmp_novel_builder_{uuid4().hex}"
    root.mkdir()
    try:
        plan, events, planned_sources, positive_nodes, rows = _build_event_inputs(
            root, REPO_ROOT / "config.yaml", shapes, shape_paths, [0.7]
        )

        assert len(plan) == 1
        assert events[plan[0]["scenario_key"]] == ("test_shape", 0.7)
        assert planned_sources[plan[0]["scenario_key"]]["path"].startswith("events/")
        assert len(positive_nodes) > 0
        assert rows[0]["closure_policy"]["policy_version"] == "linear_v1"
    finally:
        shutil.rmtree(root)
