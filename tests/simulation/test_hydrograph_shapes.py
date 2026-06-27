"""Tests for swmm_resilience.simulation.hydrograph_shapes."""
from __future__ import annotations

from pathlib import Path

import pytest

from swmm_resilience.simulation.hydrograph_shapes import (
    apply_shape,
    get_shape_stats,
    load_all_shapes,
    load_shape,
    normalize_from_csv,
)


def _write_shape_csv(tmp_path: Path, name: str, rows: list[tuple]) -> Path:
    p = tmp_path / name
    p.write_text("time_h,q_norm\n" + "\n".join(f"{t},{q}" for t, q in rows))
    return p


def test_load_shape_basic(tmp_path):
    p = _write_shape_csv(tmp_path, "s.csv", [(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)])
    shape = load_shape(p)
    assert shape == [(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)]


def test_load_shape_empty_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("time_h,q_norm\n")
    with pytest.raises(ValueError, match="vacío"):
        load_shape(p)


def test_load_shape_not_starting_at_zero_raises(tmp_path):
    p = _write_shape_csv(tmp_path, "s.csv", [(0.5, 0.0), (1.0, 1.0)])
    with pytest.raises(ValueError, match="t=0"):
        load_shape(p)


def test_load_all_shapes(tmp_path):
    _write_shape_csv(tmp_path, "shape_a.csv", [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])
    _write_shape_csv(tmp_path, "shape_b.csv", [(0.0, 0.0), (3.0, 1.0), (6.0, 0.0)])
    shapes = load_all_shapes(tmp_path)
    assert set(shapes.keys()) == {"shape_a", "shape_b"}
    assert shapes["shape_a"][1] == (1.0, 1.0)


def test_get_shape_stats():
    shape = [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0), (3.0, 0.3), (4.0, 0.0)]
    dur, t_pico = get_shape_stats(shape)
    assert dur == 4.0
    assert t_pico == 2.0


def test_get_shape_stats_empty():
    assert get_shape_stats([]) == (0.0, 0.0)


def test_apply_shape_basic():
    shape = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
    base_inflows = {"A": 10.0, "B": 5.0, "C": 0.0}  # C has no inflow
    result = apply_shape(shape, base_inflows, factor=2.0)
    assert set(result.keys()) == {"A", "B"}  # C excluded (base=0)
    assert result["A"] == [(0.0, 0.0), (1.0, 20.0), (2.0, 0.0)]
    assert result["B"] == [(0.0, 0.0), (1.0, 10.0), (2.0, 0.0)]


def test_apply_shape_scales_proportionally():
    shape = [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]
    base_inflows = {"X": 26.0}
    result = apply_shape(shape, base_inflows, factor=3.0)
    assert result["X"][1] == pytest.approx((0.5, 78.0))  # 26 * 3.0 * 1.0


def test_normalize_from_csv(tmp_path):
    csv_path = tmp_path / "val.csv"
    csv_path.write_text(
        "time_h,nodeA,nodeB\n"
        "0.0,0.0,0.0\n"
        "1.0,40.0,40.0\n"
        "2.0,20.0,20.0\n"
    )
    shape = normalize_from_csv(csv_path)
    assert shape[1] == pytest.approx((1.0, 1.0))
    assert shape[2][1] == pytest.approx(0.5)


def test_normalize_from_csv_specific_col(tmp_path):
    csv_path = tmp_path / "val.csv"
    csv_path.write_text(
        "time_h,nodeA,nodeB\n"
        "0.0,0.0,0.0\n"
        "1.0,40.0,80.0\n"
        "2.0,0.0,0.0\n"
    )
    shape = normalize_from_csv(csv_path, node_col="nodeA")
    assert shape[1][1] == pytest.approx(1.0)
