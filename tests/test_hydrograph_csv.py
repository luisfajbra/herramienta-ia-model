import pytest
from pathlib import Path
import pandas as pd

from swmm_resilience.validation.hydrograph_csv import (
    load_scenario,
    scenario_id_from_path,
    HydrographScenario,
)

NODES = {"1C", "2C"}


def _write_csv(tmp_path: Path, rows: list[dict], filename="scenario.csv") -> Path:
    path = tmp_path / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _valid_rows():
    return [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.6},
        {"node_id": "1C", "time": "0:03", "value_lps": 0.9},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.8},
        {"node_id": "2C", "time": "0:03", "value_lps": 1.2},
    ]


def test_load_valid_scenario(tmp_path):
    path = _write_csv(tmp_path, _valid_rows())
    s = load_scenario(path, NODES)
    assert s.scenario_id == "scenario"
    assert set(s.node_series.keys()) == NODES
    assert s.time_grid_hours == pytest.approx([0.0, 0.05])
    assert s.last_time_hours == pytest.approx(0.05)
    assert s.node_series["1C"] == pytest.approx([(0.0, 0.6), (0.05, 0.9)])


def test_accepts_different_durations_between_files(tmp_path):
    rows_a = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "1C", "time": "1:00", "value_lps": 1.0},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "2C", "time": "1:00", "value_lps": 0.5},
    ]
    rows_b = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "1C", "time": "5:00", "value_lps": 1.0},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "2C", "time": "5:00", "value_lps": 0.5},
    ]
    s_a = load_scenario(_write_csv(tmp_path, rows_a, "a.csv"), NODES)
    s_b = load_scenario(_write_csv(tmp_path, rows_b, "b.csv"), NODES)
    assert s_a.last_time_hours == pytest.approx(1.0)
    assert s_b.last_time_hours == pytest.approx(5.0)


def test_accepts_irregular_shared_grid(tmp_path):
    rows = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "1C", "time": "0:07", "value_lps": 1.0},
        {"node_id": "1C", "time": "1:30", "value_lps": 0.0},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "2C", "time": "0:07", "value_lps": 0.5},
        {"node_id": "2C", "time": "1:30", "value_lps": 0.0},
    ]
    s = load_scenario(_write_csv(tmp_path, rows), NODES)
    assert len(s.time_grid_hours) == 3


def test_rejects_missing_column(tmp_path):
    rows = [{"node_id": "1C", "time": "0:00"}, {"node_id": "2C", "time": "0:00"}]
    with pytest.raises(ValueError, match="Columnas"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_extra_column(tmp_path):
    rows = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0, "extra": 1},
        {"node_id": "1C", "time": "0:03", "value_lps": 0.5, "extra": 1},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.0, "extra": 1},
        {"node_id": "2C", "time": "0:03", "value_lps": 0.5, "extra": 1},
    ]
    with pytest.raises(ValueError, match="Columnas"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_missing_node(tmp_path):
    rows = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "1C", "time": "0:03", "value_lps": 1.0},
    ]
    with pytest.raises(ValueError, match="Nodos incorrectos"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_extra_node(tmp_path):
    rows = _valid_rows() + [
        {"node_id": "3C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "3C", "time": "0:03", "value_lps": 0.5},
    ]
    with pytest.raises(ValueError, match="Nodos incorrectos"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_invalid_time_format(tmp_path):
    rows = _valid_rows()
    rows[0]["time"] = "abc"
    with pytest.raises(ValueError, match="formato inválido"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_invalid_minutes(tmp_path):
    rows = _valid_rows()
    rows[0]["time"] = "0:60"
    with pytest.raises(ValueError, match="formato inválido"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_duplicate_node_time_key(tmp_path):
    rows = _valid_rows() + [{"node_id": "1C", "time": "0:00", "value_lps": 9.9}]
    with pytest.raises(ValueError, match="duplicadas"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_accepts_shuffled_row_order(tmp_path):
    rows = [
        {"node_id": "1C", "time": "0:05", "value_lps": 1.0},
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "2C", "time": "0:05", "value_lps": 1.0},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.0},
    ]
    s = load_scenario(_write_csv(tmp_path, rows), NODES)
    assert s.time_grid_hours == pytest.approx([0.0, 5 / 60])
    assert s.node_series["1C"] == pytest.approx([(0.0, 0.0), (5 / 60, 1.0)])


def test_rejects_node_not_starting_at_zero(tmp_path):
    rows = [
        {"node_id": "1C", "time": "0:01", "value_lps": 0.0},
        {"node_id": "1C", "time": "0:05", "value_lps": 1.0},
        {"node_id": "2C", "time": "0:01", "value_lps": 0.0},
        {"node_id": "2C", "time": "0:05", "value_lps": 1.0},
    ]
    with pytest.raises(ValueError, match="0:00"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_fewer_than_two_time_steps(tmp_path):
    rows = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.0},
    ]
    with pytest.raises(ValueError, match="menos de 2 tiempos"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_mismatched_time_grids(tmp_path):
    rows = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "1C", "time": "0:05", "value_lps": 1.0},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "2C", "time": "0:10", "value_lps": 1.0},
    ]
    with pytest.raises(ValueError, match="malla temporal diferente"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_negative_value(tmp_path):
    rows = _valid_rows()
    rows[0]["value_lps"] = -0.1
    with pytest.raises(ValueError, match="negativos"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_nan_value(tmp_path):
    rows = _valid_rows()
    rows[0]["value_lps"] = float("nan")
    with pytest.raises(ValueError, match="no numéricos"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_non_numeric_string_value(tmp_path):
    rows = _valid_rows()
    rows[0]["value_lps"] = "abc"
    with pytest.raises(ValueError, match="no numéricos"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_scenario_id_from_path():
    assert scenario_id_from_path(Path("pico_desplazado.csv")) == "pico_desplazado"
    assert scenario_id_from_path(Path("/some/dir/Doble Pico.csv")) == "doble_pico"
