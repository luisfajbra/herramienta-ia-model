"""Tests for swmm_resilience.validation.drainage_state, against a real .out file."""

from __future__ import annotations

from pathlib import Path

import pytest

from swmm_resilience.validation.drainage_state import read_tail_window_observables


@pytest.fixture(scope="module")
def tiny_out_file(tmp_path_factory):
    """Corre SWMM sobre un .inp minusculo y devuelve la ruta al .out resultante."""
    from pyswmm import Simulation

    out_dir = tmp_path_factory.mktemp("drainage_state_out")
    inp_path = out_dir / "tiny.inp"
    inp_path.write_text(
        """
[TITLE]
tiny network for drainage_state tests

[OPTIONS]
FLOW_UNITS           LPS
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
START_DATE           01/01/2025
START_TIME           00:00:00
REPORT_START_DATE    01/01/2025
REPORT_START_TIME    00:00:00
END_DATE             01/01/2025
END_TIME             00:20:00
REPORT_STEP          00:01:00
WET_STEP             00:01:00
DRY_STEP             00:01:00
ROUTING_STEP         0:00:30

[JUNCTIONS]
;;Name  Elevation  MaxDepth  InitDepth  SurDepth  Aponded
J1      0.0        5.0       0.0        0.0       0
J2      -1.0       5.0       0.0        0.0       0

[OUTFALLS]
;;Name  Elevation  Type      StageData  Gated
OUT1    -2.0       FREE                 NO

[CONDUITS]
;;Name  FromNode  ToNode  Length  Roughness  InOffset  OutOffset
C1      J1        J2      100     0.01       0          0
C2      J2        OUT1    100     0.01       0          0

[XSECTIONS]
;;Link  Shape       Geom1  Geom2  Geom3  Geom4
C1      CIRCULAR    0.5    0      0      0
C2      CIRCULAR    0.5    0      0      0

[INFLOWS]
;;Node  Constituent  TimeSeries  Type   Mfactor  Sfactor  Baseline  Pattern
J1      FLOW         inflow1     FLOW   1.0      1.0

[TIMESERIES]
;;Name    Date  Time     Value
inflow1         0:00     5.0
inflow1         0:05     5.0
inflow1         0:06     0.0
inflow1         0:20     0.0

[REPORT]
INPUT      NO
CONTROLS   NO
SUBCATCHMENTS  NONE
NODES      J1 J2
LINKS      ALL

[TAGS]
""",
        encoding="utf-8",
    )

    with Simulation(str(inp_path)) as sim:
        for _ in sim:
            pass

    out_path = inp_path.with_suffix(".out")
    assert out_path.exists(), "SWMM no genero el .out esperado"
    return out_path


def test_read_tail_window_observables_has_at_least_two_report_steps(tiny_out_file):
    result = read_tail_window_observables(tiny_out_file, drain_down_hours=0.25, tail_window_hours=1.0)
    assert result["n_report_steps"] >= 2


def test_read_tail_window_observables_uses_the_shorter_of_tail_window_and_drain_down(tiny_out_file):
    # drain_down_hours=0.05h (3 min) es mas corto que tail_window_hours=1.0 -> la ventana
    # efectiva son los ultimos 3 minutos, unos pocos reportes de 1 min cada uno.
    short = read_tail_window_observables(tiny_out_file, drain_down_hours=0.05, tail_window_hours=1.0)
    long = read_tail_window_observables(tiny_out_file, drain_down_hours=1.0, tail_window_hours=1.0)
    assert short["n_report_steps"] <= long["n_report_steps"]


def test_read_tail_window_observables_includes_all_expected_series(tiny_out_file):
    result = read_tail_window_observables(tiny_out_file, drain_down_hours=1.0, tail_window_hours=1.0)
    assert result["has_link_data"] is True
    assert set(result["node_total_inflow_lps"]) == {"J1", "J2"}
    assert set(result["node_flooding_lps"]) == {"J1", "J2"}
    assert set(result["node_volume_m3"]) == {"J1", "J2"}
    assert set(result["link_flow_lps"]) == {"C1", "C2"}
    for series in result["node_total_inflow_lps"].values():
        assert len(series) == result["n_report_steps"]


def test_read_tail_window_observables_flow_settles_toward_zero_at_the_end(tiny_out_file):
    """El inflow del .inp de prueba termina en 0.0 desde el minuto 6; a los 20 min
    (fin de simulacion) el aporte de J1 en la ventana final debe ser ~0."""
    result = read_tail_window_observables(tiny_out_file, drain_down_hours=1.0, tail_window_hours=1.0)
    assert result["node_total_inflow_lps"]["J1"][-1] == pytest.approx(0.0, abs=0.5)


def test_part_to_dict_normalizes_single_object_series_id(tiny_out_file):
    """SwmmOutput.get_part() devuelve una pandas.Series (no DataFrame) cuando
    la seleccion resuelve a un unico objeto (p.ej. label="J1"). En ese caso
    swmm_api todavia NO separo el label interno de columna: part.name llega
    como "node;J1;total_inflow" (kind;label;variable unidos por
    SwmmOutput._COL_SEP), no como "J1". Este test fuerza esa seleccion de un
    solo objeto directamente contra el .out real y comprueba que
    _part_to_dict() devuelve el id limpio, no el label crudo."""
    from swmm_api import SwmmOutput

    from swmm_resilience.validation.drainage_state import _part_to_dict

    with SwmmOutput(str(tiny_out_file)) as out:
        single = out.get_part(kind="node", label="J1", variable="total_inflow")

    # Confirma que esta seleccion realmente cae en el caso Series (no
    # DataFrame) -- si esto deja de ser cierto, el test ya no prueba nada.
    assert not hasattr(single, "columns")
    assert single.name == "node;J1;total_inflow"

    result = _part_to_dict(single)
    assert set(result.keys()) == {"J1"}
    assert len(result["J1"]) == len(single)
