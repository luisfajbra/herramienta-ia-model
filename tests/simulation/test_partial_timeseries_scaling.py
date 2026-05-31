from pathlib import Path

import pytest

from swmm_resilience.config import SCENARIO_MODE_TIMESERIES
from swmm_resilience.simulation.swmm_api_io import load_inp, write_scaled_inp


pytest.importorskip("swmm_api")


def _write_shared_timeseries_inp(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[TITLE]",
                "shared timeseries",
                "[OPTIONS]",
                "FLOW_UNITS           LPS",
                "INFILTRATION         HORTON",
                "FLOW_ROUTING         KINWAVE",
                "START_DATE           01/01/2020",
                "START_TIME           00:00:00",
                "REPORT_START_DATE    01/01/2020",
                "REPORT_START_TIME    00:00:00",
                "END_DATE             01/01/2020",
                "END_TIME             01:00:00",
                "SWEEP_START          01/01",
                "SWEEP_END            12/31",
                "DRY_DAYS             0",
                "REPORT_STEP          00:05:00",
                "WET_STEP             00:05:00",
                "DRY_STEP             00:05:00",
                "ROUTING_STEP         0:05:00",
                "[JUNCTIONS]",
                "J1 0 1 0 0 0",
                "J2 0 1 0 0 0",
                "[OUTFALLS]",
                "O1 0 FREE NO",
                "[CONDUITS]",
                "C1 J1 O1 100 0.013 0 0 0 0",
                "C2 J2 O1 100 0.013 0 0 0 0",
                "[XSECTIONS]",
                "C1 CIRCULAR 0.3 0 0 0 1",
                "C2 CIRCULAR 0.3 0 0 0 1",
                "[INFLOWS]",
                "J1 FLOW Shared FLOW 1.0 1.0 0.0",
                "J2 FLOW Shared FLOW 1.0 1.0 0.0",
                "[TIMESERIES]",
                "Shared 0:00 10",
                "Shared 1:00 20",
                "[END]",
            ]
        ),
        encoding="utf-8",
    )


def test_partial_scaling_duplicates_shared_timeseries_for_selected_node(tmp_path):
    inp_path = tmp_path / "shared.inp"
    out_path = tmp_path / "scaled.inp"
    _write_shared_timeseries_inp(inp_path)

    write_scaled_inp(
        inp_path,
        multiplier=2.0,
        target_nodes={"J1"},
        output_file=out_path,
        scenario_mode=SCENARIO_MODE_TIMESERIES,
    )

    scaled = load_inp(out_path)
    j1 = scaled["INFLOWS"][("J1", "FLOW")]
    j2 = scaled["INFLOWS"][("J2", "FLOW")]
    assert str(j1.time_series) != str(j2.time_series)

    j1_values = [
        value for _time, value in scaled["TIMESERIES"][str(j1.time_series)].data
    ]
    j2_values = [
        value for _time, value in scaled["TIMESERIES"][str(j2.time_series)].data
    ]
    assert j1_values == [20.0, 40.0]
    assert j2_values == [10.0, 20.0]
