from pathlib import Path

from swmm_resilience.simulation.swmm_api_io import read_continuity_error


def test_read_continuity_error_parses_worst_value(tmp_path):
    rpt = tmp_path / "run.rpt"
    rpt.write_text(
        "\n".join(
            [
                "  Runoff Quantity Continuity",
                "  Continuity Error (%) .....        -0.05",
                "  Flow Routing Continuity",
                "  Continuity Error (%) .....        -7.31",
            ]
        ),
        encoding="utf-8",
    )
    assert read_continuity_error(rpt) == -7.31


def test_read_continuity_error_returns_none_for_missing_file(tmp_path):
    assert read_continuity_error(tmp_path / "does_not_exist.rpt") is None


def test_read_continuity_error_returns_none_when_no_match(tmp_path):
    rpt = tmp_path / "run.rpt"
    rpt.write_text("no continuity section here", encoding="utf-8")
    assert read_continuity_error(rpt) is None
