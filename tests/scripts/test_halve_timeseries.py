import importlib
from pathlib import Path

import pytest

INP = """[TIMESERIES]
;;Name  Time  Value
TS1     0:00  10.000
TS1     1:00  20.000

[OTHER]
KEEP    5.0
"""


def test_importing_the_module_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module("halve_timeseries")
    importlib.reload(module)
    assert not any(tmp_path.iterdir())


def test_halves_only_the_timeseries_section(tmp_path):
    from halve_timeseries import halve_timeseries

    src = tmp_path / "in.inp"
    dst = tmp_path / "out.inp"
    src.write_text(INP, encoding="utf-8")

    count = halve_timeseries(src, dst)

    out = dst.read_text(encoding="utf-8")
    assert count == 2
    assert "5.000" in out and "10.000" in out
    assert "KEEP    5.0" in out


def test_refuses_to_overwrite_an_existing_destination_without_force(tmp_path):
    from halve_timeseries import halve_timeseries

    src = tmp_path / "in.inp"
    dst = tmp_path / "out.inp"
    src.write_text(INP, encoding="utf-8")
    dst.write_text("PRECIOUS", encoding="utf-8")

    with pytest.raises(FileExistsError):
        halve_timeseries(src, dst)
    assert dst.read_text(encoding="utf-8") == "PRECIOUS"

    halve_timeseries(src, dst, force=True)
    assert dst.read_text(encoding="utf-8") != "PRECIOUS"


def test_refuses_to_use_the_source_as_destination(tmp_path):
    from halve_timeseries import halve_timeseries

    src = tmp_path / "in.inp"
    src.write_text(INP, encoding="utf-8")
    with pytest.raises(ValueError, match="mismo archivo"):
        halve_timeseries(src, src, force=True)
