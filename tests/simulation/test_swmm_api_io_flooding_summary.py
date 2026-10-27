from types import SimpleNamespace

import pandas as pd
import pytest

from swmm_resilience.simulation import swmm_api_io


def _read_summary(monkeypatch, columns):
    source = pd.DataFrame(columns, index=pd.Index(["J1"], name="Node"))
    monkeypatch.setattr(
        swmm_api_io, "read_rpt_file", lambda *_args, **_kwargs: SimpleNamespace(node_flooding_summary=source)
    )
    monkeypatch.setattr(swmm_api_io, "_require_swmm_api", lambda: None)
    return swmm_api_io.read_node_flooding_summary("unused.rpt")


def test_selects_total_flood_volume_when_ponded_volume_is_also_reported(monkeypatch):
    result = _read_summary(
        monkeypatch,
        {
            "Total_Flooding_Volume_10^6_ltr": [0.002],
            "Maximum_Ponded_Volume_1000_m3": [5.0],
            "Flooding_Duration_hours": [0.25],
        },
    )

    assert result.loc[0, "flooding_volume_m3"] == pytest.approx(2.0)
    assert result.loc[0, "flooding_duration_min"] == pytest.approx(15.0)


def test_converts_total_flood_volume_reported_in_million_gallons(monkeypatch):
    result = _read_summary(
        monkeypatch,
        {"Total_Flooding_Volume_10^6_gal": [0.001]},
    )

    assert result.loc[0, "flooding_volume_m3"] == pytest.approx(3.785411784)


def test_rejects_ambiguous_volume_columns(monkeypatch):
    result = _read_summary(
        monkeypatch,
        {
            "Flooding_Volume_10^6_ltr": [0.002],
            "Ponded_Volume_1000_m3": [5.0],
        },
    )

    assert result is None


def test_rejects_ponded_volume_when_total_flood_volume_is_absent(monkeypatch):
    with pytest.warns(UserWarning, match="Cannot identify one total flooding-volume column"):
        result = _read_summary(
            monkeypatch,
            {"Maximum_Ponded_Volume_1000_m3": [5.0]},
        )

    assert result is None
