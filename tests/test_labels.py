import pandas as pd
import pytest

from swmm_resilience.extraction import labels


def test_extract_labels_marks_missing_rpt_nodes_as_not_flooded(monkeypatch, tmp_path):
    def fake_read_node_flooding_summary(path):
        return pd.DataFrame({"node_id": ["J2"], "flooding_volume_m3": [15.0]})

    monkeypatch.setattr(labels, "read_node_flooding_summary", fake_read_node_flooding_summary)

    df = labels.extract_labels(tmp_path / "run.rpt", ["J1", "J2"], threshold_m3=0.0)

    assert df.to_dict("records") == [
        {"node_id": "J1", "vol_inundacion_m3": 0.0, "inunda": 0},
        {"node_id": "J2", "vol_inundacion_m3": 15.0, "inunda": 1},
    ]


def test_extract_labels_applies_threshold(monkeypatch, tmp_path):
    def fake_read_node_flooding_summary(path):
        return pd.DataFrame({"node_id": ["J1"], "flooding_volume_m3": [0.5]})

    monkeypatch.setattr(labels, "read_node_flooding_summary", fake_read_node_flooding_summary)

    df = labels.extract_labels(tmp_path / "run.rpt", ["J1"], threshold_m3=1.0)

    assert df.iloc[0]["vol_inundacion_m3"] == 0.5
    assert df.iloc[0]["inunda"] == 0


def test_extract_labels_threshold_is_inclusive(monkeypatch, tmp_path):
    """Volume exactly at the threshold counts as flooded (>= rule)."""
    def fake_read_node_flooding_summary(path):
        return pd.DataFrame({"node_id": ["J1"], "flooding_volume_m3": [1.0]})

    monkeypatch.setattr(labels, "read_node_flooding_summary", fake_read_node_flooding_summary)

    df = labels.extract_labels(tmp_path / "run.rpt", ["J1"], threshold_m3=1.0)

    assert df.iloc[0]["inunda"] == 1


def test_flood_label_zero_threshold_does_not_flag_dry_nodes():
    """threshold <= 0 degrades to vol > 0 so vol=0 nodes are never flooded."""
    vols = pd.Series([0.0, 0.5])
    assert labels.flood_label(vols, 0.0).tolist() == [0, 1]


def test_read_flooding_volumes_text_fallback(monkeypatch, tmp_path):
    """When swmm_api fails, the text parser extracts volumes from the .rpt."""
    def boom(path):
        raise RuntimeError("swmm_api unavailable")

    monkeypatch.setattr(labels, "read_node_flooding_summary", boom)

    rpt = tmp_path / "run.rpt"
    rpt.write_text(
        "\n".join(
            [
                "  Node Flooding Summary",
                "  *******************",
                "  ",
                "  ------------------------------------------------",
                "  J7    1.20    150.00    0   12.50    0.003",
                "",
                "  OUTFALL LOADING SUMMARY",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.warns(UserWarning):
        df = labels.read_flooding_volumes(rpt)

    assert df["node_id"].tolist() == ["J7"]
    assert df["flooding_volume_m3"].iloc[0] == pytest.approx(3.0)
