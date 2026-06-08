import pandas as pd

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
