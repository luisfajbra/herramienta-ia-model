import pandas as pd
import pytest

from swmm_resilience.extraction import labels


def _valid_report(path, body="No nodes were flooded."):
    path.write_text(
        "\n".join(["Node Flooding Summary", "*********************", body, "", "Outfall Loading Summary"]),
        encoding="utf-8",
    )
    return path


def test_extract_labels_marks_missing_rpt_nodes_as_not_flooded(monkeypatch, tmp_path):
    def fake_read_node_flooding_summary(path):
        return pd.DataFrame({"node_id": ["J2"], "flooding_volume_m3": [15.0]})

    monkeypatch.setattr(labels, "read_node_flooding_summary", fake_read_node_flooding_summary)

    rpt = _valid_report(tmp_path / "run.rpt", "J2 0.0 0.0 0 00:00 0.015 0.000")
    df = labels.extract_labels(rpt, ["J1", "J2"], threshold_m3=0.0)

    assert df.to_dict("records") == [
        {"node_id": "J1", "vol_inundacion_m3": 0.0, "inunda": 0},
        {"node_id": "J2", "vol_inundacion_m3": 15.0, "inunda": 1},
    ]


def test_extract_labels_applies_threshold(monkeypatch, tmp_path):
    def fake_read_node_flooding_summary(path):
        return pd.DataFrame({"node_id": ["J1"], "flooding_volume_m3": [0.5]})

    monkeypatch.setattr(labels, "read_node_flooding_summary", fake_read_node_flooding_summary)

    rpt = _valid_report(tmp_path / "run.rpt", "J1 0.0 0.0 0 00:00 0.0005 0.000")
    df = labels.extract_labels(rpt, ["J1"], threshold_m3=1.0)

    assert df.iloc[0]["vol_inundacion_m3"] == 0.5
    assert df.iloc[0]["inunda"] == 0


def test_extract_labels_threshold_is_inclusive(monkeypatch, tmp_path):
    """Volume exactly at the threshold counts as flooded (>= rule)."""
    def fake_read_node_flooding_summary(path):
        return pd.DataFrame({"node_id": ["J1"], "flooding_volume_m3": [1.0]})

    monkeypatch.setattr(labels, "read_node_flooding_summary", fake_read_node_flooding_summary)

    rpt = _valid_report(tmp_path / "run.rpt", "J1 0.0 0.0 0 00:00 0.001 0.000")
    df = labels.extract_labels(rpt, ["J1"], threshold_m3=1.0)

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


@pytest.mark.parametrize(
    ("rpt_path", "contents", "expected_code"),
    [
        ("missing.rpt", None, "missing_report"),
        ("missing_summary.rpt", "SWMM report without this table", "missing_flooding_summary"),
        ("truncated.rpt", "Node Flooding Summary\n***************", "invalid_flooding_summary"),
        ("bad_value.rpt", "Node Flooding Summary\n---\nJ1 0.0 0.0 0 00:00 NaN 0.0", "invalid_flooding_volume"),
        ("negative.rpt", "Node Flooding Summary\n---\nJ1 0.0 0.0 0 00:00 -0.1 0.0", "invalid_flooding_volume"),
        ("invalid_token.rpt", "Node Flooding Summary\n---\nJ1 0.0 0.0 0 00:00 bad 0.0", "invalid_flooding_summary"),
        (
            "duplicate.rpt",
            "Node Flooding Summary\n---\nJ1 0.0 0.0 0 00:00 0.1 0.0\nJ1 0.0 0.0 0 00:00 0.2 0.0",
            "duplicate_flooding_node",
        ),
    ],
)
def test_invalid_reports_fail_closed(monkeypatch, tmp_path, rpt_path, contents, expected_code):
    monkeypatch.setattr(labels, "read_node_flooding_summary", lambda _path: None)
    path = tmp_path / rpt_path
    if contents is not None:
        path.write_text(contents + "\nOutfall Loading Summary", encoding="utf-8")

    with pytest.raises(labels.FloodingReportError) as exc:
        labels.extract_labels(path, ["J1"])

    assert exc.value.code == expected_code


def test_explicit_no_flood_summary_is_valid_empty_table(monkeypatch, tmp_path):
    monkeypatch.setattr(labels, "read_node_flooding_summary", lambda _path: None)
    rpt = _valid_report(tmp_path / "run.rpt")

    result = labels.read_flooding_volumes(rpt)

    assert result.empty
    assert result.columns.tolist() == ["node_id", "flooding_volume_m3"]


def test_empty_api_result_without_explicit_no_flood_marker_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(labels, "read_node_flooding_summary", lambda _path: pd.DataFrame())
    rpt = _valid_report(tmp_path / "run.rpt", "Summary table was truncated")

    with pytest.raises(labels.FloodingReportError) as exc:
        labels.read_flooding_volumes(rpt)

    assert exc.value.code == "invalid_flooding_summary"


def test_report_parser_manifest_records_parser_version_and_exact_report_hash(
    monkeypatch, tmp_path
):
    import hashlib

    monkeypatch.setattr(
        labels,
        "read_node_flooding_summary",
        lambda _path: pd.DataFrame({"node_id": ["J1"], "flooding_volume_m3": [2.5]}),
    )
    rpt = _valid_report(tmp_path / "manifest.rpt", "J1 0 0 0 00:00 0.0025 0")

    volumes, manifest = labels.read_flooding_volumes_with_manifest(rpt)

    assert volumes.loc[0, "flooding_volume_m3"] == 2.5
    assert manifest["parser"] == "swmm_api.node_flooding_summary"
    assert manifest["parser_version"]
    assert manifest["report_sha256"] == hashlib.sha256(rpt.read_bytes()).hexdigest()


def test_extract_labels_attaches_parser_manifest_to_dataframe(monkeypatch, tmp_path):
    import hashlib

    monkeypatch.setattr(labels, "read_node_flooding_summary", lambda _path: None)
    rpt = _valid_report(tmp_path / "fallback_manifest.rpt")

    result = labels.extract_labels(rpt, ["J1"])

    manifest = result.attrs["flooding_report_manifest"]
    assert manifest["parser"] == "strict_text_v1"
    assert manifest["report_sha256"] == hashlib.sha256(rpt.read_bytes()).hexdigest()


def test_fallback_rejects_truncated_row_after_a_valid_row(monkeypatch, tmp_path):
    monkeypatch.setattr(labels, "read_node_flooding_summary", lambda _path: None)
    rpt = tmp_path / "partial.rpt"
    rpt.write_text(
        "\n".join(
            [
                "Node Flooding Summary",
                "*********************",
                "Node Hours Flooded Maximum Rate Time Total Flood Volume Maximum Ponded Volume",
                "ID   hours         LPS       days hr:min 10^6 ltr        1000 m3",
                "--------------------------------------------------------------",
                "J1 0.1 2.0 0 00:30 0.01 0.0",
                "J2 0.0 0.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(labels.FloodingReportError) as exc:
        labels.extract_labels(rpt, ["J1", "J2"])

    assert exc.value.code == "invalid_flooding_summary"


def test_api_result_is_rejected_when_flooding_section_has_no_completion_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(
        labels,
        "read_node_flooding_summary",
        lambda _path: pd.DataFrame({"node_id": ["J1"], "flooding_volume_m3": [0.0]}),
    )
    rpt = tmp_path / "truncated_after_rows.rpt"
    rpt.write_text("Node Flooding Summary\n---\nJ1 0 0 0 00:00 0 0\n", encoding="utf-8")

    with pytest.raises(labels.FloodingReportError) as exc:
        labels.read_flooding_volumes(rpt)

    assert exc.value.code == "invalid_flooding_summary"


def test_fallback_rejects_node_id_only_row_after_valid_data(monkeypatch, tmp_path):
    monkeypatch.setattr(labels, "read_node_flooding_summary", lambda _path: None)
    rpt = tmp_path / "id_only_row.rpt"
    rpt.write_text(
        "\n".join([
            "Node Flooding Summary", "*********************",
            "Node Hours Flooded Maximum Rate Time Total Flood Volume Maximum Ponded Volume",
            "ID hours LPS days hr:min 10^6 ltr 1000 m3", "---",
            "J1 0.1 2.0 0 00:30 0.01 0.0", "J2",
            "Outfall Loading Summary",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(labels.FloodingReportError) as exc:
        labels.read_flooding_volumes(rpt)

    assert exc.value.code == "invalid_flooding_summary"
