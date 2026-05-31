import pandas as pd

from swmm_resilience.simulation.runner import (
    _flood_volume_from_timeseries_m3,
    _merge_rpt_flooding_metrics,
)


def test_flood_volume_from_timeseries_integrates_lps_to_m3():
    rows = [
        {"node_id": "J1", "time_sec": 60.0, "flooding_lps": 10.0},
        {"node_id": "J1", "time_sec": 120.0, "flooding_lps": 20.0},
        {"node_id": "J1", "time_sec": 180.0, "flooding_lps": 0.0},
        {"node_id": "J2", "time_sec": 60.0, "flooding_lps": 0.0},
        {"node_id": "J2", "time_sec": 120.0, "flooding_lps": 5.0},
    ]

    volumes = _flood_volume_from_timeseries_m3(rows)

    assert volumes["J1"] == 1.8
    assert volumes["J2"] == 0.3


def test_merge_rpt_flooding_metrics_prefers_rpt_volume_and_duration():
    node_records = [
        {
            "node_id": "J1",
            "peak_flooding_lps": 10.0,
            "total_flood_volume_m3": 1.8,
            "flooding_duration_min": 3.0,
            "flooded": 1,
        }
    ]
    rpt_df = pd.DataFrame(
        [
            {
                "node_id": "J1",
                "flooding_volume_m3": 2.5,
                "flooding_duration_min": 4.0,
            }
        ]
    )

    _merge_rpt_flooding_metrics(node_records, rpt_df)

    assert node_records[0]["total_flood_volume_m3"] == 2.5
    assert node_records[0]["flooding_duration_min"] == 4.0
    assert node_records[0]["flooded"] == 1


def test_merge_rpt_flooding_metrics_marks_flooded_from_duration_without_volume():
    node_records = [
        {
            "node_id": "J1",
            "peak_flooding_lps": 0.0,
            "total_flood_volume_m3": 0.0,
            "flooding_duration_min": 0.0,
            "flooded": 0,
        },
        {
            "node_id": "J2",
            "peak_flooding_lps": 0.0,
            "total_flood_volume_m3": 0.0,
            "flooding_duration_min": 0.0,
            "flooded": 0,
        },
    ]
    rpt_df = pd.DataFrame(
        [
            {
                "node_id": "J1",
                "flooding_volume_m3": 0.0,
                "flooding_duration_min": 5.0,
            },
            {
                "node_id": "J2",
                "flooding_volume_m3": float("nan"),
                "flooding_duration_min": 2.0,
            },
        ]
    )

    _merge_rpt_flooding_metrics(node_records, rpt_df)

    assert node_records[0]["total_flood_volume_m3"] == 0.0
    assert node_records[0]["flooding_duration_min"] == 5.0
    assert node_records[0]["flooded"] == 1
    assert node_records[1]["total_flood_volume_m3"] == 0.0
    assert node_records[1]["flooding_duration_min"] == 2.0
    assert node_records[1]["flooded"] == 1

    missing_volume_records = [
        {
            "node_id": "J3",
            "peak_flooding_lps": 0.0,
            "total_flood_volume_m3": 0.0,
            "flooding_duration_min": 0.0,
            "flooded": 0,
        }
    ]
    missing_volume_df = pd.DataFrame(
        [{"node_id": "J3", "flooding_duration_min": 3.0}]
    )

    _merge_rpt_flooding_metrics(missing_volume_records, missing_volume_df)

    assert missing_volume_records[0]["total_flood_volume_m3"] == 0.0
    assert missing_volume_records[0]["flooding_duration_min"] == 3.0
    assert missing_volume_records[0]["flooded"] == 1
