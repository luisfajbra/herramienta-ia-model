import math

import pandas as pd
import pytest

from swmm_resilience.ml.bench.holdout_audit import (
    continuity_subset, leave_one_shape_out_macro, macro_network_rmse, rmse_by_shape_network)


def frame_with_errors():
    # Errors: net1/A=3, net1/B=0, net2/A=4, net2/B=0 (truth 10 m3 everywhere).
    rows = [(1, 1, "n1", "A", 1.0, 3.0), (1, 2, "n1", "B", 1.0, 0.0),
            (2, 3, "m1", "A", 1.0, 4.0), (2, 4, "m1", "B", 1.0, 0.0)]
    return pd.DataFrame([{"network_id": n, "run_id": r, "node_id": node, "shape_id": s,
                          "factor_mult": f, "vol_inundacion_m3": 10.0, "volume_routed_m3": 10.0 + e}
                         for n, r, node, s, f, e in rows])


def test_macro_rmse_averages_network_rmse():
    assert math.isclose(macro_network_rmse(frame_with_errors()),
                        (math.sqrt(4.5) + math.sqrt(8.0)) / 2, rel_tol=1e-12)


def test_rmse_by_shape_network_reports_share_of_squared_error():
    table = rmse_by_shape_network(frame_with_errors()).set_index(["network_id", "shape_id"])
    assert table.loc[(1, "A"), "rmse_m3"] == 3.0
    assert math.isclose(table.loc[(1, "A"), "sse_share"], 9 / 25)
    assert math.isclose(table.loc[(2, "A"), "sse_share"], 16 / 25)
    assert table.loc[(2, "B"), "sse_share"] == 0.0
    assert table.n_pairs.tolist() == [1, 1, 1, 1]


def test_leave_one_shape_out_reports_all_then_each_exclusion():
    table = leave_one_shape_out_macro(frame_with_errors())
    assert table.excluded_shape.tolist() == [None, "A", "B"]
    assert table.macro_network_rmse_m3.iloc[1] == 0.0
    assert table.macro_network_rmse_m3.iloc[2] == 3.5


def test_leave_one_shape_out_rejects_dropping_a_whole_network():
    frame = frame_with_errors()
    frame.loc[frame.network_id == 2, "shape_id"] = "only_net2"
    with pytest.raises(ValueError, match="whole network"):
        leave_one_shape_out_macro(frame)


def test_continuity_subset_keeps_runs_within_absolute_limit():
    continuity = pd.Series({1: 0.5, 2: 3.0, 3: -1.0, 4: 4.9})
    subset = continuity_subset(frame_with_errors(), continuity, 2.0)
    assert sorted(subset.run_id.unique()) == [1, 3]


def test_continuity_subset_rejects_missing_runs_and_bad_limit():
    with pytest.raises(ValueError, match="missing"):
        continuity_subset(frame_with_errors(), pd.Series({1: 0.5}), 2.0)
    with pytest.raises(ValueError, match="positive"):
        continuity_subset(frame_with_errors(), pd.Series({1: 0, 2: 0, 3: 0, 4: 0}), 0.0)


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="lacks columns"):
        macro_network_rmse(frame_with_errors().drop(columns=["shape_id"]))
