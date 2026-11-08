# tests/ml/bench/test_partition_coverage.py
import pandas as pd
import pytest

from swmm_resilience.ml.bench.partition_coverage import (
    inflow_coverage, run_inflow_volumes, shape_representative_volume, stratified_shape_draw)


def node_rows():
    rows = []
    for run, network, shape, factor, per_node in [(1, 1, "dev_a", 1.0, 10.0), (2, 1, "dev_b", 1.0, 30.0),
                                                   (3, 1, "held", 1.0, 40.0), (4, 2, "dev_a", 1.0, 5.0),
                                                   (5, 2, "held", 1.0, 4.0)]:
        for node in ("x", "y"):
            rows.append({"network_id": network, "run_id": run, "node_id": node, "shape_id": shape,
                         "factor_mult": factor, "volumen_local_integrado_m3": per_node})
    return pd.DataFrame(rows)


def test_run_inflow_volumes_sums_nodes_per_run():
    volumes = run_inflow_volumes(node_rows()).set_index("run_id")
    assert volumes.loc[2, "total_inflow_m3"] == 60.0
    assert volumes.loc[2, "shape_id"] == "dev_b"


def test_run_inflow_volumes_rejects_inconsistent_run_identity_or_missing_volume():
    rows = node_rows()
    rows.loc[0, "shape_id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        run_inflow_volumes(rows)
    rows = node_rows()
    rows.loc[0, "volumen_local_integrado_m3"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        run_inflow_volumes(rows)


def test_inflow_coverage_flags_holdout_above_development_max():
    table = inflow_coverage(run_inflow_volumes(node_rows()), [1, 2, 4], [3, 5]).set_index("network_id")
    assert table.loc[1, "development_max_inflow_m3"] == 60.0
    assert table.loc[1, "holdout_max_shape"] == "held"
    assert table.loc[1, "holdout_runs_above_development_max"] == 1
    assert table.loc[1, "max_ratio_to_development_max"] == pytest.approx(80 / 60)
    assert table.loc[2, "holdout_runs_above_development_max"] == 0


def test_inflow_coverage_rejects_overlapping_or_unknown_runs():
    volumes = run_inflow_volumes(node_rows())
    with pytest.raises(ValueError, match="overlap"):
        inflow_coverage(volumes, [1, 3], [3])
    with pytest.raises(ValueError, match="unknown"):
        inflow_coverage(volumes, [1, 99], [3])


def test_shape_representative_volume_is_normalised_per_network():
    representative = shape_representative_volume(run_inflow_volumes(node_rows()))
    assert representative["held"] == 1.0  # network 1 maximum
    assert representative["dev_a"] == 1.0  # network 2 maximum
    assert representative["dev_b"] == pytest.approx(60 / 80)


def test_stratified_draw_keeps_largest_in_development_and_is_reproducible():
    volumes = pd.Series({f"s{i}": float(i) for i in range(1, 11)})
    drawn = stratified_shape_draw(volumes, k=3, seed=42)
    assert drawn == stratified_shape_draw(volumes, k=3, seed=42)
    assert "s10" not in drawn
    strata = [{"s1", "s2", "s3"}, {"s4", "s5", "s6"}, {"s7", "s8", "s9"}]
    assert all(len(set(drawn) & stratum) == 1 for stratum in strata)


def test_stratified_draw_rejects_invalid_requests():
    with pytest.raises(ValueError, match="not enough"):
        stratified_shape_draw(pd.Series({"a": 1.0, "b": 2.0}), k=2, seed=1)
    with pytest.raises(ValueError, match="finite"):
        stratified_shape_draw(pd.Series({"a": float("nan"), "b": 2.0, "c": 3.0}), k=1, seed=1)


@pytest.mark.parametrize("keep", [10, 12])
def test_stratified_draw_rejects_keep_top_not_smaller_than_shape_count(keep):
    volumes = pd.Series({f"s{i}": float(i) for i in range(1, 11)})
    with pytest.raises(ValueError, match="must be smaller than the number of shapes"):
        stratified_shape_draw(volumes, k=1, seed=1, keep_top_in_development=keep)
