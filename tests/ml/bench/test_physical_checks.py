# tests/ml/bench/test_physical_checks.py
import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.bench.physical_checks import (
    cap_at_tributary_volume, node_monotonicity_violations,
    total_monotonicity_violations, tributary_bound_violations)


def series_frame(predicted, truth, factors=(1.0, 2.0, 3.0), network=1, shape="A", node="n1"):
    return pd.DataFrame({"network_id": network, "run_id": range(len(factors)), "node_id": node,
                         "shape_id": shape, "factor_mult": list(factors),
                         "volume_routed_m3": predicted, "vol_inundacion_m3": truth,
                         "volumen_tributario_integrado_m3": 100.0})


def test_bound_violations_use_tolerance_and_report_ratio():
    frame = series_frame([50.0, 100.5, 250.0], [0.0, 0.0, 0.0])
    found = tributary_bound_violations(frame, tolerance_m3=1.0)
    assert found.factor_mult.tolist() == [3.0]
    assert found.ratio.tolist() == [2.5]


def test_cap_never_exceeds_tributary_and_rejects_invalid_bound():
    assert cap_at_tributary_volume([5.0, 300.0], [100.0, 100.0]).tolist() == [5.0, 100.0]
    with pytest.raises(ValueError, match="finite"):
        cap_at_tributary_volume([1.0], [np.nan])
    with pytest.raises(ValueError, match="negative"):
        cap_at_tributary_volume([1.0], [-1.0])
    with pytest.raises(ValueError, match="shape"):
        cap_at_tributary_volume([1.0, 2.0], [1.0])


def test_total_monotonicity_flags_drop_only_when_truth_does_not_drop():
    frame = series_frame([10.0, 30.0, 20.0], [5.0, 10.0, 15.0])
    found = total_monotonicity_violations(frame, "volume_routed_m3")
    assert found.factor_mult.tolist() == [3.0]
    assert found.predicted_drop_m3.tolist() == [-10.0]
    truth_drops = series_frame([10.0, 30.0, 20.0], [5.0, 10.0, 8.0])
    assert total_monotonicity_violations(truth_drops, "volume_routed_m3").empty


def test_node_monotonicity_counts_steps_per_node_and_shape():
    a = series_frame([10.0, 30.0, 20.0], [5.0, 10.0, 15.0], node="n1")
    b = series_frame([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], node="n2")
    result = node_monotonicity_violations(pd.concat([a, b], ignore_index=True), "volume_routed_m3")
    assert result == {"violations": 1, "steps": 4}
