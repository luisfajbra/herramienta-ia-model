import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.bench.quality import DatasetQualityError, build_quality_report
from swmm_resilience.ml.bench.schemas import KEY_COLUMNS
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17


def _frames(n_nodes=4, factors=(1.0, 2.0, 3.0)):
    keys_rows, feature_rows, clf, reg = [], [], [], []
    for run_id, factor in enumerate(factors, start=1):
        for node_idx in range(n_nodes):
            flooded = 1 if (node_idx < 2 and factor >= 2.0) else 0
            keys_rows.append(
                {
                    "run_id": run_id,
                    "network_id": 1,
                    "scenario_id": run_id,
                    "scenario_key": f"base@{factor}",
                    "scenario_kind": "base",
                    "node_id": f"N{node_idx}",
                    "factor_mult": factor,
                    "shape_id": "base",
                }
            )
            feature_rows.append({col: float(node_idx + 1) for col in FEATURE_COLUMNS_V17})
            clf.append(flooded)
            reg.append(50.0 * factor if flooded else 0.0)
    keys = pd.DataFrame(keys_rows, columns=list(KEY_COLUMNS))
    X = pd.DataFrame(feature_rows, columns=list(FEATURE_COLUMNS_V17))
    return keys, X, pd.Series(clf, name="inunda"), pd.Series(reg, name="vol_inundacion_m3")


def test_report_counts_rows_runs_and_class_balance():
    keys, X, y_clf, y_reg = _frames()
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)

    assert report["n_rows"] == 12
    assert report["n_runs"] == 3
    assert report["class_balance"]["n_flooded"] == 4
    assert report["class_balance"]["n_not_flooded"] == 8
    assert report["class_balance"]["flooded_ratio"] == pytest.approx(4 / 12)


def test_report_breaks_down_rows_by_factor_and_shape():
    keys, X, y_clf, y_reg = _frames()
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)

    assert report["rows_by_factor"] == {"1.0": 4, "2.0": 4, "3.0": 4}
    assert report["rows_by_shape"] == {"base": 12}
    assert report["flooded_by_factor"] == {"1.0": 0, "2.0": 2, "3.0": 2}


def test_report_lists_nulls_against_what_the_contract_allows():
    keys, X, y_clf, y_reg = _frames()
    X = X.copy()
    X.loc[0, "diam_max_in"] = np.nan       # nullable segun el contrato
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)

    assert report["nulls_by_column"]["diam_max_in"] == 1
    assert report["nulls_by_column"]["elev_fondo"] == 0
    assert "diam_max_in" in report["nullable_columns"]
    assert report["unexpected_null_columns"] == []


def test_report_flags_nulls_in_a_required_column():
    keys, X, y_clf, y_reg = _frames()
    X = X.copy()
    X.loc[0, "elev_fondo"] = np.nan        # NO es nullable
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)

    assert report["unexpected_null_columns"] == ["elev_fondo"]


def test_report_lists_constant_columns():
    keys, X, y_clf, y_reg = _frames()
    X = X.copy()
    X["prof_max"] = 1.5
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)

    assert "prof_max" in report["constant_columns"]
    assert "elev_fondo" not in report["constant_columns"]


def test_report_includes_percentiles_per_feature():
    keys, X, y_clf, y_reg = _frames()
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)

    stats = report["feature_stats"]["elev_fondo"]
    assert set(stats) == {"min", "p1", "p25", "p50", "p75", "p99", "max"}
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(4.0)


def test_report_counts_threshold_disagreements_without_raising():
    """inunda viene persistido; si discrepa del umbral se reporta, no se corrige."""
    keys, X, y_clf, y_reg = _frames()
    y_reg = y_reg.copy()
    y_reg.iloc[0] = 99.0            # volumen alto pero inunda == 0
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)

    assert report["threshold_disagreements"] == 1


def test_duplicate_run_node_pairs_are_a_hard_error():
    keys, X, y_clf, y_reg = _frames()
    keys = keys.copy()
    keys.loc[1, "node_id"] = "N0"    # duplica (run_id=1, node_id=N0)

    with pytest.raises(DatasetQualityError, match="duplicad"):
        build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)


def test_zero_flooded_rows_is_a_hard_error():
    keys, X, y_clf, y_reg = _frames()
    y_clf = pd.Series([0] * len(y_clf), name="inunda")

    with pytest.raises(DatasetQualityError, match="ninguna fila inundada"):
        build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)
