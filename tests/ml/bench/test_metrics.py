import numpy as np
import pytest

from swmm_resilience.ml.bench.metrics import (
    classifier_metrics,
    confusion_counts,
    end_to_end_metrics,
    end_to_end_pooled_metrics,
    fold_classifier_metrics,
    mean_metrics,
    nse,
    pooled_classifier_metrics,
    pooled_regressor_metrics,
    regressor_oracle_metrics,
    shape_breakdown_entry,
    summarize_fold_metrics,
)


def test_nse_is_one_for_a_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert nse(y, y) == pytest.approx(1.0)


def test_nse_is_zero_when_the_prediction_equals_the_mean():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert nse(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_nse_of_a_constant_truth_is_one_only_when_exact():
    y = np.array([5.0, 5.0, 5.0])
    assert nse(y, y) == pytest.approx(1.0)
    assert nse(y, np.array([5.0, 5.0, 6.0])) == pytest.approx(0.0)


def test_classifier_metrics_on_a_known_confusion_matrix():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.2, 0.1])

    result = classifier_metrics(y_true, y_pred, y_prob)

    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["auc_roc"] == pytest.approx(1.0)


def test_auc_is_nan_when_the_fold_has_a_single_class():
    y_true = np.array([1, 1, 1])
    result = classifier_metrics(y_true, np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))
    assert np.isnan(result["auc_roc"])


def test_regressor_oracle_metrics_on_a_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    result = regressor_oracle_metrics(y, y)

    assert result["nse"] == pytest.approx(1.0)
    assert result["log_nse"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["mae"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


def test_regressor_oracle_metrics_on_an_imperfect_wide_range_prediction():
    """A perfect prediction makes rmse/mae symmetric, nse/r2 coincide, and
    log_nse indistinguishable from nse. This uses an imperfect prediction over
    a wide dynamic range so a bug that reverses the (y_true, y_pred) argument
    order, or that drops the np.log1p in log_nse, changes these numbers."""
    y_true = np.array([1.0, 10.0, 100.0, 1000.0])
    y_pred = np.array([2.0, 8.0, 110.0, 900.0])

    result = regressor_oracle_metrics(y_true, y_pred)

    assert result["nse"] == pytest.approx(0.9855955793182168)
    assert result["log_nse"] == pytest.approx(0.98972348644384)
    assert result["rmse"] == pytest.approx(50.2618145315109)
    assert result["mae"] == pytest.approx(28.25)
    assert result["r2"] == pytest.approx(0.9855955793182168)


def test_end_to_end_metrics_report_totals_and_node_accuracy():
    y_true_vol = np.array([0.0, 100.0, 0.0, 50.0])
    y_pred_vol = np.array([0.0, 90.0, 0.0, 0.0])
    y_true_clf = np.array([0, 1, 0, 1])
    y_pred_clf = np.array([0, 1, 0, 0])

    result = end_to_end_metrics(y_true_vol, y_pred_vol, y_true_clf, y_pred_clf)

    assert result["pct_nodos_correctos"] == pytest.approx(0.75)
    assert result["vol_total_pred_m3"] == pytest.approx(90.0)
    assert result["vol_total_real_m3"] == pytest.approx(150.0)
    assert result["rmse_vol_todos_nodos"] == pytest.approx(np.sqrt((10**2 + 50**2) / 4))


def test_mean_metrics_averages_each_key_ignoring_nan():
    folds = [
        {"f1": 0.8, "auc_roc": float("nan")},
        {"f1": 0.6, "auc_roc": 0.9},
    ]
    result = mean_metrics(folds)
    assert result["f1"] == pytest.approx(0.7)
    assert result["auc_roc"] == pytest.approx(0.9)


def test_mean_metrics_of_an_empty_list_is_empty():
    assert mean_metrics([]) == {}


def test_pooled_regressor_metrics_concatenate_before_computing():
    """Es la asimetria heredada: el regresor NO se promedia entre folds.

    Both parts have imperfect, differently-scaled predictions so pooling
    (concatenate-then-score) and averaging (score-then-mean) provably give
    different numbers: pooled rmse=57.76 / nse=0.9845 vs a per-part average
    of rmse=51.0 / nse=0.42. A per-fold-averaging implementation would fail
    these assertions.
    """
    trues = [np.array([10.0, 20.0]), np.array([1000.0])]
    preds = [np.array([12.0, 18.0]), np.array([900.0])]
    result = pooled_regressor_metrics(trues, preds)
    assert result["rmse"] == pytest.approx(57.758116312774604)
    assert result["nse"] == pytest.approx(0.9845284963413378)


def test_pooled_regressor_metrics_of_empty_parts_is_empty():
    assert pooled_regressor_metrics([], []) == {}


def test_confusion_counts_on_a_known_case():
    counts = confusion_counts([1, 1, 0, 0], [1, 0, 0, 1])
    assert counts == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_pooled_classifier_metrics_is_perfect_even_when_a_part_has_no_positives():
    """El caso de M-03: un fold sin inundaciones no debe contar como F1 = 0."""
    true_parts = [np.array([0, 0]), np.array([1, 0])]
    pred_parts = [np.array([0, 0]), np.array([1, 0])]
    prob_parts = [np.array([0.1, 0.1]), np.array([0.9, 0.1])]

    result = pooled_classifier_metrics(true_parts, pred_parts, prob_parts)

    assert result["f1"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["specificity"] == pytest.approx(1.0)
    assert result["auc_roc"] == pytest.approx(1.0)


def test_pooled_classifier_metrics_of_no_parts_is_empty():
    assert pooled_classifier_metrics([], [], []) == {}


def test_pooled_specificity_is_nan_without_negatives():
    result = pooled_classifier_metrics(
        [np.array([1, 1])], [np.array([1, 0])], [np.array([0.9, 0.4])]
    )
    assert np.isnan(result["specificity"])


def test_fold_metrics_are_nan_when_the_fold_has_no_real_positives():
    result = fold_classifier_metrics(np.array([0, 0]), np.array([0, 0]))
    assert np.isnan(result["precision"])
    assert np.isnan(result["recall"])
    assert np.isnan(result["f1"])


def test_fold_precision_is_zero_but_recall_nan_with_only_false_positives():
    result = fold_classifier_metrics(np.array([0, 0]), np.array([1, 0]))
    assert result["precision"] == pytest.approx(0.0)
    assert np.isnan(result["recall"])
    assert np.isnan(result["f1"])


def test_fold_metrics_match_the_textbook_values():
    result = fold_classifier_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2 / 3)


def test_summarize_fold_metrics_ignores_undefined_folds_and_counts_them():
    folds = [
        {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")},
        {"precision": 1.0, "recall": 0.5, "f1": 2 / 3},
        {"precision": 1.0, "recall": 1.0, "f1": 1.0},
    ]
    summary = summarize_fold_metrics(folds)

    assert summary["f1"]["n_undefined"] == 1
    assert summary["f1"]["mean"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert summary["f1"]["min"] == pytest.approx(2 / 3)
    assert summary["f1"]["max"] == pytest.approx(1.0)
    assert summary["f1"]["median"] == pytest.approx((2 / 3 + 1.0) / 2)


def test_summarize_fold_metrics_of_all_undefined_folds_is_nan():
    folds = [{"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}]
    summary = summarize_fold_metrics(folds)
    assert np.isnan(summary["f1"]["mean"])
    assert summary["f1"]["n_undefined"] == 1


def test_end_to_end_pooled_metrics_known_values():
    true_parts = [np.array([0.0, 10.0]), np.array([40.0, 50.0])]
    routed_parts = [np.array([0.0, 10.0]), np.array([80.0, 100.0])]

    result = end_to_end_pooled_metrics(true_parts, routed_parts)

    assert result["vol_total_real_m3"] == pytest.approx(100.0)
    assert result["vol_total_pred_m3"] == pytest.approx(190.0)
    assert result["vol_total_error_pct"] == pytest.approx(90.0)
    assert result["rmse_vol_todos_nodos"] == pytest.approx(np.sqrt((40**2 + 50**2) / 4))
    assert result["mae_vol_todos_nodos"] == pytest.approx((40 + 50) / 4)


def test_end_to_end_pooled_error_pct_is_nan_without_real_volume():
    result = end_to_end_pooled_metrics([np.array([0.0])], [np.array([5.0])])
    assert np.isnan(result["vol_total_error_pct"])


def test_shape_breakdown_entry_known_values():
    entry = shape_breakdown_entry(
        yc_true=np.array([0, 0, 1, 1]),
        yc_pred=np.array([0, 0, 1, 1]),
        vol_true=np.array([0.0, 0.0, 40.0, 50.0]),
        vol_routed=np.array([0.0, 0.0, 80.0, 100.0]),
    )
    assert entry["n_rows"] == 4
    assert entry["n_positives"] == 2
    assert entry["f1"] == pytest.approx(1.0)
    assert entry["vol_real_m3"] == pytest.approx(90.0)
    assert entry["vol_pred_m3"] == pytest.approx(180.0)
    assert entry["error_vol_pct"] == pytest.approx(100.0)
    assert entry["rmse_vol"] == pytest.approx(np.sqrt((40**2 + 50**2) / 4))


def test_shape_breakdown_entry_f1_and_error_are_nan_without_positives():
    entry = shape_breakdown_entry(
        yc_true=np.array([0, 0]),
        yc_pred=np.array([0, 0]),
        vol_true=np.array([0.0, 0.0]),
        vol_routed=np.array([0.0, 0.0]),
    )
    assert np.isnan(entry["f1"])
    assert np.isnan(entry["error_vol_pct"])
