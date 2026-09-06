import csv
import json

import pandas as pd
import pytest

from swmm_resilience.ml.bench.ranking import RankingCriterion, rank_candidates
from swmm_resilience.ml.bench.reports import write_reports

CRITERION = RankingCriterion(
    primary_metric="end_to_end.rmse_vol_todos_nodos",
    primary_direction="minimize",
    tie_breakers=(("classifier.f1", "maximize"),),
)


def _metrics(rmse):
    return {
        "LOSO": {
            "classifier": {"f1": 0.8, "precision": 0.7, "recall": 0.9, "auc_roc": 0.85},
            "regressor_oracle": {"nse": 0.6, "rmse": 2.0, "mae": 1.0, "r2": 0.6, "log_nse": 0.7},
            "end_to_end": {
                "rmse_vol_todos_nodos": rmse,
                "pct_nodos_correctos": 0.9,
                "vol_total_pred_m3": 10.0,
                "vol_total_real_m3": 11.0,
            },
            "by_factor": {"1.00": {"f1": 0.8, "rmse_vol": rmse}},
        }
    }


def test_writes_one_metrics_file_per_family_plus_the_ranking(tmp_path):
    metrics = {"xgboost": _metrics(3.0), "linear": _metrics(7.0)}
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")

    paths = write_reports(metrics, ranking, tmp_path)

    assert (tmp_path / "metrics_xgboost.json").exists()
    assert (tmp_path / "metrics_linear.json").exists()
    assert (tmp_path / "ranking.json").exists()
    assert (tmp_path / "ranking.csv").exists()
    assert set(paths) >= {"ranking_json", "ranking_csv"}


def test_metrics_json_keeps_the_three_levels_and_the_factor_breakdown(tmp_path):
    metrics = {"xgboost": _metrics(3.0)}
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")
    write_reports(metrics, ranking, tmp_path)

    written = json.loads((tmp_path / "metrics_xgboost.json").read_text(encoding="utf-8"))
    assert set(written["LOSO"]) == {
        "classifier", "regressor_oracle", "end_to_end", "by_factor"
    }
    assert written["LOSO"]["by_factor"]["1.00"]["f1"] == 0.8


def test_ranking_csv_is_readable_and_ordered(tmp_path):
    metrics = {"xgboost": _metrics(3.0), "linear": _metrics(7.0)}
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")
    write_reports(metrics, ranking, tmp_path)

    with open(tmp_path / "ranking.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["family"] for row in rows] == ["xgboost", "linear"]
    assert rows[0]["rank"] == "1"


def test_ranking_files_follow_ranked_order_not_dict_insertion_order(tmp_path):
    # Insertion order is worst-first (linear, then xgboost); ranked order
    # (by ascending rmse) must be the reverse. If write_reports ever writes
    # rows in metrics_by_family's insertion order instead of the ranking's
    # order, this test must fail.
    metrics = {"linear": _metrics(7.0), "xgboost": _metrics(3.0)}
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")
    paths = write_reports(metrics, ranking, tmp_path)

    written = json.loads(paths["ranking_json"].read_text(encoding="utf-8"))
    assert [row["family"] for row in written] == ["xgboost", "linear"]

    with open(tmp_path / "ranking.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["family"] for row in rows] == ["xgboost", "linear"]


def test_report_directory_is_created_when_missing(tmp_path):
    target = tmp_path / "no" / "existe" / "todavia"
    metrics = {"xgboost": _metrics(3.0)}
    write_reports(metrics, rank_candidates(metrics, CRITERION, protocol="LOSO"), target)
    assert (target / "ranking.json").exists()


def _nan_metrics(rmse):
    # auc_roc is legitimately NaN in real LOSO runs whenever a fold has
    # only one class (e.g. low intensity factors where no node floods).
    return {
        "LOSO": {
            "classifier": {"f1": 0.8, "precision": 0.7, "recall": 0.9, "auc_roc": float("nan")},
            "regressor_oracle": {"nse": 0.6, "rmse": 2.0, "mae": 1.0, "r2": 0.6, "log_nse": 0.7},
            "end_to_end": {
                "rmse_vol_todos_nodos": rmse,
                "pct_nodos_correctos": 0.9,
                "vol_total_pred_m3": 10.0,
                "vol_total_real_m3": 11.0,
            },
            "by_factor": {},
        }
    }


def test_a_nan_metric_round_trips_to_none_through_json_load(tmp_path):
    metrics = {"broken": _nan_metrics(float("nan"))}
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")
    write_reports(metrics, ranking, tmp_path)

    written_metrics = json.loads((tmp_path / "metrics_broken.json").read_text(encoding="utf-8"))
    assert written_metrics["LOSO"]["classifier"]["auc_roc"] is None
    assert written_metrics["LOSO"]["end_to_end"]["rmse_vol_todos_nodos"] is None

    written_ranking = json.loads((tmp_path / "ranking.json").read_text(encoding="utf-8"))
    assert written_ranking[0]["primary_value"] is None

    # rank_candidates itself is untouched: NaN stays NaN in memory, only
    # the on-disk JSON representation changes.
    assert pd.isna(ranking.iloc[0]["primary_value"])


def _reject_nonfinite_token(token):
    raise ValueError(f"strict reader rejects non-finite JSON token: {token}")


def test_written_files_parse_under_a_strict_json_reader(tmp_path):
    metrics = {"broken": _nan_metrics(float("nan")), "good": _metrics(3.0)}
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")
    write_reports(metrics, ranking, tmp_path)

    for name in ("metrics_broken.json", "metrics_good.json", "ranking.json"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        # Python's json.loads accepts bare NaN/Infinity/-Infinity tokens by
        # default (they are not valid RFC 8259 JSON); routing them through
        # parse_constant makes it behave like a strict reader (e.g.
        # SQLite's json1 functions) that rejects such files outright. This
        # must not raise -- i.e. the files must contain no such tokens.
        json.loads(text, parse_constant=_reject_nonfinite_token)


def test_allow_nan_false_would_have_raised_on_the_old_unsanitized_payload():
    # Documents the failure mode this fix closes: serialising a NaN metric
    # without sanitising it first raises under allow_nan=False, which is
    # exactly why _json_safe runs before every json.dumps call now.
    payload = {"auc_roc": float("nan")}
    with pytest.raises(ValueError):
        json.dumps(payload, allow_nan=False)
