import csv
import json

import pandas as pd

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


def test_report_directory_is_created_when_missing(tmp_path):
    target = tmp_path / "no" / "existe" / "todavia"
    metrics = {"xgboost": _metrics(3.0)}
    write_reports(metrics, rank_candidates(metrics, CRITERION, protocol="LOSO"), target)
    assert (target / "ranking.json").exists()
