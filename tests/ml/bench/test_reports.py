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


import json

from swmm_resilience.ml.bench.reports import (
    write_metrics_files,
    write_oof,
    write_provenance,
)
from tests.ml.bench.prepared_factory import make_prepared, perfect_oof


def _tiny_prepared():
    rows = [
        dict(shape_id="s", node_id="N0", factor_mult=f, vol=10.0 * f,
             duracion_horas=1.0, tiempo_al_pico_h=0.5)
        for f in (1.0, 2.0, 3.0)
    ]
    return make_prepared(rows, ["LOSO"])


def test_write_metrics_files_uses_the_given_prefix_and_strict_json(tmp_path):
    metrics = {"baseline_x": {"LOSO": {"classifier": {"f1": float("nan")}}}}

    paths = write_metrics_files(metrics, tmp_path)

    written = json.loads((tmp_path / "metrics_baseline_x.json").read_text(encoding="utf-8"))
    assert written["LOSO"]["classifier"]["f1"] is None
    assert paths["metrics_baseline_x"] == tmp_path / "metrics_baseline_x.json"


def test_write_oof_stores_one_parquet_per_name_under_the_prep_id(tmp_path):
    prepared = _tiny_prepared()
    oof = perfect_oof(prepared)

    path = write_oof(oof, tmp_path, prepared.prep_id, "xgboost")

    assert path == tmp_path / prepared.prep_id / "xgboost.parquet"
    assert pd.read_parquet(path).shape == oof.shape


def test_write_provenance_records_prep_id_params_baselines_and_protocol_meaning(tmp_path):
    prepared = _tiny_prepared()
    families = {"xgboost": {"classifier": {"max_depth": 3}, "regressor": {"max_depth": 4}}}

    path = write_provenance(tmp_path, prepared, families, ["baseline_interpolacion_factor"])

    record = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "provenance.json"
    assert record["prep_id"] == prepared.prep_id
    assert record["families"]["xgboost"]["classifier"] == {"max_depth": 3}
    assert record["baselines"] == ["baseline_interpolacion_factor"]
    assert "generated_at_utc" in record
    assert "LOSO" in record["protocol_interpretation"]


def test_metrics_files_carry_the_definition_marker_and_the_prep_id(tmp_path):
    metrics = {"xgboost": {"LOSO": {"classifier": {"f1": 0.9}}}}

    write_metrics_files(metrics, tmp_path, prep_id="abc123")

    written = json.loads((tmp_path / "metrics_xgboost.json").read_text(encoding="utf-8"))
    assert written["metrics_definition"] == "classifier_pooled_v2"
    assert written["prep_id"] == "abc123"
    assert written["LOSO"]["classifier"]["f1"] == 0.9


def test_metrics_files_omit_the_prep_id_when_unknown_but_keep_the_marker(tmp_path):
    write_metrics_files({"x": {"LOSO": {}}}, tmp_path)

    written = json.loads((tmp_path / "metrics_x.json").read_text(encoding="utf-8"))
    assert written["metrics_definition"] == "classifier_pooled_v2"
    assert "prep_id" not in written


def test_write_reports_passes_the_prep_id_to_the_metrics_files(tmp_path):
    metrics = {"xgboost": _metrics(3.0)}
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")

    write_reports(metrics, ranking, tmp_path, prep_id="abc123")

    written = json.loads((tmp_path / "metrics_xgboost.json").read_text(encoding="utf-8"))
    assert written["prep_id"] == "abc123"


def test_the_marker_does_not_leak_into_the_in_memory_metrics_or_the_ranking(tmp_path):
    metrics = {"xgboost": _metrics(3.0)}
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")

    write_reports(metrics, ranking, tmp_path, prep_id="abc123")

    assert "metrics_definition" not in metrics["xgboost"]
    assert list(ranking["family"]) == ["xgboost"]
