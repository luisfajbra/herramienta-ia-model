import numpy as np
import pytest

from swmm_resilience.ml.bench.ranking import (
    RankingCriterion,
    rank_candidates,
    resolve_metric,
)

CRITERION = RankingCriterion(
    primary_metric="end_to_end.rmse_vol_todos_nodos",
    primary_direction="minimize",
    tie_breakers=(
        ("classifier.f1", "maximize"),
        ("regressor_oracle.nse", "maximize"),
    ),
)


def _metrics(rmse, f1=0.5, nse=0.5):
    return {
        "LOSO": {
            "classifier": {"f1": f1, "precision": 0.5, "recall": 0.5, "auc_roc": 0.7},
            "regressor_oracle": {"nse": nse, "rmse": 1.0, "mae": 1.0, "r2": 0.5, "log_nse": 0.5},
            "end_to_end": {
                "rmse_vol_todos_nodos": rmse,
                "pct_nodos_correctos": 0.9,
                "vol_total_pred_m3": 10.0,
                "vol_total_real_m3": 11.0,
            },
            "by_factor": {},
        }
    }


def test_resolve_metric_walks_the_dotted_path():
    metrics = _metrics(5.0)["LOSO"]
    assert resolve_metric(metrics, "end_to_end.rmse_vol_todos_nodos") == pytest.approx(5.0)
    assert resolve_metric(metrics, "classifier.f1") == pytest.approx(0.5)


def test_resolve_metric_returns_none_for_a_missing_path():
    assert resolve_metric(_metrics(5.0)["LOSO"], "classifier.no_existe") is None


def test_lowest_rmse_ranks_first():
    ranking = rank_candidates(
        {"xgboost": _metrics(3.0), "svm": _metrics(9.0), "linear": _metrics(6.0)},
        CRITERION,
        protocol="LOSO",
    )
    assert ranking["family"].tolist() == ["xgboost", "linear", "svm"]
    assert ranking["rank"].tolist() == [1, 2, 3]


def test_ties_are_broken_by_the_first_tie_breaker():
    ranking = rank_candidates(
        {"a": _metrics(5.0, f1=0.4), "b": _metrics(5.0, f1=0.9)},
        CRITERION,
        protocol="LOSO",
    )
    assert ranking["family"].tolist() == ["b", "a"]


def test_the_second_tie_breaker_is_used_when_the_first_also_ties():
    ranking = rank_candidates(
        {"a": _metrics(5.0, f1=0.7, nse=0.2), "b": _metrics(5.0, f1=0.7, nse=0.8)},
        CRITERION,
        protocol="LOSO",
    )
    assert ranking["family"].tolist() == ["b", "a"]


def test_a_candidate_with_a_nan_primary_metric_is_marked_invalid_and_ranked_last():
    ranking = rank_candidates(
        {"good": _metrics(5.0), "broken": _metrics(float("nan"))},
        CRITERION,
        protocol="LOSO",
    )
    assert ranking["family"].tolist() == ["good", "broken"]
    assert ranking.set_index("family").loc["broken", "valid"] == 0
    assert "NaN" in ranking.set_index("family").loc["broken", "invalid_reason"]


def test_a_candidate_missing_the_primary_metric_is_invalid():
    ranking = rank_candidates(
        {"good": _metrics(5.0), "empty": {"LOSO": {"end_to_end": {}, "classifier": {},
                                                   "regressor_oracle": {}, "by_factor": {}}}},
        CRITERION,
        protocol="LOSO",
    )
    assert ranking.set_index("family").loc["empty", "valid"] == 0


def test_maximize_direction_inverts_the_order():
    criterion = RankingCriterion(
        primary_metric="classifier.f1", primary_direction="maximize", tie_breakers=()
    )
    ranking = rank_candidates(
        {"low": _metrics(1.0, f1=0.2), "high": _metrics(9.0, f1=0.95)},
        criterion,
        protocol="LOSO",
    )
    assert ranking["family"].tolist() == ["high", "low"]


def test_unknown_direction_is_rejected():
    with pytest.raises(ValueError, match="minimize|maximize"):
        RankingCriterion(
            primary_metric="classifier.f1", primary_direction="sideways", tie_breakers=()
        )


def test_a_protocol_absent_from_a_candidate_is_reported_not_crashed():
    ranking = rank_candidates(
        {"only_kfold": {"GroupKFold5": _metrics(5.0)["LOSO"]}},
        CRITERION,
        protocol="LOSO",
    )
    assert ranking.set_index("family").loc["only_kfold", "valid"] == 0


def test_column_schema_is_stable_even_when_every_candidate_is_invalid():
    # When every candidate is missing the protocol, there is no valid row
    # to union against. The tie-breaker columns must still appear -- the
    # column schema is part of the declared interface and must not shrink
    # just because nothing valid happened to be ranked.
    ranking = rank_candidates(
        {"only_kfold": {"GroupKFold5": {}}},
        CRITERION,
        protocol="LOSO",
    )
    assert list(ranking.columns) == [
        "rank",
        "protocol",
        "family",
        "primary_value",
        "valid",
        "invalid_reason",
        "classifier.f1",
        "regressor_oracle.nse",
    ]


def test_ranking_records_which_protocol_produced_it():
    """ranking.csv/json is the one artifact a reader looks at; it must say
    which protocol the numbers came from, not just rank silently on whatever
    main.py passed in."""
    ranking = rank_candidates(
        {"xgboost": _metrics(3.0), "svm": _metrics(9.0)},
        CRITERION,
        protocol="LOSO",
    )
    assert (ranking["protocol"] == "LOSO").all()
