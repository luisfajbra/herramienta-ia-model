from types import SimpleNamespace
from dataclasses import replace
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import pytest

from tests.ml.bench.prepared_factory import make_prepared


def _two_network_prepared():
    rows = []
    for network_id in ("A", "B"):
        for shape_index in range(4):
            for factor in (1.0, 2.0):
                for node_index in range(2):
                    rows.append(
                        {
                            "network_id": network_id,
                            "shape_id": f"shape-{shape_index}",
                            "node_id": f"node-{node_index}",
                            "factor_mult": factor,
                            "vol": float((shape_index + node_index + factor) % 3),
                            "duracion_horas": float(shape_index + 1),
                            "tiempo_al_pico_h": float(factor),
                        }
                    )
    prepared = make_prepared(rows, ("LeaveOneNetworkOut",))
    return replace(
        prepared,
        manifest={**prepared.manifest, "flood_threshold_m3": 1.0},
    )


def test_nested_evaluation_tunes_only_outer_training_networks(monkeypatch):
    import swmm_resilience.ml.bench.evaluate as evaluate_module

    prepared = _two_network_prepared()
    tuning_views = []
    tuning_kwargs = []

    def spy_run_study(view, family, **kwargs):
        tuning_views.append(view)
        tuning_kwargs.append(kwargs)
        return SimpleNamespace(
            family=family,
            prep_id=view.prep_id,
            study_id=f"study-{len(tuning_views)}",
            best_classifier_params={"stub": True},
            best_regressor_params={"stub": True},
            best_value=1.25,
            trials=[{"number": 0, "value": 1.25}],
            seed=kwargs["seed"],
            objective_metric=kwargs["objective"],
        )

    monkeypatch.setattr(evaluate_module, "run_study", spy_run_study)
    monkeypatch.setattr(
        evaluate_module,
        "_fit_predict_fold",
        lambda prepared, family, clf, reg, train_idx, test_idx: (
            np.zeros(len(test_idx), dtype=int),
            np.full(len(test_idx), 0.25),
            np.ones(len(test_idx)),
        ),
    )

    result = evaluate_module.evaluate_nested_networks(
        prepared,
        "xgboost",
        ("LeaveOneShapeOut",),
        objective="macro_network_rmse",
        volume_band_edges_m3=(10.0, 100.0),
        n_trials=1,
        n_splits=2,
        seed=17,
    )

    assert len(result.folds) == len(tuning_views) == 2
    assert all(
        kwargs["inner_protocol"] == "LeaveOneShapeOut"
        for kwargs in tuning_kwargs
    )
    assert result.oof.groupby("sample_idx").size().eq(1).all()
    assert set(result.oof["sample_idx"]) == set(range(len(prepared.keys)))
    routed_truth = prepared.y_reg.to_numpy()
    assert result.metrics["pooled_rmse"] == pytest.approx(
        np.sqrt(np.mean(routed_truth**2))
    )
    for fold, view in zip(result.folds, tuning_views):
        outer_test_networks = set(
            prepared.keys.iloc[list(fold.outer_test_sample_idx)]["network_id"]
        )
        outer_train_networks = set(
            prepared.keys.iloc[list(fold.outer_train_sample_idx)]["network_id"]
        )
        assert len(outer_test_networks) == 1
        assert outer_train_networks.isdisjoint(outer_test_networks)
        assert set(view.keys["network_id"]) == outer_train_networks
        assert set(view.keys["run_id"]) <= set(
            prepared.keys.iloc[list(fold.outer_train_sample_idx)]["run_id"]
        )
        assert set(view.keys["run_id"]).isdisjoint(
            set(prepared.keys.iloc[list(fold.outer_test_sample_idx)]["run_id"])
        )
        assert fold.objective == "macro_network_rmse"
        assert fold.optuna_study_id


def test_nested_mlp_fits_five_fold_selected_members_and_scores_their_ensemble(monkeypatch, tmp_path):
    import swmm_resilience.ml.bench.evaluate as evaluate
    from swmm_resilience.ml.bench.models import mlp_family
    from sklearn.pipeline import Pipeline

    prepared = _two_network_prepared()
    tuned_views, classifiers, regressors, fits = [], [], [], []
    build_classifier = mlp_family.build_classifier
    build_regressor = mlp_family.build_regressor
    fit = Pipeline.fit

    def tune(view, family, **kwargs):
        assert family == "mlp"
        tuned_views.append(view)
        params = {"hidden_sizes": (len(tuned_views) + 2,), "epochs": 1, "dropout": 0.0}
        return SimpleNamespace(study_id=f"study-{len(tuned_views)}", best_classifier_params=params,
                               best_regressor_params=params, best_value=1., trials=[{"number": 0}])

    def classifier(params, weight):
        model = build_classifier(params, weight)
        classifiers.append((dict(params), model))
        return model

    def regressor(params):
        model = build_regressor(params)
        regressors.append((dict(params), model))
        return model

    def record_fit(self, X, y, **kwargs):
        fits.append((X.index.to_numpy(), np.asarray(y)))
        return fit(self, X, y, **kwargs)

    monkeypatch.setattr(evaluate, "run_study", tune)
    monkeypatch.setattr(mlp_family, "build_classifier", classifier)
    monkeypatch.setattr(mlp_family, "build_regressor", regressor)
    monkeypatch.setattr(Pipeline, "fit", record_fit)
    result = evaluate.evaluate_nested_networks(
        prepared, "mlp", ("LeaveOneShapeOut",), objective="macro_network_rmse",
        volume_band_edges_m3=(1., 2.), n_trials=1, n_splits=2, seed=19)

    assert len(tuned_views) == 2
    assert len(classifiers) == len(regressors) == 10
    assert set(result.seed_oof) == {42, 43, 44, 45, 46}
    assert set(result.metrics["seed_summary"]["per_seed"]) == {"42", "43", "44", "45", "46"}
    for index, fold in enumerate(result.folds):
        assert fold.model_seeds == (42, 43, 44, 45, 46)
        train = np.asarray(fold.outer_train_sample_idx)
        test = np.asarray(fold.outer_test_sample_idx)
        view = tuned_views[index]
        assert set(view.keys.network_id) == set(prepared.keys.iloc[train].network_id)
        np.testing.assert_array_equal(view.y_reg, prepared.y_reg.iloc[train])
        for member in range(5):
            clf_params, clf = classifiers[index * 5 + member]
            reg_params, reg = regressors[index * 5 + member]
            assert clf_params == {**fold.selected_classifier_params, "random_state": 42 + member}
            assert reg_params == {**fold.selected_regressor_params, "random_state": 42 + member}
            clf_indices, clf_y = fits[index * 10 + member * 2]
            reg_indices, reg_y = fits[index * 10 + member * 2 + 1]
            np.testing.assert_array_equal(clf_indices, train)
            np.testing.assert_array_equal(clf_y, prepared.y_clf.iloc[train])
            flooded = train[prepared.y_clf.iloc[train].to_numpy() == 1]
            np.testing.assert_array_equal(reg_indices, flooded)
            np.testing.assert_allclose(reg_y, np.log1p(prepared.y_reg.iloc[flooded]))
        X_test = prepared.X.iloc[test]
        probability = np.mean([model.predict_proba(X_test)[:, 1]
                               for _, model in classifiers[index * 5:(index + 1) * 5]], axis=0)
        conditional = np.mean([np.maximum(0., np.expm1(model.predict(X_test)))
                               for _, model in regressors[index * 5:(index + 1) * 5]], axis=0)
        oof = result.oof[result.oof.fold_id == fold.fold_id]
        np.testing.assert_allclose(oof.y_prob_clf, probability)
        np.testing.assert_array_equal(oof.y_pred_clf, (probability >= .5).astype(int))
        np.testing.assert_allclose(oof.y_pred_reg, conditional)
    sorted_oof = result.oof.sort_values("sample_idx")
    routed = np.where(sorted_oof.y_pred_clf == 1, sorted_oof.y_pred_reg, 0.)
    expected_rmse = np.mean([
        np.sqrt(np.mean((prepared.y_reg[prepared.keys.network_id == network] - routed[prepared.keys.network_id == network]) ** 2))
        for network in ("A", "B")])
    assert result.metrics["macro_network_rmse"] == pytest.approx(expected_rmse)
    output = evaluate.save_nested_network_result(result, tmp_path)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_seeds"] == [42, 43, 44, 45, 46]
    for seed, expected in result.seed_oof.items():
        pd.testing.assert_frame_equal(pd.read_parquet(output / f"seed-{seed}/oof.parquet"), expected)


def test_nested_config_entry_point_passes_configured_objective_and_budget(monkeypatch):
    import swmm_resilience.ml.bench.evaluate as evaluate_module

    prepared = _two_network_prepared()
    captured = {}

    def spy_evaluate(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "nested-result"

    monkeypatch.setattr(evaluate_module, "evaluate_nested_networks", spy_evaluate)
    config = SimpleNamespace(
        bench=SimpleNamespace(
            tuning=SimpleNamespace(
                objective="macro_network_rmse", n_trials=7, n_splits=3
            ),
            reporting=SimpleNamespace(volume_band_edges_m3=(10.0, 100.0)),
        )
    )

    result = evaluate_module.evaluate_nested_networks_from_config(
        prepared, "xgboost", ("LeaveOneShapeOut",), config, seed=13
    )

    assert result == "nested-result"
    assert captured["kwargs"]["objective"] == "macro_network_rmse"
    assert captured["kwargs"]["n_trials"] == 7
    assert captured["kwargs"]["n_splits"] == 3
    assert captured["kwargs"]["seed"] == 13
    assert captured["kwargs"]["volume_band_edges_m3"] == (10.0, 100.0)


def test_nested_config_entry_point_requires_explicit_objective():
    import swmm_resilience.ml.bench.evaluate as evaluate_module

    config = SimpleNamespace(
        bench=SimpleNamespace(
            tuning=SimpleNamespace(objective=None, n_trials=1, n_splits=2)
        )
    )
    with pytest.raises(ValueError, match="bench.tuning.objective"):
        evaluate_module.evaluate_nested_networks_from_config(
            _two_network_prepared(), "xgboost", ("LeaveOneShapeOut",), config
        )


def test_nested_report_breaks_metrics_down_by_network_and_hydrograph_shape():
    import swmm_resilience.ml.bench.evaluate as evaluate_module

    prepared = _two_network_prepared()
    oof = pd.DataFrame(
        {
            "sample_idx": np.arange(len(prepared.keys)),
            "protocol": "LeaveOneNetworkOut",
            "fold_id": np.where(
                prepared.keys["network_id"].to_numpy() == "A", 0, 1
            ),
            "y_pred_clf": 0,
            "y_prob_clf": 0.25,
            "y_pred_reg": 0.5,
        }
    )

    report = evaluate_module.nested_network_event_metrics(
        prepared, oof, volume_band_edges_m3=(1.0, 2.0)
    )

    assert set(report["by_network"]) == {"A", "B"}
    assert {
        (item["network_id"], item["shape_id"])
        for item in report["by_network_event"]
    } == {
        (network_id, shape_id)
        for network_id in ("A", "B")
        for shape_id in ("shape-0", "shape-1", "shape-2", "shape-3")
    }
    a = report["by_network"]["A"]
    assert a["n_samples"] == 16
    assert a["confusion_matrix"]["fn"] == a["n_actual_flooded"]
    a_idx = np.flatnonzero(prepared.keys["network_id"].to_numpy() == "A")
    assert a["rmse_m3"] == pytest.approx(
        np.sqrt(np.mean(prepared.y_reg.iloc[a_idx].to_numpy() ** 2))
    )
    assert a["mae_m3"] == pytest.approx(
        np.mean(prepared.y_reg.iloc[a_idx].to_numpy())
    )
    assert a["volume_bias_m3"] == pytest.approx(
        -prepared.y_reg.iloc[a_idx].sum()
    )
    assert len(a["false_negatives_by_volume_band"]) == 3
    assert sum(
        item["false_negatives"] for item in a["false_negatives_by_volume_band"]
    ) == a["confusion_matrix"]["fn"]
    assert a["pr_auc"] == pytest.approx(a["n_actual_flooded"] / a["n_samples"])


@pytest.mark.parametrize(
    "edges", [(), (1.0, 1.0), (2.0, 1.0), (-1.0, 2.0), (0.5, 2.0)]
)
def test_nested_report_rejects_invalid_volume_band_edges(edges):
    import swmm_resilience.ml.bench.evaluate as evaluate_module

    prepared = _two_network_prepared()
    oof = pd.DataFrame(
        {
            "sample_idx": np.arange(len(prepared.keys)),
            "protocol": "LeaveOneNetworkOut",
            "fold_id": 0,
            "y_pred_clf": 0,
            "y_prob_clf": 0.25,
            "y_pred_reg": 0.5,
        }
    )
    with pytest.raises(ValueError, match="volume_band_edges_m3"):
        evaluate_module.nested_network_event_metrics(
            prepared, oof, volume_band_edges_m3=edges
        )


def test_nested_report_marks_pr_auc_undefined_when_a_group_has_one_class():
    import swmm_resilience.ml.bench.evaluate as evaluate_module

    prepared = _two_network_prepared()
    no_flood = replace(
        prepared,
        y_clf=pd.Series(
            np.zeros(len(prepared.keys), dtype=int), name=prepared.y_clf.name
        ),
    )
    oof = pd.DataFrame(
        {
            "sample_idx": np.arange(len(no_flood.keys)),
            "protocol": "LeaveOneNetworkOut",
            "fold_id": np.where(
                no_flood.keys["network_id"].to_numpy() == "A", 0, 1
            ),
            "y_pred_clf": 0,
            "y_prob_clf": 0.25,
            "y_pred_reg": 0.5,
        }
    )

    report = evaluate_module.nested_network_event_metrics(
        no_flood, oof, volume_band_edges_m3=(10.0,)
    )

    assert report["by_network"]["A"]["pr_auc"] is None
    assert report["by_network"]["A"]["recall"] is None


def test_outer_test_targets_do_not_change_the_inner_training_fingerprint():
    import swmm_resilience.ml.bench.evaluate as evaluate_module

    prepared = _two_network_prepared()
    outer = prepared.folds[
        (prepared.folds["protocol"] == "LeaveOneNetworkOut")
        & (prepared.folds["fold_id"] == 0)
    ]
    test_idx = outer.loc[outer["split"] == "test", "sample_idx"].to_numpy()
    train_idx = outer.loc[outer["split"] == "train", "sample_idx"].to_numpy()
    modified_targets = prepared.y_reg.copy()
    modified_targets.iloc[test_idx] += 10000.0
    modified = replace(
        prepared,
        prep_id="fedcba9876543210",
        y_reg=modified_targets,
    )

    original_view = evaluate_module._fold_training_view(prepared, train_idx)
    modified_view = evaluate_module._fold_training_view(modified, train_idx)

    assert original_view.prep_id == modified_view.prep_id
    assert original_view.run_ids == modified_view.run_ids


def test_nested_evaluation_saves_immutable_split_and_study_provenance(
    monkeypatch, tmp_path
):
    import swmm_resilience.ml.bench.evaluate as evaluate_module

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(evaluate_module, "datetime", FrozenDatetime)

    prepared = _two_network_prepared()
    monkeypatch.setattr(
        evaluate_module,
        "run_study",
        lambda view, family, **kwargs: SimpleNamespace(
            study_id=f"study-{len(view.run_ids)}",
            best_classifier_params={"stub": True},
            best_regressor_params={"stub": True},
            best_value=1.25,
            trials=[{"number": 0, "value": 1.25}],
        ),
    )
    monkeypatch.setattr(
        evaluate_module,
        "_fit_predict_fold",
        lambda prepared, family, clf, reg, train_idx, test_idx: (
            np.zeros(len(test_idx), dtype=int),
            np.full(len(test_idx), 0.25),
            np.ones(len(test_idx)),
        ),
    )
    result = evaluate_module.evaluate_nested_networks(
        prepared,
        "xgboost",
        ("LeaveOneShapeOut",),
        objective="macro_network_rmse",
        volume_band_edges_m3=(10.0, 100.0),
        n_trials=1,
        n_splits=2,
    )

    output = evaluate_module.save_nested_network_result(result, tmp_path)

    assert (output / "oof.parquet").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["objective"] == "macro_network_rmse"
    assert manifest["network_aggregation"] == "equal_weight_per_network"
    assert manifest["reporting_config"]["volume_band_edges_m3"] == [10.0, 100.0]
    assert output.name.endswith(manifest["reporting_config_sha256"][:8])
    assert len(manifest["folds"]) == 2
    assert all(fold["optuna_study_id"] for fold in manifest["folds"])
    assert all(fold["inner_prep_id"] for fold in manifest["folds"])

    with pytest.raises(FileExistsError):
        evaluate_module.save_nested_network_result(result, tmp_path)
