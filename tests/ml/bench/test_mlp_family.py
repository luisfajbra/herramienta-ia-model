import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from swmm_resilience.ml.bench.registry import available_families, get_family
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17

PARAMS = {"hidden_sizes": (16, 8), "epochs": 20, "learning_rate": 0.01, "batch_size": 16}


def test_ensemble_averages_probabilities_and_conditional_m3_before_routing():
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from swmm_resilience.ml.bench.models.mlp_family import ensemble_predictions
    X = np.zeros((2, 2))
    classifiers = [DummyClassifier(strategy="prior").fit(np.zeros((5, 2)), y)
                   for y in ([0, 0, 0, 1, 1], [0, 0, 1, 1, 1])]
    regressors = [DummyRegressor(constant=np.log1p(v), strategy="constant").fit(X, [0, 0])
                  for v in (2., 8.)]
    pred, prob, conditional, routed = ensemble_predictions(classifiers, regressors, X)
    np.testing.assert_allclose(prob, [0.5, 0.5])
    np.testing.assert_array_equal(pred, [1, 1])
    np.testing.assert_allclose(conditional, [5., 5.])
    np.testing.assert_allclose(routed, conditional)
    dry = DummyClassifier(strategy="prior").fit(np.zeros((5, 2)), [0, 0, 0, 0, 1])
    pred, prob, conditional, routed = ensemble_predictions([dry, dry], regressors, X)
    np.testing.assert_allclose(conditional, [5., 5.])
    np.testing.assert_allclose(routed, [0., 0.])


def test_seed_report_includes_each_seed_mean_and_sample_standard_deviation():
    from swmm_resilience.ml.bench.reports import summarize_seeds
    report = summarize_seeds({42: {"rmse": 2.0}, 43: {"rmse": 4.0}})
    assert report["per_seed"] == {"42": {"rmse": 2.0}, "43": {"rmse": 4.0}}
    assert report["mean"]["rmse"] == 3.0
    assert report["sample_std"]["rmse"] == pytest.approx(np.sqrt(2.0))


def test_seeded_oof_ensemble_uses_identical_rows_and_reports_every_seed(tmp_path):
    from tests.ml.bench.prepared_factory import make_prepared
    from swmm_resilience.ml.bench.evaluate import evaluate_mlp_seeds
    prepared = make_prepared([
        {"shape_id": shape, "node_id": node, "factor_mult": float(factor),
         "vol": float(node * 2), "duracion_horas": 1., "tiempo_al_pico_h": .5}
        for shape in ("a", "b") for factor in (1, 2) for node in (0, 1)
    ], ("LeaveOneShapeOut",))
    params = {"hidden_sizes": (4,), "epochs": 1}
    oof, metrics, per_seed, report = evaluate_mlp_seeds(prepared, params, params)
    assert sorted(per_seed) == [42, 43, 44, 45, 46]
    np.testing.assert_allclose(oof.y_prob_clf, np.mean([frame.y_prob_clf for frame in per_seed.values()], axis=0))
    np.testing.assert_allclose(oof.y_pred_reg, np.mean([frame.y_pred_reg for frame in per_seed.values()], axis=0))
    np.testing.assert_array_equal(oof.y_pred_clf, (oof.y_prob_clf >= .5).astype(int))
    assert set(report["per_seed"]) == {"42", "43", "44", "45", "46"}
    assert metrics["LeaveOneShapeOut"]["end_to_end"]
    from swmm_resilience.ml.bench.reports import save_mlp_seed_evaluation
    directory = save_mlp_seed_evaluation(prepared, oof, metrics, per_seed, report, tmp_path / "run")
    for seed in per_seed:
        saved = pd.read_parquet(directory / f"seed-{seed}" / "oof.parquet")
        pd.testing.assert_frame_equal(saved, per_seed[seed])
    with pytest.raises(FileExistsError):
        save_mlp_seed_evaluation(prepared, oof, metrics, per_seed, report, tmp_path / "run")


def test_ensemble_preserves_nonnegative_conditional_volume_policy():
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from swmm_resilience.ml.bench.models.mlp_family import ensemble_predictions
    X = pd.DataFrame(np.zeros((2, 2)), columns=["a", "b"])
    classifier = DummyClassifier(strategy="constant", constant=1).fit(X, [0, 1])
    regressors = [DummyRegressor(strategy="constant", constant=v).fit(X, [0., 0.])
                  for v in (-1., np.log1p(2.))]
    _, _, conditional, routed = ensemble_predictions([classifier, classifier], regressors, X)
    np.testing.assert_allclose(conditional, [1., 1.])
    np.testing.assert_allclose(routed, [1., 1.])


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(7)
    n = 80
    X = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS_V17))), columns=list(FEATURE_COLUMNS_V17)
    )
    y_clf = (X["q_pico_nodo"] + X["prof_max"] > 0).astype(int).to_numpy()
    y_reg = np.abs(X["q_pico_nodo"].to_numpy()) * 10.0
    return X, y_clf, y_reg


def test_mlp_is_registered_as_a_family():
    assert "mlp" in available_families()
    assert get_family("mlp").SCALE_FEATURES is True


def test_pipeline_has_imputer_scaler_and_model():
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    steps = dict(pipeline.named_steps)
    assert set(steps) == {"imputer", "scaler", "model"}
    assert isinstance(steps["scaler"], StandardScaler)


def test_classifier_fits_and_returns_binary_predictions(toy_data):
    X, y_clf, _ = toy_data
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    pipeline.fit(X, y_clf)

    predictions = pipeline.predict(X)
    assert set(np.unique(predictions)) <= {0, 1}
    assert len(predictions) == len(X)


def test_predict_proba_returns_two_calibrated_columns(toy_data):
    X, y_clf, _ = toy_data
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    pipeline.fit(X, y_clf)

    probabilities = pipeline.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_regressor_fits_and_predicts_finite_values(toy_data):
    X, _, y_reg = toy_data
    pipeline = get_family("mlp").build_regressor(PARAMS)
    pipeline.fit(X, np.log1p(y_reg))

    predictions = pipeline.predict(X)
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_the_same_seed_gives_the_same_predictions(toy_data):
    """Sin esto el MLP no es comparable: cada corrida daria otro numero.

    El generador global de torch se perturba deliberadamente antes de cada
    fit (con una semilla distinta cada vez) para que el unico motivo por el
    que ambas corridas puedan coincidir sea que el estimador se resiembra a
    si mismo por dentro. Sin esta perturbacion, un estado global heredado
    de un test vecino con el mismo PARAMS podria hacer que la prueba pasara
    aunque el estimador no fuera realmente determinista.
    """
    X, y_clf, _ = toy_data
    first = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    second = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)

    torch.manual_seed(111)
    torch.rand(100)
    first.fit(X, y_clf)

    torch.manual_seed(222)
    torch.rand(100)
    second.fit(X, y_clf)

    np.testing.assert_allclose(
        first.predict_proba(X)[:, 1], second.predict_proba(X)[:, 1], rtol=1e-6
    )


def test_the_estimator_is_clonable_by_sklearn():
    """Pipeline y cross-validation dependen de get_params/set_params."""
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    cloned = clone(pipeline)
    assert cloned.named_steps["model"].get_params()["epochs"] == PARAMS["epochs"]


def test_a_legitimate_hyperparameter_reaches_the_estimator():
    """dropout es un parametro real de _DEFAULTS: debe llegar tal cual al
    estimador, sin pasar por ningun allow-list que lo reescriba u omita.
    """
    params = {**PARAMS, "dropout": 0.5}
    pipeline = get_family("mlp").build_classifier(params, scale_pos_weight=1.0)
    assert pipeline.named_steps["model"].get_params()["dropout"] == pytest.approx(0.5)


def test_an_unknown_hyperparameter_raises_instead_of_being_silently_dropped():
    """Una clave que el estimador no reconoce (typo o hiperparametro nuevo
    que nadie cableo aun) debe reventar en TypeError, no desaparecer sin
    aviso: ese fue exactamente el defecto que se corrigio en linear_family
    y no debe reaparecer aqui, porque ajustar la arquitectura del MLP es la
    razon de ser de este candidato.
    """
    params = {**PARAMS, "weight_decay": 0.01}
    with pytest.raises(TypeError):
        get_family("mlp").build_classifier(params, scale_pos_weight=1.0)


def test_scale_pos_weight_reaches_the_loss(toy_data):
    """El desbalance se aplica como pos_weight de BCEWithLogitsLoss."""
    model = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=5.0).named_steps["model"]
    assert model.get_params()["scale_pos_weight"] == pytest.approx(5.0)


def test_scale_pos_weight_actually_reaches_bcewithlogitsloss_during_fit(toy_data, monkeypatch):
    """El test anterior solo comprueba que el constructor guarda el valor;
    esta prueba espia BCEWithLogitsLoss dentro del modulo mlp_family para
    confirmar que fit() de verdad lo pasa a la perdida, no solo que el
    estimador lo recuerda.
    """
    import swmm_resilience.ml.bench.models.mlp_family as mlp_family

    X, y_clf, _ = toy_data
    captured = {}
    real_loss_cls = mlp_family.nn.BCEWithLogitsLoss

    def spying_bcewithlogitsloss(*args, **kwargs):
        captured["pos_weight"] = kwargs.get("pos_weight")
        return real_loss_cls(*args, **kwargs)

    monkeypatch.setattr(mlp_family.nn, "BCEWithLogitsLoss", spying_bcewithlogitsloss)

    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=4.0)
    pipeline.fit(X, y_clf)

    assert captured["pos_weight"] is not None
    torch.testing.assert_close(captured["pos_weight"], torch.tensor([4.0]))


def test_training_runs_on_cpu_without_requiring_cuda(toy_data):
    X, y_clf, _ = toy_data
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    pipeline.fit(X, y_clf)
    assert next(pipeline.named_steps["model"].network_.parameters()).device.type == "cpu"


def test_preprocessing_descriptor_declares_the_scaler():
    descriptor = get_family("mlp").preprocessing_descriptor(PARAMS)
    assert descriptor == {"imputer": "median", "scaler": "standard", "pca": None}
