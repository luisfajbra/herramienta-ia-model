"""
Compare regression models on the exported SWMM dataset.

Usage:
    python -m swmm_resilience.ml.train
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR

from .preprocessing import select_features_for_model
from ..config import (
    DEFAULT_MODEL_ARTIFACTS_DIR,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_RESULTS_DIR,
    ML_CV_FOLDS,
    ML_DROP_COLUMNS,
    ML_GROUP_COLUMN,
    ML_MODEL_CONFIGS,
    ML_PCA_COMPONENTS,
    ML_PCA_SVD_SOLVER,
    ML_RANDOM_STATE,
    ML_SPLIT_STRATEGY,
    ML_TARGET_CLASSIFICATION,
    ML_TARGET_REGRESSION,
    ML_TEST_SIZE,
    ML_USE_PCA,
)

try:
    from xgboost import XGBClassifier
    from xgboost import XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None


@dataclass
class ModelResult:
    model_name: str
    target: str
    feature_space: str
    pca_components: str
    split_strategy: str
    group_column: str
    train_group_count: int
    test_group_count: int
    train_size: int
    test_size: int
    random_state: int
    cv_folds: int
    mae: float
    rmse: float
    r2: float
    cv_mae_mean: float
    cv_mae_std: float
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_r2_mean: float
    cv_r2_std: float


@dataclass
class ClassificationResult:
    model_name: str
    target: str
    feature_space: str
    pca_components: str
    split_strategy: str
    group_column: str
    train_group_count: int
    test_group_count: int
    train_size: int
    test_size: int
    random_state: int
    cv_folds: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    cv_precision_mean: float
    cv_precision_std: float
    cv_recall_mean: float
    cv_recall_std: float
    cv_f1_mean: float
    cv_f1_std: float


@dataclass(frozen=True)
class SavedModelArtifact:
    task: str
    model_name: str
    target: str
    feature_columns: list[str]
    artifact_path: Path
    trained_rows: int


@dataclass(frozen=True)
class LoadedModelArtifact:
    task: str
    model_name: str
    target: str
    feature_columns: list[str]
    artifact_path: Path
    pipeline: Any
    metadata: dict[str, Any]


def load_dataset(csv_path: Path | str = DEFAULT_OUTPUT_CSV) -> pd.DataFrame:
    """Load the exported dataset."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontro el dataset: {csv_path}")
    return pd.read_csv(csv_path)


def default_artifact_dir(csv_path: Path | str = DEFAULT_OUTPUT_CSV) -> Path:
    """Return the default folder where trained model artifacts are stored."""
    csv_path = Path(csv_path)
    if csv_path == DEFAULT_OUTPUT_CSV:
        return DEFAULT_MODEL_ARTIFACTS_DIR
    return csv_path.parent / DEFAULT_MODEL_ARTIFACTS_DIR.name


def _artifact_manifest_path(artifacts_dir: Path | str) -> Path:
    return Path(artifacts_dir) / "manifest.json"


def _artifact_model_path(artifacts_dir: Path | str, task: str, model_name: str) -> Path:
    safe_name = model_name.replace(" ", "_")
    return Path(artifacts_dir) / f"{task}_{safe_name}.joblib"


def _preferred_model_name(models: dict[str, Pipeline], preferred_name: str) -> str:
    if preferred_name in models:
        return preferred_name
    return next(iter(models))


def default_regression_model_name() -> str:
    """Return the preferred regression model for persistence/inference."""
    return _preferred_model_name(build_models(), "xgboost")


def default_classification_model_name() -> str:
    """Return the preferred classification model for persistence/inference."""
    return _preferred_model_name(build_classification_models(), "xgboost_classifier")


def _selected_model_name(models: dict[str, Pipeline], selected_name: str | None, default_name: str) -> str:
    model_name = selected_name or default_name
    if model_name not in models:
        available = ", ".join(models.keys())
        raise ValueError(f"El modelo '{model_name}' no esta disponible. Opciones: {available}")
    return model_name


def _load_manifest(artifacts_dir: Path | str) -> dict[str, Any]:
    manifest_path = _artifact_manifest_path(artifacts_dir)
    if not manifest_path.exists():
        return {"latest": {}, "regression": {}, "classification": {}}

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data.setdefault("latest", {})
    data.setdefault("regression", {})
    data.setdefault("classification", {})
    return data


def _write_manifest(artifacts_dir: Path | str, manifest: dict[str, Any]) -> Path:
    manifest_path = _artifact_manifest_path(artifacts_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest_path


def _available_excel_engine() -> str | None:
    """Return the first available pandas Excel writer engine for .xlsx output."""
    if importlib.util.find_spec("openpyxl") is not None:
        return "openpyxl"
    if importlib.util.find_spec("xlsxwriter") is not None:
        return "xlsxwriter"
    return None


def select_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Select numeric input features and one target column."""
    return select_features_for_model(df, target, drop_columns=ML_DROP_COLUMNS)


def get_training_groups(df: pd.DataFrame, group_column: str = ML_GROUP_COLUMN) -> pd.Series:
    """Return the grouping series used to keep full runs together during validation."""
    if group_column not in df.columns:
        raise ValueError(
            f"No se encontro la columna de agrupacion '{group_column}' en el dataset."
        )

    groups = df[group_column].astype(str)
    if groups.nunique() < 2:
        raise ValueError(
            f"La columna de agrupacion '{group_column}' necesita al menos 2 valores distintos."
        )
    return groups


def resolve_pca_components(feature_count: int | None = None) -> int | float | None:
    """Resolve the configured PCA dimensionality for the current feature space."""
    if not ML_USE_PCA:
        return None

    n_components = ML_PCA_COMPONENTS
    if isinstance(n_components, int):
        if feature_count is None:
            return n_components
        return max(1, min(n_components, feature_count))

    if isinstance(n_components, float):
        if not 0.0 < n_components <= 1.0:
            raise ValueError("ML_PCA_COMPONENTS como float debe estar entre 0 y 1.")
        return n_components

    raise ValueError("ML_PCA_COMPONENTS debe ser int o float.")


def describe_feature_space(feature_count: int | None = None) -> tuple[str, str]:
    """Return human-readable labels for the active feature representation."""
    if not ML_USE_PCA:
        return "raw_features", "disabled"

    resolved = resolve_pca_components(feature_count)
    return "pca_components", str(resolved)


def build_pipeline(model, scale_features: bool, feature_count: int | None = None) -> Pipeline:
    """Build a training pipeline, optionally projecting inputs with PCA."""
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_features or ML_USE_PCA:
        steps.append(("scaler", StandardScaler()))

    if ML_USE_PCA:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=resolve_pca_components(feature_count),
                    svd_solver=ML_PCA_SVD_SOLVER,
                ),
            )
        )

    steps.append(("model", model))
    return Pipeline(steps=steps)


def grouped_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split rows while keeping every run_id entirely in train or test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (
        X.iloc[train_idx].copy(),
        X.iloc[test_idx].copy(),
        y.iloc[train_idx].copy(),
        y.iloc[test_idx].copy(),
        groups.iloc[train_idx].copy(),
        groups.iloc[test_idx].copy(),
    )


def resolve_group_cv_folds(groups: pd.Series, requested_folds: int) -> int:
    """Cap CV folds to the number of available groups."""
    unique_groups = int(groups.nunique())
    if unique_groups < 2:
        raise ValueError("Se necesitan al menos 2 grupos para validacion cruzada agrupada.")
    return max(2, min(requested_folds, unique_groups))


def build_models(feature_count: int | None = None) -> dict[str, Pipeline]:
    """Create the regression models to compare."""
    models: dict[str, Pipeline] = {
        "ridge": build_pipeline(
            Ridge(**ML_MODEL_CONFIGS["ridge"]),
            scale_features=True,
            feature_count=feature_count,
        ),
        "lasso": build_pipeline(
            Lasso(**ML_MODEL_CONFIGS["lasso"]),
            scale_features=True,
            feature_count=feature_count,
        ),
        "svr_rbf": build_pipeline(
            SVR(**ML_MODEL_CONFIGS["svr_rbf"]),
            scale_features=True,
            feature_count=feature_count,
        ),
    }

    if XGBRegressor is not None:
        models["xgboost"] = build_pipeline(
            XGBRegressor(**ML_MODEL_CONFIGS["xgboost"]),
            scale_features=False,
            feature_count=feature_count,
        )

    return models


def build_classification_models(feature_count: int | None = None) -> dict[str, Pipeline]:
    """Create the classification models to compare."""
    models: dict[str, Pipeline] = {
        "logistic_regression": build_pipeline(
            LogisticRegression(**ML_MODEL_CONFIGS["logistic_regression"]),
            scale_features=True,
            feature_count=feature_count,
        ),
        "svc_rbf": build_pipeline(
            SVC(**ML_MODEL_CONFIGS["svc_rbf"]),
            scale_features=True,
            feature_count=feature_count,
        ),
    }

    if XGBClassifier is not None:
        models["xgboost_classifier"] = build_pipeline(
            XGBClassifier(**ML_MODEL_CONFIGS["xgboost_classifier"]),
            scale_features=False,
            feature_count=feature_count,
        )

    return models


def _save_fitted_artifact(
    *,
    artifacts_dir: Path | str,
    task: str,
    model_name: str,
    target: str,
    feature_columns: list[str],
    pipeline: Pipeline,
    dataset_csv: Path,
    trained_rows: int,
) -> SavedModelArtifact:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = _artifact_model_path(artifacts_dir, task, model_name)
    feature_space, pca_components = describe_feature_space(len(feature_columns))
    payload = {
        "task": task,
        "model_name": model_name,
        "target": target,
        "feature_columns": feature_columns,
        "pipeline": pipeline,
        "dataset_csv": str(dataset_csv),
        "trained_rows": int(trained_rows),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_space": feature_space,
        "pca_components": pca_components,
        "group_column": ML_GROUP_COLUMN,
        "split_strategy": ML_SPLIT_STRATEGY,
    }
    joblib.dump(payload, artifact_path)
    return SavedModelArtifact(
        task=task,
        model_name=model_name,
        target=target,
        feature_columns=list(feature_columns),
        artifact_path=artifact_path,
        trained_rows=int(trained_rows),
    )


def fit_and_save_inference_models(
    csv_path: Path | str = DEFAULT_OUTPUT_CSV,
    *,
    regressor_name: str | None = None,
    classifier_name: str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, SavedModelArtifact]:
    """Fit the selected models on the full dataset and persist them for inference."""
    csv_path = Path(csv_path)
    artifacts_dir = Path(output_dir) if output_dir is not None else default_artifact_dir(csv_path)
    df = load_dataset(csv_path)

    X_reg, y_reg = select_features(df, ML_TARGET_REGRESSION)
    X_cls, y_cls = select_features(df, ML_TARGET_CLASSIFICATION)
    y_cls = y_cls.fillna(0).astype(int)
    if y_cls.nunique() < 2:
        raise ValueError("El target flooded tiene una sola clase; no se puede entrenar clasificador.")

    regression_models = build_models(feature_count=X_reg.shape[1])
    classification_models = build_classification_models(feature_count=X_cls.shape[1])
    selected_regressor_name = _selected_model_name(
        regression_models,
        regressor_name,
        default_regression_model_name(),
    )
    selected_classifier_name = _selected_model_name(
        classification_models,
        classifier_name,
        default_classification_model_name(),
    )

    regressor = regression_models[selected_regressor_name]
    classifier = classification_models[selected_classifier_name]
    regressor.fit(X_reg, y_reg)
    classifier.fit(X_cls, y_cls)

    regression_artifact = _save_fitted_artifact(
        artifacts_dir=artifacts_dir,
        task="regression",
        model_name=selected_regressor_name,
        target=ML_TARGET_REGRESSION,
        feature_columns=X_reg.columns.tolist(),
        pipeline=regressor,
        dataset_csv=csv_path,
        trained_rows=len(X_reg),
    )
    classification_artifact = _save_fitted_artifact(
        artifacts_dir=artifacts_dir,
        task="classification",
        model_name=selected_classifier_name,
        target=ML_TARGET_CLASSIFICATION,
        feature_columns=X_cls.columns.tolist(),
        pipeline=classifier,
        dataset_csv=csv_path,
        trained_rows=len(X_cls),
    )

    manifest = _load_manifest(artifacts_dir)
    manifest["latest"]["regression"] = selected_regressor_name
    manifest["latest"]["classification"] = selected_classifier_name
    manifest["regression"][selected_regressor_name] = {
        "artifact_file": regression_artifact.artifact_path.name,
        "target": regression_artifact.target,
        "feature_columns": regression_artifact.feature_columns,
        "trained_rows": regression_artifact.trained_rows,
        "dataset_csv": str(csv_path),
    }
    manifest["classification"][selected_classifier_name] = {
        "artifact_file": classification_artifact.artifact_path.name,
        "target": classification_artifact.target,
        "feature_columns": classification_artifact.feature_columns,
        "trained_rows": classification_artifact.trained_rows,
        "dataset_csv": str(csv_path),
    }
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(artifacts_dir, manifest)

    return {
        "regression": regression_artifact,
        "classification": classification_artifact,
    }


def load_saved_model_artifact(
    task: str,
    artifacts_dir: Path | str = DEFAULT_MODEL_ARTIFACTS_DIR,
    *,
    model_name: str | None = None,
) -> LoadedModelArtifact:
    """Load one persisted inference artifact from disk."""
    if task not in {"regression", "classification"}:
        raise ValueError("task debe ser 'regression' o 'classification'.")

    artifacts_dir = Path(artifacts_dir)
    manifest = _load_manifest(artifacts_dir)
    selected_name = model_name or manifest.get("latest", {}).get(task)
    if not selected_name:
        raise FileNotFoundError(
            f"No hay artefacto guardado para '{task}' en {artifacts_dir}. "
            "Entrena y guarda modelos primero."
        )

    artifact_path = _artifact_model_path(artifacts_dir, task, selected_name)
    if not artifact_path.exists():
        raise FileNotFoundError(f"No existe el artefacto esperado: {artifact_path}")

    payload = joblib.load(artifact_path)
    return LoadedModelArtifact(
        task=str(payload["task"]),
        model_name=str(payload["model_name"]),
        target=str(payload["target"]),
        feature_columns=list(payload["feature_columns"]),
        artifact_path=artifact_path,
        pipeline=payload["pipeline"],
        metadata={
            key: value
            for key, value in payload.items()
            if key not in {"pipeline", "feature_columns", "task", "model_name", "target"}
        },
    )


def compute_test_metrics(y_true, y_pred) -> tuple[float, float, float]:
    """Compute MAE, RMSE and R2 on the held-out test set."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return float(mae), float(rmse), float(r2)


def compute_cv_metrics(
    pipeline,
    X_train,
    y_train,
    groups_train: pd.Series,
    cv_folds: int,
) -> dict[str, float]:
    """Run cross-validation on the training split only."""
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    effective_folds = resolve_group_cv_folds(groups_train, cv_folds)
    cv_result = cross_validate(
        pipeline,
        X_train,
        y_train,
        groups=groups_train,
        cv=GroupKFold(n_splits=effective_folds),
        scoring=scoring,
        n_jobs=None,
        return_train_score=False,
    )
    return {
        "cv_folds_effective": float(effective_folds),
        "cv_mae_mean": float(-cv_result["test_mae"].mean()),
        "cv_mae_std": float(cv_result["test_mae"].std()),
        "cv_rmse_mean": float(-cv_result["test_rmse"].mean()),
        "cv_rmse_std": float(cv_result["test_rmse"].std()),
        "cv_r2_mean": float(cv_result["test_r2"].mean()),
        "cv_r2_std": float(cv_result["test_r2"].std()),
    }


def compute_classification_test_metrics(y_true, y_pred) -> tuple[float, float, float, float]:
    """Compute classification metrics on the held-out test set."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return float(accuracy), float(precision), float(recall), float(f1)


def compute_classification_cv_metrics(
    pipeline,
    X_train,
    y_train,
    groups_train: pd.Series,
    cv_folds: int,
) -> dict[str, float]:
    """Run cross-validation for classification on the training split only."""
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    effective_folds = resolve_group_cv_folds(groups_train, cv_folds)
    cv_result = cross_validate(
        pipeline,
        X_train,
        y_train,
        groups=groups_train,
        cv=GroupKFold(n_splits=effective_folds),
        scoring=scoring,
        n_jobs=None,
        return_train_score=False,
    )
    return {
        "cv_folds_effective": float(effective_folds),
        "cv_accuracy_mean": float(cv_result["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv_result["test_accuracy"].std()),
        "cv_precision_mean": float(cv_result["test_precision"].mean()),
        "cv_precision_std": float(cv_result["test_precision"].std()),
        "cv_recall_mean": float(cv_result["test_recall"].mean()),
        "cv_recall_std": float(cv_result["test_recall"].std()),
        "cv_f1_mean": float(cv_result["test_f1"].mean()),
        "cv_f1_std": float(cv_result["test_f1"].std()),
    }


def normalize_scenario_label(raw_value: object) -> str:
    """Map raw scenario types to compact labels used in reports."""
    value = str(raw_value).strip().lower()
    if value == "hydrograph_inflow":
        return "hydrograph"
    if value == "embedded_inflow_multiplier_sweep":
        return "embedded_hydrograph"
    if value in {"uniform_inflow_multiplier_sweep", "uniform_inflow_sweep", "steady_inflow"}:
        return "steady"

    cleaned = "".join(character if character.isalnum() else "_" for character in value)
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def available_scenario_labels(df: pd.DataFrame) -> list[str]:
    """Return the normalized scenario labels present in the dataset."""
    if "scenario_type" not in df.columns:
        return []

    labels: list[str] = []
    seen: set[str] = set()
    for raw_value in df["scenario_type"].dropna().astype(str):
        label = normalize_scenario_label(raw_value)
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def scenario_labels_for_rows(df: pd.DataFrame, row_index: pd.Index) -> pd.Series:
    """Return normalized scenario labels aligned to the provided row index."""
    if "scenario_type" not in df.columns:
        return pd.Series(index=row_index, dtype="object")
    labels = df.loc[row_index, "scenario_type"].fillna("unknown").map(normalize_scenario_label)
    labels.index = row_index
    return labels


def scenario_regression_metrics(
    y_true: pd.Series,
    y_pred,
    scenario_labels: pd.Series,
    available_labels: list[str],
) -> dict[str, float | int | None]:
    """Compute regression metrics for each scenario subset in the held-out test split."""
    metrics: dict[str, float | int | None] = {}
    prediction_series = pd.Series(np.asarray(y_pred), index=y_true.index)

    for label in available_labels:
        mask = scenario_labels == label
        count = int(mask.sum())
        metrics[f"scenario_rows_{label}"] = count
        if count == 0:
            metrics[f"mae_{label}"] = None
            metrics[f"rmse_{label}"] = None
            metrics[f"r2_{label}"] = None
            continue

        y_subset = y_true.loc[mask]
        pred_subset = prediction_series.loc[mask]
        metrics[f"mae_{label}"] = float(mean_absolute_error(y_subset, pred_subset))
        metrics[f"rmse_{label}"] = float(np.sqrt(mean_squared_error(y_subset, pred_subset)))
        metrics[f"r2_{label}"] = float(r2_score(y_subset, pred_subset)) if len(y_subset) >= 2 else None
    return metrics


def scenario_classification_metrics(
    y_true: pd.Series,
    y_pred,
    scenario_labels: pd.Series,
    available_labels: list[str],
) -> dict[str, float | int | None]:
    """Compute classification metrics for each scenario subset in the held-out test split."""
    metrics: dict[str, float | int | None] = {}
    prediction_series = pd.Series(np.asarray(y_pred), index=y_true.index)

    for label in available_labels:
        mask = scenario_labels == label
        count = int(mask.sum())
        metrics[f"scenario_rows_{label}"] = count
        if count == 0:
            metrics[f"accuracy_{label}"] = None
            metrics[f"precision_{label}"] = None
            metrics[f"recall_{label}"] = None
            metrics[f"f1_{label}"] = None
            continue

        y_subset = y_true.loc[mask]
        pred_subset = prediction_series.loc[mask]
        accuracy, precision, recall, f1 = compute_classification_test_metrics(y_subset, pred_subset)
        metrics[f"accuracy_{label}"] = accuracy
        metrics[f"precision_{label}"] = precision
        metrics[f"recall_{label}"] = recall
        metrics[f"f1_{label}"] = f1
    return metrics


def evaluate_models(
    csv_path: Path | str = DEFAULT_OUTPUT_CSV,
    target: str = ML_TARGET_REGRESSION,
    test_size: float = ML_TEST_SIZE,
    random_state: int = ML_RANDOM_STATE,
    cv_folds: int = ML_CV_FOLDS,
) -> pd.DataFrame:
    """Train and compare the available regression models."""
    df = load_dataset(csv_path)
    groups = get_training_groups(df)
    X, y = select_features(df, target)
    feature_space, pca_components = describe_feature_space(X.shape[1])
    scenario_labels = available_scenario_labels(df)

    X_train, X_test, y_train, y_test, groups_train, groups_test = grouped_train_test_split(
        X,
        y,
        groups,
        test_size=test_size,
        random_state=random_state,
    )

    results: list[dict[str, object]] = []
    models = build_models(feature_count=X.shape[1])

    for model_name, pipeline in models.items():
        cv_metrics = compute_cv_metrics(
            pipeline,
            X_train,
            y_train,
            groups_train,
            cv_folds=cv_folds,
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae, rmse, r2 = compute_test_metrics(y_test, predictions)
        test_scenarios = scenario_labels_for_rows(df, X_test.index)
        subset_metrics = scenario_regression_metrics(
            y_test,
            predictions,
            test_scenarios,
            scenario_labels,
        )

        row = ModelResult(
            model_name=model_name,
            target=target,
            feature_space=feature_space,
            pca_components=pca_components,
            split_strategy=ML_SPLIT_STRATEGY,
            group_column=ML_GROUP_COLUMN,
            train_group_count=int(groups_train.nunique()),
            test_group_count=int(groups_test.nunique()),
            train_size=len(X_train),
            test_size=len(X_test),
            random_state=random_state,
            cv_folds=int(cv_metrics["cv_folds_effective"]),
            mae=mae,
            rmse=rmse,
            r2=r2,
            cv_mae_mean=cv_metrics["cv_mae_mean"],
            cv_mae_std=cv_metrics["cv_mae_std"],
            cv_rmse_mean=cv_metrics["cv_rmse_mean"],
            cv_rmse_std=cv_metrics["cv_rmse_std"],
            cv_r2_mean=cv_metrics["cv_r2_mean"],
            cv_r2_std=cv_metrics["cv_r2_std"],
        )
        results.append({**row.__dict__, **subset_metrics})

    results_df = pd.DataFrame(results).sort_values(
        by=["cv_r2_mean", "r2", "cv_rmse_mean", "rmse", "cv_mae_mean", "mae"],
        ascending=[False, False, True, True, True, True],
    )
    return results_df


def evaluate_classification_models(
    csv_path: Path | str = DEFAULT_OUTPUT_CSV,
    target: str = ML_TARGET_CLASSIFICATION,
    test_size: float = ML_TEST_SIZE,
    random_state: int = ML_RANDOM_STATE,
    cv_folds: int = ML_CV_FOLDS,
) -> pd.DataFrame:
    """Train and compare the available classification models."""
    df = load_dataset(csv_path)
    groups = get_training_groups(df)
    X, y = select_features(df, target)
    y = y.fillna(0).astype(int)
    feature_space, pca_components = describe_feature_space(X.shape[1])
    scenario_labels = available_scenario_labels(df)

    X_train, X_test, y_train, y_test, groups_train, groups_test = grouped_train_test_split(
        X,
        y,
        groups,
        test_size=test_size,
        random_state=random_state,
    )

    results: list[dict[str, object]] = []
    models = build_classification_models(feature_count=X.shape[1])

    for model_name, pipeline in models.items():
        cv_metrics = compute_classification_cv_metrics(
            pipeline,
            X_train,
            y_train,
            groups_train,
            cv_folds=cv_folds,
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy, precision, recall, f1 = compute_classification_test_metrics(y_test, predictions)
        test_scenarios = scenario_labels_for_rows(df, X_test.index)
        subset_metrics = scenario_classification_metrics(
            y_test,
            predictions,
            test_scenarios,
            scenario_labels,
        )

        row = ClassificationResult(
            model_name=model_name,
            target=target,
            feature_space=feature_space,
            pca_components=pca_components,
            split_strategy=ML_SPLIT_STRATEGY,
            group_column=ML_GROUP_COLUMN,
            train_group_count=int(groups_train.nunique()),
            test_group_count=int(groups_test.nunique()),
            train_size=len(X_train),
            test_size=len(X_test),
            random_state=random_state,
            cv_folds=int(cv_metrics["cv_folds_effective"]),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            cv_accuracy_mean=cv_metrics["cv_accuracy_mean"],
            cv_accuracy_std=cv_metrics["cv_accuracy_std"],
            cv_precision_mean=cv_metrics["cv_precision_mean"],
            cv_precision_std=cv_metrics["cv_precision_std"],
            cv_recall_mean=cv_metrics["cv_recall_mean"],
            cv_recall_std=cv_metrics["cv_recall_std"],
            cv_f1_mean=cv_metrics["cv_f1_mean"],
            cv_f1_std=cv_metrics["cv_f1_std"],
        )
        results.append({**row.__dict__, **subset_metrics})

    return pd.DataFrame(results).sort_values(
        by=["cv_f1_mean", "f1", "cv_recall_mean", "recall", "cv_accuracy_mean", "accuracy"],
        ascending=[False, False, False, False, False, False],
    )


def save_results(
    results_df: pd.DataFrame,
    target: str,
    prefix: str = "model_comparison",
    output_dir: Path | str | None = None,
):
    results_dir = Path(output_dir) if output_dir is not None else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = results_dir / f"{prefix}_{target}.csv"
    xlsx_file = results_dir / f"{prefix}_{target}.xlsx"
    results_df.to_csv(csv_file, index=False)
    engine = _available_excel_engine()
    if engine is not None:
        try:
            results_df.to_excel(xlsx_file, index=False, engine=engine)
        except Exception:
            xlsx_file = None
    else:
        xlsx_file = None

    print("\nArchivos generados:")
    print(f"  CSV : {csv_file}")
    if xlsx_file is not None:
        print(f"  XLSX: {xlsx_file}")


def print_results_table(results_df: pd.DataFrame):
    display_columns = [
        "model_name",
        "target",
        "feature_space",
        "pca_components",
        "split_strategy",
        "train_group_count",
        "test_group_count",
        "train_size",
        "test_size",
        "mae",
        "rmse",
        "r2",
        "cv_mae_mean",
        "cv_rmse_mean",
        "cv_r2_mean",
    ]
    printable = results_df[display_columns].copy()
    numeric_cols = [
        "mae",
        "rmse",
        "r2",
        "cv_mae_mean",
        "cv_rmse_mean",
        "cv_r2_mean",
    ]
    for col in numeric_cols:
        printable[col] = printable[col].map(lambda value: f"{value:.4f}")

    separator = "=" * 110
    print(f"\n{separator}")
    print("COMPARACION DE MODELOS")
    print(separator)
    print(printable.to_string(index=False))
    print(separator)


def print_regression_scenario_breakdown(results_df: pd.DataFrame):
    scenario_labels = sorted(
        column.replace("scenario_rows_", "")
        for column in results_df.columns
        if column.startswith("scenario_rows_")
    )
    if not scenario_labels:
        return

    display_columns = ["model_name"]
    for label in scenario_labels:
        display_columns.extend(
            [f"scenario_rows_{label}", f"mae_{label}", f"rmse_{label}", f"r2_{label}"]
        )

    printable = results_df[display_columns].copy()
    for column in printable.columns:
        if column == "model_name" or column.startswith("scenario_rows_"):
            continue
        printable[column] = printable[column].map(
            lambda value: f"{value:.4f}" if pd.notna(value) else "n/a"
        )

    separator = "=" * 140
    print(f"\n{separator}")
    print("DESEMPENO POR TIPO DE ESCENARIO")
    print(separator)
    print(printable.to_string(index=False))
    print(separator)


def print_classification_results_table(results_df: pd.DataFrame):
    display_columns = [
        "model_name",
        "target",
        "feature_space",
        "pca_components",
        "split_strategy",
        "train_group_count",
        "test_group_count",
        "train_size",
        "test_size",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "cv_accuracy_mean",
        "cv_precision_mean",
        "cv_recall_mean",
        "cv_f1_mean",
    ]
    printable = results_df[display_columns].copy()
    numeric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "cv_accuracy_mean",
        "cv_precision_mean",
        "cv_recall_mean",
        "cv_f1_mean",
    ]
    for col in numeric_cols:
        printable[col] = printable[col].map(lambda value: f"{value:.4f}")

    separator = "=" * 130
    print(f"\n{separator}")
    print("COMPARACION DE MODELOS DE CLASIFICACION")
    print(separator)
    print(printable.to_string(index=False))
    print(separator)


def print_classification_scenario_breakdown(results_df: pd.DataFrame):
    scenario_labels = sorted(
        column.replace("scenario_rows_", "")
        for column in results_df.columns
        if column.startswith("scenario_rows_")
    )
    if not scenario_labels:
        return

    display_columns = ["model_name"]
    for label in scenario_labels:
        display_columns.extend(
            [
                f"scenario_rows_{label}",
                f"accuracy_{label}",
                f"precision_{label}",
                f"recall_{label}",
                f"f1_{label}",
            ]
        )

    printable = results_df[display_columns].copy()
    for column in printable.columns:
        if column == "model_name" or column.startswith("scenario_rows_"):
            continue
        printable[column] = printable[column].map(
            lambda value: f"{value:.4f}" if pd.notna(value) else "n/a"
        )

    separator = "=" * 160
    print(f"\n{separator}")
    print("CLASIFICACION POR TIPO DE ESCENARIO")
    print(separator)
    print(printable.to_string(index=False))
    print(separator)


def main():
    regression_target = ML_TARGET_REGRESSION
    classification_target = ML_TARGET_CLASSIFICATION
    preferred_regressor = default_regression_model_name()
    preferred_classifier = default_classification_model_name()
    artifacts_dir = default_artifact_dir(DEFAULT_OUTPUT_CSV)
    feature_space, pca_components = describe_feature_space()
    print(f"Comparando modelos para target de regresion: {regression_target}")
    print(f"Comparando modelos para target de clasificacion: {classification_target}")
    print(f"Dataset: {DEFAULT_OUTPUT_CSV}")
    print(f"Directorio artefactos: {artifacts_dir}")
    print(f"Test size: {ML_TEST_SIZE}")
    print(f"Random state: {ML_RANDOM_STATE}")
    print(f"CV folds: {ML_CV_FOLDS}")
    print(f"Split strategy: {ML_SPLIT_STRATEGY}")
    print(f"Group column: {ML_GROUP_COLUMN}")
    print(f"Espacio de features: {feature_space}")
    print(f"Componentes PCA: {pca_components}")
    print(f"Modelos de regresion: {', '.join(build_models().keys())}")
    print(f"Modelos de clasificacion: {', '.join(build_classification_models().keys())}")
    print(f"Regresor para persistencia: {preferred_regressor}")
    print(f"Clasificador para persistencia: {preferred_classifier}")

    regression_df = evaluate_models(
        target=regression_target,
        test_size=ML_TEST_SIZE,
        random_state=ML_RANDOM_STATE,
        cv_folds=ML_CV_FOLDS,
    )
    print_results_table(regression_df)
    print_regression_scenario_breakdown(regression_df)
    save_results(regression_df, regression_target, prefix="regression_comparison")

    classification_df = evaluate_classification_models(
        target=classification_target,
        test_size=ML_TEST_SIZE,
        random_state=ML_RANDOM_STATE,
        cv_folds=ML_CV_FOLDS,
    )
    print_classification_results_table(classification_df)
    print_classification_scenario_breakdown(classification_df)
    save_results(classification_df, classification_target, prefix="classification_comparison")
    artifacts = fit_and_save_inference_models(
        csv_path=DEFAULT_OUTPUT_CSV,
        regressor_name=preferred_regressor,
        classifier_name=preferred_classifier,
        output_dir=artifacts_dir,
    )
    print("\nArtefactos de inferencia actualizados:")
    print(f"  Regresion    : {artifacts['regression'].artifact_path}")
    print(f"  Clasificacion: {artifacts['classification'].artifact_path}")


if __name__ == "__main__":
    main()
