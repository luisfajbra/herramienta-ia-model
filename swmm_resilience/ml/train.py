"""
Compare regression models on the exported SWMM dataset.

Usage:
    python -m swmm_resilience.ml.train
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR

from ..config import (
    DEFAULT_OUTPUT_CSV,
    DEFAULT_RESULTS_DIR,
    ML_CV_FOLDS,
    ML_DROP_COLUMNS,
    ML_MODEL_CONFIGS,
    ML_RANDOM_STATE,
    ML_TARGET_CLASSIFICATION,
    ML_TARGET_REGRESSION,
    ML_TEST_SIZE,
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


def load_dataset(csv_path: Path | str = DEFAULT_OUTPUT_CSV) -> pd.DataFrame:
    """Load the exported dataset."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontro el dataset: {csv_path}")
    return pd.read_csv(csv_path)


def select_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Select numeric input features and one target column."""
    if target not in df.columns:
        raise ValueError(f"El target '{target}' no existe en el dataset.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ML_DROP_COLUMNS and col != target]
    if not feature_cols:
        raise ValueError("No se encontraron features numericas para entrenar.")

    X = df[feature_cols].copy()
    y = df[target].fillna(0.0).copy()
    return X, y


def build_scaled_pipeline(model) -> Pipeline:
    """Pipeline with imputation and scaling for models that depend on feature scale."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_tree_pipeline(model) -> Pipeline:
    """Pipeline with imputation only for tree-based models."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    """Create the regression models to compare."""
    models: dict[str, Pipeline] = {
        "ridge": build_scaled_pipeline(Ridge(**ML_MODEL_CONFIGS["ridge"])),
        "lasso": build_scaled_pipeline(Lasso(**ML_MODEL_CONFIGS["lasso"])),
        "svr_rbf": build_scaled_pipeline(SVR(**ML_MODEL_CONFIGS["svr_rbf"])),
    }

    if XGBRegressor is not None:
        models["xgboost"] = build_tree_pipeline(XGBRegressor(**ML_MODEL_CONFIGS["xgboost"]))

    return models


def build_classification_models() -> dict[str, Pipeline]:
    """Create the classification models to compare."""
    models: dict[str, Pipeline] = {
        "logistic_regression": build_scaled_pipeline(
            LogisticRegression(**ML_MODEL_CONFIGS["logistic_regression"])
        ),
        "svc_rbf": build_scaled_pipeline(SVC(**ML_MODEL_CONFIGS["svc_rbf"])),
    }

    if XGBClassifier is not None:
        models["xgboost_classifier"] = build_tree_pipeline(
            XGBClassifier(**ML_MODEL_CONFIGS["xgboost_classifier"])
        )

    return models


def compute_test_metrics(y_true, y_pred) -> tuple[float, float, float]:
    """Compute MAE, RMSE and R2 on the held-out test set."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return float(mae), float(rmse), float(r2)


def compute_cv_metrics(pipeline, X_train, y_train, cv_folds: int) -> dict[str, float]:
    """Run cross-validation on the training split only."""
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    cv_result = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=None,
        return_train_score=False,
    )
    return {
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


def compute_classification_cv_metrics(pipeline, X_train, y_train, cv_folds: int) -> dict[str, float]:
    """Run cross-validation for classification on the training split only."""
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    cv_result = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=None,
        return_train_score=False,
    )
    return {
        "cv_accuracy_mean": float(cv_result["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv_result["test_accuracy"].std()),
        "cv_precision_mean": float(cv_result["test_precision"].mean()),
        "cv_precision_std": float(cv_result["test_precision"].std()),
        "cv_recall_mean": float(cv_result["test_recall"].mean()),
        "cv_recall_std": float(cv_result["test_recall"].std()),
        "cv_f1_mean": float(cv_result["test_f1"].mean()),
        "cv_f1_std": float(cv_result["test_f1"].std()),
    }


def evaluate_models(
    csv_path: Path | str = DEFAULT_OUTPUT_CSV,
    target: str = ML_TARGET_REGRESSION,
    test_size: float = ML_TEST_SIZE,
    random_state: int = ML_RANDOM_STATE,
    cv_folds: int = ML_CV_FOLDS,
) -> pd.DataFrame:
    """Train and compare the available regression models."""
    df = load_dataset(csv_path)
    X, y = select_features(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    results: list[ModelResult] = []
    models = build_models()

    for model_name, pipeline in models.items():
        cv_metrics = compute_cv_metrics(pipeline, X_train, y_train, cv_folds=cv_folds)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae, rmse, r2 = compute_test_metrics(y_test, predictions)

        results.append(
            ModelResult(
                model_name=model_name,
                target=target,
                train_size=len(X_train),
                test_size=len(X_test),
                random_state=random_state,
                cv_folds=cv_folds,
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
        )

    results_df = pd.DataFrame([result.__dict__ for result in results]).sort_values(
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
    X, y = select_features(df, target)
    y = y.fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    results: list[ClassificationResult] = []
    models = build_classification_models()

    for model_name, pipeline in models.items():
        cv_metrics = compute_classification_cv_metrics(pipeline, X_train, y_train, cv_folds=cv_folds)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy, precision, recall, f1 = compute_classification_test_metrics(y_test, predictions)

        results.append(
            ClassificationResult(
                model_name=model_name,
                target=target,
                train_size=len(X_train),
                test_size=len(X_test),
                random_state=random_state,
                cv_folds=cv_folds,
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
        )

    return pd.DataFrame([result.__dict__ for result in results]).sort_values(
        by=["cv_f1_mean", "f1", "cv_recall_mean", "recall", "cv_accuracy_mean", "accuracy"],
        ascending=[False, False, False, False, False, False],
    )


def save_results(results_df: pd.DataFrame, target: str, prefix: str = "model_comparison"):
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_file = DEFAULT_RESULTS_DIR / f"{prefix}_{target}.csv"
    xlsx_file = DEFAULT_RESULTS_DIR / f"{prefix}_{target}.xlsx"
    results_df.to_csv(csv_file, index=False)
    try:
        results_df.to_excel(xlsx_file, index=False)
    except Exception:
        xlsx_file = None

    print("\nArchivos generados:")
    print(f"  CSV : {csv_file}")
    if xlsx_file is not None:
        print(f"  XLSX: {xlsx_file}")


def print_results_table(results_df: pd.DataFrame):
    display_columns = [
        "model_name",
        "target",
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


def print_classification_results_table(results_df: pd.DataFrame):
    display_columns = [
        "model_name",
        "target",
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


def main():
    regression_target = ML_TARGET_REGRESSION
    classification_target = ML_TARGET_CLASSIFICATION
    print(f"Comparando modelos para target de regresion: {regression_target}")
    print(f"Comparando modelos para target de clasificacion: {classification_target}")
    print(f"Dataset: {DEFAULT_OUTPUT_CSV}")
    print(f"Test size: {ML_TEST_SIZE}")
    print(f"Random state: {ML_RANDOM_STATE}")
    print(f"CV folds: {ML_CV_FOLDS}")
    print(f"Modelos de regresion: {', '.join(build_models().keys())}")
    print(f"Modelos de clasificacion: {', '.join(build_classification_models().keys())}")

    regression_df = evaluate_models(
        target=regression_target,
        test_size=ML_TEST_SIZE,
        random_state=ML_RANDOM_STATE,
        cv_folds=ML_CV_FOLDS,
    )
    print_results_table(regression_df)
    save_results(regression_df, regression_target, prefix="regression_comparison")

    classification_df = evaluate_classification_models(
        target=classification_target,
        test_size=ML_TEST_SIZE,
        random_state=ML_RANDOM_STATE,
        cv_folds=ML_CV_FOLDS,
    )
    print_classification_results_table(classification_df)
    save_results(classification_df, classification_target, prefix="classification_comparison")


if __name__ == "__main__":
    main()
