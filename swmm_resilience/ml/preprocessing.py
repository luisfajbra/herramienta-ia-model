"""
Preprocessing pipeline for ML datasets.

Handles feature selection, null handling, and data cleaning.
The actual scaling and imputation are done by model pipelines (during train/test).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from ..config import ML_DROP_COLUMNS


def select_features_for_model(
    df: pd.DataFrame,
    target: str,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select numeric features and target for model training/prediction.

    Steps:
    1. Validate target exists
    2. Drop unnecessary columns (identifiers, irrelevant results)
    3. Select only numeric columns (models need numeric input)
    4. Separate features and target

    Note: Imputation and scaling are handled by the model pipelines,
    not here. This keeps train/test separation clean.

    Args:
        df: Raw dataset from CSV
        target: Target column name ('peak_flooding_lps' or 'flooded')
        drop_columns: List of columns to exclude. If None, uses ML_DROP_COLUMNS.

    Returns:
        Tuple of (X, y) where:
        - X: Feature matrix with numeric columns only (missing values ok, pipeline handles)
        - y: Target series (NaN filled with 0.0 for regression targets)
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataset. Available: {df.columns.tolist()}")

    df = df.copy()

    # Drop unnecessary columns
    drop_list = drop_columns if drop_columns is not None else ML_DROP_COLUMNS
    cols_to_drop = [col for col in drop_list if col in df.columns and col != target]
    df_clean = df.drop(columns=cols_to_drop)

    # Get only numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target from feature list if it's there
    feature_cols = [col for col in numeric_cols if col != target]

    if not feature_cols:
        raise ValueError("No numeric features found after dropping excluded columns.")

    X = df_clean[feature_cols].copy()
    y = df_clean[target].copy()

    # For regression targets, fill NaN with 0.0 (before pipeline)
    if pd.api.types.is_numeric_dtype(y):
        y = y.fillna(0.0)

    return X, y


def get_feature_columns(
    df: pd.DataFrame,
    drop_columns: list[str] | None = None,
    target: str | None = None,
) -> list[str]:
    """
    Get list of feature column names after preprocessing.

    Args:
        df: Raw dataset
        drop_columns: List of columns to exclude (default: ML_DROP_COLUMNS)
        target: Target column (will be excluded from features)

    Returns:
        List of numeric feature column names
    """
    drop_list = drop_columns if drop_columns is not None else ML_DROP_COLUMNS
    cols_to_drop = [col for col in drop_list if col in df.columns]
    if target and target in df.columns:
        cols_to_drop.append(target)

    remaining = [col for col in df.columns if col not in cols_to_drop]
    numeric_features = df[remaining].select_dtypes(include=[np.number]).columns.tolist()
    return numeric_features


def dataset_info(df: pd.DataFrame) -> dict:
    """
    Get preprocessing info about the dataset.

    Returns:
        Dict with stats: total rows, numeric columns, missing values per column
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return {
        "total_rows": len(df),
        "numeric_columns": len(numeric_df.columns),
        "missing_per_column": numeric_df.isnull().sum().to_dict(),
        "missing_total": numeric_df.isnull().sum().sum(),
    }
