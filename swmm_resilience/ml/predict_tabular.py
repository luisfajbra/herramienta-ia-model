"""
Tabular ML prediction for new steady-flow scenarios without running PySWMM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import train
from ..config import (
    DEFAULT_OUTPUT_CSV,
    ML_RANDOM_STATE,
    ML_TARGET_CLASSIFICATION,
    ML_TARGET_REGRESSION,
)


@dataclass(frozen=True)
class TabularPredictionResult:
    """Prediction output plus the model names used."""

    predictions: pd.DataFrame
    classifier_name: str
    regressor_name: str


def available_regression_models() -> list[str]:
    """Return available tabular regression model names."""
    return list(train.build_models().keys())


def available_classification_models() -> list[str]:
    """Return available tabular classification model names."""
    return list(train.build_classification_models().keys())


def _selected_model(models: dict, selected_name: str | None, preferred_name: str):
    model_name = selected_name or preferred_name
    if model_name not in models:
        available = ", ".join(models.keys())
        raise ValueError(f"El modelo '{model_name}' no esta disponible. Opciones: {available}")
    return model_name, models[model_name]


def _scenario_input_column(df: pd.DataFrame) -> str:
    if "inflow_multiplier" in df.columns:
        return "inflow_multiplier"
    if "delta_inflow_lps" in df.columns:
        return "delta_inflow_lps"
    raise ValueError("El dataset no tiene columna inflow_multiplier ni delta_inflow_lps.")


def _build_prediction_rows(
    df: pd.DataFrame,
    inflow_multipliers: list[float],
    target_nodes: list[str] | None,
) -> pd.DataFrame:
    if "node_id" not in df.columns:
        raise ValueError("El dataset no tiene columna node_id.")
    scenario_column = _scenario_input_column(df)

    base_df = df.copy()
    if target_nodes is not None:
        target_set = {str(node_id) for node_id in target_nodes}
        base_df = base_df[base_df["node_id"].astype(str).isin(target_set)]
        missing = sorted(target_set - set(base_df["node_id"].astype(str)))
        if missing:
            raise ValueError(f"Estos nodos no existen en el dataset: {', '.join(missing)}")

    # One representative row per node keeps static topology fields. Dynamic
    # result columns are excluded later by ML_DROP_COLUMNS.
    base_rows = base_df.drop_duplicates(subset=["node_id"], keep="last")
    prediction_rows = []
    for inflow_multiplier in inflow_multipliers:
        scenario_rows = base_rows.copy()
        scenario_rows[scenario_column] = inflow_multiplier
        if "scenario_type" in scenario_rows.columns:
            scenario_rows["scenario_type"] = "ml_steady_prediction"
        if "spatial_pattern" in scenario_rows.columns:
            scenario_rows["spatial_pattern"] = "all_nodes" if target_nodes is None else "selected_nodes"
        prediction_rows.append(scenario_rows)

    return pd.concat(prediction_rows, ignore_index=True)


def align_feature_columns(rows: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned = rows.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    return aligned[feature_columns]


def predict_steady_flows(
    inflow_multipliers: list[float],
    dataset_csv: Path | str = DEFAULT_OUTPUT_CSV,
    target_nodes: list[str] | None = None,
    classifier_name: str | None = None,
    regressor_name: str | None = None,
) -> TabularPredictionResult:
    """Predict flooded state and flooding volume for new steady-flow multipliers."""
    dataset_csv = Path(dataset_csv)
    df = train.load_dataset(dataset_csv)
    scenario_column = _scenario_input_column(df)
    prediction_rows = _build_prediction_rows(df, inflow_multipliers, target_nodes)

    X_reg, y_reg = train.select_features(df, ML_TARGET_REGRESSION)
    X_cls, y_cls = train.select_features(df, ML_TARGET_CLASSIFICATION)
    y_cls = y_cls.fillna(0).astype(int)
    if y_cls.nunique() < 2:
        raise ValueError("El target flooded tiene una sola clase; no se puede entrenar clasificador.")

    regression_models = train.build_models(feature_count=X_reg.shape[1])
    classification_models = train.build_classification_models(feature_count=X_cls.shape[1])
    regressor_name, regressor = _selected_model(
        regression_models,
        regressor_name,
        train.default_regression_model_name(),
    )
    classifier_name, classifier = _selected_model(
        classification_models,
        classifier_name,
        train.default_classification_model_name(),
    )

    regressor.fit(X_reg, y_reg)
    classifier.fit(X_cls, y_cls)

    X_pred_reg = align_feature_columns(prediction_rows, X_reg.columns.tolist())
    X_pred_cls = align_feature_columns(prediction_rows, X_cls.columns.tolist())

    predicted_volume = regressor.predict(X_pred_reg)
    predicted_volume = pd.Series(predicted_volume).clip(lower=0.0).to_numpy()
    predicted_flooded = classifier.predict(X_pred_cls)
    if hasattr(classifier, "predict_proba"):
        flooded_probability = classifier.predict_proba(X_pred_cls)[:, 1]
    else:
        flooded_probability = predicted_flooded

    output = pd.DataFrame(
        {
            "node_id": prediction_rows["node_id"].astype(str),
            scenario_column: prediction_rows[scenario_column],
            "predicted_flooded": predicted_flooded.astype(int),
            "flooded_probability": flooded_probability,
            "predicted_flooding_volume_m3": predicted_volume,
        }
    )
    output["model_random_state"] = ML_RANDOM_STATE
    return TabularPredictionResult(
        predictions=output.sort_values([scenario_column, "node_id"]).reset_index(drop=True),
        classifier_name=classifier_name,
        regressor_name=regressor_name,
    )
