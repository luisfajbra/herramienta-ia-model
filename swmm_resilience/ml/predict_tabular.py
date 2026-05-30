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
from ..utils import normalize_inflow_multipliers


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


def _resolve_prediction_network(
    df: pd.DataFrame,
    network_selector: str | None,
) -> tuple[pd.DataFrame, dict[str, str] | None]:
    """Filter CSV predictions to one network, avoiding silent cross-network mixing."""
    required_columns = {"network_hash", "network_file"}
    if not required_columns.issubset(df.columns):
        return df, None

    network_rows = (
        df[["network_hash", "network_file"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["network_file", "network_hash"])
        .reset_index(drop=True)
    )
    if network_rows.empty:
        return df, None

    selected_row = None
    if network_selector:
        normalized = str(network_selector).strip()
        matches = network_rows[
            (network_rows["network_file"] == normalized)
            | (network_rows["network_hash"] == normalized)
            | (network_rows["network_hash"].astype(str).str.startswith(normalized))
        ]
        if matches.empty:
            available = ", ".join(network_rows["network_file"].astype(str).tolist())
            raise ValueError(
                f"La red '{network_selector}' no existe en el CSV. Redes disponibles: {available}"
            )
        if len(matches) > 1:
            raise ValueError(
                "El selector de red coincide con varias redes del CSV. "
                "Usa el nombre completo del .inp o un network_hash mas largo."
            )
        selected_row = matches.iloc[0]
    else:
        if len(network_rows) > 1:
            available = ", ".join(network_rows["network_file"].astype(str).tolist())
            raise ValueError(
                "El CSV contiene multiples redes y la prediccion desde CSV ya no mezcla redes "
                f"silenciosamente. Indica una red concreta. Disponibles: {available}"
            )
        selected_row = network_rows.iloc[0]

    selected_hash = str(selected_row["network_hash"])
    selected_file = str(selected_row["network_file"])
    filtered = df[
        (df["network_hash"].astype(str) == selected_hash)
        & (df["network_file"].astype(str) == selected_file)
    ].copy()
    return filtered, {"network_hash": selected_hash, "network_file": selected_file}


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
    base_rows = base_df.drop_duplicates(subset=["node_id"], keep="first")
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
    artifacts_dir: Path | str | None = None,
    network_selector: str | None = None,
) -> TabularPredictionResult:
    """Predict flooded state and flooding volume for new steady-flow multipliers."""
    dataset_csv = Path(dataset_csv)
    inflow_multipliers = normalize_inflow_multipliers(
        inflow_multipliers,
        minimum=1.0,
        label="Los factores de prediccion",
    )
    df = train.load_dataset(dataset_csv)
    df, selected_network = _resolve_prediction_network(df, network_selector)
    scenario_column = _scenario_input_column(df)
    prediction_rows = _build_prediction_rows(df, inflow_multipliers, target_nodes)

    resolved_artifacts_dir = (
        Path(artifacts_dir) if artifacts_dir is not None
        else train.default_artifact_dir(dataset_csv)
    )

    reg_artifact = train.load_saved_model_artifact(
        "regression", resolved_artifacts_dir, model_name=regressor_name
    )
    cls_artifact = train.load_saved_model_artifact(
        "classification", resolved_artifacts_dir, model_name=classifier_name
    )

    X_pred_reg = align_feature_columns(prediction_rows, reg_artifact.feature_columns)
    X_pred_cls = align_feature_columns(prediction_rows, cls_artifact.feature_columns)

    predicted_volume = reg_artifact.pipeline.predict(X_pred_reg)
    predicted_volume = pd.Series(predicted_volume).clip(lower=0.0).to_numpy()
    predicted_flooded = cls_artifact.pipeline.predict(X_pred_cls)
    if hasattr(cls_artifact.pipeline, "predict_proba"):
        flooded_probability = cls_artifact.pipeline.predict_proba(X_pred_cls)[:, 1]
    else:
        flooded_probability = predicted_flooded

    output = pd.DataFrame(
        {
            "node_id": prediction_rows["node_id"].astype(str),
            scenario_column: prediction_rows[scenario_column],
            "predicted_flooded": predicted_flooded.astype(int),
            "flooded_probability": flooded_probability,
            "predicted_peak_flooding_lps": predicted_volume,
        }
    )
    if selected_network is not None:
        output.insert(1, "network_hash", selected_network["network_hash"])
        output.insert(2, "network_file", selected_network["network_file"])
    output["model_random_state"] = ML_RANDOM_STATE
    return TabularPredictionResult(
        predictions=output.sort_values([scenario_column, "node_id"]).reset_index(drop=True),
        classifier_name=cls_artifact.model_name,
        regressor_name=reg_artifact.model_name,
    )
