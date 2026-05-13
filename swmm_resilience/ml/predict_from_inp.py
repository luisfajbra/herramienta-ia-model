"""
Inference over a new SWMM .inp using previously persisted tabular ML artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Import train/xgboost before PySWMM. On macOS, swmm-toolkit ships its own
# OpenMP runtime and loading PySWMM first can make xgboost fail to load
# `libxgboost.dylib` due to a conflicting `libomp`.
from . import train
from pyswmm import Links, Nodes, Simulation

from ..config import DEFAULT_INP_FILE, DEFAULT_MODEL_ARTIFACTS_DIR
from ..simulation.runner import extract_static_topology
from ..utils import file_hash, normalize_inflow_multipliers
from .predict_tabular import TabularPredictionResult, align_feature_columns


def _base_rows_from_inp(inp_file: Path | str, target_nodes: list[str] | None) -> tuple[pd.DataFrame, str]:
    inp_path = Path(inp_file).expanduser()
    if not inp_path.exists():
        raise FileNotFoundError(f"No existe el archivo .inp: {inp_path}")

    network_hash = file_hash(str(inp_path))
    topology = extract_static_topology(str(inp_path), network_hash, Nodes, Links, Simulation)
    base_rows = pd.DataFrame(topology["node_records"]).rename(columns={"node_uid": "node_id"})
    if base_rows.empty:
        raise ValueError(f"No se pudieron extraer nodos desde {inp_path}")

    if target_nodes is not None:
        requested = {str(node_id) for node_id in target_nodes}
        base_rows = base_rows[base_rows["node_id"].astype(str).isin(requested)]
        missing = sorted(requested - set(base_rows["node_id"].astype(str)))
        if missing:
            raise ValueError(f"Estos nodos no existen en la red: {', '.join(missing)}")

    base_rows["network_file"] = inp_path.name
    return base_rows.reset_index(drop=True), network_hash


def _prediction_rows_from_inp(
    inp_file: Path | str,
    inflow_multipliers: list[float],
    target_nodes: list[str] | None,
) -> tuple[pd.DataFrame, str]:
    base_rows, network_hash = _base_rows_from_inp(inp_file, target_nodes)
    prediction_rows: list[pd.DataFrame] = []
    spatial_pattern = "all_nodes" if target_nodes is None else "selected_nodes"

    for inflow_multiplier in inflow_multipliers:
        scenario_rows = base_rows.copy()
        scenario_rows["inflow_multiplier"] = float(inflow_multiplier)
        scenario_rows["scenario_type"] = "ml_steady_prediction"
        scenario_rows["spatial_pattern"] = spatial_pattern
        prediction_rows.append(scenario_rows)

    return pd.concat(prediction_rows, ignore_index=True), network_hash


def predict_steady_flows_from_inp(
    inflow_multipliers: list[float],
    inp_file: Path | str = DEFAULT_INP_FILE,
    *,
    target_nodes: list[str] | None = None,
    artifacts_dir: Path | str = DEFAULT_MODEL_ARTIFACTS_DIR,
    classifier_name: str | None = None,
    regressor_name: str | None = None,
) -> TabularPredictionResult:
    """Predict flooded state and flooding volume for a new .inp without running SWMM."""
    inflow_multipliers = normalize_inflow_multipliers(
        inflow_multipliers,
        minimum=1.0,
        label="Los factores de prediccion",
    )
    prediction_rows, _network_hash = _prediction_rows_from_inp(
        inp_file=inp_file,
        inflow_multipliers=inflow_multipliers,
        target_nodes=target_nodes,
    )

    regression_artifact = train.load_saved_model_artifact(
        "regression",
        artifacts_dir=artifacts_dir,
        model_name=regressor_name,
    )
    classification_artifact = train.load_saved_model_artifact(
        "classification",
        artifacts_dir=artifacts_dir,
        model_name=classifier_name,
    )

    X_pred_reg = align_feature_columns(prediction_rows, regression_artifact.feature_columns)
    X_pred_cls = align_feature_columns(prediction_rows, classification_artifact.feature_columns)

    predicted_volume = regression_artifact.pipeline.predict(X_pred_reg)
    predicted_volume = pd.Series(predicted_volume).clip(lower=0.0).to_numpy()
    predicted_flooded = classification_artifact.pipeline.predict(X_pred_cls)
    if hasattr(classification_artifact.pipeline, "predict_proba"):
        flooded_probability = classification_artifact.pipeline.predict_proba(X_pred_cls)[:, 1]
    else:
        flooded_probability = predicted_flooded

    output = pd.DataFrame(
        {
            "node_id": prediction_rows["node_id"].astype(str),
            "network_hash": prediction_rows["network_hash"].astype(str),
            "network_file": prediction_rows["network_file"].astype(str),
            "base_inflow_lps": prediction_rows["base_inflow_lps"],
            "inflow_multiplier": prediction_rows["inflow_multiplier"],
            "predicted_flooded": pd.Series(predicted_flooded).astype(int),
            "flooded_probability": flooded_probability,
            "predicted_flooding_volume_m3": predicted_volume,
        }
    )
    return TabularPredictionResult(
        predictions=output.sort_values(["inflow_multiplier", "node_id"]).reset_index(drop=True),
        classifier_name=classification_artifact.model_name,
        regressor_name=regression_artifact.model_name,
    )
