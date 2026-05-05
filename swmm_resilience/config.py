"""
Project configuration.

Keep this file limited to paths and high-level parameters.
"""

from pathlib import Path


def _decimal_range(start: float, stop: float, step: float) -> list[float]:
    """Build a float range with the same stop-exclusive behavior as Python range."""
    values: list[float] = []
    current = float(start)
    stop = float(stop)
    step = float(step)
    if step <= 0:
        raise ValueError("step must be greater than zero.")
    while current < stop - 1e-12:
        values.append(round(current, 6))
        current += step
    return values


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
NETWORKS_DIR = DATA_DIR / "networks"
HYDROGRAPHS_DIR = DATA_DIR / "hydrographs"
TRAINING_DIR = DATA_DIR / "training"
DEFAULT_DATASET_REVIEW_DIR = TRAINING_DIR / "dataset_review"
DEFAULT_NETWORK_KEY = "chico_steady"
DEFAULT_NETWORK_DIR = NETWORKS_DIR / DEFAULT_NETWORK_KEY
DEFAULT_RESULTS_DIR = DEFAULT_NETWORK_DIR / "results"

DEFAULT_INP_FILE = DEFAULT_NETWORK_DIR / "SWMM - Chico (PVC) Prueba 1 - Steady.inp"
LEGACY_INP_FILE = BASE_DIR / "SWMM - Chico (PVC) Prueba 1 - Steady.inp"

DEFAULT_DB_FILE = TRAINING_DIR / "swmm_resilience.db"
DEFAULT_OUTPUT_CSV = DEFAULT_RESULTS_DIR / "dataset_ml.csv"
DEFAULT_MODEL_ARTIFACTS_DIR = DEFAULT_RESULTS_DIR / "model_artifacts"

DEFAULT_INFLOW_MULTIPLIERS = _decimal_range(0.1, 2.0, 0.1)
DEFAULT_HYDROGRAPH_FILE = None
DEFAULT_TARGET_NODES = None
DEFAULT_SCENARIO_TYPE = "uniform_inflow_multiplier_sweep"
DEFAULT_SPATIAL_PATTERN = "uniform"

# ML configuration
ML_TARGET_REGRESSION = "flooding_volume_m3"
ML_TARGET_CLASSIFICATION = "flooded"
ML_TEST_SIZE = 0.2
ML_RANDOM_STATE = 42
ML_CV_FOLDS = 5
ML_GROUP_COLUMN = "run_id"
ML_SPLIT_STRATEGY = "grouped_by_run_id"
ML_USE_PCA = True
# PCA can receive either an integer number of components or a float between
# 0 and 1 to target a cumulative explained-variance ratio.
ML_PCA_COMPONENTS = 5
ML_PCA_SVD_SOLVER = "full"

# Temporal/CNN planning configuration. These values are placeholders for the
# future 1D CNN workflow and do not affect the current tabular models.
ML_TEMPORAL_WINDOW_MIN = 20
ML_TEMPORAL_HORIZON_MIN = 5
ML_TEMPORAL_STEP_MIN = 1
ML_TEMPORAL_TARGET = "failure_within_horizon"

ML_DROP_COLUMNS = [
    "run_id",
    "node_id",
    "scenario_type",
    "spatial_pattern",
    "applied_inflow_lps",
    "upstream_diam_avg_m",
    "downstream_diam_avg_m",
    "flooded",
    "flooding_volume_m3",
    "flooding_duration_min",
    "max_depth_m",
    "max_depth_ratio",
    "time_to_peak_min",
    "depth_rate_m_per_min",
    "in_degree",
    "out_degree",
    "upstream_capacity_lps",
    "downstream_capacity_lps",
]

ML_MODEL_CONFIGS = {
    "ridge": {
        "alpha": 1.0,
    },
    "lasso": {
        "alpha": 0.001,
        "max_iter": 20000,
    },
    "svr_rbf": {
        "kernel": "rbf",
        "C": 10.0,
        "epsilon": 0.1,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "random_state": ML_RANDOM_STATE,
    },
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 5000,
        "random_state": ML_RANDOM_STATE,
    },
    "svc_rbf": {
        "kernel": "rbf",
        "C": 10.0,
        "gamma": "scale",
    },
    "xgboost_classifier": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "random_state": ML_RANDOM_STATE,
    },
}
