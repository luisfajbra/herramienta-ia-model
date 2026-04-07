"""
Project configuration.

Keep this file limited to paths and high-level parameters.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
NETWORKS_DIR = DATA_DIR / "networks"
DEFAULT_NETWORK_KEY = "chico_steady"
DEFAULT_NETWORK_DIR = NETWORKS_DIR / DEFAULT_NETWORK_KEY
DEFAULT_RESULTS_DIR = DEFAULT_NETWORK_DIR / "results"

DEFAULT_INP_FILE = DEFAULT_NETWORK_DIR / "SWMM - Chico (PVC) Prueba 1 - Steady.inp"
LEGACY_INP_FILE = BASE_DIR / "SWMM - Chico (PVC) Prueba 1 - Steady.inp"

DEFAULT_DB_FILE = DEFAULT_RESULTS_DIR / "swmm_resilience.db"
DEFAULT_OUTPUT_CSV = DEFAULT_RESULTS_DIR / "dataset_ml.csv"

DEFAULT_DELTA_INFLOWS_M3PS = list(range(2, 102, 2))
DEFAULT_SCENARIO_TYPE = "uniform_inflow_sweep"
DEFAULT_SPATIAL_PATTERN = "uniform"

# ML configuration
ML_TARGET_REGRESSION = "flooding_volume_m3"
ML_TARGET_CLASSIFICATION = "flooded"
ML_TEST_SIZE = 0.2
ML_RANDOM_STATE = 42
ML_CV_FOLDS = 5

ML_DROP_COLUMNS = [
    "run_id",
    "node_id",
    "scenario_type",
    "spatial_pattern",
    "flooded",
    "flooding_volume_m3",
    "flooding_duration_min",
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
