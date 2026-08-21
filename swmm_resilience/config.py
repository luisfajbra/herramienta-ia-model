"""
Project configuration.

Keep this file limited to paths and high-level parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml


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
TRAINING_DIR = DATA_DIR / "training"
DEFAULT_DATASET_REVIEW_DIR = TRAINING_DIR / "dataset_review"
DEFAULT_NETWORK_KEY = "chico_hydro-qx1"
DEFAULT_NETWORK_DIR = NETWORKS_DIR / DEFAULT_NETWORK_KEY
DEFAULT_RESULTS_DIR = DEFAULT_NETWORK_DIR / "results"

DEFAULT_INP_FILE = DEFAULT_NETWORK_DIR / "SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
LEGACY_INP_FILE = BASE_DIR / "SWMM - Chico (PVC) Prueba 1 - Steady.inp"

DEFAULT_DB_FILE = TRAINING_DIR / "swmm_resilience.db"
DEFAULT_OUTPUT_CSV = DEFAULT_RESULTS_DIR / "dataset_ml.csv"
DEFAULT_MODEL_ARTIFACTS_DIR = DEFAULT_RESULTS_DIR / "model_artifacts"
DEFAULT_TEMPORAL_ARTIFACTS_DIR = DEFAULT_RESULTS_DIR / "temporal" / "model_artifacts"
DEFAULT_SURROGATE_MAPS_DIR = DEFAULT_TEMPORAL_ARTIFACTS_DIR.parent / "maps"

DEFAULT_INFLOW_MULTIPLIERS = _decimal_range(1.0, 2.5, 0.5)

# Backward-compat constants required by swmm_api_io.py and legacy code
SCENARIO_MODE_TIMESERIES = "timeseries"
SCENARIO_MODE_STEADY = "steady"
SUPPORTED_SCENARIO_MODES = (
    SCENARIO_MODE_TIMESERIES,
    SCENARIO_MODE_STEADY,
)
SCENARIO_MODE_TO_TYPE = {
    SCENARIO_MODE_TIMESERIES: "embedded_inflow_multiplier_sweep",
    SCENARIO_MODE_STEADY: "steady_inflow_multiplier_sweep",
}
SUPPORTED_MODEL_ALGORITHMS = {"xgboost", "random_forest"}
SUPPORTED_EVALUATION_METHODS = {"LOSO", "GroupKFold5"}

# When True, flooding_duration_min is read from the SWMM .rpt file after each
# simulation.  peak_flooding_lps is always taken from the simulation loop.
# PySWMM node.statistics is the source for max_depth, time_to_peak, and links.
USE_SWMM_API_RPT_RESULTS = True
DEFAULT_TARGET_NODES = None
DEFAULT_SCENARIO_MODE = SCENARIO_MODE_STEADY
DEFAULT_SCENARIO_TYPE = SCENARIO_MODE_TO_TYPE[DEFAULT_SCENARIO_MODE]
DEFAULT_SPATIAL_PATTERN = "uniform"
STRICT_INPUT_VALIDATION = True
INPUT_VALIDATION_MIN_JUNCTIONS = 20
INPUT_VALIDATION_MAX_REASONABLE_JUNCTION_DEPTH_M = 10.0
INPUT_VALIDATION_MAX_SUSPICIOUS_JUNCTION_DEPTH_FRACTION = 0.25

# ML configuration
ML_TARGET_REGRESSION = "total_flood_volume_m3"
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

# Temporal/CNN planning configuration.
ML_TEMPORAL_RESAMPLE_MIN = 5
ML_TEMPORAL_WINDOW_MIN = 20
ML_TEMPORAL_HORIZON_MIN = 5
ML_TEMPORAL_STEP_MIN = 5
ML_TEMPORAL_TARGET = "failure_within_horizon"

ML_DROP_COLUMNS = [
    "run_id",
    "node_id",
    "scenario_type",
    "spatial_pattern",
    "delta_inflow_lps",
    "upstream_diam_avg_m",
    "downstream_diam_avg_m",
    "flooded",
    "peak_flooding_lps",
    "total_flood_volume_m3",
    "flooding_duration_min",
    "max_depth_m",
    "max_depth_ratio",
    "time_to_peak_min",
    "depth_rate_m_per_min",
    "max_total_outflow_lps",
    "time_to_peak_outflow_min",
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


# ── Structured config (origin/main, used by new CLI commands) ─────────────────

def _validate_algorithm(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_MODEL_ALGORITHMS:
        allowed = ", ".join(sorted(SUPPORTED_MODEL_ALGORITHMS))
        raise ValueError(f"Algoritmo de {label} no soportado: {value}. Opciones: {allowed}")
    return normalized


def _validate_methods(methods: list) -> list[str]:
    normalized = [str(method).strip() for method in methods]
    invalid = [method for method in normalized if method not in SUPPORTED_EVALUATION_METHODS]
    if invalid:
        allowed = ", ".join(sorted(SUPPORTED_EVALUATION_METHODS))
        raise ValueError(f"Metodo de evaluacion no soportado: {', '.join(invalid)}. Opciones: {allowed}")
    return normalized


@dataclass
class NetworkConfig:
    inp_path: Path
    name: str


@dataclass
class SimulationConfig:
    factor_min: float
    factor_max: float
    factor_step: float
    hydrograph_shapes_dir: Optional[Path] = None


@dataclass
class DatasetConfig:
    output_path: Path
    flood_threshold_m3: float


@dataclass
class ClassifierConfig:
    algorithm: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    scale_pos_weight: Union[str, float]


@dataclass
class RegressorConfig:
    algorithm: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float


@dataclass
class MLConfig:
    classifier: ClassifierConfig
    regressor: RegressorConfig
    use_scaler: bool


@dataclass
class EvaluationConfig:
    methods: list
    stratify_by_factor: bool


@dataclass
class VisualizationConfig:
    factors_to_plot: list
    colormap: str
    output_path: Path
    show_labels_top_n: int


@dataclass
class ValidationConfig:
    drain_down_hours: float = 6.0


@dataclass
class Config:
    network: NetworkConfig
    simulation: SimulationConfig
    dataset: DatasetConfig
    ml: MLConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def factors(self) -> list:
        """Return list of simulation factors from factor_min to factor_max (inclusive)."""
        values = []
        current = self.simulation.factor_min
        while current <= self.simulation.factor_max + 1e-9:
            values.append(round(current, 6))
            current = round(current + self.simulation.factor_step, 6)
        return values


def load_config(config_path: str = "config.yaml") -> Config:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    base_dir = config_path.resolve().parent

    net = raw["network"]
    sim = raw["simulation"]
    ds = raw["dataset"]
    ml = raw["ml"]
    ev = raw["evaluation"]
    viz = raw["visualization"]

    inp_path = base_dir / net["inp_path"]
    if not inp_path.exists():
        raise FileNotFoundError(f"El archivo .inp no existe: {inp_path}")
    if float(sim["factor_min"]) >= float(sim["factor_max"]):
        raise ValueError("factor_min debe ser menor que factor_max")
    if float(sim["factor_step"]) <= 0:
        raise ValueError("factor_step debe ser mayor que 0")

    clf = ml["classifier"]
    reg = ml["regressor"]
    return Config(
        network=NetworkConfig(inp_path=inp_path, name=net["name"]),
        simulation=SimulationConfig(
            factor_min=float(sim["factor_min"]),
            factor_max=float(sim["factor_max"]),
            factor_step=float(sim["factor_step"]),
            hydrograph_shapes_dir=(
                base_dir / sim["hydrograph_shapes_dir"]
                if sim.get("hydrograph_shapes_dir")
                else None
            ),
        ),
        dataset=DatasetConfig(
            output_path=base_dir / ds["output_path"],
            flood_threshold_m3=float(ds["flood_threshold_m3"]),
        ),
        ml=MLConfig(
            classifier=ClassifierConfig(
                algorithm=_validate_algorithm(clf["algorithm"], "clasificador"),
                n_estimators=int(clf["n_estimators"]),
                max_depth=int(clf["max_depth"]),
                learning_rate=float(clf["learning_rate"]),
                subsample=float(clf["subsample"]),
                scale_pos_weight=clf["scale_pos_weight"],
            ),
            regressor=RegressorConfig(
                algorithm=_validate_algorithm(reg["algorithm"], "regresor"),
                n_estimators=int(reg["n_estimators"]),
                max_depth=int(reg["max_depth"]),
                learning_rate=float(reg["learning_rate"]),
                subsample=float(reg["subsample"]),
            ),
            use_scaler=bool(ml["use_scaler"]),
        ),
        evaluation=EvaluationConfig(
            methods=_validate_methods(ev["methods"]),
            stratify_by_factor=bool(ev["stratify_by_factor"]),
        ),
        visualization=VisualizationConfig(
            factors_to_plot=[float(f) for f in viz["factors_to_plot"]],
            colormap=viz["colormap"],
            output_path=base_dir / viz["output_path"],
            show_labels_top_n=int(viz["show_labels_top_n"]),
        ),
        validation=ValidationConfig(
            drain_down_hours=float(
                (raw.get("validation") or {}).get("drain_down_hours", 6.0)
            ),
        ),
    )
