from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

# Backward-compat constants required by swmm_api_io.py
SCENARIO_MODE_TIMESERIES = "timeseries"
SCENARIO_MODE_STEADY = "steady"
SUPPORTED_MODEL_ALGORITHMS = {"xgboost", "random_forest"}
SUPPORTED_EVALUATION_METHODS = {"LOSO", "GroupKFold5"}


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
class Config:
    network: NetworkConfig
    simulation: SimulationConfig
    dataset: DatasetConfig
    ml: MLConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig

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
    )
