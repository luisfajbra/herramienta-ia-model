from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype


class FeatureContractError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    units: str
    description: str
    nullable: bool = False


@dataclass(frozen=True)
class TargetDefinition:
    name: str
    task: str
    units: str
    description: str


FEATURE_DEFINITIONS_V17 = (
    FeatureDefinition("elev_fondo", "m", "Node invert elevation"),
    FeatureDefinition("prof_max", "m", "Node maximum available depth"),
    FeatureDefinition("n_tuberias_in", "count", "Incoming conduit count"),
    FeatureDefinition("n_tuberias_out", "count", "Outgoing conduit count"),
    FeatureDefinition("diam_max_in", "m", "Maximum incoming conduit diameter", True),
    FeatureDefinition("diam_max_out", "m", "Maximum outgoing conduit diameter", True),
    FeatureDefinition("pendiente_max_in", "m/m", "Maximum incoming conduit slope", True),
    FeatureDefinition("pendiente_out", "m/m", "Representative outgoing conduit slope", True),
    FeatureDefinition("base_inflow_lps", "L/s", "Persisted baseline node inflow"),
    FeatureDefinition("dist_outfall_m", "m", "Directed hydraulic distance to outfall", True),
    FeatureDefinition("n_nodos_aguas_arriba", "count", "Upstream node count"),
    FeatureDefinition("q_pico_acum_base", "L/s", "Accumulated baseline peak flow"),
    FeatureDefinition("upstream_capacity_lps", "L/s", "Aggregate upstream conveyance capacity", True),
    FeatureDefinition("q_pico_nodo", "L/s", "Scenario node peak inflow"),
    FeatureDefinition("q_pico_acum_escalado", "L/s", "Scenario-scaled accumulated peak flow"),
    FeatureDefinition("duracion_horas", "h", "Persisted scenario duration"),
    FeatureDefinition("tiempo_al_pico_h", "h", "Persisted scenario time to peak"),
)

TARGET_DEFINITIONS_V17 = (
    TargetDefinition(
        "inunda", "classification", "binary",
        "Whether the node floods during the run",
    ),
    TargetDefinition(
        "vol_inundacion_m3", "regression", "m3",
        "Total node flood volume during the run",
    ),
)

FEATURE_COLUMNS_V17 = tuple(item.name for item in FEATURE_DEFINITIONS_V17)
NULLABLE_FEATURE_COLUMNS_V17 = frozenset(
    item.name for item in FEATURE_DEFINITIONS_V17 if item.nullable
)


@dataclass(frozen=True)
class FeatureContract:
    contract_id: str
    definitions: tuple[FeatureDefinition, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.definitions)

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def descriptor_sha256(self) -> str:
        payload = json.dumps(
            {
                "contract_id": self.contract_id,
                "features": [asdict(item) for item in self.definitions],
                "targets": [asdict(item) for item in TARGET_DEFINITIONS_V17],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        actual = tuple(frame.columns)
        if actual != self.feature_names:
            raise FeatureContractError(
                f"Expected ordered features {self.feature_names}; received {actual}"
            )
        if frame.empty:
            raise FeatureContractError("Feature frame is empty")
        complex_columns = [
            name for name in self.feature_names if is_complex_dtype(frame[name])
        ]
        if complex_columns:
            raise FeatureContractError(
                f"complex-valued feature columns are not permitted: {complex_columns}"
            )
        bad_types = []
        for name in self.feature_names:
            series = frame[name]
            all_null_nullable = (
                name in NULLABLE_FEATURE_COLUMNS_V17 and series.isna().all()
            )
            if not all_null_nullable and (
                not is_numeric_dtype(series) or is_bool_dtype(series)
            ):
                bad_types.append(name)
        if bad_types:
            raise FeatureContractError(f"Non-numeric feature columns: {bad_types}")
        numeric = frame.astype(float)
        required = [name for name in self.feature_names if name not in NULLABLE_FEATURE_COLUMNS_V17]
        missing_required = [name for name in required if numeric[name].isna().any()]
        if missing_required:
            raise FeatureContractError(f"Null required feature values: {missing_required}")
        infinite = [name for name in self.feature_names if np.isinf(numeric[name].dropna()).any()]
        if infinite:
            raise FeatureContractError(f"Infinite feature values: {infinite}")
        return numeric


TABULAR_V3_17 = FeatureContract("tabular_v3_17", FEATURE_DEFINITIONS_V17)


def validate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return TABULAR_V3_17.validate_frame(frame)
