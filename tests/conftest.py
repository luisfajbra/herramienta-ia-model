import matplotlib
matplotlib.use('Agg')

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from swmm_resilience.ml.trainer import FEATURE_COLS


@pytest.fixture
def tiny_config_factory(tmp_path: Path):
    def _factory(algorithm: str = "random_forest", use_scaler: bool = False):
        inp = tmp_path / "network.inp"
        inp.write_text("network", encoding="utf-8")
        return SimpleNamespace(
            network=SimpleNamespace(inp_path=inp),
            ml=SimpleNamespace(
                classifier=SimpleNamespace(
                    algorithm=algorithm,
                    n_estimators=5,
                    max_depth=2,
                    learning_rate=0.1,
                    subsample=1.0,
                    scale_pos_weight="auto",
                ),
                regressor=SimpleNamespace(
                    algorithm=algorithm,
                    n_estimators=5,
                    max_depth=2,
                    learning_rate=0.1,
                    subsample=1.0,
                ),
                use_scaler=use_scaler,
            ),
        )

    return _factory


@pytest.fixture
def trainer_training_df():
    rows = []
    for i in range(8):
        row = {col: float(i + 1) for col in FEATURE_COLS}
        row["inunda"] = 1 if i >= 4 else 0
        row["vol_inundacion_m3"] = float(i * 10) if i >= 4 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
