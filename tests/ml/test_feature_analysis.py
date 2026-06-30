from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.trainer import FEATURE_COLS


@pytest.fixture
def analysis_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    data = {col: rng.uniform(0.1, 10.0, n) for col in FEATURE_COLS}
    data["inunda"] = np.array([1 if i < 100 else 0 for i in range(n)])
    data["vol_inundacion_m3"] = np.where(
        data["inunda"] == 1, rng.uniform(0.5, 50.0, n), 0.0
    )
    data["factor_mult"] = np.tile([0.5, 1.0, 1.5, 2.0, 2.5], 40)
    return pd.DataFrame(data)


@pytest.fixture
def analysis_config(tmp_path: Path):
    inp = tmp_path / "network.inp"
    inp.write_text("network", encoding="utf-8")
    return SimpleNamespace(
        network=SimpleNamespace(inp_path=inp),
        ml=SimpleNamespace(
            classifier=SimpleNamespace(
                algorithm="random_forest",
                n_estimators=3,
                max_depth=2,
                learning_rate=0.1,
                subsample=1.0,
                scale_pos_weight="auto",
            ),
            regressor=SimpleNamespace(
                algorithm="random_forest",
                n_estimators=3,
                max_depth=2,
                learning_rate=0.1,
                subsample=1.0,
            ),
            use_scaler=False,
        ),
    )


def test_plot_correlation_creates_files(analysis_df, tmp_path):
    from swmm_resilience.ml.feature_analysis import plot_correlation

    plot_correlation(analysis_df, tmp_path)

    assert (tmp_path / "correlation_pearson.png").exists()
    assert (tmp_path / "correlation_spearman.png").exists()
    assert (tmp_path / "feature_target_correlation.png").exists()


def test_run_ablation_returns_both_runs(analysis_df, analysis_config, tmp_path):
    from swmm_resilience.ml.feature_analysis import run_ablation

    result = run_ablation(analysis_df, analysis_config, tmp_path)

    assert "full" in result
    assert "reduced" in result
    for run_key in ("full", "reduced"):
        assert "classifier" in result[run_key]
        assert "regressor_oracle" in result[run_key]
        assert "f1" in result[run_key]["classifier"]
        assert "auc_roc" in result[run_key]["classifier"]
        assert "nse" in result[run_key]["regressor_oracle"]
        assert "r2" in result[run_key]["regressor_oracle"]


def test_run_ablation_writes_json(analysis_df, analysis_config, tmp_path):
    from swmm_resilience.ml.feature_analysis import run_ablation
    import json

    run_ablation(analysis_df, analysis_config, tmp_path)

    json_path = tmp_path / "ablation_results.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert "full" in data and "reduced" in data


def test_run_ablation_writes_chart(analysis_df, analysis_config, tmp_path):
    from swmm_resilience.ml.feature_analysis import run_ablation

    run_ablation(analysis_df, analysis_config, tmp_path)

    assert (tmp_path / "ablation_comparison.png").exists()
