from pathlib import Path

import pytest

from swmm_resilience.config import load_config


def write_config(tmp_path: Path, inp_name: str = "network.inp", factor_step: float = 0.2):
    inp = tmp_path / inp_name
    inp.write_text("[TITLE]\n;; test\n", encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
network:
  inp_path: "{inp_name}"
  name: "Test Network"
simulation:
  factor_min: 0.2
  factor_max: 0.6
  factor_step: {factor_step}
dataset:
  output_path: "data/training/dataset_final.csv"
  flood_threshold_m3: 0.0
ml:
  classifier:
    algorithm: "xgboost"
    n_estimators: 10
    max_depth: 3
    learning_rate: 0.1
    subsample: 1.0
    scale_pos_weight: "auto"
  regressor:
    algorithm: "xgboost"
    n_estimators: 10
    max_depth: 3
    learning_rate: 0.1
    subsample: 1.0
  use_scaler: false
evaluation:
  methods: ["LOSO", "GroupKFold5"]
  stratify_by_factor: true
visualization:
  factors_to_plot: [0.2, 0.4]
  colormap: "RdYlBu_r"
  output_path: "outputs/maps/"
  show_labels_top_n: 5
""",
        encoding="utf-8",
    )
    return cfg


def test_load_config_resolves_paths_and_factors(tmp_path):
    cfg = load_config(write_config(tmp_path))

    assert cfg.network.inp_path == tmp_path / "network.inp"
    assert cfg.dataset.output_path == tmp_path / "data" / "training" / "dataset_final.csv"
    assert cfg.visualization.output_path == tmp_path / "outputs" / "maps"
    assert cfg.factors() == [0.2, 0.4, 0.6]


def test_load_config_rejects_missing_inp(tmp_path):
    cfg_path = write_config(tmp_path, inp_name="missing.inp")
    (tmp_path / "missing.inp").unlink()

    with pytest.raises(FileNotFoundError, match="archivo .inp no existe"):
        load_config(cfg_path)


def test_load_config_rejects_non_positive_step(tmp_path):
    cfg_path = write_config(tmp_path, factor_step=0)

    with pytest.raises(ValueError, match="factor_step debe ser mayor que 0"):
        load_config(cfg_path)
