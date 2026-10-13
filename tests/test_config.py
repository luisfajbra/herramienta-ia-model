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

    with pytest.raises(FileNotFoundError, match=".inp file does not exist"):
        load_config(cfg_path)


def test_load_config_rejects_non_positive_step(tmp_path):
    cfg_path = write_config(tmp_path, factor_step=0)

    with pytest.raises(ValueError, match="factor_step must be greater than 0"):
        load_config(cfg_path)


def test_load_config_rejects_unknown_algorithm(tmp_path):
    cfg_path = write_config(tmp_path)
    text = cfg_path.read_text(encoding="utf-8")
    cfg_path.write_text(text.replace('algorithm: "xgboost"', 'algorithm: "svm"', 1), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported classifier algorithm"):
        load_config(cfg_path)


def test_load_config_resolves_db_path(tmp_path):
    cfg_path = write_config(tmp_path)
    text = cfg_path.read_text(encoding="utf-8")
    cfg_path.write_text(
        text.replace(
            '  flood_threshold_m3: 0.0',
            '  db_path: "outputs/custom_v17.sqlite3"\n  flood_threshold_m3: 0.0',
            1,
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.dataset.db_path == tmp_path / "outputs" / "custom_v17.sqlite3"


def test_load_config_defaults_db_path_when_absent(tmp_path):
    cfg = load_config(write_config(tmp_path))

    assert cfg.dataset.db_path == tmp_path / "outputs" / "training_v17.sqlite3"


def test_validation_inflow_closure_defaults(tmp_path):
    """write_config() no incluye bloque validation: -- deben aplicar los defaults."""
    config = load_config(write_config(tmp_path))
    assert config.validation.inflow_closure_tolerance_lps == 0.01
    assert config.validation.csv_inflow_closure_mode == "strict"


def _write_config_with_validation_block(tmp_path: Path, validation_yaml: str) -> Path:
    """Como write_config(), pero añade un bloque validation: al final del YAML."""
    cfg = write_config(tmp_path)
    text = cfg.read_text(encoding="utf-8")
    cfg.write_text(text + f"\nvalidation:\n{validation_yaml}\n", encoding="utf-8")
    return cfg


def test_validation_inflow_closure_is_parsed_when_present(tmp_path):
    cfg = _write_config_with_validation_block(
        tmp_path,
        '  drain_down_hours: 6.0\n  inflow_closure_tolerance_lps: 0.05\n'
        '  csv_inflow_closure_mode: "close"',
    )
    config = load_config(cfg)
    assert config.validation.inflow_closure_tolerance_lps == 0.05
    assert config.validation.csv_inflow_closure_mode == "close"


def test_validation_rejects_an_unknown_csv_inflow_closure_mode(tmp_path):
    cfg = _write_config_with_validation_block(
        tmp_path, '  csv_inflow_closure_mode: "auto"',
    )
    with pytest.raises(ValueError, match="strict.*close|close.*strict"):
        load_config(cfg)


def test_validation_drainage_state_defaults(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.validation.drainage_inflow_tolerance_lps == 0.01
    assert config.validation.drainage_flooding_tolerance_lps == 0.01
    assert config.validation.drainage_storage_change_tolerance_m3 == 0.01
    assert config.validation.drainage_tail_window_hours == 1.0


def test_validation_drainage_state_is_parsed_when_present(tmp_path):
    cfg = _write_config_with_validation_block(
        tmp_path,
        '  drainage_inflow_tolerance_lps: 0.05\n'
        '  drainage_flooding_tolerance_lps: 0.05\n'
        '  drainage_storage_change_tolerance_m3: 0.1\n'
        '  drainage_tail_window_hours: 2.0',
    )
    config = load_config(cfg)
    assert config.validation.drainage_inflow_tolerance_lps == 0.05
    assert config.validation.drainage_flooding_tolerance_lps == 0.05
    assert config.validation.drainage_storage_change_tolerance_m3 == 0.1
    assert config.validation.drainage_tail_window_hours == 2.0


def test_validation_max_continuity_error_pct_default(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.validation.max_continuity_error_pct == 5.0


def test_validation_max_continuity_error_pct_is_parsed_when_present(tmp_path):
    cfg = _write_config_with_validation_block(
        tmp_path, '  max_continuity_error_pct: 3.0',
    )
    config = load_config(cfg)
    assert config.validation.max_continuity_error_pct == 3.0


def test_validation_drainage_extension_defaults(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.validation.drainage_extension_increment_hours == 2.0
    assert config.validation.drainage_extension_max_retries == 2


def test_validation_drainage_extension_is_parsed_when_present(tmp_path):
    cfg = _write_config_with_validation_block(
        tmp_path,
        '  drainage_extension_increment_hours: 1.0\n'
        '  drainage_extension_max_retries: 5',
    )
    config = load_config(cfg)
    assert config.validation.drainage_extension_increment_hours == 1.0
    assert config.validation.drainage_extension_max_retries == 5


def test_validation_drainage_extension_max_retries_rejects_negative(tmp_path):
    cfg = _write_config_with_validation_block(
        tmp_path, '  drainage_extension_max_retries: -1',
    )
    with pytest.raises(ValueError, match="drainage_extension_max_retries"):
        load_config(cfg)
