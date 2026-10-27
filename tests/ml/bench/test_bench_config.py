import textwrap
from pathlib import Path

import pytest

from swmm_resilience.config import load_config

_BASE_YAML = """
network:
  inp_path: "network.inp"
  name: "Test"
simulation:
  factor_min: 1.0
  factor_max: 3.0
  factor_step: 1.0
dataset:
  output_path: "data/training/dataset_final.csv"
  db_path: "outputs/training_v17.sqlite3"
  flood_threshold_m3: 1.0
ml:
  classifier: {algorithm: "xgboost", n_estimators: 10, max_depth: 3, learning_rate: 0.1, subsample: 0.8, scale_pos_weight: "auto"}
  regressor: {algorithm: "xgboost", n_estimators: 10, max_depth: 3, learning_rate: 0.1, subsample: 0.8}
  use_scaler: false
evaluation:
  methods: ["LOSO"]
  stratify_by_factor: true
visualization:
  factors_to_plot: [1.0]
  colormap: "RdBu_r"
  output_path: "outputs/maps/"
  show_labels_top_n: 5
"""

_BENCH_YAML = """
bench:
  protocols: ["LeaveOneShapeOut", "LOSO", "GroupKFold5"]
  ranking:
    primary_metric: "end_to_end.rmse_vol_todos_nodos"
    primary_direction: "minimize"
    tie_breakers:
      - {metric: "classifier.f1", direction: "maximize"}
  promote: "none"
  families:
    xgboost:
      enabled: true
      classifier: {n_estimators: 200, max_depth: 6}
      regressor: {n_estimators: 200, max_depth: 6}
    svm:
      enabled: false
      classifier: {C: 10.0}
      regressor: {C: 10.0}
"""


def _write_config(tmp_path, extra: str = "") -> str:
    (tmp_path / "network.inp").write_text("[TITLE]\n", encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(_BASE_YAML) + textwrap.dedent(extra), encoding="utf-8")
    return str(path)


def test_config_without_a_bench_block_still_loads(tmp_path):
    """Compatibilidad: el bloque bench es opcional en esta fase."""
    config = load_config(_write_config(tmp_path))
    assert config.bench is None


def test_bench_block_is_parsed(tmp_path):
    config = load_config(_write_config(tmp_path, _BENCH_YAML))

    assert config.bench is not None
    assert config.bench.protocols == ["LeaveOneShapeOut", "LOSO", "GroupKFold5"]
    assert config.bench.promote == "none"


def test_bench_config_accepts_leave_one_network_out(tmp_path):
    yaml = _BENCH_YAML.replace(
        '"GroupKFold5"', '"GroupKFold5", "LeaveOneNetworkOut"'
    )

    config = load_config(_write_config(tmp_path, yaml))

    assert config.bench.protocols == [
        "LeaveOneShapeOut",
        "LOSO",
        "GroupKFold5",
        "LeaveOneNetworkOut",
    ]


def test_ranking_criterion_is_built_from_the_yaml(tmp_path):
    criterion = load_config(_write_config(tmp_path, _BENCH_YAML)).bench.ranking

    assert criterion.primary_metric == "end_to_end.rmse_vol_todos_nodos"
    assert criterion.primary_direction == "minimize"
    assert criterion.tie_breakers == (("classifier.f1", "maximize"),)


def test_only_enabled_families_are_returned(tmp_path):
    bench = load_config(_write_config(tmp_path, _BENCH_YAML)).bench
    assert bench.enabled_families() == ["xgboost"]


def test_family_hyperparameters_are_available_per_task(tmp_path):
    bench = load_config(_write_config(tmp_path, _BENCH_YAML)).bench
    family = bench.families["xgboost"]

    assert family.classifier["n_estimators"] == 200
    assert family.regressor["max_depth"] == 6


def test_an_unknown_family_name_is_rejected_at_load_time(tmp_path):
    bad = """
bench:
  protocols: ["LOSO"]
  ranking: {primary_metric: "classifier.f1", primary_direction: "maximize", tie_breakers: []}
  promote: "auto"
  families:
    red_neuronal_magica:
      enabled: true
      classifier: {}
      regressor: {}
"""
    with pytest.raises(ValueError, match="Unknown family"):
        load_config(_write_config(tmp_path, bad))


def test_an_unknown_protocol_is_rejected_at_load_time(tmp_path):
    bad = """
bench:
  protocols: ["NotAProtocol"]
  ranking: {primary_metric: "classifier.f1", primary_direction: "maximize", tie_breakers: []}
  promote: "auto"
  families:
    xgboost: {enabled: true, classifier: {}, regressor: {}}
"""
    with pytest.raises(ValueError, match="Unknown protocol"):
        load_config(_write_config(tmp_path, bad))


def test_promote_can_name_a_specific_family(tmp_path):
    named = _BENCH_YAML.replace('promote: "none"', 'promote: "xgboost"')
    assert load_config(_write_config(tmp_path, named)).bench.promote == "xgboost"


def test_bench_families_literal_matches_the_registry():
    """config.py validates family names against a hard-coded tuple instead of
    importing swmm_resilience.ml.bench.registry (which drags in torch and
    xgboost on every load_config() call, even for --predict/--simulate/the
    GUI, none of which touch the bench). This test is the guard against the
    two lists drifting apart; it is the one place allowed to import registry.
    """
    from swmm_resilience.config import BENCH_FAMILIES
    from swmm_resilience.ml.bench.registry import available_families

    assert BENCH_FAMILIES == available_families()


def test_loading_a_config_with_a_bench_block_does_not_import_torch_or_xgboost(tmp_path):
    """Regression guard for the import-cost bug: _parse_bench must not import
    swmm_resilience.ml.bench.registry (or anything that imports torch/xgboost
    transitively) just to validate family names."""
    import subprocess
    import sys

    config_path = _write_config(tmp_path, _BENCH_YAML)
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})\n"
        "from swmm_resilience.config import load_config\n"
        f"load_config({config_path!r})\n"
        "assert 'torch' not in sys.modules, 'torch got imported by load_config'\n"
        "assert 'xgboost' not in sys.modules, 'xgboost got imported by load_config'\n"
        "assert 'optuna' not in sys.modules, 'optuna got imported by load_config'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[3])
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_bench_tuning_defaults_when_the_block_is_absent(tmp_path):
    config = load_config(_write_config(tmp_path, _BENCH_YAML))
    assert config.bench.tuning.n_trials == 50
    assert config.bench.tuning.n_splits == 5


def test_bench_tuning_block_is_parsed_when_present(tmp_path):
    with_tuning = _BENCH_YAML.replace(
        'promote: "none"',
        'promote: "none"\n  tuning: {n_trials: 10, n_splits: 3}',
    )
    config = load_config(_write_config(tmp_path, with_tuning))
    assert config.bench.tuning.n_trials == 10
    assert config.bench.tuning.n_splits == 3


def test_bench_tuning_rejects_non_positive_n_trials(tmp_path):
    with_tuning = _BENCH_YAML.replace(
        'promote: "none"',
        'promote: "none"\n  tuning: {n_trials: 0, n_splits: 5}',
    )
    with pytest.raises(ValueError, match="n_trials"):
        load_config(_write_config(tmp_path, with_tuning))


def test_bench_tuning_rejects_non_positive_n_splits(tmp_path):
    with_tuning = _BENCH_YAML.replace(
        'promote: "none"',
        'promote: "none"\n  tuning: {n_trials: 50, n_splits: 0}',
    )
    with pytest.raises(ValueError, match="n_splits"):
        load_config(_write_config(tmp_path, with_tuning))
