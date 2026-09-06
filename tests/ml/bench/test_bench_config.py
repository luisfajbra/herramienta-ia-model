import textwrap

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
  protocols: ["LOSO", "GroupKFold5"]
  ranking:
    primary_metric: "end_to_end.rmse_vol_todos_nodos"
    primary_direction: "minimize"
    tie_breakers:
      - {metric: "classifier.f1", direction: "maximize"}
  promote: "auto"
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
    assert config.bench.protocols == ["LOSO", "GroupKFold5"]
    assert config.bench.promote == "auto"


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
    with pytest.raises(ValueError, match="Familia desconocida"):
        load_config(_write_config(tmp_path, bad))


def test_an_unknown_protocol_is_rejected_at_load_time(tmp_path):
    bad = """
bench:
  protocols: ["LeaveOneShapeOut"]
  ranking: {primary_metric: "classifier.f1", primary_direction: "maximize", tie_breakers: []}
  promote: "auto"
  families:
    xgboost: {enabled: true, classifier: {}, regressor: {}}
"""
    with pytest.raises(ValueError, match="Protocolo desconocido"):
        load_config(_write_config(tmp_path, bad))


def test_promote_can_name_a_specific_family(tmp_path):
    named = _BENCH_YAML.replace('promote: "auto"', 'promote: "xgboost"')
    assert load_config(_write_config(tmp_path, named)).bench.promote == "xgboost"
