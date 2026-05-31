# Config Cleanup And Model Quality Gate - Design Spec

## Goal

Resolve the remaining local `swmm_resilience/config.py` edits after the safe
stabilization pass without hiding model-quality risk. The cleanup should keep
physically meaningful hydraulic inputs, avoid blessing arbitrary PCA changes,
and leave the repository ready for full verification.

## Context

After commit `6643352 Stabilize hydraulic prediction pipeline`, the only tracked
unstaged file is `swmm_resilience/config.py`. Its remaining edits are:

- `ML_PCA_COMPONENTS` changed from `5` to `7`.
- `in_degree`, `out_degree`, `upstream_capacity_lps`, and
  `downstream_capacity_lps` were removed from `ML_DROP_COLUMNS`, making them
  eligible tabular ML inputs.

The audit already identifies PCA as a model-quality risk for tabular hydraulic
models, especially tree models. Existing surrogate and unified-comparison specs
treat the topology/capacity columns as inference-available static features.

## Design

### Static Hydraulic Features

Keep the topology and capacity columns as candidate tabular inputs:

- `in_degree`
- `out_degree`
- `upstream_capacity_lps`
- `downstream_capacity_lps`

These values are derived from static network structure in the local dataset
pipeline, not from post-simulation SWMM outputs. They are physically meaningful
for hydraulic prediction because they describe local connectivity and conveyance
capacity. They should not be grouped with target/result columns such as
`peak_flooding_lps`, `max_depth_m`, or `flooding_duration_min`.

The implementation plan should add a narrow feature-selection test proving that
these columns remain selected when present, while result/leakage columns remain
dropped.

### PCA Handling

Do not treat `ML_PCA_COMPONENTS = 7` as validated in this cleanup step.

The cleanup should restore or leave PCA in a known baseline state unless a
separate model-quality experiment proves that the new value improves the
hydraulic objectives. The next model-quality pass should compare at least:

- raw features vs PCA features for XGBoost/tree models
- PCA component counts or explained-variance ratio for linear/SVR models
- regression metrics for `peak_flooding_lps`, not only classification metrics

Until that experiment exists, PCA changes should be documented as pending
validation rather than committed as a silent global default.

### Verification

After resolving the config state, run the full test suite if practical:

```bash
pytest -v
```

If the full suite is too slow or has unrelated failures, record the exact
failing command and failures in the audit or follow-up plan. The focused
stabilization suite should remain green.

## Non-Goals

- Do not delete files.
- Do not push to a remote.
- Do not tune model hyperparameters in this cleanup step.
- Do not retrain or replace production artifacts in this cleanup step.
- Do not decide PCA policy without an explicit metric comparison.

## Acceptance Criteria

- The remaining `config.py` diff is resolved intentionally.
- Static topology/capacity features are either committed with a test or explicitly
  documented as deferred.
- `ML_PCA_COMPONENTS = 7` is not committed as a validated model-quality change
  without an experiment.
- Verification results are recorded in the final implementation summary.
