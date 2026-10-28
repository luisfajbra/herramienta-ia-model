# Task 4 review fixes

This report supplements `task-4-implementation-report.md` and corrects its
description of default candidate immutability/objective and nested integration.
Only synthetic fixtures were used. No real-network fitting, SWMM simulation,
full-data training, real promotion or push was performed.

## Findings addressed

1. **Legacy CLI compatibility.** `train_candidate` again writes one pair to
   `<output>/<family>/` for every family and permits retraining at that path.
   Existing CLI promotion can load MLP there. `train_seed_candidates` explicitly
   requests immutable storage, preserving the five distinct `mlp/seed-42` through
   `mlp/seed-46` directories and preflight rejection of an existing experiment.
   Tests exercise actual tiny XGBoost and MLP CLI training twice, compare repeated
   predictions, and explicitly promote only synthetic test artifacts.

2. **Honest selection provenance.** Configured-parameter training now defaults
   to JSON `objective: null` with `selection: configured_parameters`; it does not
   claim `macro_network_rmse` selected parameters. Callers with actual selection
   may supply its objective, trial evidence and inner protocols. Both direct and
   CLI metadata are covered; the configured project ranking objective itself
   remains unchanged.

3. **Nested final MLP ensemble.** Each outer MLP fold runs its inner study once
   on that outer train, then fits seeds `[42,43,44,45,46]` using that fold's own
   selected classifier/regressor parameters. Each member sees only outer train
   (regressors see its flooded subset with `log1p` targets). Outer predictions
   average flood probabilities, threshold at 0.5, and average conditional m³
   after per-member `expm1` and existing nonnegative clipping. Scoring routes dry
   ensemble classifications to zero. No seed is selected using outer test.
   `NestedNetworkFold.model_seeds` distinguishes member seeds from inner-study
   seeds. `NestedNetworkResult.seed_oof` and `metrics.seed_summary` expose all
   per-seed OOF/metrics plus mean/sample SD. Saving the immutable nested result
   writes each seed OOF separately and records seeds and ensemble rule.
   XGBoost still executes its original single fit/predict branch.

4. **OOF alignment.** Existing `evaluate_candidate` calls `_validate_oof` through
   `score_predictions`, establishing unique complete key sets; the explicit
   multiseed API then sorts every seed by `(protocol, fold_id, sample_idx)` before
   averaging. A regression supplies differently shuffled, already validated OOF
   frames and checks identical ensemble/member output. No redundant production
   alignment check was added.

## TDD evidence

- Before fixes: CLI/direct regression batch had 3 expected failures (family-root
  MLP metadata missing and falsely defaulted objective); an evidence-preservation
  test already passed because that optional interface existed.
- Before nested integration: the new real-pipeline regression failed because
  only 2 classifier/regressor pairs were fit instead of the required 10 across
  two folds.
- First green batch: **92 passed in 57.32s**, covering CLI, direct candidate
  training, MLP helpers and nested evaluation.
- The nested regression spies on real pipeline fits, verifies all training row
  indices/targets, separately tuned fold parameters and every model seed, then
  reconstructs the ensemble and macro-network RMSE from actual fitted members.

Final relevant validation:

```text
venv/Scripts/python.exe -m pytest tests/ml/bench/test_train.py
  tests/ml/bench/test_mlp_family.py tests/ml/bench/test_verify_promotion.py
  tests/ml/bench/test_multinetwork_preparation.py tests/ml/bench/test_preprocess.py
  tests/ml/bench/test_bench_cli.py tests/ml/bench/test_reports.py
  tests/ml/bench/test_nested_network_evaluation.py
  tests/ml/bench/test_parity_with_legacy_evaluator.py
  tests/ml/test_prediction_volume_output_schema.py -q
  --basetemp=.pytest-task4-review-final
166 passed, 14 warnings in 80.69s
```

Warnings are existing matplotlib/pyparsing deprecations. `git diff --check`
passed. The self-contained local fix commit uses the message
`fix: preserve candidate CLI contracts and ensemble nested MLP folds`.
No full-suite result is claimed.

## Self-review

No CLI code or configuration/ranking objective was changed. Existing single-pair
training is deliberately repeatable; explicit scientific seed experiments and
nested saved runs remain immutable. Each seed's conditional prediction retains
the existing physical clipping rule. Selection occurs once within each outer
train, before all five member fits; outer targets are only accessed by scoring.
The added per-seed reporting uses the existing score/report definitions and
does not rank seeds or replace ensemble metrics with a mean of seed metrics.
