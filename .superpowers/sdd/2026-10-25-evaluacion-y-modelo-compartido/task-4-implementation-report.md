# Task 4 implementation report

Task scope: synthetic fixtures only. No SWMM simulation, real-network fitting,
full-data training, automatic promotion, branch switch or push was performed.

## Implementation

- `train_seed_candidates(...)` fits the five fixed MLP seeds 42–46 into
  `mlp/seed-<seed>/`. Existing candidate directories raise `FileExistsError`;
  an initial preflight checks every target before starting the five fits.
- `train_mlp_ensemble(...)` returns the persisted final ensemble plus all five
  member pairs. The frozen classifier averages flood probabilities and uses
  the existing 0.5 threshold. Each regressor is inverse-transformed with
  `expm1`, clipped at the existing minimum of zero m³, and then averaged.
  The frozen regressor returns `log1p` of that conditional mean, preserving
  the existing inference interface. Dry classifications route zero volume.
- Every newly trained pair has an `experiment.json`, complete prepared source
  manifest, exact split rows, quality report, model hashes, selected parameters,
  full seed list, explicit objective, trial metadata, contract/prep identity,
  target threshold, Python/library versions (including torch), and per-run
  SWMM and hydraulic QA versions. Final full-fit scope is identified explicitly;
  its models are not represented as outer-fold models. The default objective
  remains `macro_network_rmse`; project ranking/configuration is unchanged.
- Referenced network INPs, hydrographs, dataset files and dataset manifests
  are checked against their source hashes and copied into content-addressed
  snapshot files in each candidate. Verification hashes every provenance
  document and source snapshot, including presence in the expected hash map.
  Original source paths need not remain available after snapshotting.
- Pair loading compares contract, prep identity, paired model hashes, source
  networks/runs/events, seed identity and provenance hashes before loading.
  Explicit promotion copies the full manifest and all its hashed sidecars;
  `verify_promotion` and inference pair loading check them. Existing older
  provenance records remain readable under their existing contract checks.
- `evaluate_mlp_seeds(...)` runs the same prepared folds with supplied fixed
  parameters for each of the five seeds, scores every seed, and scores the
  probability/conditional-volume ensemble separately. `summarize_seeds(...)`
  reports each seed and recursive scalar means/sample SD (`ddof=1`).
  `save_mlp_seed_evaluation(...)` writes an immutable experiment with separate
  seed OOF directories, ensemble OOF, metrics and OOF hashes.
- The design spec records owner-confirmed seed and ensemble decisions. No CLI
  behavior was changed. New entry points are explicit bench APIs; existing
  one-candidate and nested tuning entry points retain their roles.

## Scope beyond the plan's initial file list

`preprocess.py` now retrieves run/scenario configuration hashes,
`runs.swmm_version`, and the actual hydraulic QA policy/version/manifest from
the same readonly SQLite snapshot. Missing historical values are explicitly
unknown/null rather than inferred. `model_provenance.py` centralizes the added
document/pair checks so inference and promotion use the same rules.
`evaluate.py` supplies the explicit seeded OOF API. These additions are needed
to make the planned provenance and seed-spread behavior real; inference modules
and `main.py` were not modified.

## Verification and TDD

Baseline supplied for the original three focused files: 55 passed.

The initial new focused tests failed for missing APIs/source manifests (8
expected failures). A second red batch failed for missing persisted ensemble,
seeded OOF, source DB versions and promotion provenance (4 expected failures).
Further red tests exposed absent source snapshots/OOF persistence, missing
split hashing, the ensemble's nonnegative-volume policy, missing torch version
and a removable source snapshot hash. Each gap was implemented after its
failure was observed. An initial sandbox-only run hit Windows temporary-folder
permissions; the same synthetic tests ran successfully with reviewed execution
outside the sandbox.

Latest affected validation:

```text
venv/Scripts/python.exe -m pytest tests/ml/bench/test_train.py
  tests/ml/bench/test_mlp_family.py tests/ml/bench/test_verify_promotion.py
  tests/ml/bench/test_multinetwork_preparation.py tests/ml/bench/test_preprocess.py
  tests/ml/bench/test_bench_cli.py -q --basetemp=.pytest-task4-affected
128 passed, 14 warnings in 73.41s
```

Additional directly related checks:

```text
venv/Scripts/python.exe -m pytest tests/ml/bench/test_reports.py
  tests/ml/bench/test_nested_network_evaluation.py
  tests/ml/bench/test_parity_with_legacy_evaluator.py
  tests/ml/test_prediction_volume_output_schema.py -q
  --basetemp=.pytest-task4-pair-regression
34 passed, 13 warnings in 14.85s
```

Warnings are existing matplotlib/pyparsing deprecations. `git diff --check`
passed. These two affected runs total 162 passing tests.

## Self-review and practical limits

- Verified numerical reproducibility for repeated same-seed MLP predictions,
  distinct seed directories, immutable reruns, member/ensemble arithmetic,
  exact OOF row alignment, source snapshot portability/tamper detection,
  prep/source/seed pair mismatch rejection and copied promotion provenance.
- Experiment documents include separate hashes of source manifests, split
  rows and quality reports; split sensitivity is asserted independently of
  timestamps/model serialization. Model bytes themselves are covered by the
  existing and extended model-pair integrity checks.
- The APIs never select a seed on outer-test performance. The seeded OOF API
  evaluates caller-supplied fixed parameters; it does not perform nested
  hyperparameter selection or replace the existing nested tuner. Callers must
  preserve the established training-only parameter-selection boundary.
- Source copies are intentionally self-contained and may increase artifact
  storage for five seeds plus ensemble. Failed partial experiments are not
  silently resumed or overwritten; use a new output directory.
- No claim about real-network accuracy or full-suite completion is made.
