# Spatial Evaluation and Shared MLP/XGBoost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure frozen transfer from network A to B and evaluate separate single-network and shared A+B MLP/XGBoost candidates without network or event leakage.

**Architecture:** Add a network-level outer split with event/shape-level tuning confined to each outer training set. Persist row-level predictions and split provenance. Run two named experiments in immutable output paths: frozen A→B before using B labels, then A, B and A+B on explicitly reserved events of known networks. Keep these as research candidates; promotion remains a separate explicit review decision.

**Tech Stack:** Python, pandas, Optuna, scikit-learn splitters/metrics, XGBoost, MLP, Parquet, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-generalizacion-entre-redes-design.md`, section “Etapa 3”.

## Global Constraints

- Plan 1 and Plan 2 exit gates must be complete before any multired training or principal test run.
- Do not use network B labels to choose the frozen A model, features, flood threshold, tuning objective, event families or reporting thresholds in the A→B experiment.
- With two networks, report directional A→B and optionally B→A results without claiming a reliable population estimate for future networks.
- Every event family, scaling variant and node-level expansion stays in one split. Network and event generalization claims use separately named test protocols.
- Require a configured tuning objective and declared aggregation weights before trials start. Do not substitute a convenient fold average for pooled metrics after seeing results.
- Keep every run under a new directory in `outputs/bench/multinetwork/`; construct its ID from a UTC timestamp and the first 12 characters of the source `prep_id`. Never overwrite active artifacts or prior OOF results.

---

### Task 1: Add and validate a network-held-out outer protocol

**Files:** Modify `swmm_resilience/ml/bench/schemas.py`, `folds.py`; test `tests/ml/bench/test_schemas.py`, `tests/ml/bench/test_folds.py`.

**Interfaces:** Add protocol `LeaveOneNetworkOut` with group column `network_id`. Its fold rows use the current `FOLD_COLUMNS` schema; every fold tests one whole network and no `network_id` occurs in both train and test within that fold.

- [ ] Add tests with three networks, repeated node/scenario/event IDs in each, and assert each network is held out once, all of its samples are in test, and its identifiers never appear in that fold's train split. Add error tests for fewer than two networks and null `network_id`.

```python
folds = build_folds(keys, "LeaveOneNetworkOut")
for _, fold in folds.groupby("fold_id"):
    held_out = json.loads(fold["held_out_groups"].iloc[0])
    assert len(held_out) == 1
    train_ids = set(keys.iloc[fold.loc[fold.split == "train", "sample_idx"]].network_id)
    test_ids = set(keys.iloc[fold.loc[fold.split == "test", "sample_idx"]].network_id)
    assert train_ids.isdisjoint(test_ids)
    assert test_ids == set(held_out)
```
- [ ] Run `python -m pytest tests/ml/bench/test_schemas.py tests/ml/bench/test_folds.py -q`; confirm the protocol is rejected as unknown.
- [ ] Implement `LeaveOneNetworkOut` through `sklearn.model_selection.LeaveOneGroupOut`. Extend manifest protocol validation to enforce one held-out network per fold and exactly one test appearance per row.
- [ ] Preserve the existing protocol names/semantics and prepared-dataset fingerprint rules. Update `PROTOCOL_INTERPRETATION` to state this protocol tests network transfer and requires at least two networks.
- [ ] Run the focused tests and `tests/ml/bench/test_preprocess.py tests/ml/bench/test_evaluate_candidate.py -q`.
- [ ] Commit with `feat: add leave one network out evaluation folds`.

### Task 2: Add an inner tuning loop confined to each outer training set

**Files:** Modify `swmm_resilience/ml/bench/tune.py`, `evaluate.py`, and bench config validation/loader; create `tests/ml/bench/test_nested_network_evaluation.py`.

**Interfaces:** Add `evaluate_nested_networks(prepared, family, inner_protocols, objective, ...)` which builds outer network folds, creates an inner training-only `PreparedDataset` per outer fold, tunes only on those rows, fits the selected candidate on that outer train, and returns predictions/metrics for the untouched outer network plus full split/trial provenance.

- [ ] Add a tiny deterministic fixture with two networks, multiple event families and at least enough event/shape groups within each network for the configured inner splitter. Spy on `run_study` to prove it receives only outer-train rows; assert all outer-test rows appear once and none appear in Optuna's training fingerprint.

```python
result = evaluate_nested_networks(prepared, "xgboost", ("LeaveOneShapeOut",), objective="macro_network_rmse")
for fold in result.folds:
    assert set(fold.inner_run_ids) <= set(fold.outer_train_run_ids)
    assert set(fold.outer_test_run_ids).isdisjoint(fold.inner_run_ids)
assert result.oof.groupby("sample_idx").size().eq(1).all()
```
- [ ] Run `python -m pytest tests/ml/bench/test_nested_network_evaluation.py -q`; confirm it fails because no nested evaluator exists.
- [ ] Implement a fold-local preparation view that carries only the outer-train runs, builds inner event/shape folds, and creates a fresh preprocessing/model pipeline per inner split. Prevent any scaler/imputer fit on outer-test data.
- [ ] Persist outer fold ID, held-out network, train/test run IDs, inner prep IDs, Optuna study ID, selected params, score definition, seed and OOF row keys.
- [ ] Run the new test plus `tests/ml/bench/test_tune.py tests/ml/bench/test_evaluate_candidate.py tests/ml/bench/test_parallel_folds.py -q`.
- [ ] Commit with `feat: nest model tuning inside network-held-out folds`.

### Task 3: Make the tuning objective and network aggregation explicit

**Files:** Modify `swmm_resilience/config.py`, `swmm_resilience/ml/bench/tune.py`, `metrics.py`, `reports.py`; test `tests/ml/bench/test_bench_config.py`, `test_tune.py`, `test_metrics.py`, `test_reports.py`.

**Interfaces:** `validate_tuning_objective(protocols, objective)` in `swmm_resilience/config.py` requires `pooled_rmse` or `macro_network_rmse` when network-held-out evaluation is enabled. Configuration records the per-network aggregation rule. Each Optuna trial record persists the objective name, value, network IDs and fold scores. Metrics output contains both pooled and per-network values regardless of which one selects a candidate.

- [ ] Add tests that absent/unknown objective fails before Optuna starts, pooled RMSE is calculated on concatenated held-out rows, and macro-network RMSE is the unweighted mean of network RMSEs independent of network row counts.

```python
with pytest.raises(ValueError, match="tuning_objective"):
    validate_tuning_objective(("LeaveOneNetworkOut",), None)
assert macro_network_rmse({"A": [1.0, 1.0], "B": [2.0, 2.0]},
                          {"A": [0.0, 0.0], "B": [0.0, 0.0]}) == pytest.approx(1.5)
```

Define `macro_network_rmse(y_true_by_network: dict[str, np.ndarray], y_pred_by_network: dict[str, np.ndarray]) -> float` beside the existing pooled metric helpers in `swmm_resilience/ml/bench/metrics.py`.
- [ ] Run the focused tests; confirm current tuning starts without an explicit objective and optimizes mean fold RMSE.
- [ ] Implement objective dispatch and fail-fast config validation. Keep existing same-network benchmark configuration readable through an explicit compatibility setting, but require an objective for any `LeaveOneNetworkOut` run.
- [ ] Emit per-network/event sample counts, RMSE/MAE m³, volume bias, confusion counts, recall, PR-AUC where both classes permit it, and false negatives grouped by configured actual-volume bands. Store band edges in experiment config and include them in its identity.
- [ ] Run the focused tests plus `tests/ml/bench/test_parity_with_legacy_evaluator.py`; confirm old evaluator behavior remains labelled as legacy and new ranking uses the recorded objective.
- [ ] Commit with `feat: report explicit network weighted model objectives`.

### Task 4: Add paired artifact manifests and deterministic multi-seed MLP runs

**Files:** Modify `swmm_resilience/ml/bench/train.py`, `promote.py`, `models/mlp_family.py`, `reports.py`; test `tests/ml/bench/test_train.py`, `test_mlp_family.py`, `test_verify_promotion.py`.

**Interfaces:** An experiment manifest identifies both classifier and regressor, family, selected parameters, all random seeds, contract ID/hash, prep ID, source network/run/event hashes, outer/inner split protocol, objective, target threshold, quality policy version and library/SWMM versions.

- [ ] Add tests that two MLP seeds yield two distinct candidate directories/manifests and a report of mean/spread, and that mismatched classifier/regressor prep IDs or source-run hashes fail pair loading.

```python
assert len({candidate.metadata["random_seed"] for candidate in candidates}) == len(candidates)
assert len({candidate.classifier_path.parent for candidate in candidates}) == len(candidates)
with pytest.raises(ModelProvenanceError, match="prep_id"):
    load_candidate_pair(classifier_dir, regressor_dir)
```
- [ ] Run the focused tests; confirm current artifact metadata records one seed and does not encode a complete multired source manifest.
- [ ] Implement deterministic seed enumeration for MLP and preserve the best-selection rule in config. Save each seed separately; never overwrite an OOF directory with another seed. Keep XGBoost trial/seed metadata explicit too.
- [ ] Extend `verify_promotion`/candidate verification to compare both artifacts with one complete manifest and hash all source-manifest files. Do not automatically promote the best result.
- [ ] Run focused tests and verify manifest hash changes if one source network, event, split, seed or model file changes.
- [ ] Commit with `feat: persist paired multired model experiment provenance`.

### Task 5: Execute the frozen A→B transfer experiment

**Files:** Create a runbook `docs/experimentos/plantilla-transferencia-A-a-B.md` and an experiment command in `main.py` only if the existing bench CLI cannot express frozen-model inference; add `tests/ml/bench/test_frozen_transfer_experiment.py`.

**Interfaces:** The runner consumes Plan 1/2 experiment manifests, an A-only prepared artifact/model pair, and B input/runs. It emits a separate immutable report with B labels excluded from model selection and a record of whether each B run passed hydraulic admission.

- [ ] Add a test around a spy predictor and B labels: model training/tuning inputs must contain A IDs only; labels are joined only after `predict` has written immutable B predictions. Changing B labels must not change A candidate hash or selection.
- [ ] Run `python -m pytest tests/ml/bench/test_frozen_transfer_experiment.py -q`; confirm no current command records this sequence end to end.
- [ ] Specify the pre-registration fields in the runbook: A-only training run IDs, contract hash, selection objective, threshold, selected params, seeds, B run/event hashes, hydraulic policy, and report metrics. Lock these before simulation/prediction.
- [ ] Implement the smallest runner that loads the verified A model pair and contract, validates B domain/quality, predicts B once, and only then joins B targets for scoring. Failed domain or hydraulic QA yields a separate rejected/ineligible report, not a fallback score.
- [ ] Run focused tests on synthetic manifests. Then, when the owner supplies B and approves the hydraulic policy, create the experiment ID with `datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + source_prep_id[:12]`, and write the frozen predictions under `experiment_dir / "A_to_B_frozen"`. Seal its hashes.
- [ ] Commit code, test and runbook with `feat: record frozen network transfer experiment`.

### Task 6: Train and compare A, B and A+B on reserved events

**Files:** Extend the experiment runner and reports from Tasks 2–5; test `tests/ml/bench/test_multinetwork_event_holdout.py` and update the experiment runbook.

**Interfaces:** A run config lists included networks, reserved event-family IDs, all scaling variants assigned to each event, nested tune protocol, objective, seeds and output directory. The split builder rejects an event/family/hash appearing on both sides across any included network when the stated claim is simultaneous new-event transfer.

- [ ] Add a two-network fixture with the same physical event family represented on both networks and multiple factors. Assert the entire family and its variants stay in one split; assert a claimed new-event test fails if a sibling variant is in train.

```python
test_families = set(keys.loc[folds.split == "test", "event_family_id"])
train_families = set(keys.loc[folds.split == "train", "event_family_id"])
assert train_families.isdisjoint(test_families)
assert folds.loc[folds.split == "test"].groupby("sample_idx").size().eq(1).all()
```
- [ ] Run `python -m pytest tests/ml/bench/test_multinetwork_event_holdout.py -q`; confirm current shape-only split allows a cross-network event-family overlap.
- [ ] Implement separate named experiments: A-only, B-only and A+B. For known-network event evaluation, reserve event families before tuning and apply the same event assignment to both networks when testing simultaneous event transfer.
- [ ] Run nested tuning only within the permitted training side. Save OOF predictions keyed by network/run/event/node; assert each test key occurs once and no held-out event or test network entered preprocessing, Optuna or threshold selection.
- [ ] Produce side-by-side per-network and per-event results for XGBoost and MLP, with MLP seed spread and both pooled/macro-network aggregates. State clearly that A+B results are on known networks and cannot replace the sealed A→B result.
- [ ] Run all tests introduced by this plan plus `tests/ml/bench tests/database` in the selected project environment. Execute full-data tuning only after the synthetic integration tests pass and only into a new experiment directory.
- [ ] Commit with `feat: compare shared models on reserved network events`.

## Plan 3 completion gate

The sealed A→B report uses an A-only pair and records B hydraulic eligibility and model errors separately. The A/B/A+B study uses nested model selection, event-family holdout, complete manifests and predeclared aggregation. No code path calls these two-network results proof of performance on a third unseen network, and no candidate is promoted automatically.
