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

**Files:** Modify `swmm_resilience/ml/bench/schemas.py`, `folds.py`, `preprocess.py`, and `config.py`; test `tests/ml/bench/test_schemas.py`, `tests/ml/bench/test_folds.py`, and `tests/ml/bench/test_bench_config.py`.

**Interfaces:** Add protocol `LeaveOneNetworkOut` with group column `network_id`. Its fold rows use the current `FOLD_COLUMNS` schema; every fold tests one whole network and no `network_id` occurs in both train and test within that fold.

- [x] Add tests with three networks, repeated node/scenario/event IDs in each, and assert each network is held out once, all of its samples are in test, and its identifiers never appear in that fold's train split. Add error tests for fewer than two networks and null `network_id`.

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
- [x] Run `python -m pytest tests/ml/bench/test_schemas.py tests/ml/bench/test_folds.py -q`; confirm the protocol is rejected as unknown.
- [x] Implement `LeaveOneNetworkOut` through `sklearn.model_selection.LeaveOneGroupOut`. Extend manifest protocol validation to enforce one held-out network per fold and exactly one test appearance per row.
- [x] Preserve the existing protocol names/semantics and prepared-dataset fingerprint rules. Update `PROTOCOL_INTERPRETATION` to state this protocol tests network transfer and requires at least two networks.
- [x] Run the focused tests and `tests/ml/bench/test_preprocess.py tests/ml/bench/test_evaluate_candidate.py -q`.
- [x] Commit with `feat: add leave one network out evaluation folds`.
### Task 2: Add an inner tuning loop confined to each outer training set

**Files:** Modify `swmm_resilience/ml/bench/tune.py`, `evaluate.py`, and bench config validation/loader; create `tests/ml/bench/test_nested_network_evaluation.py`.

**Interfaces:** Add `evaluate_nested_networks(prepared, family, inner_protocols, objective, ...)` which builds outer network folds, creates an inner training-only `PreparedDataset` per outer fold, tunes only on those rows, fits the selected candidate on that outer train, and returns predictions/metrics for the untouched outer network plus full split/trial provenance.

- [x] Add a tiny deterministic fixture with two networks, repeated node/scenario identifiers, and enough shape groups within each network for the configured inner splitter. Spy on `run_study` to prove it receives only outer-train rows; assert all outer-test rows appear once and changes to outer-test targets do not alter Optuna's training fingerprint. Physical event-family reservation remains in Task 6.

```python
result = evaluate_nested_networks(prepared, "xgboost", ("LeaveOneShapeOut",), objective="macro_network_rmse")
for fold in result.folds:
    assert set(fold.inner_run_ids) <= set(fold.outer_train_run_ids)
    assert set(fold.outer_test_run_ids).isdisjoint(fold.inner_run_ids)
assert result.oof.groupby("sample_idx").size().eq(1).all()
```
- [x] Run `python -m pytest tests/ml/bench/test_nested_network_evaluation.py -q`; confirm it fails because no nested evaluator exists.
- [x] Implement a fold-local preparation view that carries only the outer-train runs, builds inner event/shape folds, and creates a fresh preprocessing/model pipeline per inner split. Prevent any scaler/imputer fit on outer-test data.
- [x] Persist outer fold ID, held-out network, train/test run IDs, inner prep IDs, Optuna study ID, selected params, score definition, seed and OOF row keys.
- [x] Run the new test plus `tests/ml/bench/test_tune.py tests/ml/bench/test_evaluate_candidate.py tests/ml/bench/test_parallel_folds.py -q`.
- [x] Commit with `feat: nest model tuning inside network-held-out folds`.

### Task 3: Make the tuning objective and network aggregation explicit

**Project decision (2026-10-27):** use `macro_network_rmse` as the configured selection objective. Each network contributes equally; retain `pooled_rmse` as a separately named descriptive metric.

**Interpretation for the current two-network study:** each outer fold trains on the other single network. Therefore its inner `macro_network_rmse` contains one network and numerically equals that network's RMSE; it does not estimate an average across training networks. The outer A-to-B and B-to-A scores remain separate transfer results. They are not evidence of expected performance over a population of unseen networks. Revisit this limitation when additional admitted networks are available.

**Files:** Modify `swmm_resilience/config.py`, `swmm_resilience/ml/bench/tune.py`, `metrics.py`, `reports.py`; test `tests/ml/bench/test_bench_config.py`, `test_tune.py`, `test_metrics.py`, `test_reports.py`.

**Interfaces:** `validate_tuning_objective(protocols, objective)` in `swmm_resilience/config.py` requires `pooled_rmse` or `macro_network_rmse` when network-held-out evaluation is enabled. Configuration records the per-network aggregation rule. Each Optuna trial record persists the objective name, value, network IDs and fold scores. Metrics output contains both pooled and per-network values regardless of which one selects a candidate.

- [x] Add tests that absent/unknown objective fails before Optuna starts, pooled RMSE is calculated on concatenated held-out rows, and macro-network RMSE is the unweighted mean of network RMSEs independent of network row counts.

```python
with pytest.raises(ValueError, match="tuning_objective"):
    validate_tuning_objective(("LeaveOneNetworkOut",), None)
assert macro_network_rmse({"A": [1.0, 1.0], "B": [2.0, 2.0]},
                          {"A": [0.0, 0.0], "B": [0.0, 0.0]}) == pytest.approx(1.5)
```

Define `macro_network_rmse(y_true_by_network: dict[str, np.ndarray], y_pred_by_network: dict[str, np.ndarray]) -> float` beside the existing pooled metric helpers in `swmm_resilience/ml/bench/metrics.py`.
- [x] Run the focused tests; confirm current tuning starts without an explicit objective and optimizes mean fold RMSE.
- [x] Implement objective dispatch and fail-fast config validation. Keep existing same-network benchmark configuration readable through an explicit compatibility setting, but require an objective for any `LeaveOneNetworkOut` run.
- [x] Connect the configured objective and tuning budget to the nested-evaluation entry point; reject absent objectives before trials start.
- [x] Record each trial's state, objective, inner-fold RMSE values and per-network RMSE values for later explanation and audit.
- [x] Emit per-network and per-network/shape sample counts, RMSE/MAE m³, volume bias, confusion counts, recall, PR-AUC where both classes permit it, and false negatives grouped by configured actual-volume bands. Require fixed band edges in nested network configuration and include their hash in the immutable experiment ID.
- [x] Run the focused tests plus `tests/ml/bench/test_parity_with_legacy_evaluator.py`; confirm old evaluator behavior remains labelled as legacy and new ranking uses the recorded objective.
- [x] Confirm the A-derived volume-band cut points (24, 108, 344 m³) and record them in the project config before launching a real nested evaluation. These are descriptive reporting strata derived from A-only eligible targets; for B→A they are used only after prediction, never for tuning or model selection.
- [x] Commit implementation with `feat: report nested network and event metrics` (`a207b61`); record the approved reporting bands in a follow-up decision commit.

### Task 4: Add paired artifact manifests and deterministic multi-seed MLP runs

**Files:** Modify `swmm_resilience/ml/bench/train.py`, `promote.py`, `models/mlp_family.py`, `reports.py`; test `tests/ml/bench/test_train.py`, `test_mlp_family.py`, `test_verify_promotion.py`.

**Interfaces:** An experiment manifest identifies both classifier and regressor, family, selected parameters, all random seeds, contract ID/hash, prep ID, source network/run/event hashes, outer/inner split protocol, objective, target threshold, quality policy version and library/SWMM versions.

- [x] Add tests that two MLP seeds yield two distinct candidate directories/manifests and a report of mean/spread, and that mismatched classifier/regressor prep IDs or source-run hashes fail pair loading.

```python
assert len({candidate.metadata["random_seed"] for candidate in candidates}) == len(candidates)
assert len({candidate.classifier_path.parent for candidate in candidates}) == len(candidates)
with pytest.raises(ModelProvenanceError, match="prep_id"):
    load_candidate_pair(classifier_dir, regressor_dir)
```
- [x] Run the focused tests; confirm current artifact metadata records one seed and does not encode a complete multired source manifest.
- [x] Implement deterministic seed enumeration for MLP and preserve the best-selection rule in config. Save each seed separately; never overwrite an OOF directory with another seed. Keep XGBoost trial/seed metadata explicit too.
- [x] Extend `verify_promotion`/candidate verification to compare both artifacts with one complete manifest and hash all source-manifest files. Do not automatically promote the best result.
- [x] Run focused tests and verify manifest hash changes if one source network, event, split, seed or model file changes.
- [x] Commit with `feat: persist paired multired model experiment provenance`.

### Task 5: Execute the frozen A→B transfer experiment

**Files:** Create a runbook `docs/experimentos/plantilla-transferencia-A-a-B.md` and an experiment command in `main.py` only if the existing bench CLI cannot express frozen-model inference; add `tests/ml/bench/test_frozen_transfer_experiment.py`.

**Interfaces:** The runner consumes Plan 1/2 experiment manifests, an A-only prepared artifact/model pair, and B input/runs. It emits a separate immutable report with B labels excluded from model selection and a record of whether each B run passed hydraulic admission.

- [ ] Add a test around a spy predictor and B labels: model training/tuning inputs must contain A IDs only; labels are joined only after `predict` has written immutable B predictions. Changing B labels must not change A candidate hash or selection.
- [x] Run `python -m pytest tests/ml/bench/test_frozen_transfer_experiment.py -q`; confirm no current command records this sequence end to end.
- [x] Specify the pre-registration fields in the runbook: A-only training run IDs, contract hash, selection objective, threshold, selected params, seeds, B run/event hashes, hydraulic policy, and report metrics. Lock these before simulation/prediction.
- [ ] Implement the smallest runner that loads the verified A model pair and contract, validates B domain/quality, predicts B once, and only then joins B targets for scoring. Failed domain or hydraulic QA yields a separate rejected/ineligible report, not a fallback score.
- [ ] Run focused tests on synthetic manifests. Then, when the owner supplies B and approves the hydraulic policy, create the experiment ID with `datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + source_prep_id[:12]`, and write the frozen predictions under `experiment_dir / "A_to_B_frozen"`. Seal its hashes.
- [ ] Commit code, test and runbook with `feat: record frozen network transfer experiment`.

**Synthetic implementation status (real scoring is blocked in code):** The API uses a real-format `study.json`, verifies its hash and A-only trial network scores, and seals predictions before labels. It excludes and reports ineligible runs. A synthetic actual-estimator test confirms changed B labels leave candidate and predictions unchanged; real-source bindings remain subject to independent review; do not mark real execution complete. The test uses trained synthetic estimators, not real B. Event files, feature/target sources and automatic study-to-preparation linkage require verification before real use. See `docs/experimentos/plantilla-transferencia-A-a-B.md`.

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

**Task 5 source-binding gate after review:** `score_transfer` raises unconditionally. Implement and test a verified label-free B input view whose keys/features and eligible node coverage bind to actual INP/event files, run QA and SWMM report; only after prediction seals, load labels from the same verified artifact and check their hashes/report linkage. `PreparedDataset.load` alone currently does not authenticate parquet contents or prove report linkage. Synthetic-only helper outputs must never be used to claim transfer results.

**Integrity fix, 2026-10-28:** `PreparedDataset.load` reconstructs the SQL-ordered training frame from `keys.parquet`, `features.parquet` and `targets.parquet` and checks its fingerprint against `manifest.json`. It also regenerates the fold assignment from the declared protocols and rejects altered `folds.parquet`. Tests first reproduced silent acceptance of changed sample/fold data; 80 focused schema/preparation tests and the full 512-test ML bench suite passed. These checks detect changed sample content and non-reproducible splits, but do not authenticate the artifact against SWMM reports or input files, and do not open the Task 5 scoring gate. See the source-binding gate above.

**Task 5 source-registration decision, 2026-10-28:** keep source files in their persistent experiment directories and link by resolved path, byte size and SHA-256; do not copy them. Create the B `run_id` in SQLite before prediction and QA, but do not write `node_results` until the prediction seal exists. Historical links may be populated only from the frozen protocol and exact `(network_id, shape_id, factor_mult)` identity. A missing or changed source fails closed. See the matching decision in the design spec.

#### Task 5a: Persist immutable source artifact links

**Files:** Add migration `swmm_resilience/database/sql/010_run_source_artifacts.sql` and identity hardening migration `011_run_source_artifact_identity.sql`; add file registration and verification helpers in `swmm_resilience/database/run_source_artifacts.py`; extend migration and database tests.

**Interface:** `register_run_source_artifacts(conn, run_id, artifacts)` accepts one complete set of linked paths for a completed run with immutable hydraulic QA. Required roles are the source-network INP, effective INP, event source, SWMM `.rpt` and `.out`; the source INP hash must match both `networks.network_sha256` and `sha256(networks.inp_bytes)`. `verify_run_source_artifacts(conn, run_id)` re-hashes every linked file, checks run identity snapshots and the `.rpt`/QA link, and recomputes a canonical manifest digest from run identity and all role/path/hash/size records. The API returns that canonical manifest and digest; callers cannot supply the digest. Rows are append-only and keyed by `(run_id, artifact_role)`. A run's `run_id`, `network_id`, `scenario_id`, `config_sha256` and status cannot change after links exist. Registration is atomic; mismatches fail without inserting rows.

- [x] Add failing SQLite tests for successful complete registration; missing role; changed file after registration; `.rpt` hash/QA mismatch; duplicate registration; and update/delete rejection.
- [x] Add migration 010 with role/path/hash/size/manifest-hash columns, strict checks, FK to `runs`, and immutable insert/update/delete guards.
- [x] Add migration 011 with source-network-INP role, run identity snapshots, run identity/config/status guard, and canonical digest verification.
- [x] Implement the helper to resolve and hash all five linked files, validate run and QA identity, verify source INP against network bytes/hash, compute canonical digest, and atomically insert the role set.
- [x] Run the amended migration/API tests plus existing migration 009 and catalog tests; confirm all negative cases fail closed.
- [x] Commit the review fix.

**Task 5a outcome:** the initial four-file foundation is extended by the review fix to bind five persistent file links to immutable run identity and a recomputed canonical source manifest. Older four-role rows are preserved in a legacy table and remain unverified under the new contract. The experiment source manifest is not yet consumed by `predict_transfer`; feature/target derivation and label-free B preparation remain outstanding, and `score_transfer` remains unconditionally blocked.

#### Task 5b: Register the historical A/B source manifests

**Files:** Add a source-manifest importer beside the database helper and tests using synthetic protocols; update the Task 5 runbook.

- [x] Add failing tests for exact mapping from `(network hash, shape_id, factor_mult)` to database `run_id`, exact frozen `expected_scenario_keys` (including explicit `base` keys), plus rejection of duplicate, missing, ambiguous, ineligible or hash-mismatched mappings.
- [x] Import only records present in the frozen `protocol.json`/`batch_progress.json`; verify the protocol seal, each referenced source file, `.rpt` and `.out`, report hash in QA, INP hash, event-series hash, and exact scenario key before writing artifact links.
- [x] Write an immutable experiment manifest with run IDs, relative artifact paths under the recorded experiment root, file hashes/sizes, protocol hash and link-table hash. Preserve original files in place.
- [x] Test against synthetic manifests and run read-only inventory of the stored A/B experiments; do not modify the canonical database or open real scoring.
- [x] Require a connection without an active transaction so the importer owns the complete SQLite/file transaction and cannot leave an orphan manifest after a caller rollback.
- [x] Commit with `feat: bind frozen simulation manifests to run ids`.

**Task 5b outcome (implementation ready for review):** the importer is fail-closed and supports explicit, frozen artifact links. Synthetic fixtures verify exact matching and rejection cases. The existing experiment manifests are not admissible yet: `protocol.json` has shape CSV hashes but omits their paths, and `batch_progress.json` stores RPT references only, without `scenario_key`, event series, effective INP, OUT, or their hashes. Their INP paths also resolve from repository root rather than experiment root. No historical rows/files were imported or changed; regenerate an explicit frozen manifest before Task 5c. The actual scoring barrier remains closed.

#### Task 5c: Build label-free B rows with preassigned SQLite run IDs

**Files:** Add a B preparation entry point in `swmm_resilience/ml/bench` or `database` after locating the narrowest existing loader; add tests in `tests/ml/bench/test_frozen_transfer_experiment.py` and database tests.

- [x] Add synthetic integration tests showing fresh B `run_id`, scenario, hydraulic QA, source artifact links and contract features are staged before prediction while `node_results` stays empty.
- [x] Add exclusive pre-simulation `freeze_b_transfer_protocol` plus separate external append-only SQLite receipt (chained hash, Git commit and execution context); require the receipt in batch and stage, including re-verification before inference. Link the source INP by absolute path/hash when it lives outside experiment output; keep the other four artifacts root-confined.
- [x] Allocate run IDs in the anchored protocol's declared stable order; require explicit scenario keys, protocol SHA, five file links and exact junction coverage in the effective INP. Preserve QA-ineligible runs with rejection codes but write no feature/key rows for them.
- [x] Derive continuity from RPT, validate the flooding-summary report with `read_flooding_volumes_with_manifest` without building a target vector, and derive drainage/OUT finite coverage from `read_tail_window_observables` + `classify_drainage_state`. Require exact complete node/link identity and static-feature parity (except dynamic `base_inflow_lps`, NaNs matched explicitly, `rtol=atol=1e-9`) between source/effective INP, plus exact finite OUT coverage for all node variables and every conduit, pump, orifice, weir and outlet flow series. Compare batch QA declarations against derived values; persist and predict from derived QA only.
- [x] Build v3.17/v4.21 label-free frames with the shared static, topology, scenario-dynamic and tributary feature functions. Reject target fields and reject frozen duration/peak times that disagree with the event series.
- [x] Reverify source manifest/protocol bytes, SQLite run identity, source registry links/hashes, QA and persisted features immediately before prediction. The wrapper reconstructs preregistration plans from SQLite, invokes `register_transfer`, then passes exact `KEY_COLUMNS`, features and reconstructed QA metadata to `predict_transfer`.
- [x] Keep B label parsing/import explicitly deferred to Task 5d; Task 5c has no target argument or `node_results` write path. `score_transfer` remains unconditionally blocked.
- [x] Run focused synthetic/source-binding regressions and commit only Task 5c as `feat: prepare verified label-free B transfer runs`.

**Task 5c scope note:** the previous 5b importer remains for historical runs with exact database mappings, and it does not consume the new protocol or enforce its external receipt. The reusable freeze/anchor API is now the required pre-run gate for B; a regeneration runner/adapter must add equivalent anchor verification before importing regenerated A. New B regeneration uses a two-document protocol: `protocol.json` is frozen before simulation with an external append-only receipt; `frozen_b_manifest.json` is completed afterward and must carry that receipt, explicit source paths/hashes and QA declarations that staging recalculates from RPT/OUT/INP. The external SQLite triggers/hash chain detect accidental/tampering edits but cannot authenticate against an operator who can replace the external DB. No current output databases were used or changed.

**Task 5c crash windows:** protocol and external receipt are separate writes; a crash can leave an unanchored protocol or an anchored protocol whose receipt was not returned. Neither is recoverable through the current API, so they cannot feed a valid batch. SQLite staging and the exclusive JSON output seal are also not one atomic transaction; a crash can leave an orphan seal or populated DB without the in-memory staged frame. Fresh-empty DB and exclusive-manifest checks make retries fail closed. Preserve the artifacts for audit and restart in a new experiment root/database; do not edit/recreate the anchor manually.

#### Task 5d: Open real scoring only after end-to-end source verification

- [ ] Add a failing test proving `score_transfer` rejects a missing run-source manifest, altered INP/event/RPT/OUT, altered features, or an incomplete source-to-label join.
- [ ] Enable `score_transfer` only for sealed predictions whose A candidate/preparation/study and B source manifests all verify; compute labels by parsing the verified `.rpt` after prediction seal, never from caller-supplied labels.
- [ ] Produce a real report only after the source links and hydraulic admission pass. Until then, preserve the unconditional scoring block and synthetic-only status.
- [ ] Run the complete relevant suite plus independent review. Commit with `feat: score only source-bound frozen transfer predictions`.
