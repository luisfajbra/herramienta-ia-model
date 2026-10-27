# Hydraulic Admission and Simulation Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure each network and SWMM result used for training has an explicit supported-domain decision, consistent physical inputs/units, valid labels, and auditable hydraulic quality.

**Architecture:** Add a read-only INP admission inventory, then repair the shared scenario-to-feature and simulation-to-label paths. Record a versioned QA decision for every run and make training consume only explicitly eligible runs. Keep policies such as boundary-condition support, initial state, flood threshold and continuity tolerance in experiment configuration; require an owner-approved value and rationale instead of inventing engineering values in code.

**Tech Stack:** Python, swmm_api, PySWMM/SWMM, pandas, SQLite, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-generalizacion-entre-redes-design.md`, section “Etapa 2”.

## Global Constraints

- Keep the confirmed input as flow hydrographs at nodes. Rainfall-runoff modeling is outside this plan.
- Inspect network B before admitting it or adding any B record to the training database. Its INP is not currently in this repository.
- Do not choose a new universal initial condition, flood threshold, continuity limit, drainage window or boundary-condition domain in implementation code.
- Never modify the source INP. Run generated scenarios in disposable experiment directories and preserve INP/RPT/OUT plus configuration hashes.
- A software QA pass does not certify real-network calibration. Report SWMM data quality separately from model error.

---

### Task 1: Create a read-only network admission inventory

**Files:** Create `swmm_resilience/validation/network_admission.py`, `scripts/audit_network_inp.py`, and `tests/validation/test_network_admission.py`; update CLI usage in README.

**Interfaces:** `inspect_network_inp(inp_path: Path) -> dict` returns a deterministic JSON-serializable profile including SHA-256, SWMM version/options, FLOW_UNITS, rain/subcatchment presence, node/link/section counts and types, parallel-link groups, outfalls and boundary types, storage/pumps/controls, offsets, barrels, time-series names/ranges, initial flows/depths and missing/unresolved references. It makes no writes beside the INP.

- [x] Add compact fixture INPs containing a circular-conduit network, two parallel links, a storage node, a pump and a fixed-stage outfall. Assert inventory counts and options; assert source file hash/mtime is unchanged.
- [x] Run `python -m pytest tests/validation/test_network_admission.py -q`; confirm the import fails because the inventory module does not exist.
- [x] Implement `inspect_network_inp` with the existing `swmm_resilience.simulation.swmm_api_io.load_inp` parser. Normalize enums/IDs to stable strings, sort all emitted records, and hash raw source bytes.
- [x] Make `scripts/audit_network_inp.py INPUT.inp --output PROFILE.json` call the same function. Require output path outside the INP source; emit JSON and a short text summary of facts, not a pass/fail domain decision.
- [x] Run `python -m pytest tests/validation/test_network_admission.py -q` and execute the command against the small test fixture. Verify a byte-for-byte source hash match and deterministic JSON across two invocations.
- [ ] Commit inventory, tests and README with `feat: add read-only SWMM network admission inventory`.

### Task 2: Make hydraulic-domain policy explicit and testable

**Files:** Create `swmm_resilience/validation/network_policy.py` and `tests/validation/test_network_policy.py`; modify the admission CLI/config parser only to accept an explicit policy file.

**Interfaces:** `validate_network_profile(profile: dict, policy: dict) -> list[dict]` returns an empty list only when every required policy decision is present and the profile matches; otherwise it returns stable issue codes with paths/values. An absent policy is an error, not a permissive default.

- [x] Define policy fields for flow/geometry units, supported node/link/section types, parallel links, outfall/boundary types, storage/pumps/controls, offsets, persistent inflows, missing geometry, and feature contract version. Write tests for one allowed fixture, one issue per unsupported fixture element, and missing policy values.
- [x] Run `python -m pytest tests/validation/test_network_policy.py -q`; verify missing `policy.json` and each unsupported element is reported before any simulation.
- [x] Implement deterministic policy comparison. Each issue includes `code`, object ID, observed property and permitted property. Do not encode the policy as a global default in Python or `config.yaml`.
- [x] Add `scripts/audit_network_inp.py INPUT.inp --policy POLICY.json --output PROFILE.json`; output both profile and admission issues. Use exit code 0 for supported, 2 for policy rejection, 3 for invalid/missing input.
- [x] Run the admission and policy tests. Review the resulting JSON schema against the accepted design and keep profile generation independent of a future B INP.
- [ ] Commit with `feat: require an explicit hydraulic admission policy`.

### Task 3: Unify effective inflow, units, and time support

**Files:** Modify `swmm_resilience/extraction/static_features.py`, `swmm_resilience/extraction/tributary_features.py`, `swmm_resilience/simulation/hydrograph_shapes.py`, `swmm_resilience/simulation/timeseries_scenario.py`, and `swmm_resilience/validation/hydrograph_batch.py`; test `tests/extraction/test_static_features.py`, `tests/extraction/test_tributary_features.py`, `tests/simulation/test_partial_timeseries_scaling.py`, and `tests/test_hydrograph_batch.py`.

**Interfaces:** Introduce one shared effective-flow helper used by input rewriting and feature construction. It returns finite `(time_h, q_lps)` samples plus the applied conversion/scaling/baseline/pattern metadata; it rejects unsupported constituents, units, missing option values and unsupported patterns.

- [ ] Add regression tests for an inflow series with value 10 L/s, Sfactor 2 and baseline 5 L/s, an absent FLOW_UNITS option, a non-LPS unit, a pattern, non-finite sample, a series beginning after `t=0`, and two nodes with different time series. Assert generated INP values and feature summaries represent the same effective flow.
- [ ] Run the focused tests and confirm the feature path currently reports raw peak 10 L/s where SWMM receives 25 L/s, and that missing FLOW_UNITS is not rejected by hydrograph validation.
- [ ] Check the project's pinned SWMM/PySWMM version and construct the expected inflow transform from a disposable minimal INP. Measure PySWMM node inflow over the event; do not infer Mfactor/Sfactor/baseline/pattern ordering from feature names alone.
- [ ] Implement the shared helper, convert or reject units explicitly, use zero outside a source series only where SWMM uses that convention, and reject incompatible grids before both scenario writing and feature computation.
- [ ] Run `python -m pytest tests/extraction/test_static_features.py tests/extraction/test_tributary_features.py tests/simulation/test_partial_timeseries_scaling.py tests/test_hydrograph_batch.py -q`. Add one PySWMM integration assertion comparing the shared helper's flow to observed node inflow with a stated tolerance tied to the engine/report step.
- [ ] Commit helper, callers and tests with `fix: align effective inflows in simulation and features`.

Progress note (2026-10-26): the shared `EffectiveInflow` evaluator is implemented in `simulation/effective_inflow.py`, and is used by static peak extraction, simulation telemetry and scenario writing. The absolute CSV scenario writer resets Sfactor/Base/pattern so the generated INP matches those values. A PySWMM fixture confirms `Base=5`, `Sfactor=2`, `TS=10` yields 25 L/s; the isolated change set passed its focused tests. The full suite remains unverified in this Windows sandbox because temporary-file ACL failures affect pytest and code paths that use `tempfile`.

### Task 4: Preserve all hydraulic links or reject unsupported geometry

**Files:** Modify `swmm_resilience/extraction/topology.py`, `swmm_resilience/extraction/static_features.py`, `swmm_resilience/extraction/tributary_features.py`; test `tests/extraction/test_topology.py`, `tests/extraction/test_static_features.py`, `tests/extraction/test_tributary_features.py`.

**Interfaces:** `build_network_graph` preserves each link ID, endpoints and type, including parallel conduits. Topology/tributary computation consumes edge IDs explicitly and either calculates supported geometry correctly or raises a domain error naming the unsupported INP object.

- [ ] Add a fixture with two A→B conduits of different capacities and length. Assert both IDs survive, incoming link count is 2, distance is deterministic, and aggregate proxy capacity uses both links. Add cases for non-circular section, multiple barrels, adverse slope, ELEVATION offsets, pump, orifice, weir and storage.
- [ ] Run the focused extraction tests and verify the parallel-link test fails under current `nx.DiGraph` behavior.
- [ ] Move to a keyed multigraph or an equivalent link-record representation. Update predecessor/edge iteration to read every link record. Do not call `G[pred][node]` as if there were one edge.
- [ ] For each tested geometry/link type, either implement and independently verify its physical contribution or make admission reject it. Do not calculate circular Manning capacity from a non-circular `height`, drop barrels, or erase slope sign without a documented physically valid treatment.
- [ ] Run the focused extraction tests and `tests/ml/test_feature_contract_v21.py`. Verify existing Chico feature outputs against a frozen test fixture within exact or explicitly justified floating-point tolerance.
- [ ] Commit with `fix: preserve parallel links and enforce topology support`.

Progress note (2026-10-26): extraction now uses a keyed `MultiDiGraph`; immediate incoming conduit capacity is summed across edge IDs. A regression fixture with two different parallel conduits passes. Broader link/geometry support remains policy-dependent and is not claimed complete.

### Task 5: Fail closed when flooding reports are missing or malformed

**Files:** Modify `swmm_resilience/extraction/labels.py` and callers in `swmm_resilience/simulation/runner.py` / `swmm_resilience/validation/hydrograph_batch.py`; test `tests/test_labels.py`, `tests/simulation/test_label_extraction.py`.

**Interfaces:** `read_flooding_volumes(rpt_path)` distinguishes a validated report (including zero-flood case) from missing, malformed, duplicate, non-finite or unit-ambiguous data. Failures raise a typed extraction error; they never become zero labels.

- [ ] Add RPT fixtures for valid no-flood summary, valid ponding summary, missing file, truncated/missing section, duplicate node, `NaN`, and an invalid last token. Assert absent nodes get zero only when the flooding-summary section itself parsed successfully and the configured SWMM report semantics permit it.
- [ ] Run `python -m pytest tests/test_labels.py tests/simulation/test_label_extraction.py -q`; confirm the corrupt-report cases currently return zero or accept the last column.
- [ ] Parse SWMM report headers and columns by name/validated layout rather than `parts[-1]`. If `swmm_api` fails, use a fallback with the same explicit report-layout validation. Validate nonnegative finite volumes and node IDs before mapping all junctions.
- [ ] Persist parser source, parser version, report hash and failure code in run provenance. Make batch/runner abort or mark the run ineligible on parser failure.
- [ ] Run focused label tests and one SWMM-generated `.rpt` fixture with the project's supported engine. Check node-level agreement between primary and fallback parser on a representative flooding case and a no-flood case.
- [ ] Commit with `fix: reject invalid SWMM flooding reports`.

### Task 6: Make run quality and drainage admission finite and versioned

**Files:** Modify `swmm_resilience/validation/drainage_state.py`, `swmm_resilience/validation/hydrograph_batch.py`, `swmm_resilience/simulation/batch.py`, `swmm_resilience/database/training_queries.py`, and quality schema/migrations if needed; test `tests/validation/test_drainage_state.py`, `tests/database/test_load_training_frame.py`, and batch tests.

**Interfaces:** Each run stores a `hydraulic_quality` manifest with policy version, continuity value, finite/full-coverage checks, drainage state, initial condition, and eligibility plus reason. Training selection requires eligible quality and finite continuity according to the experiment's explicit policy.

- [ ] Add unit tests for all-NaN and partially missing observables, too-short node/link series, insufficient time window, absent link output, finite drained values, finite not-drained values, NULL continuity and retry exhaustion. None of the unverifiable cases may return `drained` or enter principal training.
- [ ] Run the focused tests and verify the all-NaN reproducer currently yields `drained` and a NULL-continuity run is currently selected.
- [ ] Validate finiteness and expected object/time coverage before threshold comparisons. Return `unverified` with stable failed-check codes on missing data; retain `not_drained` for finite, adequately observed failures.
- [ ] Unify training and batch-validation admission so exhausted retries, base cases and continuity violations cannot silently become eligible. Require configured continuity policy and rationale; do not add a guessed numeric threshold.
- [ ] Add migration and tests for the run-quality manifest, including a round-trip and an immutability assertion for pinned runs. Preserve the historical value and mark old runs' quality as unknown until independently backfilled.
- [ ] Run `python -m pytest tests/validation/test_drainage_state.py tests/database/test_load_training_frame.py tests/simulation/test_batch.py tests/test_hydrograph_batch.py -q`.
- [ ] Commit with `fix: gate training on finite verified hydraulic runs`.

### Task 7: Record initial conditions and run a controlled hydraulic smoke study

**Files:** Modify run/simulation metadata in `swmm_resilience/simulation/batch.py`, `swmm_resilience/simulation/runner.py`, and database provenance; add `tests/simulation/test_initial_condition_provenance.py` and an experiment-only script under `scripts/experiments/`.

**Interfaces:** A run manifest records every conduit/node initialization value from the input hash and the named initial-state policy. The controlled experiment operates only on a copied INP under its output directory.

- [x] Add a test proving the manifest changes if a single `InitFlow` changes, records the exact initialization policy ID, and does not rewrite the source INP.
- [x] Run the test and confirm current provenance did not previously record the initial-flow variants as a run manifest.
- [x] Implement only the provenance/versioning change. Do not force initial flow to zero or select a dry-state policy in code.
- [ ] Prepare a reproducible SWMM experiment for at least a no-flood event, a long event and a short-pulse event. Use paired copies with the INP's current state and an owner-approved alternative state. Store `.inp`, `.rpt`, `.out`, SHA-256, settings and per-node volume/continuity/drainage differences together.
- [ ] Execute the controlled experiment only after the owner records which initial-state cases are scientifically relevant and what continuity/drainage policy to apply. Do not use its outputs in the canonical database.
- [ ] Commit code, tests and experiment recipe (not generated `.out` files) with `feat: record hydraulic initial-state provenance`.

## Plan 2 completion gate

The B INP profile and selected policy are reviewed before simulation. Every pilot run has a validated report and finite targets, explicit units and event support, recorded initial state, a versioned hydraulic-quality result, and a documented eligibility decision. Any unsupported geometry, boundary or initial condition either has a tested implementation or is rejected with an object-specific reason. No run is declared physically valid solely because the software suite passes.

### Additional gate: independent generalization across networks

The current benchmark protocols hold out factors or hydrograph shapes, not networks. A multi-network `network_ids` filter can create a pooled dataset but does not make a network independent. Before fitting or reporting the requested shared MLP/XGBoost models, implement an outer LeaveOneNetworkOut evaluation with all feature fitting, hyperparameter selection, classification thresholds and model choices nested inside each training-network fold. First report frozen A→B transfer; after using B labels, treat B as development data and reserve a third network for claims about future-network generalization. Do not use B's labels to select a model and then report B as an untouched test.

Progress note (2026-10-26): Task 5 has a fail-closed `FloodingReportError` with stable codes for missing/unreadable reports, missing/truncated summaries, invalid or duplicate rows, and non-finite/negative volumes. A zero-filled label frame is produced only after an explicit no-flood marker or a validated flooding table. The SWMM API mapper now selects total flooding volume separately from maximum ponded volume and converts supported report units explicitly. Parser/version and report-hash provenance are generated and attached to label frames; persisting this metadata into the run-quality table and an engine-generated flooded-report comparison remain open.

Progress note (2026-10-26): Task 6 classifier now marks NaN/None observables, partial node-variable coverage, short observed windows and invalid tolerances as `unverified`. Migration 009 stores immutable quality decisions; SQLite enforces matching continuity and required provenance structure, and the training query requires an eligible manifest. The historical database still has 1,000 COMPLETE runs with NULL continuity and remains unmodified. Propagation from simulation through database ingestion and the common eligibility decision remain open; the 5% threshold and configured drain tolerances are provisional.


Progress note (2026-10-26): the owner approved recalculating historical A from the current source INP and shape CSV files as a new reproducible version with hashes, preserving historical runs; the 5% routing-continuity limit and current drain-down/retry tolerances are approved provisionally and must be reported as provisional. The 1,000 historical run IDs are pinned and have no stored `scenario_inflows`; none were modified. The base INP in SQLite matches the repository source by SHA-256.

Progress note (2026-10-26): at the user's direction, all recalculation simulations are paused pending clarification of the negative `455C` hydrograph's origin. Software fixes that do not depend on network B continue using fixtures only. SQLite eligibility now independently checks required initial-condition and flood-report provenance and consistency with the quality columns, closing a direct-insert path that could otherwise bypass the Python quality writer. The database loader, training view, migration and quality-writer suites pass (81 tests); no network simulation or training run was performed for this verification.

Progress note (2026-10-26): Task 7's software-only initial-state provenance is now emitted for base and shape batch scenarios and the single-factor runner. The source INP is hashed byte-for-byte; parsed conduit initial flows and junction/storage/divider initial depths are sorted and hashed under policy ID `source_inp_initial_state_v1`. The manifest records values as found, including parsed defaults, and makes no claim that the state is hydraulically justified. Four fixture tests pass, including changed-InitFlow sensitivity and proof that the source INP is not modified. No SWMM experiment was run. The separate path from these run manifests into DB quality records remains open.

Progress note (2026-10-26): The normal extraction pipeline now applies the same versioned quality decision before dataset assembly. It excludes and records any scenario with incomplete/non-finite node coverage, continuity beyond the owner-approved provisional 5% limit, unverified drainage, missing initial-state provenance, or missing/invalid flood-report provenance; it stops before training when no scenario remains. `--persist-sql` now requires the corresponding quality report and stores the eligible decisions alongside the imported runs, so SQLite's training selector can use the same gate. Excluded scenarios stay in the JSON report and are not imported as training runs. Focused software tests pass; no network simulation, recalibration, or model training was run.

Decision update (2026-10-26): the user explicitly directed that the 455C hydrograph be set to zero. The 61 values in the repository copy and in the supplied Downloads INP were changed to `0.000`. Both now hash to `1c46db327638e023f86f7f0061774abe708f01b8df625f6b6c7f58af36a1ace1`; the source hash before the edit was `9f8792abecd86b90187b010acb98cc2147ad114fb6be199d5417f79cd2cc45e7`. This supersedes the earlier pause that required clarification of 455C's treatment. The choice is user-directed input handling, not an independent hydraulic validation. Initial conduit `InitFlow` provenance remains unresolved, and no simulation or training has been run.

Progress note (2026-10-26): following the user's instruction to compare initial states, a paired exploratory SWMM run was executed on the supplied 3-hour Chico Norte Qx1 event. Only the 522 conduit `InitFlow` values differed (source values versus all zero); all node initial depths and the 455C zero hydrograph were identical. Both cases met the provisional 5% continuity threshold (2.971% versus -0.066%) and had 0 m? flooding, but both were `not_drained` because the INP ends at 3 h while the provisional drain-down requirement is 6 h. Initial state affected 274/523 node peak depths and 300/522 link peak absolute flows; the largest time-series differences were 0.895 m and 3,103 L/s. Detailed hashes and object-level CSVs are retained under `outputs/experiments/initial_conditions_chico_norte_qx1_20261026/`; interpretation is documented in `docs/superpowers/reports/2026-10-26-comparacion-initflow-chico-norte-qx1.md`. This is diagnostic only, not admission or training. In this session, 6 effective-inflow unit tests and 20 tributary-feature tests passed; the 59-case focused suite could not complete because Pytest hit WinError 5 on Windows basetemp directories, so temporary-INP integration tests remain unverified here. Task 7 remains incomplete: there is only one event, no extended drain-down output, and no basis to select zero InitFlow as physically valid.

Progress note (2026-10-26): the owner-approved frozen synthetic batch for Chico Norte B ran all 39 shapes × 25 peak-preserving factors (975 scenarios) with source `InitFlow`, the source INP unchanged, and no SQLite writes or training. The explicit `linear_v1` endpoint closure added a zero-flow point for 75 scenarios (three shapes), affecting 39,030 scenario–node series. All reports parsed with 522-junction flood-label coverage; all scenarios drained. Under the provisional absolute 5% continuity limit, 859 passed software QA and 116 were excluded for continuity only. Flood volume was positive in 763 scenarios, reaching 537,492 m³ for `shape_quasi_uniform_block_f5.000`; these synthetic magnitudes and the unresolved physical validity of the source initial state require review. Results are retained under `outputs/experiments/chico_norte_exploratory_a_to_b_20261026/` and summarized in `docs/superpowers/reports/2026-10-26-lote-exploratorio-chico-norte-a-b.md`. This creates B labels but does not evaluate A-trained models on B, admit B, or authorize use of B labels for model/threshold selection. The B network-admission and independent-network evaluation gates remain open.


Progress note (2026-10-26): after the user selected peak-preserving amplitudes, four isolated synthetic SWMM pilots were run: a 0.5 h pulse and a 14 h shape, each with source `InitFlow` and all conduit `InitFlow=0`. Amplitudes were each node's maximum effective FLOW from the supplied 3 h INP; factor=1; node 455C remains zero. The 0.5 h/source case had 12.721% continuity and fails the provisional 5% limit. The other three were below 5%. All four drained only after 8?10 h of zero-flow tail (the 0.5 h cases needed the full two extensions), and all had 0 m3 flooding, so this pilot supplies no positive flood-label examples. Initial state remains materially influential (largest paired peak depth differences: 0.405 m short shape, 0.137 m long shape). The artifact set and hashes are in `outputs/experiments/synthetic_hydrograph_pilot_chico_norte_20261026/`; a readable report is `docs/superpowers/reports/2026-10-26-piloto-hidrogramas-sinteticos-chico-norte.md`. These are synthetic exploratory simulations only; not admitted or written to SQLite. Task 7 and the training-readiness gates remain incomplete.

Progress note (2026-10-26; status superseded by the 2026-10-27 note below): the owner-authorized A requalification was executed as a new experiment: 1,000/1,000 reports validated, 963 eligible under the provisional gate, 37 excluded for continuity only, and source INP hash unchanged. A-only dataset and quality records were rebuilt into `outputs/experiments/requalification_chico_hydro_qx1_20261026/training_a_only.sqlite3`; 154,080 rows were prepared as `758d3a0ad79e8506`, without touching canonical SQLite or CSV. Full details: `docs/superpowers/reports/2026-10-26-recalificacion-chico-hydro-qx1.md`. At that time, XGBoost's 50-trial A-only Optuna study was complete, while MLP and frozen B evaluation were pending. B labels were kept out of tuning and model/threshold selection. The result is an exploratory synthetic transfer test to one network; formal hydraulic admission and broader generalization remain open.

Progress note (2026-10-27): MLP's sequential A-only study completed all 50/50 trials at the approved 200 epochs; best exploratory five-fold GroupKFold RMSE is 117.0970 m³/node (trial 35). Both final candidates were fitted on the 963 software-eligible A scenarios, then evaluated once on frozen B without using B labels for tuning or selection. B admitted 859/975 scenarios under the provisional 5% continuity filter; 116 were excluded. On 448,398 node-scenario rows, MLP end-to-end RMSE is 773.58 m³/node and its total predicted flood volume is 67.65 million m³ versus 41.60 million m³ actual; XGBoost RMSE is 273.79 m³/node and total predicted volume is 24.21 million m³. XGBoost transfers better in this test, but still underpredicts total volume by about 41.8%; MLP overpredicts by about 62.6%. Both remain exploratory: B is not hydraulically admitted, its initial-flow provenance and the provisional continuity threshold remain unresolved, the synthetic hydrograph shapes are shared with A, and B features materially exceed/miss A feature support. This test does not establish readiness for unseen networks or storms. Full results: `docs/superpowers/reports/2026-10-26-recalificacion-chico-hydro-qx1.md` and `outputs/experiments/requalification_chico_hydro_qx1_20261026/frozen_b_evaluation/`.
