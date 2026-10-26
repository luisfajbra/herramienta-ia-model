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

- [ ] Add compact fixture INPs containing a circular-conduit network, two parallel links, a storage node, a pump and a fixed-stage outfall. Assert inventory counts and options; assert source file hash/mtime is unchanged.
- [ ] Run `python -m pytest tests/validation/test_network_admission.py -q`; confirm the import fails because the inventory module does not exist.
- [ ] Implement `inspect_network_inp` with the existing `swmm_resilience.simulation.swmm_api_io.load_inp` parser. Normalize enums/IDs to stable strings, sort all emitted records, and hash raw source bytes.
- [ ] Make `scripts/audit_network_inp.py INPUT.inp --output PROFILE.json` call the same function. Require output path outside the INP source; emit JSON and a short text summary of facts, not a pass/fail domain decision.
- [ ] Run `python -m pytest tests/validation/test_network_admission.py -q` and execute the command against the small test fixture. Verify a byte-for-byte source hash match and deterministic JSON across two invocations.
- [ ] Commit inventory, tests and README with `feat: add read-only SWMM network admission inventory`.

### Task 2: Make hydraulic-domain policy explicit and testable

**Files:** Create `swmm_resilience/validation/network_policy.py` and `tests/validation/test_network_policy.py`; modify the admission CLI/config parser only to accept an explicit policy file.

**Interfaces:** `validate_network_profile(profile: dict, policy: dict) -> list[dict]` returns an empty list only when every required policy decision is present and the profile matches; otherwise it returns stable issue codes with paths/values. An absent policy is an error, not a permissive default.

- [ ] Define policy fields for flow/geometry units, supported node/link/section types, parallel links, outfall/boundary types, storage/pumps/controls, offsets, persistent inflows, missing geometry, and feature contract version. Write tests for one allowed fixture, one issue per unsupported fixture element, and missing policy values.
- [ ] Run `python -m pytest tests/validation/test_network_policy.py -q`; verify missing `policy.json` and each unsupported element is reported before any simulation.
- [ ] Implement deterministic policy comparison. Each issue includes `code`, object ID, observed property and permitted property. Do not encode the policy as a global default in Python or `config.yaml`.
- [ ] Add `scripts/audit_network_inp.py INPUT.inp --policy POLICY.json --output PROFILE.json`; output both profile and admission issues. Use exit code 0 for supported, 2 for policy rejection, 3 for invalid/missing input.
- [ ] Run the admission and policy tests. Review the resulting JSON schema against the accepted design and keep profile generation independent of a future B INP.
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

### Task 4: Preserve all hydraulic links or reject unsupported geometry

**Files:** Modify `swmm_resilience/extraction/topology.py`, `swmm_resilience/extraction/static_features.py`, `swmm_resilience/extraction/tributary_features.py`; test `tests/extraction/test_topology.py`, `tests/extraction/test_static_features.py`, `tests/extraction/test_tributary_features.py`.

**Interfaces:** `build_network_graph` preserves each link ID, endpoints and type, including parallel conduits. Topology/tributary computation consumes edge IDs explicitly and either calculates supported geometry correctly or raises a domain error naming the unsupported INP object.

- [ ] Add a fixture with two A→B conduits of different capacities and length. Assert both IDs survive, incoming link count is 2, distance is deterministic, and aggregate proxy capacity uses both links. Add cases for non-circular section, multiple barrels, adverse slope, ELEVATION offsets, pump, orifice, weir and storage.
- [ ] Run the focused extraction tests and verify the parallel-link test fails under current `nx.DiGraph` behavior.
- [ ] Move to a keyed multigraph or an equivalent link-record representation. Update predecessor/edge iteration to read every link record. Do not call `G[pred][node]` as if there were one edge.
- [ ] For each tested geometry/link type, either implement and independently verify its physical contribution or make admission reject it. Do not calculate circular Manning capacity from a non-circular `height`, drop barrels, or erase slope sign without a documented physically valid treatment.
- [ ] Run the focused extraction tests and `tests/ml/test_feature_contract_v21.py`. Verify existing Chico feature outputs against a frozen test fixture within exact or explicitly justified floating-point tolerance.
- [ ] Commit with `fix: preserve parallel links and enforce topology support`.

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

- [ ] Add a test proving the manifest changes if a single `InitFlow` changes, records the exact initialization policy ID, and does not rewrite the source INP.
- [ ] Run the test and confirm current provenance does not distinguish the initial-flow variants shown in the audit.
- [ ] Implement only the provenance/versioning change. Do not force initial flow to zero or select a dry-state policy in code.
- [ ] Prepare a reproducible SWMM experiment for at least a no-flood event, a long event and a short-pulse event. Use paired copies with the INP's current state and an owner-approved alternative state. Store `.inp`, `.rpt`, `.out`, SHA-256, settings and per-node volume/continuity/drainage differences together.
- [ ] Execute the controlled experiment only after the owner records which initial-state cases are scientifically relevant and what continuity/drainage policy to apply. Do not use its outputs in the canonical database.
- [ ] Commit code, tests and experiment recipe (not generated `.out` files) with `feat: record hydraulic initial-state provenance`.

## Plan 2 completion gate

The B INP profile and selected policy are reviewed before simulation. Every pilot run has a validated report and finite targets, explicit units and event support, recorded initial state, a versioned hydraulic-quality result, and a documented eligibility decision. Any unsupported geometry, boundary or initial condition either has a tested implementation or is rejected with an object-specific reason. No run is declared physically valid solely because the software suite passes.
