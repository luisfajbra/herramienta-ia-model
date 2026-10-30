# Figuras de transferencia a Chico Norte Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Generate and preserve auditable parity plots and three comparable SWMM/XGBoost/MLP flood maps from the already sealed Chico Norte transfer evaluation.

**Architecture:** A focused analysis script will authenticate the sealed registration, predictions, staging manifest, eligible-run RPTs, and target hash before drawing anything. It will reconstruct labels transiently from verified RPTs, select low/median/high eligible scenarios by the nearest positive-volume p10/p50/p90 quantiles, draw figures from the existing routed predictions and INP geometry, and write only PNGs plus an aggregate provenance manifest. No labels will be exported or inserted into SQLite.

**Tech Stack:** Python, pandas, NumPy, matplotlib, existing SWMM RPT label parser and INP geometry parser, pytest.

**Spec:** Approved in conversation on 2026-10-30; results and prior protocol are documented in `docs/superpowers/reports/2026-10-30-transferencia-verificada-chico-norte.md`.

## Global Constraints

- Use the 787 frozen QA-eligible Chico Norte runs and their 410,814 run-node predictions; leave the 188 excluded runs out.
- Keep the external score, candidate artifacts, model parameters, thresholds, and SQLite `node_results` unchanged.
- Verify file hashes and exact run/node coverage before parsing targets or rendering figures.
- Extract target volumes from their manifest-linked RPTs with the shared validated parser, verify the canonical label hash against the frozen score, then discard row-level labels after rendering.
- Define low/median/high scenarios as eligible runs with total SWMM flooding at or above 1 m³ nearest the p10/p50/p90 quantiles of positive eligible-run totals; break ties by `scenario_key` ascending and record the quantile, target value, selected run, and total.
- Use actual flooded class `SWMM volume >= 1 m³`, predicted flooded class from sealed `y_pred_clf`, and routed predicted volume from sealed `volume_routed_m3`.
- Save Spanish, publication-ready PNGs and an aggregate-only JSON manifest under `docs/figures_chico_norte_transferencia/`; do not write node-level target tables.
- Do not retrain, rerun SWMM, alter score JSONs, or modify SQLite.

---

### Task 1: Test the deterministic scenario selection and integrity gates

**Files:**
- Create: `tests/visualization/test_chico_norte_transfer_figures.py`
- Create later: `scripts/figures_part2/make_chico_norte_transfer_figures.py`

**Interfaces:**
- `select_representative_scenarios(totals: pandas.DataFrame) -> list[dict]` consumes eligible scenario keys and total SWMM volumes, returns p10/p50/p90 selections in fixed order.
- `verify_sha256(path: Path, expected: str, role: str) -> None` fails closed on mismatch.

- [x] **Step 1: Write failing tests** for quantile selection, excluding dry/ineligible rows, deterministic ties, and SHA mismatch rejection.
- [x] **Step 2: Run the tests and confirm failure** because the plotting module did not yet expose the helpers.
- [x] **Step 3: Implement the tested selection and hash helpers** in the focused plotting module.
- [x] **Step 4: Run the focused tests and confirm they pass.**

### Task 2: Reconstruct authenticated labels and validate the frozen transfer

**Files:**
- Modify: `scripts/figures_part2/make_chico_norte_transfer_figures.py`
- Modify: `tests/visualization/test_chico_norte_transfer_figures.py`

**Interfaces:**
- `load_verified_transfer(model_dirs: dict[str, Path], source_root: Path) -> tuple[pandas.DataFrame, pandas.DataFrame, dict]` returns the joined figure frame, per-run scenario totals, and aggregate provenance after validating both sealed candidates and all eligible source links.

- [x] **Step 1: Add failing tests** for exact prediction/input keys and cascade, run admission binding, registration schema, expected node inventories, label-hash metric reproduction, and SHA mismatch rejection.
- [x] **Step 2: Run those tests and confirm the unimplemented guards fail first.**
- [x] **Step 3: Implement fail-closed loading**: compare candidate-independent source seals across models; verify score and registration hashes, B INP hash, manifest SHA, all prediction/input hashes and seals, 975 run-status decisions, 787 eligible IDs, QA eligibility, and eligible RPT/report hashes; parse each eligible RPT with the shared validated flooding-volume parser and require all reported nodes to belong to the frozen prediction node inventory.
- [x] **Step 4: Join both candidates' predictions by `(run_id, node_id)` to transient labels and require exact one-to-one coverage; recompute the canonical frozen label hash and require it to equal both score reports and the score metrics.**
- [x] **Step 5: Assert the source manifest declares zero `node_results` rows and open staging SQLite read-only to confirm it remains empty.**
- [x] **Step 6: Run all focused tests and confirm they pass.**

### Task 3: Render parity plots and comparable flood maps

**Files:**
- Modify: `scripts/figures_part2/make_chico_norte_transfer_figures.py`
- Modify: `tests/visualization/test_chico_norte_transfer_figures.py`

- [x] **Step 1: Add failing tests** for positive eligible quantiles, dry/ineligible exclusion, deterministic tie handling, exact node coverage, and readable parity ticks.
- [x] **Step 2: Run tests red before implementing each helper and the readable tick rule.**
- [x] **Step 3: Implement one 2×2 parity figure** showing all eligible run-node pairs and the subset flooded by SWMM (`>=1 m³`) for XGBoost and MLP; use `log10(1 + volume)` axes, a 1:1 line, log-density bins, sample counts, and metrics from the sealed routed predictions.
- [x] **Step 4: Implement three maps** (SWMM, XGBoost, MLP panels) for the selected p10/p50/p90 cases using the frozen INP node coordinates and links, shared volume scale, a 1 m³ classification threshold, and distinct false-positive/false-negative symbols.
- [x] **Step 5: Write only PNG files and an aggregate JSON provenance manifest** with hashes, parser/software versions, selection rule/results, row counts, and scenario metrics; never persist row-level actual labels.
- [x] **Step 6: Run the end-to-end generator and inspect each PNG for legibility and comparable scales.**

### Task 4: Document the figures and verification

**Files:**
- Create: `docs/figures_chico_norte_transferencia/INDEX.md`
- Create: generated PNGs and `figure_manifest.json` in the same directory
- Modify: `docs/superpowers/reports/2026-10-30-transferencia-verificada-chico-norte.md`
- Modify: `.gitignore` to version only the approved figures in this folder despite the repository-wide PNG ignore rule.

- [x] **Step 1: Document the scenario rule, plotting definitions, interpretation, known limitations, and exact regeneration command.**
- [x] **Step 2: Append a dated figures section to the transfer report** linking every figure and explaining that visual selection does not change the external score or candidate.
- [x] **Step 3: Run `git diff --check`, focused tests, and the full figure command from a clean output directory.**
- [x] **Step 4: Confirm the two score JSON hashes are unchanged, the SQLite node-results count remains zero, and no row-level label file was created.**
- [x] **Step 5: Stage only the plan, script, tests, documentation, provenance JSON, and PNGs on `fix/test-suite-contracts-2026-10-27`.**
