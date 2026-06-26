# Factor Comparison CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable CLI mode that generates SWMM/XGBoost parity and per-node flood-volume plots for every factor in the configured dataset.

**Architecture:** Keep comparison assembly and plotting in the existing `swmm_resilience.analysis.factor_comparison` module. Add only a thin orchestration branch to `main.py`, writing results under `outputs/metrics/factor_comparison`.

**Tech Stack:** Python, argparse, pandas, matplotlib, pytest

---

### Task 1: CLI integration

**Files:**
- Modify: `main.py`
- Create: `tests/test_factor_comparison_cli.py`

- [ ] **Step 1: Write the failing test**

Add a CLI test that replaces `generate_factor_comparisons`, invokes `main()` with
`--factor-comparison`, and verifies the configured dataset, model directory, and
output directory.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_factor_comparison_cli.py`

Expected: FAIL because `--factor-comparison` is not registered.

- [ ] **Step 3: Write minimal implementation**

Import `generate_factor_comparisons`, register `--factor-comparison`, invoke the
generator with `config.dataset.output_path`, `MODELS_DIR`, and
`METRICS_DIR / "factor_comparison"`, then print the generated paths.

- [ ] **Step 4: Run focused tests**

Run:
`python -m pytest -q tests/test_factor_comparison_cli.py tests/analysis/test_factor_comparison.py tests/visualization/test_factor_comparison_plots.py`

Expected: all tests pass.

### Task 2: Generate and verify artifacts

**Files:**
- Create: `outputs/metrics/factor_comparison/*.png`

- [ ] **Step 1: Generate plots**

Run: `python main.py --factor-comparison`

Expected: two PNG files per factor in the dataset.

- [ ] **Step 2: Verify artifacts**

Confirm the output directory contains non-empty `volume_by_node_factor_*.png`
and `parity_factor_*.png` files in equal counts.
