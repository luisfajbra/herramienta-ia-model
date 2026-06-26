# Factor Comparison Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate two SWMM-versus-XGBoost plots for every multiplier factor.

**Architecture:** A focused analysis coordinator will merge stored SWMM volumes with predictions for each factor. Existing visualization conventions will render one node profile and one parity plot into a dedicated output folder.

**Tech Stack:** Python, pandas, matplotlib, pytest, existing joblib models.

---

### Task 1: Per-Factor Plot Renderer

**Files:**
- Modify: `swmm_resilience/visualization/model_comparison.py`
- Test: `tests/visualization/test_factor_comparison_plots.py`

- [ ] Write a failing test for the two output names and visible node labels.
- [ ] Verify the test fails because `plot_factor_comparison` is missing.
- [ ] Implement one blue/orange node-volume plot and one parity plot.
- [ ] Run the focused test and confirm it passes.

### Task 2: Factor Comparison Coordinator

**Files:**
- Create: `swmm_resilience/analysis/factor_comparison.py`
- Test: `tests/analysis/test_factor_comparison.py`

- [ ] Write a failing test that supplies two factors and expects four files.
- [ ] Implement dataset filtering, prediction merging, and renderer calls.
- [ ] Validate node sets and duplicate node IDs before plotting.
- [ ] Run the focused tests and confirm they pass.

### Task 3: CLI and Current Outputs

**Files:**
- Modify: `main.py`
- Create: `outputs/factor_comparisons/*.png`

- [ ] Add the `--factor-comparisons` command.
- [ ] Run the command with the configured dataset and models.
- [ ] Verify 50 non-empty PNG files exist.
- [ ] Compile modified modules and run relevant tests.
