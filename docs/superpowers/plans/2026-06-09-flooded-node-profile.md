# Flooded Node Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate an additional per-factor node-volume plot containing only nodes with SWMM flood volume greater than zero.

**Architecture:** Add a focused plotting function beside the existing factor comparison plots. Call it from the existing factor generator while preserving the two current output files.

**Tech Stack:** Python, pandas, NumPy, matplotlib, pytest

---

### Task 1: Filtered Plot

**Files:**
- Modify: `swmm_resilience/visualization/model_comparison.py`
- Test: `tests/visualization/test_factor_comparison_plots.py`

- [ ] Write and run a failing test with flooded and non-flooded SWMM rows.
- [ ] Implement `plot_flooded_swmm_node_profile`.
- [ ] Run the focused visualization test.

### Task 2: Generator Integration

**Files:**
- Modify: `swmm_resilience/analysis/factor_comparison.py`
- Test: `tests/analysis/test_factor_comparison.py`

- [ ] Update the generator test to expect the additional path.
- [ ] Invoke the filtered plot after the existing comparison plots.
- [ ] Run all factor-comparison tests.

### Task 3: Generate Artifacts

**Files:**
- Create: `outputs/metrics/factor_comparison/volume_by_node_flooded_swmm_factor_*.png`

- [ ] Run `python main.py --factor-comparison`.
- [ ] Verify old and new PNGs are present and non-empty.
- [ ] Inspect a representative plot.
