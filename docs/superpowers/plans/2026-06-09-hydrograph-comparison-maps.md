# Hydrograph Comparison Maps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add directly comparable SWMM and ML flood maps to hydrograph validation outputs.

**Architecture:** Build a focused adapter around the existing flood-map renderer, using the canonical comparison DataFrame and a shared volume maximum. Invoke it from batch validation and reuse it to generate maps from an existing summary CSV.

**Tech Stack:** Python, pandas, Matplotlib, pytest

---

### Task 1: Comparison Map Adapter

**Files:**
- Modify: `swmm_resilience/visualization/model_comparison.py`
- Modify: `tests/test_hydrograph_batch.py`

- [ ] Add a failing test that captures two flood-map calls, verifies SWMM/ML column mappings, shared `vmax`, English titles, and output paths in the root output directory.
- [ ] Run the focused test and verify the helper is missing.
- [ ] Implement `plot_scenario_flood_maps`.
- [ ] Run the focused test and verify it passes.

### Task 2: Batch Integration

**Files:**
- Modify: `swmm_resilience/validation/hydrograph_batch.py`
- Modify: `tests/test_hydrograph_batch.py`

- [ ] Add a failing test that verifies batch validation calls the map helper for every scenario.
- [ ] Integrate the helper after the canonical comparison DataFrame is built.
- [ ] Run all hydrograph validation tests.

### Task 3: Existing Results

**Files:**
- Generate: `outputs/new_csv_comparison/flood_map_swmm_*.png`
- Generate: `outputs/new_csv_comparison/flood_map_ml_*.png`

- [ ] Load `comparison_summary.csv`, group by scenario, and call the new helper.
- [ ] Verify four non-empty PNG files exist directly in `outputs/new_csv_comparison`.
