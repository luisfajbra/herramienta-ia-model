# Critical-Node CSV Hydrograph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate the highest-peak inflow hydrograph for every evaluated CSV scenario.

**Architecture:** Extend the existing hydrograph visualization module with a renderer for `HydrographScenario`. Reuse the established plot styling and call the renderer from the batch coordinator with the main output directory.

**Tech Stack:** Python, pandas-backed scenario loading, matplotlib, pytest.

---

### Task 1: Scenario Hydrograph Renderer

**Files:**
- Modify: `swmm_resilience/visualization/hydrograph.py`
- Test: `tests/test_hydrograph.py`

- [ ] Write a failing test with two node series and assert that the renderer selects the node with the largest single flow value.
- [ ] Capture matplotlib values and assert hours are converted to minutes, the title is English, and the node suffix is hidden.
- [ ] Run `python -m pytest tests/test_hydrograph.py -q` and confirm the new test fails because the renderer is missing.
- [ ] Implement `plot_scenario_hydrograph(scenario, output_path)` using the same line, fill, labels, grid, DPI, and layout as `plot_hydrograph`.
- [ ] Re-run `python -m pytest tests/test_hydrograph.py -q` and confirm it passes.

### Task 2: Batch Integration

**Files:**
- Modify: `swmm_resilience/validation/hydrograph_batch.py`
- Test: `tests/test_hydrograph_batch.py`

- [ ] Write a failing integration test asserting one renderer call per scenario with `out_dir / f"hydrograph_{scenario_id}.png"`.
- [ ] Run the focused test and confirm no call occurs.
- [ ] Import and call `plot_scenario_hydrograph` after loading each validated scenario.
- [ ] Run the hydrograph and batch test modules and confirm they pass.

### Task 3: Existing Outputs

**Files:**
- Read: `data/hidrogramas_prueba/*.csv`
- Create: `outputs/new_csv_comparison/hydrograph_*.png`

- [ ] Load each CSV through `load_scenario` using nodes derived from the base INP.
- [ ] Render one critical-node hydrograph per scenario directly in `outputs/new_csv_comparison`.
- [ ] Verify both PNG files exist and are non-empty.
- [ ] Run `python -m compileall swmm_resilience/visualization/hydrograph.py swmm_resilience/validation/hydrograph_batch.py`.
