# English Plot Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all generated chart text English, show descriptive feature names, and remove trailing node `I/O` markers from visible labels.

**Architecture:** Introduce a pure display-label module and call it only at rendering boundaries. Translate static Matplotlib strings and title-producing callers without changing model columns, node IDs, output paths, or data processing.

**Tech Stack:** Python, Matplotlib, pandas, pytest

---

### Task 1: Shared Display Labels

**Files:**
- Create: `swmm_resilience/visualization/labels.py`
- Create: `tests/visualization/test_labels.py`

- [ ] Write tests asserting `123I -> 123`, `123O -> 123`, unchanged nonmatching IDs, all current feature mappings, and readable unknown-feature fallback.
- [ ] Run `pytest tests/visualization/test_labels.py -q` and verify import failure.
- [ ] Implement `format_node_label()` and `feature_display_name()`.
- [ ] Run `pytest tests/visualization/test_labels.py -q` and verify all tests pass.

### Task 2: Feature Importance and Core Curves

**Files:**
- Modify: `swmm_resilience/ml/feature_importance.py`
- Modify: `swmm_resilience/visualization/hydrograph.py`
- Modify: `swmm_resilience/visualization/resilience_curve.py`
- Modify: `swmm_resilience/visualization/flood_volume_curve.py`
- Modify: `tests/test_hydrograph.py`
- Create: `tests/ml/test_feature_importance_labels.py`

- [ ] Write plot-text tests for English labels and descriptive feature names.
- [ ] Run the focused tests and verify Spanish/current internal labels cause failures.
- [ ] Translate plot strings and apply shared display labels.
- [ ] Re-run focused tests and verify they pass.

### Task 3: Maps and Comparison Charts

**Files:**
- Modify: `swmm_resilience/visualization/flood_map.py`
- Modify: `swmm_resilience/visualization/network_map.py`
- Modify: `swmm_resilience/visualization/model_comparison.py`
- Modify: `swmm_resilience/visualization/runner.py`
- Modify: `swmm_resilience/ml/temporal/predict.py`
- Modify: `swmm_resilience/desktop/app.py`
- Modify: `tests/visualization/test_flood_volume_map_contract.py`
- Create: `tests/visualization/test_plot_text_contract.py`

- [ ] Write tests for English map text and display-only node suffix removal.
- [ ] Run focused tests and verify failures on current Spanish labels/raw node IDs.
- [ ] Translate all chart strings and title-producing callers; apply node formatting to annotations and tick labels.
- [ ] Re-run focused tests and verify they pass.

### Task 4: Verification

**Files:**
- Verify all modified Python files and tests.

- [ ] Run `rg` over plotting calls to find remaining Spanish chart strings.
- [ ] Run `pytest -q`.
- [ ] Run `python -m compileall swmm_resilience`.
- [ ] Review `git diff` and confirm unrelated local changes remain untouched.
