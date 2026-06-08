# Flood Volume Curve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--flood-volume-curve` CLI command that plots total flooded volume per factor (SWMM vs ML), with linear and log scale subplots side by side in each PNG.

**Architecture:** Two new modules mirroring the resilience curve pattern — `swmm_resilience/analysis/flood_volume.py` for pure computation and `swmm_resilience/visualization/flood_volume_curve.py` for plotting. Each PNG has two subplots (linear left, log right). Same `_predict_fn` injection pattern as `compute_resilience_curve` for testability.

**Tech Stack:** Python 3.10+, pandas, matplotlib, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `swmm_resilience/analysis/flood_volume.py` | Create | `compute_flood_volume_curve(...)` — pure computation |
| `swmm_resilience/visualization/flood_volume_curve.py` | Create | `plot_flood_volume_curve(...)` — two PNGs with dual subplots |
| `tests/test_flood_volume.py` | Create | Unit tests for computation and plotting |
| `main.py` | Modify | Add `--flood-volume-curve` flag and handler |

---

## Task 1: Flood Volume Computation

**Files:**
- Create: `swmm_resilience/analysis/flood_volume.py`
- Create: `tests/test_flood_volume.py` (computation tests only)

- [ ] **Step 1: Write failing computation tests**

Create `tests/test_flood_volume.py`:

```python
import pandas as pd
import pytest

from swmm_resilience.analysis import flood_volume


def swmm_df():
    rows = []
    for factor in [1.0, 2.0]:
        for node_idx in range(4):
            flooded = factor == 2.0 and node_idx >= 2
            rows.append({
                "node_id": f"J{node_idx}",
                "factor_mult": factor,
                "inunda": 1 if flooded else 0,
                "vol_inundacion_m3": 10.0 if flooded else 0.0,
            })
    return pd.DataFrame(rows)


def test_compute_flood_volume_swmm_values():
    df = swmm_df()
    factors = [1.0, 2.0]

    result = flood_volume.compute_flood_volume_curve(
        df_swmm=df,
        factors=factors,
        config=None,
        models_dir=None,
        _predict_fn=None,
    )

    assert list(result["factor"]) == [1.0, 2.0]
    assert result.loc[result["factor"] == 1.0, "vol_total_swmm"].iloc[0] == pytest.approx(0.0)
    assert result.loc[result["factor"] == 2.0, "vol_total_swmm"].iloc[0] == pytest.approx(20.0)


def test_compute_flood_volume_ml_values():
    def fake_predict(factor, config, models_dir):
        return pd.DataFrame({
            "node_id": ["J0", "J1", "J2", "J3"],
            "inunda_pred": [0, 0, 1, 1] if factor == 2.0 else [0, 0, 0, 0],
            "vol_pred_m3": [0.0, 0.0, 15.0, 5.0] if factor == 2.0 else [0.0] * 4,
        })

    df = swmm_df()
    factors = [1.0, 2.0]

    result = flood_volume.compute_flood_volume_curve(
        df_swmm=df,
        factors=factors,
        config=None,
        models_dir=None,
        _predict_fn=fake_predict,
    )

    assert result.loc[result["factor"] == 1.0, "vol_total_ml"].iloc[0] == pytest.approx(0.0)
    assert result.loc[result["factor"] == 2.0, "vol_total_ml"].iloc[0] == pytest.approx(20.0)
```

- [ ] **Step 2: Run failing tests**

```bash
python -m pytest tests/test_flood_volume.py::test_compute_flood_volume_swmm_values tests/test_flood_volume.py::test_compute_flood_volume_ml_values -v
```

Expected: FAIL — `ModuleNotFoundError` because the module does not exist yet.

- [ ] **Step 3: Implement the computation module**

Create `swmm_resilience/analysis/flood_volume.py`:

```python
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from ..ml.predict import predict_network


def compute_flood_volume_curve(
    df_swmm: pd.DataFrame,
    factors: list[float],
    config,
    models_dir: Optional[Path],
    _predict_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """Compute total flooded volume per factor for SWMM data and ML predictions.

    vol_total = sum of vol_inundacion_m3 (or vol_pred_m3) across all nodes.

    ML predictions are computed if:
      - _predict_fn is explicitly provided (used for testing), OR
      - config and models_dir are both non-None (production path, uses predict_network)
    Pass _predict_fn=None with config=None for SWMM-only (vol_total_ml = NaN).
    """
    use_ml = _predict_fn is not None or (config is not None and models_dir is not None)
    actual_predict = _predict_fn if _predict_fn is not None else predict_network

    rows = []
    for factor in factors:
        df_f = df_swmm[abs(df_swmm["factor_mult"] - factor) < 1e-6]
        vol_swmm = float(df_f["vol_inundacion_m3"].sum())

        if use_ml:
            pred = actual_predict(factor, config, models_dir)
            vol_ml = float(pred["vol_pred_m3"].sum())
        else:
            vol_ml = float("nan")

        rows.append({"factor": factor, "vol_total_swmm": vol_swmm, "vol_total_ml": vol_ml})

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run computation tests**

```bash
python -m pytest tests/test_flood_volume.py::test_compute_flood_volume_swmm_values tests/test_flood_volume.py::test_compute_flood_volume_ml_values -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/analysis/flood_volume.py tests/test_flood_volume.py
git commit -m "feat: add flood volume computation module"
```

---

## Task 2: Flood Volume Plot

**Files:**
- Create: `swmm_resilience/visualization/flood_volume_curve.py`
- Modify: `tests/test_flood_volume.py` (append plot test)

- [ ] **Step 1: Write failing plot test**

Append to `tests/test_flood_volume.py`:

```python
from swmm_resilience.visualization import flood_volume_curve


def test_plot_flood_volume_curve_writes_two_pngs(tmp_path):
    df = pd.DataFrame({
        "factor": [1.0, 2.0, 3.0],
        "vol_total_swmm": [0.0, 20.0, 80.0],
        "vol_total_ml":   [0.0, 18.0, 75.0],
    })

    path_swmm, path_ml = flood_volume_curve.plot_flood_volume_curve(df, tmp_path)

    assert path_swmm == tmp_path / "flood_volume_swmm.png"
    assert path_ml   == tmp_path / "flood_volume_ml.png"
    assert path_swmm.exists() and path_swmm.stat().st_size > 0
    assert path_ml.exists()   and path_ml.stat().st_size > 0
```

- [ ] **Step 2: Run failing test**

```bash
python -m pytest tests/test_flood_volume.py::test_plot_flood_volume_curve_writes_two_pngs -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the plot module**

Create `swmm_resilience/visualization/flood_volume_curve.py`:

```python
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd


def _plot_dual(fig, ax_lin, ax_log, x, y, color, marker):
    for ax, scale in ((ax_lin, "linear"), (ax_log, "log")):
        ax.plot(x, y, color=color, marker=marker, linewidth=2)
        ax.set_xlabel("Factor multiplicador de caudal")
        ax.set_ylabel("Volumen total inundado (m³)")
        ax.set_yscale(scale)
        ax.set_title("Escala lineal" if scale == "linear" else "Escala logarítmica")
        ax.grid(True, alpha=0.3)


def plot_flood_volume_curve(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Generate two flood-volume PNGs — one for SWMM data, one for ML predictions.

    Each PNG contains two subplots: linear scale (left) and log scale (right).
    df must have columns: factor, vol_total_swmm, vol_total_ml.
    Returns (path_swmm, path_ml).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    path_swmm = output_dir / "flood_volume_swmm.png"
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5))
    _plot_dual(fig, ax_lin, ax_log, df["factor"], df["vol_total_swmm"], "#2176ae", "o")
    fig.suptitle("Volumen total de inundación — SWMM (real)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path_swmm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Curva de volumen SWMM guardada: {path_swmm}")

    path_ml = output_dir / "flood_volume_ml.png"
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5))
    _plot_dual(fig, ax_lin, ax_log, df["factor"], df["vol_total_ml"], "#e07b39", "s")
    fig.suptitle("Volumen total de inundación — Predicción ML", fontsize=13)
    fig.tight_layout()
    fig.savefig(path_ml, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Curva de volumen ML guardada: {path_ml}")

    return path_swmm, path_ml
```

- [ ] **Step 4: Run plot test**

```bash
python -m pytest tests/test_flood_volume.py::test_plot_flood_volume_curve_writes_two_pngs -v
```

Expected: PASS.

- [ ] **Step 5: Run all flood volume tests**

```bash
python -m pytest tests/test_flood_volume.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/visualization/flood_volume_curve.py tests/test_flood_volume.py
git commit -m "feat: add flood volume curve plot module"
```

---

## Task 3: CLI Integration

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add imports to main.py**

Read `main.py` first. Add these two imports after the existing resilience imports (after `from swmm_resilience.visualization.resilience_curve import plot_resilience_curve`):

```python
from swmm_resilience.analysis.flood_volume import compute_flood_volume_curve
from swmm_resilience.visualization.flood_volume_curve import plot_flood_volume_curve
```

- [ ] **Step 2: Add CLI argument**

In the parser block, after `--resilience-curve`:

```python
    parser.add_argument("--flood-volume-curve", action="store_true",
                        help="Graficar volumen total de inundación por factor (SWMM vs ML)")
```

- [ ] **Step 3: Add handler**

After the `--resilience-curve` handler block (after its `return`) and before `if args.only_maps:`, add:

```python
    # ── Modo: curva de volumen de inundación ─────────────────────────────────
    if args.flood_volume_curve:
        print("\nCalculando curva de volumen de inundación...")
        df = pd.read_csv(config.dataset.output_path)
        factors = sorted(df["factor_mult"].unique())
        result = compute_flood_volume_curve(df, factors, config, MODELS_DIR)
        print("\nVolumen total por factor:")
        print(result.to_string(index=False))
        plot_flood_volume_curve(result, METRICS_DIR)
        return
```

- [ ] **Step 4: Compile check**

```bash
python -m compileall main.py swmm_resilience/analysis/flood_volume.py swmm_resilience/visualization/flood_volume_curve.py
```

Expected: no errors.

- [ ] **Step 5: Smoke test**

```bash
python main.py --flood-volume-curve
```

Expected output includes:
```
Calculando curva de volumen de inundación...
Curva de volumen SWMM guardada: outputs/metrics/flood_volume_swmm.png
Curva de volumen ML guardada: outputs/metrics/flood_volume_ml.png
```

Verify both files:
```bash
python -c "
from pathlib import Path
assert Path('outputs/metrics/flood_volume_swmm.png').exists()
assert Path('outputs/metrics/flood_volume_ml.png').exists()
print('ok')
"
```

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests -v
```

Expected: 28 tests PASS (25 existing + 3 new in `test_flood_volume.py`).

- [ ] **Step 7: Commit**

```bash
git add main.py
git commit -m "feat: add --flood-volume-curve CLI command"
```
