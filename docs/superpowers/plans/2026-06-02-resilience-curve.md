# Resilience Curve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--resilience-curve` CLI command that computes and plots network resilience (fraction of non-flooded nodes) per factor, comparing SWMM ground truth vs ML predictions on the same chart.

**Architecture:** Two new modules — `swmm_resilience/analysis/resilience.py` for pure computation (no matplotlib) and `swmm_resilience/visualization/resilience_curve.py` for plotting. The `swmm_resilience/analysis/` package is new and needs an `__init__.py`. `predict_network` from `swmm_resilience/ml/predict.py` is called once per factor to get ML predictions on-the-fly.

**Tech Stack:** Python 3.10+, pandas, matplotlib, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `swmm_resilience/analysis/__init__.py` | Create | Empty package marker |
| `swmm_resilience/analysis/resilience.py` | Create | `compute_resilience_curve(df_swmm, factors, config, models_dir) → DataFrame` |
| `swmm_resilience/visualization/resilience_curve.py` | Create | `plot_resilience_curve(df, output_path) → Path` |
| `tests/test_resilience.py` | Create | Unit tests for computation and plotting |
| `main.py` | Modify | Add `--resilience-curve` flag and handler |

---

## Task 1: Resilience Computation

**Files:**
- Create: `swmm_resilience/analysis/__init__.py`
- Create: `swmm_resilience/analysis/resilience.py`
- Create: `tests/test_resilience.py` (partial — computation tests only)

- [ ] **Step 1: Create the analysis package**

Create `swmm_resilience/analysis/__init__.py` as an empty file.

- [ ] **Step 2: Write failing computation tests**

Create `tests/test_resilience.py`:

```python
import pandas as pd
import pytest

from swmm_resilience.analysis import resilience


def swmm_df():
    rows = []
    for factor in [1.0, 2.0]:
        for node_idx in range(4):
            rows.append({
                "node_id": f"J{node_idx}",
                "factor_mult": factor,
                "inunda": 1 if (factor == 2.0 and node_idx >= 2) else 0,
                "vol_inundacion_m3": 10.0 if (factor == 2.0 and node_idx >= 2) else 0.0,
            })
    return pd.DataFrame(rows)


def test_compute_resilience_swmm_values():
    df = swmm_df()
    factors = [1.0, 2.0]

    result = resilience.compute_resilience_curve(
        df_swmm=df,
        factors=factors,
        config=None,
        models_dir=None,
        _predict_fn=None,
    )

    assert list(result["factor"]) == [1.0, 2.0]
    assert result.loc[result["factor"] == 1.0, "resilience_swmm"].iloc[0] == pytest.approx(1.0)
    assert result.loc[result["factor"] == 2.0, "resilience_swmm"].iloc[0] == pytest.approx(0.5)


def test_compute_resilience_ml_values(monkeypatch):
    import pandas as pd

    def fake_predict(factor, config, models_dir):
        return pd.DataFrame({
            "node_id": ["J0", "J1", "J2", "J3"],
            "inunda_pred": [0, 0, 1, 1] if factor == 2.0 else [0, 0, 0, 0],
            "vol_pred_m3": [0.0, 0.0, 5.0, 5.0] if factor == 2.0 else [0.0] * 4,
        })

    df = swmm_df()
    factors = [1.0, 2.0]

    result = resilience.compute_resilience_curve(
        df_swmm=df,
        factors=factors,
        config=None,
        models_dir=None,
        _predict_fn=fake_predict,
    )

    assert result.loc[result["factor"] == 1.0, "resilience_ml"].iloc[0] == pytest.approx(1.0)
    assert result.loc[result["factor"] == 2.0, "resilience_ml"].iloc[0] == pytest.approx(0.5)
```

- [ ] **Step 3: Run failing tests**

```bash
python -m pytest tests/test_resilience.py::test_compute_resilience_swmm_values tests/test_resilience.py::test_compute_resilience_ml_values -v
```

Expected: FAIL — `ModuleNotFoundError` because the module does not exist yet.

- [ ] **Step 4: Implement the computation module**

Create `swmm_resilience/analysis/resilience.py`:

```python
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from ..ml.predict import predict_network


def compute_resilience_curve(
    df_swmm: pd.DataFrame,
    factors: list[float],
    config,
    models_dir: Optional[Path],
    _predict_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """Compute resilience per factor for SWMM data and ML predictions.

    resilience = non-flooded nodes / total nodes (range 0–1).

    ML predictions are computed if:
      - _predict_fn is explicitly provided (used for testing), OR
      - config and models_dir are both non-None (production path, uses predict_network)
    Pass _predict_fn=None with config=None to compute SWMM-only (resilience_ml = NaN).
    """
    use_ml = _predict_fn is not None or (config is not None and models_dir is not None)
    actual_predict = _predict_fn if _predict_fn is not None else predict_network

    rows = []
    for factor in factors:
        df_f = df_swmm[abs(df_swmm["factor_mult"] - factor) < 1e-6]
        n_total = len(df_f)
        res_swmm = float((df_f["inunda"] == 0).sum()) / n_total if n_total > 0 else float("nan")

        if use_ml:
            pred = actual_predict(factor, config, models_dir)
            n_pred = len(pred)
            res_ml = float((pred["inunda_pred"] == 0).sum()) / n_pred if n_pred > 0 else float("nan")
        else:
            res_ml = float("nan")

        rows.append({"factor": factor, "resilience_swmm": res_swmm, "resilience_ml": res_ml})

    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run computation tests**

```bash
python -m pytest tests/test_resilience.py::test_compute_resilience_swmm_values tests/test_resilience.py::test_compute_resilience_ml_values -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/analysis/__init__.py swmm_resilience/analysis/resilience.py tests/test_resilience.py
git commit -m "feat: add resilience computation module"
```

---

## Task 2: Resilience Curve Plot

**Files:**
- Create: `swmm_resilience/visualization/resilience_curve.py`
- Modify: `tests/test_resilience.py` (add plot test)

- [ ] **Step 1: Write failing plot test**

Append to `tests/test_resilience.py`:

```python
from swmm_resilience.visualization import resilience_curve


def test_plot_resilience_curve_writes_png(tmp_path):
    df = pd.DataFrame({
        "factor": [1.0, 2.0],
        "resilience_swmm": [1.0, 0.5],
        "resilience_ml": [1.0, 0.6],
    })

    output = resilience_curve.plot_resilience_curve(df, tmp_path / "resilience.png")

    assert output == tmp_path / "resilience.png"
    assert (tmp_path / "resilience.png").exists()
    assert (tmp_path / "resilience.png").stat().st_size > 0
```

- [ ] **Step 2: Run failing test**

```bash
python -m pytest tests/test_resilience.py::test_plot_resilience_curve_writes_png -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the plot module**

Create `swmm_resilience/visualization/resilience_curve.py`:

```python
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd


def plot_resilience_curve(df: pd.DataFrame, output_path: Path) -> Path:
    """Plot resilience vs factor for SWMM data and ML predictions.

    df must have columns: factor, resilience_swmm, resilience_ml.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["factor"], df["resilience_swmm"],
        color="#2176ae", marker="o", linewidth=2, label="SWMM (real)",
    )
    ax.plot(
        df["factor"], df["resilience_ml"],
        color="#e07b39", marker="s", linewidth=2, label="Predicción ML",
    )

    ax.set_xlabel("Factor multiplicador de caudal")
    ax.set_ylabel("Resiliencia (fracción de nodos no inundados)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Curva de resiliencia de la red")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Curva de resiliencia guardada: {output_path}")
    return output_path
```

- [ ] **Step 4: Run plot test**

```bash
python -m pytest tests/test_resilience.py::test_plot_resilience_curve_writes_png -v
```

Expected: PASS.

- [ ] **Step 5: Run all resilience tests**

```bash
python -m pytest tests/test_resilience.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/visualization/resilience_curve.py tests/test_resilience.py
git commit -m "feat: add resilience curve plot module"
```

---

## Task 3: CLI Integration

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add imports to main.py**

Read `main.py` first. Add these two imports after the existing visualization imports (after `from swmm_resilience.visualization.network_map import generate_network_map`):

```python
from swmm_resilience.analysis.resilience import compute_resilience_curve
from swmm_resilience.visualization.resilience_curve import plot_resilience_curve
```

- [ ] **Step 2: Add CLI argument**

In the parser block, after `--network-map`:

```python
    parser.add_argument("--resilience-curve", action="store_true",
                        help="Calcular y graficar curva de resiliencia SWMM vs ML")
```

- [ ] **Step 3: Add handler**

After the `--network-map` handler block and before `if args.only_maps:`, add:

```python
    # ── Modo: curva de resiliencia ────────────────────────────────────────────
    if args.resilience_curve:
        print("\nCalculando curva de resiliencia...")
        df = pd.read_csv(config.dataset.output_path)
        factors = sorted(df["factor_mult"].unique())
        result = compute_resilience_curve(df, factors, config, MODELS_DIR)
        print("\nResiliencia por factor:")
        print(result.to_string(index=False))
        out = METRICS_DIR / "resilience_curve.png"
        plot_resilience_curve(result, out)
        return
```

- [ ] **Step 4: Compile check**

```bash
python -m compileall main.py swmm_resilience/analysis/resilience.py swmm_resilience/visualization/resilience_curve.py
```

Expected: no errors.

- [ ] **Step 5: Smoke test**

```bash
python main.py --resilience-curve
```

Expected output includes:
```
Calculando curva de resiliencia...
Resiliencia por factor:
 factor  resilience_swmm  resilience_ml
    0.2              1.0            1.0
    ...
Curva de resiliencia guardada: outputs/metrics/resilience_curve.png
```

Verify:
```bash
python -c "from pathlib import Path; p = Path('outputs/metrics/resilience_curve.png'); assert p.exists() and p.stat().st_size > 0; print('ok')"
```

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "feat: add --resilience-curve CLI command"
```

---

## Task 4: Full Verification

**Files:** No changes expected.

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests -v
```

Expected: 25 tests PASS (22 existing + 3 new in `test_resilience.py`).

- [ ] **Step 2: Compile all modules**

```bash
python -m compileall main.py swmm_resilience
```

Expected: no errors.

- [ ] **Step 3: Commit if any fixes were needed**

If any fixes were required:

```bash
git add main.py swmm_resilience tests
git commit -m "fix: stabilize resilience curve feature"
```

If no fixes were needed, skip this commit.
