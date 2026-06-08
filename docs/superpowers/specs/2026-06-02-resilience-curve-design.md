# Design: Resilience Curve

**Date:** 2026-06-02
**Feature:** `--resilience-curve` CLI command
**Status:** Design approved, pending implementation plan

---

## Purpose

Evaluate network resilience per rainfall scenario and plot how it degrades as the flow multiplier increases. Allows comparing ground-truth SWMM behavior against ML model predictions on the same chart.

---

## Definition

```
resilience(factor) = nodos no inundados / total nodos
```

Range: 0 (all nodes flooded) to 1 (no nodes flooded). Computed independently for SWMM data and ML predictions at each factor.

---

## Architecture

Two new modules with a single responsibility each:

| File | Responsibility |
|---|---|
| `swmm_resilience/analysis/resilience.py` | Pure computation — no matplotlib |
| `swmm_resilience/visualization/resilience_curve.py` | Plotting only — receives a DataFrame |
| `tests/test_resilience.py` | Unit tests for both |
| `main.py` | Add `--resilience-curve` flag and handler |

---

## Module 1: `swmm_resilience/analysis/resilience.py`

### Public function

```python
compute_resilience_curve(
    df_swmm: pd.DataFrame,
    factors: list[float],
    config,
    models_dir: Path,
) -> pd.DataFrame
```

### Returns

DataFrame with columns:

| Column | Type | Description |
|---|---|---|
| `factor` | float | Factor multiplicador |
| `resilience_swmm` | float | Fracción de nodos no inundados según SWMM |
| `resilience_ml` | float | Fracción de nodos no inundados según predicción ML |

### SWMM calculation (per factor)

```python
df_f = df_swmm[abs(df_swmm["factor_mult"] - factor) < 1e-6]
n_total = len(df_f)
n_ok = (df_f["inunda"] == 0).sum()
resilience_swmm = n_ok / n_total if n_total > 0 else float("nan")
```

### ML calculation (per factor)

Calls `predict_network(factor, config, models_dir)` — already available in `swmm_resilience/ml/predict.py`. Then:

```python
n_total = len(result)
n_ok = (result["inunda_pred"] == 0).sum()
resilience_ml = n_ok / n_total
```

---

## Module 2: `swmm_resilience/visualization/resilience_curve.py`

### Public function

```python
plot_resilience_curve(df: pd.DataFrame, output_path: Path) -> Path
```

### Visual spec

- X axis: Factor multiplicador de caudal
- Y axis: Resiliencia (0–1)
- Line 1: SWMM real — color `#2176ae` (azul), marker `o`
- Line 2: Predicción ML — color `#e07b39` (naranja), marker `s`
- Legend, grid, `fig.savefig / plt.close(fig)` pattern
- Output: `outputs/metrics/resilience_curve.png`

---

## CLI

```bash
python main.py --resilience-curve
```

Handler in `main.py` (standalone mode, before `--only-maps`):

1. Reads `config.dataset.output_path` (CSV existente)
2. Extracts sorted unique `factor_mult` values
3. Calls `compute_resilience_curve(df, factors, config, MODELS_DIR)`
4. Calls `plot_resilience_curve(result, output_path)`
5. Prints resilience values per factor to terminal

Requires: `dataset_final.csv` + trained models in `outputs/models/`.

---

## Tests (`tests/test_resilience.py`)

### Test 1: SWMM computation

Synthetic DataFrame with 4 nodes × 2 factors. Factor 1.0: 0 flooded → resilience 1.0. Factor 2.0: 2 flooded → resilience 0.5. Assert both values.

### Test 2: ML computation

Monkeypatch `predict_network` to return a fixed DataFrame. Assert ML resilience column matches expected values.

### Test 3: Plot writes PNG

Call `plot_resilience_curve` with a minimal 2-row DataFrame. Assert output file exists and is non-empty.

---

## Output

- `outputs/metrics/resilience_curve.png` — gráfica de dos curvas
