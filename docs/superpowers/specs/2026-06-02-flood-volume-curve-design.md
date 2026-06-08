# Design: Flood Volume Curve

**Date:** 2026-06-02
**Feature:** `--flood-volume-curve` CLI command
**Status:** Design approved, pending implementation plan

---

## Purpose

Show how the **total flooded volume of the network** grows as the flow multiplier increases, comparing SWMM ground truth vs ML predictions. Complements the resilience curve: resilience shows how many nodes fail (binary), this shows how much volume floods (magnitude).

---

## Computation

```
vol_total_swmm(factor) = Σ vol_inundacion_m3  for all nodes at that factor (from CSV)
vol_total_ml(factor)   = Σ vol_pred_m3         from predict_network(factor, config, models_dir)
```

---

## Architecture

Two new modules:

| File | Responsibility |
|---|---|
| `swmm_resilience/analysis/flood_volume.py` | `compute_flood_volume_curve(...)` — pure computation, no matplotlib |
| `swmm_resilience/visualization/flood_volume_curve.py` | `plot_flood_volume_curve(...)` — plotting only |
| `tests/test_flood_volume.py` | Unit tests for both |
| `main.py` | Add `--flood-volume-curve` flag and handler |

---

## Module 1: `swmm_resilience/analysis/flood_volume.py`

### Public function

```python
compute_flood_volume_curve(
    df_swmm: pd.DataFrame,
    factors: list[float],
    config,
    models_dir: Optional[Path],
    _predict_fn: Optional[Callable] = None,
) -> pd.DataFrame
```

### Returns

DataFrame with columns:

| Column | Type | Description |
|---|---|---|
| `factor` | float | Factor multiplicador |
| `vol_total_swmm` | float | Volumen total inundado según SWMM (m³) |
| `vol_total_ml` | float | Volumen total inundado según predicción ML (m³) |

### ML injection logic (identical to resilience module)

```python
use_ml = _predict_fn is not None or (config is not None and models_dir is not None)
actual_predict = _predict_fn if _predict_fn is not None else predict_network
```

### SWMM calculation (per factor)

```python
df_f = df_swmm[abs(df_swmm["factor_mult"] - factor) < 1e-6]
vol_swmm = float(df_f["vol_inundacion_m3"].sum())
```

### ML calculation (per factor)

```python
pred = actual_predict(factor, config, models_dir)
vol_ml = float(pred["vol_pred_m3"].sum())
```

---

## Module 2: `swmm_resilience/visualization/flood_volume_curve.py`

### Public function

```python
plot_flood_volume_curve(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]
```

### Returns

`(path_swmm, path_ml)` where:
- `path_swmm = output_dir / "flood_volume_swmm.png"`
- `path_ml   = output_dir / "flood_volume_ml.png"`

### Visual spec

Each PNG has **two subplots side by side** — left: linear scale, right: log scale (`ax.set_yscale("log")`).

| Element | Value |
|---|---|
| Figure size | `(14, 5)` |
| Left subplot title | `"Escala lineal"` |
| Right subplot title | `"Escala logarítmica"` |
| X label (both) | `"Factor multiplicador de caudal"` |
| Y label (both) | `"Volumen total inundado (m³)"` |
| SWMM PNG overall title | `"Volumen total de inundación — SWMM (real)"` |
| ML PNG overall title | `"Volumen total de inundación — Predicción ML"` |
| SWMM line color | `#2176ae` (azul), marker `o` |
| ML line color | `#e07b39` (naranja), marker `s` |
| Grid | `True, alpha=0.3` on both subplots |
| Save | `fig.savefig`, `plt.close(fig)` |
| Parent dir | `output_dir.mkdir(parents=True, exist_ok=True)` |

**Note on log scale:** If all values for a factor are 0 (no flooding at low factors), `yscale("log")` will produce warnings/gaps — this is acceptable and expected behavior.

---

## CLI

```bash
python main.py --flood-volume-curve
```

Handler in `main.py` (after `--resilience-curve` block, before `--only-maps`):

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

Requires: `dataset_final.csv` + trained models in `outputs/models/`.

---

## Tests (`tests/test_flood_volume.py`)

### Test 1: SWMM volume computation

Synthetic DataFrame: 4 nodes × 2 factors. Factor 1.0: all vol=0. Factor 2.0: two nodes with vol=10.0 each → sum=20.0. Assert both values.

### Test 2: ML volume computation

Monkeypatch `predict_network` to return fixed `vol_pred_m3`. Assert ML vol column matches expected sum.

### Test 3: Plot writes two PNGs

Call `plot_flood_volume_curve` with a minimal 3-row DataFrame (factors 1.0, 2.0, 3.0 with non-zero vols). Assert both output files exist and are non-empty.

---

## Output

- `outputs/metrics/flood_volume_swmm.png` — SWMM (linear + log subplots)
- `outputs/metrics/flood_volume_ml.png` — ML (linear + log subplots)
