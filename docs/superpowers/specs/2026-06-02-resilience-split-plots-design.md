# Design: Split Resilience Curve Into Two Separate Plots

**Date:** 2026-06-02
**Feature:** Modify `--resilience-curve` to generate two PNGs instead of one
**Status:** Design approved, pending implementation plan

---

## Change

Split the single `resilience_curve.png` (two lines on one chart) into two separate PNGs, each with one line. One command still generates both.

---

## Function signature change

**Before:**
```python
plot_resilience_curve(df: pd.DataFrame, output_path: Path) -> Path
```

**After:**
```python
plot_resilience_curve(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]
```

Returns `(path_swmm, path_ml)` where:
- `path_swmm = output_dir / "resilience_swmm.png"`
- `path_ml   = output_dir / "resilience_ml.png"`

---

## Visual spec

### `resilience_swmm.png`
- Single line, color `#2176ae` (azul), marker `o`
- Title: `"Curva de resiliencia — SWMM (real)"`
- X: Factor multiplicador de caudal
- Y: Resiliencia (0–1.05)
- Grid, no legend needed (single line)

### `resilience_ml.png`
- Single line, color `#e07b39` (naranja), marker `s`
- Title: `"Curva de resiliencia — Predicción ML"`
- X: Factor multiplicador de caudal
- Y: Resiliencia (0–1.05)
- Grid, no legend needed (single line)

---

## `main.py` change

Handler passes `METRICS_DIR` (directory) instead of `METRICS_DIR / "resilience_curve.png"`:

```python
    if args.resilience_curve:
        print("\nCalculando curva de resiliencia...")
        df = pd.read_csv(config.dataset.output_path)
        factors = sorted(df["factor_mult"].unique())
        result = compute_resilience_curve(df, factors, config, MODELS_DIR)
        print("\nResiliencia por factor:")
        print(result.to_string(index=False))
        plot_resilience_curve(result, METRICS_DIR)
        return
```

---

## File map

| File | Action |
|---|---|
| `swmm_resilience/visualization/resilience_curve.py` | Modify — new signature, two figures |
| `tests/test_resilience.py` | Modify — update plot test for two output files |
| `main.py` | Modify — pass `METRICS_DIR` instead of file path |

---

## Test update

Replace `test_plot_resilience_curve_writes_png` with:

```python
def test_plot_resilience_curve_writes_two_pngs(tmp_path):
    df = pd.DataFrame({
        "factor": [1.0, 2.0],
        "resilience_swmm": [1.0, 0.5],
        "resilience_ml":   [1.0, 0.6],
    })

    path_swmm, path_ml = resilience_curve.plot_resilience_curve(df, tmp_path)

    assert (tmp_path / "resilience_swmm.png").exists()
    assert (tmp_path / "resilience_swmm.png").stat().st_size > 0
    assert (tmp_path / "resilience_ml.png").exists()
    assert (tmp_path / "resilience_ml.png").stat().st_size > 0
    assert path_swmm == tmp_path / "resilience_swmm.png"
    assert path_ml   == tmp_path / "resilience_ml.png"
```
