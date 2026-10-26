# figures_part2

Parity plots, per-shape errors and flood maps (English labels) for the three
tuned models of Part 2: XGBoost, MLP and approximate SVM, LeaveOneShapeOut
out-of-fold predictions (each of the 40 shapes is predicted by a model trained
without it).

## Run

From the repository root (the worktree), with the project virtual environment:

```
../../../venv/Scripts/python.exe scripts/figures_part2/make_parity_and_maps.py
```

It takes about one minute and writes only to `docs/figures_part2/` (PNG at
200 dpi plus `figure_numbers.json`, the numbers printed on the figures).

## Inputs (read-only)

- `outputs/bench/parte2/ca706e27654e0cf5/{xgboost,mlp,svm}/tuned/oof.parquet`
- `outputs/bench/prepared_v4_21/ca706e27654e0cf5/` (keys, targets), loaded with
  `swmm_resilience.ml.bench.preprocess.resolve_prepared`
- `data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp`
  (node coordinates and conduits, project parser)
- `docs/auditoria/correcciones/datos_parte2/comparacion_protocolos.csv`

The script stops with an error if the pooled RMSE it computes differs from the
CSV by more than 0.01 m3 (xgboost_tuned 69.64, mlp_tuned 101.12, svm_tuned
214.87).

## Notes

- Predicted routed volume of a row = `y_pred_reg` where `y_pred_clf == 1`, else 0.
- The shared plotting functions in `swmm_resilience/visualization/model_comparison.py`
  hardcode "XGBoost" in some labels and draw one model per figure, so the
  figures here are drawn with matplotlib in this script (the modules are not
  modified); only the `.inp` parser is reused.
- Colour language: XGBoost blue, MLP orange, approximate SVM green, SWMM dark grey.
- Scenario selection rules for the maps are documented in
  `docs/figures_part2/INDEX.md` and in `select_scenarios()`.
