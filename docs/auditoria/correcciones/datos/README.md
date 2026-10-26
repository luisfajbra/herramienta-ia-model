# Evaluation data for the audit corrections (M-01, M-02, M-03)

These tables are the source of the figures quoted by the thesis report on
the evaluation of XGBoost and of the no-learning references. Every quoted
number must come from here and be traceable with `tables_provenance.json`.

## What was evaluated

- **Model:** XGBoost with the **default hyperparameters of `config.yaml`**
  (`bench.families.xgboost`: 200 trees, depth 6, learning rate 0.05,
  subsampling 0.8, the same in the classifier and in the regressor). It is **not**
  the model tuned with Optuna: these tables measure protocols and metrics.
- **No-learning references:** `baseline_interpolacion_factor`
  (interpolates the volume of the same shape and node between neighbouring factors) and
  `baseline_forma_mas_cercana` (copies the volume of the same node and factor from the
  training shape most similar in duration and time to peak).
- **Data:** `tabular_v4_21` contract (21 features), PreparedDataset
  `ca706e27654e0cf5`. There are 154,080 rows: 963 of the 1000 runs (40 shapes × 25
  factors) × 160 nodes. The continuity filter (|error| ≤ 5 %) leaves out
  37 (shape, factor) cells. A row is flooded if its volume is ≥ 1 m³.

  > **Correction note (2026-10-11, spike D-04):** the volume is that of the `.rpt`, with a resolution of 1 m³. In the re-simulated sample (one network, 17 simulated scenarios, 15 with overflow; the threshold bound was computed on the 15 with per-node records, 13 with overflow), "≥ 1 m³" is consistent with an effective threshold of ≈ 0.5 m³ of continuous volume, bounded to (0.4988, 0.5002] m³. See `../03-resolucion-de-la-etiqueta.md`.

- **Protocols:**
  - `LOSO` and `GroupKFold5` group by intensity factor, so they measure
    **interpolation between factors**: all shapes remain in the
    training set. With this grid, `GroupKFold5` never holds out two contiguous
    factors, and the interpolation reference gives exactly the same
    predictions as in `LOSO`.
  - `LeaveOneShapeOut` holds out one complete hydrograph shape: it measures
    generalization to **unseen shapes**. It is the main protocol.

## Files

The "pooled" metrics are computed over all the test rows of the
protocol; "per-fold mean" is the average of the folds.

- `comparacion_protocolos.csv`: one row per (model, protocol). Columns:
  - `modelo` (model), `protocolo` (protocol), `que_mide` (what it measures);
  - `f1_agregado` (pooled F1), `recall_agregado` (pooled recall), `precision_agregada` (pooled precision);
  - `f1_media_por_fold_con_ceros` (zero-filled per-fold mean F1): the previous definition, which counts folds without positives as 0;
  - `folds_f1_indefinido` (folds with undefined F1);
  - `rmse_vol_agregado_m3` (pooled volume RMSE), `rmse_vol_media_por_fold_m3` (per-fold-mean volume RMSE): volume routed by the classifier, over all nodes;
  - `error_vol_total_pct` (total volume error);
  - `nse_regresor_oracle` (oracle regressor NSE): the regressor evaluated only on the nodes that actually flooded.
- `matriz_confusion.csv`: columns `modelo`, `protocolo`, `tn`, `fp`, `fn`, `tp`. In each row, tn + fp + fn + tp = 154,080.
- `banda_limitrofe.csv`: columns `modelo`, `protocolo`, `fn_total`, `fn_en_banda_1_2m3` (true volume in [1, 2) m³), `fn_fuera_de_banda` (outside the band) and `fn_con_vol_ge_100m3` (true volume ≥ 100 m³). It classifies the false negatives by true volume.
- `por_forma.csv`: one row per (model, protocol, shape). Columns: `modelo`, `protocolo`, `forma` (shape), `n_rows`, `n_positives`, `f1`, `rmse_vol`, `vol_real_m3` (true volume), `vol_pred_m3` (predicted volume) and `error_vol_pct` (volume error).
- `tables_provenance.json`: the provenance of the four CSVs. It records:
  - the exact command, the git commit and whether the versioned tree was dirty;
  - the SHA-256 of each OOF read and of each CSV written;
  - n_rows, n_runs, the complete grid and the missing cells.
- `runs/metrics_<modelo>.json`: the full metrics returned by
  `score_predictions` for each model or reference. They include `classifier`,
  `confusion_matrix`, `classifier_per_fold`,
  `classifier_legacy_mean_zero_fill`, `regressor_oracle`, `end_to_end`,
  `end_to_end_pooled`, `by_factor` and `by_shape`, separated by protocol.
- `runs/provenance.json`: the provenance of the run. It records:
  - prep_id, contract, features, rows, threshold and continuity filter;
  - the hyperparameters, the commit, the versions and the seconds of each stage;
  - the fallback rows of each reference (`baseline_fallback`).

## How to reproduce

```
# 1. Corrida completa (prepara el dataset, evalúa XGBoost y las referencias,
#    escribe los OOF en outputs/bench/oof/<prep_id>/ y runs/*.json). ~6 min.
venv/Scripts/python.exe scripts/ejecutar_evaluacion_v4_21.py

# 2. Tablas a partir de los OOF guardados (sin reentrenar). ~1,5 min.
venv/Scripts/python.exe scripts/informe_evaluacion.py --prep-id ca706e27654e0cf5 --prepared-dir outputs/bench/prepared_v4_21 --oof-root outputs/bench/oof --out docs/auditoria/correcciones/datos --flood-threshold 1.0
```

## Traceability note

The files in `runs/` (metrics and `provenance.json`) came from
`scripts/ejecutar_evaluacion_v4_21.py` at commit `4a2a87c`. That run
also wrote the CSVs. Afterwards, the four CSVs were regenerated with
`scripts/informe_evaluacion.py` from the same OOFs, without retraining, at
the commit that records `tables_provenance.json`. That version sorts the rows
by (model, protocol, shape) and writes with LF line endings. The values are
identical to those of the original run: this was checked with pandas, comparing
value by value.

The run of step 1 also writes the CSVs, but not
`tables_provenance.json`. Therefore, if it is repeated, step 2 must be run
afterwards so that the tables and their provenance match again.
