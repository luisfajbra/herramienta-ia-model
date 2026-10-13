# Audit corrections — index

- **Date:** 2026-10-11
- **Branch:** `worktree-optuna-tuning-familia-ganadora` (series base `9ad4df0`).
- **Source audit:** [`../2026-10-10-auditoria-completa.md`](../2026-10-10-auditoria-completa.md) and [`../anexos/`](../anexos/).
- **Spec and plan:** [`../../superpowers/specs/2026-10-11-correcciones-auditoria-design.md`](../../superpowers/specs/2026-10-11-correcciones-auditoria-design.md), [`../../superpowers/plans/2026-10-11-correcciones-auditoria.md`](../../superpowers/plans/2026-10-11-correcciones-auditoria.md).

## Reading order

1. [`04-informe-consolidado.md`](04-informe-consolidado.md) — status of each finding, before/after tables, limitations, what was not done, and the **summary for the research report** (§11). Start here.
2. [`01-evaluacion-honesta.md`](01-evaluacion-honesta.md) — M-01 (no-learning references and what each protocol measures), M-02 (per-shape error with `tabular_v4_21`), M-03 (F1 aggregation), M-08/M-10 (persisted predictions and confusion matrix). Read its §0 before quoting any figure.
3. [`03-resolucion-de-la-etiqueta.md`](03-resolucion-de-la-etiqueta.md) — D-04/H-06: volume resolution and label semantics. **Path A/B decision pending for the author** (recommendation: A).
4. [`02-trazabilidad-y-seguridad.md`](02-trazabilidad-y-seguridad.md) — V-01 (figures on a sample), V-07 (`halve_timeseries.py`), M-07/V-04 (model verification against `promotion.json` and figure provenance) and the test-isolation defect.
5. [`datos/README.md`](datos/README.md) — what each table contains, how it was generated and how to reproduce it.

Each finding in `01` and `02` follows the same structure: **Problem and evidence** · **Cause** · **What changed** · **Verification** · **Remaining limits** · **Text for the thesis**.

## Status summary

| Finding | Status | Commits | Document |
|---|---|---|---|
| M-01 | Corrected in the evaluation; measured with `tabular_v4_21` | `9c8db1b`, `02cdaab`, `6c48c46`, `833760d`, `5f0586d` + measurement | [01](01-evaluacion-honesta.md) §1 |
| M-02 | Measured; partly mitigated by `tabular_v4_21`; open | `b831f0f`, `bd0f23f` + measurement | [01](01-evaluacion-honesta.md) §2 |
| M-03 | Corrected in the benchmark | `b831f0f`, `2871917`, `bd0f23f`, `5528479` | [01](01-evaluacion-honesta.md) §3 |
| M-08 / T-19, M-10 (partial) | Corrected / partial | `bd0f23f`, `b95a480`, `4a2a87c`, `e26fc5a` | [01](01-evaluacion-honesta.md) §4 |
| V-01 (and M-13 in part) | Corrected for new figures | `93e34b4` | [02](02-trazabilidad-y-seguridad.md) §1 |
| V-07 | Corrected | `57dc170` | [02](02-trazabilidad-y-seguridad.md) §2 |
| M-07 / V-04 | Corrected in `predict_network` and figures; other loaders not verified | `8f17147`, `0c8a3f4` | [02](02-trazabilidad-y-seguridad.md) §3 |
| Test isolation (new defect) | Corrected | `398d6bc` | [02](02-trazabilidad-y-seguridad.md) §4 |
| D-04 / H-06 | Investigated; decision pending | `2035a51`, `867ce6e` | [03](03-resolucion-de-la-etiqueta.md) |

Measurement: `a26155e`, `4a2a87c`, `4717885`, `395a2ee`, `e26fc5a`, `284927d`. Full detail in [`04-informe-consolidado.md`](04-informe-consolidado.md) §1.

## Evaluation data (`datos/`)

| File | Contents |
|---|---|
| [`datos/comparacion_protocolos.csv`](datos/comparacion_protocolos.csv) | One row per (model, protocol): pooled F1, recall and precision; F1 under the previous definition; pooled and per-fold-mean RMSE; total volume error; oracle NSE |
| [`datos/matriz_confusion.csv`](datos/matriz_confusion.csv) | TN, FP, FN, TP per (model, protocol) |
| [`datos/banda_limitrofe.csv`](datos/banda_limitrofe.csv) | False negatives by true volume (1 m³ band, outside the band, ≥ 100 m³) |
| [`datos/por_forma.csv`](datos/por_forma.csv) | One row per (model, protocol, shape): F1, RMSE, true and predicted volume, total volume error |
| [`datos/tables_provenance.json`](datos/tables_provenance.json) | Command, commit, SHA-256 of inputs and outputs, rows and runs |
| [`datos/runs/metrics_xgboost.json`](datos/runs/metrics_xgboost.json) | Full XGBoost metrics per protocol |
| [`datos/runs/metrics_baseline_interpolacion_factor.json`](datos/runs/metrics_baseline_interpolacion_factor.json) | Full metrics of the interpolation reference |
| [`datos/runs/metrics_baseline_forma_mas_cercana.json`](datos/runs/metrics_baseline_forma_mas_cercana.json) | Full metrics of the nearest-shape reference |
| [`datos/runs/provenance.json`](datos/runs/provenance.json) | Run provenance: `prep_id`, features, hyperparameters, fallback rows, versions, timings |

**Scope of all figures:** XGBoost with the default hyperparameters of `config.yaml` (not the Optuna-tuned model), one network, `tabular_v4_21` contract, `prep_id` `ca706e27654e0cf5`, 154,080 rows = 963 of 1000 runs × 160 nodes.

## Part 2 — fair XGBoost / MLP / approximate SVM comparison (2026-10-12/13)

| Document / data | Contents |
|---|---|
| [06-seleccion-de-modelos.md](06-seleccion-de-modelos.md) | Design of the comparison, results with unseen shapes, effect of tuning, limits and text for the thesis |
| [datos_parte2/](datos_parte2/) | Run tables (comparison, confusion matrix, 1–2 m³ band, per shape, summary and tuning convergence) and `tables_provenance.json` |
| `../../superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md` | Part 2 spec |
| `scripts/ejecutar_parte2_v4_21.py` | Resumable script (`--family {xgboost,mlp,svm}` and `--build-tables`) |
