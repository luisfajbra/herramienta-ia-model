# Correcciones de la auditoría — índice

- **Fecha:** 2026-10-11
- **Rama:** `worktree-optuna-tuning-familia-ganadora` (base de la serie `9ad4df0`).
- **Auditoría de origen:** [`../2026-10-10-auditoria-completa.md`](../2026-10-10-auditoria-completa.md) y [`../anexos/`](../anexos/).
- **Spec y plan:** [`../../superpowers/specs/2026-10-11-correcciones-auditoria-design.md`](../../superpowers/specs/2026-10-11-correcciones-auditoria-design.md), [`../../superpowers/plans/2026-10-11-correcciones-auditoria.md`](../../superpowers/plans/2026-10-11-correcciones-auditoria.md).

## Orden de lectura

1. [`04-informe-consolidado.md`](04-informe-consolidado.md) — estado de cada hallazgo, tablas antes/después, limitaciones, lo que no se hizo y el **resumen para el informe de la investigación** (§11). Empezar aquí.
2. [`01-evaluacion-honesta.md`](01-evaluacion-honesta.md) — M-01 (referencias sin aprendizaje y qué mide cada protocolo), M-02 (error por forma con `tabular_v4_21`), M-03 (agregación de F1), M-08/M-10 (predicciones persistidas y matriz de confusión). Leer su §0 antes de citar cualquier cifra.
3. [`03-resolucion-de-la-etiqueta.md`](03-resolucion-de-la-etiqueta.md) — D-04/H-06: resolución del volumen y semántica de la etiqueta. **Decisión A/B pendiente del autor** (recomendación: A).
4. [`02-trazabilidad-y-seguridad.md`](02-trazabilidad-y-seguridad.md) — V-01 (figuras en muestra), V-07 (`halve_timeseries.py`), M-07/V-04 (verificación del modelo contra `promotion.json` y procedencia de figuras) y el defecto de aislamiento de pruebas.
5. [`datos/README.md`](datos/README.md) — qué contiene cada tabla, cómo se generó y cómo reproducirla.

Cada hallazgo de `01` y `02` sigue la misma estructura: **Problema y evidencia** · **Causa** · **Qué cambió** · **Verificación** · **Límites que quedan** · **Texto para la tesis**.

## Estado resumido

| Hallazgo | Estado | Commits | Documento |
|---|---|---|---|
| M-01 | Corregido en la evaluación; medido con `tabular_v4_21` | `9c8db1b`, `02cdaab`, `6c48c46`, `833760d`, `5f0586d` + medición | [01](01-evaluacion-honesta.md) §1 |
| M-02 | Medido; mitigado en parte por `tabular_v4_21`; abierto | `b831f0f`, `bd0f23f` + medición | [01](01-evaluacion-honesta.md) §2 |
| M-03 | Corregido en el banco | `b831f0f`, `2871917`, `bd0f23f`, `5528479` | [01](01-evaluacion-honesta.md) §3 |
| M-08 / T-19, M-10 (parcial) | Corregido / parcial | `bd0f23f`, `b95a480`, `4a2a87c`, `e26fc5a` | [01](01-evaluacion-honesta.md) §4 |
| V-01 (y M-13 en parte) | Corregido para figuras nuevas | `93e34b4` | [02](02-trazabilidad-y-seguridad.md) §1 |
| V-07 | Corregido | `57dc170` | [02](02-trazabilidad-y-seguridad.md) §2 |
| M-07 / V-04 | Corregido en `predict_network` y figuras; otros cargadores sin verificar | `8f17147`, `0c8a3f4` | [02](02-trazabilidad-y-seguridad.md) §3 |
| Aislamiento de pruebas (defecto nuevo) | Corregido | `398d6bc` | [02](02-trazabilidad-y-seguridad.md) §4 |
| D-04 / H-06 | Investigado; decisión pendiente | `2035a51`, `867ce6e` | [03](03-resolucion-de-la-etiqueta.md) |

Medición: `a26155e`, `4a2a87c`, `4717885`, `395a2ee`, `e26fc5a`, `284927d`. Detalle completo en [`04-informe-consolidado.md`](04-informe-consolidado.md) §1.

## Datos de evaluación (`datos/`)

| Archivo | Contenido |
|---|---|
| [`datos/comparacion_protocolos.csv`](datos/comparacion_protocolos.csv) | Una fila por (modelo, protocolo): F1, recall y precisión agregados; F1 con la definición anterior; RMSE agregado y medio por pliegue; error de volumen total; NSE oráculo |
| [`datos/matriz_confusion.csv`](datos/matriz_confusion.csv) | TN, FP, FN, TP por (modelo, protocolo) |
| [`datos/banda_limitrofe.csv`](datos/banda_limitrofe.csv) | Falsos negativos por volumen real (banda de 1 m³, fuera de la banda, ≥ 100 m³) |
| [`datos/por_forma.csv`](datos/por_forma.csv) | Una fila por (modelo, protocolo, forma): F1, RMSE, volumen real y predicho, error de volumen total |
| [`datos/tables_provenance.json`](datos/tables_provenance.json) | Comando, commit, SHA-256 de entradas y salidas, filas y corridas |
| [`datos/runs/metrics_xgboost.json`](datos/runs/metrics_xgboost.json) | Métricas completas de XGBoost por protocolo |
| [`datos/runs/metrics_baseline_interpolacion_factor.json`](datos/runs/metrics_baseline_interpolacion_factor.json) | Métricas completas de la referencia de interpolación |
| [`datos/runs/metrics_baseline_forma_mas_cercana.json`](datos/runs/metrics_baseline_forma_mas_cercana.json) | Métricas completas de la referencia de forma más cercana |
| [`datos/runs/provenance.json`](datos/runs/provenance.json) | Procedencia de la corrida: `prep_id`, features, hiperparámetros, filas de respaldo, versiones, tiempos |

**Alcance de todas las cifras:** XGBoost con hiperparámetros por defecto de `config.yaml` (no el modelo ajustado con Optuna), una red, contrato `tabular_v4_21`, `prep_id` `ca706e27654e0cf5`, 154 080 filas = 963 de 1000 corridas × 160 nodos.

## Parte 2 — comparación justa XGBoost / MLP / SVM aproximado (2026-10-12/13)

| Documento / datos | Contenido |
|---|---|
| [06-seleccion-de-modelos.md](06-seleccion-de-modelos.md) | Diseño de la comparación, resultados con formas no vistas, efecto de la afinación, límites y texto para la tesis |
| [datos_parte2/](datos_parte2/) | Tablas de la corrida (comparación, matriz de confusión, banda de 1–2 m³, por forma, resumen y convergencia de la afinación) y `tables_provenance.json` |
| `../../superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md` | Spec de la parte 2 |
| `scripts/ejecutar_parte2_v4_21.py` | Script resumible (`--family {xgboost,mlp,svm}` y `--build-tables`) |
