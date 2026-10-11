# Correcciones de la auditoría (pasos 1 a 6) — diseño

**Fecha:** 2026-10-11
**Rama:** `fix/cierre-auditoria-hidrogramas-20260927` (checkout en el worktree `optuna-tuning-familia-ganadora`).
**Origen:** `docs/auditoria/2026-10-10-auditoria-completa.md` y anexos 01–05.
**Alcance:** hallazgos M-01, M-02, M-03, V-01, V-07, M-07/V-04 y D-04/H-06, más la persistencia de las predicciones fuera de fold (M-08/T-19). La Parte 2 (SVM justo y selección final XGBoost + MLP) es un subproyecto aparte con su propia spec.

## 1. Estado de partida verificado (re-triaje en el worktree)

| Hallazgo | Estado en el worktree |
|---|---|
| M-01 sin referencias sin aprendizaje | Abierto. `LeaveOneShapeOut` ya es el protocolo del ranking (`bench.protocols[0]`) |
| M-02 sin volumen del evento | Mitigado en parte: `tabular_v4_21` añade `volumen_local_integrado_m3`, `volumen_tributario_integrado_m3` e `integral_exceso_capacidad_m3`. RMSE de XGBoost por forma 78,04 → 56,81 m³. Falta medir NSE y error por forma con v4_21 |
| M-03 F1 promediado con ceros | Abierto (`ml/bench/metrics.py:41-43` y `mean_metrics`) |
| V-01 figuras en muestra | Abierto (`visualization/resilience_curve.py`, `flood_volume_curve.py`; salida de `--evaluate-shapes`) |
| V-07 `halve_timeseries.py` | Abierto |
| M-07/V-04 modelo ≠ `promotion.json` | En el worktree `outputs/models` está versionado y los SHA-256 coinciden con `promotion.json`. Falta una verificación automática y la procedencia en las figuras |
| D-04/H-06 resolución de la etiqueta | Abierto (`flood_threshold_m3: 1.0` = resolución del `.rpt`) |
| OOF no persistido | Abierto (`main.py`, `_run_bench`: `_, metrics = evaluate_candidate(...)`) |

Línea base de tests antes de empezar: **1.058 passed, 0 failed** (commit `1ecca50`).

## 2. Decisiones de diseño

### 2.1 Punto de enchufe
`score_predictions(prepared, oof, provenance)` en `ml/bench/evaluate.py` es la capa núcleo: puntúa predicciones fuera de fold sin saber qué modelo las generó. Las referencias sin aprendizaje, la agregación nueva de M-03 y el desglose por forma entran ahí, de modo que todos los candidatos y todas las referencias caen en la misma tabla con las mismas métricas.

### 2.2 Referencias sin aprendizaje (M-01)
Módulo nuevo `ml/bench/baselines.py`. Cada referencia recibe un `PreparedDataset` y devuelve un frame OOF con las columnas de `OOF_COLUMNS`; después se puntúa con `score_predictions`.
- `interpolacion_factor`: para una fila de test (forma, nodo, factor), interpola linealmente el volumen del mismo nodo y forma entre los factores vecinos del train. En los extremos usa el vecino más cercano. Aplica a protocolos agrupados por `factor_mult`. En `LeaveOneShapeOut` no aplica (la forma no está en train) y se omite explícitamente.
- `forma_mas_cercana`: para una fila de test (forma retenida, nodo, factor), toma el volumen del mismo nodo y factor en la forma de train más cercana en (`duracion_horas`, `tiempo_al_pico_h`) estandarizados. Aplica a `LeaveOneShapeOut`.
- Etiqueta predicha: `volumen_predicho >= threshold_m3`. Probabilidad: igual a la etiqueta predicha (0/1). El AUC-ROC de una referencia sin probabilidades reales no se interpreta; se documenta.
- Cada protocolo lleva en `schemas.py` una descripción de qué mide (`PROTOCOL_INTERPRETATION`): `LOSO` y `GroupKFold5` miden interpolación entre factores; `LeaveOneShapeOut` mide generalización a formas nuevas. Esa descripción se escribe en los JSON de reportes.

### 2.3 Agregación de métricas del clasificador (M-03)
Para cada protocolo:
- `classifier` (cifra principal) = métricas calculadas sobre todas las filas de test concatenadas del protocolo, más la matriz de confusión agregada `confusion = {tn, fp, fn, tp}` y `specificity`.
- `classifier_per_fold` = estadísticos de dispersión entre folds (`min`, `median`, `max`, `mean`) de precisión, recall y F1, calculados solo sobre folds donde la métrica está definida. En un fold sin positivos reales, recall y F1 son **NaN**, no 0. Se reporta `n_folds_undefined`.
- `classifier_legacy_mean_zero_fill` = el valor antiguo (media por fold con `zero_division=0`), conservado con nombre explícito para poder citar el antes y el después.
- El criterio de desempate `classifier.f1` del ranking pasa a leer el valor agregado.
- El test de paridad con `ml/evaluator.py` se ajusta para comparar contra `classifier_legacy_mean_zero_fill`: la paridad con el código legado sigue demostrada, y el valor nuevo se prueba aparte.
- `regressor_oracle` y `end_to_end` no cambian de definición en esta fase. Se añade a `end_to_end` el error relativo del volumen total (`vol_total_error_pct`).

### 2.4 Desglose por forma (M-02)
`score_predictions` añade `by_shape` por protocolo: para cada `shape_id` en el test, `n_positivos`, `f1`, `rmse_vol`, `vol_real_m3`, `vol_pred_m3` y `error_vol_pct`. Solo tiene sentido en `LeaveOneShapeOut` (una forma por fold), pero se calcula para cualquier protocolo. Si `shape_id` no está en `prepared.keys`, `by_shape` queda vacío.

### 2.5 Persistencia de predicciones fuera de fold (M-08/T-19)
`_run_bench` conserva el primer elemento que devuelve `evaluate_candidate` y lo escribe en `outputs/bench/oof/<prep_id>/<familia>.parquet`. Los JSON de métricas incorporan un bloque `provenance` con `prep_id`, `family`, los hiperparámetros usados y la fecha UTC. Las referencias de 2.2 se persisten igual bajo la familia `baseline_*`.

### 2.6 Figuras y salidas en muestra (V-01)
Los títulos y leyendas de `resilience_curve`, `flood_volume_curve` y la comparación por factor añaden "(en muestra)" y una nota al pie: "El modelo fue entrenado con estos escenarios; no es validación". `--evaluate-shapes` imprime un aviso en la cabecera de su salida. Las versiones fuera de muestra derivables de las predicciones del paso 2.5 quedan como mejora opcional, fuera de esta fase.

### 2.7 `halve_timeseries.py` (V-07)
Se convierte en una función `halve_timeseries(src, dst, force=False)` con `main()` bajo `if __name__ == "__main__"` y argumentos explícitos `--src` y `--dst`. Se niega a sobrescribir un `dst` existente sin `force`. Importar el módulo no tiene efectos. No se borra el archivo.

### 2.8 Coherencia modelo–`promotion.json` (M-07/V-04)
- `verify_promotion(models_dir)` en `ml/bench/promote.py`: compara el SHA-256 de `classifier.joblib` y `regressor.joblib` con `promotion.json`. Devuelve un resultado estructurado (`ok`, lista de discrepancias).
- Se invoca antes de cargar los modelos de `outputs/models` para predecir (`ml/predict.py`). Con discrepancia, lanza `ModelProvenanceError` con ambos SHA. Si no existe `promotion.json`, advierte y continúa (compatibilidad con modelos legado).
- Las rutas de generación de figuras (`visualization/runner.py`) escriben `provenance.json` junto a las figuras: familia, SHA-256 de los modelos, `prep_id`, fecha.
- Test de contrato: `--bench` y `--bench-evaluate` no escriben en `outputs/models/`.
- No se cambia `promote: "xgboost"` de `config.yaml`; la promoción sigue siendo explícita con `--bench-promote`.

### 2.9 Resolución de la etiqueta (D-04/H-06) — spike antes de código
No se cambia código en este paso hasta una decisión. El spike mide, sobre copias y en el scratchpad:
1. si el volumen inundado por nodo se puede obtener del `.out` binario (serie de inundación integrada en el tiempo) con resolución mejor que 1 m³, y cuánto difiere del `.rpt` en las combinaciones de control;
2. el tiempo por simulación y el costo estimado de re-simular las 1.000 corridas;
3. cuántas etiquetas cambiarían y qué pasa con la banda 0,5–2 m³ al usar un umbral continuo.

Con esos números el usuario decide entre:
- **A:** conservar la resolución y el umbral actuales, declararlos como supuesto y reportar una matriz de confusión que separa la banda limítrofe (volumen real de 1 a 2 m³);
- **B:** re-simular, regenerar el dataset (nuevo `prep_id`) y re-ejecutar el banco.

## 3. Archivos

| Archivo | Cambio |
|---|---|
| `swmm_resilience/ml/bench/baselines.py` | Nuevo |
| `swmm_resilience/ml/bench/metrics.py` | Agregación agregada, dispersión por fold, matriz de confusión |
| `swmm_resilience/ml/bench/evaluate.py` | `score_predictions` con las métricas nuevas y `by_shape` |
| `swmm_resilience/ml/bench/schemas.py` | `PROTOCOL_INTERPRETATION` |
| `swmm_resilience/ml/bench/reports.py` | Bloque `provenance`, descripción del protocolo |
| `swmm_resilience/ml/bench/ranking.py` / `config.yaml` | Ruta del desempate `classifier.f1` hacia el valor agregado |
| `swmm_resilience/ml/bench/promote.py`, `swmm_resilience/ml/predict.py` | `verify_promotion` y su uso |
| `swmm_resilience/visualization/*.py`, `swmm_resilience/analysis/*.py`, `main.py` | Rótulos en muestra, OOF, aviso en `--evaluate-shapes`, `provenance.json` |
| `halve_timeseries.py` | Guarda y función |
| `tests/ml/bench/…` | Pruebas nuevas y ajuste del test de paridad |
| `docs/auditoria/correcciones/` | Documentación (sección 5) |

## 4. Pruebas

Cada cambio nace de una prueba que falla (TDD) y cierra con la suite completa en verde. Pruebas clave:
- La agregación: un predictor perfecto da F1 = 1,0 agregado; en un fold sin positivos el recall del fold es NaN y `n_folds_undefined` lo cuenta; la matriz de confusión suma el número de filas.
- Las referencias: la interpolación sobre una función lineal conocida reproduce exactamente los valores omitidos; la forma más cercana elige la forma esperada en un caso mínimo; el frame OOF cumple `_validate_oof`.
- `by_shape`: error relativo calculado a mano en un caso de 2 formas.
- `verify_promotion`: coincide, no coincide y falta `promotion.json`.
- `halve_timeseries`: importar el módulo no escribe; sobrescribir sin `force` falla.
- Contrato: `--bench` y `--bench-evaluate` no tocan `outputs/models`.
- Ajuste del test de paridad: el valor legado sigue coincidiendo con `ml/evaluator.py`.

## 5. Documentación para la investigación

Directorio `docs/auditoria/correcciones/`:
- `00-indice.md`: tabla de hallazgos → estado → commit → documento.
- `01-evaluacion-honesta.md` (M-01, M-02, M-03, OOF y matriz de confusión).
- `02-trazabilidad-y-seguridad.md` (V-01, V-07, M-07/V-04).
- `03-resolucion-de-la-etiqueta.md` (D-04/H-06; resultados del spike y decisión).
- `04-informe-consolidado.md`: resultados antes/después, tablas, y el resumen que se pueda incorporar al informe de la investigación.

Cada hallazgo se documenta con: problema y evidencia, causa, cambio realizado (commit), verificación con cifras antes y después, límites que quedan y un bloque **"Texto para la tesis"**.

## 6. Ejecución

- `subagent-driven-development`: un subagente por tarea, revisión en dos etapas; modelo potente (Opus) para métricas, referencias y mediciones; Sonnet para tareas mecánicas.
- Un commit por tarea y push a `fix/cierre-auditoria-hidrogramas-20260927` al cerrar cada paso.
- Compuertas: el paso 6 se detiene tras el spike y espera la decisión A/B. La Parte 2 no empieza hasta cerrar la Parte 1.

## 7. Fuera de alcance

- Rediseñar las features de volumen del evento más allá de lo que aporta v4_21 (se decide tras medir en el paso 1).
- Curvas de resiliencia fuera de muestra (mejora opcional posterior).
- SVM, tuning justo y selección final: Parte 2.
- Alinear el checkout principal con la rama remota.
