# Benchmark justo de 4 familias tras Fase C2 — resultados (2026-10-05)

Primera corrida real de punta a punta del pipeline (Optuna, Fase B1, Fase B2,
Fase C1, Fase C2) contra la red SWMM real, después de que todo ese trabajo se
validó solo con fixtures sintéticos en tests. Este documento registra los
números para usar en la presentación y el informe de tesis.

Artefactos pesados (mapas, CSVs, reportes crudos) archivados en
`resultados_archivados/2026-10-05-benchmark-fase-c2/` (fuera de git — son
binarios grandes y reproducibles con el pipeline; ver sección final).

## 1. Dataset generado

- **39 formas de hidrograma** × **25 factores de intensidad** (0.2 a 5.0,
  paso 0.2) + la forma base = **1000 corridas SWMM**, 160 nodos cada una =
  **160,000 filas** en `dataset_final.csv`.
- **0 corridas sin drenar.** Ningún escenario necesitó el reintento con
  extensión de drenaje (Fase C2) — todos clasificaron `drained` en el primer
  intento.
- **0 errores, 0 crashes** en toda la corrida.

## 2. Exclusión por continuidad de routing (Fase C2)

Primera vez que el filtro corre contra datos reales — hasta ahora solo se
había probado con fixtures.

- **37 de 1000 corridas (3.7%)** superan el umbral de continuidad de rutina
  del 5% y quedan excluidas del entrenamiento y del benchmark principal.
- Concentradas en **factores bajos (0.2–0.8)** de formas de pico corto/abrupto
  (`shape_ultra_short_impulse`, `rare_extreme`, `common_bogota`,
  `shape_flash_flood_urban`, `shape_micro_burst_45min`) — consistente con que
  SWMM tiene más ruido numérico en corridas de bajo caudal y duración corta.
- Peor caso: `shape_ultra_short_impulse` a factor 0.2, **12.38%** de error de
  continuidad.
- La forma base (`base__f0.20`) también quedó excluida (5.17%) — confirma que
  el fix del hallazgo I2 (las formas base antes nunca recibían su valor de
  continuidad) funciona en producción.
- Reporte completo: `continuity_excluded.csv` (copiado en el archivo de
  resultados).
- **Entrenamiento legacy (XGBoost vía `--persist-sql`)**: 5920 de 160,000
  filas (37 corridas × 160 nodos) descartadas por el mismo filtro.

## 3. Ranking final del banco de 5 familias

El SVM RBF exacto de sklearn se reemplazó por una aproximación escalable:
`StandardScaler` → `Nystroem(kernel="rbf", n_components=200)` →
`SGDClassifier`/`SGDRegressor`. El clasificador usa `loss="log_loss"` para
mantener `predict_proba` sin el calibrado interno de `SVC(probability=True)`.
Esta configuración conserva una frontera no lineal tipo RBF, pero su costo
crece aproximadamente de forma lineal con las filas. Científicamente debe
reportarse como **aproximación de kernel RBF con Nystroem + SGD**: el
clasificador usa regresión logística sobre el mapeo RBF y el regresor usa
pérdida insensible a epsilon. La clave interna `svm` se conserva solo por
compatibilidad con el registro del banco.

Un fold real de `LeaveOneShapeOut` (150,240 filas de entrenamiento y 3,840 de
prueba) tardó **11.2 s** en ajustar clasificador y regresor y producir las
predicciones. La corrida de entrenamiento y evaluación completa terminó en
aproximadamente **9 minutos**; la versión exacta anterior había acumulado más
de 1 día y 15 horas de CPU sin terminar.

Ranking sobre el protocolo `LeaveOneShapeOut` (criterio primario:
`end_to_end.rmse_vol_todos_nodos`, menor es mejor):

| Rank | Familia | RMSE vol. todos los nodos | F1 clasificador | NSE regresor oracle |
|------|---------|---------------------------|------------------|----------------------|
| 1 | **XGBoost** | **78.04** | 0.979 | 0.815 |
| 2 | MLP | 99.61 | 0.977 | 0.631 |
| 3 | Kernel RBF aproximado (Nystroem + SGD) | 155.21 | 0.843 | 0.530 |
| 4 | Random Forest | 185.46 | 0.890 | 0.277 |
| 5 | Lineal | 201.18 | 0.817 | 0.071 |

**XGBoost gana con margen claro**: 22% mejor RMSE que el segundo lugar (MLP),
y el NSE del regresor oracle (0.815) es muchísimo más sólido que el de MLP
(0.631). La familia `svm` reconfigurada queda tercera: reduce el RMSE de
Random Forest en 16% y el del modelo lineal en 23%, aunque su F1 es menor que
el de Random Forest.

Dataset evaluado: 154,080 filas (160,000 − 5,920 excluidas por continuidad),
46,225 inundadas.

**XGBoost fue promovido** (`config.bench.promote: "xgboost"`,
`--bench-promote`, `prep_id=2db0578f875526cb`).

### Nota sobre MLP (investigado y cerrado, 2026-10-05)

El entrenamiento de MLP emitió 70 `RuntimeWarning: overflow encountered in
exp` durante la evaluación. Investigado: es **cosmético, no un bug de
predicción** — viene de una fórmula manual de sigmoide
(`1/(1+exp(-x))`) en `predict_proba()` del clasificador; cuando el logit es
muy negativo (red confiada en la clase mayoritaria), `exp()` desborda pero
el resultado sigue siendo matemáticamente correcto (satura a `0.0`, nunca
`NaN`) — verificado numéricamente. El F1 alto del clasificador (0.977) ya lo
confirmaba indirectamente. Corregido en el commit `cb661d6` (sigmoide
piecewise con indexado booleano) para limpiar el log; no cambia ningún
resultado ya reportado.

**El NSE del regresor (0.631 vs. 0.815 de XGBoost) no está relacionado con
el overflow** — el overflow venía solo del clasificador (sigmoide); el
regresor usa `MSELoss` directo sobre el target ya transformado con `log1p`,
sin ninguna sigmoide de por medio. Es simplemente un hecho real sobre esta
arquitectura en este problema (una red densa simple ajusta peor la
regresión de volumen que el gradient boosting), no algo roto que arreglar.

## 4. Búsqueda de hiperparámetros con Optuna sobre XGBoost (familia ganadora)

50 trials, `GroupKFold(5)` interno por `shape_id`, semilla 42, ~21 minutos de
corrida total (1252.6 s).

**Status: `estimacion_exploratoria_condicionada`** — el propio estudio se
etiqueta así (spec §12): *"No se reservó un holdout de formas separado ni se
aplicó selección de familia anidada; este resultado no debe presentarse como
confirmatorio."* Es una búsqueda exploratoria sobre la familia ya elegida por
el ranking de la sección 3, no una validación independiente.

**Mejor trial: #47**, valor objetivo (`end_to_end.rmse_vol_todos_nodos`) =
**108.84**:

| Hiperparámetro | Default (config.yaml) | Mejor trial (#47) |
|---|---|---|
| Clasificador `max_depth` | 6 | 5 |
| Clasificador `learning_rate` | 0.05 | 0.136 |
| Clasificador `n_estimators` | 200 | 313 |
| Clasificador `subsample` | 0.8 | 0.791 |
| Regresor `max_depth` | 6 | 8 |
| Regresor `learning_rate` | 0.05 | 0.199 |
| Regresor `n_estimators` | 200 | 421 |
| Regresor `subsample` | 0.8 | 0.910 |

Estudio completo (50 trials con sus parámetros y valores) guardado en
`resultados_archivados/2026-10-05-benchmark-fase-c2/tuning/xgboost/2db0578f875526cb_20261006T015644Z/study.json`.

**Nota metodológica importante para no confundir los números:** Optuna
evalúa cada combinación de hiperparámetros con su propio `GroupKFold(5)` por
`shape_id` (5 folds, ~8 formas retenidas a la vez por fold) — **no** es el
mismo split que el ranking de la sección 3 (`LeaveOneShapeOut`, 40 folds, 1
forma retenida a la vez). Ambos promedian el RMSE por fold de la misma
manera (promedio simple, no agrupado), pero el split de Optuna es
deliberadamente más exigente (menos formas para entrenar, más formas nuevas
a la vez para predecir), así que sus valores de `objective` **no son
comparables directamente** con el 78.04 de la tabla de arriba — son más
altos porque el fold es más difícil, no porque el modelo sea peor. Es el
diseño correcto (evita que la búsqueda de hiperparámetros se sobreajuste al
protocolo más amable del ranking).

## 5. Dónde está todo

- Dataset real: `data/training/dataset_final.csv` (27 MB, no versionado)
- SQLite v17 con el filtro de continuidad aplicado: `outputs/training_v17.sqlite3`
- Reporte de excluidos: `outputs/continuity_excluded.csv`
- Ranking y reportes del banco: `outputs/bench/reports/`
- **Copia archivada (fuera de git, no se sobreescribe con próximas corridas):**
  `resultados_archivados/2026-10-05-benchmark-fase-c2/`
  - `mapas/` — 980 mapas de inundación (SWMM vs. predicción, por forma y factor)
  - `ranking_4_familias.csv`
  - `ranking_5_familias.csv` — ranking final con la familia `svm` reconfigurada
  - `continuity_excluded.csv`
  - `bench_reports/` — ranking.json/csv crudos del banco
  - `tuning/xgboost/2db0578f875526cb_20261006T015644Z/study.json` — estudio
    Optuna completo (50 trials)
  - `paridad/<familia>/` — gráficos de paridad OOF por familia (los mismos
    usados en la presentación)
- **Presentación**: `Presentacion_MLSMFP_05-10_resultados_fase_c2.pptx`
  (versionada en git) — replica `Presentacion_MLSMFP_15-09_top5_resultados.pptx`
  (30 slides originales intactas, verificado) + 6 slides nuevas con todo lo
  de este informe.

## 6. Pendiente

- [x] Reconfigurar la familia `svm` para que corra en tiempo razonable a esta
      escala y sumarla al ranking — Nystroem RBF + SGD, tercera con RMSE
      155.21.
- [x] Revisar la inestabilidad numérica de MLP — era cosmético, corregido
      (ver nota de la sección 3).
- [x] Generalizar mapas de paridad/inundación a las 5 familias — hecho para
      XGBoost, Random Forest, Lineal, MLP y la familia `svm` reconfigurada,
      usando predicciones OOF consistentes con el ranking.
- [x] `config_7shapes.yaml` desactualizado — eliminado (commit `6f4b202`):
      ninguna de las tolerancias/bloques agregados desde Fase A quedaban
      reflejados ahí y `promote: "auto"` ya era inválido.
- [x] Entrenar sobre el contrato `tabular_v4_21` (features tributarias,
      Fase B2) y conectar `ScenarioPredictor` a ese modelo — hecho en el
      plan de migración `2026-10-06-migracion-tabular-v4-21` (tarea 7):
      confirmado +27.20% RMSE con el protocolo oficial `LeaveOneShapeOut`
      (ver `docs/resultados/2026-10-06-explicacion-completa.md`, sección 11),
      modelo XGBoost final promovido a `outputs/models/`.
- [ ] Decisión de integración (merge a la rama real) — pospuesta por el
      usuario.
