# Corrección 02 — Trazabilidad de figuras y modelos, y seguridad de scripts (V-01, V-07, M-07/V-04)

- **Fecha:** 2026-10-11
- **Rama:** `worktree-optuna-tuning-familia-ganadora`. Commits `93e34b4`, `57dc170`, `8f17147`, `0c8a3f4` y `398d6bc`.
- **Origen de los hallazgos:** [`../2026-10-10-auditoria-completa.md`](../2026-10-10-auditoria-completa.md), [`../anexos/05-analisis-visualizacion-soporte.md`](../anexos/05-analisis-visualizacion-soporte.md) (V-01, V-04, V-07) y [`../anexos/02-ml-protocolo-evaluacion.md`](../anexos/02-ml-protocolo-evaluacion.md) (M-07, M-13).
- **Alcance:** estos cambios no modifican ningún modelo, dato ni métrica. Hacen visible si una figura es de ajuste en muestra, con qué modelo se generó, y evitan que un script o una prueba alteren archivos de referencia.

---

## 1. V-01 (y M-13 en parte) — Figuras y salidas en muestra sin rotular

### Problema y evidencia

Las curvas de resiliencia, de volumen total inundado y la comparación por factor evalúan el modelo sobre los mismos factores y nodos con los que se entrenó: `train_models` ajusta con el dataset completo y `predict_network` se evalúa sobre los 25 factores del dataset (0,2 a 5,0) y los 160 nodos (anexo 05, V-01). Los títulos decían "SWMM (Actual)" y "ML Prediction" sin indicar que es ajuste en muestra. Lo mismo ocurría con la salida de `--evaluate-shapes`, que imprime F1 y NSE sobre formas y factores de entrenamiento (anexo 02, M-13). Un lector puede tomar esas figuras y cifras como validación.

### Causa

Las figuras se diseñaron para mostrar el comportamiento del modelo final sobre la malla de simulaciones; no había una distinción explícita entre ajuste y validación en las rutinas de visualización ni en el CLI.

### Qué cambió

Commit `93e34b4`:

| Archivo | Cambio |
|---|---|
| `swmm_resilience/visualization/labels.py` | Constantes `IN_SAMPLE_NOTE` (pie de figura, en inglés: "In-sample: the model was trained on these scenarios; this is not an independent validation.") e `IN_SAMPLE_BANNER` (aviso de consola, en español: "[EN MUESTRA] Estos escenarios (formas y factores) se usaron para entrenar el modelo; las métricas no son una validación independiente."), y las funciones `add_in_sample_note(fig)` y `print_in_sample_banner()` |
| `swmm_resilience/visualization/resilience_curve.py` | Pie en `resilience_ml.png` (la figura solo-SWMM no cambia) |
| `swmm_resilience/visualization/flood_volume_curve.py` | Pie en `flood_volume_ml.png` y `flood_volume_combined.png` |
| `swmm_resilience/visualization/model_comparison.py` | Pie en las figuras de comparación por factor (`volume_by_node_factor_*`, `parity_factor_*`) |
| `main.py` | `--evaluate-shapes` imprime el aviso antes de sus métricas |

### Verificación

| Comprobación | Antes | Después | Fuente |
|---|---|---|---|
| Familias de figuras ML con aviso "en muestra" | 0 | 5 (`resilience_ml`, `flood_volume_ml`, `flood_volume_combined`, `volume_by_node_factor_*`, `parity_factor_*`) | Commit `93e34b4`; `tests/visualization/test_in_sample_note.py` |
| Aviso en la salida de `--evaluate-shapes` | No | Sí | `test_the_evaluate_shapes_branch_prints_the_in_sample_banner` |
| Pruebas nuevas | — | 6, todas en verde | `tests/visualization/test_in_sample_note.py` (`test_add_in_sample_note_writes_the_note_into_the_figure`, `test_resilience_ml_figure_carries_the_in_sample_note`, `test_flood_volume_figures_carry_the_in_sample_note`, `test_factor_comparison_figures_carry_the_in_sample_note`, `test_print_in_sample_banner_emits_the_banner`, `test_the_evaluate_shapes_branch_prints_the_in_sample_banner`) |

Según el registro de la revisión de la tarea, el revisor inspeccionó las figuras renderizadas.

### Límites que quedan

- **Las figuras ya existentes en `outputs/metrics/` no se regeneraron.** No llevan el aviso. Solo lo llevan las que se generen de ahora en adelante.
- **Se añadió un pie de figura, no se cambiaron títulos ni leyendas.** La spec (§2.6) pedía añadir "(en muestra)" también al título; la implementación se limitó al pie.
- **El pie está en inglés**, como el resto del texto de las figuras. Si una figura entra en la tesis, el carácter "en muestra" debe decirse también en la leyenda en español.
- **M-13 queda parcialmente abierto.** `--evaluate-generalization` (interpolación entre factores vistos) y `--evaluate-validation-shapes` (que protege la condición fuera de muestra solo por el nombre de la forma) no se rotularon.
- **No se generaron versiones fuera de muestra de las curvas** a partir de las predicciones fuera de pliegue (mejora opcional, spec §7).
- Pruebas: la de la figura comprueba el texto de la nota, no su posición en la imagen; con más de 30 nodos la nota queda cerca del rótulo del eje (minucias diferidas, ver `04`).

### Texto para la tesis

> Las curvas de resiliencia, de volumen total inundado y las comparaciones por factor que se presentan para el modelo final se calcularon sobre las mismas combinaciones de forma, factor y nodo con las que se entrenó el modelo. Describen, por tanto, el ajuste del modelo a los datos de entrenamiento y no su capacidad predictiva; así se indica en cada figura. La evaluación fuera de muestra es la de la validación cruzada por forma retenida y la de los hidrogramas de validación no usados en el entrenamiento.

---

## 2. V-07 — `halve_timeseries.py` sobrescribía la red base al importarse

### Problema y evidencia

`halve_timeseries.py` era un script versionado sin guarda `if __name__ == "__main__"` y con rutas relativas fijas: al importarlo o ejecutarlo leía la red Qx2 y **sobrescribía** el `.inp` base `chico_hydro-qx1/...Qx1.00.inp` (el que usa `config.yaml`), dividiendo la serie por 2 (anexo 05, V-07; confirmado por lectura en la auditoría §3.1). Eso cambiaría el hash MD5 de la red, y `predict_network` rechazaría los modelos (`training_inp_hash.txt`). La auditoría comprobó además que el `.inp` Qx1 actual es consistente con Qx2/2 (`max|Qx1·2 − Qx2| = 0,001`), es decir, el script o un equivalente ya se aplicó en algún momento.

### Causa

Código ejecutable a nivel de módulo, sin argumentos ni protección contra sobrescritura.

### Qué cambió

Commit `57dc170`, archivos `halve_timeseries.py` y `tests/scripts/test_halve_timeseries.py`:

- Función `halve_timeseries(src, dst, force=False)` que divide por 2 solo la sección `[TIMESERIES]`.
- `main()` con argumentos explícitos `--src`, `--dst` y `--force`, bajo `if __name__ == "__main__"`. Importar el módulo no tiene efectos.
- Se niega a sobrescribir un destino existente sin `force` y a usar el origen como destino.
- El archivo no se borró.

### Verificación

| Comprobación | Antes | Después | Fuente |
|---|---|---|---|
| Importar el módulo escribe en `data/` | Sí (auditoría §3.1) | No | `test_importing_the_module_has_no_side_effects` |
| Sobrescribe un destino existente sin `--force` | Sí (siempre escribía) | No | `test_refuses_to_overwrite_an_existing_destination_without_force` |
| Acepta el origen como destino | — | No | `test_refuses_to_use_the_source_as_destination` |
| Solo cambia `[TIMESERIES]` | — | Sí | `test_halves_only_the_timeseries_section` |
| `git status --short data/` antes y después de las pruebas | — | Vacío en ambos casos | Informe de la tarea 6 (notas de trabajo no versionadas) |

### Límites que quedan

- La procedencia de las redes Qx1, Qx2 y Qx3 (Qx1 como derivado redondeado de Qx2) sigue sin documentarse (recomendación del anexo 05).
- Minucias diferidas: aserciones laxas (`in`) en la prueba de división, `main()` sin prueba de CLI, ventana de carrera entre comprobar y escribir el destino, y ningún aviso si la sección no tiene filas (ver `04`).
- El hallazgo emparejado en el plan de la auditoría, T-04 (`python -m swmm_resilience.ml.train --help` entrena), no se trató en esta serie.

### Texto para la tesis

No requiere texto en la tesis: es un riesgo de integridad del repositorio, no de los resultados. Si se quiere mencionar en la sección de reproducibilidad: *"Los scripts auxiliares que derivan redes de otras redes se ejecutan solo de forma explícita y no sobrescriben archivos existentes, de modo que la red base usada para generar los datos no puede alterarse por accidente."*

---

## 3. M-07 / V-04 — Modelo en disco distinto del promovido; figuras sin procedencia

### Problema y evidencia

En el checkout auditado (rama `fix/cierre-auditoria-hidrogramas-20260927`, commit `6e8579b`), `outputs/models/classifier.joblib` tenía SHA-256 `660be918…` mientras `promotion.json` declaraba `e2e7925e…`; el regresor, `6b61f627…` frente a `40e67e22…` (auditoría §3.1; anexo 05, V-04). El modelo en disco lo había reentrenado el pipeline por defecto (`train_models`) el 3 de octubre, con 40 formas, mientras la promoción registraba un candidato del banco de 19 formas (anexo 02, M-07). Las figuras de `outputs/metrics/` eran del 6 al 15 de septiembre, anteriores al modelo cargado. No se podía afirmar qué modelo había generado cada figura ni si `promotion.json` decía la verdad.

**En el worktree donde se hicieron estas correcciones el estado de partida era distinto:** los modelos de `outputs/models` están versionados y sus SHA-256 ya coincidían con `promotion.json` (spec §1). El propio `promotion.json`, en cambio, estaba ignorado por `.gitignore` (`outputs/models/*`) y no se versionaba; desde el commit `25eb169` está versionado con `git add -f`, y `.gitignore` tiene la excepción explícita `!outputs/models/promotion.json` (commit `e90448d`), de modo que futuras promociones no se ignoran en silencio. `verify_promotion` detiene la predicción (`ModelProvenanceError`) solo cuando `promotion.json` EXISTE y no coincide con los modelos; si falta, advierte y continúa, y las figuras quedan con familia "desconocida". Para este documento se repitió la comprobación con `verify_promotion("outputs/models")`, que devuelve `{'status': 'verified', 'ok': True, 'discrepancies': []}`; el `promotion.json` del worktree declara `family: xgboost`, `classifier_sha256` `a7bb8f8f…`, `source` `scripts/comparar_contratos_xgboost.py` y fecha 2026-10-06. Es decir, la discrepancia de la auditoría era del checkout atrasado, no del worktree. Lo que faltaba en ambos era una **verificación automática** al cargar el modelo y la **procedencia** en las figuras.

### Causa

`predict_network` cargaba `classifier.joblib` y `regressor.joblib` sin comparar sus SHA-256 con `promotion.json`, y el pipeline por defecto podía reescribir `outputs/models` sin tocar `promotion.json` (anexo 02, M-07: `main.py:1082`, `ml/trainer.py:78-80` en el checkout auditado). En el worktree, `train_models` ya se negaba a sobrescribir un modelo con otro número de features y escribía su propio `promotion.json` (cambios anteriores a esta serie, no revisados aquí).

### Qué cambió

| Commit | Archivo | Cambio |
|---|---|---|
| `8f17147` | `swmm_resilience/ml/bench/promote.py` | `verify_promotion(models_dir)` compara el SHA-256 de `classifier.joblib` y `regressor.joblib` con `promotion.json` y devuelve un resultado estructurado; `assert_promotion_consistent` lanza `ModelProvenanceError` con ambos SHA si no coinciden, y advierte y continúa si no existe `promotion.json` (modelos legado); `model_provenance` resume el modelo activo |
| `8f17147` | `swmm_resilience/ml/predict.py` | `predict_network` verifica antes de cargar los modelos (importación diferida para no cargar torch) |
| `8f17147` | `swmm_resilience/visualization/runner.py` | `write_figure_provenance` escribe `provenance.json` junto a las figuras ML de `generate_ml_map` y del lote `run` (en el lote, solo si se generó al menos una figura nueva, para no atribuir figuras viejas al modelo actual) |
| `0c8a3f4` | `promote.py`, `main.py` | Sin `promotion.json`, la procedencia registra el SHA-256 real de los archivos del modelo (familia "desconocida"); `main.py` estampa `provenance.json` tras `--resilience-curve`, `--flood-volume-curve` y la comparación por factor |
| `0c8a3f4` | `tests/test_ml_trainer_predict.py`, `tests/test_predict_shape_features.py` | Las pruebas existentes de predicción crean un `promotion.json` coherente con su modelo de prueba (eliminan avisos espurios) |

### Verificación

| Comprobación | Antes | Después | Fuente |
|---|---|---|---|
| `predict_network` con un modelo que no coincide con `promotion.json` | Carga y predice | Se niega (`ModelProvenanceError` con ambos SHA) | `test_predict_network_refuses_models_that_do_not_match_promotion_json`, `test_assert_raises_on_mismatch_naming_the_hashes` |
| Sin `promotion.json` | Carga sin aviso | Advierte y continúa | `test_predict_network_warns_and_proceeds_without_promotion_json` |
| Figuras de curvas y comparación con `provenance.json` | No | Sí | `test_figure_provenance_records_the_active_model`, `test_figure_provenance_without_a_record_stamps_the_artifact_hashes`, pruebas de `tests/test_cli_sql_reads.py` |
| Importar `ml/predict.py` carga torch | No | No (se mantiene) | `test_importing_predict_does_not_load_torch` |
| Estado de `outputs/models` en el worktree | SHA coincidentes (spec §1) | `verified`, 0 discrepancias | `verify_promotion("outputs/models")`, ejecutado 2026-10-11 |
| Pruebas nuevas | — | 15 en `tests/ml/bench/test_verify_promotion.py` y 2 en `tests/visualization/test_figure_provenance.py` | Archivos citados |

Durante todas las corridas de esta serie, el SHA-256 de `outputs/models/*`, de `outputs/training_v17.sqlite3` y de `outputs/bench/reports/*` no cambió (informe de la tarea 8, `sha256sum -c` antes y después; notas de trabajo no versionadas).

### Límites que quedan

- **Otros cargadores de modelos no pasan por `verify_promotion`.** Siguen cargando sin verificar (registro de decisiones R8; números de línea al momento de la revisión de la tarea 7):
  - `ScenarioPredictor` en `main.py` (líneas 614, 702 y 831 en ese momento);
  - el análisis SHAP (`main.py`, línea 522: `joblib.load` directo);
  - `predict_from_inp` a través de `ml/train.py` (`load_saved_model_artifact`), que usa también el mapa ML de la aplicación de escritorio (`generate_ml_map`).
  El mapa de escritorio sí recibe `provenance.json`, pero el modelo que dibuja no se verifica contra `promotion.json`.
- **Un `promotion.json` o `manifest.json` malformado** produce un `JSONDecodeError` sin mensaje de dominio.
- **Granularidad gruesa:** `provenance.json` describe el modelo activo en el momento de escribir, por carpeta, no por figura.
- **Las figuras existentes no se regeneraron** con el modelo declarado (recomendación de la auditoría, P0 punto 4): siguen sin procedencia. El modelo final de la tesis aún no está fijado (Parte 2).
- **Sin `promotion.json`, la procedencia identifica archivos, no candidatos:** el SHA-256 de los artefactos no dice con qué `prep_id` ni con qué candidato del banco se entrenaron.

### Texto para la tesis

> Cada figura generada con el modelo de aprendizaje automático se acompaña de un registro de procedencia con la familia del modelo, la huella SHA-256 de los archivos del clasificador y del regresor, el conjunto de datos de entrenamiento y la fecha. Antes de predecir, el sistema comprueba que los archivos del modelo coinciden con el registro de promoción y se detiene si no es así, de modo que no puede atribuirse a un modelo una figura producida con otro.

(Si se usa este párrafo, conviene limitarlo a las figuras de la tesis que efectivamente se regeneren con el modelo final, y no aplicarlo a las rutas de predicción por escenario, SHAP o escritorio, que no están verificadas.)

---

## 4. Defecto encontrado durante el trabajo — pruebas que escribían procedencia falsa en `outputs/metrics`

### Problema y evidencia

Al añadir la procedencia en `main.py` (commit `0c8a3f4`), tres pruebas existentes que ejecutan `main.main()` empezaron a escribir `provenance.json` en el `outputs/metrics` **real** del repositorio: `tests/test_cli_sql_reads.py::test_resilience_curve_reads_the_frame_from_sql`, `::test_flood_volume_curve_reads_the_frame_from_sql` y `tests/test_factor_comparison_cli.py::test_factor_comparison_cli_generates_plots_in_metrics_directory`. Se detectó durante la corrida de la tarea 8: la suite completa sobrescribió `outputs/metrics/provenance.json` y `outputs/metrics/factor_comparison/provenance.json` (informe de la tarea 8). Esos archivos atribuían las figuras existentes al modelo de prueba: procedencia falsa.

### Causa

Las pruebas sustituían solo el cálculo y el dibujo, pero dejaban `main.METRICS_DIR` apuntando a la carpeta real, y nada impedía que una prueba escribiera allí.

### Qué cambió

Commit `398d6bc` (registro de decisiones R10):

- Las tres pruebas redirigen `main.METRICS_DIR` a una carpeta temporal; sus aserciones no cambian.
- `tests/conftest.py` añade un guardia automático: antes y después de cada prueba toma (tamaño, fecha de modificación) de cada archivo bajo el `outputs/metrics` real y falla si alguno se creó o cambió.
- Se borraron los dos `provenance.json` espurios (ignorados por git; no había versión legítima que restaurar).

### Verificación

| Comprobación | Resultado | Fuente |
|---|---|---|
| Suite con el guardia y sin la corrección | 1145 pasan; 3 errores en el desmontaje, exactamente las tres pruebas culpables | Informe de la tarea 7, segunda ronda (notas de trabajo no versionadas) |
| Suite con el guardia y la corrección | 1145 pasan, 0 errores; los dos archivos no reaparecen | ídem; registro de decisiones R10 |
| Suite completa al cierre de la serie | Ver `04` §7 | Comando de `04` §7 |

### Límites que quedan

- El guardia vigila solo `outputs/metrics`, no `outputs/models` ni `outputs/bench`.
- El guardia no tiene una prueba propia que demuestre que detecta una escritura.
- Quedó una carpeta vacía `outputs/metrics/factor_comparison/`.

### Texto para la tesis

No requiere texto en la tesis. Es un defecto del entorno de pruebas, corregido antes de generar ninguna cifra o figura de la tesis.
