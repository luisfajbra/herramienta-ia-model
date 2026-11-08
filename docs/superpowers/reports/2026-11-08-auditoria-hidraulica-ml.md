# Auditoría hidráulica y de ML: del 4 al 8 de noviembre de 2026

## Estado (actualizado)

Actualización del 8 de noviembre de 2026, rama `fix/test-suite-contracts-2026-10-27`, sin push. El detalle por hallazgo (problema, plan, solución y pendientes) está en el [documento de resolución](2026-11-08-resolucion-hallazgos-auditoria.md).

- **1, 2, 4 y 5:** atendidos con análisis post hoc (`scripts/auditar_reserva_conjunta.py`) y documentados en el [informe de la reserva](2026-11-08-evaluacion-reserva-conjunta.md#sensibilidad-post-hoc). El reentrenamiento queda pendiente de decisión (ver "Decisiones pendientes" en el [plan](../plans/2026-11-08-resolver-hallazgos-auditoria.md)).
- **3:** **fuera de alcance por decisión del investigador (8 de noviembre)**. Se considera parte del diseño de la red; se mantiene como está.
- **6:** respaldo local creado y verificado (`33_reserva_conjunta_y_base_20261108.zip`, SHA-256 `2f1ea8857d45e5fcefa6eb2faad435b35ffb2f869f8168ea163fe78b049b4022`). **La subida a Drive está pendiente**: la debe hacer el investigador y confirmarla. Hasta entonces el hallazgo 6 no se considera resuelto.
- **7:** se retira el punto de `with sqlite3.connect(...)`. Al revisarlo, queda rebajado a observación: en código dentro de una función la conexión se cierra al reasignar la variable o al terminar la función, y solo en `scripts/train_joint_final_20261108.py:88` la última conexión de instantánea (la de `mlp`) queda abierta hasta que `main` termina, es decir, durante todo el entrenamiento. Es inocuo (el bloque `with` ya hizo el commit y no hay transacción pendiente) y no se cambia el script, porque ya fue ejecutado y está congelado. Los demás puntos ya estaban documentados. La longitud de ruta de los tests se documenta en el [README](../../../README.md#verificacion).
- **Presentación:** la diapositiva "Hallazgos y próximos pasos" de `scripts/build_weekly_2026_11_08_joint_deck.py` (línea 182) todavía dice `'MLP logró menor RMSE de volumen en esta reserva.'`, que conserva la afirmación de una sola forma del hallazgo 1. No se editó el script ni se regeneró el `.pptx`; hace falta la aprobación del investigador. El texto propuesto es `'MLP logró menor RMSE macro, pero por una sola forma; sin ella gana XGBoost.'`.

**Alcance:** commits `47848f3c` a `d4e1e6a7` de la rama `fix/test-suite-contracts-2026-10-27`, es decir, todo lo posterior a la revisión anterior (`8d1e1ceb`, [informe](2026-11-04-revision-rama-test-suite-contracts.md)). Cubre la base conjunta Chico Sur + Chico Norte, la partición por formas, el tuning recuperable, el ajuste final, la evaluación de la reserva y el diagnóstico hidráulico.

**Método:** lectura del código nuevo y comprobaciones independientes en solo lectura sobre los artefactos sellados:

- `outputs/experiments/joint_chico_sur_norte_20261104_v1`
- `outputs/experiments/evaluacion_reserva_conjunta_20261108_v1`
- los INP, RPT y CSV enlazados de las 1.975 corridas.

No se modificó código, ni modelos, ni resultados. Los scripts de comprobación quedaron fuera del repositorio. Las cifras de este informe se pueden reproducir con las consultas descritas en cada sección.

## 1. Resumen

| # | Hallazgo | Tipo | Gravedad |
|---|---|---|---|
| 1 | La conclusión "MLP tiene menor error de volumen" depende de una sola forma; sin ella el orden se invierte | ML, interpretación | **Alta** |
| 2 | El sorteo dejó en la reserva las dos formas con más volumen de entrada; la de bloque supera en 16 % a cualquier forma de desarrollo | ML, cobertura | **Alta** (probablemente explica la brecha entre CV y reserva) |
| 3 | La cola de drenaje es una rampa lineal cuando el CSV no termina en cero; cuatro formas antiguas reciben de 16 % a 41 % de volumen extra y nunca drenan | Hidráulica, generación de escenarios | **Media** (la QA las excluye, pero se pierden 175 corridas y el motivo registrado es engañoso) |
| 4 | El MLP extrapola de forma exponencial en m³ (hasta 3 veces el valor de SWMM en un nodo) y su volumen total baja entre los factores 4,8 y 5,0 | ML, física | Media |
| 5 | El límite de continuidad del 5 % es laxo para etiquetas pequeñas | Hidráulica, QA | Baja |
| 6 | Los artefactos sellados de la reserva no están versionados ni tienen respaldo documentado | Reproducibilidad | Media |
| 7 | Detalles menores de código | Código | Baja |

**Verificado correcto** (sección 8): las etiquetas cuadran con el balance de SWMM en las 1.653 corridas elegibles, las entradas CSV cuadran con el "External Inflow" de SWMM, no hay fuga entre la reserva y el desarrollo, la cadena de sellos se respeta, la transformación `log1p`/`expm1` es coherente y la corrección de tiempos del OUT es correcta y no afecta resultados previos.

## 2. Hallazgo 1: la ventaja del MLP depende de una sola forma

**Qué se ve.** El [informe de la reserva](2026-11-08-evaluacion-reserva-conjunta.md) reporta un RMSE macro de 234,54 m³ para XGBoost y 184,63 m³ para MLP. Concluye que "en esta reserva MLP tiene menor error de volumen". Recalculé esas cifras desde las predicciones selladas y coinciden. Pero:

| RMSE macro (m³) | Todas las formas | Sin `shape_quasi_uniform_block` | Mediana por forma y red |
|---|---:|---:|---:|
| XGBoost | 234,54 | **66,68** | 40,81 |
| MLP | 184,63 | **87,23** | 29,29 |

RMSE por forma y red (m³):

| Forma | XGB Sur | XGB Norte | MLP Sur | MLP Norte |
|---|---:|---:|---:|---:|
| common_storm | 49,3 | 48,0 | 39,5 | 35,3 |
| rare_extreme | 2,7 | 5,8 | 4,0 | 4,4 |
| shape_delayed_abrupt_peak | 40,8 | 40,8 | 28,1 | 27,2 |
| shape_flashy_long_tail_9h5 | 20,2 | 23,1 | 25,5 | 23,1 |
| shape_front_loaded_broad | 24,1 | 21,3 | 29,9 | 28,7 |
| **shape_quasi_uniform_block** | **583,8** | **563,0** | **398,2** | **443,0** |
| shape_twin_plateau_15h | 147,0 | 130,0 | 183,2 | 203,9 |

**Por qué importa.** El RMSE macro es una suma de errores cuadrados, y una forma concentra el 93 % del error de XGBoost y el 81 % del de MLP. Por eso la métrica principal mide casi solo esa forma. Sin ella, XGBoost gana. Por forma, el resultado es mixto: MLP gana en `common_storm` y `delayed_abrupt_peak`, y XGBoost gana en `twin_plateau_15h`, `front_loaded_broad` y `flashy_long_tail`. En la otra forma de gran volumen (`twin_plateau_15h`), XGBoost es mejor que MLP. Esto contradice la idea de que MLP "extrapola mejor" en general.

**Recomendación.** No cambiar la métrica principal acordada, porque eso sería un ajuste post hoc. Sí conviene reportar junto a ella esta tabla y aclarar que el orden depende de una forma. La frase de conclusión debería decir que MLP tiene menor RMSE macro en esta reserva, que eso se debe a `shape_quasi_uniform_block` y que en las otras seis formas el resultado está repartido.

## 3. Hallazgo 2: el sorteo dejó lo más extremo fuera del entrenamiento

**Qué se ve.** Ordené las corridas elegibles de factor 5 por "External Inflow" del RPT:

| Red | 1.º | 2.º | 3.º |
|---|---|---|---|
| Chico Sur | quasi_uniform_block 262.697 m³ (**reserva**) | sustained_plateau_14h 225.640 m³ (desarrollo) | twin_plateau_15h 223.169 m³ (**reserva**) |
| Chico Norte | quasi_uniform_block 780.411 m³ (**reserva**) | sustained_plateau_14h 670.288 m³ (desarrollo) | twin_plateau_15h 662.949 m³ (**reserva**) |

El sorteo `random.Random(42).sample` es correcto y estaba fijado de antemano. Por azar, tocó la forma de mayor volumen de toda la población y la tercera. En factor 5, la forma de bloque introduce 16 % más agua que cualquier evento de desarrollo, y en esas corridas cerca del 70 % de la entrada se pierde como inundación. La red trabaja limitada por la capacidad de su salida.

**Matiz importante.** El problema no es solo que la etiqueta salga de rango. Solo 18 pares nodo–corrida de la reserva superan la etiqueta máxima de desarrollo (6.562 m³ en Sur y 12.639 m³ en Norte), y explican el 5,8 % del error de XGBoost. En esa forma, XGBoost predice solo entre el 28 % y el 49 % del volumen total de SWMM desde el factor 1,4 ([volumen_por_factor.csv](../../figures_reserva_diagnostico_20261108/volumen_por_factor.csv)), donde las variables siguen dentro del rango. El [diagnóstico](2026-11-08-diagnostico-hidraulico-cobertura.md) ya advertía esto. Una hipótesis que vale la pena probar es que las variables no capturan bien la duración sostenida sobre la capacidad. En una red saturada, el volumen inundado es aproximadamente la entrada menos la capacidad de salida por la duración, y eso depende más de la duración que del pico.

**Consistencia con la CV interna.** Sin esa forma, el RMSE macro de la reserva (66,7 y 87,2 m³) queda cerca del de la selección interna (58,9 y 74,7 m³). Esta comparación es post hoc, pero sugiere que la brecha de cuatro veces entre CV y reserva se debe sobre todo a esa forma y no a un sobreajuste general.

**Recomendación.** En un próximo diseño, revisar la cobertura de volumen de entrada antes de sortear, por ejemplo estratificando por volumen total, o declarar explícitamente que la reserva mide extrapolación. La reserva actual ya fue consultada y no debe reutilizarse como prueba ciega.

## 4. Hallazgo 3: la cola de drenaje es una rampa cuando el CSV no termina en cero

**Dónde.** [timeseries_scenario.py:122-124](../../../swmm_resilience/simulation/timeseries_scenario.py#L122-L124):

```python
data = list(scenario.node_series[node_id])
if drain_down_hours > 0:
    data.append((scenario.last_time_hours + drain_down_hours, 0.0))
```

**Por qué falla.** SWMM interpola linealmente entre los puntos de un `[TIMESERIES]`. Si el último valor del CSV es cero, el punto agregado no tiene efecto y se obtiene el drenaje buscado. Si el CSV termina con caudal distinto de cero, el punto agregado crea una rampa de entrada desde ese valor hasta cero a lo largo de todo el período de drenaje. El docstring dice que se agrega "zero-flow ... so the network can drain", pero en ese caso no hay drenaje. Además, el bucle de reintentos de [batch.py:113-160](../../../swmm_resilience/simulation/batch.py#L113-L160) y de `frozen_pair_runner.py` alarga `drain_down_hours`, lo que solo alarga la rampa. Nunca converge.

**Evidencia.** Comparé el volumen integrado del CSV de cada corrida con el "External Inflow" del bloque "Flow Routing Continuity" de su RPT:

| Grupo | Corridas | Exceso de entrada en SWMM frente al CSV |
|---|---:|---|
| Elegibles | 1.653 | media 0,00 %, máximo 0,18 % (redondeo) |
| Excluidas, CSV termina en cero | 147 | media 0,00 % |
| Excluidas, CSV termina > 1 % del pico | **175** | **16 % a 41 %** |

Las 175 corridas son de las formas `base` (solo en Sur), `long_moderate`, `shape_2h` y `shape_5h`, en todos sus factores. Por ejemplo, la corrida 13 (`base`, Sur) termina el CSV en 2,32 L/s a las 3:00, y el INP lleva el cero a las 13:00. Todas quedan como `drainage_not_drained` con volumen almacenado final distinto de cero. `write_scenario_inp` ya emite un `warnings.warn` en este caso, pero no se detiene.

**Impacto.**

- En los resultados actuales: **ninguno**. La QA excluye las 175 corridas y ninguna entra en desarrollo ni en la reserva. Las features (integral del CSV) coinciden con lo que SWMM recibe en todas las corridas elegibles.
- En la cobertura: se pierden cuatro formas completas, 8,9 % de la población, sin que el motivo registrado (`drainage_not_drained`) revele la causa.
- Riesgo latente: si alguien relaja la QA de drenaje, esas corridas tendrían etiquetas calculadas con 16 % a 41 % más agua que la que describen sus features.
- No revisé si estudios anteriores a la QA de drenaje (migración 009) usaron estas formas. Conviene comprobarlo antes de citar resultados históricos que las incluyan.

**Corrección propuesta (no aplicada).** Elegir una de estas opciones:

1. Agregar el cero un paso después del último punto del CSV (por ejemplo `last + 1 s`, o el paso de reporte), para que la entrada caiga a cero y el resto del período sea drenaje real.
2. Rechazar el escenario si el CSV no termina en cero, y que el autor del CSV agregue su recesión.

Cualquiera de las dos cambia la entrada que SWMM recibe en esas formas. Requiere decisión del investigador, una nueva simulación de esas 175 corridas, y debe registrarse como un cambio de protocolo. Si se elige la opción 1, la integral de las features y la del INP deben seguir coincidiendo. El cotejo actual (`scripts/cotejo_fuentes_bloque_20261108.py`) ya exige que la cola valga cero.

## 5. Hallazgo 4: el MLP extrapola de forma exponencial en m³

**Qué se ve.**

- Mayor predicción nodal del MLP: 17.440 m³ en Sur (nodo `52C`, corrida 699, con probabilidad exactamente 0,5) y 21.193 m³ en Norte (nodo `268I`, corrida 1675, donde SWMM da 6.814 m³). Las etiquetas máximas de la reserva son 7.418 y 14.928 m³.
- XGBoost, al ser un modelo de árboles, se satura (la predicción es constante por tramos y deja de crecer fuera del rango de entrenamiento), pero no queda acotado por la etiqueta máxima vista: sus máximos son 6.098 y 14.068 m³, y el de Chico Norte supera la etiqueta máxima de desarrollo (12.639 m³).
- El volumen total del MLP para `quasi_uniform_block` en Sur **baja** de 185.554 m³ (factor 4,8) a 182.383 m³ (factor 5,0), mientras SWMM sube. Es la única violación de monotonía en los totales, entre ambos modelos.
- A nivel nodal, la predicción baja al subir el factor (con SWMM sin bajar) en 63 de 101.972 pasos para MLP y en 1.242 para XGBoost (1,2 %, por los saltos de los árboles).

**Por qué.** El regresor aprende `log1p(volumen)`. Una red con activaciones ReLU extrapola de forma aproximadamente lineal en el espacio logarítmico, lo que equivale a un crecimiento exponencial en m³. Los árboles, en cambio, se saturan. Esto explica que MLP supere a SWMM en Norte a partir del factor 4,4 y que XGBoost se quede corto. Ninguno de los dos modelos tiene una garantía física de monotonía respecto al factor.

**Recomendación.** Documentarlo como limitación. Si se reentrena, se pueden evaluar restricciones de monotonía (XGBoost admite `monotone_constraints` en variables de volumen) o un límite físico superior: el volumen inundado de un nodo no puede superar su volumen tributario de entrada.

## 6. Hallazgo 5: el límite de continuidad es del 5 %

La QA admite corridas con error de continuidad de hasta 5 % (máximo admitido: 4,98 % en Sur y 4,99 % en Norte; mediana 0,40 % y 0,73 %). Hay 99 elegibles en Sur y 151 en Norte por encima de 2 %. En 119 corridas elegibles con inundación (101 de desarrollo y 18 de la reserva), el volumen de error de continuidad supera la mitad del volumen inundado. Son corridas pequeñas: mediana de 66 m³ inundados, factores de 1,2 a 2,6, y 0,05 % del volumen inundado total.

**Impacto:** casi nulo sobre el RMSE, pero no despreciable cerca del umbral de 1 m³, que coincide con la resolución del RPT (0,001 × 10⁶ L). Las exclusiones por continuidad se concentran además en factores bajos, sobre todo en Chico Norte (63 de 113 en factor ≤ 1). **Recomendación:** si se cita la clasificación cerca del umbral, agregar un análisis de sensibilidad con el límite en 2 %.

Las condiciones iniciales también pesan. El volumen almacenado inicial es constante (855 m³ en Sur y 4.048 m³ en Norte), porque todos los conductos tienen `InitFlow > 0`. Representa una mediana de 3 % a 4 % de la entrada y llega a 59 % en eventos pequeños. Esto ya está documentado como un origen no aclarado de los caudales iniciales de Chico Norte, y aplica también a Chico Sur.

## 7. Hallazgo 6 (reproducibilidad) y hallazgo 7 (código menor)

**Hallazgo 6: respaldo.** `outputs/*` está en `.gitignore`. El paquete conjunto (247 MB) y la evaluación sellada de la reserva (6,7 MB, con sello, predicciones, etiquetas y scores) no tienen respaldo documentado en Drive. Los estudios, XGBoost y MLP sí lo tienen (`27_`, `29_` y `32_`). Como la reserva ya fue consultada y no puede repetirse a ciegas, perder `evaluacion_reserva_conjunta_20261108_v1` sería irrecuperable en sentido científico. **Recomendación:** respaldarla junto con su SHA-256.

**Hallazgo 7: detalles de código** (ninguno cambia resultados):

- `scripts/prepare_joint_dataset.py:35` y `scripts/train_joint_final_20261108.py:88` usan `with sqlite3.connect(...) as db`. En Python, eso no cierra la conexión, solo hace commit o rollback. Las lecturas quedan con el archivo abierto hasta que termina el proceso. Es inocuo en un script, pero en Windows impide borrar o mover el archivo mientras tanto.
- `scripts/cotejo_fuentes_bloque_20261108.py:127` integra el OUT desde la primera muestra y no usa el `time_sec` corregido (que parte del inicio real). El informe lo declara ("omiten el primer intervalo"), así que no es un error, pero el script no aprovecha la corrección.
- Las etiquetas vienen de dos parsers de RPT (`strict_text_v1` en las corridas antiguas de Sur y `swmm_api.node_flooding_summary` en otras). El cuadre con el balance de la sección 8 indica que ambos producen lo mismo.
- `verify_joint_materialization` lee los manifiestos fuente por ruta absoluta, así que el paquete no se puede mover a otra carpeta sin ajustes. Ya está documentado.
- **Tests y longitud de ruta en Windows.** La suite completa (8 de noviembre) dio 1.706 aprobados y 1 fallido: `test_joint_builder_verifies_two_sources_and_publishes_new_dataset[False-joint]`, con `FileNotFoundError [WinError 3]` en `shutil.copy`. El test copia archivos con nombre `source-<sha256 de 64 caracteres>.bin`, y con un `--basetemp` largo (la carpeta temporal de la sesión, unos 110 caracteres) la ruta supera el límite de 260 caracteres de Windows. Con `--basetemp C:/ptj/a` pasan sus 6 casos. Es un problema del entorno, no del código, pero conviene usar un `--basetemp` corto o habilitar rutas largas en Windows (`LongPathsEnabled`).

## 8. Verificado correcto

- **Etiquetas frente al balance de SWMM.** En las 1.653 corridas elegibles, la suma de `vol_inundacion_m3` por corrida coincide con el "Flooding Loss" del bloque de continuidad del RPT. La diferencia máxima es 70 m³ en la corrida 1100, frente a una tolerancia de redondeo de 0,5 m³ por nodo. Ninguna corrida queda fuera de tolerancia.
- **Entradas frente a SWMM.** En todas las corridas elegibles, el volumen del CSV coincide con el "External Inflow" (diferencia de 0,18 % o menos). No hay volumen extra no descrito por las features.
- **Topología.** Ambas redes son árboles: 160 y 522 nodos, cada uno con un conducto, una descarga, y sin vertederos, orificios ni bombas. La variable tributaria por ancestros únicos es coherente con esa estructura.
- **Partición.** `development_run_ids` recalcula la pertenencia desde la QA y la forma, y exige que coincida con el JSON sellado. El entrenador comprueba que no haya formas reservadas. El evaluador comprueba que no haya solape entre corridas y que la cobertura nodal esté completa. `ShapeGroupKFold5` y `build_tuning_folds` usan el mismo `GroupKFold(shuffle=True, random_state=42)` sobre `shape_id`; scikit-learn 1.8.0 lo admite.
- **Orden de sellado.** El evaluador guarda y sella ambas predicciones antes de leer `node_results`, y verifica después que la base y la partición no cambiaron.
- **Transformación del volumen.** El regresor se entrena con `log1p`. `SeedEnsembleRegressor.predict` devuelve `log1p` de la media en m³, y el evaluador aplica `expm1` con recorte en cero. Es coherente.
- **Tuning recuperable.** El contrato inmutable incluye los hashes de todo el código `.py`. Un trial `RUNNING` interrumpido se marca como `FAIL` y cuenta en el presupuesto. El bloqueo es exclusivo vía SQLite. `select_best_trial` exige 50 trials `COMPLETE`.
- **Corrección de tiempos del OUT** (`a2d0bdda`). Es correcta. Su único consumidor es el script de cotejo, que pasa `simulation_start` desde el INP efectivo. No hubo consumidores antes, así que no afecta features, etiquetas ni métricas. El primer reporte es a las 00:03:00 porque SWMM reporta en múltiplos de `REPORT_STEP` desde `START_TIME`; `REPORT_START_TIME 00:00:05` solo filtra.
- **Materialización conjunta.** Verifica hashes antes y después, detecta WAL pendiente, comprueba que la QA coincida con la fuente, exige que la cobertura de features y etiquetas sea exactamente el inventario elegible, revisa claves foráneas, y revierte la escritura (savepoint) si falla.

## 9. Pendientes de la revisión anterior

Siguen igual que en el [informe del 4 de noviembre](2026-11-04-revision-rama-test-suite-contracts.md), sección 3: docstrings del filtro de continuidad, error poco claro en `preprocess` con una base sin migrar, presentaciones `.pptx` versionadas (167 MB en el árbol actual) y unas 185 carpetas `.pytest-*` y `tmp*` sin seguimiento en la raíz. Nada está publicado (sin push).

## 10. Orden sugerido

1. Corregir la redacción de la conclusión de la reserva (hallazgo 1) y añadir la tabla por forma. Solo cambia documentación.
2. Respaldar `evaluacion_reserva_conjunta_20261108_v1` y el paquete conjunto (hallazgo 6).
3. Decidir el tratamiento de la cola de drenaje (hallazgo 3) antes de cualquier nueva simulación. Si se corrige, volver a simular las 175 corridas como un protocolo nuevo.
4. Para un próximo entrenamiento: estratificar la reserva por volumen (hallazgo 2) y evaluar restricciones de monotonía o un límite físico (hallazgo 4).
