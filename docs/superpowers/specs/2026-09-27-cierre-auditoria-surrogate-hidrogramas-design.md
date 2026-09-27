# Cierre de la auditoría del surrogate de hidrogramas

**Estado:** Propuesta para revisión; implementación pendiente de aprobación.

**Fecha:** 2026-09-27  
**Documento de origen:** `AUDITORIA_SURROGATE_HIDROGRAMAS.md`

## 1. Objetivo y alcance

Esta especificación define cómo cerrar los pendientes verificables de la auditoría sobre la red actual.
Conserva los logros ya implementados y separa con claridad correcciones de código, pruebas y evidencia experimental.
Una corrección queda demostrada primero por contrato y pruebas; un resultado científico solo existe después de ejecutar el experimento correspondiente.

La coordinación, las preguntas, las decisiones difíciles y el diseño corresponden a Astra.
La escritura, implementación, pruebas y ejecuciones corresponden a Sol.
Luna podrá intervenir únicamente en tareas mecánicas delimitadas si se decide usarla.
No se cambiará la configuración global de modelos como efecto lateral de este trabajo.
No habrá commits, push, promoción de modelos ni reemplazo de artefactos activos automáticos.
El árbol de trabajo está sucio y todas las modificaciones locales existentes deben preservarse.
El usuario autorizó cómputo por etapas, sujeto a observar primero el costo del piloto antes de confirmar corridas largas.

Quedan fuera del alcance la ampliación general de la persistencia completa y el retiro total del pipeline legacy.
Los planes existentes sobre esas capas no se reabren aquí.
Sí se implementará la trazabilidad mínima necesaria para que los experimentos de esta auditoría sean reproducibles y comparables.

## 2. Evidencia comprobada

La base `outputs/training_v17.sqlite3` contiene 160.000 muestras:

- 40 formas de hidrograma;
- 25 factores por forma;
- 160 nodos por corrida;
- 1.000 corridas sobre una red.

Cada `shape_id` aporta exactamente 4.000 filas y no hay identificadores de forma nulos.
El PreparedDataset disponible bajo `outputs/bench/prepared/` tiene 76.000 filas, 19 formas y 475 corridas.
Los reportes existentes fueron producidos en momentos distintos y el ranking actual contiene solo SVM.
Esos artefactos son históricos y no prueban una comparación conjunta de las cinco familias sobre el dataset actual.

El banco vigente entrega `factor_mult` como grupo tanto a LOSO como a GroupKFold5.
Por ello, LOSO actual deja fuera una intensidad y mantiene todas las formas en train y test.
No mide generalización a una forma temporal desconocida.

El contrato `tabular_v3_17` contiene 17 features.
Solo `duracion_horas` y `tiempo_al_pico_h` describen directamente el tiempo del evento.
En entrenamiento esos valores son escalares globales de la forma.
En inferencia `scenario_predict.py` toma el tiempo al pico de la primera serie no vacía, lo que no garantiza la misma semántica.

La regla canónica de inundación vigente es `volumen_m3 >= 1.0`.
El extractor y sus pruebas aplican la comparación inclusiva.
Los 1.902 casos observados por `quality.py` tienen volumen exactamente igual a 1 m³ y son falsos desacuerdos producidos por usar `>` en ese reporte.
No se reetiquetarán retrospectivamente muestras correctas para ocultar ese defecto.

Tres formas generadas, `long_moderate`, `shape_2h` y `shape_5h`, no terminan con caudal cero.
El writer actual agrega un único cero seis horas después del último punto.
Eso interpola una cola de aporte artificial durante esas seis horas.
Los siete CSV independientes de `data/hydrograph_validation/` sí terminan en cero y no deben confundirse con esas tres formas de entrenamiento.
No se afirmará que todos los escenarios actuales estén afectados sin revisar la serie efectiva de cada corrida.

Los hashes y fechas de los modelos activos, el manifiesto de promoción y la evidencia SQL no coinciden plenamente.
Se registrará esta inconsistencia sin atribuir a los modelos un entrenamiento que no consta.
No se fabricará una promoción histórica para reconciliar fechas.

## 3. Relación con diseños anteriores

La spec `2026-06-26-surrogate-shape-aware-features-design.md` y el plan del 2026-06-27 introdujeron las 17 features actuales.
Su decisión de descartar el volumen del evento quedó superada por la evidencia física y experimental posterior.
Se conserva su aporte de duración y tiempo al pico como contrato legacy, no como cierre suficiente de C1.

La spec del banco del 2026-09-05 fijó ambos protocolos sobre grupos de factor.
El nuevo protocolo por forma es una extensión incompatible en semántica, no una reinterpretación silenciosa de sus resultados.

La spec y el plan Optuna del 2026-08-04 siguen pendientes.
Además, asumen grupos por factor y XGBoost como candidato.
No deben ejecutarse tal cual; la optimización se rediseñará después del benchmark por forma.

El backfill desde CSV puede conservar tiempos no medidos como cero.
Ese cero no se interpretará como una observación física.
La migración mínima debe representarlo como nulo o `not_measured`, sin inventar mediciones retrospectivas.

## 4. Alternativas consideradas

La alternativa recomendada corrige primero integridad, protocolo, aporte y drenaje; después agrega features compartidas y compara modelos con los mismos folds.
Este orden permite atribuir cada cambio y evita entrenar sobre una verdad hidráulica que luego deba regenerarse.

Corregir únicamente la validación cruzada produciría una estimación más honesta, pero dejaría pendiente la sensibilidad física del modelo a volumen, excedencia y forma tributaria.

Ir directamente a CNN o LSTM aumentaría el costo y la superficie de depuración sin evidencia de que el tabular enriquecido sea insuficiente.
Las redes temporales quedan condicionadas a una brecha residual medida después de las fases anteriores.

## 5. Fase A: integridad y protocolo

La regla `volumen_m3 >= 1.0` se centralizará y será usada por extracción, calidad, entrenamiento, inferencia y pruebas.
El umbral y el operador inclusivo formarán parte del manifiesto.

`LOSO` conservará su semántica legacy por factor para que los resultados históricos sigan interpretables.
Se añadirá el protocolo explícito `LeaveOneShapeOut`, con `group_column=shape_id`, como protocolo principal.
Los protocolos por factor serán diagnósticos secundarios y opcionales.

En el dataset original completo, cada fold por forma tendría en test las 25 intensidades y los 160 nodos de una sola forma.
La regla general será retener todas las filas elegibles del `shape_id`, no imponer esos conteos después de aplicar controles de calidad hidráulica.
Las exclusiones y sus motivos se registrarán, serán comunes a todas las familias y obligarán a recalcular y reportar la cobertura real por fold.
Nunca se declarará cobertura completa cuando se hayan excluido escenarios.
Ningún `shape_id` de test podrá aparecer en train.
Cada muestra aparecerá exactamente una vez en OOF para ese protocolo.
`shape_id` será una clave de agrupación y procedencia, nunca una feature del modelo.

Se persistirán la definición del split y hashes SHA-256 del dataset, contrato, configuración, semilla y código de protocolo.
La procedencia incluirá el commit y un snapshot del código con hash SHA-256; si el worktree está sucio, incluirá además un hash del diff y las diferencias relevantes necesarias para reproducir la corrida.
Ese registro se limitará al repositorio, excluirá secretos y no incorporará otras carpetas del usuario.
El manifiesto incluirá la columna de agrupación y los identificadores retenidos por fold.
La carga de un cache validará su semántica, no solo la presencia de archivos.
Un PreparedDataset obsoleto o construido con otra agrupación será rechazado con un error explícito.

## 6. Fase B: aporte, drenaje y features coherentes

El fin del aporte y el fin de simulación serán conceptos diferentes.
El colchón de drenaje no se usará para cerrar gradualmente un aporte que quedó positivo.

Para una forma generada cuyo endpoint supere la tolerancia, una política explícita y versionada cerrará linealmente hasta cero en el siguiente paso de la malla.
Después mantendrá cero durante el drenaje.
La corrida registrará la fuente, la serie efectiva enviada a SWMM, su duración final y una advertencia del cambio.
La verdad SWMM se regenerará para todo escenario cuya serie efectiva haya cambiado.

Para CSV aportados por el usuario habrá dos modos explícitos: rechazo estricto o cierre con la misma política versionada.
Nunca se alterará una serie del usuario en silencio.
La tolerancia de fin de aporte se expresará en L/s y su valor aparecerá en todo manifiesto de corrida.
El default estricto exigirá caudal final cero a la precisión de entrada.

Las tolerancias hidráulicas se definirán y congelarán en un piloto anterior a la comparación de familias.
Se justificarán con observables y estado basal.
Si una decisión física no está respaldada por evidencia, requerirá aprobación específica.
Hasta entonces el drenaje será `unverified`, M3 seguirá abierto y esa cohorte no se presentará como validación confirmatoria.

Los descriptores se calcularán sobre la serie efectivamente enviada a SWMM y excluirán el tramo de ceros de drenaje.
Entrenamiento e inferencia usarán exactamente la misma función y el mismo validador.
El nuevo contrato será deliberadamente incompatible con `tabular_v3_17`; los modelos antiguos se preservarán.

Las features nuevas por nodo serán:

- volumen local integrado, en m³;
- volumen tributario integrado, en m³;
- duración efectiva del caudal tributario por encima del 1 % de su pico;
- tiempo al primer máximo tributario, con cero para una serie nula;
- tiempo sobre `upstream_capacity_lps`;
- integral de `max(Q_tributario - capacidad, 0)`, en m³.

`Q_tributario` sumará las series del nodo y de sus ancestros únicos sobre una misma malla.
La interpolación será lineal y explícita.
Nunca se sustituirá la suma temporal por una suma de máximos.
Las integrales usarán cortes exactos en los cruces de umbral por segmento.
La conversión de una integral en L/s·h a m³ será multiplicar por 3,6.

Una capacidad faltante o menor o igual a cero se marcará inválida; no se harán divisiones artificiales.
El contrato nuevo admitirá faltantes solo en variables de capacidad marcadas explícitamente.
El imputador se ajustará únicamente con train y el reporte de calidad registrará los flags correspondientes.

`q_pico_acum_escalado` se conservará y documentará como proxy.
No se rebautizará como caudal enrutado porque ignora desfase, atenuación y routing.
Duración y tiempo al pico actuales permanecerán únicamente en el contrato legacy.
El contrato nuevo tendrá semántica por nodo inequívoca.
Descriptores adicionales de multipico o recesión se añadirán solo si una ablación demuestra su necesidad.

El nuevo dataset tendrá identificador y hashes propios y se regenerará desde las series efectivas.
Nunca se mezclarán labels obtenidos con una hidráulica anterior ni se sobrescribirá el histórico.

## 7. Fase C: verdad SWMM y elegibilidad

La última ventana se verificará con observables de `.out`: flooding, total/lateral inflow, depth, volumen en nodos y enlaces cuando esté disponible, y flujo final de enlaces frente a su estado basal.
También se guardarán continuidad de runoff y routing, unidades y el `.rpt` como apoyo.

Cada escenario tendrá estado `drained`, `not_drained` o `unverified`.
Ni `END_TIME + 6 h` ni profundidad cero aislada demostrarán drenaje.

La regla operativa inicial usará la última ventana de `min(1 h, tiempo de drenaje)` y al menos dos reportes.
Exigirá ausencia de aporte e inundación sobre tolerancias y cambio estabilizado de almacenamiento.
Exigirá además que el flujo residual de nodos y enlaces respecto al estado basal quede dentro de una tolerancia configurada y congelada en el piloto.
Si `.out` solo ofrece volumen nodal, no se declarará estabilidad ni vaciado total de la red.
Las tolerancias físicas serán explícitas en configuración y manifiesto.
Si no están fijadas o faltan observables, el estado será `unverified`.
Profundidad o volumen remanentes se informarán, pero por sí solos no implicarán fallo.
`drained` significará cumplimiento del criterio operativo de estabilidad, ausencia de aporte e inundación; no significará profundidad cero ni una red completamente vacía.

Un error absoluto de continuidad de routing superior a 5 % marcará la referencia como cuestionada.
Ese escenario saldrá del benchmark principal y aparecerá en un reporte separado.
Todas las familias usarán exactamente el mismo conjunto elegible.
Si falta o se rechaza un escenario, no se presentará una comparación final completa hasta re-simularlo o declarar la limitación.

No se extenderá una simulación indefinidamente.
Las extensiones repetidas tendrán incremento y límite configurables, ambos registrados.

## 8. Fase D: benchmark justo

Se compararán XGBoost, Random Forest, lineal, SVM y MLP con el mismo dataset, features, folds, umbral y semilla.
El baseline de 17 features se recreará sobre las mismas corridas hidráulicas válidas y los mismos folds.
Si el costo lo exige, la ablación 17 versus contrato nuevo se hará primero con XGBoost antes de ampliar a cinco familias.

Se persistirán predicciones OOF y métricas por forma y escenario.
Clasificación incluirá CSI, PR-AUC, precisión, recall y F1.
Volumen incluirá RMSE y MAE sobre todos los nodos y sobre nodos inundados.
El regresor oracle se mantendrá separado del resultado end-to-end.

Por escenario se informarán volumen real, predicho, error absoluto y error relativo.
Si el volumen real es cero, el error relativo será NaN y no se añadirá un epsilon engañoso.
Se reportarán resultados pooled y macro por forma: media, desviación, mediana y peor forma.
El ranking ordenará por menor macro-RMSE end-to-end por forma, luego mayor CSI macro y después menor media del error absoluto de volumen total por escenario.
Solo ante empate numérico exacto en esos tres criterios se usará `candidate_id` en orden lexicográfico.

Los tiempos separarán carga, construcción de features e inferencia.
Hardware, hilos y dispositivo quedarán registrados.
CPU y GPU se reportarán por separado.

## 9. Ranking y promoción

El benchmark solo producirá ranking por defecto.
La promoción será una operación explícita, transaccional y con rollback a los artefactos previos.
Antes de promover verificará hashes de dataset, configuración, modelos, contrato, INP, protocolo, splits y cohorte del reporte.

Los guardrails conservadores iniciales exigirán:

- métrica primaria estrictamente mejor que el baseline comparable;
- CSI macro no menor;
- media del error absoluto de volumen total por escenario no mayor;
- todas las métricas requeridas finitas.

Si no se cumplen, no se promoverá.
Las tolerancias futuras deberán decidirse antes del experimento y no después de mirar los resultados.
El modelo activo podrá conservarse si ningún candidato mejora.
La inconsistencia histórica actual se documentará, no se reescribirá.

## 10. Reconciliación y extrapolación

Se conservarán probabilidad raw, clase raw y volumen raw no negativo.
El volumen final será cero cuando la clase raw sea negativa; en otro caso será la salida no negativa del regresor.
`inunda_pred` final será `volumen_final >= 1.0`.
No se inflará un volumen hasta 1 m³ para forzar consistencia.
Las métricas del clasificador raw y las reconciliadas end-to-end se distinguirán y la política será versionada.

Los límites de descriptores para OOF se aprenderán exclusivamente con train de cada fold.
En inferencia procederán del manifiesto del entrenamiento final.
Se reportará `temporal_descriptor_out_of_range`.
`shape_status` será `known` o `unseen` únicamente cuando exista identidad verificable de la forma; sin esa evidencia será `unknown`.
Los límites de descriptores serán un diagnóstico independiente y no determinarán `shape_status`.
Estar dentro de límites no certificará generalización.
Se mantendrá el indicador de extrapolación de magnitud.

Se documentará que `base_inflow_lps` representa un pico y que el proxy acumulado ignora routing, desfase, atenuación y cuellos de botella.
La capacidad mínima aguas abajo se pospone hasta contar con una formulación física validada.
Esta limitación documentada será el cierre aceptado de MO1 por ahora.

El parser de ponding será dirigido por encabezados y habrá una única extracción de labels.
Si la columna `Total Flood Volume` no puede identificarse inequívocamente, abortará con un diagnóstico; nunca tomará el último token como sustituto.
Se mantendrán pruebas de unidades, umbral exacto, nodos sin aporte, hash del INP y las demás fortalezas de regresión.
Los hallazgos menores ya cerrados se verificarán sin rehacerlos.

## 11. Ejecución por etapas

Primero se ejecutarán pruebas unitarias de contrato, folds, fórmulas y paridad, seguidas por un smoke SWMM pequeño.
Luego se hará un piloto de tiempos por familia con grupos completos, marcado como no comparativo y excluido del ranking.

El benchmark completo por forma requiere 40 folds × 5 familias × 2 modelos, es decir, 400 ajustes.
Tendrá checkpoints reanudables cuya identidad se validará antes de continuar.
El trabajo será recuperable por fold.

SVM RBF puede no ser viable con 156.000 filas de train por fold.
Tendrá timeout explícito y estado `not_run_resource_limit`.
No se cambiará de familia ni se submuestreará silenciosamente.
Si SVM falta, no se afirmará que las cinco familias fueron reevaluadas.
El presupuesto de la corrida costosa se confirmará con las cifras del piloto.

## 12. Optimización posterior

Optuna se abordará solo después de las fases anteriores y sobre la familia candidata elegida por el benchmark fijo.
El inner GroupKFold agrupará por forma y verá exclusivamente el train del fold exterior.
Preprocesamiento, early stopping, pruning y selección de umbral nunca usarán outer test.
La comparación contra la configuración fija reutilizará exactamente los mismos outer folds.

Cada estudio registrará trials, tiempo, semilla, espacio, parámetros y resultados.
Elegir familia y evaluarla sobre los mismos OOF introduce sesgo de selección.
Una afirmación confirmatoria requerirá un holdout de formas no usado para seleccionar ni ajustar, o selección de familia anidada.
Sin esa separación, el resultado se etiquetará como estimación exploratoria condicionada.

CNN o LSTM solo se considerarán si queda una brecha residual medida.
Naive Bayes podrá añadirse como baseline de clasificación separado, nunca como sustituto del surrogate de volumen.

## 13. Matriz de aceptación

La tabla de cierre distinguirá cuatro estados: código implementado, pruebas aprobadas, corrida ejecutada y evidencia suficiente.
Ningún hallazgo se marcará resuelto solo porque exista código.

| Hallazgo | Criterio de aceptación |
|---|---|
| C1 | Contrato nuevo compartido, ablación ejecutada y sensibilidad por forma reportada. |
| C2 | `LeaveOneShapeOut` sin intersección de formas y OOF completo. |
| M1 | Paridad exacta de features entre entrenamiento e inferencia. |
| M2 | Extrapolación de magnitud y descriptores reportada con límites solo de train. |
| M3 | Política de cierre aplicada, escenarios afectados regenerados y drenaje verificable. |
| M4 | Todos los nodos elegibles, incluidos los que no tienen aporte directo. |
| M5 | Regla inclusiva única y reconciliación versionada. |
| MO1 | Limitación del proxy documentada; capacidad aguas abajo explícitamente pospuesta. |
| MO2 | Métricas pooled, macro, por forma y por escenario persistidas. |
| MO3 | Unidades verificadas y registradas. |
| MO4 | `base_inflow_lps` documentado como pico. |
| MO5 | Volúmenes totales y errores por escenario disponibles. |
| MO6 | Carga, features e inferencia medidas por separado. |
| Menores | Parser, continuidad, hash, nodos y conciliación cubiertos por regresión. |

## 14. Mapa de archivos

Los puntos existentes que concentran los cambios son:

- `swmm_resilience/ml/bench/schemas.py`, `folds.py`, `preprocess.py`, `evaluate.py`, `metrics.py`, `ranking.py`, `reports.py` y `promote.py`;
- `swmm_resilience/ml/contracts.py`, `scenario_predict.py` y `predict.py`;
- `swmm_resilience/extraction/dynamic_features.py` y `labels.py`;
- `swmm_resilience/dataset/assembler.py` y `validator.py`;
- `swmm_resilience/simulation/timeseries_scenario.py`, `hydrograph_shapes.py`, `swmm_api_io.py` y `runner.py`;
- `swmm_resilience/validation/hydrograph_csv.py` y `hydrograph_batch.py`;
- `swmm_resilience/database/csv_backfill.py` y `training_queries.py`;
- `swmm_resilience/config.py`, `config.yaml` y `main.py`.

El módulo nuevo propuesto para las fórmulas compartidas es `swmm_resilience/extraction/event_features.py`.
Su única responsabilidad será transformar series efectivas y topología en descriptores reproducibles; no entrenará modelos ni ejecutará SWMM.

Las pruebas se extenderán en `tests/ml/bench/`, `tests/ml/`, `tests/simulation/`, `tests/database/`, `tests/test_scenario_predict.py`, `tests/test_timeseries_scenario.py`, `tests/test_hydrograph_batch.py` y `tests/test_labels.py`.
La implementación futura deberá preservar los nombres y responsabilidades de los archivos existentes salvo que un plan posterior justifique una migración explícita.
