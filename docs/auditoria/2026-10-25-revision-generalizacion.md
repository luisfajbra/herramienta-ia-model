# Revisión de seguimiento: hidráulica, MLP/XGBoost y generalización entre redes

Fecha: **25 de octubre de 2026**. Versión revisada: worktree `optuna-tuning-familia-ganadora`, rama `worktree-optuna-tuning-familia-ganadora`, commit **78e551a**. La carpeta principal continúa en `6e8579b` con cambios locales; no representa el estado que el autor confirmó utilizar.

**Dictamen: todavía no está listo el flujo completo para entrenar y validar de forma fiable un modelo común a varias redes.** Sí hay una base útil para preparar un piloto. XGBoost y MLP son candidatos razonables para ese piloto por sus resultados actuales, pero esos resultados no demuestran transferencia a otras redes. Antes del entrenamiento conjunto hay que resolver los defectos de contrato/persistencia y definir el dominio hidráulico y la evaluación por red.

La segunda red no está en el proyecto. No se presupone su geometría, tipo de descarga, unidades, calidad de calibración, presencia de almacenamiento/bombas ni disponibilidad de hidrogramas. El objetivo confirmado por el autor es un modelo compartido que pueda predecir en más redes.

## Evidencia y límites

Se contrastaron la auditoría del 10 de octubre, sus cinco anexos, los documentos de correcciones del 11 de octubre y la selección de modelos del 13 de octubre, además del código y los artefactos actuales. **El informe posterior de correcciones es indispensable:** modifica algunas interpretaciones de la auditoría inicial, particularmente las relativas al redondeo y al umbral de inundación.

Trabajo realizado:

- Inspección de generación SWMM, features estáticas/dinámicas/tributarias, etiquetas, cierre/drenaje, SQL, preparación del banco, particiones, ajuste, inferencia y promoción.
- Consultas a la BD real con conexión `mode=ro`; integridad y FK, continuidad, cobertura y etiquetas.
- Recálculo independiente de métricas a partir de los OOF guardados de XGBoost y MLP ajustados, sin reentrenar esos modelos.
- **1.035 pruebas seleccionadas aprobadas**: 611 del banco y cadena hidráulica/inferencia; 424 de SQL, scripts, CLI, visualización y contratos; 3 pruebas de escala excluidas. No es una ejecución de toda la suite.
- **Ocho simulaciones nuevas**: seis controles sobre copias de Chico y dos sobre redes sintéticas de dos pozos para comprobar el efecto de la descarga.
- Reproducciones mínimas de pérdida de conductos paralelos, semántica de inflows, colisiones entre redes, metadatos incorrectos y aceptación de NaN como drenaje. Mutaciones de SQL únicamente sobre copia en memoria.
- Sin cambios al código funcional, a la configuración, al dataset canónico ni a los modelos activos. Los modelos diminutos de diagnóstico se guardaron en una carpeta separada.

Las primeras ejecuciones de pytest fallaron por configuración de temporales de esta auditoría y por permisos del sandbox de Windows. Se corrigió el destino temporal y se repitieron fuera del sandbox: los resultados válidos son `verified_tests.log` y `data_cli_tests.log`. Esos incidentes no se clasifican como defectos del proyecto.

No se repitieron las 1.000 simulaciones, el ajuste Optuna completo, las 26 mutaciones antiguas, las pruebas de escala/empaquetado ni la GUI interactiva. Tampoco se validó contra mediciones reales. Los hallazgos secundarios no revalidados se identifican en la [matriz de seguimiento](2026-10-25-matriz-seguimiento.md); no se declaran cerrados por ausencia de fallos en tests. Esta revisión no garantiza encontrar todos los defectos posibles.

## Resultados actuales que sí se confirmaron

La BD contiene **una red, 1.000 corridas, 40 formas, 160 nodos y 160.000 filas**. `integrity_check` devuelve `ok`, no hay violaciones de FK y todas las etiquetas coinciden con `vol_inundacion_m3 >= 1`. Los 1.000 errores de continuidad están informados; 37 superan 5 % en valor absoluto. El conjunto preparado `ca706e27654e0cf5` contiene **963 corridas y 154.080 filas**.

Los siguientes números se recalcularon sobre 154.080 predicciones únicas por modelo, en 40 folds LeaveOneShapeOut. El volumen del sistema es cero cuando el clasificador predice no inundación.

| Medida | XGBoost ajustado | MLP ajustado |
|---|---:|---:|
| RMSE agregado del volumen, todos los nodos | 69,6424 m³ | 101,1170 m³ |
| Sesgo del volumen total | −2,95 % | −5,67 % |
| TN / FP / FN / TP | 106.472 / 1.383 / 378 / 45.847 | 106.302 / 1.553 / 473 / 45.752 |
| Falsos negativos con volumen real ≥100 m³ | 0 | 4 |
| Error del volumen total por forma: mínimo / máximo | −31,22 % / +33,02 % | −26,29 % / +61,62 % |
| Porcentaje del SSE concentrado en `shape_quasi_uniform_block` | 61,02 % | 72,96 % |

Son cifras **exploratorias condicionadas al tuning sobre las mismas formas**, sobre una red simulada. El conjunto de 154.080 filas no equivale a 154.080 eventos independientes. La concentración del error en una forma y los falsos negativos volumétricamente grandes del MLP deben mantenerse visibles al comunicar el desempeño.

Los SHA-256 de los modelos activos coinciden con `promotion.json`: se ha corregido la inconsistencia concreta de archivos descrita en octubre. Sin embargo, **el par activo es XGBoost de 21 variables con 200 árboles, profundidad 6, learning rate 0,05 y subsample 0,8**, procedente de la migración del 6 de octubre. No es el candidato ajustado que obtiene 69,64 m³. Los OOF de la parte 2 son evidencia de evaluación, no dos pares finales listos para desplegar. La trazabilidad del par activo tampoco incluye `prep_id` en su registro actual.

## Hallazgos prioritarios de esta revisión

P0 significa bloqueo de la conclusión de generalización amplia o del experimento propuesto; P1 requiere corrección antes de un entrenamiento conjunto confiable; P2 es una limitación o mejora que debe quedar explícita. Los ejemplos de redes sintéticas no afirman que la red futura tenga esas propiedades.

### N-01 · P0 · Las 21 variables no distinguen condiciones de descarga hidráulicamente distintas

**Evidencia reproducida:** dos redes con las mismas cotas, profundidades, tuberías y aportes; solo cambia el outfall de `FREE` a `FIXED 11.5`. Las 21 features son exactamente iguales. SWMM predice 0 m³ en ambos pozos en el primer caso y **1.308 m³ en J2** en el segundo. Errores de continuidad: 0,000 % y −0,004 %, respectivamente.

Ubicación: `swmm_resilience/ml/contracts.py` (contrato v4_21), `extraction/static_features.py:35`, `extraction/topology.py:16`. No se representa la condición de borde de la descarga. Tampoco hay una representación completa del almacenamiento, bombeo o de la capacidad limitante aguas abajo.

**Consecuencia:** ningún MLP o XGBoost que reciba exclusivamente esas mismas variables puede producir dos respuestas distintas para estos dos casos. No se resuelve solo incorporando más filas. Hay que incorporar las variables necesarias y verificar su representación o restringir y validar el dominio admitido, por ejemplo, redes con determinadas condiciones de descarga. La onda dinámica de SWMM sí contempla remanso, sobrecarga y flujo inverso; no se deben eliminar esos mecanismos al definir el surrogate sin explicitarlo. [EPA: capacidades hidráulicas de SWMM](https://www.epa.gov/water-research/storm-water-management-model-swmm).

### N-02 · P1 · La ruta de entrenamiento y serialización todavía mezcla contratos de 17 y 21 variables

`main.py:187` prepara el banco usando los valores por defecto v17; el pipeline general también entrena v17. En `main.py:1292` se calculan las variables tributarias pero se descartan antes de persistir el dataset. Los scripts particulares de la parte 2 sí usan v4_21, por lo que ejecutar el comando general no reproduce automáticamente la metodología reciente.

**Defecto reproducido:** `ml/bench/train.py:101` escribe siempre 17 nombres y `tabular_v3_17` en los metadatos, aunque el estimador se haya ajustado a `prepared.X` con 21 columnas. Un entrenamiento mínimo de 80 filas produjo **modelo de 21 variables con manifiesto de 17**. `promote.py` usa esos metadatos para proteger contratos, por lo que la inconsistencia puede impedir una promoción correcta o declarar incorrectamente el contrato en un destino nuevo.

Corrección necesaria: seleccionar un contrato explícito de extremo a extremo, derivar la metadata del PreparedDataset validado y comprobar columnas ordenadas/identidad del contrato al cargar y promover. Los nombres, unidades y semántica deben coincidir, además del número de columnas.

### N-03 · P1 · Identidad y lectura de datos todavía asumen una sola red

`database/training_queries.py:375` identifica un snapshot por `(shape_id, factor_mult, node_id)` sin `network_id`. Dos redes legítimas con nodos `J1` y las mismas formas/factores se interpretan como snapshots duplicados. El lector recomienda recrear la BD, lo que sería una respuesta equivocada al objetivo de acumular redes.

`ml/bench/baselines.py:86,119,243` también construye las referencias por nodo/forma/factor sin red. Reproducción con dos redes y nodo `J1`: `ValueError` por celdas duplicadas. Si los nombres no colisionan, la omisión no se convierte por ello en una evaluación válida entre redes.

`main.py:1034` carga `data/training/dataset_final_v4_21.csv` de una ruta fija, independientemente del CSV configurado. `csv_backfill.py:236` lo une por forma/factor y nodo, sin comprobar un manifiesto de origen de red y evento. Al cambiar de red, un archivo residual puede atribuir features de la red anterior a los nuevos runs si coinciden los IDs.

Corrección necesaria: identidad compuesta de red/nodo/evento, selección explícita de redes/runs y comprobación de procedencia de todos los CSV. El `network_id` debe usarse para identidad, trazabilidad y partición; convertirlo en una entrada categórica del modelo no soluciona la transferencia a redes desconocidas.

### N-04 · P0 · No existe evaluación de transferencia entre redes y el tuning no es anidado

`ml/bench/schemas.py:53` solo reconoce `LOSO`, `GroupKFold5` y `LeaveOneShapeOut`, agrupados por factor o forma. Ninguno retiene una red completa. `ml/bench/tune.py:265` ajusta hiperparámetros con todas las formas del PreparedDataset; la parte 2 evalúa posteriormente esas configuraciones en las mismas 40 formas.

El código declara correctamente `estimacion_exploratoria_condicionada`. Es una mejora de honestidad, no una eliminación del sesgo de selección. Hace falta separar la red y/o eventos de prueba antes de cualquier selección, o anidar el tuning en el entrenamiento de cada fold externo. [scikit-learn: validación anidada frente a no anidada](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

Además, `tune.py:205` minimiza la media de RMSE de cinco folds; el informe presenta como principal el RMSE agregado. `config.yaml` también conserva el ranking por media de folds. Ambas medidas son legítimas, pero diferentes: hay que elegir y registrar una función objetivo coherente, especialmente cuando las redes tengan distinto tamaño.

### N-05 · P1 · El grafo pierde conductos paralelos y no cubre todas las estructuras SWMM

En `extraction/topology.py:16` se usa `nx.DiGraph`; `add_edge(A,B,...)` sobrescribe los atributos de una arista anterior con los mismos extremos. **Dos tuberías paralelas P1 y P2 producen una sola arista P2** en la reproducción. El cálculo de capacidad por predecesor utiliza el `link_id` sobreviviente.

El grafo agrega solamente `CONDUITS`; no representa enlaces de bombas, vertederos y orificios. En bifurcaciones, sumar íntegramente los aportes de cada ancestro no representa el reparto de caudal; en la reconfluencia el conjunto de ancestros evita duplicar IDs, pero no corrige las asignaciones incorrectas de las ramas. No se incluyen propagación, atenuación ni inversión de flujo.

Para el árbol de Chico el proxy tiene un dominio claro. Antes de admitir redes adicionales se necesita conservar enlaces paralelos, representar las estructuras admitidas y rechazar explícitamente topologías no soportadas, o desarrollar un cálculo tributario compatible con ellas. H-12 deja de ser un riesgo menor al cambiar de red.

### N-06 · P1 · Unidades, factores de inflow y geometría no se validan como contrato hidráulico

`static_features.py:21` toma el máximo de la serie cruda y omite el factor multiplicador y la suma del caudal basal cuando hay una serie. Reproducción: pico de serie 10 L/s, factor 2 y basal 5 L/s → feature de pico **10** frente a aporte efectivo **25 L/s**. La escritura de escenarios conserva `[INFLOWS]`, por lo que SWMM puede aplicar un caudal diferente del representado por el modelo. Los patrones basales tampoco están incorporados.

El entrenamiento no exige LPS; `hydrograph_batch.py:110` solo rechaza unidades diferentes si la opción existe. Si falta, no hay rechazo, aunque SWMM tiene su propio valor por defecto. La BD fija LPS en el backfill. Es necesario convertir explícitamente o rechazar unidades no soportadas y opciones omitidas; no basta renombrar columnas.

`topology.py:49` interpreta `XSECTIONS.height` como diámetro y aplica Manning circular a cualquier sección; omite número de barriles y toma el valor absoluto de la pendiente (`:103`). `static_features.py:82` solo conoce cotas de junctions y supone offsets de profundidad. Secciones no circulares, barriles múltiples, `LINK_OFFSETS=ELEVATION`, pendientes adversas y otros tipos de nodo necesitan tratamiento correcto o rechazo explícito.

La semántica de aporte basal más serie escalada está definida en el [manual oficial de SWMM 5.2](https://www.epa.gov/system/files/documents/2022-04/swmm-users-manual-version-5.2.pdf). Las propiedades concretas de la futura red siguen sin conocerse.

### N-07 · P1 · La nueva tabla v4_21 perdió la protección de procedencia

`database/sql/006_tributary_contract.sql:22` crea `node_features_v4_21` sin los triggers de inmutabilidad presentes en `node_features`. La migración 007 añade la vista, no esas protecciones.

**Prueba sobre copia en memoria de la BD:** un run referenciado en `training_run_inputs` rechaza `UPDATE node_features SET elev_fondo=elev_fondo+1` con `pinned by training provenance`, pero acepta la misma modificación en `node_features_v4_21`. También acepta cambiar su continuidad a 99 %, reproduciendo la brecha antigua D-01 (`005_provenance_integrity.sql:992`).

Los fingerprints de nuevos PreparedDataset pueden detectar contenido distinto al preparar de nuevo, pero no impiden alterar retrospectivamente la evidencia SQL ni garantizan que los artefactos históricos sigan correspondiendo a esa BD. Hay que extender la protección al contrato activo y a toda la identidad/evidencia del run.

### N-08 · P1 · La calidad hidráulica no es todavía una condición obligatoria de admisión

La continuidad ya se persiste y se excluyen 37 runs conocidos >5 %. Sin embargo, `training_queries.py:304` admite continuidad NULL. `simulation/batch.py:206` devuelve resultados incluso después de agotar reintentos con estado `not_drained` o `unverified`; ese estado queda en un sidecar y no participa en la selección SQL/PreparedDataset.

La forma `base` sigue otra ruta sin la misma comprobación de cola. `validation/hydrograph_batch.py:341` avisa ante continuidad excesiva, pero sigue calculando e integrando esas métricas; tampoco aplica el control de drenaje completo del batch de entrenamiento. No se debe mezclar referencia SWMM no verificada con la evaluación principal sin una etiqueta y exclusión coherentes.

Además, `validation/drainage_state.py` acepta como **`drained`** un conjunto de observables con todos los valores NaN: las comparaciones `abs(NaN)>tolerancia` son falsas. Se reprodujo directamente. Debe validar finitud, cobertura por objeto y duración efectiva de la ventana antes de decidir el estado. La hipótesis de caudal basal cero tampoco sirve automáticamente para una red con flujo permanente.

### N-09 · P1 · El parser de respaldo sigue convirtiendo inundación real en cero

`extraction/labels.py:79` toma `parts[-1]`, que en los reportes probados corresponde a profundidad embalsada y no a volumen total. En los seis controles produjo **0 m³**; el parser principal produjo, entre otros, **123.097 m³** para `long_moderate`, **13.235 m³** para `shape_2h` y **1.320 m³** para el impulso corto (todos f=5, condiciones iniciales originales).

El problema se activa cuando falla la lectura principal; no invalida por sí solo el dataset actual, cuyos tres controles originales coincidieron por nodo con SQL. Pero no debe permanecer como recuperación silenciosa. Un reporte inexistente/ilegible o un número inválido también puede terminar tratado como ausencia de inundación (`fillna(0.0)`). Se necesita distinguir explícitamente reporte válido sin inundación de fallo de lectura. H-07 y D-18 permanecen abiertos.

### N-10 · P1 · Condición inicial heredada y omisiones de capacidad aguas abajo

Los **160 conductos** del INP conservan `InitFlow != 0`. En el impulso de 0,5 h a f=5, cambiando únicamente InitFlow a cero en una copia, el volumen reportado pasó de **1.320 a 1.195 m³**; cambiaron 60 nodos, con diferencias de hasta 13 m³ por nodo. Las continuidades fueron 1,978 % y −0,924 %, ambas dentro del filtro de 5 %.

Esto demuestra que aprobar continuidad no demuestra que la condición inicial represente el evento que se quiere estudiar. No se recomienda poner todo a cero sin decidir si el experimento parte seco o de un estado previo; la condición debe justificarse y aplicarse de manera consistente entre redes/eventos.

La capacidad comparada con el aporte tributario sigue siendo `upstream_capacity_lps`, que es un proxy y no la capacidad limitante de evacuación del pozo. En el PreparedDataset hay **54.891 filas (35,625 %)** sin capacidad aguas arriba; también faltan allí las dos nuevas variables derivadas de capacidad. Es una ausencia estructural en 57 nodos de cabecera, no una medición desconocida aleatoria. La mediana puede ocultar esa diferencia. Falta evaluar capacidad aguas abajo, almacenamiento, condición de salida y representación explícita de esa ausencia. No debe interpretarse `integral_exceso_capacidad_m3` como un balance físico de inundación.

### N-11 · P1 · Integridad de modelos y dominio de inferencia incompletos

`predict.py` verifica `promotion.json`; `scenario_predict.py:69` carga directamente ambos joblib sin verificar la promoción ni que el par provenga del mismo experimento. Solo exige el número de variables y posteriormente valida el frame. `hydrograph_batch` comprueba el hash del INP junto al clasificador, no la coherencia del par completo.

La bandera `extrapolated` usa pico/base y límites de factor de la configuración; no compara el dominio estático, duración, volumen, capacidad, condición inicial o descarga con los del entrenamiento. En una red nueva, ese denominador es el caudal base de la **nueva** red: estar dentro del rango de factores no acredita estar dentro del dominio aprendido.

La protección por un único `training_inp_hash.txt` tampoco representa un modelo entrenado en varias redes. `--allow-inp-mismatch` permite ejecutar un experimento, pero no implementa la validación de transferencia. Se requiere manifiesto con redes/runs/contrato, par de modelos coherente y diagnóstico de dominio separado de la coincidencia de archivos.

### N-12 · P2 · Inferencia por factor supone una forma espacial común

`predict.py:17` toma la primera TIMESERIES del INP y reconstruye todas las series usando esa forma normalizada. La migración de datos (`scripts/migrar_dataset_v4_21.py:49`) utiliza las series propias de cada nodo en la forma base. En una red con series heterogéneas o una primera TIMESERIES ajena al caudal, las dos rutas no representan el mismo evento.

También `tributary_features.py:20` sostiene el valor del extremo al interpolar fuera del rango. En la reproducción con un aporte que comienza en t=1 h, inventa aporte antes de ese instante: integral de 54 frente a 18 m³ bajo la convención de caudal cero fuera de la serie. El CSV actual exige inicio común en cero, por lo que este último caso no es un fallo demostrado de esos CSV válidos; es una limitación de la función para series de INP con distintos rangos. Conviene unificar construcción del evento efectivo entre entrenamiento y todas las rutas de inferencia.

### N-13 · P2 · La generación actual cubre hidrogramas de aporte, no generalización hidrológica completa

El INP activo no contiene RAINGAGES ni SUBCATCHMENTS. Los eventos son formas de caudal en L/s escaladas sobre los nodos; no hay entradas de lluvia, áreas de subcuencas, infiltración o transformación lluvia–escorrentía en el contrato tabular. `apply_shape` conserva la distribución espacial base y aplica una forma/factor común.

Por tanto, los resultados respaldan un surrogate hidráulico para **aportes prescritos**. No demuestran que el modelo transforme aguaceros de otra cuenca en inundaciones, ni que genere aguaceros. Si se desea predecir desde lluvia, hay que resolver explícitamente la generación de aportes o incorporar la representación hidrológica correspondiente. Si la entrada seguirá siendo un hidrograma por nodo, debe declararse así y ampliar ensayos de heterogeneidad espacial y desfases.

## Qué se corrigió desde la auditoría previa

La [matriz completa](2026-10-25-matriz-seguimiento.md) conserva los 111 identificadores originales y distingue cierre, cierre parcial, persistencia y falta de revalidación. Los cambios con mayor valor para esta decisión son:

| Hallazgo previo | Estado verificado en el worktree |
|---|---|
| M-01: sin referencias y evaluación solo por factor | Corregido para el benchmark de formas: referencias incorporadas y LeaveOneShapeOut principal. No cubre otras redes. |
| H-02 / M-02: ausencia de volumen y mala generalización | Ausencia de volumen corregida por v4_21; desempeño mejorado, pero M-02 sigue parcial y dependiente de la forma. |
| M-03: F1 promedio con folds secos como cero | Corregido en las métricas agregadas del banco; el evaluador legacy conserva su definición. |
| M-08 / T-19: OOF descartados | Predicciones y procedencia persistidas; se pudieron recalcular resultados. La parte de tiempos no queda cerrada en todas las rutas. |
| M-10: faltan métricas | Matriz de confusión, especificidad y desglose por forma disponibles; PR-AUC/MCC/balanced accuracy no se incorporaron al núcleo del banco. |
| H-04 / D-02: continuidad ausente | Ya hay valores en 1.000/1.000 runs y filtro de 37; faltan otras evidencias y admisión estricta de runs no verificados. |
| H-05: drenaje convertido en rampa de 6 h | Cierre explícito y controles de cola añadidos. La ruta base y los requisitos de admisión no están unificados. |
| M-07 / V-04: modelos no coinciden con promoción | Los hashes actuales sí coinciden y `predict_network` comprueba procedencia. Cobertura incompleta en otros cargadores y par activo distinto del ajustado. |
| V-01 / M-13: gráficos en muestra sin indicarlo | Advertencias y notas añadidas en rutas concretas; existen además figuras OOF de la parte 2. No equivale a validación entre redes. |
| V-07: script sobrescribe INP al importarse | Corregido con función, main y control de sobrescritura. |
| T-06 / M-20: inferencia sin validar features | Frame validado en las dos rutas tabulares activas. Persiste el problema diferente de manifiestos/cargadores de N-02/N-11. |
| M-06 / M-15: comparación y tuning | Tres familias comparadas con 20 trials cada una; mejora real del procedimiento. No anidado y con una semilla: cierre parcial, no confirmatorio. |
| H-01, H-07, H-08, H-09, H-11/H-12 | Persisten condiciones iniciales, parser, capacidad, cota de outfall, unidades/semántica y limitaciones topológicas. Algunas se vuelven bloqueantes al salir de Chico. |
| D-01 / D-03 / D-05 / D-06 / D-12 | Persisten brechas en inmutabilidad de runs, metadatos legacy, migraciones al leer, persistencia no atómica/idempotente y hash de evento incompleto. |

**Corrección de interpretación necesaria:** no debe repetirse la afirmación original de que todas las transiciones de inundado a seco al subir el factor son «ruido de redondeo». La investigación posterior encontró descensos del volumen continuo en los 12 casos re-simulados, y el umbral nominal de 1 m³ reportado se comportó aproximadamente como 0,5 m³ continuo en aquella muestra. Esta auditoría no atribuye las 72 transiciones a una causa única. El informe posterior también muestra que integrar el `.out` cada 3 minutos no es automáticamente más exacto que el `.rpt`; para un dataset nuevo conviene estudiar el acumulador de PySWMM y fijar explícitamente la semántica del umbral.

## Preparación para un modelo común: condiciones concretas de avance

| Actividad | Estado hoy | Condición para avanzar |
|---|---|---|
| Elegir XGBoost y MLP como candidatos de investigación | Sí | Mantener las cifras como exploratorias y no confundirlas con los modelos activos. |
| Inspeccionar y simular un piloto de la segunda red | Pendiente de INP | Confirmar unidades, topología, estructuras, descarga, aportes, calibración y condiciones iniciales. |
| Generar un dataset comparable de ambas redes | No listo como flujo integrado | Corregir N-02/N-03/N-06/N-08/N-09 y versionar el procedimiento/etiquetas. |
| Entrenar un modelo único con A+B | Algoritmos disponibles; infraestructura incompleta | Identidad compuesta, contrato hidráulico válido, datasets y manifests reproducibles. |
| Afirmar transferencia a una red no vista | No | Evaluación por red retenida; ninguna selección con sus etiquetas. |
| Afirmar generalización simultánea a red y evento nuevos | No | Exclusión de red y de familia/evento, con todas sus escalas/variantes, según la pregunta experimental. |

Secuencia recomendada, sin entrenar todavía el experimento definitivo:

1. **Fijar el alcance físico.** Precisar si se aceptarán únicamente redes por gravedad con conducciones circulares y descarga libre, o también remanso, almacenamiento, bombas, estructuras especiales y aportes permanentes. La decisión depende del INP nuevo y del objetivo científico. No asumirla a partir de Chico.
2. **Reparar el recorrido v4_21 y la identidad entre redes.** Un comando documentado debe producir datos, preparar, entrenar ambas familias, evaluar y guardar artefactos coherentes, usando rutas por experimento/red. Añadir regresiones para los casos reproducidos en esta revisión. Mantener las protecciones de modelos activos.
3. **Definir la verdad hidráulica.** Misma semántica de aporte efectivo, condición inicial y umbral en ambas redes; balance y drenaje válidos; registro de versión SWMM, inputs, settings, cierre, reporte y motivo de cualquier exclusión. Una continuidad menor de 5 % es un criterio operativo actual, no una garantía universal de exactitud.
4. **Medir primero la transferencia sin utilizar B para ajustar.** Congelar la selección basada en A y evaluar A→B; para cada arquitectura se permite el ajuste interno solo en A. Conservar esa medición como resultado previo a incorporar B. Revisar B→A con la misma separación si se quiere estudiar asimetría.
5. **Después entrenar el modelo conjunto A+B.** Compararlo con modelos individuales en eventos reservados de redes conocidas. Ese ensayo mide mejora dentro de A/B, no transferencia a una tercera red. Si se usan datos de B para entrenar, B deja de ser una red externa para ese modelo.
6. **Evaluar redes completas retenidas.** Con dos redes se pueden hacer A→B y B→A, pero solo hay dos unidades espaciales; no bastan para estimar ampliamente variabilidad entre redes. Para evaluar un modelo que aprende de varias redes y se prueba en otra conviene incorporar más redes, con al menos una red realmente no utilizada en la selección. El tuning debe quedar dentro del conjunto de entrenamiento de cada evaluación externa.
7. **Separar redes y eventos conforme a la afirmación.** Retener una red completa prueba transferencia espacial; retener una forma/evento en redes conocidas prueba transferencia de evento. Para afirmar ambas a la vez, el evento de test y sus variantes tampoco pueden participar en el ajuste en otras redes. Usar IDs de evento físicos o familias generadoras, no solo nombres de CSV.
8. **Reportar errores por red y por evento.** RMSE/MAE en m³ y agregaciones con peso explícito por red, sesgo de volumen, recall, PR-AUC, falsos negativos de volumen alto y límites de dominio. Añadir variabilidad entre semillas de MLP y sensibilidad a la representación hidráulica. No ocultar los peores eventos detrás del promedio de todas las filas.

No hay evidencia de que haga falta abandonar MLP/XGBoost inmediatamente. Primero hay que verificar que el problema sea identificable con sus entradas y que las pruebas separen correctamente redes y eventos. Una arquitectura más compleja no recupera información hidráulica que no recibe.

## Artefactos de esta revisión

Carpeta de ejecución: `herramienta-ia-model/outputs/audit_20261025/`, fuera del worktree de trabajo y separada de los modelos/resultados de investigación existentes.

| Archivo | Contenido |
|---|---|
| `inspect_current.py` / `evidence.json` | Consultas de BD, métricas OOF, modelos activos y reproducciones sintéticas. |
| `hydraulic_probe.py` / `hydraulic_evidence.json` | Seis simulaciones de control; comparación por nodo contra la BD y sensibilidad a InitFlow. |
| `boundary_probe.py` / `boundary_evidence.json` | Mismas 21 variables, distintas condiciones de descarga y distinta inundación. |
| `persistence_probe.py` / `persistence_evidence.json` | Modelo 21/metadata 17 y mutaciones SQL en copia en memoria. |
| `verified_tests.log` | 611 passed, 37 warnings, 101,74 s. |
| `data_cli_tests.log` | 424 passed, 3 deselected, 432 warnings, 78,43 s. |
| `swmm/` y `boundary/` | INP, RPT y OUT de los ocho controles, reproducibles sin tocar el INP base. |

Para reproducir los cuatro diagnósticos, desde la carpeta principal: `venv\Scripts\python.exe outputs\audit_20261025\<script>.py`. No requieren ejecutar entrenamientos definitivos; `persistence_probe.py` ajusta solamente un par diminuto de diagnóstico en `disposable_candidate/`.

Información pendiente del autor para el siguiente trabajo: archivo de la nueva red y sus condiciones de borde/aportes; si el modelo recibirá caudales por nodo o lluvia; criterio de error y de falso negativo aceptable para el uso previsto. Estas respuestas no se han supuesto ni sustituyen los defectos ya demostrados.
