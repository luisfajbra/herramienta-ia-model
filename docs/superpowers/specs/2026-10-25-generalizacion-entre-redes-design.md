# Diseño: modelo hidráulico compartido entre redes

Fecha: 25 de octubre de 2026

Proyecto: worktree `optuna-tuning-familia-ganadora`, commit base `78e551a`

Auditoría de referencia: `docs/auditoria/2026-10-25-revision-generalizacion.md`

## Objetivo

Preparar el proyecto para evaluar de forma reproducible y físicamente defendible si un modelo tabular compartido puede predecir el volumen y la presencia de inundación en más de una red de alcantarillado pluvial. La entrada confirmada para esta etapa son hidrogramas de caudal por nodo; el modelo no recibirá lluvia directamente ni calculará transformación lluvia–escorrentía.

El resultado buscado es un flujo seguro para incorporar una red, validar y versionar sus simulaciones, preparar un dataset multired, comparar MLP y XGBoost con una prueba espacial independiente, y producir modelos cuyos contratos y procedencia sean verificables.

## Evidencia que motiva el diseño

La revisión de seguimiento confirma una sola red en el dataset actual. Aunque hay resultados OOF para 40 formas, las formas participaron en el ajuste de hiperparámetros; esos resultados no miden transferencia a una red no vista. Los artefactos activos tampoco son el par ajustado cuyos OOF se analizaron.

La auditoría reprodujo bloqueos que afectan la siguiente etapa:

- Un entrenamiento de 21 columnas puede persistir metadata de 17 columnas.
- Lectura, agrupación y backfill omiten la red en algunas claves y asumen identificadores únicos entre redes.
- El contrato actual omite condiciones de borde capaces de cambiar la inundación sin cambiar ninguna de sus 21 features.
- Los controles de unidades, finitud, cierre, condición inicial, parser de inundación y admisión hidráulica no están aplicados de manera uniforme.
- El grafo pierde enlaces paralelos y simplifica topologías que podrían existir en otra red.
- Los cargadores y manifests no identifican de forma completa una pareja de modelos entrenada con varias redes.

La revisión contabiliza 1.035 pruebas seleccionadas aprobadas y ocho simulaciones de diagnóstico. No se corrieron la suite completa, las 1.000 simulaciones ni un tuning completo. Esos resultados son evidencia acotada, no una certificación del sistema entero.

## Alcance y límites

Incluye:

1. Reparar el recorrido del contrato tabular v4_21 y su metadata, y hacer que red, nodo, escenario y evento conserven identidades inequívocas en lectura, almacenamiento, preparación y evaluación.
2. Inspeccionar la nueva red antes de admitirla: unidades, elementos, topología, condiciones de borde, aportes, estado inicial, discretización temporal y compatibilidad con las variables disponibles.
3. Establecer una política uniforme de calidad y procedencia para resultados SWMM, incluyendo valores finitos, balance, cierre, reporte válido, unidad, versión del motor y condición inicial.
4. Mantener separados el experimento A→B con modelo congelado y la evaluación posterior de un modelo entrenado con A+B.
5. Comparar MLP y XGBoost por red y evento, incluyendo sensibilidad entre semillas para MLP, métricas de clasificación y volumen, y criterios de falsos negativos grandes.

Fuera de alcance:

- Entrada directa de lluvia, generación de lluvia, subcuencas, infiltración o modelación hidrológica lluvia–escorrentía.
- Afirmar que dos redes bastan para estimar la variabilidad entre todas las redes posibles.
- Promover automáticamente un modelo a producción o reemplazar el modelo activo.
- Elegir silenciosamente una tolerancia universal de continuidad, un umbral de inundación o una condición seca/inicial para las redes futuras.
- Añadir features físicas específicas antes de inspeccionar el INP B y comprobar que sus entradas estarán disponibles al predecir.

## Diseño por etapas

### Etapa 1: contrato, identidad y procedencia multired

Definir un único contrato explícito desde la consulta de entrenamiento hasta preparación, entrenamiento, evaluación, carga y promoción. El contrato declarado en `PreparedDataset`, las columnas ordenadas del frame, los estimadores, metadata y manifests deben coincidir en nombre, orden, unidades, nulabilidad e identidad, y deben rechazar incompatibilidades antes de ajustar o predecir.

Usar claves compuestas que incluyan la red y la identidad del escenario/evento donde corresponda. Las coincidencias de `node_id`, `shape_id` o factor entre redes son válidas si la red difiere; duplicados dentro de la misma identidad completa sí deben fallar con un mensaje que indique la clave afectada. Las rutas de CSV deben venir de configuración explícita, y cada fuente debe tener un manifiesto de red, INP, evento y hash.

Los datos ya usados por un experimento deben quedar protegidos y ser reproducibles. La migración de procedencia debe aplicar a las features v4_21, runs, entradas y elementos hidráulicos que definen esos datos. La apertura de una base para leer o entrenar no debe modificarla con migraciones implícitas; una operación de migración debe ser explícita.

### Etapa 2: admisión hidráulica, simulaciones y etiquetas

Antes de ampliar o regenerar datos, perfilar el INP B y decidir qué tipos de red pertenecen al dominio científico del modelo. El informe de admisión documentará como mínimo FLOW_UNITS, unidades geométricas, tipos de nodos/enlaces, enlaces paralelos, secciones y barriles, cotas/offsets, rutas y descargas, almacenamiento/bombeo/control, aportes basales y series, intervalos temporales, condiciones iniciales y configuración del solver. Una propiedad incompatible debe producir un rechazo explicado o una transformación probada y documentada.

Unificar la interpretación del caudal efectivo usado por SWMM y por las features; validar unidades ausentes o no compatibles, cobertura temporal y valores finitos. Corregir el grafo para conservar enlaces paralelos y hacer explícitos los tipos de estructura soportados. Donde la feature sea un proxy (por ejemplo, capacidad aguas arriba), nombrarla y evaluarla como tal; no interpretarla como capacidad de evacuación del pozo.

Una corrida solamente será elegible si sus entradas y reporte son identificables, sus targets son finitos y consistentes, y pasa la política versionada de balance y cierre. “Drenada” exige cobertura observable finita y suficiente, y una cola compatible con la condición base de esa red. Reporte faltante o ilegible es fallo de extracción, nunca inundación cero. El umbral de inundación, condiciones iniciales y límites de balance deben escribirse en la configuración del experimento con su justificación; la suite de software puede probar la aplicación de una regla, pero no escoger por sí sola el criterio de ingeniería.

### Etapa 3: evaluación espacial, entrenamiento y transferencia

Antes de usar targets de B para ajustar, guardar un experimento A→B: congelar el contrato, selección de hiperparámetros y modelos definidos solo con A; ejecutar B con el mismo procedimiento hidráulico; reportar por separado compatibilidad de dominio, calidad SWMM y error del modelo. Ese resultado responde a transferencia del modelo de A a B. B debe mantenerse como red de desarrollo después de analizar su desempeño, y no volver a llamarse test externo para decisiones que utilicen sus etiquetas.

Después, comparar modelos A, B y A+B en eventos reservados por identidad física y sus variantes. Esa evaluación responde al desempeño en redes conocidas. Con dos redes se permiten A→B y B→A como casos direccionales, sin afirmar que estiman la variabilidad de redes futuras. El protocolo LeaveOneNetworkOut puede medir cada red retenida en un dataset multired, siempre que todo el tuning y selección se repita dentro del conjunto de entrenamiento de cada fold externo. La prueba de una red verdaderamente futura requiere reservar una tercera red que no participe en selección.

Los reportes mostrarán métricas por red y por evento y definirán los pesos agregados antes de comparar: RMSE/MAE de volumen en m³, sesgo del volumen total, recall, PR-AUC y falsos negativos por bandas de volumen, además de calibración/umbral del clasificador cuando corresponda. El RMSE pooled y la media de RMSE por red responderán a preguntas distintas; no se intercambiarán sin declararlo. Los artefactos de XGBoost y MLP incluirán el mismo `prep_id`, contrato y procedencia y solo se promoverán mediante el mecanismo existente, bajo una acción explícita separada de este plan.

### Decisión registrada — 27 de octubre de 2026

Para seleccionar hiperparámetros en evaluación multired se usará `macro_network_rmse`: calcular el RMSE de volumen en cada red y promediar esos RMSE con el mismo peso por red, sin ponderar por su número de nodos o corridas. La razón es que la pregunta principal es la transferencia entre redes y una red grande no debe dominar la selección. El `pooled_rmse` se conservará como resultado complementario para describir error por fila; no reemplazará el objetivo macro ni se usará para cambiar retrospectivamente la selección. Cada configuración, estudio Optuna y reporte registrará el objetivo explícito. La decisión fue confirmada por el usuario el 27 de octubre de 2026.

**Alcance estadístico con dos redes:** en cada fold externo LeaveOneNetworkOut solo queda una red en entrenamiento. Los folds internos LeaveOneShapeOut particionan sus formas, por lo que `macro_network_rmse` dentro de cada estudio tiene una sola red y equivale a su RMSE. No debe describirse como una media macro entre varias redes ni como prueba de selección transferible a una población de redes. Se conservan por separado los resultados A→B y B→A; una tercera red reservada es necesaria para una verificación externa independiente de redes no vistas.

La entrada de ejecución `evaluate_nested_networks_from_config(...)` toma objetivo, número de trials y folds de `bench.tuning`; no se permite que una llamada de producción sustituya silenciosamente la decisión configurada. El manifiesto de cada trial registra estado Optuna, métrica objetivo, valor, RMSE por fold interno y RMSE por red interna, además de parámetros y duración. Estos campos dejan rastro suficiente para reconstruir por qué se seleccionó cada candidato.

Los resultados nested se desglosan por red y por red–`shape_id`. `shape_id` representa la forma del hidrograma usada en una familia de escenarios; se informa como estrato descriptivo, no como validación independiente de familias físicas de tormenta. Cada estrato registra número de filas, positivos reales, matriz de confusión, precisión, recall, F1, average precision como área bajo la curva precisión–recall, RMSE/MAE de volumen, sesgo total y falsos negativos/recall por banda de volumen real. El PR-AUC queda `null` si falta una de las clases. Las bandas usan el umbral de inundación del `PreparedDataset` como límite inferior, cortes finitos ascendentes configurados en `bench.reporting.volume_band_edges_m3` y una última banda abierta; los cortes, umbral y unidad del estrato quedan en el manifiesto y el hash de configuración participa en el ID inmutable del experimento.

**Decisión registrada — 28 de octubre de 2026:** el usuario confirmó usar los cortes A-derivados de 24, 108 y 344 m³. La propuesta se obtuvo sin usar predicciones ni etiquetas de B: resumió los 154.080 targets del artefacto A-only software-elegible `758d3a0ad79e8506`, bajo el umbral configurado de 1 m³. Había 46.225 positivos; los cuartiles de su volumen fueron 24, 108 y 344 m³. Las bandas resultantes son `[1,24]`, `(24,108]`, `(108,344]` y `>344` m³, con recuentos A de 11.689, 11.506, 11.488 y 11.542 positivos. Son estratos descriptivos, no límites de severidad hidráulica ni criterios de admisión. Se fijan en `bench.reporting.volume_band_edges_m3`; para B→A se aplicarán solo al reporte posterior a la predicción y nunca al tuning o selección del modelo.

### Semillas y ensemble MLP — decisión confirmada el 28 de octubre de 2026

Cada experimento MLP completo usará exactamente cinco semillas deterministas,
`[42, 43, 44, 45, 46]`, derivadas de `ML_RANDOM_STATE=42`. Cada pareja
clasificador/regresor se conservará en su propio directorio inmutable. Se
informarán las métricas de cada semilla, su media y su desviación estándar
muestral (`ddof=1`); no se elegirá una semilla por su desempeño en el test externo.

El ensemble final promediará las probabilidades de inundación de las cinco
semillas y aplicará el umbral de clasificación existente de 0,5 a esa media.
Para el volumen condicional, primero se aplicará `expm1` a cada regresor y se
mantendrá el mínimo físico existente de 0 m³ por semilla; después se promediarán
esas predicciones en m³. El volumen encaminado será cero si la clasificación
del ensemble es seca y será la media condicional si es inundada. La pareja
persistida conservará las interfaces de inferencia existentes: el regresor
devolverá `log1p(media_condicional_m3)` para que el consumidor aplique `expm1`.

Estas decisiones no cambian `macro_network_rmse` como objetivo configurado de
selección de hiperparámetros ni autorizan una promoción automática. La evaluación
por semillas y los artefactos ajustados con todas las filas preparadas son
productos distintos; el manifiesto identifica expresamente ese alcance y no
presenta los artefactos finales como modelos de un fold externo. Esta tarea se
valida únicamente con fixtures sintéticas, sin simulaciones ni ajuste de redes
reales. Las entradas nuevas son APIs del banco; no añaden comandos CLI.

La evaluación nested MLP aplica las cinco semillas después de seleccionar los
hiperparámetros dentro del train de cada fold externo. Las cinco parejas usan
esos mismos parámetros seleccionados y solo las filas del train externo. Se
conservan el OOF de cada semilla, la media/desviación estándar muestral de sus
métricas y las métricas del ensemble como resultado principal. La semilla del
estudio interno se registra separada de las cinco semillas de los modelos.

La API y el CLI anteriores para ajustar una sola pareja conservan su ruta
`candidates/<familia>/` y permiten repetir el ajuste. Los directorios inmutables
por semilla pertenecen a las APIs explícitas de experimentos multisemilla.
Un ajuste con parámetros configurados, sin selección, registra `objective: null`
y `selection: configured_parameters`; solo una selección efectivamente realizada
se describe con su objetivo y evidencia de tuning.

## Componentes esperados

- `swmm_resilience/ml/contracts.py`: catálogo y validación de contratos, incluyendo la selección explícita de v4_21.
- `swmm_resilience/ml/bench/preprocess.py` y `schemas.py`: construcción y validación de datasets y particiones sin mezclar redes o contratos.
- `swmm_resilience/ml/bench/train.py`, `promote.py` y cargadores en `predict.py`/`scenario_predict.py`: metadata verdadera, pareja coherente y validación de artefactos.
- `swmm_resilience/database/training_queries.py`, `csv_backfill.py`, migraciones SQL y `schema.py`: claves de red, selección de runs, procedencia y protección de las features activas.
- `swmm_resilience/extraction/labels.py`, `static_features.py`, `topology.py`, `tributary_features.py`; `simulation/batch.py`, `runner.py`, `inflow_closure.py`; y `validation/drainage_state.py`/`hydrograph_batch.py`: interpretación hidráulica común y admisión verificable.
- `swmm_resilience/ml/bench/folds.py`, `schemas.py`, `tune.py`, `evaluate.py`, `metrics.py` y sus reportes: folds externos por red, ajuste interno y resultados por red/evento.
- Pruebas unitarias con SQLite/INP/OUT sintéticos y pruebas de regresión para los casos reproducidos en la auditoría. Simulaciones y cambios en el dataset canónico usan una base, directorio y copias INP dedicados al experimento.

Este mapa señala responsabilidades previstas; cada plan hijo deberá confirmar nombres y puntos de extensión contra el código antes de modificarlos.

## Criterios globales de aceptación

1. Una fixture de dos redes con nodos y formas con nombres iguales puede prepararse sin colisiones, mientras una duplicación con la misma clave de red/escenario/nodo se rechaza.
2. Un modelo de 21 features no puede guardarse, cargarse ni promoverse con metadata de 17; reordenar columnas o cambiar el hash del contrato también falla antes de predecir.
3. La entrada y las features calculadas corresponden al mismo caudal efectivo, unidad y ventana temporal; valores no finitos o estados hidráulicos no verificados no entran silenciosamente al entrenamiento principal.
4. El parser diferencia un reporte válido con cero inundación de un reporte que falta o no puede leerse; conserva unidades y coincide con el extractor principal en fixtures representativas.
5. Ningún enlace paralelo se pierde; geometría o estructuras fuera del alcance se aceptan con cálculo verificado o se rechazan de manera explícita.
6. Las migraciones y consultas de datos preservan la procedencia y no mutan la base real durante una operación de lectura.
7. La partición externa reserva una red completa; transformadores, tuning, selección y umbrales se ajustan solo dentro del train de cada fold externo. Cada muestra de test aparece exactamente una vez por protocolo.
8. El informe distingue A→B antes de entrenar con B del resultado posterior A+B, y no afirma desempeño en una tercera red sin datos realmente retenidos de esa red.
9. Se puede regenerar una ejecución identificada desde INP/eventos/configuración/versiones hasta preparación, folds, OOF y artefactos XGBoost/MLP, sin sobreescribir resultados ajenos.

## Decisiones de ingeniería que se mantienen abiertas

Se resolverán al inspeccionar el INP B y la necesidad científica, antes de incorporar sus filas al dataset:

- Qué elementos y condiciones de borde se admiten, y cuáles se rechazan: el modelo actual no distingue por sí mismo FREE de FIXED.
- Si cotas, elevaciones y features de capacidad son comparables entre redes y qué datum/unidades comparten.
- Si los caudales son libres de variar por nodo en tiempo y forma, o si el protocolo limita espacialmente las entradas.
- Estado hidráulico inicial representativo de cada evento: seco, permanente, antecedente u otro estado justificado.
- Umbral de volumen inundado, tolerancia de balance por uso y ventana de cola. No se fijan aquí valores nuevos.
- Si los límites de falsos negativos por volumen alto tienen un valor de aceptación definido por el uso previsto.

Si alguna decisión implica ampliar de forma sustancial el contrato de features o la pregunta científica, se actualizará este diseño antes de implementar esa ampliación.

### Protocolo A→B congelado: implementación sintética y límites

Se implementan APIs separadas para preregistrar, predecir y evaluar: el primer paso recibe políticas y eventos planeados, la inferencia no recibe etiquetas y el scoring exige una predicción sellada. El registro identifica par, contrato, estudio y hash, redes/corridas, objetivo, semillas, umbrales y criterios de reporte. Se vuelve a aplicar la política de admisión de dominio y la política de calidad de las corridas; las corridas no elegibles se excluyen y documentan en un informe separado; sin corridas elegibles o con rechazo de dominio no se produce un score.

La procedencia del estudio y su pertenencia exclusiva a A debe generarse y auditarse desde el pipeline real; una declaración manual coherente no demuestra ausencia de fuga. Aún se debe vincular los hashes de eventos con los archivos efectivamente usados en SWMM y las tablas de features/etiquetas con los reportes y datasets de la corrida. Estas limitaciones impiden declarar listo el experimento real. El detalle y los campos requeridos están en `docs/experimentos/plantilla-transferencia-A-a-B.md`. Este trabajo usa solamente fixtures sintéticas.


**Corrección posterior a revisión independiente:** `score_transfer` está bloqueada incondicionalmente hasta que el proyecto implemente una vinculación verificable entre INP y eventos efectivos, calidad y reporte SWMM por corrida, cobertura nodal y archivos de features/etiquetas. `PreparedDataset.load` actualmente valida estructura y contrato, pero no verifica los contenidos parquet contra esa procedencia; cargarlo no basta para abrir la barrera. El helper privado y sus archivos `synthetic_scores.json` son solo pruebas, sin validez sobre B real. La cobertura y el `shape_id` exigidos por el prototipo se verifican contra la lista preregistrada declarada, que todavía debe derivarse y comprobarse desde la fuente hidráulica. El MLP de transferencia exige el ensemble persistido de cinco semillas y su regla aprobada; un par de una sola semilla se rechaza. La tarea 5 real permanece incompleta.

**Actualización de integridad de artefactos:** `PreparedDataset.load` reconstruye ahora el frame ordenado de entrenamiento con `keys.parquet`, `features.parquet` y `targets.parquet`, y lo compara con `training_frame_sha256`. También regenera los folds desde los protocolos declarados y rechaza asignaciones alteradas en `folds.parquet`. Estas comprobaciones detectan cambios en filas y particiones, pero aún no vinculan features o targets al `.rpt/.out`, no verifican los archivos de entrada de SWMM ni autentican la cobertura nodal esperada. `score_transfer` sigue bloqueada.
