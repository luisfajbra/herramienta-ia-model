# Revisión de admisión estructural: Chico Norte Qx1

## Identificación y trazabilidad

- INP: `data/networks/chico_norte_93_qx1/SWMM - Chico Norte 93% - Qx1.00.inp`
- SHA-256: `9f8792abecd86b90187b010acb98cc2147ad114fb6be199d5417f79cd2cc45e7`
- Tamaño: 1,525,204 bytes. La copia coincide byte por byte con el archivo indicado por el usuario.
- Parser local: `swmm-api 0.4.73`.
- Motor SWMM disponible en el entorno: `5.2.4` (build `52004`). El INP no declara la versión de motor que originó sus condiciones iniciales o series.
- El inventario reproducible se genera con `scripts/audit_network_inp.py`; sus salidas viven en `outputs/network_admission/` y no junto al INP.

## Hechos que expone el INP

- Unidades: caudal `LPS`, geometría en metros según las unidades SWMM.
- Red: 522 junctions, 1 outfall (`523C`, tipo `FREE`), 522 conducciones y 523 nodos con coordenadas. Un componente débilmente conectado, sin ciclos dirigidos ni pares de enlaces paralelos.
- Todas las conducciones tienen sección circular y un barril. El modo de offsets es `DEPTH`; 177 conducciones tienen al menos un offset distinto de cero.
- No hay subcuencas ni pluviómetros. El alcance de este archivo es entrada por hidrogramas de caudal nodal, coherente con la etapa hidráulica seleccionada; no es una simulación lluvia–escorrentía.
- Hay 522 aportes de constituyente `FLOW`, uno por nodo, cada uno con serie propia. Las 522 series cubren 0–3 h, con 61 puntos separados 3 minutos. Son curvas nodales de un único escenario sincronizado, no 522 eventos. Se contaron 514 formas temporales únicas después de normalizar cada curva por su máximo absoluto y redondear la razón a 9 decimales.
- Los factores de escala y unidad de los aportes son 1. No hay patrón declarado. La base declarada de 455C es cero.
- No se detectaron referencias estructurales sin resolver dentro del alcance listado por el perfil. Las expresiones de `[CONTROLS]` se inventarían, pero sus referencias internas no se analizan todavía.

## Hallazgos que requieren resolución

1. La serie de `455C` es negativa en sus 61 puntos: mínimo `−0.226 L/s`, máximo `−0.011 L/s` y valor final `−0.017 L/s`. El usuario indicó que desconoce su origen. Debe rastrearse si representa extracción/retorno intencional o un problema de datos antes de entrenar.
2. Las 522 conducciones tienen `InitFlow` distinto de cero (`1.5` a `4529.9992 L/s`), mientras las 522 junctions empiezan con profundidad `0`. El usuario desconoce cómo se generaron esos caudales y pidió comparar estados. No se debe reemplazarlos por cero ni tratarlos como estado calibrado sin evidencia.
3. El nombre de la red indica Chico Norte, pero se reutilizan identificadores de la red Chico presente en entrenamiento. En la comparación estructural local, 77 IDs de junction coinciden; coinciden 160 IDs de conduit y los 160 tienen pares de extremos distintos. El outfall cambia de `109C` a `523C`. La red requiere un `network_id` propio y no se deben emparejar enlaces o nodos únicamente por nombre.
4. El perfil del INP no identifica el motor SWMM usado para generar estas series ni el origen/calibración de `InitFlow`. La versión disponible localmente (`5.2.4`) describe el motor instalado para trabajo futuro, no la procedencia del archivo.

## Estado de admisión

El inventario estructural está completo y es de solo lectura. La política hidráulica todavía está incompleta: no se creó una política que declare admisible a Chico Norte y el CLI la reporta como `policy_incomplete`. Esto no es una aprobación o rechazo de calidad hidráulica. No se ejecutó SWMM, no se añadió la red ni resultados a la base de entrenamiento y no se habilitó entrenamiento multired.

Antes de simulaciones comparativas faltan el origen verificable de los hidrogramas y de `InitFlow`, la definición de estados alternativos científicamente relevantes, la versión que se usará para reproducirlos y las reglas de continuidad/cierre que permitirán comparar resultados.

## Interpretación del flujo efectivo y límites de generalización

- Para una entrada SWMM de constituyente `FLOW`, el flujo total se interpreta como `Base + Sfactor × serie`; fuera de la cobertura temporal de la serie, su componente es cero y permanece la base. El motor local 5.2.4 y el evaluador compartido coinciden en una fixture: `Base=5`, `Sfactor=2`, `serie=10` produce `25 L/s`. Los CSV de escenarios son flujos absolutos, por lo que su INP derivado neutraliza base, escala y patrón previos. [Manual oficial EPA SWMM 5.2](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P10145M6.TXT)
- La política ahora rechaza, por defecto, series con muestras no finitas o flujo efectivo negativo, fuentes activas `[DWF]`, `[RDII]`, `[GROUNDWATER]` o `[SNOWPACKS]` no declaradas, patrones sin evaluador, y geometría `CUSTOM`, `IRREGULAR` o `STREET` con referencias faltantes. Cada issue incluye `object_id`. La inspección actual de Chico Norte sigue en `policy_incomplete`; estos controles no equivalen a una aprobación hidráulica.
- El grafo de extracción ahora conserva el identificador de cada conducción paralela y suma la capacidad proxy de todas las conducciones inmediatas de entrada. La fixture de dos tubos confirma que no se pierde uno de los enlaces.
- El banco ML deja filtrar por varios `network_id`, pero sus protocolos publicados son `LOSO`/`GroupKFold5` por factor y `LeaveOneShapeOut` por forma. No hay todavía `LeaveOneNetworkOut` ni ajuste de hiperparámetros encapsulado dentro del conjunto de entrenamiento de cada red reservada. Por ello el código actual no puede producir una estimación independiente válida de generalización a Chico Norte, y los OOF por forma del entrenamiento existente no la sustituyen.
- Aún no se debe entrenar el modelo conjunto A+B con Chico Norte. Primero debe cerrarse la admisión y ejecutarse A→B con modelos e hiperparámetros congelados desde A. Tras usar las etiquetas de B para ajustar, B pasa a ser red de desarrollo; haría falta una tercera red independiente para medir transferencia a una red verdaderamente futura.

## Verificación de software de esta revisión

- 44 pruebas enfocadas de admisión, política, flujo efectivo, escenarios y escalado pasaron, incluida una comparación con `Node.lateral_inflow` del motor SWMM local.
- Las pruebas de topología paralela, features tributarias y contrato v4_21 pasaron (29).
- La suite completa seleccionó 1.336 pruebas: 1.233 pasaron, 53 fallaron y 50 dieron error. Una repetición de casos representativos confirmó `WinError 5` al usar temporales en `%LOCALAPPDATA%\Temp` o en `outputs`, y la prueba de GUI no puede inicializar Tcl en este entorno. También hay tests de predicción que fallan porque su fixture de modelo no incluye `feature_contract_id`, requerido por el validador de procedencia actual. No atribuyo el resto de fallas a estos cambios sin repetirlas en un entorno limpio.


Progress note (2026-10-26): an independent review found that a permissive policy could admit CFS even though the effective FLOW evaluator only supports LPS. Admission now rejects non-LPS units whenever FLOW inflows are present, including when CFS is explicitly listed as allowed; non-FLOW-only inflows are not subject to that evaluator-specific restriction. The focused admission/effective-flow suite passes (47 tests).

Progress note (2026-10-26): fail-closed flood-label extraction now rejects absent, incomplete, duplicate, non-finite, negative, or malformed reports. A review of the existing `outputs/training_v17.sqlite3` was read-only: it contains 1,000 COMPLETE runs, all 1,000 with NULL `continuity_error_pct`; the scenarios also contain no `hydraulic_quality` or `drainage_state` manifest keys. These historical runs therefore have no stored evidence for the run-quality gate.


## Decisiones y trazabilidad para la recalificaci?n de A

- El usuario aprob? volver a ejecutar y recalificar las 1.000 corridas de entrenamiento; tambi?n aprob? conservar provisionalmente el l?mite de continuidad de 5%.
- El usuario aprob? aplicar provisionalmente la configuraci?n existente de cierre (`drain_down_hours=6`, incrementos de 2 h, m?ximo 2 reintentos, tolerancias de 0,01 L/s y 0,01 m?) dejando expl?cito que falta sustento hidr?ulico documentado.
- El usuario autoriz? reconstruir estas corridas como una versi?n nueva desde el INP y los CSV actuales, con hashes, conservando intacta la evidencia hist?rica. No se deben reemplazar ni editar las corridas antiguas.
- Verifiqu? en modo de solo lectura que `outputs/training_v17.sqlite3` tiene una sola red (`Chico Sur`) y que su INP en la base y `data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp` son id?nticos (421.820 bytes; SHA-256 `ea48571b013b9749cd6292b1b4e3be620f397bb61dbee795a53cdee88782cb89`).
- Las 1.000 corridas corresponden a 40 formas por 25 factores (0,2?5,0). La base registra 0 filas en `scenario_inflows`; los archivos de resultados hist?ricos tampoco est?n vinculados por `run_id`. Hay 39 CSV versionados en `data/hydrograph_shapes`; la forma `base` est? representada por el hidrograma del INP. Por ello la reejecuci?n ser? reproducible desde las fuentes actuales, pero no se podr? afirmar que replica bit a bit las entradas ef?meras originales.
- Los 1.000 `run_id` hist?ricos est?n fijados en `training_run_inputs`. La recalificaci?n debe crear evidencia nueva y mantener los antiguos como hist?ricos/no verificados; ning?n campo fijado se editar?.
- La suite completa, antes de corregir el fixture de RPT que no ten?a marcador de cierre, ejecut? 1.367 pruebas: 1.268 pasaron, 55 fallaron y 44 dieron error. Las fallas incluyen restricciones de temporales de Windows y m?dulos ajenos a esta admisi?n; la prueba de lote afectada por el nuevo requisito de cierre fue corregida. Una ejecuci?n posterior de las pruebas de labels, lote y estado de drenaje pas? (42 tests). No he repetido la suite completa despu?s de esa correcci?n.

## Estado del cambio

Se a?adi? rechazo del resumen RPT sin marcador siguiente `Outfall Loading Summary` antes de usar incluso una tabla API no vac?a; se rechaza un identificador aislado despu?s del inicio de filas de datos. Los diagn?sticos del estado de drenaje representan m?tricas ausentes como `null` y clasifican el caso como `unverified`. No se ha ejecutado ninguna de las 1.000 corridas nuevas ni se ha escrito a la base can?nica.

## Control de calidad previo a entrenamiento

En seguimiento a la decisi?n del usuario, los escenarios que no cumplan el control se excluyen y se reportan; el proceso se detiene si no queda ninguno elegible. La ruta principal ahora eval?a cobertura completa y finita, continuidad, drenaje, manifiesto de estado inicial y procedencia del reporte RPT antes de ensamblar el CSV de entrenamiento. El informe queda junto al CSV como `.hydraulic_quality.json` e incluye razones por escenario. La importaci?n `--persist-sql` exige ese informe, adjunta los dictámenes a las corridas elegibles y deja los registros antiguos sin recalificar.

La verificación enfocada combinada pasó (76 pruebas), incluyendo calidad, backfill CSV, CLI, ensamblador y fixtures de simulación. Pytest necesitó un plugin local temporal porque el entorno deniega el acceso a la carpeta temporal predeterminada de Windows. No se ejecutaron SWMM, la recalificación de las 1.000 corridas ni entrenamiento; la red B continúa sin simulaciones y sin uso en entrenamiento. Los RPT malformados ahora se conservan como escenarios no elegibles con el código de error, para que el informe de calidad se escriba antes de detenerse si ninguno pasa.

## Decisión posterior sobre la serie 455C

El usuario indicó reemplazar por cero el aporte de 455C. Se aplicó a las 61 muestras de la serie `455C` en la copia del proyecto `data/networks/chico_norte_93_qx1/SWMM - Chico Norte 93% - Qx1.00.inp`; cada valor ahora es `0.000` entre 0:00 y 3:00. El inventario confirma mínimo y máximo iguales a cero y ninguna muestra negativa.

La fuente original en Descargas y la copia del proyecto ahora contienen el mismo cambio y tienen SHA-256 `1c46db327638e023f86f7f0061774abe708f01b8df625f6b6c7f58af36a1ace1`. El hash anterior a la modificación era `9f8792abecd86b90187b010acb98cc2147ad114fb6be199d5417f79cd2cc45e7`. Este cambio refleja una decisión explícita del usuario; por sí solo no demuestra que el caudal original fuera un error físico ni valida la red. No se ha ejecutado SWMM ni se han entrenado modelos. La procedencia de los `InitFlow` no nulos sigue pendiente.


## Actualizaci?n posterior ? 2026-10-26

Por instrucci?n del usuario, las 61 muestras de la serie de 455C se cambiaron a cero en la fuente entregada y en la copia versionada. El SHA-256 actual del INP es `1c46db327638e023f86f7f0061774abe708f01b8df625f6b7c58af36a1ace1` (1.525.143 bytes); el hash en la secci?n hist?rica de arriba corresponde a la versi?n anterior a ese cambio. La serie de 455C tiene ahora 61 valores iguales a cero.

Tambi?n se ejecut? una pareja diagn?stica de SWMM para el evento original de tres horas, comparando los `InitFlow` actuales con una copia que tiene todos los `InitFlow` de conducciones en cero. El resultado mostr? sensibilidad hidr?ulica a ese estado inicial; no selecciona ninguno como f?sicamente correcto. Los dos casos quedaron `not_drained` al terminar a las tres horas, por lo que no se consideran admisibles ni se usan para entrenamiento. El detalle verificable est? en `docs/superpowers/reports/2026-10-26-comparacion-initflow-chico-norte-qx1.md` y los artefactos en `outputs/experiments/initial_conditions_chico_norte_qx1_20261026/`.

El estado de admisi?n contin?a `policy_incomplete` (19 decisiones de pol?tica sin declarar). A?n no se han generado ni ingresado corridas sint?ticas de entrenamiento para Chico Norte.
