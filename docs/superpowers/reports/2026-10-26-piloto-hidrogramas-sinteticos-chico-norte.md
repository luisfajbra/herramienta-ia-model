# Piloto sint?tico de hidrogramas: Chico Norte Qx1

Fecha: 2026-10-26. Resultado exploratorio; no es una admisi?n hidr?ulica ni un conjunto de entrenamiento.

## Dise?o ejecutado

Se usaron las formas versionadas `shape_ultra_short_impulse.csv` (0,5 h) y `shape_sustained_plateau_14h.csv` (14 h). Por decisi?n del usuario se conservaron los picos nodales: para cada uno de los 522 nodos con FLOW se multiplic? `q_norm(t)` por el m?ximo caudal efectivo de ese nodo en el INP actualizado. El factor fue 1. Los 521 nodos con pico positivo conservaron ese pico de referencia; 455C permaneci? en cero. Esta operaci?n redistribuye el caudal en el tiempo y cambia el volumen total de cada forma; los picos heredados de un solo evento de tres horas tampoco son caudales de dise?o validados.

Cada forma se ejecut? con `InitFlow` fuente y con todos los `InitFlow` de conducciones en cero, en INP separados. Se comprob? que los INP pareados difieren solo en esos 522 campos. Tras la duraci?n de cada forma se a?adieron seis horas de caudal cero; al no drenar, se aplic? la configuraci?n aprobada de hasta dos extensiones de dos horas. No se modific? la fuente. Hash de fuente: `1c46db327638e023f86f7f0061774abe708f01b8df625f6b6c7f58af36a1ace1`. Motor: PySWMM 2.1.0 / SWMM Toolkit 0.17.0.

## Resultados finales

| Forma | InitFlow | Cola final | Continuidad | Inundaci?n total | Drenaje |
|---|---|---:|---:|---:|---|
| Pulso 0,5 h | Fuente | 10 h | 12,721% | 0 m? | `drained` |
| Pulso 0,5 h | Cero | 10 h | ?0,182% | 0 m? | `drained` |
| Evento 14 h | Fuente | 8 h | 0,574% | 0 m? | `drained` |
| Evento 14 h | Cero | 8 h | 0,000% | 0 m? | `drained` |

La corrida corta con `InitFlow` fuente excede el l?mite provisional de continuidad de 5%, y ampliar la cola no lo corrigi?. Las otras tres corridas quedan bajo 5% y drenaron usando las tolerancias provisionales. Los RPT no contienen l?neas de advertencia o error del motor.

El estado inicial tuvo efectos grandes, en particular para el pulso corto: entre las dos variantes cambiaron los picos de profundidad de 523/523 nodos y los picos de caudal de 522/522 conducciones; la diferencia m?xima de profundidad pico fue 0,405 m, la de caudal pico 1.749 L/s, y las diferencias m?ximas entre series fueron 0,885 m y 3.195 L/s. Para la forma de 14 h cambiaron los picos de profundidad de 183/523 nodos y los picos de caudal de 226/522 conducciones; las diferencias m?ximas fueron 0,137 m y 204 L/s. Los CSV por nodo y conducci?n se conservan junto con cada salida.

## Decisi?n y l?mites

El piloto confirma que SWMM ejecuta estas dos formas, pero **las cuatro variantes produjeron cero inundaci?n**. Por lo tanto este piloto no aporta ejemplos positivos para entrenar el clasificador de inundaci?n. Adem?s, el estado inicial cambia mucho la hidr?ulica, y la variante fuente del pulso corto falla el 5% provisional. No se debe seleccionar o mezclar los estados iniciales para entrenamiento bas?ndose solo en este piloto.

La generaci?n us? formas normalizadas sint?ticas y amplitudes tomadas del pico de un evento contenido en el propio INP. No son lluvia observada, hidrogramas medidos independientes ni caudales de dise?o. La admisi?n de red contin?a `policy_incomplete` con 19 valores de pol?tica pendientes. No se guard? ning?n resultado en SQLite ni en el conjunto can?nico.

Para convertir esto en una evaluaci?n ?til falta congelar una regla de amplitudes/intensidades que produzca el dominio de eventos previsto y ejemplos de ambas clases; cerrar las decisiones de admisi?n; justificar o resolver el estado inicial; y mantener B fuera del ajuste hasta completar la transferencia A?B con MLP/XGBoost congelados. El manifiesto JSON con hashes, configuraciones, resultados de cada intento y comparaciones se guarda en `outputs/experiments/synthetic_hydrograph_pilot_chico_norte_20261026/pilot_manifest.json`.
