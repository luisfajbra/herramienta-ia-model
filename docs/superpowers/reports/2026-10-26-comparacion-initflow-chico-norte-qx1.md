# Comparaci?n diagn?stica de condiciones iniciales: Chico Norte Qx1

Fecha: 2026-10-26.

## Alcance

Se ejecut? con PySWMM 2.1.0 / SWMM Toolkit 0.17.0 una pareja de simulaciones sobre el INP de Chico Norte Qx1 actualizado. La corrida `current` conserva exactamente el archivo fuente, SHA-256 `1c46db327638e023f86f7f0061774abe708f01b8df625f6b7c58af36a1ace1`. La corrida `zero` se cre? como copia, poniendo en cero los `InitFlow` de las 522 conducciones. Se verific? que las ?nicas diferencias entre INP son esos 522 campos y que las profundidades iniciales de nodos son id?nticas. Los 61 caudales de la serie 455C siguen en cero en la fuente.

El evento disponible dura tres horas. Las dos corridas utilizaron las mismas condiciones hidrol?gicas, opciones, geometr?a y motor. No se modific? la fuente durante la prueba. Los artefactos, hashes y resultados detallados est?n en `outputs/experiments/initial_conditions_chico_norte_qx1_20261026/`; incluyen INP, RPT, OUT, manifiesto JSON y tablas CSV por nodo y conducci?n.

## Resultados

| M?trica | InitFlow actual | InitFlow cero | Diferencia cero ? actual |
|---|---:|---:|---:|
| Error de continuidad | 2,971% | ?0,066% | ?3,037 puntos porcentuales |
| Volumen de inundaci?n | 0 m? | 0 m? | 0 m? |
| Estado terminal de drenaje | `not_drained` | `not_drained` | ? |
| Nodos con cambio de profundidad m?xima | ? | ? | 274 de 523 |
| M?xima diferencia de profundidad de series | ? | ? | 0,895 m |
| Conducciones con cambio de caudal m?ximo absoluto | ? | ? | 300 de 522 |
| M?xima diferencia de series de caudal | ? | ? | 3.103 L/s |

El mayor cambio en profundidad m?xima fue 0,0746 m en 126C. El mayor cambio de caudal m?ximo absoluto fue 148,4 L/s en T521. Los cambios de series pueden ser mayores que los cambios de m?ximos porque los caudales pico pueden ocurrir en tiempos distintos.

Ambas continuidades cumplen el umbral provisional de 5%. Los reportes SWMM no muestran advertencias ni errores. La clasificaci?n `not_drained` no permite admitir estas corridas: el INP termina a las tres horas, mientras la configuraci?n provisional exige observar seis horas de drenaje.

## Interpretaci?n y l?mites

Los valores actuales de `InitFlow` cambian materialmente la respuesta hidr?ulica para este evento. Que ambas corridas cumplan el umbral provisional de continuidad no demuestra que cualquiera de los dos estados iniciales sea f?sicamente correcto. El estado de caudal cero se us? ?nicamente como comparador solicitado; no se selecciona como estado general de la red. Un evento de tres horas tampoco satisface los casos adicionales del plan (evento sin inundaci?n, evento largo y pulso corto) ni el periodo de drenaje.

Estas salidas son diagn?sticas, no fueron insertadas en SQLite ni habilitan entrenamiento o admisi?n de red. La selecci?n del estado de inicializaci?n para experimentos adicionales requiere una base hidr?ulica del proyecto o los eventos/configuraciones que correspondan.

## Verificaci?n de software en esta sesi?n

Los tests unitarios de la interpretaci?n com?n del caudal (incluyendo `10 L/s`, `Sfactor=2`, `Base=5 L/s`, unidades no LPS, patr?n, valores no finitos, series que empiezan despu?s de `t=0` y series distintas por nodo) est?n presentes en `tests/simulation/test_effective_inflow.py`; las 6 pruebas sin archivos temporales pasaron. Las 20 pruebas de `tests/extraction/test_tributary_features.py` tambi?n pasaron. La suite focalizada de 59 casos no pudo completar porque Pytest recibi? `PermissionError (WinError 5)` al acceder/limpiar los directorios `basetemp` de Windows; los casos que requieren INP temporal quedaron con error y no se pueden declarar verificados en esta sesi?n. La simulaci?n diagn?stica en s? s? produjo y valid? ambos RPT/OUT mediante SWMM.
