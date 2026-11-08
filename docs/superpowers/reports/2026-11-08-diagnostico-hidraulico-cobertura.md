# Diagnóstico del hidrograma de mayor error

**Fecha:** 8 de noviembre de 2026. Análisis descriptivo posterior a la evaluación de la reserva, sin reentrenar, ajustar umbrales ni excluir escenarios. El objetivo es distinguir una posible falta de cobertura del entrenamiento de una conclusión sobre estabilidad hidráulica, que este análisis no permite establecer.

## Forma y comprobaciones hidráulicas disponibles

El archivo anclado `shape_quasi_uniform_block.csv` define una forma normalizada de **11 horas**. La integración trapezoidal de `q/qmáx` equivale a **9,57 horas al pico** y la interpolación lineal indica **8,24 horas con caudal normalizado mayor o igual a 0,9**. Son propiedades del hidrograma sintético de entrada, no una caracterización de lluvia observada ni una validación física de la red.

Se revisaron las corridas de factores 2, 3, 4 y 5 en ambas redes. Sus dictámenes registrados indican elegibilidad y estado drenado; sus errores de continuidad están entre **0,042 % y 0,243 %**. Estos controles usan el QA provisional acordado y no comprueban por sí solos la convergencia de todos los pasos ni justifican los caudales iniciales de Chico Norte, cuyo origen sigue sin conocerse.

## Cobertura respecto al entrenamiento

Para cada nodo y red se calculó el máximo de cada variable en las corridas elegibles de desarrollo; después se comparó el valor del mismo nodo para cada escenario reservado. Se verificaron las huellas de las tablas de entrenamiento contra el estudio congelado y la forma contra su ancla. El máximo de entrenamiento es una referencia empírica, no un límite hidráulico físico.

| Variable, factor 5 | Chico Sur | Chico Norte |
|---|---:|---:|
| Caudal pico local por encima del máximo visto en el mismo nodo | 0/160 | 0/522 |
| Volumen tributario integrado por encima del máximo visto en el mismo nodo | 160/160 | 522/522 |
| Volumen local integrado por encima del máximo visto en el mismo nodo | 147/160 | 521/522 |
| Integral de exceso sobre capacidad por encima del máximo visto | 92/103 nodos con valor válido | 263/283 nodos con valor válido |

Para factores 2, 3 y 4, el volumen tributario integrado no superó el máximo del mismo nodo en las corridas de entrenamiento. Esto es relevante: también existen errores en esos factores, de modo que salir del rango marginal en factor 5 no explica por sí solo todo el comportamiento observado. Las variables de exceso sobre capacidad tienen valores faltantes para parte de los nodos; los denominadores anteriores reflejan solo comparaciones válidas, sin imputar datos.

Como comprobación independiente de dos ejemplos, para el nodo `100C` de Chico Sur el máximo de volumen tributario integrado en desarrollo fue 3.366,09 m³ y el valor reservado con factor 5 fue 3.918,92 m³; en el nodo `100C` de Chico Norte fueron 7.201,07 y 8.383,72 m³, respectivamente. Son identidades de nodo locales a redes diferentes, no el mismo nodo físico.

La variable tributaria es una **aproximación de entradas acumuladas**, construida sumando las series del nodo y de sus ancestros únicos en el grafo de conducciones. No es una medición de caudal que pasó por el nodo ni reproduce por sí misma la propagación, el almacenamiento o el remanso hidráulico; esas dinámicas corresponden a SWMM. Esta distinción limita la interpretación física de sus máximos y de la extrapolación detectada.

## Respuesta de volumen y límites

En Chico Norte con factor 5, SWMM reportó un volumen total de **537.489 m³**, XGBoost **261.982,55 m³** y MLP **607.470,27 m³**. En Chico Sur, SWMM reportó **183.407 m³** y MLP **182.383,36 m³**, pero el RMSE nodal de MLP para ese escenario fue aproximadamente **735,6 m³**. Por tanto, un volumen total parecido puede ocultar errores espaciales entre nodos. Las curvas por factor están en el [informe de evaluación](2026-11-08-evaluacion-reserva-conjunta.md).

Los datos respaldan la observación de una forma muy sostenida y de extrapolación en volumen integrado para factor 5; **no prueban que esa extrapolación cause el error**, ni descartan problemas en la generación de variables, la discretización, los reportes o el modelo. La siguiente revisión debe contrastar hidrogramas efectivos, integración y unidades por nodo con los INP/RPT/OUT enlazados y, si se diseña un nuevo entrenamiento, usar una prueba independiente nueva: la reserva actual ya fue consultada.

Reproducción: `scripts/diagnostico_hidraulico_bloque_reservado_20261108.py`. Los resultados detallados, incluidos denominadores por variable, red y factor, y QA de las ocho corridas, están en `outputs/experiments/diagnostico_bloque_reservado_20261108/`.

## Cotejo puntual de CSV, INP, RPT y OUT

Se revisó el nodo local `100C` en cuatro corridas de esta forma: factores 3 y 5 en ambas redes. Se verificaron los hashes de los artefactos enlazados y los sellos de las tablas de entrada y etiquetas. Los 16 puntos del CSV coincidieron con los del INP efectivo en cada caso; el INP añadió un punto de caudal cero para el drenaje. Una integración trapezoidal independiente de las entradas locales y de los ancestros únicos reprodujo las variables guardadas de volumen local y tributario para los cuatro casos. Este cotejo no cubre todos los nodos ni escenarios.

| Red y factor | Volumen de inundación en RPT (m³) | Integral aproximada de la serie OUT (m³) |
|---|---:|---:|
| Chico Sur, 3 | 0 | 0,00 |
| Chico Sur, 5 | 672 | 672,28 |
| Chico Norte, 3 | 801 | 800,44 |
| Chico Norte, 5 | 1.864 | 1.862,91 |

El RPT expresa esa columna en `10^6 ltr`; multiplicar por 1.000 convierte el valor mostrado a m³ y reproduce la etiqueta guardada. La columna mostrada tiene tres decimales en esa unidad: su resolución de presentación es 1 m³, por lo que un `0,000` impreso no demuestra ausencia física exacta de desbordamiento inferior a esa resolución. Las integrales OUT de la tabla usan muestras cada 180 s desde la primera salida, a los tres minutos; omiten el primer intervalo y no son idénticas a la integración interna de SWMM ni una prueba de convergencia. La cercanía en estos cuatro ejemplos no valida el conjunto completo. Reproducción con el lector corregido: `scripts/cotejo_fuentes_bloque_20261108.py`; salida `outputs/experiments/cotejo_fuentes_bloque_20261108_corregido.json`. Se conservó la salida anterior al cambio como `cotejo_fuentes_bloque_20261108.json`.

**Hallazgo técnico independiente y corrección:** antes del cambio, `swmm_resilience.simulation.swmm_api_io.get_node_series` devolvía una columna llamada `time_sec` con el valor entero de una fecha en nanosegundos desde época cuando el índice era `datetime`; la primera muestra devolvía `1735948980000000000`, no segundos transcurridos. Se reprodujo el mismo comportamiento en `get_link_series` con la conducción `T1` de la corrida 700. La búsqueda de llamadas en `swmm_resilience`, `scripts` y `tests` no encontró consumidores de estas funciones, aparte de sus definiciones antes del nuevo cotejo; no hay evidencia de que esa columna haya intervenido en las variables, predicciones o métricas aquí presentadas.

El investigador decidió contar desde el inicio real de la simulación, indicado explícitamente. Las dos funciones ahora exigen `simulation_start` para índices `datetime`, rechazan muestras anteriores a ese inicio y conservan segundos transcurridos para índices de tipo duración. Con el inicio leído del INP efectivo (`2025-01-04 00:00:00`), tanto el nodo `100C` como la conducción `T1` devuelven 180, 360 y 540 segundos para las primeras muestras del OUT; seis pruebas específicas y siete pruebas existentes relacionadas pasaron. Cambiar este archivo de código **no modifica los estudios ni modelos guardados**; una ejecución futura que exija las huellas exactas del código del estudio anterior deberá usar la versión congelada de aquel estudio o declarar una ejecución nueva, sin relajar sus controles.

La batería ampliada de simulación, topología e importaciones terminó con **156 pruebas aprobadas**, incluidas ocho pruebas del contrato temporal. En una ejecución anterior de esa batería falló una prueba del mecanismo de reemplazo de archivos temporales de Windows; esa prueba pasó al repetirse sola y la batería completa pasó al repetirse. Se conserva esta observación sin atribuirle una causa no demostrada.
