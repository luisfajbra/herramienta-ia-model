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

## Respuesta de volumen y límites

En Chico Norte con factor 5, SWMM reportó un volumen total de **537.489 m³**, XGBoost **261.982,55 m³** y MLP **607.470,27 m³**. En Chico Sur, SWMM reportó **183.407 m³** y MLP **182.383,36 m³**, pero el RMSE nodal de MLP para ese escenario fue aproximadamente **735,6 m³**. Por tanto, un volumen total parecido puede ocultar errores espaciales entre nodos. Las curvas por factor están en el [informe de evaluación](2026-11-08-evaluacion-reserva-conjunta.md).

Los datos respaldan la observación de una forma muy sostenida y de extrapolación en volumen integrado para factor 5; **no prueban que esa extrapolación cause el error**, ni descartan problemas en la generación de variables, la discretización, los reportes o el modelo. La siguiente revisión debe contrastar hidrogramas efectivos, integración y unidades por nodo con los INP/RPT/OUT enlazados y, si se diseña un nuevo entrenamiento, usar una prueba independiente nueva: la reserva actual ya fue consultada.

Reproducción: `scripts/diagnostico_hidraulico_bloque_reservado_20261108.py`. Los resultados detallados, incluidos denominadores por variable, red y factor, y QA de las ocho corridas, están en `outputs/experiments/diagnostico_bloque_reservado_20261108/`.
