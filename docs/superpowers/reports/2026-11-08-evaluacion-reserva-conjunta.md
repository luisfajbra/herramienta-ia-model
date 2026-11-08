# Evaluación de los modelos conjuntos con formas reservadas

**Fecha:** 8 de noviembre de 2026. Los modelos ajustados conjuntamente con Chico Sur y Chico Norte se evaluaron contra SWMM sobre siete formas apartadas del desarrollo, incluidos todos sus factores elegibles. Este informe no mide transferencia a una red no vista. Los lotes se habían explorado en etapas anteriores del proyecto, por lo que la reserva no constituye una muestra completamente virgen de la investigación.

## Diseño y trazabilidad

- Las 321 corridas evaluadas comprenden 168 de Chico Sur y 153 de Chico Norte; producen 26.880 y 79.866 pares nodo–escenario, respectivamente: **106.746 pares por modelo**. Las corridas que fallaron el QA provisional acordado permanecen excluidas.
- Las formas reservadas fueron `common_storm`, `rare_extreme`, `shape_delayed_abrupt_peak`, `shape_flashy_long_tail_9h5`, `shape_front_loaded_broad`, `shape_quasi_uniform_block` y `shape_twin_plateau_15h`. Los modelos finales se ajustaron solo con las 1.332 corridas de desarrollo.
- Las predicciones de XGBoost y del ensemble MLP se guardaron y sellaron antes de leer las etiquetas para el cálculo de métricas. La tabla de etiquetas procede de la base conjunta sellada, cuyas fuentes SWMM y resultados fueron verificados durante su materialización y se revalidaron antes de la evaluación. La clasificación usa umbral de probabilidad 0,5; un nodo clasificado como seco recibe volumen predicho cero.
- El umbral para considerar inundación en las etiquetas es 1 m³. Se mantuvieron los cortes de 24, 108 y 344 m³ para desglosar falsos negativos. Las métricas completas, por forma, red y corrida, se encuentran en `outputs/experiments/evaluacion_reserva_conjunta_20261108_v1/scores.json`; el sello, las predicciones y las etiquetas están en ese mismo directorio.
- Los controles independientes verificaron los hashes del sello, predicciones, etiquetas y reporte y reprodujeron las métricas macro a partir de los 106.746 pares por modelo.

## Resultados por red

| Modelo | Red | RMSE volumen (m³) | MAE volumen (m³) | Precisión | Recall | F1 | Sesgo del volumen total |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | Chico Sur | 233,86 | 50,92 | 92,97 % | 97,82 % | 95,33 % | −16,96 % |
| XGBoost | Chico Norte | 235,23 | 48,36 | 93,23 % | 97,23 % | 95,19 % | −15,56 % |
| MLP | Chico Sur | 170,79 | 32,68 | 83,73 % | 92,86 % | 88,06 % | −1,82 % |
| MLP | Chico Norte | 198,47 | 34,65 | 85,61 % | 93,78 % | 89,51 % | −0,78 % |

El **RMSE macro por red**, promedio simple de los dos RMSE anteriores, es **234,54 m³ para XGBoost** y **184,63 m³ para MLP**. En esta reserva MLP tiene menor RMSE macro de volumen y XGBoost obtiene mayor precisión y recall de clasificación. **La ventaja de MLP en volumen depende de una sola forma:** sin `shape_quasi_uniform_block`, el RMSE macro es 66,68 m³ para XGBoost y 87,23 m³ para MLP. Por forma y red el resultado está repartido (tabla siguiente). Este análisis se hizo después de ver los resultados: es descriptivo, no cambia la métrica principal acordada ni selecciona un modelo.

RMSE de volumen por forma y red (m³). Menor es mejor.

| Forma | XGB Sur | XGB Norte | MLP Sur | MLP Norte |
|---|---:|---:|---:|---:|
| common_storm | 49,3 | 48,0 | 39,5 | 35,3 |
| rare_extreme | 2,7 | 5,8 | 4,0 | 4,4 |
| shape_delayed_abrupt_peak | 40,8 | 40,8 | 28,1 | 27,2 |
| shape_flashy_long_tail_9h5 | 20,2 | 23,1 | 25,5 | 23,1 |
| shape_front_loaded_broad | 24,1 | 21,3 | 29,9 | 28,7 |
| **shape_quasi_uniform_block** | **583,8** | **563,0** | **398,2** | **443,0** |
| shape_twin_plateau_15h | 147,0 | 130,0 | 183,2 | 203,9 |

Fuente: outputs/experiments/auditoria_reserva_conjunta_20261108_v1/audit.json (scripts/auditar_reserva_conjunta.py).

El RMSE global agrupado por pares es 234,88 m³ para XGBoost y 191,88 m³ para MLP; el R² global es 0,786 y 0,857, respectivamente. Este RMSE global no coincide necesariamente con el macro: Chico Norte aporta más nodos y, por tanto, pesa más en el cálculo agrupado. Las paridades muestran ese cálculo global; la métrica principal acordada para comparar redes es el macro.

Durante la selección interna los mejores valores macro fueron 58,89 m³ para XGBoost y 74,73 m³ para MLP. La reserva tiene errores considerablemente mayores, pero se trata de particiones y poblaciones de escenarios distintas: la diferencia por sí sola no demuestra una causa única ni una capacidad de generalización a otras redes.

## Análisis descriptivo de errores

La forma `shape_quasi_uniform_block` concentra **93,38 %** de la suma de errores cuadrados agrupada de XGBoost y **81,32 %** de la de MLP. Su RMSE es 583,8 y 563,0 m³ en Chico Sur y Chico Norte para XGBoost, y 398,2 y 443,0 m³ para MLP. Los factores altos tienen errores particularmente grandes. Esta contribución se examinó después de obtener los resultados: es un diagnóstico, no un criterio para retirar escenarios ni una nueva métrica principal. Debe revisarse la respuesta hidráulica de esa forma y la cobertura del entrenamiento antes de atribuir el fenómeno a una causa.

Para ese mismo hidrograma, las curvas del volumen total según el factor están en [Chico Sur](../../figures_reserva_diagnostico_20261108/volumen_factor_chico_sur.png) y [Chico Norte](../../figures_reserva_diagnostico_20261108/volumen_factor_chico_norte.png). La forma fue seleccionada después de calcular los errores. En Chico Norte, XGBoost subestima el volumen total en factores altos; MLP se aproxima al total de SWMM en parte del rango y lo supera hacia el extremo. La cercanía del total no garantiza que el volumen se ubique correctamente nodo por nodo.

La revisión posterior de la forma, los dictámenes QA y la cobertura de variables por nodo se documenta en el [diagnóstico hidráulico y de cobertura](2026-11-08-diagnostico-hidraulico-cobertura.md). Esa revisión es descriptiva y no atribuye causalidad al rango de entrenamiento.

Los volúmenes totales negativos en la columna de sesgo indican subestimación agregada respecto a SWMM; un sesgo total pequeño puede ocultar errores en nodos individuales. Los resultados siguen sujetos a la admisión provisional de Chico Norte, el origen no aclarado de sus caudales iniciales y el QA ya documentado; no sustituyen una validación de campo.

## Sensibilidad post hoc

Esta sección resume el análisis [`scripts/auditar_reserva_conjunta.py`](../../../scripts/auditar_reserva_conjunta.py), ejecutado el 8 de noviembre sobre las predicciones ya selladas. Todo es **post hoc** y está rotulado `post_hoc_audit_not_model_selection`: no reemplaza las métricas oficiales de `scores.json`, no reentrena nada y no selecciona un modelo. La métrica principal sigue siendo `macro_network_rmse`. Las cifras salen de `outputs/experiments/auditoria_reserva_conjunta_20261108_v1/audit.json`. La resolución completa de cada hallazgo está en el [documento de resolución](2026-11-08-resolucion-hallazgos-auditoria.md).

**Cobertura del volumen de entrada.** En ambas redes, la forma con más volumen de entrada en desarrollo es `shape_sustained_plateau_14h` (225.639 m³ en Chico Sur y 670.283 m³ en Chico Norte). En la reserva es `shape_quasi_uniform_block` (262.697 m³ y 780.366 m³). El máximo de la reserva supera al de desarrollo en una razón de 1,164 en las dos redes (16,4 % más), y cuatro corridas de la reserva en cada red lo superan (`max_ratio_to_development_max`). El sorteo de formas fue correcto y quedó fijado de antemano; por azar dejó en la reserva la forma de mayor volumen. Esto no invalida la evaluación, pero significa que la reserva mide en parte extrapolación. La función `stratified_shape_draw` queda disponible para el diseño de una reserva futura; no se usó para volver a sortear esta reserva, que ya fue consultada y no debe reutilizarse como prueba ciega.

**Coherencia física.** El volumen inundado de un nodo no puede superar el volumen que le llega de sus tributarios. Con una tolerancia de 1 m³, XGBoost tiene 0 violaciones de esa cota (`tributary_bound.violations`) y MLP tiene 46, con una razón máxima de 2,27 veces la cota (`max_ratio`). Si se recortaran las predicciones a la cota, el RMSE macro de MLP bajaría de 184,63 a 174,29 m³ (`capped_macro_network_rmse_m3`); es un **recorte post hoc, no adoptado**, que XGBoost no necesita (queda en 234,54 m³). En el total por forma y factor hay una violación de monotonía en MLP (Chico Sur, `shape_quasi_uniform_block`, el volumen total predicho baja 3.170,65 m³ al pasar al factor 5,0 mientras SWMM sube) y ninguna en XGBoost. Por nodo, la predicción baja al subir el factor, sin que SWMM baje, en 63 de 101.972 pasos para MLP y en 1.242 para XGBoost (1,2 %). La explicación es que el regresor aprende `log1p(volumen)`: el MLP extrapola de forma aproximadamente lineal en el espacio logarítmico, es decir exponencial en m³, mientras que XGBoost se satura porque los árboles no predicen por encima de lo visto.

**Continuidad.** La QA admite corridas con error de continuidad de hasta 5 %. Se recalculó el RMSE macro y la clasificación con subconjuntos más estrictos (`continuity_sensitivity`):

| Límite de \|continuidad\| | Corridas | Pares | RMSE macro XGBoost (m³) | RMSE macro MLP (m³) |
|---|---:|---:|---:|---:|
| Oficial (≤ 5 %) | 321 | 106.746 | 234,54 | 184,63 |
| ≤ 2 % | 283 | 93.064 | 249,96 | 196,91 |
| ≤ 1 % | 245 | 79.744 | 268,87 | 211,94 |

| Límite | Modelo | Precisión Sur / Norte | Recall Sur / Norte |
|---|---|---:|---:|
| ≤ 2 % | XGBoost | 94,3 % / 93,5 % | 98,0 % / 97,6 % |
| ≤ 2 % | MLP | 85,5 % / 86,0 % | 93,1 % / 93,9 % |
| ≤ 1 % | XGBoost | 94,4 % / 93,9 % | 98,2 % / 97,8 % |
| ≤ 1 % | MLP | 85,7 % / 86,4 % | 93,7 % / 94,5 % |

El orden entre modelos no cambia con los límites más estrictos, y la precisión y el recall varían menos de 2 puntos porcentuales respecto a los valores oficiales (XGBoost 93,0 % / 93,2 % de precisión y 97,8 % / 97,2 % de recall; MLP 83,7 % / 85,6 % y 92,9 % / 93,8 %). El RMSE macro sube al restringir; no se investigó la causa, pero el subconjunto cambia la mezcla de corridas y de formas que se promedia. Si se cita la clasificación cerca del umbral de 1 m³, conviene acompañarla de esta sensibilidad.

## Figuras

Las figuras usan datos sellados y se encuentran en [`docs/figures_reserva_conjunta_20261108_v2`](../../figures_reserva_conjunta_20261108_v2). Hay una paridad global por modelo y mapas separados de SWMM, MLP y XGBoost para cada red. Cada comparación de mapas de una misma red y escenario comparte escala de colores.

- [Paridad global XGBoost](../../figures_reserva_conjunta_20261108_v2/paridad_global_xgboost.png) y [paridad global MLP](../../figures_reserva_conjunta_20261108_v2/paridad_global_mlp.png).
- Ejemplo `common_storm`, factor 3: Chico Sur ([SWMM](../../figures_reserva_conjunta_20261108_v2/mapa_chico_sur_swmm_factor_3.png), [MLP](../../figures_reserva_conjunta_20261108_v2/mapa_chico_sur_mlp_factor_3.png), [XGBoost](../../figures_reserva_conjunta_20261108_v2/mapa_chico_sur_xgboost_factor_3.png)); Chico Norte ([SWMM](../../figures_reserva_conjunta_20261108_v2/mapa_chico_norte_swmm_factor_3.png), [MLP](../../figures_reserva_conjunta_20261108_v2/mapa_chico_norte_mlp_factor_3.png), [XGBoost](../../figures_reserva_conjunta_20261108_v2/mapa_chico_norte_xgboost_factor_3.png)).

Los mapas del factor 1,0 se habían fijado antes de consultar los resultados; SWMM no reportó volumen de inundación en ninguna de las dos redes para ese caso. Se conservan en la misma carpeta. Los mapas de factor 3 se añadieron **después de observar los resultados** para ilustrar un caso con inundación y no son evidencia independiente adicional. La escala es común entre los tres modelos para cada red y factor; puede diferir entre redes o factores. En SWMM se muestran todos los volúmenes positivos reportados, mientras que los modelos muestran el volumen después de la decisión de clasificación.
