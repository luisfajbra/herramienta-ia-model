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

El **RMSE macro por red**, promedio simple de los dos RMSE anteriores, es **234,54 m³ para XGBoost** y **184,63 m³ para MLP**. En esta reserva MLP tiene menor error de volumen; XGBoost obtiene mayor precisión y recall de clasificación. Son métricas distintas y no establecen un ganador universal para todos los objetivos.

El RMSE global agrupado por pares es 234,88 m³ para XGBoost y 191,88 m³ para MLP; el R² global es 0,786 y 0,857, respectivamente. Este RMSE global no coincide necesariamente con el macro: Chico Norte aporta más nodos y, por tanto, pesa más en el cálculo agrupado. Las paridades muestran ese cálculo global; la métrica principal acordada para comparar redes es el macro.

Durante la selección interna los mejores valores macro fueron 58,89 m³ para XGBoost y 74,73 m³ para MLP. La reserva tiene errores considerablemente mayores, pero se trata de particiones y poblaciones de escenarios distintas: la diferencia por sí sola no demuestra una causa única ni una capacidad de generalización a otras redes.

## Análisis descriptivo de errores

La forma `shape_quasi_uniform_block` concentra **93,38 %** de la suma de errores cuadrados agrupada de XGBoost y **81,32 %** de la de MLP. Su RMSE es 583,8 y 563,0 m³ en Chico Sur y Chico Norte para XGBoost, y 398,2 y 443,0 m³ para MLP. Los factores altos tienen errores particularmente grandes. Esta contribución se examinó después de obtener los resultados: es un diagnóstico, no un criterio para retirar escenarios ni una nueva métrica principal. Debe revisarse la respuesta hidráulica de esa forma y la cobertura del entrenamiento antes de atribuir el fenómeno a una causa.

Para ese mismo hidrograma, las curvas del volumen total según el factor están en [Chico Sur](../../figures_reserva_diagnostico_20261108/volumen_factor_chico_sur.png) y [Chico Norte](../../figures_reserva_diagnostico_20261108/volumen_factor_chico_norte.png). La forma fue seleccionada después de calcular los errores. En Chico Norte, XGBoost subestima el volumen total en factores altos; MLP se aproxima al total de SWMM en parte del rango y lo supera hacia el extremo. La cercanía del total no garantiza que el volumen se ubique correctamente nodo por nodo.

La revisión posterior de la forma, los dictámenes QA y la cobertura de variables por nodo se documenta en el [diagnóstico hidráulico y de cobertura](2026-11-08-diagnostico-hidraulico-cobertura.md). Esa revisión es descriptiva y no atribuye causalidad al rango de entrenamiento.

Los volúmenes totales negativos en la columna de sesgo indican subestimación agregada respecto a SWMM; un sesgo total pequeño puede ocultar errores en nodos individuales. Los resultados siguen sujetos a la admisión provisional de Chico Norte, el origen no aclarado de sus caudales iniciales y el QA ya documentado; no sustituyen una validación de campo.

## Figuras

Las figuras usan datos sellados y se encuentran en [`docs/figures_reserva_conjunta_20261108_v2`](../../figures_reserva_conjunta_20261108_v2). Hay una paridad global por modelo y mapas separados de SWMM, MLP y XGBoost para cada red. Cada comparación de mapas de una misma red y escenario comparte escala de colores.

- [Paridad global XGBoost](../../figures_reserva_conjunta_20261108_v2/paridad_global_xgboost.png) y [paridad global MLP](../../figures_reserva_conjunta_20261108_v2/paridad_global_mlp.png).
- Ejemplo `common_storm`, factor 3: Chico Sur ([SWMM](../../figures_reserva_conjunta_20261108_v2/mapa_chico_sur_swmm_factor_3.png), [MLP](../../figures_reserva_conjunta_20261108_v2/mapa_chico_sur_mlp_factor_3.png), [XGBoost](../../figures_reserva_conjunta_20261108_v2/mapa_chico_sur_xgboost_factor_3.png)); Chico Norte ([SWMM](../../figures_reserva_conjunta_20261108_v2/mapa_chico_norte_swmm_factor_3.png), [MLP](../../figures_reserva_conjunta_20261108_v2/mapa_chico_norte_mlp_factor_3.png), [XGBoost](../../figures_reserva_conjunta_20261108_v2/mapa_chico_norte_xgboost_factor_3.png)).

Los mapas del factor 1,0 se habían fijado antes de consultar los resultados; SWMM no reportó volumen de inundación en ninguna de las dos redes para ese caso. Se conservan en la misma carpeta. Los mapas de factor 3 se añadieron **después de observar los resultados** para ilustrar un caso con inundación y no son evidencia independiente adicional. La escala es común entre los tres modelos para cada red y factor; puede diferir entre redes o factores. En SWMM se muestran todos los volúmenes positivos reportados, mientras que los modelos muestran el volumen después de la decisión de clasificación.
