# Evaluación de modelos conjuntos con diez escenarios de Chico Norte

## Alcance y procedencia

Se evaluaron los candidatos finales MLP y XGBoost entrenados conjuntamente con Chico Sur y Chico Norte, frente a los resultados SWMM ya conservados de dos hidrogramas sintéticos en Chico Norte. Cada forma se combinó con los factores 0,7; 1,3; 2,7; 3,9 y 4,3: **10 simulaciones**, 522 nodos por simulación y **5.220 pares nodo–escenario**. Los escenarios originales estaban separados de los datos utilizados para entrenar los candidatos conjuntos, pero se habían evaluado y examinado anteriormente. Esta evaluación es **exploratoria y posterior**, no una prueba ciega ni una evaluación de una red desconocida. No se usaron sus resultados para seleccionar modelos y no se reentrenó.

Los caudales son hidrogramas de entrada nodal sintéticos, no lluvia medida. Los factores son distintos de la grilla anterior y están dentro de su intervalo 0,2–5,0. Se reutilizaron los reportes SWMM, no se volvieron a simular escenarios. Las diez corridas habían pasado el QA provisional: continuidad absoluta ≤5 %, cobertura finita y drenaje; no se usó no convergencia como exclusión adicional. Se conservó el estado inicial original de Chico Norte, cuyo origen no se ha aclarado.

Las predicciones se obtuvieron con los candidatos de `outputs/experiments/final_joint_candidates_20261108_v1/{xgboost,mlp/ensemble}` y el contrato `tabular_v4_21`. Las entradas provinieron de los CSV sellados de la evaluación anterior: se verificó su SHA-256 contra ambos sellos y que ambas familias tuvieran idénticas entradas. La rutina histórica `load_verified_frame` comprobó los sellos de predicciones históricas, el diseño, el manifiesto y los RPT de SWMM, de los cuales se reconstruyeron las etiquetas. Se comprobó la correspondencia uno a uno entre pares nodo–corrida. Estos controles preservan trazabilidad; el hecho de escribir predicciones nuevas no convierte este lote ya observado en ciego.

Los resultados están en `outputs/experiments/conjuntos_diez_hidrogramas_chico_norte_20261108_v1/`: variables, etiquetas y predicciones en Parquet, `comparacion.parquet` y `resultados.json`, con hashes de modelos, entradas y figuras. Las formas de este lote no figuran entre las formas de entrenamiento del candidato XGBoost revisado; la red sí participó en el desarrollo de ambos candidatos. El umbral de inundación de referencia es 1 m³ y el umbral de clasificación 0,5; los volúmenes se ponen en cero si el clasificador predice seco. El MLP es un ensamble de cinco semillas.

## Resultados globales

RMSE y MAE se calculan sobre los 5.220 pares, no como promedio de los diez RMSE. El sesgo compara la suma de volúmenes predichos con la suma de los reportados por SWMM. Hay 1.433 pares con inundación de al menos 1 m³.

| Modelo conjunto | RMSE (m³) | MAE (m³) | Precisión | Recall | F1 | Sesgo del volumen total |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 174,76 | 44,48 | 91,27 % | 96,30 % | 93,72 % | +71,74 % |
| MLP | 50,68 | 12,95 | 86,08 % | 94,07 % | 89,90 % | +15,61 % |

En este lote, el MLP tiene menor error de volumen y XGBoost mayor precisión y recall. El sesgo positivo significa sobreestimación agregada; no implica que todos los nodos o escenarios estén sobreestimados. No debe generalizarse esta comparación a redes o eventos no estudiados.

## Por hidrograma

| Forma | RMSE XGBoost (m³) | RMSE MLP (m³) | Volumen SWMM (m³) | Volumen XGBoost (m³) | Volumen MLP (m³) |
|---|---:|---:|---:|---:|---:|
| Pulsos dobles | 46,33 | 18,69 | 84.943 | 118.981 | 96.467 |
| Meseta sostenida | 242,76 | 69,19 | 238.173 | 435.940 | 277.087 |

Los volúmenes son la suma de los cinco factores de cada forma, no el volumen de una tormenta única. En la meseta con factor 4,3, SWMM reporta 116.608 m³, XGBoost predice 209.188 m³ y MLP 134.438 m³. Un volumen total próximo tampoco garantiza localización correcta en los nodos.

## Figuras y verificación

Las 23 figuras están en `docs/figures_conjuntos_diez_hidrogramas_chico_norte_20261108_v1`: dos hidrogramas, 18 mapas separados SWMM/MLP/XGBoost para los factores 0,7; 2,7 y 4,3, una paridad global por modelo y las curvas de volumen total por factor. Los mapas muestran inundación sobre la red y tienen escalas individuales; los colores de mapas distintos no deben compararse sin leer sus barras. Las paridades usan puntos y ejes `log10(1 + volumen)`; las métricas se calculan en m³ originales.

Se recalcularon de forma independiente los RMSE y sesgos desde `comparacion.parquet`; coincidieron con `resultados.json`. Se verificaron los hashes de las 23 figuras. La prueba de alineación rechaza nodos ausentes o duplicados y pasó. Las curvas por factor se agregaron tras la primera ejecución a partir de las mismas predicciones; el script de reproducción quedó actualizado para generarlas junto con las demás figuras. El script se ejecuta con `venv/Scripts/python.exe scripts/evaluacion_conjunta_hidrogramas_nuevos.py` y se niega a sobrescribir sus directorios de salida.

La evaluación y las figuras locales también se agruparon en el [respaldo 44 de Drive](https://drive.google.com/file/d/1Ec-HpQ5TWhGdQ9fNfedXsWgahB0SA1Lu/view?usp=drivesdk), junto con su [manifiesto](https://drive.google.com/file/d/14FshOg1aUVPSvwZBLzkrskhQNXfNnvtK/view?usp=drivesdk). El ZIP local contiene 30 archivos, pasó la verificación de sus miembros y tiene SHA-256 `33097353121aac964dd0db20fb8b1aec9e1fbe58dbe887b1b76ff9e1b71eab6f`. Los metadatos de Drive confirman carpeta privada y tamaño de 6.824.936 bytes. Queda pendiente descargar y verificar el hash de la copia remota. El informe dentro del ZIP corresponde a la versión anterior a la adición de este párrafo de respaldo. `outputs/` permanece ignorado por Git.
