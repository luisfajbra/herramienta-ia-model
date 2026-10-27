# Lote exploratorio sintético de Chico Norte

**Fecha:** 2026-10-26
**Protocolo:** `chico_norte_exploratory_A_to_B_v1`
**Rol de la red:** B congelada para una futura prueba de transferencia
**Resultado:** 975/975 escenarios simulados y reportes validados; 859 pasan el QA de software provisional.

## Diseño ejecutado

Se ejecutaron 39 hidrogramas normalizados existentes por 25 factores entre 0,2 y 5,0, para 975 escenarios SWMM. En cada nodo, el caudal aplicado fue su máximo `FLOW` efectivo del INP de Chico Norte multiplicado por el factor y la forma normalizada. El INP conserva los `InitFlow` de origen; 455C conserva el hidrograma de caudal cero indicado por el usuario. El hash del INP al inicio y al cierre es `1c46db327638e023f86f7f0061774abe708f01b8df625f6b6c7f58af36a1ace1`.

El protocolo se congeló antes de ejecutar las etiquetas. Aplicó 6 horas de drenaje, extensiones de 2 horas hasta un máximo de dos reintentos y el límite provisional de continuidad absoluta del 5%. Cuando una forma terminaba en caudal distinto de cero, la política explícita `linear_v1` añadió un punto de cierre a cero en el siguiente paso de la malla: ocurrió en 75 escenarios de tres formas (`long_moderate`, `shape_2h`, `shape_5h`), afectando 39.030 pares escenario–nodo. El manifiesto de cierre conserva el detalle por nodo. Los 975 escenarios quedaron drenados y con procedencia de estado inicial. No se escribieron filas en SQLite, no se ensambló un dataset de entrenamiento y no se entrenaron ni ajustaron modelos.

## Resultado de la calificación

Los 975 reportes se pudieron leer, todos contienen etiquetas finitas para las 522 uniones, y no faltan pares forma–factor. El calificador marcó 859 escenarios elegibles por QA de software y excluyó 116 por exceder el 5% provisional de continuidad. No hubo otras causas de exclusión. Este umbral es un filtro exploratorio acordado; no representa calibración hidráulica ni admisión de la red.

Se observó volumen de inundación positivo en 763 escenarios: 723 elegibles por software y 40 excluidos por continuidad. El primer escenario con inundación positiva aparece en el factor 1,2; desde el factor 1,8, las 39 formas tienen volumen positivo. El mayor volumen agregado de inundación fue 537.492 m³, en `shape_quasi_uniform_block_f5.000`, con 288 nodos positivos; el nodo con mayor volumen en ese caso fue `464C` con 14.928 m³. Para esa misma forma, factor 1,0 dio 0 m³ y factor 1,2 dio 2.668 m³ en 19 nodos. La mediana entre escenarios fue 13.391 m³ y el percentil 95 fue 186.262,8 m³. Esta transición rápida y los volúmenes altos deben revisarse hidráulicamente antes de interpretarlos como cargas plausibles de una red real.

## Interpretación y límites

Este lote genera etiquetas SWMM en la red B congelada; **todavía no ejecuta una evaluación de transferencia A→B de MLP o XGBoost**. Esa comparación requerirá ajustar el modelo solo con A y luego predecir B sin usar sus etiquetas para seleccionar arquitectura, hiperparámetros o umbrales. Los hidrogramas son sintéticos y conservan picos nodales, no provienen de observaciones ni de una transformación lluvia–escorrentía.

El software confirma integridad de reportes, cobertura, procedencia, drenaje y continuidad según el protocolo. No confirma que `InitFlow` sea físicamente correcto, que los picos nodales sean compatibles entre sí, ni que la escala de factores hasta 5 represente tormentas de diseño. El volumen alto y la inundación positiva extendida desde factores moderados hacen necesario revisar balances y magnitudes de entrada/salida, los nodos que dominan el volumen, y la justificación de los escenarios antes de considerar estos objetivos aptos para una evaluación de modelos. La admisión de B permanece pendiente y las etiquetas continúan congeladas como prueba exploratoria.

## Artefactos y reproducibilidad

Los archivos de escenario, reportes, salidas SWMM, manifiestos de drenaje, estado inicial y cierre están bajo `outputs/experiments/chico_norte_exploratory_a_to_b_20261026/runs/`. El protocolo congelado está en `protocol.json`; los resultados de QA por escenario, en `scenario_qa.csv`; el resumen y las etiquetas validadas, en `batch_progress.json`; faltantes, en `missing_reports.json`. El script `finalize_batch.py` vuelve a validar los reportes y genera esos artefactos sin acceder a SQLite.

El resumen de ejecución registra cero errores de worker, cero reportes faltantes, INP sin cambios, `sqlite_written=false` y `models_trained=false`. El CSV de QA se verificó con 975 filas. El protocolo y los hashes de los 39 archivos de forma quedan almacenados junto a los resultados.
