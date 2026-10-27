# Recalificación de red A y preparación de modelos

## Resultado de la recalificación

Se reejecutaron como experimento nuevo las 1.000 combinaciones de `chico_hydro-qx1`: 39 formas actuales más la forma base del INP, cada una con 25 factores de 0,2 a 5,0. El INP fuente no cambió durante las simulaciones (SHA-256 `ea48571b013b9749cd6292b1b4e3be620f397bb61dbee795a53cdee88782cb89`). Se conservaron las condiciones iniciales tal como estaban en el archivo; su origen hidráulico sigue sin estar documentado.

Las 1.000 salidas `.rpt` se validaron y no faltan reportes. Con el filtro de software acordado provisionalmente, 963 escenarios son elegibles y 37 se excluyen únicamente por superar el 5% de error absoluto de continuidad. Todas las corridas tienen cobertura de 160 nodos y estado de drenaje `drained`. Ese resultado describe los criterios computacionales usados; no equivale a admitir hidráulicamente la red ni justifica por sí solo los límites provisionales.

Se observa volumen positivo de inundación en 794 escenarios elegibles. El máximo agregado es 183.407 m³ para `shape_quasi_uniform_block_f5.000`, con 93 nodos por encima del umbral de 1 m³. En esa corrida, los mayores volúmenes nodales son 7.211 m³ en 36C, 6.845 m³ en 8C y 6.155 m³ en 102C. Estos valores sintéticos de caudal pico conservado requieren revisión de unidades, escalas y plausibilidad antes de interpretarlos como escenarios de diseño.

Los artefactos de corridas y protocolo están en `outputs/experiments/requalification_chico_hydro_qx1_20261026/`. El finalizador confirmó 1.000 filas QA únicas, 963 elegibles, cero reportes faltantes, cero errores de worker y que el INP fuente no cambió. El cierre `linear_v1` agregó un punto terminal de caudal cero en 100 escenarios (14.700 pares escenario-nodo).

## Dataset aislado A-only

Se reconstruyeron las features y etiquetas usando los extractores y reglas activos del proyecto. El resultado contiene 154.080 filas (963 corridas × 160 nodos) y una copia SQLite aislada con 963 registros explícitos de calidad elegible. `PreparedDataset 758d3a0ad79e8506` pasó la validación del banco con los contratos y protocolos configurados. Ni la base histórica/canónica ni `dataset_final.csv` fueron modificados.

El entrenamiento representa un solo INP y una sola red fuente. La división por formas permite estudiar sensibilidad a las formas sintéticas dentro de esa red, pero no establece generalización entre redes. Las features de entrada actuales incluyen rasgos geométricos/topológicos, flujos pico del INP y descriptores escalares de la forma; no incluyen la serie completa del hidrograma.

## Ajuste y evaluación de modelos

El estudio Optuna XGBoost terminó 50 trials de cinco folds agrupados por `shape_id`. Su mejor RMSE interno fue 108,8445 m³ por nodo. El clasificador seleccionado usa profundidad 5, `learning_rate=0,13587`, 313 árboles y `subsample=0,7909`; el regresor usa profundidad 8, `learning_rate=0,19891`, 421 árboles y `subsample=0,9100`. El candidato final A-only quedó ajustado con 154.080 filas y 46.225 filas inundadas para regresión, y sus metadatos validan hashes pareados de los artefactos. El estudio está etiquetado por el código como `estimacion_exploratoria_condicionada`; se usó únicamente A y no sirve como resultado confirmatorio. Los parámetros y los 50 trials están en `outputs/experiments/requalification_chico_hydro_qx1_20261026/tuning/xgboost/`.

El estudio MLP terminó 50 trials de 200 épocas en modo secuencial de un hilo porque el sandbox Windows denegó los pipes requeridos por `loky`. El mejor RMSE interno fue 117,0970 m³/nodo (trial 35), con arquitectura de clasificador `(128, 64)` y regresor `(64, 32, 16)`. El candidato MLP se ajustó luego con todo A elegible. Los dos estudios A-only están etiquetados como estimaciones exploratorias condicionadas; sus scores internos no son confirmatorios ni se usaron para afirmar generalización.

## Prueba congelada de transferencia A→B

Una vez fijados los parámetros y artefactos de A, se evaluaron una sola vez los 859 escenarios B que pasaron el filtro de software; se excluyeron 116/975 por continuidad >5%. Son 448.398 filas nodo-escenario (522 nodos por escenario). No se usaron etiquetas B para tuning, umbral o selección de familia. Los artefactos y métricas reproducibles están en `outputs/experiments/requalification_chico_hydro_qx1_20261026/frozen_b_evaluation/`.

| Modelo | A-only CV RMSE (m³/nodo) | B AUC inundación | B F1 | B RMSE end-to-end (m³/nodo) | Volumen total predicho / real |
|---|---:|---:|---:|---:|---:|
| MLP | 117,10 | 0,885 | 0,739 | 773,58 | 67,65 / 41,60 millones m³ (1,63×; +62,6%) |
| XGBoost | 108,84 | 0,926 | 0,766 | 273,79 | 24,21 / 41,60 millones m³ (0,58×; −41,8%) |

XGBoost supera al MLP en RMSE end-to-end y volumen agregado para este ensayo congelado, aunque aún subestima el volumen total aproximadamente 41,8%; el MLP lo sobreestima aproximadamente 62,6%. XGBoost tiene mayor precisión de clasificación (0,875 frente a 0,719) y AUC, pero menor recall (0,680 frente a 0,760), así que pierde más nodos inundados. En regresión oracle sobre nodos realmente inundados, XGBoost obtiene RMSE 466,35 m³ y NSE 0,266, mientras MLP obtiene RMSE 1.354,54 m³ y NSE −5,190. El error mediano por escenario aumenta con los factores de escala: a factor 5,0 es 259,0 m³/nodo para XGBoost y 711,9 para MLP. Por tanto, ninguno demuestra precisión hidráulica transferible suficiente en todo el rango probado.

Hay desajuste claro entre las features de A y B: 46,2% de los valores de `elev_fondo` y 43,3% de `dist_outfall_m` de B quedan fuera del rango observado en A. Además, `diam_max_in` y `upstream_capacity_lps` faltan en 45,8% de los nodos B. Esas filas se procesaron con el pipeline del contrato existente, pero indican que la geometría/topología de B está parcialmente fuera del soporte de entrenamiento y deben figurar en cualquier interpretación del resultado.

La prueba reutiliza las 39 formas sintéticas y los factores 0,2–5,0 de A; mide transferencia a una segunda red bajo formas compartidas, no generalización a lluvias no vistas. B tampoco cuenta con admisión hidráulica formal: los 859 escenarios solo son elegibles por el filtro de software provisional del 5%, y siguen abiertos el origen de `InitFlow`, la justificación del límite de continuidad y la interpretación física de los hidrogramas sintéticos. **Conclusión: estos candidatos pueden servir para investigación y diagnóstico, pero no están listos para respaldar predicciones en redes nuevas ni decisiones de diseño.** Hace falta admitir hidráulicamente B, investigar el desajuste de features y evaluar en redes y eventos independientes con caudales defendibles antes de alegar generalización.

## Límites hidráulicos pendientes

- El 5% de continuidad y los criterios actuales de drenaje, tolerancias y cierre son provisionales; requieren justificación hidráulica.
- Los `InitFlow` originales se conservaron; su procedencia y significado todavía no están resueltos.
- Los perfiles de entrada se escalaron para conservar picos nodales desde los datos actuales; no son mediciones de lluvia/caudal ni hidrogramas calibrados.
- El INP de Chico Norte B sigue sin admisión hidráulica formal. El cambio de 455C a cero fue una decisión del usuario sobre el archivo de entrada, no una verificación de campo.
- Un único ensayo A→B no basta para afirmar transferencia general a otras redes ni a lluvias no vistas.

