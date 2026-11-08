# Presentación con el formato del 20 de octubre

Esta versión conserva el contenido y los resultados del corte del 30 de octubre, con el formato visual oscuro de `Presentacion_MLSMFP_20-10_avances_auditoria_svm.pptx`: fondo azul marino, títulos Libre Baskerville azul claro, texto Montserrat y acentos dorados. Las figuras se mantienen sobre fondo blanco, como las paridades de la referencia.

Las dos láminas de paridad usan puntos, no hexágonos. XGBoost aparece en azul y MLP en naranja, siguiendo los colores de modelo de la presentación de referencia. Los ejes muestran `log10(1 + volumen)` para conservar los ceros. La línea punteada `y = x` indica igualdad: el volumen predicho coincide con SWMM.

- La paridad global incluye 410.814 pares corrida–nodo de 787 escenarios elegibles. RMSE: XGBoost 270,86 m³; MLP 435,93 m³.
- La paridad de positivos incluye 131.502 pares con volumen SWMM ≥1 m³. Sus RMSE condicionales son 477,94 y 748,07 m³; no sustituyen los globales.

Los cambios de estilo no modifican modelos, predicciones ni resultados. La fuente de datos pasó la misma verificación de hashes y trazabilidad que el score externo. El manifiesto conserva el hash de etiquetas y las métricas de ambas gráficas.

Presentación: `Presentacion_MLSMFP_30-10_avance_chico_norte_y_modelos_formato_20-10_puntos_v2.pptx`. Para regenerar figuras y presentación desde la raíz del repositorio: `venv\Scripts\python.exe scripts/build_2026_10_30_dark_format.py`.
