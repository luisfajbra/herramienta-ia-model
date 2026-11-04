# Guion de la presentación semanal · 2 de noviembre de 2026

[Presentacion PowerPoint](../../Presentacion_MLSMFP_02-11_avance_chico_norte_mapas_separados_inundacion.pptx).

La presentacion parte del deck aprobado del 30 de octubre. Conserva 14 diapositivas, retira las diapositivas 13–15 del original porque contenian mapas comparativos con marcas de errores, actualiza la fecha y añade 24 diapositivas nuevas: hidrogramas, mapas individuales, paridades y rectas por factor.

## Transición

“En este corte dejamos congelados los candidatos MLP y XGBoost entrenados con Chico Sur. Probamos dos formas sintéticas de caudal por nodo en Chico Norte, con cinco factores seleccionados reproduciblemente. No corrimos lluvia–escorrentía ni reentrenamos los modelos. Las figuras que siguen comparan las salidas con SWMM.”

## Diapositivas 15–16 · Hidrogramas

“Primero vemos cada forma por separado, con la misma curva, marcadores y relleno usados en la gráfica anterior del hidrograma. Son pulsos dobles y una meseta sostenida con un pico tardío. El eje vertical muestra caudal normalizado, no litros por segundo observados. Cada forma se aplicó a los nodos usando el pico nodal de referencia del INP y el factor del escenario. Se probaron los factores 0,7; 1,3; 2,7; 3,9 y 4,3. Los factores 0,7, 2,7 y 4,3 se usan también en los mapas que vienen a continuación.”

## Diapositivas 17–34 · Mapas de inundación

“Después mostramos por separado los mapas de SWMM, MLP y XGBoost para los factores 0,7; 2,7 y 4,3 de cada hidrograma. Se ve la red en gris y únicamente los nodos inundados, coloreados por volumen con su barra de color. No se dibujan nodos secos, cruces ni anillos. Como cada mapa tiene su propia escala, los colores se interpretan junto con su barra y no se comparan directamente entre escenarios.”

“Con factor 0,7, SWMM no reportó volumen de inundación en ninguna de las dos formas; por eso para los modelos se informa el volumen adicional en metros cúbicos, no un porcentaje relativo a cero. En el factor 2,7, los totales SWMM fueron 11.224 m³ para pulsos dobles y 30.829 m³ para la meseta sostenida. En factor 4,3, fueron 41.008 m³ y 116.608 m³, respectivamente.”

## Diapositivas 35–36 · Paridades globales

“Ahora cada modelo tiene su propia gráfica de paridad. Cada punto representa el volumen predicho frente al volumen SWMM para un nodo en un escenario. La línea roja discontinua es igualdad. Los ejes simétricos logarítmicos permiten mostrar ceros y volúmenes altos sin agrupar los datos en hexágonos. Cada panel reporta métricas sobre los mismos 5.220 pares; no son promedios de diez métricas por escenario.”

“XGBoost tuvo RMSE 186,18 m³, precisión 82,2%, recall 55,9%, 632 omisiones y 174 falsas alarmas. MLP tuvo RMSE 186,26 m³, precisión 66,2%, recall 89,6%, 149 omisiones y 656 falsas alarmas. El error volumétrico global es casi igual, pero el equilibrio entre omisiones y falsas alarmas es distinto.”

## Diapositiva 37 · Rectas por factor

“Al final de las figuras dejamos las rectas que comparan volumen total por factor. Cada panel corresponde a uno de los dos hidrogramas y las líneas comparan SWMM, XGBoost y MLP. Esta vista muestra cómo crece el volumen total con el factor y deja visible que XGBoost tiende a subestimar los eventos altos en este lote.”

## Diapositiva 38 · Conclusiones

“Los diez escenarios pasaron la QA vigente, pero continuidad del 5% y las tolerancias hidráulicas siguen siendo provisionales. El lote sirve para comparar estos candidatos en Chico Norte con formas sintéticas nuevas. No demuestra confiabilidad general en otras redes y no se incorporaron estos datos al entrenamiento.”

“Lo que sigue es revisar desacuerdos por nodo, ampliar formas y factores, y validar más redes. Antes de admitir Chico Norte a entrenamiento hay que justificar QA y revisar convergencia. Si luego se reentrena con Chico Norte, reservaremos una red o escenarios independientes para verificar si la transferencia mejora de verdad.”

## Respuestas breves

- **¿Son lluvias observadas?** No; son hidrogramas sintéticos de caudal por nodo.
- **¿Por qué se muestran como curvas?** La línea une los caudales normalizados en el tiempo y conserva el estilo de la gráfica de hidrograma presentada anteriormente; cada forma tiene su propia diapositiva.
- **¿Por qué hay seis mapas?** Se eligieron tres factores (bajo, medio y alto) para cada una de las dos formas.
- **¿Por qué los mapas tienen distintas intensidades de color?** Cada mapa tiene su propia barra de color y escala; compara los valores con la barra de ese mismo mapa.
- **¿Qué significa RMSE 186 m³?** Es la raíz del error cuadrático medio agrupando los 5.220 pares nodo–escenario, no el promedio de diez RMSE.
- **¿Cuál modelo ganó?** No se declara ganador: MLP omitió menos inundaciones y produjo más falsas alarmas; XGBoost fue más preciso y omitió más.
- **¿Esto valida cualquier red?** No; solo se probó una red objetivo y dos formas sintéticas.
