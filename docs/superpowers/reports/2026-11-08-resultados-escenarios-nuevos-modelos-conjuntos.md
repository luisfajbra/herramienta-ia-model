# Resultados: escenarios nuevos para modelos conjuntos

## Protocolo y fuentes

El [protocolo previo](2026-11-08-protocolo-escenarios-nuevos-modelos-conjuntos.md) fijó dos formas sintéticas nuevas de caudal por nodo y los factores 1,9; 2,1; 2,5; 2,9 y 3,7 antes de calcular resultados. Son factores que no figuran en la grilla de entrenamiento conjunto ni en la prueba anterior de diez escenarios. La ejecución fue en Chico Norte, red incluida en el entrenamiento de ambos candidatos; no es una prueba de transferencia a una red desconocida ni de validación con mediciones.

La ejecución `outputs/experiments/conjuntos_escenarios_nuevos_chico_norte_20261108` conserva la selección, las series por nodo, el protocolo anclado, los INP efectivos, RPT/OUT y manifiestos. El archivo de ancla SQLite está junto al directorio de ejecución. Las diez simulaciones terminaron, todas fueron elegibles con el QA provisional acordado y no hubo exclusiones. Los errores de continuidad estuvieron entre 0,105 % y 1,208 % y los escenarios se declararon drenados; no se aplicó una exclusión adicional por no convergencia. Se mantuvieron las condiciones iniciales originales de la red.

Los hashes de metadatos, clasificadores y regresores de los modelos conjuntos quedaron registrados antes de la predicción. La base temporal `joint_staging.sqlite3` se construyó con el verificador de fuentes y permaneció sin etiquetas nodales; las etiquetas se extrajeron de los RPT verificados después de sellar las predicciones. La carpeta `evaluacion_modelos_conjuntos` contiene variables, claves, predicciones, etiquetas, sellos, métricas y cobertura. Se usaron las 21 variables `tabular_v4_21`, umbral de inundación de 1 m³ y umbral de clasificación de 0,5. El MLP es un ensamble de cinco semillas.

## Resultados agrupados

Los valores siguientes se calculan sobre **5.220 pares nodo–escenario**; SWMM reportó 1.766 pares con inundación ≥1 m³. RMSE no es el promedio de diez RMSE y las filas no representan redes independientes.

| Modelo conjunto | RMSE (m³) | MAE (m³) | Precisión | Recall | F1 | Sesgo de volumen total |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 51,12 | 16,64 | 90,27 % | 98,19 % | 94,06 % | +24,41 % |
| MLP | 27,02 | 7,92 | 85,14 % | 94,39 % | 89,53 % | +0,03 % |

Para estas formas y factores, MLP obtuvo menor error volumétrico; XGBoost identificó una proporción mayor de los nodos con inundación. Un sesgo total cercano a cero no garantiza buena localización ni errores pequeños por escenario: aquí el MLP sobreestima el volumen del ascenso escalonado y subestima el del pulso, y ambos efectos se compensan al sumar el conjunto.

| Hidrograma, cinco factores agrupados | RMSE XGBoost (m³) | RMSE MLP (m³) | Volumen SWMM (m³) | XGBoost (m³) | MLP (m³) |
|---|---:|---:|---:|---:|---:|
| Ascenso escalonado y recesión | 50,35 | 36,34 | 246.252 | 278.752 | 252.845 |
| Pulso agudo y recesión | 51,88 | 11,80 | 86.948 | 135.788 | 80.456 |

Los volúmenes de esta tabla suman cinco simulaciones: no representan una sola tormenta. Las métricas de cada combinación están en `scores.json`. En el ascenso escalonado con factor 3,7, los RMSE nodales son 79,93 m³ para XGBoost y 61,79 m³ para MLP; en el pulso con el mismo factor son 73,32 y 19,28 m³, respectivamente. Los resultados son específicos de esta selección sintética.

## Interpolación y extrapolación

Los factores son nuevos pero están dentro del rango 0,2–5,0 de entrenamiento: la prueba de factor es de **interpolación**. Las formas temporales son nuevas. Para evaluar explícitamente extrapolación, se comparó cada variable de cada nodo con sus mínimos y máximos de desarrollo en Chico Norte. En la forma de ascenso escalonado, `tiempo_primer_pico_tributario_h` vale 9 h y supera el máximo de desarrollo correspondiente a cada nodo en **2.610 pares**, que son los 522 nodos en los cinco factores. Así hay extrapolación univariada comprobada en esa variable. La forma de pulso tiene tiempo al pico de 1,5 h.

Las demás variables comparables no superaron sus rangos univariados por nodo; algunas tienen valores nulos y no cuentan con límite comparable. Estar dentro de rangos individuales no demuestra estar dentro del dominio conjunto de las 21 variables. La comparación de rangos por nodo está en `cobertura_por_variable.json`; no acredita generalización a redes nuevas.

## Figuras y verificación

Se generaron 23 figuras en `docs/figures_conjuntos_escenarios_nuevos_chico_norte_20261108`: dos hidrogramas, 18 mapas individuales de SWMM, MLP y XGBoost para los factores fijados 1,9; 2,5 y 3,7 de cada forma, dos paridades separadas y una figura de curvas por factor. Los tres mapas de cada escenario comparten escala de color; muestran inundación sin marcas de errores. Las paridades muestran ejes `log10(1 + volumen)`; sus errores se calculan en m³.

Se recalcularon independientemente los RMSE globales desde las predicciones y etiquetas guardadas: coincidieron con `scores.json`. Se verificaron los SHA-256 de las 23 figuras y no faltó ningún mapa planificado. Las pruebas focalizadas de novedad y correspondencia nodo–corrida pasaron. La evaluación, los protocolos, los modelos fijados y las figuras deben conservarse juntos y respaldarse externamente; `outputs/` está ignorado por Git.

## Respaldo

Se creó el ZIP local `outputs/backups/45_escenarios_nuevos_conjuntos_chico_norte_20261108.zip`, de 109.808.200 bytes y SHA-256 `b98827f676c8c9f1c31d46c8168c6928468609af9348ee6159f093662f9b4aa4`. Sus 134 archivos incluyen simulaciones, INP y reportes, protocolo anclado, selección, predicciones, figuras, informes y scripts de evaluación; los modelos conjuntos originales permanecen en sus respaldos previos. El ZIP y sus miembros pasaron verificación local.

La integración de Drive rechazó el ZIP completo por superar su límite de 100 MiB. Se subieron cinco partes `part00` a `part04`, el [manifiesto de partes](https://drive.google.com/file/d/1swP5nEM0rqfzcdJL-2u0IJGOLCJq7Bvi/view?usp=drivesdk) y el [manifiesto de archivos](https://drive.google.com/file/d/1Lua5Fy0Dds1F2YkhnDZCyBVTS0kMIzzD/view?usp=drivesdk) a la [carpeta privada de la tesis](https://drive.google.com/drive/folders/1chsofbBXPKxzUNbwYir5wmgw4K6tSk0G). Se verificó localmente que las partes, concatenadas en orden, reconstruyen el hash del ZIP. Los metadatos de Drive confirman la presencia y el tamaño de cada parte, pero no se verificó por descarga el hash de la copia remota. Esta sección se añadió después de generar el ZIP; la copia del informe dentro de ese ZIP no la contiene.
