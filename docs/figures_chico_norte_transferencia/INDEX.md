# Figuras de transferencia: Chico Sur → Chico Norte

Estas figuras describen la evaluación externa congelada de XGBoost y el ensemble MLP entrenados únicamente con Chico Sur y evaluados en Chico Norte. Se generaron después de cerrar el score; no cambiaron modelos, umbrales, selección de candidatos ni métricas oficiales.

## Archivos

- [Paridad de volumen nodal](paridad_volumen_chico_norte.png): cuatro paneles con todos los pares corrida–nodo y solo los nodos inundados por SWMM.
- [Mapa de volumen bajo, p10](mapa_inundacion_bajo_chico_norte.png)
- [Mapa de volumen mediano, p50](mapa_inundacion_mediano_chico_norte.png)
- [Mapa de volumen alto, p90](mapa_inundacion_alto_chico_norte.png)
- [Manifiesto de generación y hashes](figure_manifest.json)

El manifiesto JSON contiene hashes de las fuentes y las imágenes, regla de selección, resúmenes agregados por modelo y versiones del software. No contiene etiquetas por nodo.

## Población y controles

La paridad usa los 787 escenarios aceptados por QA, 522 nodos por corrida y 410.814 pares corrida–nodo. Los 188 escenarios excluidos no entran en las figuras. La etiqueta real se reconstruye en memoria desde los RPT ligados a cada corrida; el SHA-256 canónico reconstruido (`e52cd199f1a0de11e98461007ed3313e80bb895a851c7ed6b9863b0a0489529f`) coincide con ambos reportes de score. También se verifican los sellos de registro, entradas y predicciones, los hashes de candidato, INP, manifiestos y RPT, las decisiones QA, la cobertura exacta de nodos y la reproducción de las métricas generales.

La base SQLite se abre en modo de solo lectura y se comprueba que `node_results` siga en cero. El script no crea archivos de objetivos por nodo; conserva solamente las imágenes y este manifiesto de métricas agregadas.

## Selección de mapas

Se calcula el volumen SWMM total de cada escenario elegible. Se excluyen de esta selección los escenarios con menos de 1 m³ de inundación total; sobre los restantes se calculan cuantiles lineales p10, p50 y p90. Para cada cuantil se elige el escenario elegible más cercano al valor objetivo. Los tres escenarios deben ser distintos; los empates se resuelven ordenando `scenario_key` alfabéticamente. Es una selección descriptiva y no modifica el score externo.

| Nivel | Cuantil objetivo | Escenario seleccionado | Total SWMM | Nodos inundados SWMM |
|---|---:|---|---:|---:|
| Bajo | 1.886,3 m³ (p10) | `B:shape_micro_burst_45min:f2.400` | 1.886,0 m³ | 163 |
| Mediano | 31.130,0 m³ (p50) | `B:new_hydrograph:f4.200` | 31.263,0 m³ | 310 |
| Alto | 134.346,3 m³ (p90) | `B:shape_double_peak_equal_separated:f4.400` | 134.462,0 m³ | 273 |

## Cómo leerlas

La paridad conserva ceros usando el cambio de eje `log10(1 + volumen en m³)` y muestra densidad de pares en escala logarítmica. La diagonal roja es igualdad entre SWMM y el volumen predicho enrutado. Los paneles superiores incluyen todos los pares y reproducen el RMSE/MAE usados en el score: XGBoost 270,86/64,32 m³; MLP 435,93/74,95 m³. Los inferiores restringen el análisis a los 131.502 pares inundados por SWMM (umbral de 1 m³): RMSE 477,94 m³ para XGBoost y 748,07 m³ para MLP. Esta segunda vista es condicional y sus RMSE no reemplazan la métrica oficial agregada.

Los tres mapas muestran SWMM, XGBoost y MLP en paneles contiguos, con la misma geometría del INP y una escala de color común para los volúmenes. Los nodos secos aparecen en gris; los falsos positivos del modelo llevan un círculo morado y los falsos negativos una cruz roja. La clase real se define como volumen SWMM ≥1 m³ y la clase predicha se toma de `y_pred_clf` sellado. El color de los nodos usa una transformación raíz para que los volúmenes pequeños sigan siendo visibles; no es una escala lineal de color.

| Nivel | Modelo | Volumen predicho | RMSE por nodo | Falsos positivos | Falsos negativos |
|---|---|---:|---:|---:|---:|
| Bajo | XGBoost | 852 m³ | 9,64 m³ | 15 | 93 |
| Bajo | MLP | 1.787 m³ | 9,45 m³ | 80 | 29 |
| Mediano | XGBoost | 13.435 m³ | 98,71 m³ | 14 | 143 |
| Mediano | MLP | 31.583 m³ | 106,35 m³ | 77 | 42 |
| Alto | XGBoost | 68.301 m³ | 402,46 m³ | 20 | 110 |
| Alto | MLP | 148.706 m³ | 445,99 m³ | 103 | 22 |

En estos tres casos XGBoost subestima el volumen total y omite más nodos inundados. MLP se acerca más al total en los casos bajo y mediano y omite menos nodos, pero genera más falsos positivos; en el caso alto sobreestima el volumen. Son ejemplos seleccionados por volumen real, no una muestra aleatoria ni un nuevo criterio para declarar ganador.

## Límites de interpretación

Las coordenadas del INP permiten representar la red espacialmente, pero no se verificó un sistema de referencia geográfica; los mapas son esquemas de red, no mapas base georreferenciados. Las entradas son hidrogramas sintéticos de caudal por nodo. Los 410.814 pares comparten una sola red y escenarios relacionados, por lo que no son observaciones independientes ni prueban generalización a otras redes o eventos observados. Los umbrales hidráulicos de QA continúan siendo provisionales.

## Regeneración

Desde la raíz del repositorio:

```powershell
.\venv\Scripts\python.exe scripts/figures_part2/make_chico_norte_transfer_figures.py
```

El comando requiere los artefactos originales en `outputs/`, que Git ignora. Se detiene si los hashes, etiquetas, modelos, entradas, scores, QA o cobertura dejan de coincidir. También rehúsa sobrescribir figuras existentes; para regenerar, archiva primero la carpeta actual y revisa el cambio esperado.
