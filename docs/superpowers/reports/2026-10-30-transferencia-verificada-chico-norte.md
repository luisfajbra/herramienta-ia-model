# Transferencia congelada Chico Sur → Chico Norte

**Corte:** 2026-10-30 UTC (2026-10-29, hora de Bogotá)

**Estado:** predicciones y scoring externo completados para XGBoost y el ensemble MLP. No se reentrenó ni se ajustó ningún modelo con etiquetas de Chico Norte.

## Resultado

Los dos candidatos pasaron la misma cadena de preregistro, verificación label-free, predicción sellada y extracción de etiquetas desde los RPT autenticados. En 787 escenarios elegibles de Chico Norte se evaluaron 410.814 pares corrida–nodo (787 × 522); 188 escenarios quedaron excluidos por QA. La SQLite de staging conserva cero filas en `node_results`, incluso después del scoring: las etiquetas se extrajeron en memoria y se guardaron únicamente los resúmenes y hashes de auditoría.

| Métrica agregada en Chico Norte | XGBoost | MLP, ensemble de 5 semillas |
|---|---:|---:|
| Pares corrida–nodo | 410.814 | 410.814 |
| Pares inundados reales (umbral 1 m³) | 131.502 | 131.502 |
| Verdaderos positivos | 70.451 | 115.017 |
| Falsos negativos | 61.051 | 16.485 |
| Falsos positivos | 12.976 | 59.879 |
| Precisión | 84,45 % | 65,76 % |
| Recall | 53,57 % | 87,46 % |
| F1 | 65,56 % | 75,08 % |
| PR-AUC | 0,8140 | 0,8184 |
| RMSE de volumen, m³ | **270,86** | 435,93 |
| MAE de volumen, m³ | **64,32** | 74,95 |
| Sesgo total de volumen, m³ | −20.589.535 | +4.132.866 |
| Sesgo relativo | −55,11 % | +11,06 % |

La regla de cascada fue la misma en ambos casos: clasificar inundación con probabilidad ≥ 0,5 y enrutamiento de volumen cero cuando la clase es seca. El sesgo total es `suma(volumen_predicho_enrutado) − suma(volumen_real)` sobre los 410.814 pares, no un sesgo promedio por nodo. Las bandas de falsos negativos usan cortes aprobados con datos de Chico Sur: [1,24], (24,108], (108,344] y >344 m³. El recall por banda fue:

| Banda de volumen real | XGBoost | MLP |
|---|---:|---:|
| [1,24] m³ | 40,32 % | 76,52 % |
| (24,108] m³ | 53,32 % | 87,94 % |
| (108,344] m³ | 59,08 % | 92,37 % |
| >344 m³ | 62,99 % | 94,12 % |

En esta prueba, XGBoost tuvo menor error de volumen y menos falsas alarmas; MLP omitió menos nodos inundados y elevó el recall, a cambio de más falsos positivos y mayor RMSE/MAE de volumen. No se declara un ganador único: depende del costo de omitir una inundación frente al costo de sobrepredecirla. Ninguno queda aprobado como herramienta operativa con estos resultados.

## Qué se entrenó y qué significa el RMSE de selección

Ambos modelos se seleccionaron y ajustaron solo con Chico Sur, contrato `tabular_v4_21` de 21 features, umbral de inundación 1 m³ y objetivo interno `macro_network_rmse`.

| Candidato | Búsqueda | Ensamble final | Mejor objetivo interno |
|---|---:|---|---:|
| XGBoost | 50/50 trials, trial 13 | Una pareja clasificador/regresor | 73,7821 m³ |
| MLP | 50/50 trials, trial 48 | Cinco parejas, semillas 42–46 | 86,4202 m³ |

Estos valores son resultados de selección interna agrupada por forma en Chico Sur; no son los RMSE externos de Chico Norte ni el promedio simple de 975 escenarios individuales. El objetivo calcula RMSE por red y luego lo promedia entre redes del fold; durante la selección actual solo había una red fuente, Chico Sur. Para MLP se promediaron las probabilidades y los volúmenes condicionales en m³ después de aplicar `expm1` a las cinco semillas; el umbral de cascada permaneció en 0,5. El ensemble final no se seleccionó por una semilla individual.

La búsqueda interna fue exploratoria y no constituye una validación independiente sobre nuevas formas físicas. El score externo descrito arriba no se usó para cambiar hiperparámetros, umbrales, semillas ni preprocesamiento.

## Chico Norte, escenarios y control de calidad

El INP congelado de Chico Norte tiene SHA-256 `1c46db327638e023f86f7f0061774abe708f01b8df625f6b6c7f58af36a1ace1`. El protocolo generó 975 corridas a partir de 39 formas de hidrograma y una grilla de 25 factores. Son entradas sintéticas de caudal por nodo, no lluvia observada ni aforos. Treinta y cinco formas tuvieron al menos una corrida elegible; el resumen contiene 787 escenarios elegibles.

El criterio de QA conservó provisionalmente el límite de continuidad del 5 % y la política configurada de drenaje (6 horas iniciales, extensiones de 2 horas y hasta dos reintentos), además de cobertura finita y trazabilidad de RPT/OUT/INP/evento. Se excluyeron 188 escenarios únicos: 116 por continuidad superior al 5 % y 75 por no alcanzar drenaje aceptable; tres escenarios tienen ambos motivos. No se detectaron brechas de cobertura finita. Esos límites siguen pendientes de justificación hidráulica; su aplicación no convierte las simulaciones en observaciones validadas.

Los reportes externos registran 787 enlaces RPT usados y 188 exclusiones. No se extrajeron targets de escenarios excluidos. Ningún target se insertó en SQLite. El SHA-256 canónico de etiquetas derivadas en ambos reportes es `e52cd199f1a0de11e98461007ed3313e80bb895a851c7ed6b9863b0a0489529f`; los dos scores apuntan al mismo manifiesto de fuentes (hash de bytes `68bed0e283ddafbfada42743ae49f28d4611aa1585069a8d2f6b19815313e37b`).

### Política explícita de dominio

Como no había una política persistida para admisión, se registró para este par la política `chico_sur_chico_norte_pair_scope_v1`, después de inspeccionar ambos INP. Los dos usan LPS y metros, junctions/outfalls, conduits de sección circular, caudales FLOW, offset DEPTH y outfall FREE; no contienen lluvia, enlaces paralelos, patrones de entrada, secciones hidráulicas adicionales ni geometría faltante. Ambos pasaron sin incidencias. Se permitió entrada persistente porque los INP examinados contienen colas/baselines no nulos en 147/160 nodos de Chico Sur y 521/522 de Chico Norte; no se cambió ni se filtró esa condición.

Esta política describe solo el dominio observado de Chico Sur y Chico Norte. No es un valor por defecto del proyecto ni una autorización automática para redes futuras. La decisión anterior de dejar en cero el hidrograma negativo de 455C se conservó en el INP congelado; el perfil actual no detecta muestras negativas.

## Incidencias encontradas durante la ejecución

1. **Colisión de IDs SQLite locales.** La preregistración rechazó inicialmente Chico Norte porque ambas bases independientes habían asignado `network_id=1`. Esos enteros son claves locales, no identidad global de red. Se corrigió la admisión para exigir un único B y comparar el SHA-256 del INP B con el SHA-256 fuente de A. La prueba acepta IDs locales iguales con INP distintos y rechaza el mismo INP aunque los IDs locales difieran.
2. **Redondeo de probabilidad `float32`.** El primer intento de scoring XGBoost se detuvo antes de extraer etiquetas: el CSV predeterminado redondeó probabilidades `float32` a siete cifras y el chequeo de reproducción exigía tolerancia `1e-9`. Los volúmenes sí reproducían. Se mantuvieron intactos los CSV ya sellados, se ajustó el chequeo a la precisión de salida del clasificador y las predicciones nuevas se serializan con 17 cifras significativas. El scoring de ambos candidatos se volvió a ejecutar y pasó.

Las dos incidencias se corrigieron antes de usar targets. El intento fallido no creó un reporte de score ni alteró los sellos de predicción. La barrera compara el hash de cada reporte antes de materializar etiquetas en memoria.

## Archivos y trazabilidad

Los artefactos operacionales están bajo `outputs/` (directorios ignorados por Git); los reportes enlazados son los resultados fuente:

- XGBoost: `outputs/bench/multinetwork/20261030T005341Z-813323e3ef75/A_to_B_frozen/verified_scores.json`
- MLP: `outputs/bench/multinetwork/20261030T011406Z-813323e3ef75/A_to_B_frozen/verified_scores.json`
- SQLite label-free: `outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z/B/B_verified_label_free_v1.sqlite3`
- Manifiesto de fuentes B: `outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z/B/experiment_source_manifest.json`
- Candidatos A-only: `outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z/A/candidates/xgboost/` y `.../A/candidates/mlp/ensemble/`

SHA-256 de los artefactos de inferencia y score:

| Artefacto | XGBoost | MLP |
|---|---|---|
| Metadata de candidato | `cdf6ed18a10e2233e9a3670dcac63ac0e2474aa284e1457e865cb41d15444027` | `5da1af38519cab23569e2d73c9671be0471c8ab76f55971a521a66393151c8b8` |
| Clasificador | `a1a6e504cca9a4553e93edeed3340da48c70b5b74167716c4bc96946f376046f` | `0bb1045feeb5bdcacc82ca05b96fe7e382756acb8ab2b501dc450360933ba321` |
| Regresor | `045526a91a4a99a89bde1f473bfc4e67eefa950c5baacf835dcc595c5ecc1b7f` | `a450253d23daba66d032daf3e1723b8fbecd9281f30d3cca06efa01dadba45bb` |
| Predicciones selladas | `105df19f6e1fd178a0215fe9d1120471f3db5bade5d9b01ca4aa5b7a8ac85cdc` | `2e98e5f687587bfef0baab7ed5d3b939f294429416353242575790e3a08e9a0c` |
| Entradas selladas | `4087e16ccc682430bf0d805af718df4b7453a8fcd6ce16cc67f1b3d2d472d363` | igual |
| Reporte de score | `dee28ebd39c8eec4f6dd4c5faa5908f7be9f7d9eb6eb861fb386d06d1363f683` | `ead28ebd5bb396af65cbaf21c476023874d139cc95a486bfce7fa09c789a0555` |

Los estudios de selección también permanecen sellados: XGBoost `5b3ace9ece0223b86c538c63e56ccaff60f6ff37de8e0221021adca57764cd93` y MLP `2644ac54d45df36e6b77f74c29806fa68e0ae0af3adddc9ce8ecade470826fbe`.

## Interpretación y siguientes pasos

El resultado demuestra que los dos candidatos pueden producir predicciones en Chico Norte bajo este contrato, esta política de dominio y estas entradas sintéticas. No demuestra que el modelo generalice a otras redes de drenaje ni a tormentas observadas. Los 410.814 pares nodo–corrida comparten solo una red y escenarios relacionados; no son 410.814 observaciones independientes. Además, 188 corridas no entraron al score y la QA usa umbrales provisionales.

Antes de entrenar o desplegar con una tercera red, se debe: (1) revisar hidráulicamente los umbrales QA y el manejo de estados iniciales; (2) conseguir observaciones o un conjunto de eventos de prueba representativo; (3) declarar una política de dominio para esa red y generar un staging nuevo; y (4) acordar el costo relativo de falsos negativos y errores de volumen para definir un criterio de aceptación. Los resultados externos actuales deben permanecer congelados y no utilizarse para reajustar estos candidatos.

## Verificación de código

Después de los dos arreglos se ejecutaron 39 pruebas enfocadas de preregistro, inferencia sellada y scoring verificado: **39 passed**. Incluyen el caso de IDs locales repetidos, identidad por SHA-256, el redondeo CSV de probabilidades `float32`, la negativa a aceptar etiquetas externas y la lectura de targets únicamente desde reportes ligados.
