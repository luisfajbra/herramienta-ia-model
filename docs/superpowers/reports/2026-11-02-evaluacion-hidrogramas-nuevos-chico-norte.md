# Evaluación de modelos con hidrogramas nuevos en Chico Norte

**Fecha de ejecución:** 2 de noviembre de 2026 (hora de Colombia)

**Estado:** completada; evaluación exploratoria y reproducible

**Alcance:** comparar, contra SWMM, los candidatos congelados MLP y XGBoost entrenados previamente con Chico Sur. No se reentrenaron ni ajustaron los modelos.

## Qué se evaluó

Se construyeron dos formas sintéticas de caudal nodal para la red Chico Norte: **Pulsos dobles separados**, de 7 horas, y **Meseta sostenida y pico tardío**, de 16 horas. Las ordenadas de cada forma se normalizaron entre cero y uno; luego se multiplicaron por el pico de caudal nodal definido en el INP de Chico Norte y por el factor de cada escenario. Así se conserva el pico nodal de referencia antes del escalamiento. No son mediciones observadas ni lluvia; la entrada evaluada sigue siendo el hidrograma de caudal por nodo.

Se seleccionaron cinco factores mediante `random.Random(20261102).sample`, sin reemplazo, de los décimos que no pertenecen a la grilla de entrenamiento. El orden de selección fue 3,9; 0,7; 2,7; 4,3; 1,3. Para informar se ordenaron como **0,7; 1,3; 2,7; 3,9; 4,3**. La combinación de dos formas y cinco factores produjo **10 escenarios**. Los factores están dentro del intervalo de entrenamiento [0,2; 5,0], por lo que el ensayo evalúa interpolación del factor y formas temporales nuevas; no es extrapolación del factor.

Se conservaron los criterios QA vigentes: cobertura finita completa, drenaje final aceptable y error absoluto de continuidad ≤5 %. El límite de 5 % y las tolerancias de drenaje siguen siendo provisionales, como está documentado en la revisión hidráulica. No se agregó una exclusión por FII o porcentaje de pasos sin convergencia. Los diez escenarios terminaron drenados y con continuidad entre **0,191 % y 3,007 %**, así que los diez fueron elegibles y ninguno se excluyó. Todos necesitaron extender la ventana inicial de drenaje de 6 a 8 horas; los escenarios de pulsos dobles con factor 3,9 y 4,3 necesitaron una extensión adicional hasta 10 horas. Se conservaron los intentos y sus artefactos en el historial.

El umbral de inundación para la etiqueta fue 1 m³ y el umbral de clasificación fue probabilidad 0,5. La cascada del modelo asigna volumen cero si no predice inundación. Las métricas de volumen de abajo usan las **5.220 filas nodo–escenario agrupadas** (522 nodos por 10 corridas); el RMSE es el RMSE calculado sobre esas filas, no el promedio de los diez RMSE por escenario. Las filas repetidas por nodo no son observaciones independientes de redes distintas; no se calcularon intervalos de confianza.

## Resultados principales

SWMM registró inundación ≥1 m³ en 1.433 de las 5.220 filas (27,5 %). MLP obtuvo más recall y F1; XGBoost fue más preciso y tuvo menos falsas alarmas. Sus RMSE volumétricos agrupados fueron prácticamente iguales, pero el sesgo agregado muestra diferencias que ese RMSE por sí solo oculta.

| Candidato | Precisión | Recall | F1 | PR-AUC | RMSE (m³) | MAE (m³) | Sesgo total de volumen |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 82,2 % | 55,9 % | 0,665 | 0,796 | 186,18 | 48,39 | −40,9 % |
| MLP, ensemble de 5 semillas | 66,2 % | 89,6 % | 0,761 | 0,831 | 186,26 | 47,94 | +10,1 % |

XGBoost omitió 632 inundaciones y marcó 174 falsas alarmas; MLP omitió 149 y marcó 656 falsas alarmas. En este pequeño lote, MLP detectó muchas más inundaciones, a costa de más falsas alarmas. XGBoost fue más conservador y subestimó el volumen total. No se declara un ganador definitivo: cuál error pesa más depende del uso hidráulico previsto, y diez escenarios sintéticos no bastan para fijar una política operacional.

| Forma de hidrograma | Modelo | Recall | F1 | RMSE (m³) | MAE (m³) | Sesgo de volumen |
|---|---|---:|---:|---:|---:|---:|
| Pulsos dobles separados | XGBoost | 54,2 % | 0,656 | 85,79 | 26,21 | −30,8 % |
| Pulsos dobles separados | MLP | 88,1 % | 0,764 | 86,39 | 25,99 | +7,3 % |
| Meseta sostenida y pico tardío | XGBoost | 57,8 % | 0,675 | 248,92 | 70,57 | −44,5 % |
| Meseta sostenida y pico tardío | MLP | 91,3 % | 0,759 | 248,84 | 69,90 | +11,2 % |

El evento sostenido produjo más volumen de inundación con factores altos y dominó el RMSE agrupado. En ambas formas la diferencia de RMSE entre modelos fue pequeña, aunque su balance de falsos negativos y falsas alarmas fue distinto. En los factores 0,7 y 1,3 SWMM tuvo 0 y 5 m³ para la forma de pulsos, y 0 y 128 m³ para la forma sostenida; por eso esos puntos de baja magnitud son sensibles a falsas alarmas y al corte de 1 m³.

## Figuras

Los enlaces de esta sección conservan las figuras originales de la evaluación, incluidas las comparaciones históricas con símbolos de error. La presentación revisada utiliza [figuras independientes por modelo](../../figures_chico_norte_novel_eval_2026_11_02_presentacion/INDEX.md), con la red y únicamente los nodos inundados coloreados por volumen. Cada mapa tiene su propia escala de color y debe interpretarse con su barra.

- [Hidrogramas sintéticos y factores](../../figures_chico_norte_novel_eval_2026_11_02/hidrogramas_y_factores.png)
- [Paridad global](../../figures_chico_norte_novel_eval_2026_11_02/paridad_global.png)
- [Volumen total por factor](../../figures_chico_norte_novel_eval_2026_11_02/volumen_total_por_factor.png)
- [Mapa: pulsos dobles, factor 2,7](../../figures_chico_norte_novel_eval_2026_11_02/mapa_novel_20261102_cascade_peaks_f2.7.png)
- [Mapa: meseta y pico tardío, factor 2,7](../../figures_chico_norte_novel_eval_2026_11_02/mapa_novel_20261102_sustained_late_surge_f2.7.png)

## Lectura y límites

Esta prueba confirma que los dos candidatos congelados pueden ejecutar predicciones sobre Chico Norte con estas entradas y que sus resultados se pueden contrastar con SWMM. También muestra una diferencia práctica: para reducir inundaciones omitidas, MLP rindió mejor en este lote; para reducir falsas alarmas, XGBoost rindió mejor. Las métricas no permiten afirmar todavía una confiabilidad general de ninguno de los modelos en cualquier red.

La forma de evento sí es nueva, pero sólo se probaron dos perfiles sintéticos, cinco factores dentro del soporte de entrenamiento y una única red objetivo. Los resultados no validan hidrogramas medidos, otras redes no revisadas, cambios en condiciones iniciales ni lluvia–escorrentía. El umbral hidráulico de continuidad continúa siendo provisional y no se filtraron corridas por no convergencia. La prueba tampoco reentrenó modelos ni amplió el set de entrenamiento.

El siguiente paso recomendado es revisar en conjunto los puntos de desacuerdo, sobre todo las falsas alarmas de MLP en factores bajos y los falsos negativos de XGBoost, y ampliar la evaluación con más formas de entrada y redes objetivo antes de usar estas métricas como afirmación de transferencia general. La admisión hidráulica de Chico Norte al entrenamiento debe seguir separada hasta justificar los criterios QA y revisar sus diagnósticos numéricos.

## Trazabilidad y reproducción

- INP de Chico Norte: `data/networks/chico_norte_93_qx1/SWMM - Chico Norte 93% - Qx1.00.inp`; SHA-256 `1c46db327638e023f86f7f0061774abe708f01b8df625f6b6c7f58af36a1ace1`.
- Formas normalizadas: `data/hydrograph_validation/chico_norte_nuevos_2026_11_02/`; ambos CSV tienen SHA-256 en el diseño congelado.
- Candidatos congelados del lote de entrenamiento del 28 de octubre: XGBoost metadata `cdf6ed18a10e2233e9a3670dcac63ac0e2474aa284e1457e865cb41d15444027`; MLP ensemble metadata `5da1af38519cab23569e2d73c9671be0471c8ab76f55971a521a66393151c8b8`.
- Ejecución, manifiestos, reportes SWMM, bases temporales, predicciones y salidas verificadas: `outputs/experiments/chico_norte_novel_hydrograph_eval_20261102/run_20261103T034108Z/`. El paquete de artefactos SWMM ocupa aproximadamente 199 MB. Se mantuvo `node_results` en cero; las etiquetas se extrajeron desde RPT verificados durante el scoring.
- Figuras: `docs/figures_chico_norte_novel_eval_2026_11_02/`; su índice registra los SHA-256.
- Comando para repetir el experimento completo en una nueva carpeta: `.\venv\Scripts\python.exe -m scripts.run_chico_norte_novel_hydrograph_eval --execute`.
- Se corrigió el lector CSV del scoring para preservar exactamente los factores decimales escritos. Sin esa lectura de ida y vuelta, `0.7` podía redondearse a un float binario distinto al guardado en SQLite y el validador cerraba correctamente el scoring. La regresión se reprodujo en esta ejecución antes de puntuar y los 10 registros quedaron luego verificados.

## Material para presentación

- Presentación semanal basada en la versión aprobada del 30 de octubre: conserva 14 diapositivas, elimina las tres diapositivas antiguas de mapas con marcas de errores, y agrega 24 diapositivas para esta evaluación: [`Presentacion_MLSMFP_02-11_avance_chico_norte_mapas_separados_inundacion.pptx`](../../../Presentacion_MLSMFP_02-11_avance_chico_norte_mapas_separados_inundacion.pptx).
- Guion oral para explicar los resultados, el RMSE agrupado, las figuras y los límites de la evaluación: [`guion_presentacion_2026-11-02-chico-norte.md`](../../resultados/guion_presentacion_2026-11-02-chico-norte.md).
- Script reproducible que actualiza la presentación a partir del PPTX anterior y las figuras/resultados verificados: [`update_weekly_2026_11_02_deck.py`](../../../scripts/update_weekly_2026_11_02_deck.py). Comando: `.\venv\Scripts\python.exe -m scripts.update_weekly_2026_11_02_deck` (genera la versión revisada en un archivo independiente).



La version revisada reorganiza las figuras en el orden solicitado: dos hidrogramas separados en curvas de linea, con marcadores y relleno azul claro como en el flujo completo; 18 mapas independientes, uno por modelo (SWMM, MLP o XGBoost) y escenario, cada uno con escala YlOrRd propia y sin marcas de errores de clasificacion; una paridad global por modelo y al final las rectas por factor. Hidrogramas, paridades y mapas se regeneraron desde los artefactos sellados; la figura de rectas quedo sin cambios. El [indice de figuras](../../figures_chico_norte_novel_eval_2026_11_02_presentacion/INDEX.md) documenta escalas y reproduccion.


## Estado de integración y próximos pasos (2026-11-03)

**Actualización posterior:** la opción optativa `main.py --evaluate-frozen-chico-norte` ya permite inspeccionar y ejecutar específicamente esta evaluación; se comprobó con diez simulaciones y figuras separadas. No permite aún escoger otras entradas. Los comandos, pruebas y límites se detallan en el [informe de integración](2026-11-03-integracion-evaluacion-congelada.md). Los párrafos siguientes conservan el estado anterior a esa integración.

Esta evaluación de hidrogramas nuevos es un experimento externo reproducible, pero aún se ejecuta por sus scripts específicos. Cargó los candidatos congelados de MLP (ensemble de cinco semillas) y XGBoost, verificó predicciones y reportes SWMM enlazados, y guardó sus resultados, figuras y presentación en artefactos separados. No reentrenó ni promovió modelos y no incorporó etiquetas a la base de entrenamiento. Ejecutar los comandos habituales del flujo principal no vuelve a generar automáticamente estos resultados; el flujo principal todavía no ofrece esta evaluación como opción integrada.

La continuación recomendada es:

1. **Continuar la revisión hidráulica de Chico Norte.** El [seguimiento del 3 de noviembre](2026-11-03-revision-hidraulica-sensibilidad-chico-norte.md) documenta sensibilidades y problemas pendientes de cierre y tiempo. La continuidad del 5 % y la política de drenaje permanecen como QA provisional vigente, sin filtro automático adicional por no convergencia o FII.
2. **Hacer reproducible la evaluación desde el flujo normal.** Preparar una opción explícita para cargar el par congelado, puntuar un lote elegido y generar mapas/paridades con manifiestos y hashes. Debe producir salidas en un directorio de experimento separado y no entrenar, promover modelos ni escribir etiquetas en la base canónica por defecto.
3. **Decidir la admisión de Chico Norte antes de ampliar el entrenamiento.** Si la revisión y las decisiones de QA permiten usar sus etiquetas, ejecutar el plan de modelo compartido con evaluación anidada por red y evento. Un resultado sobre redes ya usadas para desarrollo no demuestra transferencia a una red futura.
4. **Reservar una tercera red verdaderamente no usada** para evaluar generalización a redes nuevas.

Este estado complementa, pero no marca como cerrados, los gates del [roadmap multired](../plans/2026-10-25-modelo-multired-roadmap.md) ni los del plan de evaluación anidada.
