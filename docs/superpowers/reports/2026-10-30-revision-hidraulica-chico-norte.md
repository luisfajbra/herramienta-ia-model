# Revisión hidráulica de Chico Norte

**Fecha de corte:** 2026-10-30
**Alcance:** revisión de los artefactos congelados del lote de 975 escenarios de Chico Norte y del código que genera sus INP efectivos. No se ejecutaron nuevas simulaciones ni se modificaron modelos, INP, reportes, bases de datos o umbrales durante esta revisión.

## Decisión de proyecto registrada

Se mantiene, por ahora, la QA vigente como regla operativa para admitir corridas al set de datos: cobertura finita completa, continuidad absoluta ≤5 % y drenaje aceptable según la configuración actual. No se añade todavía un filtro automático por porcentaje de pasos no convergentes o por FII. La decisión conserva la población y los resultados congelados ya usados por el proyecto y evita introducir un corte nuevo sin una regla acordada.

Esta decisión define la admisión provisional del proyecto; no significa que los umbrales tengan justificación hidráulica definitiva ni que cada corrida admitida esté hidráulicamente validada. Se seguirá registrando y revisando la no convergencia como diagnóstico, y los 75 casos cuyo INP efectivo no cerró correctamente sus hidrogramas permanecen excluidos hasta que puedan calificarse con una entrada corregida y trazable. La corrección del generador `linear_v1` sigue siendo necesaria para nuevas ejecuciones, pero no cambia retrospectivamente la QA ni los scores congelados.

## Dictamen

El lote está trazado y se puede reconstruir, pero **no debe describirse como hidráulicamente validado**. La QA aplicada hasta ahora comprueba cobertura finita, continuidad y drenaje según umbrales provisionales; no considera el indicador de pasos de enrutamiento que no convergen, aunque SWMM lo imprime en el RPT. Además, el ejecutor congelado omitió el cierre explícito de hidrogramas no nulos que sí aplica el flujo normal del proyecto. Esa omisión explica de forma directa los 75 casos marcados `not_drained` de tres formas de evento.

Los scores externos actuales utilizaron 787 escenarios que pasaron la QA vigente. Las tres formas afectadas por el cierre defectuoso quedaron completamente excluidas, por lo que no aportaron etiquetas al score actual. Sin embargo, 690 de esos 787 escenarios elegibles tienen un porcentaje distinto de cero de pasos no convergentes. Los resultados externos siguen siendo reproducibles como **evaluación bajo la QA provisional vigente**, pero su interpretación hidráulica y su cobertura necesitan esta salvedad.

## Base y reproducción de la revisión

- INP base: `data/networks/chico_norte_93_qx1/SWMM - Chico Norte 93% - Qx1.00.inp`.
- SHA-256 del INP: `1c46db327638e023f86f7f0061774abe708f01b8df625f6b6c7f58af36a1ace1`, coincidente con el protocolo congelado.
- Experimento: `outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z/B/`.
- Población: 39 formas sintéticas × 25 factores = 975 escenarios. Son entradas de caudal por nodo, no lluvia observada ni mediciones de campo.
- Integridad revisada: los SHA-256 de los 975 RPT coinciden con el manifiesto congelado; no se detectaron faltantes ni discrepancias. El INP fuente también coincide con el protocolo.
- Configuración relevante del INP: unidades LPS; `DYNWAVE`; paso de reporte 3 min; paso de enrutamiento 5 s; `VARIABLE_STEP=0.75`; `ALLOW_PONDING=NO`; `MIN_SLOPE=0`. Los valores cero de `MAX_TRIALS` y `HEAD_TOLERANCE` requieren interpretarse conforme a los valores por defecto de la versión de SWMM usada, no como una orden de “cero tolerancia”.

La revisión leyó el manifiesto y los reportes existentes y contrastó la ruta del generador congelado con la política de cierre usada por los flujos de lote habituales. No alteró los artefactos que sustentan el scoring.

## Resultados de QA hidráulica registrada

| Resultado del lote congelado | Corridas |
|---|---:|
| Total | 975 |
| `drained` según la política configurada | 900 |
| `not_drained` según la política configurada | 75 |
| Continuidad absoluta >5 % | 116 |
| Tanto continuidad >5 % como `not_drained` | 3 |
| Elegibles según la QA vigente | 787 |
| Excluidas por la QA vigente | 188 |
| Cobertura finita completa | 975 |

El filtro usa |continuidad| ≤5 % provisionalmente, drenaje según la política configurada y cobertura completa. El límite del 5 % no tiene aún una justificación hidráulica específica para esta red. El manual EPA cita 10 % como ejemplo de nivel por encima del cual debe cuestionarse la validez; no lo convierte en un límite universal ni justifica por sí solo elegir 5 %.

La continuidad absoluta tuvo mediana 0,864 %, percentil 90 de 5,469 %, percentil 95 de 8,053 %, percentil 99 de 13,400 % y máximo 17,699 %. Veintisiete escenarios superaron 10 %. El límite actual excluye 116 escenarios por continuidad, pero continuidad aceptable no demuestra por sí sola estabilidad numérica ni plausibilidad física.

## Hallazgos

### 1. El control actual omite la no convergencia de pasos de enrutamiento — crítico

Los RPT contienen la estadística SWMM “% of Steps Not Converging”. Aparece en los 975 informes. En 801 informes el valor es mayor que cero; otros 174 indican convergencia en todos los pasos. De los 787 escenarios elegibles bajo la QA vigente, 690 (87,7 %) reportan algún paso no convergente. En ese subconjunto elegible, la mediana es 24,67 %, el percentil 90 es 41,96 % y el máximo 72,09 %.

El caso más alto es `B:shape_quasi_uniform_block:f1.400`: 72,09 % de pasos no convergentes; SWMM destaca el nodo 523C y, después, 396I. El reporte también presenta un “Highest Flow Instability Index” (FII); T508 es el elemento con mayor índice en 799 de 975 casos, T463 en 159 y T514 en 17. El valor principal de FII para cada corrida tiene mediana 47, percentil 90 de 69 y máximo 116. La estadística de pasos no convergentes y FII son indicadores para investigar: en esta revisión no se les asignó un umbral automático de rechazo.

T508 conecta 193C con 191C; mide 7,172 m, tiene diámetro 1,350 m, Manning n=0,0084, pendiente geométrica aproximada 0,001394 y `InitFlow` de 2.909,06 L/s. El reporte identifica T507 como el elemento que domina la limitación de paso de tiempo en 974 de 975 escenarios; su contribución mediana es 77,69 %. Esto prioriza una revisión de series temporales y sensibilidad numérica en T507/T508, pero no prueba que sus dimensiones sean erróneas.

No hubo líneas literales `WARNING` o `ERROR` en 948 RPT; 27 sí contienen esas cadenas bajo el análisis de líneas ancladas. La finalización normal de SWMM y la ausencia de advertencias textuales no eliminan la evidencia del estadístico de convergencia.

### 2. Falta el cierre `linear_v1` en el ejecutor congelado — crítico para drenaje y cobertura

El flujo normal de `swmm_resilience/simulation/batch.py` aplica `apply_closure_policy(...)` antes de escribir el INP. El ejecutor de `swmm_resilience/simulation/frozen_pair_runner.py` construye `HydrographScenario` directamente desde el CSV y llama a `write_scenario_inp(...)`, sin aplicar esa política. El escritor añade el único punto cero en `último_punto + horas_de_drenaje`.

Por tanto, cuando el último valor CSV era positivo, SWMM interpoló linealmente desde ese caudal hasta cero durante todo el periodo de drenaje: el aporte seguía activo mientras se esperaba que la red drenara. En el lote anterior, esos casos incluían un punto de cierre a cero en el siguiente paso temporal, conforme a `linear_v1`, y el periodo de drenaje empezaba después de cerrar el aporte. La comparación de INP para `long_moderate`, factor 0,2, muestra concretamente el punto `13:00 = 0` en el INP anterior y su ausencia en el congelado actual; en este último, el siguiente cero quedó al final de la cola de drenaje.

Los 75 `not_drained` pertenecen exactamente a `long_moderate`, `shape_2h` y `shape_5h`, los 25 factores de cada forma. Los tres grupos tienen finales de aporte positivos. La causa del estado `not_drained` es por ello la semántica de entrada generada, no evidencia suficiente de que la red real tarde más en drenar. Los 75 casos están excluidos del score actual, pero esta falla reduce cobertura de eventos y hace que el criterio de drenaje de esos casos no sea comparable con el lote anterior.

### 3. `InitFlow` es una condición inicial importante y aún no está justificada

El INP asigna `InitFlow` no nulo a los 522 conductos (1,5–4.529,9992 L/s), mientras los 523 nodos comienzan secos. La procedencia de esos caudales iniciales sigue desconocida. En una comparación ya registrada para un evento Qx1 de 3 h, el estado actual y `InitFlow=0` dieron ambos 0 m³ de inundación y ambos incumplieron el criterio de drenaje de 6 h; pero la continuidad cambió de 2,971 % a −0,066 %, el máximo de profundidad cambió en 274/523 nodos y los máximos absolutos de caudal cambiaron en 300/522 conducciones. La mayor diferencia temporal reportada fue 0,895 m de profundidad y 3.103 L/s de caudal.

Esto demuestra sensibilidad al estado inicial; no demuestra cuál estado representa la condición real. El manual hidráulico de referencia de SWMM describe el flujo inicial asignado a una conducción como una condición uniforme de flujo permanente y explica que puede intervenir en la determinación de la profundidad normal. No se debe borrar ni escalar `InitFlow` hasta determinar su fuente o documentar explícitamente un análisis de escenarios iniciales.

### 4. El umbral de continuidad y los criterios de drenaje siguen siendo provisionales

La política actual usa 5 % de continuidad, seis horas iniciales de drenaje, extensiones de dos horas hasta dos reintentos, ventana final de una hora, tolerancias de 0,01 L/s para aporte/inundación/flujo residual y 0,01 m³ para cambio de almacenamiento. La configuración está registrada, pero los umbrales de tolerancia no tienen aún una justificación específica basada en resolución, balance, escala de caudales o uso previsto. Mantener una regla reproducible es valioso; llamarla validación hidráulica completa sería excesivo.

Además, `ALLOW_PONDING=NO` impide que el modelo represente almacenamiento superficial por ponding en los nodos. El volumen de inundación reportado debe interpretarse según la formulación de flooding de SWMM y esta opción; no equivale a volumen o extensión de inundación urbana observada.

### 5. Entradas sintéticas y red sin observaciones

Los eventos son hidrogramas sintéticos nodales. La decisión previa fue mantener 455C en cero porque se desconocía el origen de la señal negativa. Ese ajuste está congelado, pero no valida retrospectivamente la serie original ni el escenario de cero. La prueba con Chico Norte evalúa transferencia entre dos modelos de red bajo entradas sintéticas; no valida precisión contra observaciones de lluvia, caudal o inundación.

## Consecuencia para los resultados ya calculados

El score externo usa 787 corridas elegibles y 410.814 pares corrida–nodo. Las 75 corridas con el defecto de cierre quedaron excluidas; el scoring actual no debe describirse como si incluyera sus etiquetas. El conjunto elegible conserva los 690 casos con estadístico de no convergencia distinto de cero, pues el filtro no inspecciona esa estadística. Por ello:

- se conserva el score como resultado congelado y reproducible de la QA vigente;
- se acompaña de la limitación de que la no convergencia no se filtró ni se analizó por sensibilidad;
- no se usa para afirmar validación hidráulica, confiabilidad operativa ni generalización a redes observadas;
- si la revisión de convergencia cambia reglas de admisión o salidas, cualquier evaluación revisada será una nueva versión, con nuevos hashes y sin sobrescribir manifiestos o RPT históricos.

## Acciones recomendadas, en orden

1. Corregir el ejecutor congelado para aplicar y registrar `linear_v1` antes de escribir el INP. Rechazar o marcar explícitamente cualquier entrada cuyo final efectivo no sea cero dentro de la tolerancia; preservar CSV fuente, INP previo, manifiesto de cierre y hashes.
2. Añadir a QA el registro por corrida de “% of Steps Not Converging”, FII máximo y elementos críticos. No fijar todavía un corte universal: primero identificar cuándo y dónde aparecen los fallos y si cambian variables relevantes.
3. Ejecutar un análisis hidráulico focalizado sobre los casos extremos y representativos —incluidos `shape_quasi_uniform_block:f1.400`, los tres grupos afectados por cierre, y escenarios de factores bajos/altos—. Graficar caudal/carga/profundidad de T507/T508 y 523C/396I con paso temporal fino en las ventanas inestables.
4. Hacer una sensibilidad numérica controlada en una muestra pequeña: comparar paso de enrutamiento más fino con la configuración actual, conservando los hidrogramas, estados iniciales y demás parámetros constantes. Comparar continuidad, máximos, tiempos de pico, volúmenes de inundación y FII. Una corrida de 5 s puede no resolver fenómenos más rápidos; no asumir que reducir el paso necesariamente corrige todo.
5. Investigar la fuente de `InitFlow`. Mientras siga desconocida, mantener el valor actual como escenario histórico y registrar `InitFlow=0` como sensibilidad separada, sin combinar resultados de ambos estados en un único set de etiquetas.
6. Con los diagnósticos anteriores, justificar los criterios de continuidad/drenaje/tolerancia por escala y objetivo, emitir una nueva calificación versionada y luego reconstruir el score de transferencia con la población aceptada.

Las acciones 1 y 2 son correcciones de software y observabilidad. Las decisiones sobre umbrales, estado inicial y aceptación de resultados deben basarse en la sensibilidad y la procedencia hidráulica, no en buscar el corte que aumente el número de corridas elegibles.

## Fuentes técnicas

- EPA, *SWMM 5.2 User's Manual*: balance de continuidad; advertencia de que una continuidad excesiva cuestiona validez; inestabilidades de flujo en Dynamic Wave, timestep y revisión de los elementos con mayor FII. [Manual oficial EPA](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P10145M6.TXT).
- EPA, *SWMM Reference Manual, Volume II — Hydraulics*: interpretación de caudales iniciales asignados y condiciones hidráulicas. [Manual oficial EPA](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100S9AS.TXT).
- EPA, historial de actualización 5.2: incorporación del porcentaje de pasos donde no se obtuvo convergencia. [Notas oficiales de actualización](https://www.epa.gov/system/files/documents/2022-02/updatebug20220222_1.pdf).

## Estado de preparación

**No se recomienda iniciar todavía un nuevo entrenamiento que trate estas etiquetas como verdad hidráulica depurada.** Los modelos entrenados con Chico Sur y el score externo congelado pueden preservarse y reportarse con su alcance actual; antes de crear un set de entrenamiento ampliado con Chico Norte, hay que corregir el cierre, analizar la no convergencia y decidir/documentar la política de estado inicial. No se requiere cambiar los resultados ya archivados para completar esta auditoría.

### Seguimiento del 3 de noviembre

Se verificaron los dos lotes congelados y se ejecutaron nueve corridas exploratorias de sensibilidad en tres escenarios, además de seis repeticiones para registrar los pasos internos. El [informe de seguimiento](2026-11-03-revision-hidraulica-sensibilidad-chico-norte.md) documenta los resultados, las series locales, los balances y una incompatibilidad de mallas al intentar aplicar el cierre a parte de los 75 escenarios históricos. Se conserva la QA provisional acordada; siguen pendientes la corrección del cierre antes de la congelación, una sensibilidad más amplia y la justificación del estado inicial. Los scores y artefactos históricos no se reemplazaron.
