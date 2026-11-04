# Revisión hidráulica y sensibilidad de Chico Norte

**Ampliación posterior:** el [informe de sensibilidad ampliada](2026-11-03-sensibilidad-hidraulica-ampliada.md) compara los diez escenarios de los dos hidrogramas nuevos, incluidos cambios de clase por nodo, y enlaza el plan de integración. Este documento conserva los resultados y límites de la revisión original.

**Fecha:** 3 de noviembre de 2026.

**Alcance:** diagnóstico de los lotes congelados, nueve simulaciones exploratorias en copias de INP y seis repeticiones para registrar pasos internos. Las entradas son hidrogramas sintéticos de caudal por nodo; no se evaluó lluvia–escorrentía ni se contó con mediciones de campo.

**Estado:** revisión y sensibilidades documentadas; admisión hidráulica definitiva y origen de los caudales iniciales pendientes.

## Resultado principal

La continuidad y el drenaje por sí solos no describen toda la respuesta numérica. En los dos casos ensayados, el porcentaje reportado de pasos sin convergencia disminuyó al reducir el paso máximo de enrutamiento de 5 a 1 s, pero permaneció alto. Los volúmenes de inundación por nodo cambiaron poco a la resolución de los RPT, mientras algunas series locales cambiaron apreciablemente. La revisión no demuestra que una de las configuraciones represente la condición real.

Se conserva el criterio de QA aceptado por el proyecto: cobertura finita, continuidad absoluta ≤5 % y drenaje conforme a las tolerancias configuradas. El 5 % y esas tolerancias siguen siendo provisionales. **No se ha introducido un filtro por no convergencia o FII**, ni se ha modificado la población histórica o los scores de los modelos.

## Comprobación de los lotes existentes

Se verificó el SHA-256 de los **985 RPT** contra sus manifiestos: 975 del lote de transferencia y 10 de los hidrogramas nuevos. El valor de continuidad extraído de cada reporte coincidió con su manifiesto. Los resultados anteriores de transferencia se confirmaron: entre 787 corridas elegibles, **690** registraron algún paso sin convergencia; la mediana fue **24,67 %** y el máximo **72,09 %**.

En los diez escenarios nuevos, **8 de 10** registraron pasos sin convergencia; la mediana fue **22,915 %** y el máximo **49,50 %**. Por ejemplo, la meseta sostenida con factor 4,3 registró **0,191 % de error de continuidad** y **49,50 % de pasos sin convergencia**. Los diez cumplieron la QA provisional existente; la no convergencia no se había usado para excluirlos.

El inventario de esta revisión registra esos indicadores en un CSV verificable por corrida. Todavía no están incorporados automáticamente como campos de diagnóstico del QA para ejecuciones futuras; esa integración queda pendiente y no implica agregar un umbral de exclusión.

También se verificaron los diez CSV de evento nuevos contra sus hashes: cada uno tiene 521 series nodales de entrada y termina exactamente en cero. El nodo 455C mantiene la decisión anterior de entrada cero. Estos son los eventos de la evaluación reciente, no una corrección retroactiva de los 75 escenarios históricos excluidos.

## Casos y método de sensibilidad

Se seleccionaron intencionalmente dos casos, no una muestra estadística:

1. **Histórico crítico:** `shape_quasi_uniform_block`, factor 1,4; fue el caso de mayor porcentaje de pasos no convergentes en los 787 elegibles del lote de transferencia.
2. **Meseta nueva:** `novel_20261102_sustained_late_surge`, factor 4,3; presenta el mayor porcentaje de pasos no convergentes entre los diez eventos nuevos.

Para cada caso se simularon tres variantes con SWMM 5.2.4, PySWMM 2.1.0 y SWMM Toolkit 0.17.0:

- **Referencia:** InitFlow del INP efectivo y paso máximo de enrutamiento de 5 s.
- **Paso de 1 s:** los mismos InitFlow y entradas, con paso máximo de enrutamiento de 1 s.
- **InitFlow cero:** los 522 InitFlow de conductos en cero, con paso máximo de 5 s; el resto de condiciones se conserva.

Las tres variantes usan `VARIABLE_STEP=0,75` y `MINIMUM_STEP=0,5 s`. Por eso, 5 y 1 s son **máximos configurados**, no pasos constantes. Se cambió el intervalo de reporte de 3 a 1 minuto en todas las variantes para obtener series comparables. SWMM controló sus pasos internos; no se forzó el integrador al tomar muestras. Se verificaron los INP línea por línea: frente a cada referencia, la variante de 1 s cambia solo `ROUTING_STEP`, y la de InitFlow cero cambia solo los 522 campos `InitFlow`.

Antes de interpretar las diferencias se compararon las referencias de 5 s con sus RPT congelados: coincidieron la continuidad, el porcentaje de no convergencia, el mayor FII y los volúmenes de inundación por nodo a la resolución reportada. Las seis variantes tienen continuidad dentro de 5 %, cobertura finita y drenaje aceptable según la política provisional. En la variante con InitFlow cero esta comprobación usa su propio INP como fuente del estado inicial; **no certifica equivalencia con el estado del INP original ni constituye admisión para entrenamiento**.

| Caso | Variante | Continuidad | Pasos sin convergencia | Mayor FII (conducto) | Inundación nodal total | Nodos ≥1 m³ |
|---|---|---:|---:|---:|---:|---:|
| Histórico crítico | Referencia, máx. 5 s | 0,328 % | 72,09 % | 112 (T463) | 18.046 m³ | 64 |
| Histórico crítico | Máx. 1 s | 0,328 % | 56,26 % | 87 (T463) | 18.046 m³ | 64 |
| Histórico crítico | InitFlow cero, máx. 5 s | −0,034 % | 72,89 % | 113 (T463) | 18.053 m³ | 64 |
| Meseta nueva, factor 4,3 | Referencia, máx. 5 s | 0,191 % | 49,50 % | 80 (T463) | 116.608 m³ | 253 |
| Meseta nueva, factor 4,3 | Máx. 1 s | 0,203 % | 42,29 % | 74 (T463) | 116.583 m³ | 253 |
| Meseta nueva, factor 4,3 | InitFlow cero, máx. 5 s | −0,020 % | 50,44 % | 82 (T463) | 116.613 m³ | 253 |

El porcentaje de no convergencia usa como denominador el número de pasos realizados; ese número cambia al cambiar el paso máximo. La reducción del porcentaje y del FII observado no demuestra por sí sola convergencia temporal ni valida hidráulicamente la corrida. Los volúmenes anteriores se obtuvieron sumando la tabla de inundación por nodo del RPT, cuya resolución es **1 m³ por nodo**; cambios menores pueden quedar ocultos. Los totales de la tabla de balance del RPT son otra salida redondeada y no tienen que coincidir exactamente con la suma de las filas nodales.

En el caso histórico, las dos variantes de paso reprodujeron los mismos volúmenes de los 522 nodos a esa resolución. En la meseta, el paso de 1 s cambió el volumen reportado en **21 nodos**, con diferencia absoluta máxima de **6 m³** en un nodo y **−25 m³** en la suma total. En ambos casos se conservaron las mismas clases inundado/seco al umbral de 1 m³. Esto solo describe estas dos entradas y esa resolución de salida.

### Series locales y condición inicial

En la meseta, al comparar paso máximo de 1 s con 5 s en los tiempos comunes del OUT de un minuto, la máxima diferencia absoluta observada fue **669,6 L/s en T508** y **0,103 m de profundidad en 396I**. Son diferencias instantáneas dentro de la malla de reporte, no diferencias entre máximos de la serie. Las oscilaciones visibles ameritan inspección más fina; un muestreo de un minuto puede ocultar variaciones más rápidas. Véase la [serie de T508](../../resultados/evidencia_revision_hidraulica_2026_11_03/sensibilidad/graficas_sensibilidad/meseta_nueva_factor_4_3_T508_flow.png) y la [profundidad en 396I](../../resultados/evidencia_revision_hidraulica_2026_11_03/sensibilidad/graficas_sensibilidad/meseta_nueva_factor_4_3_396I_depth.png).

El estado inicial también cambia la respuesta temprana. En la meseta, T508 difirió hasta **2.239,4 L/s** entre InitFlow fuente y cero en los registros de un minuto. El balance de SWMM reportó **4.048 m³** de almacenamiento inicial con los InitFlow de origen y **0 m³** con InitFlow cero; ambos valores están redondeados en el RPT. Aunque las profundidades iniciales asignadas a los nodos son cero, las condiciones iniciales de los conductos forman parte del estado hidráulico. La procedencia de esos caudales sigue desconocida y esta comparación no permite elegir cuál es físicamente correcto.

El nodo **523C es un outfall FREE**, no uno de los 522 junctions etiquetados para el modelo. El conducto T92 llega a ese outfall; T507 y T508 forman un tramo cercano entre 194C, 193C y 191C. T463 es el conducto con mayor FII de los dos casos seleccionados. Estas identificaciones orientan la inspección; no prueban que sus geometrías sean erróneas.

### Ampliación con un factor bajo

Se añadió un tercer caso, **pulsos dobles con factor 0,7**, que en el lote reciente no había reportado inundación. Se repitieron las mismas tres variantes y el control de 5 s reprodujo su reporte congelado. Las tres corridas registraron **0 % de pasos sin convergencia**, **0 m³** de inundación nodal reportada y drenaje/cobertura aceptables bajo el QA provisional. La continuidad fue **3,007 %** con InitFlow fuente, tanto con máximo de 5 s como de 1 s, y **0,001 %** con InitFlow cero.

El balance del caso con InitFlow fuente registra 22.318 m³ de aporte externo y 4.048 m³ de almacenamiento inicial: este último equivale aproximadamente al **18,1 % del aporte externo** de ese caso. Son componentes distintos del balance, no una afirmación de que toda esa agua se convierta en inundación. En la [serie de T508](../../resultados/evidencia_revision_hidraulica_2026_11_03/factor_bajo/graficas_sensibilidad/pulsos_nuevos_factor_0_7_T508_flow.png), la diferencia máxima entre InitFlow fuente y cero en registros de un minuto fue **2.239,4 L/s**. Un porcentaje de no convergencia igual a cero tampoco identifica el estado inicial real ni sustituye la revisión de su procedencia.

### Inspección de pasos internos

Para los dos primeros casos se repitieron las seis variantes y se registró cada paso interno de SWMM en las ventanas **0–15 min** y **11 h 15 min–12 h 30 min** desde el inicio. La segunda ventana se eligió después de observar diferencias en las series de un minuto; es una inspección exploratoria, no una ventana definida antes de ver los resultados. En las seis repeticiones coincidieron la continuidad, la no convergencia, el mayor FII y los volúmenes nodales del RPT con sus controles correspondientes.

Los pasos de las variantes no ocurren necesariamente en los mismos instantes. Para cuantificar diferencias se interpoló linealmente cada serie a una **malla común de un segundo**, únicamente dentro del intervalo observado de ambas; las figuras conservan los puntos de los pasos internos sin remuestrearlos. En la meseta, el máximo de la diferencia absoluta entre paso máximo de 1 y 5 s en esa malla fue **739,9 L/s en T508** y **0,353 m en 396I** dentro de la ventana crítica. En comparación, al usar únicamente el OUT de un minuto los máximos observados fueron 669,6 L/s y 0,103 m, respectivamente. Los valores corresponden a diferencias instantáneas interpoladas, no a diferencias entre máximos de cada serie ni a un error frente a una medición real. Véanse las figuras de pasos internos para [T508](../../resultados/evidencia_revision_hidraulica_2026_11_03/pasos_nativos/graficas/meseta_nueva_factor_4_3_ventana_critica_T508.png) y [396I](../../resultados/evidencia_revision_hidraulica_2026_11_03/pasos_nativos/graficas/meseta_nueva_factor_4_3_ventana_critica_396I.png).

En la ventana inicial de la meseta, la diferencia máxima entre InitFlow fuente y cero fue **2.572,9 L/s en T508** sobre esa misma malla interpolada. Las oscilaciones y diferencias locales justifican más revisión, aunque la suma de volúmenes de inundación resulte parecida a la resolución del RPT.

La propiedad `sim.current_time` de PySWMM entrega una marca de fecha redondeada al segundo y puede repetirla cuando SWMM toma pasos de 0,5 s. Para esta inspección se utilizó el tiempo transcurrido devuelto directamente por `swmm_step`, en días decimales, convertido a segundos; se comprobó con pasos de 0,5 y 1,0 s y no se forzó ningún `stride`. El [código fuente de SWMM de EPA](https://github.com/USEPA/Stormwater-Management-Model/blob/develop/src/solver/swmm5.c) documenta esa salida en días decimales. El primer intento con marcas redondeadas se interrumpió y no se usó para las comparaciones finales.

## Cierre de aportes: hallazgo pendiente de corrección

Los 75 escenarios históricos con estado `not_drained` abarcan los 25 factores de `long_moderate`, `shape_2h` y `shape_5h`. Sus CSV terminaban con al menos un nodo por encima de la tolerancia de 0,01 L/s. En el lote exploratorio independiente con cierre `linear_v1`, los 75 casos correspondientes aparecen drenados y 71 cumplen su QA de software. Es una comparación entre lotes, no una nueva calificación ni un reemplazo automático de los 75 resultados congelados.

El ejecutor congelado actual incorpora un guardia que **rechaza** eventos con extremo no cero por encima de la tolerancia, pero `prepare_pair` todavía materializa esos eventos sin aplicar el cierre antes de congelarlos. Además, se reprodujo una incompatibilidad al intentar componer la política con el contrato de malla común: `apply_closure_policy` añade un tiempo a los nodos que superan la tolerancia y deja intactos los demás, mientras `event_stats` exige la misma malla en todos. Según los finales de los CSV originales, **22 de los 75 escenarios** tienen una mezcla de nodos por encima y por debajo de la tolerancia. En `shape_2h`, factor 1,4, se agregaría un punto a 520 de 521 nodos y `event_stats` rechaza el resultado por mallas distintas.

La corrección pendiente debe definir y probar cómo representar en la malla compartida los nodos ya dentro de tolerancia, conservar y hashear los insumos originales y efectivos, aplicar `linear_v1` antes de congelar el protocolo y simular, y recalificar cualquier corrida regenerada como una versión nueva. **No se ha implementado esa corrección en esta revisión.**

## Hallazgo adicional en la lectura temporal

Las funciones `get_node_series` y `get_link_series` de `swmm_resilience/simulation/swmm_api_io.py` pueden etiquetar como `time_sec` un entero en **nanosegundos desde la época Unix** cuando el OUT devuelve un índice de fechas. Se reprodujo con el OUT de referencia: entre `00:01:00` y `00:02:00`, la columna cambia de `1735948860000000000` a `1735948920000000000`, una diferencia de `60000000000`, aunque el intervalo es de 60 s. Por tanto, esa columna no debe interpretarse como segundos transcurridos sin corregir el origen y la unidad.

La búsqueda en el código del proyecto solo encontró las definiciones de esas dos funciones, no llamadas internas; esto no demuestra que ningún usuario externo las use. Las figuras de esta revisión no utilizan esa columna: leen directamente el índice de fechas del OUT o el tiempo transcurrido devuelto por SWMM. Queda pendiente corregir y probar el contrato temporal, con un origen explícito para los segundos transcurridos, antes de usar esas funciones para nuevos análisis.

## Interpretación y límites

- Este estudio comprueba sensibilidad para tres casos elegidos, y ventanas de pasos internos para dos de ellos; no establece estabilidad de los 985 escenarios ni convergencia de la solución a medida que el paso tiende a cero. Falta ampliar casos, explorar otros pasos y revisar configuración hidráulica.
- Los resultados de las seis variantes son diagnósticos: no se usaron para entrenar, seleccionar o promover MLP/XGBoost, y no se añadieron etiquetas a la base de entrenamiento.
- El origen de InitFlow y la representación física de los hidrogramas sintéticos siguen sin verificarse con observaciones. `ALLOW_PONDING=NO`; los volúmenes SWMM no son mapas observados de extensión de inundación superficial.
- Se conserva la QA provisional decidida por el usuario. Justificarla para el uso de tesis y decidir la admisión de etiquetas al entrenamiento exige integrar esta evidencia con las condiciones reales y los demás escenarios.

El manual de SWMM recomienda usar intervalos de reporte de un minuto o menos para una primera inspección de inestabilidades y examinar series temporales en ubicaciones relevantes. El FII y las frecuencias de no convergencia son diagnósticos que requieren interpretación contextual. [EPA, Manual de usuario SWMM 5.2](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P10145M6.TXT).

## Siguiente trabajo recomendado

1. Corregir y verificar el cierre `linear_v1` con malla común antes de cualquier nueva congelación de los 75 escenarios; conservar íntegros los resultados históricos.
2. Ampliar la sensibilidad numérica con más escenarios y pasos adicionales, para evaluar si las oscilaciones y las diferencias locales observadas en las ventanas subminuto persisten. No fijar nuevos cortes QA a partir de solo tres casos.
3. Investigar la procedencia de los 522 InitFlow y contrastar condiciones iniciales con datos de operación o con un estado inicial hidráulicamente justificado. Mantener InitFlow fuente y cero como escenarios separados mientras tanto.
4. Registrar automáticamente en el QA de futuras corridas la no convergencia, el FII y los elementos críticos como diagnósticos, sin introducir nuevos filtros. Corregir el contrato de `time_sec` para fechas de OUT y comprobarlo con series temporales reales antes de reutilizar los lectores auxiliares.
5. Documentar la justificación de continuidad, drenaje y demás tolerancias para el objetivo de la tesis y, solo después, decidir si Chico Norte se incorpora al entrenamiento multired. Mantener el resultado externo actual identificado como evaluación bajo la QA provisional.

## Archivos y reproducción

- [Inventario, tablas, figuras, INP y RPT de sensibilidad](../../resultados/evidencia_revision_hidraulica_2026_11_03/README.md).
- [Revisión hidráulica anterior](2026-10-30-revision-hidraulica-chico-norte.md) y [evaluación con hidrogramas nuevos](2026-11-02-evaluacion-hidrogramas-nuevos-chico-norte.md).
- Experimento completo, incluidos los seis OUT de un minuto: `outputs/experiments/chico_norte_hydraulic_sensitivity_20261103T130312Z/`. Los OUT ocupan aproximadamente **178,3 MiB** y `outputs/` está excluido de Git; deben respaldarse por separado. La carpeta de evidencia conserva copias acotadas de los seis INP y RPT, series seleccionadas, tablas y figuras, pero por sí sola no contiene todos los OUT.
- Extensión de factor bajo: `outputs/experiments/chico_norte_hydraulic_sensitivity_20261103T132030Z/`; sus tres INP, RPT y series seleccionadas también están en la carpeta de evidencia. Los OUT completos de esta extensión permanecen en `outputs/`.
- Repeticiones de pasos internos: `outputs/experiments/chico_norte_native_windows_20261103T132556Z/`. La carpeta de evidencia conserva las ventanas comprimidas, RPT, tablas y figuras; los OUT completos permanecen en `outputs/`. El intento interrumpido con marcas de tiempo redondeadas está en `outputs/experiments/chico_norte_native_windows_20261103T132158Z/` y no se usa para los resultados.
- El primer intento `chico_norte_hydraulic_sensitivity_20261103T130114Z` fue interrumpido: usar `step_advance(60)` alteró los pasos numéricos respecto del control. No se usa como resultado; las seis corridas reportadas se hicieron sin forzar ese avance y sus referencias reprodujeron los RPT congelados.
- Los archivos fuente, candidatos y bases históricas no se sobrescribieron. Los scripts, sus dependencias, comandos y hashes están descritos en el índice de evidencia.
- La reconstrucción independiente del inventario produjo los mismos hashes en sus seis tablas. El resumen de pasos internos se regeneró en otra carpeta y sus **39 archivos** coincidieron byte a byte con la evidencia. Se verificó la integridad de los archivos de la carpeta de evidencia mediante su manifiesto SHA-256.
- Verificación de software: **18 pruebas pasaron** (sensibilidad, cierre y rechazo de extremos no cerrados). Una prueba adicional que usa `tmp_path` no pudo iniciar por `WinError 5` en el directorio temporal de pytest, incluso al seleccionar una ruta dentro del proyecto; se excluyó de esa ejecución y se comprobó manualmente la escritura y lectura del manifiesto en `outputs/experiments/manual_closure_manifest_check_20261103/`. La limitación de pytest no equivale a una prueba automática aprobada.
