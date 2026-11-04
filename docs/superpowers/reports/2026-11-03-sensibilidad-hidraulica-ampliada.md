# Sensibilidad hidráulica ampliada: Chico Norte

**Fecha:** 3 de noviembre de 2026. **Estado:** estudio exploratorio terminado; integración al flujo principal pendiente. Este informe complementa la [revisión hidráulica inicial](2026-11-03-revision-hidraulica-sensibilidad-chico-norte.md). El [protocolo](2026-11-03-protocolo-sensibilidad-ampliada.md) se escribió antes de las 24 simulaciones adicionales; los dos hidrogramas y sus diez escenarios ya se habían evaluado, por lo que no es una prueba ciega.

## Qué se estudió

Se revisaron dos hidrogramas sintéticos de entrada por nodo —pulsos y meseta— con factores 0,7; 1,3; 2,7; 3,9 y 4,3. Para cada uno se compararon tres variantes de SWMM: **referencia** con InitFlow original y paso máximo de 5 s; **paso máximo de 1 s**, conservando InitFlow; e **InitFlow cero** en 522 conductos con paso máximo de 5 s. En todas las variantes, `VARIABLE_STEP=0,75`, `MINIMUM_STEP=0,5 s` y reporte cada minuto. El paso real es variable: 1 y 5 s son máximos, no intervalos constantes.

Son **diez escenarios y treinta variantes**: seis variantes de dos casos analizados previamente y 24 simulaciones nuevas. El caso histórico crítico y las inspecciones a pasos internos de la revisión inicial se conservan por separado; no se cuentan entre estos diez casos. Se utilizó SWMM 5.2.4, PySWMM 2.1.0 y SWMM Toolkit 0.17.0. Las diferencias se refieren a simulaciones con las mismas entradas, no a mediciones de campo.

## Resultados principales

- **Calidad provisional:** las 24 nuevas variantes tuvieron cobertura finita, drenaje aceptable y error absoluto de continuidad inferior a 5 %; el máximo absoluto entre ellas fue **1,727 %**. Las seis variantes reutilizadas también habían cumplido la QA provisional en la revisión anterior. Esta comprobación no demuestra que el estado inicial sea físicamente correcto; en InitFlow cero la verificación usa el propio INP modificado.
- **Paso máximo:** al pasar de 5 a 1 s, el porcentaje de pasos sin convergencia disminuyó en ocho escenarios y quedó en 0 % en los otros dos. Por ejemplo, en meseta factor 4,3 pasó de **49,50 % a 42,29 %**. Este porcentaje depende del número de pasos, que cambia entre variantes; su reducción no prueba convergencia ni es una confiabilidad.
- **Inundación total:** el cambio máximo absoluto en la suma de volúmenes nodales al reducir el paso fue **56 m³**, en pulsos factor 3,9: de **32.706 a 32.650 m³**. La mayor diferencia absoluta en un nodo fue **7 m³**, en pulsos factor 4,3. Seis de los diez escenarios presentaron cambios en al menos un volumen nodal reportado.
- **Cambio de clase:** con paso de 1 s hubo **dos pares nodo–escenario** que cruzaron el umbral de 1 m³: nodo 372I en pulsos factor 2,7 y nodo 396I en pulsos factor 3,9, ambos de 0 a 1 m³ reportado. Con InitFlow cero hubo **siete pares nodo–escenario** que cambiaron de clase en tres escenarios de pulsos; no se deben sumar como siete nodos únicos, pues el nodo 314C aparece en dos factores. Todos estos cambios fueron entre 0 y 1 m³ reportado.
- **Estado inicial:** la diferencia de la suma nodal frente a la referencia estuvo entre **−5 y +7 m³** en estos diez escenarios, y la mayor diferencia absoluta en un nodo fue **5 m³**. El balance SWMM redondeado registró **4.048 m³ de almacenamiento inicial** en las variantes con InitFlow original y **0 m³** con InitFlow cero. La diferencia no demuestra cuál estado corresponde a la realidad.

La [tabla de los treinta casos](../../resultados/sensibilidad_ampliada_2026_11_03/comparacion_diez_escenarios.csv) contiene continuidad, no convergencia, FII, volumen, número de nodos y diferencias respecto a cada referencia. La [tabla de cambios de clase](../../resultados/sensibilidad_ampliada_2026_11_03/cambios_clase_por_nodo.csv) identifica cada nodo y los dos volúmenes reportados. El volumen proviene de la tabla de inundación por nodo del RPT, con **resolución de 1 m³ por nodo**; cambios menores pueden quedar ocultos. El umbral vigente es al menos 1 m³, por lo cual los casos reportados como 0 y 1 m³ están en su frontera de resolución.

| Hidrograma | Factor | Volumen referencia (m³) | Diferencia paso 1 s (m³) | Diferencia InitFlow cero (m³) |
|---|---:|---:|---:|---:|
| Pulsos | 0,7 | 0 | 0 | 0 |
| Pulsos | 1,3 | 5 | 0 | 0 |
| Pulsos | 2,7 | 11.224 | −18 | −5 |
| Pulsos | 3,9 | 32.706 | −56 | +5 |
| Pulsos | 4,3 | 41.008 | −54 | 0 |
| Meseta | 0,7 | 0 | 0 | 0 |
| Meseta | 1,3 | 128 | 0 | +7 |
| Meseta | 2,7 | 30.829 | −11 | +3 |
| Meseta | 3,9 | 90.608 | −16 | −3 |
| Meseta | 4,3 | 116.608 | −25 | +5 |

Las gráficas de [volumen para pulsos](../../resultados/sensibilidad_ampliada_2026_11_03/pulsos_volumen_total_m3.png), [volumen para meseta](../../resultados/sensibilidad_ampliada_2026_11_03/meseta_volumen_total_m3.png), [no convergencia para pulsos](../../resultados/sensibilidad_ampliada_2026_11_03/pulsos_pasos_sin_convergencia_pct.png) y [no convergencia para meseta](../../resultados/sensibilidad_ampliada_2026_11_03/meseta_pasos_sin_convergencia_pct.png) muestran cada variante frente al factor. Las líneas de volumen pueden superponerse porque las diferencias son pequeñas frente al total; la tabla anterior las cuantifica.

## Verificación y trazabilidad

Los SHA-256 del INP efectivo, evento, RPT y OUT fuente se comprobaron antes de simular y al resumir. Se verificaron además los hashes de los archivos de cada variante. Las diez referencias de 5 s reprodujeron continuidad, porcentaje de no convergencia, mayor FII y volúmenes nodales del RPT original. Para volúmenes se aplicó una tolerancia absoluta de **0,000001 m³** con tolerancia relativa cero: en meseta factor 3,9 la lectura del CSV introdujo una diferencia de representación numérica de aproximadamente 1,14 × 10⁻¹³ m³ en el nodo 2C. Esta tolerancia es mucho menor que la resolución de 1 m³ del RPT y no equivale a una tolerancia de aceptación hidráulica.

Para las 24 variantes nuevas se verificó contra los INP que los cambios fueran exclusivamente los declarados: referencia frente a fuente cambia las opciones de reporte y paso; 1 s frente a referencia cambia el máximo de paso; InitFlow cero frente a referencia cambia los 522 campos iniciales. No se identificaron otros cambios. La [QA](../../resultados/sensibilidad_ampliada_2026_11_03/qa/sensibilidad/qa_sensibilidad.csv), el [balance](../../resultados/sensibilidad_ampliada_2026_11_03/qa/sensibilidad/balance_sensibilidad.csv) y las [comprobaciones de entrada](../../resultados/sensibilidad_ampliada_2026_11_03/qa/sensibilidad/cambios_entrada_verificados.csv) quedan por corrida. Las seis variantes reutilizadas y su QA figuran en el [índice de evidencia anterior](../../resultados/evidencia_revision_hidraulica_2026_11_03/README.md).

Se regeneraron las tablas y seis gráficas agregadas en un directorio independiente y sus archivos coincidieron por SHA-256. El [índice de evidencia](../../resultados/sensibilidad_ampliada_2026_11_03/README.md) detalla código, archivos y preservación. Los OUT completos siguen en `outputs/experiments/`, ignorado por Git: los hashes verifican archivos disponibles pero **no constituyen una copia de seguridad**.

## Interpretación y siguientes pasos

Las diferencias agregadas son reducidas en comparación con algunos volúmenes totales, pero eso no garantiza series locales estables ni clases nodales idénticas. La revisión previa mostró diferencias mayores en pasos internos que a intervalos de un minuto para ubicaciones específicas. No se investigaron todas las series ni se demostró convergencia temporal: hacen falta niveles adicionales de paso y un diseño de selección más amplio. Una reducción del máximo de paso puede modificar tanto el número como la distribución de los pasos calculados.

Los escenarios representan dos formas sintéticas en una sola red; no sustentan generalización a otros hidrogramas o redes, validación con observaciones ni un porcentaje de confiabilidad. No se recalibró SWMM, no se seleccionaron hiperparámetros, no se entrenó ningún modelo y no se cambiaron las etiquetas históricas ni la QA. Los parámetros físicos quedan fuera de esta primera etapa por decisión del investigador.

El [plan de implementación](../plans/2026-11-03-sensibilidad-hidraulica.md) incorpora el estudio al proyecto mediante una interfaz optativa, verificaciones de procedencia, controles de cierre y tiempo, resultados por pareja y evidencia reproducible. La adopción de Chico Norte para entrenamiento conjunto sigue requiriendo revisar su estado inicial y admisión hidráulica; las variantes cero no son datos de entrenamiento autorizados automáticamente.
