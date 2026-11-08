# Inventario y justificación de variables de los modelos conjuntos

Fecha de revisión: 8 de noviembre de 2026. Alcance: candidatos finales MLP (ensamble de cinco semillas) y XGBoost ajustados con Chico Sur y Chico Norte. Ambos metadatos registran el mismo contrato `tabular_v4_21` y las mismas **21 variables ordenadas**, tanto para clasificación como para regresión. El inventario se contrastó con los metadatos de `outputs/experiments/final_joint_candidates_20261108_v1/{mlp/ensemble,xgboost}/metadata.json` y con `swmm_resilience/ml/contracts.py`. Los flujos anteriores pueden usar el contrato legado de 17 variables; esta lista no supone que todos los comandos históricos carguen los candidatos conjuntos.

Cada fila de datos representa un nodo en una simulación. Las justificaciones que siguen son razones hidráulicas para aportar información al modelo; **no demuestran importancia estadística, causalidad ni mejora individual**. Medir eso requiere un estudio específico, sin ajustar decisiones sobre la reserva ya consultada.

## Geometría y conectividad local

| N.º | Variable | Unidad | Definición implementada | Justificación técnica |
|---:|---|---|---|---|
| 1 | `elev_fondo` | m | Cota de fondo del nodo en el INP. | Contextualiza la posición vertical y el gradiente general de la red; no es por sí sola la carga hidráulica. Puede depender del datum. |
| 2 | `prof_max` | m | Profundidad máxima del nodo declarada en el INP. | Describe la altura disponible hasta la profundidad máxima configurada, relacionada con la posibilidad de sobrecarga y desbordamiento. La configuración del nodo y de SWMM también interviene. |
| 3 | `n_tuberias_in` | cantidad | Número de conducciones que llegan directamente al nodo según el grafo. | Identifica confluencias y complejidad local de aportes; no mide sus caudales. |
| 4 | `n_tuberias_out` | cantidad | Número de conducciones que salen directamente del nodo. | Describe la conectividad de salida y posibles bifurcaciones; no garantiza su capacidad real. |
| 5 | `diam_max_in` | m | Mayor valor `height` de las secciones de las conducciones entrantes. | Resume el tamaño de conducciones que llegan. Es diámetro en secciones circulares; para otras geometrías es altura, por lo que el nombre puede inducir a error. |
| 6 | `diam_max_out` | m | Mayor valor `height` de las secciones salientes. | Aproxima la escala geométrica disponible para evacuar; no equivale a capacidad, especialmente con varias salidas o secciones no circulares. |
| 7 | `pendiente_max_in` | m/m | Mayor pendiente geométrica con signo de las conducciones entrantes para las que se puede calcular. | Resume el desnivel por longitud asociado al transporte por gravedad; no representa directamente el gradiente de energía bajo remanso. |
| 8 | `pendiente_out` | m/m | Pendiente geométrica con signo de la **primera salida calculable**, según el orden de extracción. | Aporta información del desnivel de salida. Si existen varias salidas, una sola pendiente puede no ser representativa; depende de ese orden. |

Las pendientes usan cotas de fondo, offsets de los conductos y longitud. Si falta una cota en un extremo no se calcula ese conducto. El código asigna longitud 1,0 cuando la longitud declarada es falsa o ausente; ello requiere revisar datos de entrada antes de extrapolar a otras redes. Fuente: `swmm_resilience/extraction/static_features.py`.

## Aportes de referencia y posición en la red

| N.º | Variable | Unidad | Definición implementada | Justificación técnica |
|---:|---|---|---|---|
| 9 | `base_inflow_lps` | L/s | Pico del perfil de entrada efectiva de referencia del nodo obtenido del INP, considerando basal y puntos de la serie. | Describe la escala de aporte de referencia y se utiliza para construir escenarios. **Es un pico de referencia; el nombre no significa necesariamente caudal base constante.** |
| 10 | `dist_outfall_m` | m | Distancia mínima dirigida por longitud desde el nodo hasta un outfall alcanzable. | Resume posición y longitud de la ruta hasta descarga, asociadas a conectividad y posible tránsito; no es un tiempo de viaje ni una medida de remanso. |
| 11 | `n_nodos_aguas_arriba` | cantidad | Número de ancestros únicos del nodo en el grafo dirigido. | Aproxima el tamaño y la conectividad de la zona aportante sin duplicar ancestros. |
| 12 | `q_pico_acum_base` | L/s | Suma de `base_inflow_lps` del nodo y sus ancestros. | Resume la magnitud potencial de aportes de referencia. Es una **suma de picos individuales**, que pueden ocurrir en momentos diferentes; no es necesariamente un pico simultáneo. |
| 13 | `upstream_capacity_lps` | L/s | Suma de capacidades estimadas de conductos inmediatamente entrantes, mediante Manning a sección circular llena, a partir de altura de sección, rugosidad y valor absoluto de pendiente. | Ofrece una referencia geométrica simplificada para comparar aportes. No es la capacidad hidráulica efectiva de todo el sistema ni de salida del nodo; el cálculo circular y el valor absoluto de pendiente limitan su interpretación. |

El grafo de esta implementación se construye con `CONDUITS`. La distancia se calcula por la ruta dirigida más corta. Una red con otros tipos de elementos hidráulicos, geometrías no circulares, pendientes adversas o conectividad distinta requiere revisar la validez de estas representaciones antes de inferir. Fuente: `swmm_resilience/extraction/topology.py`.

## Hidrograma y escenario

Para cada nodo se define un **hidrograma tributario construido**, sumando instante a instante los hidrogramas externos del propio nodo y de sus ancestros únicos, interpolados linealmente sobre una malla común. Es una variable de ingeniería derivada del grafo: no simula el transporte por las tuberías, su desfase, almacenamiento, remanso ni pérdidas. La suma de aportes no debe confundirse con el caudal real de SWMM en un conducto. Véase `swmm_resilience/extraction/tributary_features.py`.

| N.º | Variable | Unidad | Definición implementada | Justificación técnica |
|---:|---|---|---|---|
| 14 | `q_pico_nodo` | L/s | Pico del aporte de entrada del escenario en el nodo. En el cálculo por factor se obtiene como pico de referencia por factor; existe cálculo desde picos nodales para escenarios arbitrarios. | La intensidad máxima local ayuda a caracterizar la solicitación del nodo. |
| 15 | `q_pico_acum_escalado` | L/s | Suma de picos de referencia tributarios multiplicada por el factor; en la ruta arbitraria suma los picos nodales del escenario. | Aproxima la escala máxima potencial aportante. No es el máximo de la suma temporal ni un caudal realmente enrutado. |
| 16 | `volumen_local_integrado_m3` | m³ | Integral temporal del hidrograma externo del nodo. | Cuantifica el agua aportada localmente durante el evento, información que el pico no aporta. |
| 17 | `volumen_tributario_integrado_m3` | m³ | Integral del hidrograma tributario construido. | Resume el volumen de entrada potencial del nodo y sus ancestros; útil en eventos prolongados, sujeto a las limitaciones del grafo y el estado inicial. |
| 18 | `duracion_efectiva_tributario_h` | h | Tiempo en que el hidrograma tributario construido supera el 1 % de su propio pico. | Resume cuánto tiempo permanece activo el aporte. El 1 % es una convención de cálculo, no un umbral hidráulico universal. |
| 19 | `tiempo_primer_pico_tributario_h` | h | Instante del primer máximo de ese hidrograma; cero para serie nula. | Describe el momento de la demanda máxima y su relación con la forma del evento. |
| 20 | `tiempo_sobre_capacidad_h` | h | Tiempo en que el hidrograma tributario construido supera `upstream_capacity_lps`. | Resume duración de una demanda superior a la capacidad de referencia; no demuestra que una tubería esté sobrecargada porque compara agregados aproximados. |
| 21 | `integral_exceso_capacidad_m3` | m³ | Integral de `max(Q_tributario - upstream_capacity_lps, 0)`. | Resume magnitud y duración del exceso frente a esa referencia; **no es volumen inundado físico** ni incorpora almacenamiento, transporte o salida efectiva. |

Las integrales usan interpolación lineal y conversión de L/s·h a m³ por factor 3,6. Las dos últimas variables quedan nulas si la capacidad falta o no es positiva. La lectura fuera del rango temporal de cada serie mantiene el valor del extremo: requiere coherencia con el cierre y drenaje del escenario efectivo.

## Objetivos, identificadores y preparación

- `inunda`: objetivo de clasificación binaria derivado del volumen SWMM; se considera inundado cuando el volumen reportado es **≥ 1 m³**. El clasificador predice una probabilidad y se aplica umbral 0,5.
- `vol_inundacion_m3`: objetivo de regresión, volumen de inundación nodal reportado por SWMM. La cascada usa un regresor condicional y asigna cero cuando el clasificador no predice inundación. El regresor trabaja con `log1p` y se invierte a m³. El ensamble MLP promedia las probabilidades y los volúmenes condicionales de las cinco semillas.
- `node_id`, `run_id`, `scenario_id`, `network_id`, `shape_id` y `factor_mult` permiten identificar, agrupar y rastrear datos; **no forman parte de las 21 entradas**. El factor afecta indirectamente las variables calculadas del escenario.
- `continuity_error_pct`, el estado de drenaje y los sellos de procedencia son controles de elegibilidad y trazabilidad; no son entradas predictoras. El límite absoluto de continuidad del 5 % es provisional.
- Ambos modelos imputan valores faltantes mediante la mediana aprendida en el ajuste. El MLP además estandariza las variables; XGBoost no. Los metadatos indican que no se usa PCA. Los valores faltantes pueden tener significado físico, como un nodo de cabecera sin tuberías entrantes; imputarlos no significa que la tubería exista.
- `duracion_horas` y `tiempo_al_pico_h` aparecen en el contrato histórico de 17 variables y en metadatos auxiliares, pero no están entre las 21 entradas del modelo conjunto; allí se usan las variables tributarias de duración y tiempo al pico descritas arriba. La lluvia, la intensidad de precipitación y las características de transformación lluvia–escorrentía no son entradas de estos candidatos: se trabaja con hidrogramas de caudal por nodo.

## Límites y prioridades para la tesis

1. La cota absoluta y los picos de referencia pueden reflejar características particulares de cada red, datum y escala de los datos. Una buena justificación física no asegura generalización a una tercera red.
2. La sección máxima y una pendiente de salida no describen completamente la capacidad de un nodo. Rugosidad y longitud intervienen en cálculos derivados, pero no son variables individuales de entrada; la rugosidad y otros parámetros siguen interviniendo en SWMM.
3. El resumen de un hidrograma en unos pocos valores no conserva toda su evolución temporal ni el tránsito hidráulico. La auditoría de la reserva muestra sensibilidad importante a una forma prolongada; la hipótesis de que faltan descriptores debe contrastarse en un protocolo nuevo.
4. No se debe inferir que la cota del volumen tributario sea una ley universal: las condiciones iniciales y los fenómenos hidráulicos pueden afectar el balance. La revisión documenta el análisis como exploratorio.
5. Los umbrales, criterios QA, exclusiones y origen no aclarado de `InitFlow` deben acompañar cualquier interpretación de resultados. Las decisiones experimentales y resultados post hoc están en `2026-11-08-resolucion-hallazgos-auditoria.md` y `2026-11-08-revision-independiente-resolucion-auditoria.md`.

Como referencia hidráulica general, el [manual de SWMM 5.2 de la EPA](https://www.epa.gov/system/files/documents/2022-04/swmm-users-manual-version-5.2.pdf) explica la modelación hidráulica y sus opciones; la definición exacta de las 21 variables anteriores procede del código y metadatos locales del proyecto.
