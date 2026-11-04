# Diseño propuesto: base supervisada verificable de Chico Sur y Chico Norte

**Estado:** diseño aprobado por el investigador el 3 de noviembre de 2026; no se ha materializado una base conjunta ni se han entrenado modelos. El [plan de implementación](../plans/2026-11-03-base-conjunta-verificable.md) desglosa las tareas.

## Objetivo y decisiones existentes

Preparar una base nueva que permita comparar y entrenar MLP y XGBoost con ambas redes, sin alterar las bases ni la evaluación de transferencia ya cerrada. El investigador autorizó la admisión provisional de Chico Norte con `InitFlow` fuente y QA vigente, y decidió usar los lotes principales elegibles, manteniendo separados los diez escenarios de la presentación. El objetivo de selección aprobado sigue siendo `macro_network_rmse`. La separación de esos diez escenarios no los convierte en una evaluación ciega, porque ya se revisaron sus resultados.

## Alternativas consideradas

1. **Copiar y fusionar las SQLite directamente.** Menor desarrollo aparente, pero sus claves locales pueden coincidir, Chico Norte no tiene objetivos guardados y no habría una demostración suficiente de correspondencia entre fuentes, etiquetas y filas. Se descarta como procedimiento manual.
2. **Construir una base nueva con importación verificada.** Revalidar los sellos y archivos de las dos fuentes, trasladar los datos de Chico Sur y extraer los objetivos de Chico Norte desde sus RPT elegibles, reconstruir claves globales y generar un manifiesto. Requiere implementación y pruebas, pero preserva el encadenamiento de la evidencia. Es la opción recomendada.
3. **Volver a simular los lotes.** Aumenta tiempo y complejidad y no es necesario si los archivos actuales superan verificaciones; tendría sentido si faltan o fallan las fuentes. No se programan nuevas simulaciones sin comprobarlo.

## Alcance y fuentes

Se inspeccionarán las dos bases y manifiestos congelados identificados en el [inventario](../reports/2026-11-03-preparacion-entrenamiento-conjunto.md). Las fuentes permanecen de solo lectura, y la salida utiliza una ruta nueva y se rechaza si ya existe. Se conservarán las 1.000 decisiones QA de Chico Sur y las 975 de Chico Norte, incluidas las exclusiones y sus motivos; solo corridas elegibles pueden aportar filas supervisadas.

El contrato de variables es `tabular_v4_21` y el umbral de inundación vigente es 1 m³. Para cada corrida se verificarán identidad de red, protocolo y ancla, QA, archivos fuente y hashes, y cobertura exacta de variables y objetivos. Las etiquetas de Chico Norte se extraerán del RPT enlazado de la corrida elegible, con el mismo parser y umbral que el scoring congelado; las excluidas no aportarán etiquetas. El resultado será una base y manifiesto nuevos con la correspondencia entre cada `run_id` de origen y de destino, identidad de red por hash del INP, escenarios, nodos, contratos, fuentes, versión de software y recuentos. Las colisiones de identificadores de nodos o corridas entre redes se resolverán mediante las claves compuestas, nunca por nombre solamente.

Los resultados de evaluación externa previos no se sobrescribirán. La base label-free y sus predicciones seguirán intactas. Los diez escenarios de la presentación no entrarán en esta base. Si falta o cambia un artefacto, la materialización se detendrá y reportará la causa; no se inferirán objetivos ausentes ni se escribirá una base marcada completa. Los hashes detectan alteraciones accidentales, pero no constituyen firmas de autenticidad.

## Grupos y evaluación

Los dos protocolos congelados declaran 39 formas comunes y Chico Sur tiene además la forma `base`. Los recibos de anclaje registran los mismos hashes de los CSV para las 39 formas comunes; se deberán revalidar los archivos efectivos y resolver explícitamente cualquier discrepancia o duplicado físico antes de particionar. Una misma forma y sus factores se mantendrán en el mismo grupo entre redes para evitar que un factor o una red de la misma forma aparezcan a ambos lados de una partición. La identidad y distribución final se documentarán antes del entrenamiento; no se decide aquí qué formas concretas serán de evaluación ni cómo tratar equivalencias no demostradas.

Se mantendrán métricas por red y globales según el diseño previo. Tras usar Chico Norte en ajuste o selección, su resultado sobre eventos reservados corresponde a una red conocida; una red verdaderamente no vista requiere una tercera red independiente. El análisis de sensibilidad sigue documentado por separado y la falta de calibración/validación de campo continúa como límite científico.

## Pruebas y criterios de aceptación

- Con redes y reportes sintéticos, verificar salida supervisada de dos redes, claves únicas aun con IDs y nodos repetidos, QA mixto y cero filas de excluidos.
- Rechazar ancla, INP, evento, RPT u OUT alterado; rechazar duplicados, cobertura incompleta, ausencia de etiquetas, cambios de contrato, base de destino existente y cero elegibles de una red. Simular una falla a mitad del trabajo y comprobar que no se publica una salida como completa.
- Comparar objetivos verificados de Chico Norte con los sellos y recuentos de la evaluación anterior, sin reusar métricas de esa evaluación para seleccionar el modelo; verificar recuentos y hashes de Chico Sur.
- Comprobar el funcionamiento del preparador multired, la integridad de features/objetivos y que los grupos de forma no cruzan particiones. Las pruebas sintéticas no reemplazan la validación hidráulica.
- Producir un informe reproducible con inclusión/exclusión, trazabilidad, recuentos, advertencias y diferencias frente al inventario esperado. Mantener modelos y datos originales sin cambios.

## Decisiones aún no tomadas

La asignación concreta de formas a las particiones, el presupuesto de búsqueda para el entrenamiento conjunto y los criterios de aceptación del modelo no se han fijado. Deben acordarse antes de ejecutar la selección o el entrenamiento; este diseño no autoriza a promover un candidato ni a afirmar capacidad general sobre otras redes.
