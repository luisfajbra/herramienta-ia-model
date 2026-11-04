# Diseño: sensibilidad hidráulica reproducible

## Pregunta y alcance

El análisis debe mostrar cómo cambian los resultados SWMM de Chico Sur y Chico Norte al variar, por separado, el máximo del paso de enrutamiento y la condición inicial de los conductos. El investigador acordó empezar por estas dos variables el 3 de noviembre de 2026. Un estudio de rugosidad, diámetros u otros parámetros físicos requiere un diseño y rangos sustentados adicionales y no forma parte de esta etapa. Los experimentos operan con hidrogramas nodales, no con lluvia–escorrentía.

El [protocolo exploratorio inmediato](../reports/2026-11-03-protocolo-sensibilidad-ampliada.md) amplía la revisión de Chico Norte a los diez escenarios nuevos ya evaluados. El producto principal será un flujo optativo y parametrizado; no se presenta el piloto como si ya estuviera integrado ni se cambia la configuración ordinaria.

## Contrato experimental

Un protocolo explícito identifica versión, red, selección exacta de escenarios, fuentes y sus SHA-256, motor SWMM y versiones de paquetes, unidades, opciones hidráulicas, estado inicial, política de cierre, máximo del paso, paso mínimo, paso variable, intervalo de reporte, variables, elementos y mallas de comparación. La referencia y las variantes comparten fuente y escenario. Cada combinación tiene una identidad propia; registrar el cambio exacto y verificar que no cambió ninguna otra entrada. Crear directorios nuevos; no sobrescribir experimentos. No reutilizar un RPT/OUT por nombre: verificar hashes y asociación de identidad.

La referencia conserva la configuración de origen. Para sensibilidad de paso se modifica solo su máximo; para sensibilidad de estado inicial se reemplazan los InitFlow explícitamente y se mantiene el paso. Valores ensayados, número de niveles, redes y selección se especifican por protocolo, no por valores universales. Los pasos 5 y 1 s y estado cero son el piloto vigente, no un estándar. La comparación con al menos un paso intermedio o menor requiere probar la compatibilidad con el mínimo y el paso variable, los costos y la sintaxis del motor. No prometer convergencia sin evidencia de refinamiento suficiente y criterios adecuados al uso.

En cada pareja registrar continuidad, porcentaje de pasos no convergentes con su denominador si está disponible, FII y elementos destacados, almacenamiento inicial, drenaje, cobertura, inundación por nodo, suma de inundación y número de nodos inundados. Comparar volúmenes nodales, cambios de clasificación al umbral vigente y diferencias absolutas; reportar el porcentaje relativo solo con denominador no nulo y explicar el redondeo RPT de 1 m³. Los tiempos de reporte, unidades y nodos deben coincidir o la comparación falla. Para series a pasos nativos con mallas distintas, permitir interpolación exclusivamente en la intersección temporal y documentarla; conservar los datos nativos. Sin extrapolación. Seleccionar elementos y ventanas antes de la corrida cuando sea posible, indicar las seleccionadas después y no inferir estabilidad global a partir de ellas.

La QA existente se conserva: cobertura finita, continuidad absoluta ≤5 % provisional y drenaje; no convertir FII o no convergencia en exclusión automática. Registrar los fallos y las corridas incompletas y permitir comparaciones diagnósticas identificadas sin incorporarlas a entrenamiento. No definir tolerancias de equivalencia ni declarar admisión de la red sin justificación acordada. Un estado inicial cero es un supuesto y no una verdad física.

## Integración y seguridad del resultado

Crear un comando optativo explícito de preparación/ejecución/reporte que reuse la infraestructura de corridas y QA existente y que no afecte los comandos normales, los modelos, las bases canónicas ni los resultados congelados. Hacer preparación sin simulaciones y requerir `--execute` para ejecutar. Registrar progreso por corrida, reanudación verificada y fallos; no compartir en paralelo una misma instancia de SWMM. Producir un manifiesto versionado, trazabilidad de archivos y un paquete de evidencia con tablas y figuras legibles. Los OUT grandes y cualquier copia de respaldo deben tener ubicación, tamaño, hash y política de preservación explícitos: un SHA-256 no es una copia.

Antes de usar escenarios que necesiten cierre, corregir y probar el cierre de malla común y materializar el evento efectivo antes de congelarlo; conservar fuente y versión nueva. Antes de utilizar `get_node_series` o `get_link_series`, corregir su origen y unidad de tiempo y probarlo. Los diez eventos de este piloto ya terminan en cero y no requieren esa corrección para sus comparaciones aisladas.

Se entregará un informe de tesis que separe resultados, decisiones, limitaciones, sensibilidad local y validación externa; no convertir diez escenarios sintéticos de una red en evidencia de generalización a otras redes. Las imágenes de inundación mostrarán volúmenes SWMM por nodo con escalas comparables, separadas por variante; cualquier comparación con ML requerirá definir previamente qué modelo y entrada se comparan y no incluir marcas adicionales ajenas al volumen.

## Criterio de terminación

El flujo solo se considera incorporado cuando funcione desde su comando documentado en ambas redes con datos de prueba controlados, compruebe cambios y hashes, reanude sin sobrescribir, documente exclusiones y genere informe reproducible. Una corrida de prueba real debe documentar su cobertura y no equivale por sí sola a admisión hidráulica, entrenamiento conjunto ni validación con mediciones.
