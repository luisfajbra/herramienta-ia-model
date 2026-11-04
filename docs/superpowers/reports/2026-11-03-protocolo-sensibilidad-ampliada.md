# Protocolo exploratorio: sensibilidad hidráulica ampliada de Chico Norte

**Resultado:** las simulaciones y verificaciones completadas se describen en el [informe de sensibilidad ampliada](2026-11-03-sensibilidad-hidraulica-ampliada.md).

**Fecha:** 3 de noviembre de 2026. Este protocolo se escribió antes de ejecutar las ocho comparaciones adicionales. No constituye un registro externo independiente.

## Objetivo y alcance

Determinar cómo cambian los resultados de SWMM al modificar por separado el máximo del paso de enrutamiento y el caudal inicial de los conductos. La primera etapa fue acordada con el investigador; no se modifican rugosidades, pendientes ni diámetros. Las entradas son hidrogramas sintéticos de caudal por nodo, no lluvia ni observaciones de campo.

## Casos

Se toman **todos los diez escenarios** de los dos hidrogramas sintéticos nuevos de la evaluación anterior: pulsos sucesivos y meseta con incremento tardío; factores 0,7; 1,3; 2,7; 3,9 y 4,3 para cada uno. No son una muestra aleatoria representativa de tormentas ni de redes. Las formas y factores ya se habían usado para evaluar modelos y sus resultados se conocían; esta es una ampliación exploratoria, no una prueba ciega independiente.

Se reutilizan las tres variantes ya realizadas para pulsos factor 0,7 y meseta factor 4,3. Se ejecutan los ocho casos restantes, con tres variantes cada uno: **24 simulaciones nuevas**, además de las seis ya disponibles. El caso histórico crítico de la revisión inicial se mantiene como caso complementario separado, no como parte de los diez.

Para cada caso, el código comprueba los hashes de INP efectivo, evento, RPT y OUT de la corrida congelada antes de ejecutar. Las variantes se hacen sobre copias en un directorio nuevo. No se sustituyen resultados históricos, no se modifican los modelos ni se escribe en la base de entrenamiento.

## Variantes y comparación

1. Referencia: InitFlow del INP fuente y máximo de enrutamiento de 5 s.
2. Paso: los mismos InitFlow e hidrogramas, máximo de enrutamiento de 1 s.
3. Estado inicial: InitFlow cero en los 522 conductos, máximo de 5 s.

Se conserva `VARIABLE_STEP=0,75` y `MINIMUM_STEP=0,5 s`; los máximos no son pasos constantes. Se usa reporte cada minuto para comparar en las mismas fechas. Se comprobará que la referencia reproduzca los volúmenes nodales y los diagnósticos principales del RPT congelado. Una discrepancia se investigará y no se ocultará.

Se informarán, por caso y variante: continuidad, porcentaje de pasos sin convergencia y FII máximo, suma de volúmenes nodales inundados y número de nodos con al menos 1 m³. Frente a la referencia se documentarán la diferencia total, el número de nodos cuyo volumen cambió y la diferencia absoluta máxima por nodo. Se compararán también las series de T507, T508, T463, T92, 523C y 396I a resolución de un minuto, sin presentar esas ubicaciones como representativas de toda la red. Para identificar oscilaciones más rápidas se requiere inspección de pasos internos adicional; el reporte de un minuto no basta para descartarlas.

El volumen nodal del RPT está redondeado a 1 m³. El porcentaje de pasos no convergentes tiene un denominador que cambia al variar el paso; una reducción no demuestra convergencia temporal. Se señalarán por separado los escenarios sin inundación y no se dividirá por un volumen de referencia cero. No se calculará ningún porcentaje de confiabilidad sin un método y observaciones que lo sustenten.

La QA existente (cobertura finita, continuidad absoluta ≤5 % y drenaje bajo las tolerancias vigentes) sigue siendo provisional. No se crea un criterio automático de rechazo por FII ni por no convergencia. InitFlow cero es una hipótesis de sensibilidad y no una condición validada para generar datos de entrenamiento.

## Límites y trazabilidad

La comparación es numérica y de hipótesis iniciales, no una validación hidráulica con mediciones. Una tercera red y eventos observados harían falta para estudiar generalización a redes y fenómenos reales. Los diez escenarios estaban disponibles antes del estudio; no se usan para seleccionar hiperparámetros ni modificar los modelos en este análisis.

El código de ejecución es `scripts/run_chico_norte_extended_sensitivity.py`, que invoca `scripts/review_chico_norte_hydraulics.py`. Cada ejecución genera `design.json`, INP/RPT/OUT, resultados nodales, hashes y `results.json` incremental. Los OUT son grandes y están bajo `outputs/`, directorio ignorado por Git: conservar o respaldar esos archivos requiere una acción específica. El informe final indicará rutas, verificaciones y cualquier fallo.

Como referencia metodológica, el [manual de usuario SWMM 5.2 de EPA](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P10145M6.TXT) distingue paso de cálculo y de reporte y recomienda examinar series temporales para buscar inestabilidades.
