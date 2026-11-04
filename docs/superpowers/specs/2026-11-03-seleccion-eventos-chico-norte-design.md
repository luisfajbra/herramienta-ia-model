# Diseño: selección verificable de eventos para Chico Norte

## Alcance

La siguiente etapa permite seleccionar expresamente formas de hidrograma normalizadas y factores para Chico Norte, conservando la regla ya acordada para sus pilotos sintéticos: multiplicar cada forma por el pico de caudal del nodo en el INP y por el factor indicado. El resultado es una serie de caudal por nodo, no lluvia ni una transformación lluvia–escorrentía. Esta regla no representa automáticamente hidrogramas medidos o independientes por nodo; un flujo para archivos de caudal absoluto por nodo es una capacidad distinta.

La primera fase valida un protocolo sin ejecutar SWMM. La integración de ejecución posterior debe producir predicciones y evaluación verificadas, respetar exclusiones de QA, escribir en un directorio nuevo y generar gráficos para los casos elegibles. No habilita por sí sola otra red: se exige el SHA-256 de Chico Norte y la política de dominio ya registrada. Redes distintas requieren una política y admisión revisadas. Sin autorización expresa no se sustituyen candidatos, umbrales ni decisiones QA.

## Protocolo versionado

Un documento JSON identifica: versión `chico_norte_event_selection_v1`; ruta y SHA-256 del INP; una lista no vacía de objetos `id`, `path`, `sha256` de CSV normalizados con columnas `time_h,q_norm`; lista explícita de factores; y la regla `nodal_peaks_from_inp`. Las rutas relativas se resuelven respecto al directorio del protocolo y se reportan como rutas absolutas y hashes, no se escriben ni copian durante la inspección. Se rechazan identidades duplicadas, rutas inexistentes, archivos alterados, otras redes, formas inválidas o no finitas, finales distintos de cero y tiempos no representables al minuto según la validación existente. El archivo debe estar en UTF-8 y carecer de claves inesperadas.

Los factores son números finitos dentro del soporte 0,2–5,0 documentado en el experimento anterior, positivos, con como máximo tres decimales y sin duplicados después de esa representación. Esto preserva identidades de escenario inequívocas y evita extrapolación en factor; no demuestra que una forma esté dentro del dominio de entrenamiento. El protocolo debe indicar su naturaleza sintética y no afirma que los eventos sean observados ni que estén reservados respecto al entrenamiento. Seleccionar eventos ya evaluados constituye repetición exploratoria, no nueva prueba ciega.

## Trazabilidad y barreras de seguridad

El resumen incluye SHA-256 del protocolo, de cada forma y de la red, factores, identidades, total de escenarios y regla de generación. Para ejecución se verificarán nuevamente esas fuentes inmediatamente antes de materializar los eventos; se congelarán política, entradas efectivas, QA, modelos y predicciones antes del scoring. Ninguna selección puede modificar el resultado histórico, las bases canónicas o los modelos congelados. Fallos y exclusiones deben quedar identificados; nunca reemplazar por ceros una entrada inválida ni generar métricas sin datos elegibles.

La ejecución y las figuras para eventos seleccionados no quedan habilitadas hasta que existan pruebas sobre configuraciones no históricas, QA parcial y flujo completo. Las gráficas de inundación deben ser independientes por SWMM/MLP/XGBoost y representar únicamente inundación, según la presentación aprobada. No se crean nuevos límites hidráulicos o criterios de aceptación a partir de estos ensayos.
