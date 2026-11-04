# Selección verificable de eventos de Chico Norte

**Fecha:** 3 de noviembre de 2026. **Objetivo:** permitir seleccionar formas y factores explícitos para volver a simular Chico Norte con SWMM y evaluar los candidatos congelados entrenados con Chico Sur. Esta funcionalidad no entrena con Chico Norte ni incorpora una tercera red.

## Procedimiento y trazabilidad

El protocolo JSON declara las formas normalizadas, sus SHA-256, el INP aprobado y los factores. El lector comprueba el formato y la integridad. Se generan hidrogramas por nodo multiplicando la forma por el pico nodal del INP y el factor seleccionado; son entradas sintéticas, no mediciones. Los factores admitidos están entre 0,2 y 5,0: pertenecer a ese intervalo no demuestra por sí solo que una forma esté dentro del dominio aprendido.

La congelación registra fuentes, configuración, eventos y un ancla SQLite separada. La ejecución comprueba el ancla y los insumos, conserva los reportes SWMM y resultados de QA. Una etapa posterior registra las predicciones de ambos modelos y las compara con las etiquetas extraídas de las simulaciones verificadas, sin escribirlas en la base canónica. Se generan mapas individuales por SWMM, MLP y XGBoost, paridades separadas, hidrogramas y curvas de volumen por factor. Solo entran al scoring y figuras los escenarios elegibles; los fallidos se reportan. La paridad se omite cuando los valores verdaderos son constantes y el R² no está definido. Los comandos están en el README.

Se conservan los criterios provisionales ya acordados: cobertura finita, drenaje y continuidad absoluta hasta 5 %. No se añadió exclusión por FII. Las tolerancias requieren justificación hidráulica futura. Esta entrega reutiliza la implementación de QA existente. Posteriormente se verificó con un lote sintético mixto la importación, el scoring y la unión para las figuras; la inferencia real y el renderizado completo con lote mixto siguen pendientes. Véase el [informe de prueba](2026-11-03-prueba-qa-mixto.md).

## Pilotos de verificación

Se repitió el hidrograma de pulsos dobles previamente documentado con factores 0,7 y 1,3, en ejecuciones aisladas. Se contrastaron sus métricas por escenario con la evaluación congelada anterior. Son repeticiones para comprobar la implementación, **no dos escenarios nuevos independientes**. Cada piloto tiene 522 nodos y fue elegible; ninguna simulación se excluyó.

| Factor | SWMM: volumen inundado | MLP: RMSE nodal | XGBoost: RMSE nodal |
|---|---:|---:|---:|
| 0,7 | 0 m³ | 0,680 m³ | 0,801 m³ |
| 1,3 | 5 m³ | 2,398 m³ | 6,203 m³ |

Para 1,3 SWMM identificó un nodo inundado y ambos modelos lo omitieron. MLP señaló 34 nodos adicionales y XGBoost 20. Estos resultados muestran por qué un RMSE pequeño en un conjunto dominado por ceros no basta para sostener que el modelo predice bien todas las inundaciones. El umbral de inundación es 1 m³. Para 0,7 no se genera paridad porque la referencia es constante. En los mapas del piloto 1,3 cada panel registra su propia escala de color: los colores iguales entre paneles no equivalen necesariamente al mismo volumen. Consultar las barras y el manifiesto antes de compararlos.

Las ejecuciones completas están en `outputs/experiments/piloto_seleccion_swmm_20261103` y `outputs/experiments/piloto_seleccion_inundado_20261103`; las figuras están en `selected_figures`. Mantener junto a cada carpeta su archivo adyacente `*_protocol_anchor.sqlite3`. Los manifiestos incluyen rutas absolutas originales: una restauración en otra ruta puede necesitar ajustes antes de repetir el flujo.

## Pendientes de presentación registrados

En la revisión posterior, el investigador decidió documentar estos puntos y continuar primero con el flujo de entrenamiento:

1. **Escalas independientes en mapas:** los mapas individuales calculan sus propios máximos de color y tamaño de punto. Un color igual en SWMM, MLP y XGBoost puede representar volúmenes diferentes. Por ejemplo, en el piloto de factor 1,3 los máximos fueron 5,0, 25,738 y 86,394 m³, respectivamente. La escala correspondiente queda registrada en el manifiesto de cada figura. Para una futura comparación visual se propone una escala común por escenario; mientras tanto deben consultarse las barras de color.
2. **Redondeo del factor en el título:** el protocolo admite factores con hasta tres decimales, pero el título del mapa presenta uno. Por ejemplo, factores 0,234 y 0,249 aparecerían como 0,2. Los archivos y el manifiesto conservan el factor con tres decimales. Ajustar el título queda pendiente antes de presentar escenarios con mayor precisión; esto no cambia el factor simulado ni las métricas.

Ambos puntos afectan la interpretación o identificación visual; no se han modificado las figuras ni los resultados congelados.

## Alcance pendiente

Antes de usar otra red se necesita su admisión hidráulica y un contrato de entradas y features probado. También faltan ampliar la prueba de QA parcial a inferencia y renderizado reales, una evaluación independiente adicional y el diseño de entrenamiento conjunto. No debe atribuirse el resultado de transferencia a entrenamiento con Chico Norte.
