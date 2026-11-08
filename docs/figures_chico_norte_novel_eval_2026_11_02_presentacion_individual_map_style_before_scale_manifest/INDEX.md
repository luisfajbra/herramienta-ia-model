# Figuras para la presentación · Chico Norte

Material basado en la evaluación congelada y en predicciones/RPT verificados por hash. No se ejecutó SWMM ni se reentrenaron modelos para construir estas figuras.

## Orden y estilo

1. Dos hidrogramas separados, como curvas azules con marcadores y relleno claro, siguiendo el estilo del flujo completo.
2. Dieciocho mapas independientes: SWMM, MLP y XGBoost para cada uno de seis escenarios (factores 0,7, 2,7 y 4,3 en ambas formas). Cada mapa dibuja la red en gris, los nodos secos en gris claro, los volúmenes inundados en `YlOrRd` y su propia barra de color. No se superponen marcas de falsas alarmas ni omisiones. Las escalas son independientes y los colores no se comparan entre mapas.
3. Dos paridades globales de puntos, separadas por modelo, con los colores de XGBoost y MLP, línea de igualdad y ejes `log10(1 + volumen)`.
4. La gráfica de rectas de volumen total por factor, copiada sin cambios desde la evaluación original.

El manifiesto `presentation_figures_manifest.json` registra hashes de imágenes, escenarios y resumen por mapa. El guion y los resultados están en [el reporte de evaluación](../superpowers/reports/2026-11-02-evaluacion-hidrogramas-nuevos-chico-norte.md) y [el guion oral](../resultados/guion_presentacion_2026-11-02-chico-norte.md).
