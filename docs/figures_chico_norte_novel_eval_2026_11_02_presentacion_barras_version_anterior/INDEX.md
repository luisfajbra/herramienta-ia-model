# Figuras para la presentación · Chico Norte

Este paquete contiene figuras de presentación basadas en la evaluación congelada del 2 de noviembre. No cambia simulaciones, predicciones, candidatos ni etiquetas. Se vuelven a leer los 5.220 pares verificados y los RPT originales enlazados por `run_id`; cada hash se valida antes de dibujar.

## Orden y contenido

1. Dos hidrogramas independientes como curvas con marcadores y relleno azul claro, siguiendo el formato de la gráfica anterior de hidrograma. El eje vertical muestra el caudal normalizado `q/qmáx` respecto al tiempo. Son formas sintéticas, no caudales observados.
2. Seis mapas de inundación: factor 0,7, 2,7 y 4,3 para cada forma. Cada mapa compara SWMM, XGBoost y MLP sobre los mismos 522 nodos. La escala de color es común a los tres paneles dentro de cada mapa; cambia entre mapas. Círculos marcan falsas alarmas y cruces las inundaciones omitidas. Si SWMM reporta 0 m³, el cambio de volumen se presenta en m³ y no como porcentaje relativo a cero.
3. Dos paridades globales independientes, una por modelo. Se conservan los puntos y la línea de igualdad; cada figura tiene sus métricas y cubre los mismos 5.220 pares. Los ejes symlog permiten mostrar cero y volúmenes grandes.
4. La figura `volumen_total_por_factor.png` se copia sin cambios desde la evaluación original. Conserva sus rectas y sus dos paneles, uno por hidrograma.

## Archivos

- `hidrograma_novel_20261102_cascade_peaks.png`
- `hidrograma_novel_20261102_sustained_late_surge.png`
- Seis archivos `mapa_<hidrograma>_f<factor>.png`.
- `paridad_global_xgboost.png` y `paridad_global_mlp.png`.
- `volumen_total_por_factor.png`, reutilizada sin modificaciones.
- `presentation_figures_manifest.json` registra fuente de resultados, INP, escenarios y SHA-256 de cada imagen.

## Reproducción

Desde la raíz del repositorio, con los artefactos originales de la evaluación disponibles:

```powershell
.\venv\Scripts\python.exe scripts/build_chico_norte_presentation_figures.py
.\venv\Scripts\python.exe -m scripts.update_weekly_2026_11_02_deck
```

El primer comando no ejecuta SWMM ni entrena modelos. El segundo parte de la presentación aprobada del 30 de octubre y escribe una presentación revisada separada.

El generador de figuras conserva los archivos existentes y se niega a sobrescribir la carpeta de salida; para reproducirlas de nuevo, archive primero `docs/figures_chico_norte_novel_eval_2026_11_02_presentacion/`.

El informe de la evaluación, QA y limitaciones está en [evaluacion-hidrogramas-nuevos-chico-norte.md](../superpowers/reports/2026-11-02-evaluacion-hidrogramas-nuevos-chico-norte.md); el guion oral está en [guion_presentacion_2026-11-02-chico-norte.md](../resultados/guion_presentacion_2026-11-02-chico-norte.md).
