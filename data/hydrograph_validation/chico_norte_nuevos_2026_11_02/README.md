# Hidrogramas sintéticos de validación — Chico Norte

Estos dos CSV contienen formas temporales sintéticas normalizadas (`time_h`, `q_norm`) creadas para la evaluación externa de modelos en Chico Norte, ejecutada el 2 de noviembre de 2026. `q_norm` está en [0, 1], empieza y termina en cero y alcanza un pico de uno. El ejecutor multiplica la forma por los picos nodales del INP de Chico Norte y por cada factor seleccionado.

- `novel_20261102_cascade_peaks.csv`: pulsos dobles separados, duración 7 h.
- `novel_20261102_sustained_late_surge.csv`: meseta sostenida con pico tardío, duración 16 h.

Son entradas inventadas para validación, no observaciones ni lluvia. **No forman parte del directorio de formas de entrenamiento** y no deben incorporarse al set de entrenamiento sin una decisión explícita y una nueva versión del experimento. Los hashes, factores aleatorios, sellos, QA y resultados están en `docs/resultados/evidencia_evaluacion_chico_norte_2026_11_02/`; la interpretación está en el informe de evaluación correspondiente.
