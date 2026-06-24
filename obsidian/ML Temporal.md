# ML Temporal

> Surrogate CNN/LSTM para predicción temporal de inundación por multiplicador

**Ruta:** `swmm_resilience/ml/temporal/`

## Archivos clave

- [[dataset.py]] — Construye ventanas temporales y dataset surrogate desde parquets
- [[train_surrogate.py]] — Entrena CNN/LSTM surrogate con GroupKFold y guarda artefactos
- [[predict.py]] — Inferencia surrogate por multiplicador de caudal
- [[compare_surrogate.py]] — Compara XGBoost vs CNN vs LSTM en splits idénticos
- [[surrogate_cnn.py]] — Arquitectura CNN dual-branch para predicción de inundación
- [[surrogate_lstm.py]] — Arquitectura LSTM dual-branch para predicción de inundación

## Recibe datos de
[[Parquets Timeseries]] · [[SQLite DB]] · [[Config]]

## Produce
[[Surrogate Weights]] · [[Surrogate Maps]]

## Depende de
[[Database]] · [[Config]]
