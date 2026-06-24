# ML Tabular

> Pipeline XGBoost/SVC para predicción tabular de inundación sin SWMM

**Ruta:** `swmm_resilience/ml/`

## Archivos clave

- [[train.py]] — Entrena y guarda modelos tabulares (clasificación y regresión)
- [[predict_tabular.py]] — Predicción desde CSV con modelos tabulares guardados
- [[predict_from_inp.py]] — Predicción desde .inp sin simulación previa
- [[preprocessing.py]] — Selección de features, limpieza y normalización

## Recibe datos de
[[Dataset ML CSV]] · [[Config]]

## Produce
[[Tabular Model Artifacts]]

## Depende de
[[Config]]
