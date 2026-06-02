# Quickstart

## 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

## 2. Revisar Configuracion

Edita `config.yaml` y confirma:

- `network.inp_path`
- `simulation.factor_min`, `simulation.factor_max`, `simulation.factor_step`
- `dataset.output_path`
- algoritmos en `ml.classifier.algorithm` y `ml.regressor.algorithm`
- factores de `visualization.factors_to_plot`

## Ejecucion Recomendada

1. Revisa `config.yaml`.
2. Ejecuta `python main.py --only-ml` si ya existe
   `data/training/dataset_final.csv`.
3. Ejecuta `python main.py --only-maps` para regenerar mapas desde el CSV.
4. Ejecuta `python main.py --predict --factor 3.5` para inferencia sin SWMM.
5. Ejecuta `python main.py` solo cuando quieras recalcular simulaciones SWMM.

## Comandos Utiles

```bash
python main.py --only-ml
python main.py --only-maps
python main.py --predict --factor 3.5
python main.py --skip-extraction
python main.py --skip-simulation --skip-extraction
```

## Resultados

- Dataset: `data/training/dataset_final.csv`
- Modelos: `outputs/models/classifier.joblib` y
  `outputs/models/regressor.joblib`
- Metricas: `outputs/metrics/metrics_classifier.json`,
  `outputs/metrics/metrics_regressor.json`,
  `outputs/metrics/metrics_endtoend.json` y
  `outputs/metrics/metrics_by_factor.json`
- Mapas: `outputs/maps/*.png`

## Verificacion Rapida

```bash
python -m pytest tests -v
python -m compileall main.py swmm_resilience
```
