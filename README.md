# SWMM Resilience

Pipeline spec v4 para generar escenarios SWMM, construir un dataset tabular,
entrenar modelos de inundacion por nodo y producir metricas, mapas e
inferencia rapida sin volver a correr SWMM.

El flujo principal se configura desde `config.yaml`.

## Pipeline Spec V4

```bash
python main.py
python main.py --skip-extraction
python main.py --only-ml
python main.py --only-maps
python main.py --predict --factor 3.5
```

Salidas principales:

- `data/training/dataset_final.csv`
- `outputs/models/classifier.joblib`
- `outputs/models/regressor.joblib`
- `outputs/models/training_inp_hash.txt`
- `outputs/metrics/*.json`
- `outputs/maps/*.png`

## Modos De Uso

- `python main.py`: ejecuta el pipeline completo desde `config.yaml`.
- `python main.py --skip-extraction`: reutiliza `dataset.output_path` y salta
  extraccion/simulacion.
- `python main.py --skip-simulation --skip-extraction`: modo honesto para
  reutilizar CSV existente sin intentar reconstruir reportes `.rpt`.
- `python main.py --only-ml`: entrena, evalua y genera importancia de variables
  desde `data/training/dataset_final.csv`.
- `python main.py --only-maps`: regenera mapas desde el CSV y la red del config.
- `python main.py --predict --factor 3.5`: predice nodos inundados y volumen
  usando los modelos guardados en `outputs/models`.

## Arquitectura Actual

```text
main.py
config.yaml
swmm_resilience/
  config.py
  simulation/
    batch.py
    runner.py
  extraction/
    static_features.py
    dynamic_features.py
    labels.py
    assembler.py
  dataset/
    validator.py
  ml/
    trainer.py
    evaluator.py
    feature_importance.py
    predict.py
  visualization/
    flood_map.py
```

La version spec v4 no usa frontend desktop, SQLite ni los modulos temporales
legacy como flujo principal. Es un pipeline CLI basado en CSV, modelos joblib,
metricas JSON y mapas PNG.

## Modelos Y Metricas

El clasificador predice `inunda`. El regresor predice `vol_inundacion_m3` solo
para nodos inundados; se entrena en espacio `log1p` y las predicciones se
devuelven a m3 con `expm1`.

La evaluacion reporta:

- clasificador aislado
- regresor oracle con etiquetas reales para filtrar inundados
- sistema end-to-end con etiquetas predichas
- estratificacion por `factor_mult`

`metrics_regressor.json` incluye `nse` y `log_nse`. El `log_nse` se calcula en
espacio logaritmico sobre predicciones out-of-fold apiladas para evitar que
folds LOSO con muy pocos nodos inundados dominen la metrica global.

## Instalacion

```bash
pip install -r requirements.txt
```

Dependencias principales:

- `pyswmm` y `swmm-api` para SWMM
- `pandas`, `numpy`, `scikit-learn` y `xgboost` para ML
- `matplotlib` y `networkx` para mapas
- `pytest` para verificacion

## Verificacion

```bash
python -m pytest tests -v
python -m compileall main.py swmm_resilience
```
