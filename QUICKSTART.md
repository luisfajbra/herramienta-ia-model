# Quickstart

## 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 2. Poner el archivo SWMM

Coloca el `.inp` en:

```text
data/networks/
```

## 3. Ajustar caudales de inyección

Edita [swmm_resilience/config.py](swmm_resilience/config.py):

```python
DEFAULT_DELTA_INFLOWS_M3PS = [0.005 * step for step in range(1, 21)]
```

## 4. Ejecutar

```bash
python main.py
```

## 5. Revisar resultados

- SQLite: `data/results/swmm_resilience.db`
- CSV: `data/results/dataset_ml.csv`
