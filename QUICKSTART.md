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
DEFAULT_DELTA_INFLOWS_LPS = [5 * step for step in range(1, 21)]
```

## 4. Ejecutar

Para abrir la aplicacion local con formulario:

```bash
python app.py
```

Tambien puedes ejecutar el pipeline directo desde consola.

```bash
python main.py
```

## 5. Revisar resultados

- SQLite central: `data/training/swmm_resilience.db`
- CSV por red: `data/networks/chico_steady/results/dataset_ml.csv`
