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

Opcional: para usar un hidrograma externo, crea un CSV como `data/hydrographs/example_hydrograph.csv` con este formato:

```csv
minute,inflow_lps
0,0
5,10
10,25
15,40
20,30
25,15
30,0
```

Luego edita [swmm_resilience/config.py](swmm_resilience/config.py):

```python
DEFAULT_HYDROGRAPH_FILE = HYDROGRAPHS_DIR / "example_hydrograph.csv"
DEFAULT_TARGET_NODES = None
```

Usa `DEFAULT_TARGET_NODES = None` para todos los nodos, o una lista como `["NODO_1", "NODO_2"]` para un subgrupo.

```bash
python main.py
```

## 5. Revisar resultados

- SQLite central: `data/training/swmm_resilience.db`
- CSV por red: `data/networks/chico_steady/results/dataset_ml.csv`
