# Próximos pasos — pipeline spec v4

Estado al 2026-06-02. Modelo entrenado, 19 tests pasan, smoke run exitoso.

---

## 1. Métricas en terminal

**Qué:** Al finalizar `--only-ml`, imprimir en consola todos los bloques de métricas:
- LOSO y GroupKFold5 (ambos métodos, actualmente solo se imprime LOSO)
- Los 3 niveles: clasificador, regresor oracle, end-to-end
- Tabla de F1 y RMSE por factor

**Dónde:** Solo `main.py`, bloque de prints al final de `main()`.

**Esfuerzo:** ~30 min.

---

## 2. Curvas de aprendizaje

**Qué:** Gráfico de F1 (clasificador) y NSE (regresor) vs tamaño del training set, usando `sklearn.model_selection.learning_curve`. Sirve para saber si más datos mejorarían el modelo.

**Dónde:** Función nueva en `swmm_resilience/ml/evaluator.py` + flag `--learning-curves` en `main.py`. Guarda `outputs/metrics/learning_curves.png`.

**Esfuerzo:** ~1 h (incluye test).

---

## 3. Visualización del hidrograma

**Qué:** Gráfico de la forma del hidrograma de entrada a la red (tiempo en minutos vs caudal en L/s), para un nodo representativo al factor base Qx1. Útil para entender qué tan agresivo es cada escenario de tormenta.

**Dónde:** Función nueva en `swmm_resilience/visualization/` + flag `--hydrograph` en `main.py`. Lee datos de `get_node_inflow_profiles` (ya disponible). Guarda `outputs/maps/hydrograph_Qx1.png`.

**Esfuerzo:** ~30 min.

---

## 4. Mapas por factor — ya implementado

`--only-maps` ahora genera un mapa por cada factor en el dataset (25 mapas), en lugar de los 5 configurados en `visualization.factors_to_plot`.

---

## 5. Gráfica de la red (`--network-map`)

**Qué:** PNG estático de la topología de la red con tuberías coloreadas por tipo y flechas de dirección de flujo.

**Comportamiento:**
- Cada tubería se dibuja como una línea con una flecha en el extremo del nodo destino.
- **Tubería inicial** (azul): el nodo de origen no tiene ninguna tubería de entrada aguas arriba. Es cabecera de cuenca.
- **Tubería continua** (naranja): el nodo de origen tiene al menos una tubería de entrada aguas arriba. El flujo viene de otra tubería.
- Los nodos se dibujan como puntos neutros sin diferenciación visual.
- Se incluye leyenda con los dos tipos de tubería.
- Guarda en `outputs/maps/network_map.png`.

**Definición formal de "inicial" vs "continua":**
```
nodos_con_entrada = {to_node para cada conduit en la red}
tubería es INICIAL  si su from_node NO está en nodos_con_entrada
tubería es CONTINUA si su from_node SÍ está en nodos_con_entrada
```

**Dónde:** Función nueva `generate_network_map(inp_path, output_path)` en `swmm_resilience/visualization/network_map.py` + flag `--network-map` en `main.py`. Lee coordenadas y conduits de `load_inp` (ya disponible en `swmm_api_io.py`).

**Esfuerzo:** ~1 h (incluye test con red sintética de 3 nodos).

---

## Comandos de referencia actuales

```bash
# Re-entrenar y evaluar desde CSV existente (~30 s)
python main.py --only-ml

# Generar todos los mapas SWMM (25 factores)
python main.py --only-maps

# Predicción ML para un factor específico (genera mapa automáticamente)
python main.py --predict --factor 2.0
python main.py --predict --factor 3.5
python main.py --predict --factor 5.0
```

Salidas principales:
- `outputs/maps/flood_map_factor_*.png` — mapas con datos reales SWMM
- `outputs/maps/flood_map_pred_*.png` — mapas con predicción ML
- `outputs/maps/network_map.png` — topología de la red (pendiente)
- `outputs/metrics/metrics_*.json` — métricas en JSON
- `outputs/metrics/feature_importance_*.png` — importancia de variables
- `outputs/metrics/learning_curves.png` — curvas de aprendizaje (pendiente)
- `outputs/maps/hydrograph_Qx1.png` — hidrograma base (pendiente)
