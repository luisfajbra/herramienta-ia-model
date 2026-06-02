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
- `outputs/metrics/metrics_*.json` — métricas en JSON
- `outputs/metrics/feature_importance_*.png` — importancia de variables
