# Próximos pasos — pipeline spec v4

Estado al 2026-06-02. Modelo entrenado, 25 tests pasan, smoke run exitoso.

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

## ✅ 3. Visualización del hidrograma — IMPLEMENTADO

Comando: `python main.py --hydrograph`

Selecciona automáticamente el nodo con mayor caudal pico (nodo 87I, 26 L/s) y genera `outputs/maps/hydrograph_Qx1.png` con el hidrograma de entrada (tiempo en min vs caudal en L/s).

---

## 4. Etiquetas descriptivas en gráficas de importancia de variables

**Qué:** En las dos gráficas de importancia (`feature_importance_classifier.png` y `feature_importance_regressor.png`), reemplazar los nombres técnicos de las variables por su nombre completo descriptivo en español. El cambio es **únicamente visual** — el dataset, los modelos y el código interno no se tocan.

**Mapa de nombres (validar antes de implementar):**

| Variable técnica | Nombre en la gráfica |
|---|---|
| `elev_fondo` | Elevación del fondo del nodo (m) |
| `prof_max` | Profundidad máxima del nodo (m) |
| `n_tuberias_in` | Número de tuberías de entrada |
| `n_tuberias_out` | Número de tuberías de salida |
| `diam_max_in` | Diámetro máximo de tubería de entrada (m) |
| `diam_max_out` | Diámetro máximo de tubería de salida (m) |
| `pendiente_max_in` | Pendiente máxima de tubería de entrada (m/m) |
| `pendiente_out` | Pendiente de tubería de salida (m/m) |
| `base_inflow_lps` | Caudal de entrada base (L/s) |
| `dist_outfall_m` | Distancia al punto de descarga (m) |
| `n_nodos_aguas_arriba` | Número de nodos aguas arriba |
| `q_pico_acum_base` | Caudal pico acumulado base (L/s) |
| `upstream_capacity_lps` | Capacidad de transporte aguas arriba (L/s) |
| `factor_mult` | Factor de escala de lluvia |
| `q_pico_nodo` | Caudal pico en el nodo escalado (L/s) |
| `q_pico_acum_escalado` | Caudal pico acumulado (L/s) |

**Dónde:** `swmm_resilience/ml/feature_importance.py`, diccionario de mapeo aplicado solo al eje Y del gráfico antes de `plt.savefig`.

**Esfuerzo:** ~20 min.

---

## ✅ 5. Mapas por factor — IMPLEMENTADO

`--only-maps` genera un mapa por cada factor en el dataset (25 mapas).

---

## ✅ 6. Gráfica de la red (`--network-map`) — IMPLEMENTADO

Comando: `python main.py --network-map`

Genera `outputs/maps/network_map.png` con tuberías iniciales (azul), continuas (gris), flechas de flujo en el punto medio de cada tubería, nodos como círculos negros y nodo de salida como triángulo.

---

## ✅ 7. Curva de resiliencia (`--resilience-curve`) — IMPLEMENTADO

Comando: `python main.py --resilience-curve`

Calcula `resiliencia = nodos no inundados / total nodos` por factor, comparando datos reales SWMM vs predicción ML. Genera dos PNGs separados:
- `outputs/metrics/resilience_swmm.png` — curva SWMM (azul)
- `outputs/metrics/resilience_ml.png` — curva ML (naranja)

Imprime la tabla completa de resiliencia por factor en terminal.

---

## 8. Curva de volumen de inundación total (`--flood-volume-curve`)

**Qué:** Curva que muestra el **volumen total inundado de la red** (suma de `vol_inundacion_m3` de todos los nodos) por cada factor multiplicador, comparando datos reales SWMM vs predicción ML. Complementa la curva de resiliencia: mientras esta dice cuántos nodos fallan (binario), esta dice cuánto volumen se acumula (magnitud).

**Fórmula:**
```
vol_total_swmm(factor) = Σ vol_inundacion_m3  para todos los nodos en ese factor
vol_total_ml(factor)   = Σ vol_pred_m3         de predict_network(factor)
```

**Dónde:** Función nueva `compute_flood_volume_curve` en `swmm_resilience/analysis/resilience.py` + función `plot_flood_volume_curve(df, output_dir)` en `swmm_resilience/visualization/resilience_curve.py` + flag `--flood-volume-curve` en `main.py`. Genera dos PNGs: `outputs/metrics/flood_volume_swmm.png` y `outputs/metrics/flood_volume_ml.png`.

**Esfuerzo:** ~45 min (estructura idéntica a la curva de resiliencia).

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

# Hidrograma del nodo con mayor caudal pico
python main.py --hydrograph

# Mapa de topología de la red
python main.py --network-map

# Curva de resiliencia SWMM vs ML
python main.py --resilience-curve

# Curva de volumen de inundación total SWMM vs ML (pendiente)
python main.py --flood-volume-curve
```

Salidas principales:
- `outputs/maps/flood_map_factor_*.png` — mapas con datos reales SWMM ✅
- `outputs/maps/flood_map_pred_*.png` — mapas con predicción ML ✅
- `outputs/maps/network_map.png` — topología de la red ✅
- `outputs/maps/hydrograph_Qx1.png` — hidrograma base ✅
- `outputs/metrics/metrics_*.json` — métricas en JSON
- `outputs/metrics/feature_importance_*.png` — importancia de variables
- `outputs/metrics/resilience_swmm.png` — curva de resiliencia SWMM ✅
- `outputs/metrics/resilience_ml.png` — curva de resiliencia ML ✅
- `outputs/metrics/flood_volume_swmm.png` — curva de volumen total SWMM (pendiente)
- `outputs/metrics/flood_volume_ml.png` — curva de volumen total ML (pendiente)
- `outputs/metrics/learning_curves.png` — curvas de aprendizaje (pendiente)
