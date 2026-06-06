# Spec: Transformación Logarítmica del Target del Regresor

**Fecha:** 2026-06-01  
**Estado:** Aprobado para implementación

---

## Problema

Las métricas del regresor de volumen de inundación (`vol_inundacion_m3`) son pobres. La causa raíz es que el target tiene distribución muy sesgada a la derecha: la mayoría de nodos inundados tienen volúmenes pequeños (1–50 m³) y unos pocos tienen volúmenes grandes (200–300 m³). El error cuadrático medio penaliza proporcionalmente más los errores en valores grandes, haciendo que el modelo optimice casi exclusivamente para los picos y tenga mala calibración en el rango frecuente.

## Solución

Aplicar transformación `log1p` al target durante el entrenamiento del regresor. Las predicciones se revierten con `expm1` antes de calcular métricas o exportar resultados. El usuario nunca ve valores en escala logarítmica.

Se añade `log_nse` (NSE calculado en escala log) como métrica adicional al regresor, complementaria al NSE en m³. En hidrología el log-NSE es estándar porque es más sensible a errores en eventos de baja magnitud.

## Archivos afectados

| Archivo | Cambio |
|---|---|
| `swmm_resilience/ml/trainer.py` | fit regresor con `np.log1p(y_reg)` |
| `swmm_resilience/ml/evaluator.py` | fit por fold con `np.log1p`, expm1 en predict, añadir `log_nse` |
| `swmm_resilience/ml/predict.py` | expm1 al resultado de `reg.predict(...)` |

## Cambios detallados

### trainer.py

```python
# ANTES
reg.fit(df_flooded[FEATURE_COLS], df_flooded["vol_inundacion_m3"])

# DESPUÉS
reg.fit(df_flooded[FEATURE_COLS], np.log1p(df_flooded["vol_inundacion_m3"]))
```

### evaluator.py — fit por fold

```python
# ANTES
reg.fit(X_tr[flooded_tr], yr_tr[flooded_tr])

# DESPUÉS
reg.fit(X_tr[flooded_tr], np.log1p(yr_tr[flooded_tr]))
```

### evaluator.py — Level 2 (oracle)

```python
# ANTES
yr_pred_oracle = reg.predict(X_te[flooded_te])

# DESPUÉS
yr_pred_oracle = np.expm1(reg.predict(X_te[flooded_te]))
# yr_true_oracle permanece en m³ (sin transformar)
```

Métricas del regresor tras el cambio:

```python
reg_m.append({
    "nse":     _nse(yr_true_oracle, yr_pred_oracle),                        # m³
    "log_nse": _nse(np.log1p(yr_true_oracle), np.log1p(yr_pred_oracle)),    # log space
    "rmse":    float(np.sqrt(mean_squared_error(yr_true_oracle, yr_pred_oracle))),
    "mae":     float(mean_absolute_error(yr_true_oracle, yr_pred_oracle)),
    "r2":      float(r2_score(yr_true_oracle, yr_pred_oracle)),
})
```

### evaluator.py — Level 3 (end-to-end)

```python
# ANTES
yr_pred_e2e[clf_flood_mask] = reg.predict(X_te[clf_flood_mask])

# DESPUÉS
yr_pred_e2e[clf_flood_mask] = np.expm1(reg.predict(X_te[clf_flood_mask]))
```

Las métricas end-to-end (`rmse_vol_todos_nodos`, `vol_total_pred_m3`, `vol_total_real_m3`) ya trabajan en m³ y no cambian de definición.

### predict.py

```python
# ANTES
vol_pred[flood_mask] = reg.predict(X.values[flood_mask])

# DESPUÉS
vol_pred[flood_mask] = np.expm1(reg.predict(X.values[flood_mask]))
```

## Invariantes

- El dataset `dataset_final.csv` **no cambia** — `vol_inundacion_m3` siempre se almacena en m³.
- `log1p` y `expm1` son inversas exactas: `expm1(log1p(x)) == x` para cualquier `x ≥ 0`.
- El regresor sigue entrenándose **solo sobre filas con `inunda=1`** — no hay ceros en el target de entrenamiento del regresor (los nodos con `vol=0` tienen `inunda=0` y no llegan al regresor).
- Todos los outputs visibles al usuario (métricas JSON, mapas, predicciones) permanecen en m³.

## Métricas de éxito

El cambio es exitoso si, después de re-entrenar:
- `log_nse` > 0 (el modelo supera la media como predictor en escala log).
- `nse` en m³ mejora respecto al valor anterior (baseline: −88.5).
- `rmse` en m³ se mantiene igual o baja (puede subir ligeramente si el modelo ahora prioriza eventos pequeños sobre grandes).
- `log_nse − nse_m³` ≤ 0.3 — si el gap es mayor, el modelo predice bien eventos pequeños pero sigue fallando en los grandes. Un gap amplio es una señal de análisis relevante para la tesis: indica que la transformación corrigió el sesgo en eventos frecuentes pero la capacidad predictiva en eventos extremos sigue siendo limitada.
