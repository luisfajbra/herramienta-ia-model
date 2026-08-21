# Spec — Surrogate consciente de la forma del hidrograma

**Fecha:** 2026-06-26
**Origen:** hallazgo C1 de `AUDITORIA_SURROGATE_HIDROGRAMAS.md` — el espacio de features es
ciego a la forma del hidrograma. Verificado empíricamente: ML da ~750 m³ para un CSV de 2 h
y otro de 5 h con el mismo pico, mientras SWMM da 450 m³ y 1200 m³ respectivamente.
**Fuera de alcance:** arquitecturas temporales (LSTM/GNN), heterogeneidad espacial por nodo,
variación de rugosidad Manning.

## Decisiones tomadas

| Decisión | Elección | Razón |
|---|---|---|
| Features nuevas | `duracion_horas` + `tiempo_al_pico_h` | Únicas dimensiones ortogonales ausentes |
| `vol_nodo_m3` / `vol_acum_m3` | Descartadas | `= q_pico × D × C_forma` → collineales con features existentes + las nuevas |
| Factor de fricción Manning | Descartado | Red toda en PVC; rango físico n=0.009–0.011 aporta < 10% variación en capacidad |
| Nivel de surrogate | Feature engineering + XGBoost | < 500 corridas SWMM, objetivo operacional, no demostración de arquitectura temporal |

---

## 1. Diagnóstico raíz

El vector de features actual captura dos dimensiones del hidrograma:

| Dimensión | Feature(s) existentes |
|---|---|
| Topología estática | `elev_fondo`, `prof_max`, diámetros, pendientes, etc. (13 features) |
| Intensidad de pico | `q_pico_nodo`, `q_pico_acum_escalado`, `base_inflow_lps`, `q_pico_acum_base` |
| **Forma temporal** | **ninguna** ← hueco que produce el bug C1 |

Con el dataset actual (barrido de factor sobre un solo hidrograma base), `duracion_horas` es
constante para las 4,000 filas → XGBoost no puede aprender de ella aunque se incluyera. El
fix de features y el nuevo dataset son inseparables.

---

## 2. Features nuevas

Ambas son **atributos del escenario** — escalares idénticos para todos los nodos de un mismo
escenario. Esto es consistente con el tratamiento de `factor_mult` (metadato de escenario que
no entra al modelo).

### 2.1 `duracion_horas`

$$\text{duracion\_horas} = t_n$$

donde $t_n$ es el **último timestamp** de la serie del hidrograma (horas), excluyendo el punto
de drain-down que añade `write_scenario_inp`. Equivale a `scenario.last_time_hours` en
`HydrographScenario`.

Captura: extensión temporal del evento. Distingue 2 h de 5 h con el mismo pico.

### 2.2 `tiempo_al_pico_h`

$$\text{tiempo\_al\_pico\_h} = t_{i^*}, \quad i^* = \arg\max_i\, q_i$$

donde la serie es `[(t_0, q_0), ..., (t_n, q_n)]` en horas/L·s⁻¹. Si hay empate se toma la
primera ocurrencia.

Para nodos sin inflow directo (serie ausente o todo-cero): `tiempo_al_pico_h` toma el valor
del escenario calculado sobre el nodo representativo (primer nodo con serie no-vacía), igual
que `duracion_horas`. Esto refleja que la forma del evento es una propiedad del escenario, no
del nodo.

Captura: agresividad de la subida. Distingue Rare/Extreme (pico a 10 min) de Long Moderate
(pico a 360 min) aunque tengan la misma duración y pico.

### 2.3 Por qué estas dos y no más

Con 7 formas distintas, las dimensiones verdaderamente independientes del hidrograma son:

| Eje | Feature | Independiente de los otros dos |
|---|---|---|
| Intensidad | `q_pico_nodo` (existe) | ✅ |
| Extensión temporal | `duracion_horas` | ✅ |
| Forma de la subida | `tiempo_al_pico_h` | ✅ |


---

## 3. Dataset de entrenamiento ampliado

### 3.1 Formas de hidrograma

| ID | Descripción | Duración | Tiempo al pico |
|---|---|---|---|
| `base` | Hidrograma base del `.inp` (original) | D_base | t_pico_base |
| `common_bogota` | Común Bogotá — subida rápida, pico ~10 min | 120 min | ~10 min |
| `rare_extreme` | Raro/extremo — spike muy agudo | 60 min | ~10 min |
| `common_storm` | Tormenta común — campana simétrica | 360 min | ~150 min |
| `long_moderate` | Larga moderada — rampa gradual | 720 min | ~360 min |
| `shape_2h` | Forma del CSV de validación 2 h — reescalada proporcionalmente | ~120 min | según CSV |
| `shape_5h` | Forma del CSV de validación 5 h — reescalada proporcionalmente | ~300 min | según CSV |

Los CSVs de validación (`csv_2h`, `csv_5h`) tienen valores **absolutos** (p.ej. 40 L/s
por nodo, uniforme). Usarlos directamente como base de factor-augmentation rompería la
proporcionalidad por nodo: nodo 1C (base = 0.5 L/s en `.inp`) recibiría 40 × F L/s →
factor efectivo de hasta 80×, inconsistente con el rango de entrenamiento del resto de
nodos.

La solución: extraer la **forma normalizada** del CSV (dividir cada valor por el pico
máximo del CSV) y reconstruir la serie de entrenamiento por nodo aplicando su
`base_inflow_lps` del `.inp`:

$$q_i^{(v, F)} = \text{base\_inflow\_lps}(v) \times F \times \frac{q_i^{\text{CSV}}}{\max_j q_j^{\text{CSV}}}$$

Así `q_pico_nodo(v) = base_inflow_lps(v) × F` se mantiene idéntico a las demás formas,
preservando la coherencia del dataset. La forma temporal (duración, tiempo al pico) queda
capturada fielmente en `duracion_horas` y `tiempo_al_pico_h`. Los CSVs originales siguen
siendo exclusivamente para **validación**, no para entrenamiento.

### 3.2 Generación de corridas

Para cada forma se generan los 25 factores existentes (0.2–5.0, paso 0.2), escalando la
amplitud por el factor y dejando la forma temporal intacta:

$$q_i^{(F)} = q_i^{\text{base\_forma}} \times F, \quad t_i^{(F)} = t_i^{\text{base\_forma}}$$

| Fuente | Formas | Factores | Corridas SWMM | Filas (×160 nodos) |
|---|---|---|---|---|
| Dataset original | 1 | 25 | 25 | 4,000 |
| Formas nuevas | 6 | 25 | 150 | 24,000 |
| **Total** | **7** | **25** | **175** | **28,000** |

Las 175 corridas usan el mismo `.inp` base de Chico Sur; solo cambia la sección
`[TIMESERIES]` vía `write_scenario_inp` (ya implementado). El drain-down de 6 h se aplica
igual que en validación.

### 3.3 Columnas nuevas en `dataset_final.csv`

Se añaden `duracion_horas` y `tiempo_al_pico_h` como columnas escalares de escenario junto a
`factor_mult` (que permanece como metadato, no como input al modelo).

---

## 4. Cambios de código

### 4.1 `swmm_resilience/extraction/dynamic_features.py`

**`compute_dynamic_features`** (entrenamiento) — añadir dos parámetros:

```python
def compute_dynamic_features(
    static_topo_df: pd.DataFrame,
    factor: float,
    duracion_horas: float = 0.0,     # nuevo
    tiempo_al_pico_h: float = 0.0,   # nuevo
) -> pd.DataFrame:
    df = static_topo_df[["node_id", "base_inflow_lps", "q_pico_acum_base"]].copy()
    df["factor_mult"] = round(factor, 6)
    df["q_pico_nodo"] = df["base_inflow_lps"] * factor
    df["q_pico_acum_escalado"] = df["q_pico_acum_base"] * factor
    df["duracion_horas"] = duracion_horas        # constante por escenario
    df["tiempo_al_pico_h"] = tiempo_al_pico_h    # constante por escenario
    return df[["node_id", "factor_mult", "q_pico_nodo", "q_pico_acum_escalado",
               "duracion_horas", "tiempo_al_pico_h"]]
```

**`compute_scenario_dynamic_features`** (inferencia) — añadir dos parámetros:

```python
def compute_scenario_dynamic_features(
    static_topo_df: pd.DataFrame,
    peak_map: dict[str, float],
    graph: nx.DiGraph,
    duracion_horas: float = 0.0,     # nuevo
    tiempo_al_pico_h: float = 0.0,   # nuevo
) -> pd.DataFrame:
    rows = []
    for nid in static_topo_df["node_id"].astype(str):
        own_peak = float(peak_map.get(nid, 0.0))
        ancestors = nx.ancestors(graph, nid) if graph.has_node(nid) else set()
        accumulated = sum(float(peak_map.get(str(n), 0.0)) for n in ancestors | {nid})
        rows.append({
            "node_id": nid,
            "q_pico_nodo": own_peak,
            "q_pico_acum_escalado": accumulated,
            "duracion_horas": duracion_horas,
            "tiempo_al_pico_h": tiempo_al_pico_h,
        })
    return pd.DataFrame(rows)
```

### 4.2 `swmm_resilience/ml/scenario_predict.py`

Añadir helper y computar escalares antes de llamar a `compute_scenario_dynamic_features`:

```python
def _time_to_peak_h(series: list[tuple[float, float]]) -> float:
    """Tiempo (horas) en que la serie alcanza su máximo. 0.0 si serie vacía."""
    if not series:
        return 0.0
    return max(series, key=lambda x: x[1])[0]
```

En `predict_timed`:

```python
duracion_horas = scenario.last_time_hours

_rep = next((s for s in scenario.node_series.values() if s), [])
tiempo_al_pico_h = _time_to_peak_h(_rep)

dynamic_df = compute_scenario_dynamic_features(
    self.full_df, peak_map, self.graph,
    duracion_horas=duracion_horas,
    tiempo_al_pico_h=tiempo_al_pico_h,
)
```

### 4.3 `swmm_resilience/ml/trainer.py`

```python
FEATURE_COLS = [
    "elev_fondo", "prof_max", "n_tuberias_in", "n_tuberias_out",
    "diam_max_in", "diam_max_out", "pendiente_max_in", "pendiente_out",
    "base_inflow_lps", "dist_outfall_m", "n_nodos_aguas_arriba",
    "q_pico_acum_base", "upstream_capacity_lps",
    "q_pico_nodo", "q_pico_acum_escalado",
    "duracion_horas",     # nuevo
    "tiempo_al_pico_h",   # nuevo
]
```

15 → 17 features. Exige reentrenamiento completo.

### 4.4 Pipeline de entrenamiento (batch + ensamble del dataset)

El loop de entrenamiento necesita recibir, por cada corrida, la forma del hidrograma para
calcular `duracion_horas` y `tiempo_al_pico_h`. La estrategia:

1. Definir las 7 formas como objetos (`HydrographShape`) en config o como archivos CSV de
   referencia (un archivo por forma, mismo formato que los CSVs de validación).
2. El loop externo itera sobre formas; el interno sobre factores — igual que el actual pero
   con una dimensión extra.
3. Para cada forma, `duracion_horas` y `tiempo_al_pico_h` se computan una vez desde la serie
   de referencia y se pasan a `compute_dynamic_features` para las 25 × 160 filas de esa forma.

---

## 5. Contrato de features v3

| # | Feature(s) | Tipo | Origen |
|---|---|---|---|
| 1–8 | `elev_fondo`, `prof_max`, `n_tuberias_in`, `n_tuberias_out`, `diam_max_in`, `diam_max_out`, `pendiente_max_in`, `pendiente_out` | fijo geométrico | `.inp` |
| 9–13 | `base_inflow_lps`, `dist_outfall_m`, `n_nodos_aguas_arriba`, `q_pico_acum_base`, `upstream_capacity_lps` | fijo red/capacidad | `.inp` |
| 14 | `q_pico_nodo` | dinámico por nodo | pico de la serie del escenario |
| 15 | `q_pico_acum_escalado` | dinámico por nodo | suma de picos sobre ancestros topológicos |
| 16 | `duracion_horas` | dinámico por escenario | `scenario.last_time_hours` |
| 17 | `tiempo_al_pico_h` | dinámico por escenario | `t` en `argmax(q)` de la serie |

**17 features** — `factor_mult` permanece como metadato del dataset (agrupación LOSO,
estratificación), nunca como input al modelo.

---

## 6. Estrategia de pruebas

Tests nuevos (TDD):

| Test | Verifica |
|---|---|
| `test_time_to_peak_h_basic` | Serie triangular: devuelve el t correcto |
| `test_time_to_peak_h_tie` | Empate de picos: devuelve primera ocurrencia |
| `test_time_to_peak_h_empty` | Serie vacía: devuelve 0.0 |
| `test_compute_scenario_dynamic_features_shape_cols` | DataFrame resultante tiene `duracion_horas` y `tiempo_al_pico_h` como constantes |
| `test_dynamic_features_training_shape_cols` | Mismo contrato para `compute_dynamic_features` |
| `test_predict_timed_uses_scenario_duration` | Dos escenarios con pico idéntico y duraciones distintas → features distintas en el DataFrame `X` |
| `test_feature_cols_count` | `len(FEATURE_COLS) == 17` |

Tests existentes que requieren actualización de contrato:
- `test_scenario_predict`, `test_hydrograph_batch`, `tests/ml/test_preprocessing_feature_contract`

---

## 7. Criterios de aceptación

1. `pytest tests` en verde con el contrato v3.
2. `duracion_horas` y `tiempo_al_pico_h` toman valores distintos en el DataFrame `X` al
   predecir con el CSV de 2 h vs el de 5 h — verificable con un assertion antes de `clf.predict`.
3. Dataset regenerado con 28,000 filas, con columnas `duracion_horas` y `tiempo_al_pico_h`
   no constantes entre escenarios.
4. Modelos reentrenados con 17 features; `training_inp_hash.txt` actualizado.
5. Al correr `--evaluate-hydrographs` con los dos CSVs (2 h y 5 h), `vol_pred_m3` difiere
   entre ellos — el bug C1 queda cerrado empíricamente.
