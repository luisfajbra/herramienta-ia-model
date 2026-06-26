# Tiempo de cómputo estampado en los mapas de volumen (SWMM y ML)

Fecha: 2026-06-20

## Objetivo

Que cada vez que se corra la validación batch (`--evaluate-hydrographs`) —donde
SWMM y el ML se ejecutan de verdad por escenario— los mapas de volumen de
inundación incluyan, **en la esquina derecha del mapa**, el **tiempo de corrida**
de cada uno (en segundos). Así se ve directamente que el surrogate (XGBoost)
tiene menor costo computacional.

El usuario calcula el speed-up por su cuenta en una tabla aparte; los mapas
**solo** muestran el tiempo en segundos (sin speed-up, sin texto extra).

No se corre nada extra: el batch ya ejecuta y ya cronometra SWMM y ML por
escenario; solo falta llevar esos tiempos ya medidos a la esquina de los mapas.

## Alcance

- **Incluido:** mapas de volumen del batch `--evaluate-hydrographs`
  (`plot_scenario_flood_maps`): mapa SWMM y mapa ML, por escenario.
- **Fuera de alcance (YAGNI):** speed-up en los mapas (lo saca el usuario aparte),
  `--predict`, `--factor-comparison`, tablas-imagen de tiempos, módulo de tiempos
  compartido, flags nuevos, runs extra de SWMM.

## Comportamiento

Por cada escenario, el batch ya mide (en el bucle de `run_batch_validation`,
hydrograph_batch.py):

- `t_swmm` (tiempo de correr SWMM),
- `t_features_s` + `t_inference_s` (tiempo del ML).

Esos valores se pasan a `plot_scenario_flood_maps`, que los dibuja como una
anotación en la **esquina inferior derecha** de cada mapa:

- Mapa SWMM: `Tiempo de cómputo: 1.85 s`
- Mapa ML:   `Tiempo de cómputo: 0.0240 s`

(Los números son ejemplos de formato; los reales los pone el cronómetro.) Los
títulos de los mapas no cambian ("SWMM Simulation" / "ML Prediction").

## Arquitectura

### Helper puro de formato — `swmm_resilience/visualization/runtime_caption.py`

```python
def format_runtime_text(seconds: float | None) -> str | None:
    """Texto del tiempo de cómputo para la esquina del mapa.

    - seconds None -> None (no se dibuja anotación).
    - Formato: 'Tiempo de cómputo: {s} s', con {s} a '.2f' si seconds >= 1, o a
      '.4f' si es menor (tiempos de ML en milésimas de segundo).
    """
```

Función pura, sin matplotlib, testeable de forma aislada. Sin speed-up.

### `plot_flood_map` (flood_map.py)

Añadir un parámetro opcional `runtime_text: str | None = None`. Si viene, se
dibuja como anotación en la esquina inferior derecha de los ejes, antes de
`savefig`:

```python
if runtime_text:
    ax.text(
        0.98, 0.02, runtime_text,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, bbox=ANNOTATION_BBOX, zorder=6,
    )
```

Con `None` (por defecto) el mapa queda idéntico al actual.

### `plot_scenario_flood_maps` (model_comparison.py)

Añadir dos parámetros opcionales y pasar el texto a cada `plot_flood_map`:

```python
def plot_scenario_flood_maps(
    comp_df, inp_path, output_dir, scenario_id,
    t_swmm_s: float | None = None,
    t_ml_s: float | None = None,
) -> tuple[Path, Path]:
    ...
    # plot_flood_map(..., title="...\nSWMM Simulation",
    #                runtime_text=format_runtime_text(t_swmm_s))
    # plot_flood_map(..., title="...\nML Prediction",
    #                runtime_text=format_runtime_text(t_ml_s))
```

Con valores por defecto `None`, el comportamiento sin tiempos es idéntico al
actual.

### `run_batch_validation` (hydrograph_batch.py)

En la llamada existente (paso g del bucle), pasar los tiempos ya medidos:

```python
t_ml = pred_timings["t_features_s"] + pred_timings["t_inference_s"]
plot_scenario_flood_maps(
    comp_df, scenario_inp_path, out_dir, scenario.scenario_id,
    t_swmm_s=t_swmm, t_ml_s=t_ml,
)
```

`t_swmm` ya existe en el bucle; `t_ml` se calcula igual que para `timing_rows`.

## Casos borde

- `seconds None` (llamada sin pasar tiempos) -> sin anotación; mapas como hoy.

## Pruebas

- `tests/visualization/test_runtime_caption.py`:
  - `format_runtime_text` con seconds >= 1 (formato `.2f`), seconds < 1 (`.4f`),
    y `seconds=None` (devuelve None).
- `plot_scenario_flood_maps`: test que capture el kwarg `runtime_text` pasado a
  `plot_flood_map` (monkeypatch) y verifique que, al pasar `t_swmm_s`/`t_ml_s`,
  cada mapa recibe su texto de tiempo; y que sin pasarlos `runtime_text` es None
  (no romper `test_plot_scenario_flood_maps_uses_shared_scale_and_root_output`).
- `plot_flood_map`: smoke test de que con `runtime_text` el PNG se genera y los
  ejes contienen un texto con la cadena del tiempo.

## Restricciones operativas

- Trabajo en branch nueva (independiente), sin borrar archivos/outputs/branches.
