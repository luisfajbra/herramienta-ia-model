# Tabla-imagen de volumen total de inundación (SWMM vs XGBoost)

Fecha: 2026-06-20
Branch: `feature/tabla-volumen-total`

## Objetivo

Reportar, como una imagen de tabla sencilla, la sumatoria del volumen de
inundación en toda la red para cada escenario, comparando SWMM y XGBoost, con
una columna de diferencia porcentual entre ambos. La tabla se genera en dos
contextos: validación de hidrogramas (`--evaluate-hydrographs`) y comparación
por factor (`--factor-comparison`).

## Columnas de la tabla

| Encabezado | Contenido |
|---|---|
| `Escenario` / `Factor` | Identificador del escenario (hidrograma) o factor de caudal |
| `Volumen SWMM (m³)` | Suma de `vol_swmm_m3` sobre todos los nodos del escenario |
| `Volumen XGBoost (m³)` | Suma de `vol_pred_m3` sobre todos los nodos del escenario |
| `Diferencia (%)` | `(XGBoost − SWMM) / SWMM × 100`, con signo |

Fila final **TOTAL**: suma de SWMM, suma de XGBoost y `Diferencia (%)` global
(calculada sobre las sumas totales, no como promedio de los porcentajes).

### Convención de la diferencia porcentual

`(XGBoost − SWMM) / SWMM × 100`. Positivo = XGBoost sobreestima respecto a
SWMM; negativo = subestima. Coincide con `error_pct` ya existente en
`hydrograph_batch.py`.

## Arquitectura

Módulo nuevo `swmm_resilience/visualization/summary_table.py` con dos funciones
de responsabilidad única (cálculo separado del renderizado):

```python
def build_total_volume_table(rows: list[dict]) -> pd.DataFrame:
    """Función pura.

    Entrada: lista de dicts con claves "label", "vol_swmm_m3", "vol_pred_m3".
    Salida: DataFrame con columnas [label, vol_swmm_m3, vol_pred_m3, diff_pct]
    más una fila final cuyo label es "TOTAL".

    diff_pct = (vol_pred_m3 - vol_swmm_m3) / vol_swmm_m3 * 100.
    diff_pct = None cuando vol_swmm_m3 == 0 (evita división por cero).
    La fila TOTAL suma vol_swmm_m3 y vol_pred_m3 y recalcula diff_pct sobre
    esas sumas (None si la suma SWMM == 0).
    """

def render_total_volume_table(
    table_df: pd.DataFrame,
    output_path: Path,
    label_header: str,
    title: str | None = None,
) -> Path:
    """Renderiza table_df como imagen PNG con matplotlib y la guarda en
    output_path. Devuelve output_path."""
```

**Separación clave:** `build_total_volume_table` es pura y testeable sin dibujar;
`render_total_volume_table` solo dibuja. Sin dependencias nuevas (solo
matplotlib, ya en uso).

## Cableado en los comandos

### `--evaluate-hydrographs` (`swmm_resilience/validation/hydrograph_batch.py`)

Tras el bucle de escenarios ya existe la lista `per_scenario` con
`vol_total_swmm_m3` y `vol_total_pred_m3` por escenario. Construir las `rows`
desde esa lista (label = `scenario_id`), llamar a `build_total_volume_table` y
`render_total_volume_table` con `label_header="Escenario"`, guardando en
`<out_dir>/scenario_totals_table.png` (junto al `scenario_totals.csv` existente).

### `--factor-comparison` (`swmm_resilience/analysis/factor_comparison.py`)

Dentro del bucle por factor ya se dispone del DataFrame `comparison`. Acumular
`comparison["vol_swmm_m3"].sum()` y `comparison["vol_pred_m3"].sum()` por factor.
Al final del bucle, construir las `rows` (label = factor formateado, p. ej.
`"3.00"`), renderizar a
`outputs/metrics/factor_comparison/total_volume_by_factor_table.png` con
`label_header="Factor"`. El path de la imagen se añade a la lista `paths` que
devuelve `generate_factor_comparisons`.

## Formato de la imagen

- Números con separador de miles y 1 decimal (`1,266.0`).
- `Diferencia (%)` con signo y 1 decimal (`+77.9%`); `—` cuando es None.
- Fila TOTAL en negrita con sombreado gris claro.
- Encabezados con fondo de color; celdas numéricas alineadas a la derecha.
- `dpi=150`, fondo blanco, ejes ocultos (`ax.axis("off")`).

## Casos borde

- `vol_swmm_m3 == 0` en un escenario → `diff_pct = None` → se muestra `—`.
- `rows` vacío (ningún escenario/factor) → no se genera imagen; la función de
  cableado retorna sin error y sin añadir path.

## Pruebas

- `tests/visualization/test_summary_table.py`:
  - `build_total_volume_table`: verifica diff_pct (valor y signo), fila TOTAL
    (sumas y % global), y SWMM=0 → diff_pct None (fila y total).
  - `render_total_volume_table`: smoke test — genera el PNG y comprueba que el
    archivo existe y pesa > 0 bytes.

## Fuera de alcance (YAGNI)

- No se añade CSV nuevo para factor-comparison (el usuario pidió solo la imagen).
- No se usan librerías externas de tablas (plotly/dataframe_image).
- No se modifica la lógica de cálculo de volúmenes existente.

## Restricciones operativas

- Todo el trabajo en la branch `feature/tabla-volumen-total`.
- No borrar archivos, outputs ni branches existentes.
