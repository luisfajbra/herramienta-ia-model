# Design: Hydrograph Visualization & Network Map

**Date:** 2026-06-02
**Features:** `--hydrograph` and `--network-map` CLI flags
**Status:** Design approved, pending implementation plan

---

## 1. Hydrograph Visualization (`--hydrograph`)

### Purpose

Show the shape of the design storm hydrograph used as input to the network. Useful for understanding how aggressive each rainfall scenario is before running predictions or comparisons.

### Behavior

- Reads all node inflow profiles from the `.inp` using `get_node_inflow_profiles` (already available in `swmm_resilience/simulation/swmm_api_io.py`).
- Auto-selects the node with the highest peak flow value (max value across all `points` tuples) as the representative node.
- Plots time (minutes) on X axis vs flow (L/s) on Y axis.
- Title includes the selected node ID: e.g. `Hidrograma de entrada — Nodo 30C (Qx1)`.
- Output saved to `outputs/maps/hydrograph_Qx1.png`.

### New file

`swmm_resilience/visualization/hydrograph.py`

```
plot_hydrograph(inp_path: Path, output_path: Path) -> Path
```

- Reads profiles, finds peak node, plots with matplotlib, saves PNG, returns `output_path`.

### CLI

```bash
python main.py --hydrograph
```

Added as standalone mode in `main.py` (same pattern as `--only-maps`). Does not require a trained model or dataset.

### Test

`tests/test_hydrograph.py` — monkeypatches `get_node_inflow_profiles` with a fake two-node profile, calls `plot_hydrograph`, asserts output file exists and is non-empty.

---

## 2. Network Map (`--network-map`)

### Purpose

Reference visualization of the drainage network topology: pipe flow directions, pipe type classification, and outfall locations. Independent of flood results.

### Pipe classification

```
nodos_destino = { c.to_node for c in CONDUITS }

tubería INICIAL   → from_node NOT in nodos_destino  (cabecera de cuenca, azul)
tubería CONTINUA  → from_node IN  nodos_destino      (recibe flujo de otra tubería, gris)
```

### Node classification

- **Nodo normal:** aparece en `JUNCTIONS`. Dibujado como círculo pequeño gris.
- **Nodo de salida (outfall):** aparece en sección `OUTFALLS` del `.inp`. Dibujado como triángulo negro apuntando hacia abajo.

### Visual spec

| Elemento | Estilo |
|---|---|
| Tubería inicial | Línea azul, grosor 1.5 |
| Tubería continua | Línea gris, grosor 1.0 |
| Flecha de flujo | En el punto medio de cada tubería, misma dirección que el conduit (`from_node` → `to_node`) |
| Nodo normal | Círculo gris, radio pequeño (3 px) |
| Nodo de salida | Triángulo negro apuntando hacia abajo |
| ID de nodo | Texto pequeño (fontsize 5), centrado sobre cada nodo |
| Leyenda | Tubería inicial / Tubería continua / Nodo de salida |
| Fondo | Blanco |
| Título | Nombre de la red (`config.network.name`) |

### Arrow implementation

`matplotlib.patches.FancyArrowPatch` posicionada en el punto medio del segmento `(from_node → to_node)`, con `arrowstyle="->"` y `mutation_scale=8`. Color igual al de la tubería.

### New file

`swmm_resilience/visualization/network_map.py`

```
generate_network_map(inp_path: Path, output_path: Path, network_name: str) -> Path
```

- Carga `inp` con `load_inp`.
- Construye `nodos_destino` y clasifica tuberías.
- Identifica outfalls desde sección `OUTFALLS`.
- Dibuja con matplotlib, guarda PNG, retorna `output_path`.

### CLI

```bash
python main.py --network-map
```

Modo standalone. No requiere dataset ni modelos. Guarda en `outputs/maps/network_map.png`.

### Test

`tests/test_network_map.py` — red sintética de 4 nodos (2 cabecera, 1 intermedio, 1 outfall), monkeypatcha `load_inp`, llama `generate_network_map`, verifica que el PNG existe y es no vacío.

---

## File map

| Archivo | Acción |
|---|---|
| `swmm_resilience/visualization/hydrograph.py` | Crear |
| `swmm_resilience/visualization/network_map.py` | Crear |
| `tests/test_hydrograph.py` | Crear |
| `tests/test_network_map.py` | Crear |
| `main.py` | Modificar — agregar `--hydrograph` y `--network-map` |
