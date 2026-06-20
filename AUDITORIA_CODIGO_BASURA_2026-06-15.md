# Auditoría de código y artefactos inútiles — 2026-06-15

Auditoría conservadora de código/archivos basura en el repositorio `herramienta`.

## Criterio aplicado

- **Solo se marca para borrar** lo que se pudo **verificar como huérfano** (sin imports, sin referencias en tests, ni en docs/configs) o como **artefacto regenerable duplicado**.
- Si algo es ambiguo, podría ser un fixture, o tocarlo puede romper el pipeline, **NO se incluye en la lista de borrado** y se mueve a "Revisar manualmente".
- **El retiro de Pipeline A queda explícitamente FUERA de esta auditoría.** Es un refactor deliberado (decidido el 2026-06-15), no código basura: todos sus módulos están cableados a la GUI de escritorio, a la inferencia (`predict_from_inp.py`) y a tests. Eliminarlo debe hacerse como tarea aparte y controlada.

## Método

- Listado de entradas top-level y de los 90+ módulos `.py`.
- Análisis de alcanzabilidad (grafo de imports con AST) sobre todo el repo, incluyendo entrypoints (`main.py`, `desktop/app.py`), `conftest.py` y `tests/`.
- `grep` de referencias de cada candidato en archivos `.py` **y** no-Python (docs, `pytest.ini`, `comandos.txt`).
- Verificación de estado en git (tracked/untracked) y de reglas `.gitignore`.

**Resultado del análisis de alcanzabilidad:** ningún módulo del paquete `swmm_resilience` quedó huérfano. No hay módulos del paquete para borrar.

---

## ✅ Confirmado seguro de borrar (alta confianza)

| # | Ruta | Tipo | Evidencia | Git |
|---|------|------|-----------|-----|
| 1 | `pruebas_locales.py` | Script scratch | Bucle manual de `print` para inspeccionar una simulación. Cero referencias en todo el repo. | tracked → `git rm` |
| 2 | `verificar.py` | Script scratch | Verificación manual one-off (`python verificar.py <factor>`). Cero referencias (las coincidencias de "verificar" en docs son la palabra en español, no el archivo). | tracked → `git rm` |
| 3 | `halve_timeseries.py` | Script one-off **riesgoso** | Munge de datos que **sobrescribe el `.inp` de producción** (`DST = chico_hydro-qx1/...Qx1.00.inp`). Cero referencias. Tenerlo suelto es un riesgo de sobrescritura accidental. | tracked → `git rm` |
| 4 | `outputs/factor_comparisons/` | Carpeta de salida obsoleta | Duplicado viejo (50 archivos, 2026-06-09 08:59), **superado** por `outputs/metrics/factor_comparison/` (70 archivos, 09:19). Ningún código la referencia; la ruta actual la fija `main.py:148`. | untracked (ignorada) |
| 5 | `__pycache__/`, `.pytest_cache/` | Cachés de build | Regenerables, ya en `.gitignore`. Limpieza trivial. | untracked (ignoradas) |

### Comandos sugeridos (NO ejecutados)

```bash
# Scripts scratch (tracked):
git rm pruebas_locales.py verificar.py halve_timeseries.py

# Carpeta de salida obsoleta (local):
rm -rf outputs/factor_comparisons

# Cachés (opcional, regenerables):
find . -type d -name __pycache__ -exec rm -rf {} +
rm -rf .pytest_cache
```

---

## ⚠️ Revisar manualmente (NO borrar automáticamente)

| Ruta / tema | Por qué no se incluye en el borrado |
|-------------|-------------------------------------|
| **Pipeline A** (`ml/train.py`, `ml/predict_from_inp.py`, `ml/predict_tabular.py`, `ml/scenario_predict.py`, artefactos `model_artifacts/`, EDA `analysis/eda.py`) | Refactor deliberado, no basura. Cableado a la GUI de escritorio, a la inferencia y a tests. Retirarlo es una tarea controlada aparte. |
| `validation_output_smoke/` | Salida de un smoke test. Los PNG/CSV son regenerables (untracked, seguros), pero `inp/hydrograph_2h.inp` e `inp/hydrograph_5h.inp` **están trackeados** y podrían ser fixtures de un flujo de validación. Verificar antes de tocar. |
| `outputs/new_csv_comparison/`, `outputs/maps/`, `outputs/metrics/` (contenido) | Artefactos locales regenerables (ignorados por git). No son "basura de código"; el usuario tiene `metrics_per_scenario.csv` abierto. Limpieza a discreción del usuario. |
| `comandos.txt` | **NO borrar.** Es una chuleta útil de comandos del CLI, sirve como documentación. |
| Archivos `.md` de raíz y `docs/` | Documentación/specs/planes. Fuera de alcance de esta auditoría. |
| Código muerto intra-archivo | No se auditó a nivel de funciones/ramas sueltas para evitar falsos positivos. Requiere revisión dedicada y con tests. |

---

## Resumen

- **3 scripts scratch** + **1 carpeta de salida obsoleta** + **cachés** = lo único confirmado como seguro de borrar.
- **0 módulos del paquete** son huérfanos.
- **Pipeline A** se trata por separado (retiro deliberado, no incluido aquí).
- **Nada fue borrado** al generar este reporte; es solo el inventario para tu aprobación.
