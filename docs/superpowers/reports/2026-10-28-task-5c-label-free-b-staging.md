# Task 5c — staging B label-free con procedencia

**Estado:** implementación sintética terminada; no habilita scoring real ni incorpora etiquetas B.

## Alcance y decisiones aplicadas

- Las corridas nuevas usan un flujo separado del importador histórico 5b y requieren una base SQLite fresca y migrada.
- `protocol.json` se congela antes de simular. Contiene hash/ruta de la red y claves explícitas en orden estable, más forma, factor, duración y tiempo al pico.
- `frozen_b_manifest.json` se genera al terminar el lote y se liga al hash exacto del protocolo. Cada clave explícita se liga a cinco fuentes: INP de red, INP efectivo, serie de evento, RPT y OUT.
- QA se reconstruye desde campos explícitos y política persistida. Runs `COMPLETE` no elegibles se guardan y reportan con razones, sin features, keys ni predicciones.
- Las features de ambas versiones soportadas usan los extractores compartidos de training/inferencia; no aceptan targets. La cobertura de junctions, tiempos del evento y hashes se verifican. Se compara también la configuración de escenario persistida con el manifiesto sellado antes de inferencia.
- La interfaz previa al predictor recibe el esquema exacto `KEY_COLUMNS`, features ordenadas y QA observado reconstruido; `node_results` queda vacío.
- El wrapper conecta stage → `register_transfer` → reverificación → `predict_transfer`; no modifica `score_transfer`, no lee etiquetas del RPT y no escribe métricas reales.

## Límites de procedencia

Los hashes detectan cambios contra el manifiesto, pero no prueban autoría ni son firmas independientes. En particular, Task 5c no ejecuta un parser nuevo sobre el RPT para regenerar cada medida QA: requiere campos QA caller-supplied y valida consistencia/hash del reporte usando la política existente. Para las corridas reales será necesario emitir esos campos desde el parser/calificador aprobado y revisar el resultado; el staging por sí solo no demuestra que sean físicamente correctos ni justifica los umbrales hidráulicos provisionales. No se ejecutaron simulaciones ni se importaron/modificaron bases/output canónicos.

## TDD y verificación

**RED:** antes de crear el módulo, las pruebas sintéticas fallaron por ausencia de `label_free_b_staging` (3 fallos esperados). Durante la revisión de la integración se añadió un caso adversarial adicional: alterar en SQLite `factor_mult` luego del staging debe impedir inferencia. El primer intento de regresión dentro del sandbox tampoco pudo crear/leer temporales pytest preexistentes por `PermissionError`; se volvió a ejecutar con acceso normal a temporales, sin cambiar outputs del proyecto.

**GREEN:** suite de staging, registro de artefactos, importador histórico y predictor congelado pasó con `60 passed`. Incluye orden estable de IDs aunque las filas del manifiesto de lote estén reordenadas; campos target prohibidos; hash modificado; cobertura de nodos incompleta; tiempo/duración discordantes; corrida QA no elegible excluida y reportada; escenario SQLite alterado; artefactos y QA/features disponibles antes de predicción; `node_results` vacío; esquema exacto de keys y wrapper de registro/inferencia.

Comando de regresión:

```powershell
venv\Scripts\python.exe -m pytest tests/database/test_label_free_b_staging.py tests/database/test_run_source_artifacts.py tests/database/test_source_manifest_importer.py tests/ml/bench/test_frozen_transfer_experiment.py -q --basetemp=<directorio-temporal-nuevo>
```

La pasada completa final, compilación y `git diff --check` deben confirmarse antes del commit. Linter Ruff no está instalado en el entorno virtual. No hubo revisión independiente de subagente en este turno; se efectuó revisión local enfocada en la frontera de predicción, escenarios mutables, hashes, cobertura y flujo stage→register→predict.

## Próximo paso

La siguiente fase autorizada debe generar los manifiestos completos y corridas nuevas A/B en infraestructura/datos explícitamente elegidos, empezando por el pipeline de simulación y QA que produzca los cinco archivos por corrida. B solo se puede predecir tras congelar y verificar esos artefactos; Task 5d debe conservar la barrera de etiquetas hasta el sello de predicciones.
