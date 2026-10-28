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

## Hallazgos de revisión independiente corregidos

1. **QA caller-supplied no era evidencia.** El staging ahora deriva continuidad de RPT, QA de inundación/procedencia con los parsers compartidos, finite coverage con RPT/OUT/INP, estado de drenaje con la ventana final OUT y las condiciones iniciales con INP fuente y efectivo. Rechaza declaraciones del batch que discrepen. Solo persiste/pasa al predictor esos valores derivados; los valores de inundación que construye `extract_labels` se descartan y nunca llegan a la interfaz de inferencia ni a `node_results`.
2. **Un SHA final era recalculable junto con el protocolo.** `freeze_b_transfer_protocol` escribe el protocolo en modo exclusivo y añade un receipt en una SQLite externa con triggers append-only, hash encadenado, commit Git y contexto de ejecución. Batch y stage deben verificar el receipt y los bytes anclados; la prueba modifica el protocolo y recalcula el SHA del batch y confirma rechazo antes de crear filas.

La ancla separada mejora integridad operativa, no proporciona firma ni identidad autenticada. Un operador con capacidad para reemplazar/recrear también la SQLite externa puede reconstruir una cadena falsa; evitar ese riesgo requiere servicio/medio independiente con escritura controlada. Los umbrales hidráulicos continúan siendo decisiones provisionales documentadas. El INP fuente puede permanecer en `data/networks` como enlace absoluto con hash; no se copia. INP efectivo, CSV de evento, RPT y OUT se mantienen bajo la raíz del experimento.

El importador histórico Task 5b aún no consume la nueva ancla ni el esquema de protocolo. La API de freeze es reutilizable, pero un runner/adapter deberá verificar el mismo receipt al generar/importar A; el importador actual no se considera admisión de A regenerada. No se ejecutaron simulaciones ni se importaron/modificaron bases/output canónicos.

## TDD y verificación

**RED:** las pruebas sintéticas iniciales fallaron por ausencia de `label_free_b_staging`. Después de la revisión independiente, un nuevo test falló porque el staging aceptaba una continuidad declarada distinta del RPT; los dos tests del freeze/anchor fallaron por falta de API. Esos fallos precedieron a las correcciones. El primer intento de regresión dentro del sandbox tampoco pudo crear/leer temporales pytest preexistentes por `PermissionError`; se volvió a ejecutar con acceso normal a temporales.

**GREEN:** los tests sintéticos focalizados de ancla/staging pasaron con `16 passed`. Incluyen triggers no-update/no-delete; fuente INP externa enlazada; QA caller alterada en continuidad/cobertura/drenaje/initial-state rechazada; protocolo y hash del batch reescritos sin receipt rechazados; orden estable de IDs aunque batch llegue reordenado; hash de artefactos alterado; cobertura/tiempos incorrectos; QA no elegible excluida; escenario SQLite alterado; esquema exacto `KEY_COLUMNS`; features/QA disponibles antes de predecir; sin volumen en la salida y `node_results` vacío.

Comando de regresión:

```powershell
venv\Scripts\python.exe -m pytest tests/database/test_b_transfer_protocol_anchor.py tests/database/test_label_free_b_staging.py tests/database/test_run_source_artifacts.py tests/database/test_source_manifest_importer.py tests/ml/bench/test_frozen_transfer_experiment.py -q --basetemp=<directorio-temporal-nuevo>
```

La regresión completa (ancla, staging, fuente SQLite, importador histórico y predictor congelado) pasó: **66 passed, 13 warnings**. `compileall` y `git diff --check` también pasaron. Ruff no está instalado en el entorno virtual. La revisión independiente solicitó estos dos bloqueos P1 y ambos quedan cubiertos por tests sintéticos; se revisó localmente también el flujo stage→register→predict y el sello de metadatos.

## Próximo paso

La siguiente fase autorizada debe generar los manifiestos completos y corridas nuevas A/B en infraestructura/datos explícitamente elegidos, empezando por el pipeline de simulación y QA que produzca los cinco archivos por corrida. B solo se puede predecir tras congelar y verificar esos artefactos; Task 5d debe conservar la barrera de etiquetas hasta el sello de predicciones.
