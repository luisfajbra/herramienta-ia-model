# Task 5c — staging B label-free con procedencia

**Estado:** implementación sintética terminada; no habilita scoring real ni incorpora etiquetas B.

## Alcance y decisiones aplicadas

- Las corridas nuevas usan un flujo separado del importador histórico 5b y requieren una base SQLite fresca y migrada.
- `protocol.json` se congela antes de simular. Contiene hash/ruta de la red y claves explícitas en orden estable, más forma, factor, duración y tiempo al pico.
- `frozen_b_manifest.json` se genera al terminar el lote y se liga al hash exacto del protocolo. Cada clave explícita se liga a cinco fuentes: INP de red, INP efectivo, serie de evento, RPT y OUT.
- QA se reconstruye desde campos explícitos y política persistida. Runs `COMPLETE` no elegibles se guardan y reportan con razones, sin features, keys ni predicciones.
- Las features de ambas versiones soportadas usan los extractores compartidos de training/inferencia; no aceptan targets. La cobertura de junctions, tiempos del evento y hashes se verifican. Se compara también la configuración de escenario persistida con el manifiesto sellado antes de inferencia.
- La interfaz previa al predictor recibe el esquema exacto `KEY_COLUMNS`, features ordenadas y QA observado reconstruido; `node_results` queda vacío.
- El wrapper conecta stage → `register_transfer` → reverificación → `predict_transfer`; no modifica `score_transfer` ni escribe métricas reales. El staging valida transitoriamente el resumen de inundación del RPT, descarta los valores y no construye/entrega un vector objetivo.

## Hallazgos de revisión independiente corregidos

1. **QA caller-supplied no era evidencia.** El staging deriva continuidad de RPT, valida el resumen de inundación/procedencia con el parser compartido, exige cobertura OUT exacta y finita para todo nodo y enlace hidráulico declarado en INP (junctions, outfalls, storage, dividers; conduits, pumps, orifices, weirs, outlets), calcula estado de drenaje en la ventana final OUT y verifica condiciones iniciales desde INP fuente/efectivo. Rechaza declaraciones del batch que discrepen. Además compara el descriptor canónico de todas las secciones parseadas fuente/efectivo, excluyendo solo `INFLOWS`, `TIMESERIES` y `OPTIONS.END_DATE`/`OPTIONS.END_TIME`; detecta cambios físicos como roughness de un conduit no dominante que no altera máximos estáticos agregados. También compara todas las columnas de `extract_static_features`, salvo `base_inflow_lps`, con NaN-parity explícita y `rtol=atol=1e-9`. No construye un vector objetivo: solo verifica que los IDs del resumen sean conocidos; el cuadro de volúmenes parseado se descarta y nunca se persiste ni se entrega a inferencia o `node_results`.
2. **Un SHA final era recalculable junto con el protocolo.** `freeze_b_transfer_protocol` escribe el protocolo en modo exclusivo y añade un receipt en una SQLite externa con triggers append-only, hash encadenado, commit Git y contexto de ejecución. Batch y stage deben verificar el receipt y los bytes anclados; la prueba modifica el protocolo y recalcula el SHA del batch y confirma rechazo antes de crear filas.

La ancla separada mejora integridad operativa, no proporciona firma ni identidad autenticada. Un operador con capacidad para reemplazar/recrear también la SQLite externa puede reconstruir una cadena falsa; evitar ese riesgo requiere servicio/medio independiente con escritura controlada. Los umbrales hidráulicos continúan siendo decisiones provisionales documentadas. El INP fuente puede permanecer en `data/networks` como enlace absoluto con hash; no se copia. INP efectivo, CSV de evento, RPT y OUT se mantienen bajo la raíz del experimento.

La nueva comparación de descriptor es semántica según `swmm-api`, no un diff de bytes. Comentarios/formato normalizados por el parser y campos no expuestos por esa versión no pueden verificarse aquí; el `execution_context` debe registrar la versión instalada y cambios de parser requieren revisión del contrato.

Las escrituras del protocolo/receipt y de SQLite/manifiesto de staging no son transacciones atómicas conjuntas. Una caída puede dejar un protocolo sin ancla, un receipt cuyo retorno se perdió, un manifiesto huérfano o una DB ya poblada sin frame devuelto. Todos fallan cerrados ante staging/reintento; no existe API de recuperación. Se conserva la evidencia y se reanuda en una raíz y DB nuevas después de revisar el caso.

El importador histórico Task 5b aún no consume la nueva ancla ni el esquema de protocolo. La API de freeze es reutilizable, pero un runner/adapter deberá verificar el mismo receipt al generar/importar A; el importador actual no se considera admisión de A regenerada. No se ejecutaron simulaciones ni se importaron/modificaron bases/output canónicos.

## TDD y verificación

**RED:** las pruebas sintéticas iniciales fallaron por ausencia de `label_free_b_staging`. Después de la revisión independiente, un nuevo test falló porque el staging aceptaba una continuidad declarada distinta del RPT; los dos tests del freeze/anchor fallaron por falta de API. Esos fallos precedieron a las correcciones. El primer intento de regresión dentro del sandbox tampoco pudo crear/leer temporales pytest preexistentes por `PermissionError`; se volvió a ejecutar con acceso normal a temporales.

**GREEN:** las pruebas sintéticas incluyen triggers no-update/no-delete; fuente INP externa enlazada; QA caller alterada en continuidad/cobertura/drenaje/initial-state rechazada; protocolo y hash del batch reescritos sin receipt rechazados; orden estable de IDs aunque batch llegue reordenado; QA no elegible excluida; esquema exacto `KEY_COLUMNS`; features/QA disponibles antes de predecir; `node_results` vacío; diferencias de identidad, elevación, diámetro, roughness no dominante y endpoint del INP efectivo rechazadas; descriptor permite solo los campos event/timing autorizados; OUT con un enlace esperado omitido excluido con `incomplete_link_coverage`. Las series OUT completas cubren exactamente todos los nodos hidráulicos y los cinco tipos de enlace reconocidos. QA lee y valida el resumen de inundación transitoriamente, sin llamar a `extract_labels`, crear un vector objetivo ni pasar volúmenes a inferencia.

Comando de regresión:

```powershell
venv\Scripts\python.exe -m pytest tests/database/test_b_transfer_protocol_anchor.py tests/database/test_label_free_b_staging.py tests/database/test_run_source_artifacts.py tests/database/test_source_manifest_importer.py tests/ml/bench/test_frozen_transfer_experiment.py -q --basetemp=<directorio-temporal-nuevo>
```

La regresión completa (ancla, staging, fuente SQLite, importador histórico y predictor congelado) pasó: **73 passed, 13 warnings**. `compileall` y `git diff --check` también pasaron. Ruff no está instalado en el entorno virtual. Las revisiones identificaron y esta regresión cubre la cobertura de enlaces, paridad de features y descriptor hidráulico completo; también se revisó localmente el flujo stage→register→predict y el sello de metadatos.

## Próximo paso

La siguiente fase autorizada debe generar los manifiestos completos y corridas nuevas A/B en infraestructura/datos explícitamente elegidos, empezando por el pipeline de simulación y QA que produzca los cinco archivos por corrida. B solo se puede predecir tras congelar y verificar esos artefactos; Task 5d debe conservar la barrera de etiquetas hasta el sello de predicciones.
