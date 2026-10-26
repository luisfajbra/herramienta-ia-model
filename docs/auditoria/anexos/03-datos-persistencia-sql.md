# Anexo 03 — Datos y persistencia SQL

**Auditoría:** proyecto de tesis `herramienta-ia-model` (modelo sustituto ML de inundación nodal sobre SWMM)
**Rama / commit base:** `fix/cierre-auditoria-hidrogramas-20260927` @ `6e8579b` (con cambios sin commitear ajenos a este anexo, que no se tocaron)
**Fecha:** 2026-10-10
**Área:** datos, esquema SQLite v17, migraciones, proveniencia, backfill CSV→SQL, bloqueo de flujo, reset y calidad de los datos reales.

---

## 1. Alcance y método

### 1.1 Código revisado línea a línea

| Módulo | Líneas | Rol |
|---|---|---|
| `swmm_resilience/database/sql/001…005_*.sql` | 2.107 | Esquema v17 real (tablas, CHECK, FK, triggers, vistas) |
| `swmm_resilience/database/migrations.py` | 226 | Motor de migraciones con checksum |
| `swmm_resilience/database/connection.py` | 36 | Política de conexión (FK, WAL, `sha256()`) |
| `swmm_resilience/database/training_queries.py` | 387 | Lector canónico del frame de entrenamiento |
| `swmm_resilience/database/csv_backfill.py` | 451 | Escritura CSV→SQL y cadena de evidencia de entrenamiento |
| `swmm_resilience/database/upgrade.py`, `maintenance.py`, `recovery.py`, `workflow_lock.py`, `migration_005_validator.py`, `queries.py` | 371 | Upgrade con respaldo, recuperación y bloqueo |
| `swmm_resilience/database/schema.py`, `repository.py` | 567 | Esquema *legacy* (desktop), que **convive** con el v17 |
| `swmm_resilience/reset.py` | 245 | Borrado de datos |
| `swmm_resilience/dataset/*.py`, `swmm_resilience/analysis/dataset.py` | 179 | Ensamblado y validación del dataset y export legacy |
| `swmm_resilience/extraction/labels.py` | 140 | Regla de etiqueta (leída por relevancia directa para los datos) |
| `main.py` (líneas 865-916, 1050-1080) | — | Llamadores de `--persist-sql` y `load_training_frame` |
| `tests/database/*.py` | 6.338 | Cobertura real (27 archivos) |

No existen migraciones Alembic: el proyecto usa su propio catálogo `sql/NNN_*.sql`.

Especificaciones contrastadas: `docs/superpowers/specs/2026-08-21-sqlite-v17-pipeline-consolidation-design.md`, `2026-08-22-sqlite-v17-provenance-hardening-design.md` y `2026-09-03-sqlite-v17-training-frame-cutover-design.md`. La de 08-21 quedó parcialmente reemplazada por la de 09-03, que define un camino mínimo. Las desviaciones de 08-21 se reportan solo cuando la 09-03 no las declara fuera de alcance.

### 1.2 Diagnósticos ejecutados (todos de solo lectura sobre los datos reales)

- Las bases reales `outputs/training_v17.sqlite3` (55,3 MB) y `outputs/training_v17_19shapes.sqlite3` (27,3 MB) se abrieron **siempre** con `file:…?mode=ro&immutable=1`. `immutable=1` evita tocar los ficheros `-wal`/`-shm`. El `-wal` real mide 0 bytes, así que el fichero principal está completo.
- Antes y después de la auditoría, el SHA-256 de ambos ficheros siguió siendo el mismo (`d6b8ae75afb04dba…` y `bfc3af8c83e4e694…`). Las fechas de modificación tampoco cambiaron (2026-09-15 09:49 y 2026-09-06 20:18). El `-shm` de `training_v17` cambió de fecha durante la sesión (16:53 y 17:03), pero no por estos scripts: otros agentes que comparten el scratchpad abrieron la base en paralelo.
- Los experimentos de comportamiento (triggers, reset, idempotencia, atomicidad) se ejecutaron sobre **bases sintéticas o copias** creadas en el scratchpad. Las copias se hicieron con la API `backup` desde una conexión `immutable`.
- `pytest tests/database -p no:cacheprovider --basetemp=<scratchpad>` dio **208 passed, 3 deselected (scale), 28 s**. No se ejecutó la suite completa ni las pruebas `-m scale`.
- No se modificó código, `config.yaml`, `outputs/` ni git. `git status` es idéntico al inicial.

Python 3.12.10, SQLite 3.49.1 (venv del proyecto). Los scripts están en la sección 5.

---

## 2. Hallazgos

Resumen por severidad: **🔴 0 · 🟠 4 · 🟡 9 · 🔵 8** (21 hallazgos).

| ID | Sev. | Archivo:línea | Descripción | Impacto cuantificado | Evidencia reproducible | Recomendación |
|---|---|---|---|---|---|---|
| **D-01** | 🟠 MAYOR | `sql/005_provenance_integrity.sql:991-998` | El *pinning* de proveniencia de `runs` solo bloquea `UPDATE` de `status` y `network_id`. Se pueden reescribir `scenario_id`, `config_sha256`, `continuity_error_pct`, `node_count` y timestamps de un run usado por un `training_run`. `scenario_inflows` no está fijado en absoluto. La spec 08-22 §3.2.1 exige que "the selected `runs` … rows cannot be updated or deleted". | Permutar `scenario_id` entre dos runs fijados es aceptado por los triggers y por `load_training_frame()`. Cada (forma, factor, nodo) queda reetiquetada en silencio con el volumen de otro factor, y la feature `q_pico_nodo`, guardada en `node_features`, deja de corresponder al `factor_mult` del escenario. En la base sintética de prueba cambiaron 6 de 96 volúmenes sin ningún error. `_check_single_snapshot` solo detecta duplicados, no permutaciones. | Script D05 (`d05_swap.py`) y D04-E1. Resultado: `runs.scenario_id … UPDATE ACEPTADO`; `scenario_inflows … INSERT/UPDATE/DELETE ACEPTADOS`; "filas cuyo vol cambió para la misma clave: 6". | Cambiar el trigger a `BEFORE UPDATE ON runs WHEN EXISTS(pinned)` sin condición de columna (o con `NEW.* IS NOT OLD.*` para todas). Añadir triggers de *pinning* a `scenario_inflows`. En lectura, verificar `runs.config_sha256 = scenarios.config_sha256` (hoy coincide en 1000/1000, pero nadie lo comprueba). Añadir test. |
| **D-02** | 🟠 MAYOR | `csv_backfill.py:147-156`, `sql/001_v17_initial.sql:79-97,132-153` | Los runs se insertan directamente como `COMPLETE` a partir del CSV, sin evidencia de la simulación SWMM. `swmm_version` y `continuity_error_pct` quedan NULL. `started_at_utc` y `completed_at_utc` son dos llamadas a `_now()` en el momento del backfill. `node_results.peak_flooding_lps`, `flooding_duration_min`, `max_depth_m` y `max_depth_ratio` quedan NULL. `scenario_inflows` (el hidrograma aplicado) queda vacía y `links` también. La spec 08-21 §7.2/§11 exigía versión SWMM, error de continuidad y verificación de continuidad antes de `COMPLETE`. La 09-03 aceptó el camino mínimo, pero no renunció explícitamente a la continuidad. | Ninguno de los **1000/1000** runs (ni de los 475/475 de 19shapes) tiene versión SWMM ni error de continuidad. En 988/1000 los timestamps de inicio y fin son idénticos, así que son fabricados. El 100 % de las columnas hidráulicas secundarias es NULL. Con la base actual no se puede demostrar que alguna de las 1000 simulaciones cumpla el balance de masa. El pipeline de entrenamiento no lee la continuidad: solo lo hace el arnés de validación (`validation/hydrograph_batch.py:180-318`, umbral 5 %). | D02, sección "runs por status": `swmm_null=1000 cont_null=1000 same_ts=988`; `peak_null=160000 …`. D01: `scenario_inflows 0`, `links 0`, `node_timeseries 0`. | En la generación del dataset, leer el error de continuidad del `.rpt` de cada simulación, persistirlo en `runs.continuity_error_pct` junto con `swmm_version`, y rechazar o marcar runs por encima del 5 %. Mientras tanto, re-ejecutar `_read_continuity_error` sobre los `.rpt` de entrenamiento, si se conservan (**no verificado** que existan), y documentar el máximo observado en la tesis. |
| **D-03** | 🟠 MAYOR | `csv_backfill.py:234-235, 248-249, 295, 425-431` | La cadena de evidencia del modelo persistido está vacía en sus campos de reproducibilidad. `hyperparameters_json='{}'`, `preprocessing_json='{}'` y `library_versions_json='{}'`, aunque el modelo real es un `Pipeline(SimpleImputer(median)+XGB(n_estimators, max_depth, lr, subsample…))`. `fit_seconds=0`. `training_query_contract_sha256` es el SHA-256 del literal `"SELECT * FROM training_samples_v17"`, no de la definición de la vista. Los modelos se entrenan con el DataFrame **leído del CSV** (`main.py:874,900`), no con la vista SQL que declara `query_sql`. La spec 08-22 §3.5 exige igualdad de *library-version JSON*, pero con `'{}'` en ambos lados la comprobación pasa trivialmente. | Ambas bases: 2 `trained_models` con blob joblib de 0,8-0,96 MB sin versión de xgboost/sklearn registrada, así que el pickle puede no deserializar o dar otro resultado con otra versión. Hash de la vista real: `9f6c21c3…`. Hash registrado: `01bc303d…`. Un cambio de la vista no invalidaría la proveniencia. | D03: tabla `trained_models` y `model_evaluations` (`hyperparameters_json {}`, `fit_seconds 0.0`); línea "sha256(definición vista) 9f6c21c3… vs contract 01bc303d…". | Serializar los hiperparámetros efectivos (`model.get_params()`), el preprocesado y `{python, numpy, pandas, sklearn, xgboost, joblib}` en los JSON. Calcular el contrato sobre `sqlite_master.sql` de la vista. Entrenar `persist_training_run` sobre `load_training_samples(conn, run_ids)` en lugar del CSV. Medir los tiempos reales. |
| **D-04** | 🟠 MAYOR | `extraction/labels.py:19-23,79-86,110-112`; `config.yaml:14` | El volumen de inundación viene del `.rpt` en 10⁶ L con 3 decimales, es decir, con **resolución de 1 m³**. El umbral de la etiqueta es `flood_threshold_m3: 1.0`, **igual a la resolución**. En la práctica, `inunda = "SWMM imprimió ≥ 0,001 ML"`: el umbral no filtra nada distinto del redondeo del informe. El volumen objetivo del regresor está cuantizado a enteros. | El 100 % de los 46.242 volúmenes positivos son enteros (2.737 valores únicos). El volumen mínimo positivo es exactamente 1,0 m³ y no hay ninguno en (0, 1). El 7,9 % de los positivos (3.642) está entre 1 y 3 m³, donde el error de cuantización relativo llega al 50 %. Hay 72 transiciones 1→0 al **aumentar** el factor (21 nodos, 17 formas), todas con volumen previo de 1-2 m³, más 79 caídas de volumen. Son ruido de redondeo y de cálculo en el borde, no física. | D02 (`frac enteros=1.0000`, `0<vol<thr=0`) y D06 (`vol previo en esas transiciones: {1.0: 68, 2.0: 4}`). | Declarar en la tesis que la etiqueta equivale a "volumen reportado > 0 con resolución de 1 m³". Evaluar un umbral con margen sobre la resolución (p. ej. 5-10 m³) o leer el volumen desde el `.out` binario, que no está redondeado. Analizar la sensibilidad de F1 y de las métricas de volumen al umbral. |
| **D-05** | 🟡 MODERADO | `training_queries.py:380-382`; `main.py:888-889` | `load_training_frame()` es una función de **lectura**, pero llama a `apply_migrations()` sobre la base real sin `WorkflowLock` y sin respaldo. `--persist-sql` hace lo mismo. La spec 08-22 (líneas 539-545) exige `upgrade_database_with_backup()` para bases en 001-004 y que "all managed writers must acquire it [lock]". Una futura migración 006 se aplicaría sin respaldo en el primer `--only-maps` o `--resilience-curve`. | Base sintética en versión 4 → tras `load_training_frame()` queda en **versión 5**, con 0 respaldos creados. Con la base al día, la lectura no cambia bytes (D04-E6: SHA idéntico, sin sidecars residuales). | D04-E7: "versión de esquema tras 'leer': 5 \| backups creados: []". | En lectura, comprobar la versión y **fallar** si hay migraciones pendientes, indicando que se ejecute `upgrade_database_with_backup`. Quitar `apply_migrations` de los caminos de lectura. |
| **D-06** | 🟡 MODERADO | `csv_backfill.py:202, 386, 450`; `main.py:865-916` | `--persist-sql` no es atómico ni idempotente. (a) El backfill hace commit (l. 202) antes de entrenar. (b) `persist_training_run` hace un commit intermedio tras la CV (l. 386): si falla después (ajuste final, `joblib.dump`, inserción del modelo), queda un `training_run` en `RUNNING` permanente, que no se puede borrar por trigger y que fija todos los runs. (c) Repetir la ejecución con el mismo CSV falla con `UNIQUE(scenarios…)`. El único remedio documentado es "borrar el fichero de la base", sin respaldo. `main.py` no usa `try/rollback/close`. | Fallo simulado tras el commit intermedio: `training_runs=[(1,'RUNNING')]`, 10 evaluaciones COMPLETE, 192 OOF, 0 modelos; `DELETE training_runs → 'training runs cannot be deleted'`. Segunda ejecución: `IntegrityError UNIQUE constraint failed: scenarios…`. | D04-E4 y D04-E5. | Un único `BEGIN IMMEDIATE … COMMIT` para el entrenamiento completo, o `recover_abandoned_training_run` en un `except`. Hacer el backfill idempotente (`INSERT … ON CONFLICT DO NOTHING` más verificación de igualdad de contenido). Respaldar con `checkpoint_and_backup` antes de recrear la base. |
| **D-07** | 🟡 MODERADO | `workflow_lock.py` (solo usado en `upgrade.py:78`) | `WorkflowLock` solo se adquiere en `upgrade_database_with_backup`. Ni `--persist-sql`, ni `recover_abandoned_training_run` (que lo delega al llamador), ni las 8 lecturas de `main.py` lo usan. La transacción de escritura de `persist_training_run` dura toda la CV de 5 folds (≈60 s según los timestamps de la base real) y bloquea a otros escritores, cuyo `busy_timeout` es de 5 s. | Dos `--persist-sql` concurrentes o `--persist-sql` + `upgrade` no se serializan. El segundo falla con `database is locked` tras 5 s y deja el estado parcial de D-06. | `grep WorkflowLock swmm_resilience` → única llamada en `upgrade.py:78`. Timestamps de `training_runs` en la base real: 14:48:48 → 14:49:49. | Adquirir `WorkflowLock` en `--persist-sql`, en la recuperación y en `apply_migrations`. |
| **D-08** | 🟡 MODERADO | `reset.py:38-47, 66-78`; `desktop/app.py:835-846` | `reset_db` borra tablas de una lista *legacy* cuyos nombres **coinciden** con tablas v17 (`runs`, `node_results`). Desactiva `PRAGMA foreign_keys`, no respalda y no comprueba qué esquema tiene la base. La GUI permite apuntar `db_var` a cualquier fichero. Solo pide confirmación sí/no (CLI `s/N`, GUI `askyesno`). | Sobre una base v17 **sin** *pinning* (estado posible tras D-06(a)), borra 96 `node_results` y 12 `runs` y deja **96 violaciones FK** (`node_features` huérfanas). Sobre las bases reales (100 % de runs fijados) el trigger lo bloquea (`IntegrityError pinned by training provenance`) y la transacción se revierte sin pérdida. | D04-E3. | Detectar el esquema (`schema_migrations`) y rechazar bases v17. Mantener `foreign_keys=ON`. Hacer `checkpoint_and_backup` antes de borrar. Retirar `--db` para v17 (la spec 08-21 §13 ya preveía retirar el reset *legacy*). |
| **D-09** | 🟡 MODERADO | `database/schema.py:6-122,334-388`; `repository.py:24-28`; `visualization/loaders.py:60-62`; `visualization/runner.py:72-88` | Conviven dos esquemas incompatibles con **tablas homónimas** (`runs` con `run_id TEXT` frente a `INTEGER`, `node_results` con `node_id/flooded` frente a `node_pk/inunda`). `create_schema()` (legacy) ejecutado sobre una base v17 aplica DDL en autocommit y luego falla a mitad. | Sobre una base v17 sintética: se añaden `runs.inflow_multiplier` y `runs.input_source`, se crean 6 tablas legacy (`network_nodes`, `network_links`, `run_inputs`, `link_results`, `run_summary`, `temporal_artifacts`) y termina con `OperationalError no such column: runs.delta_inflow_lps`. El esquema queda contaminado de forma permanente, aunque `load_training_frame` sigue funcionando. | D04-E2. | Guardar en `create_schema` y `connect_db` la condición "si existe `schema_migrations`, abortar". Renombrar o aislar el esquema legacy, o retirarlo (spec 08-21 §13). |
| **D-10** | 🟡 MODERADO | `csv_backfill.py:73-74`; git `core.autocrlf=true` | `network_sha256 = sha256(bytes del .inp en disco)`. El `.inp` está en git con LF (`i/lf`) pero en disco tiene CRLF (`w/crlf`) por `autocrlf`. La identidad de la red **depende de la plataforma o configuración del checkout**. | SHA del blob en git: `1cad5f54fd9b9b0e…`. SHA en disco y en la base: `ea48571b013b9749…`. Un clon en Linux/macOS o con `autocrlf=false` generaría otra `network_id`. `_check_single_snapshot` fallaría al mezclar, o la proveniencia no casaría con la de la tesis. | `git ls-files --eol` y `sha256sum` (sección 5, comandos C1). | Añadir `*.inp -text` (o `binary`) en `.gitattributes` y re-normalizar. Alternativa: documentar el hash canónico y su convención de EOL. |
| **D-11** | 🟡 MODERADO | `.gitignore:68,82` | Los datos canónicos (`outputs/training_v17*.sqlite3`, `data/training/dataset_final.csv`) están ignorados por git y no hay un manifiesto versionado con su hash. Los modelos `outputs/models/*.joblib` sí están versionados. Reconstruir la base exige re-simular SWMM. | Si la base de 55 MB se pierde, desaparece la única copia de 160.000 filas etiquetadas y de la evidencia OOF. El manifiesto del bench `outputs/bench/prepared/af3751551ca5df98` sí registra `source_snapshot_sha256=a88d3b99…`, que **coincide** con un backup de la base actual. El de 19 formas (`4a48359e7e5b53c6`) no tiene ese campo. | D07. | Versionar un `DATA_MANIFEST` con el SHA-256 de la base, del CSV y del `training_frame_sha256`. Archivar la base en almacenamiento externo (Zenodo u OSF) para la defensa. |
| **D-12** | 🟡 MODERADO | `csv_backfill.py:121-128` | `scenarios.config_sha256` solo cubre `{shape_id, factor_mult, duracion_horas, tiempo_al_pico_h}`. No cubre las ordenadas del hidrograma (`scenario_inflows` está vacía), ni las opciones SWMM, ni el `.inp` (este sí va por `network_id`). Dos hidrogramas distintos con el mismo `shape_id`, la misma duración y el mismo tiempo al pico producen el mismo hash. `scenario_key` usa `factor:.2f`. | Las 40 formas reales no colisionan hoy (D02: 0 duplicados de clave natural, 0 colisiones de `scenario_key`), pero el hash no prueba qué hidrograma se simuló. Combinado con D-02, la forma real aplicada no es verificable desde la base. | D02 y lectura de código. | Incluir en `config_json` el SHA-256 del CSV de la forma (ordenadas normalizadas) y de las opciones `[OPTIONS]` usadas. Poblar `scenario_inflows` o, al menos, guardar el hash de la serie. |
| **D-13** | 🟡 MODERADO | spec 09-03 §6; `dataset/assembler.py:12-38`; `main.py:1052` | La Fase 2 del *cutover* no está implementada: `assemble_dataset` solo escribe el CSV, `--export-csv` no existe y la base solo se puebla con `--persist-sql` manual. Hay dos fuentes de verdad sin control automático de paridad: el CSV (modificado el **2026-10-03**) es más nuevo que la base (**2026-09-15**), y todos los lectores usan la base. | Hoy son **idénticos** (verificado: 160.000/160.000 filas, 0 diferencias en las 22 columnas compartidas). Si el CSV se regenera tras re-simular, los comandos seguirán leyendo la base antigua sin aviso. | D03, sección "SQL vs CSV". | Implementar la Fase 2 o, como mínimo, un chequeo de paridad (hash del CSV registrado en la base) que avise al leer. |
| **D-14** | 🔵 MENOR | `sql/001_v17_initial.sql` (todas las tablas) | Las tablas no son `STRICT` (SQLite ≥3.37 lo permite). `CHECK(v >= 0)` acepta `'abc'` (TEXT > número) y `+Inf`. NaN se rechaza porque SQLite lo convierte en NULL. | El riesgo lo mitiga la validación en lectura (`TABULAR_V3_17.validate_frame`, `_validate_targets`). En las bases reales hay 0 inf y 0 tipos inválidos. | Comando C2 (sección 5): `[('abc','text'), (inf,'real')]` aceptados. | Migrar `node_features` y `node_results` a `STRICT` o añadir `CHECK(typeof(x)='real' AND x < 1e308)`. |
| **D-15** | 🔵 MENOR | `training_queries.py:291`; `ml/bench/preprocess.py:47`; `main.py:1057-1068` | El orden canónico del frame es `ORDER BY run_id, node_id`, donde `run_id` es una clave sustituta asignada según el orden de escenarios del CSV. `main.py` afirma que equivale a ordenar por `(shape_id, factor_mult, node_id)`. Es **cierto hoy**, pero solo porque `base` y las demás formas se ensamblan en orden alfabético. Una forma cuyo id ordene antes que `base` rompería la equivalencia y XGBoost (`subsample`) entrenaría con otro orden de filas. | Verificado `True` en ambas bases. Lecturas repetidas son deterministas (D04-E6). | Comando C3 (sección 5). | Ordenar por la clave natural `(shape_id, factor_mult, node_id)` en la vista o el loader, y en `SORT_KEY` del bench. |
| **D-16** | 🔵 MENOR | `sql/005…:1049-1069`; `sql/001…:215-228` | Faltan índices para los triggers de *pinning* y para las FK hijas: `node_features(node_pk)`, `oof_predictions(run_id,node_pk)` y `node_features(node_pk,network_id)`. | `EXPLAIN QUERY PLAN`: `SCAN node_features` (160.000 filas) y `SCAN oof_predictions` (320.000 filas) por cada fila de `nodes` o `node_results` afectada. Es solo rendimiento. | Comando C4 (sección 5). | `CREATE INDEX … ON node_features(node_pk, network_id)` y `… ON oof_predictions(run_id, node_pk)`. |
| **D-17** | 🔵 MENOR | `upgrade.py:53-62, 110-116`; `maintenance.py:22-31` | La "huella lógica" del upgrade solo usa `PRAGMA schema_version` más los checksums: no detecta cambios de **datos** entre el respaldo y la migración. Se revalida fuera de `BEGIN IMMEDIATE` (la spec 08-22, líneas 551-553, pide hacerlo dentro). El respaldo se verifica con `integrity_check`, pero no con `foreign_key_check`. | Ventana de carrera con escritores que no toman el lock (D-07). Sin incidencia observada. | Lectura de código. | Incluir conteos o un hash de tablas en la huella, revalidar dentro de la transacción de la migración y añadir `foreign_key_check` al respaldo. |
| **D-18** | 🔵 MENOR | `extraction/labels.py:110-112`; `dataset/assembler.py:29-31` | Los volúmenes no numéricos del `.rpt` se convierten en 0,0 (`fillna(0.0)`), y los nodos sin etiqueta tras el `merge` reciben `inunda=0`, sin aviso. Un desajuste de formato de `node_id` (mayúsculas, espacios) etiquetaría nodos como "no inunda" en silencio. | No hay evidencia de que haya ocurrido: 0 discrepancias etiqueta/volumen. Los 48 nodos que nunca inundan (30 %) **no se verificaron** contra los `.rpt`. | Lectura de código. | Contar y avisar (o fallar) cuando hay coerciones o nodos sin correspondencia en el `.rpt`. |
| **D-19** | 🔵 MENOR | `repository.py:31-130`; `analysis/dataset.py:10-68` | Camino *legacy*: `save_results` hace 4 `to_sql` con commit independiente (escrituras parciales), usa `INSERT OR REPLACE` en la topología y FK desactivadas (`sqlite3.connect` sin pragma). El export filtra `status='completed'`. | La base legacy `data/training/swmm_resilience.db` no existe hoy; el camino está inactivo y no afecta a la tesis. | Lectura de código y `ls data/training`. | Retirarlo o aislarlo (spec 08-21 §13). |
| **D-20** | 🔵 MENOR | `csv_backfill.py:274-284` | La métrica persistida en la base (`f1=0,9658`, `rmse=125,10`) procede de `GroupKFold(5)` por `factor_mult`, con factores intercalados (fold 0 = {1,2,3,4,5}) y **todas las formas presentes en entrenamiento y validación**. No es una métrica de generalización a formas no vistas (LOSO). | Si se citara en la tesis, sobreestimaría la generalización frente al protocolo LOSO. Sin solapamiento de runs entre entrenamiento y validación (0). | D03: "factores de validación por fold". | Etiquetar `scope` con el protocolo (p. ej. `group_kfold5_by_factor`) y no usar este valor como métrica principal. Ver el anexo de ML. |
| **D-21** | 🔵 MENOR | features topológicas (extracción) | El nodo `108C` tiene `n_tuberias_out=1` y `diam_max_out=0,7`, pero `pendiente_out=NULL`. Es una incoherencia de la feature. | 1000 filas (0,6 %). Pasa el contrato porque `pendiente_out` es *nullable*. | D06. | Revisar el tipo del enlace de salida de `108C` (probablemente vierte a un *outfall* o no es conducto) y documentarlo. |
| **D-22** | 🔵 MENOR | `tests/database/` | Cobertura: 208 tests verdes, pero **ninguno** cubre: `UPDATE runs.scenario_id` o `config_sha256` sobre runs fijados (D-01), `scenario_inflows` fijado, fallo de `persist_training_run` a mitad (D-06), re-ejecución con el mismo CSV, `reset_db` sobre v17 (D-08), `create_schema` legacy sobre v17 (D-09), migración disparada por una lectura (D-05) ni dependencia del EOL (D-10). `test_backfill_reuses_existing_network_on_rerun` usa escenarios **disjuntos**, así que no prueba la idempotencia real. | La cobertura de triggers 002-005 es amplia (más de 4.000 líneas de tests), pero los caminos operativos reales (CLI) están poco probados. | `grep "UPDATE runs" tests/database` → sin resultados. | Añadir tests de regresión para D-01, D-05, D-06, D-08 y D-09. |

---

## 3. Lo que está bien (verificado)

| # | Aspecto | Evidencia |
|---|---|---|
| B-01 | Integridad física y referencial de ambas bases: `PRAGMA integrity_check = ok` y `foreign_key_check = 0` violaciones. | D01 |
| B-02 | Historial de migraciones 001-005 contiguo. Los checksums guardados coinciden con el catálogo actual (`430d13d7…`, `ac73354c…`, `d90c151f…`, `142d235e…`, `bc296815…`). La normalización de saltos de línea de `read_text()` hace el checksum inmune a CRLF/LF en los `.sql`. El validador de 005 está registrado (`validate_before_005`, `37c7a858…`). | D01 y cálculo con `_migration_catalog` |
| B-03 | El motor de migraciones es atómico: cada migración va en `BEGIN IMMEDIATE` con *rollback*, prohíbe `BEGIN`/`SAVEPOINT` dentro del SQL mediante el *authorizer*, comprueba el prefijo y el checksum, rechaza historial adelantado, hace *preflight* de FK e integridad antes de 005 y rechaza ejecutarse con una transacción abierta. | `migrations.py:86-226`, tests de migración en verde |
| B-04 | Las conexiones gestionadas aplican `foreign_keys=ON`, `recursive_triggers=ON` y `journal_mode=WAL`, comprobando el valor devuelto, más `busy_timeout`. `sha256()` se registra como determinista y falla cerrado con NULL o un tipo inválido. El trigger de *pinning* de red falla cerrado en conexiones no gestionadas (sin `sha256()`). | `connection.py`, `005:1117-1129` |
| B-05 | Cobertura completa forma × factor × nodo: 40 × 25 × 160 = **160.000** (y 19 × 25 × 160 = 76.000). 0 celdas faltantes, 0 duplicados de clave natural, 1 run por escenario, 0 colisiones de `scenario_key`, 0 huérfanos (nodos, features, resultados, escenarios), 100 % de runs fijados por el `training_run`. | D02 |
| B-06 | Consistencia de la etiqueta: `inunda == (vol ≥ 1,0 m³)` en 160.000/160.000 filas. 0 volúmenes negativos, 0 inf, 0 NaN en los objetivos. `tiempo_al_pico_h ≤ duracion_horas` siempre. | D02 |
| B-07 | Coherencia física de las features de escala: `q_pico_nodo = base_inflow × factor` y `q_pico_acum_escalado = q_pico_acum_base × factor` exactos (cociente min = max = 1,000000). Las 13 features estáticas son invariantes por nodo entre los 1000 runs. `nodes` y `node_features` coinciden en elevación, profundidad y caudal base. Los NaN de `diam_max_in`, `pendiente_max_in` y `upstream_capacity_lps` (57.000) corresponden exactamente a nodos de cabecera (`n_tuberias_in=0`). | D02 |
| B-08 | **SQL ≡ CSV**: 160.000 filas, unión externa 100 % `both` y 0 diferencias en todas las columnas compartidas. La base de 19 formas es un subconjunto exacto de la de 40 (0 diferencias en las 76.000 claves comunes). | D03 |
| B-09 | La red está autocontenida: `network_sha256 == sha256(inp_bytes)` y `inp_bytes` es igual al `.inp` actual en disco (421.820 bytes). Los blobs de los modelos coinciden con `model_sha256`. El hash del contrato de features es `56af955c…`, igual al publicado en la spec 08-22 §3.5. | D03 |
| B-10 | Evidencia OOF completa y sin fuga: `included_run_ids_json` es igual a `training_run_inputs` (1000). Hay 160.000 OOF únicas por objetivo y 0 runs en entrenamiento y validación del mismo fold. Las evaluaciones están COMPLETE, 5 por tarea. | D03 |
| B-11 | Lectura reproducible: dos llamadas a `load_training_frame` devuelven frames idénticos, en el mismo orden. Sobre una base al día, la lectura no modifica bytes ni deja sidecars. El loader valida cardinalidad, simetría de claves, contrato y dominio de los objetivos dentro de un `SAVEPOINT`, que da una instantánea consistente en WAL. | D04-E6, `training_queries.py` |
| B-12 | Trazabilidad del bench: el manifiesto `af3751551ca5df98` (160.000 filas) registra `source_snapshot_sha256 = a88d3b99…`, que coincide con un backup tomado hoy de la base real. Dos backups del mismo origen dan el mismo SHA, así que la huella es estable. | D07, D04-E6 |
| B-13 | El *pinning* protege `node_results`, `node_features`, `scenarios`, `networks` y `nodes`: el `reset_db` sobre una base fijada se revierte por completo. Tablas de evidencia append-only (OOF, métricas, modelos, promociones). Existe una recuperación explícita `RUNNING→FAILED` que no borra filas. | D04-E3, `005`, `recovery.py` |
| B-14 | `WorkflowLock` usa un bloqueo de SO (`msvcrt`/`fcntl`) que se libera automáticamente si el proceso muere, así que no hay locks huérfanos. El fichero `.workflow.lock` persiste pero es inocuo. El upgrade hace *checkpoint* WAL, respaldo, `integrity_check` y SHA del respaldo antes de migrar. | `workflow_lock.py`, `upgrade.py`, tests en verde |
| B-15 | `export_training_samples_csv` escribe de forma atómica (`mkstemp` + `replace`). | `training_queries.py:310-331` |
| B-16 | `pytest tests/database`: **208 passed**, 3 deselected (scale), 28 s. | Ejecución local |

**No verificado:** los valores reales de error de continuidad de las 1000 simulaciones (no están en la base; ver D-02), la existencia de los `.rpt` de entrenamiento, las pruebas `-m scale`, el comportamiento POSIX de `WorkflowLock` y la seguridad de hilos de la GUI.

---

## 4. Resultados de los chequeos de calidad sobre los datos reales

### 4.1 Inventario (filas por tabla)

| Tabla | `training_v17` | `training_v17_19shapes` |
|---|---:|---:|
| networks | 1 | 1 |
| nodes | 160 | 160 |
| links | 0 | 0 |
| scenarios | 1.000 | 475 |
| scenario_inflows | 0 | 0 |
| runs (todos COMPLETE) | 1.000 | 475 |
| node_features | 160.000 | 76.000 |
| node_results | 160.000 | 76.000 |
| node_timeseries | 0 | 0 |
| training_runs | 1 | 1 |
| training_run_inputs | 1.000 | 475 |
| model_evaluations | 10 | 10 |
| model_evaluation_runs | 10.000 | 4.750 |
| oof_predictions | 320.000 | 152.000 |
| trained_models | 2 | 2 |
| model_metrics | 2 | 2 |
| candidates/rankings/promotions/selections | 0 | 0 |
| schema_migrations | 5 | 5 |
| Objetos: tablas / índices / triggers / vistas | 33 / 43 / 97 / 4 | 33 / 43 / 97 / 4 |
| integrity_check / foreign_key_check | ok / 0 | ok / 0 |
| Cabecera WAL (bytes 18-19) | 2 / 2 | 2 / 2 |

### 4.2 Cobertura y claves

| Chequeo | `training_v17` | `19shapes` |
|---|---:|---:|
| Formas × factores × nodos | 40 × 25 × 160 | 19 × 25 × 160 |
| Filas esperadas / presentes | 160.000 / 160.000 | 76.000 / 76.000 |
| Celdas faltantes | 0 | 0 |
| Duplicados (shape, factor, node) | 0 | 0 |
| Escenarios con más de 1 run | 0 | 0 |
| `runs.config_sha256 ≠ scenarios.config_sha256` | 0 / 1.000 | 0 / 475 |
| Rango de factor (paso) | 0,2-5,0 (0,2) | 0,2-5,0 (0,2) |
| Nodos sin features / features sin resultado / resultados sin feature | 0 / 0 / 0 | 0 / 0 / 0 |
| Runs no fijados por un training_run | 0 | 0 |

### 4.3 Rangos físicos e imposibles

| Chequeo | `training_v17` | `19shapes` |
|---|---:|---:|
| `vol_inundacion_m3 < 0` | 0 | 0 |
| `factor_mult ≤ 0` | 0 | 0 |
| `prof_max ≤ 0` | 0 | 0 |
| `q_pico_nodo < 0` | 0 | 0 |
| `tiempo_al_pico_h > duracion_horas` | 0 | 0 |
| inf en features | 0 | 0 |
| `q_pico_acum_escalado < q_pico_nodo` | 0 | 0 |
| `n_in=0` con diámetro de entrada / `n_in>0` sin diámetro | 0 / 0 | 0 / 0 |
| NaN `diam_max_in`, `pendiente_max_in`, `upstream_capacity_lps` | 57.000 c/u (nodos de cabecera) | 27.075 c/u |
| NaN `pendiente_out` (nodo 108C, ver D-21) | 1.000 | 475 |
| `vol_inundacion_m3` máximo | 7.418 m³ | 7.126 m³ |
| `duracion_horas` / `tiempo_al_pico_h` | 0,5-15 h / 0,083-11 h | 1-13 h / 0,167-6,5 h |
| `peak_flooding_lps`, `flooding_duration_min`, `max_depth_m` NULL | 160.000 (100 %) | 76.000 (100 %) |
| `swmm_version` / `continuity_error_pct` NULL | 1.000 / 1.000 | 475 / 475 |
| Runs con `started_at = completed_at` | 988 | 473 |

### 4.4 Balance de clases

| Métrica | `training_v17` | `19shapes` |
|---|---:|---:|
| `inunda=1` / total | 46.242 / 160.000 (**28,90 %**) | 22.058 / 76.000 (29,02 %) |
| Prevalencia por factor (mín-máx) | 0,0 % - 57,67 % | 0,0 % - 58,03 % |
| Factores sin ningún inundado | 5 (0,2-1,0) = 32.000 filas (20 %) | 5 |
| Factores con 100 % inundados | 0 | 0 |
| Prevalencia por forma (mín-máx) | 21,85 % (`shape_ultra_short_impulse`) - 30,62 % (`new_hydrograph`) | 25,15 % (`rare_extreme`) - 30,62 % |
| Nodos que nunca inundan / que siempre inundan | 48 / 0 (de 160) | 53 / 0 |

### 4.5 Etiqueta frente a volumen (umbral 1,0 m³)

| `inunda` \ volumen | vol = 0 | 0 < vol < 1 | vol ≥ 1 |
|---|---:|---:|---:|
| 0 | 113.758 | 0 | 0 |
| 1 | 0 | 0 | 46.242 |

| Métrica de cuantización y monotonía | `training_v17` | `19shapes` |
|---|---:|---:|
| Discrepancias `inunda ≠ (vol ≥ 1)` | 0 | 0 |
| Fracción de volúmenes positivos enteros | 100 % | 100 % |
| Volumen positivo mínimo | 1,0 m³ | 1,0 m³ |
| Positivos con 1-3 m³ | 3.642 (7,9 %) | n/d |
| Cuantiles de volumen de inundados (0 / 25 / 50 / 75 / 99 / 100 %) | 1 / 24 / 107 / 343,75 / 3.016 / 7.418 | 1 / 21 / 98 / 313 / 2.569 / 7.126 |
| Transiciones `inunda` 1→0 al subir el factor | 72 de 153.600 (21 nodos, 17 formas; volumen previo 1-2 m³) | 48 de 72.960 |
| Caídas de volumen al subir el factor (>1 m³) | 79 (7); caída máxima 4 m³ | 50 (3) |

### 4.6 Proveniencia

| Chequeo | `training_v17` | `19shapes` |
|---|---|---|
| `network_sha256 == sha256(inp_bytes)` | Sí | Sí |
| `inp_bytes ==` `.inp` actual en disco | Sí (`ea48571b…`) | Sí |
| SHA del blob git del mismo `.inp` (LF) | `1cad5f54…` (≠, D-10) | ídem |
| `sha256(model_blob) == model_sha256` | 2/2 | 2/2 |
| `feature_contract_sha256` | `56af955c…` (= spec) | ídem |
| `training_query_contract_sha256` frente a SHA de la vista | `01bc303d…` ≠ `9f6c21c3…` (D-03) | ídem |
| `hyperparameters_json`, `preprocessing_json`, `library_versions_json` | `{}`, `{}`, `{}` | ídem |
| Métrica persistida (GroupKFold5 por factor) | F1 0,9658; RMSE 125,10 m³ | F1 0,9646; RMSE 88,17 m³ |
| Folds de validación (factores) | f0 {1..5}, f1 {0,8..4,8}, f2 {0,6..4,6}, f3 {0,4..4,4}, f4 {0,2..4,2} | ídem |
| SQL frente a `dataset_final.csv` | 0 diferencias / 160.000 filas | n/a |
| Manifiesto del bench frente a la base | `a88d3b99…` = backup actual | sin campo de snapshot |

---

## 5. Scripts reproducibles

Todos son de solo lectura sobre `outputs/` (`?mode=ro&immutable=1`). Los experimentos D04 y D05 crean bases sintéticas en un subdirectorio temporal junto al script. Se ejecutan con `venv\Scripts\python.exe -I <script>` desde cualquier directorio fuera de `outputs/`; los scripts usan rutas absolutas al repositorio. Copias usadas en esta auditoría: `%TEMP%\claude\…\scratchpad\auditoria_datos_sql\`.

### D01 — Inventario (`d01_inventario.py`)

```python
"""D01 - Inventario de esquema y conteos (SOLO LECTURA, immutable=1: no toca -wal/-shm)."""
import sqlite3
from pathlib import Path

ROOT = Path(r"C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model")
DBS = [ROOT / "outputs/training_v17.sqlite3", ROOT / "outputs/training_v17_19shapes.sqlite3"]

def ro(p):
    return sqlite3.connect(f"{p.resolve().as_uri()}?mode=ro&immutable=1", uri=True)

for db in DBS:
    c = ro(db)
    print("=" * 80); print(db.name)
    hdr = open(db, "rb").read(100)
    print("header write/read version (2=WAL):", hdr[18], hdr[19])
    for pragma in ("journal_mode", "page_size", "page_count", "freelist_count",
                   "user_version", "schema_version", "foreign_keys", "auto_vacuum"):
        print(f"  PRAGMA {pragma} =", c.execute(f"PRAGMA {pragma}").fetchone()[0])
    print("  integrity_check:", c.execute("PRAGMA integrity_check").fetchall()[:5])
    fk = c.execute("PRAGMA foreign_key_check").fetchall()
    print("  foreign_key_check violaciones:", len(fk), fk[:5])
    for r in c.execute("SELECT version,name,substr(checksum_sha256,1,12),applied_at_utc FROM schema_migrations ORDER BY version"):
        print("   ", r)
    for r in c.execute("SELECT * FROM schema_migration_validators"):
        print("   validator", r[0], r[1], r[2][:12])
    print("   ", c.execute("SELECT type, COUNT(*) FROM sqlite_master GROUP BY type").fetchall())
    for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall():
        print(f"   {t:40s} {c.execute(f'SELECT COUNT(*) FROM \"{t}\"').fetchone()[0]:>10d}")
    c.close()
```

### D02 — Calidad de datos (`d02_calidad.py`)

```python
"""D02 - Chequeos de calidad de datos (SOLO LECTURA, immutable=1)."""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model")
DBS = [ROOT / "outputs/training_v17.sqlite3", ROOT / "outputs/training_v17_19shapes.sqlite3"]
THRESHOLD = 1.0

def ro(p):
    return sqlite3.connect(f"{p.resolve().as_uri()}?mode=ro&immutable=1", uri=True)

def q(c, sql):
    return pd.read_sql_query(sql, c)

for db in DBS:
    c = ro(db); print("=" * 90); print(db.name)
    print(q(c, "SELECT r.status, s.scenario_kind, COUNT(*) n, MIN(r.node_count) mn, MAX(r.node_count) mx, "
               "SUM(r.swmm_version IS NULL) swmm_null, SUM(r.continuity_error_pct IS NULL) cont_null, "
               "SUM(r.started_at_utc=r.completed_at_utc) same_ts FROM runs r JOIN scenarios s USING(scenario_id) GROUP BY 1,2"))
    print(q(c, "SELECT COUNT(*) esc_multi_run FROM (SELECT scenario_id FROM runs GROUP BY scenario_id HAVING COUNT(*)>1)"))
    print(q(c, "SELECT SUM(r.config_sha256<>s.config_sha256) difieren, COUNT(*) total FROM runs r JOIN scenarios s USING(scenario_id)"))
    print(q(c, "SELECT COUNT(*) dup FROM (SELECT network_id, shape_id, factor_mult FROM scenarios GROUP BY 1,2,3 HAVING COUNT(*)>1)"))
    print(q(c, "SELECT COUNT(*) dup_key FROM (SELECT scenario_key FROM scenarios GROUP BY 1 HAVING COUNT(DISTINCT factor_mult)>1)"))
    f = q(c, "SELECT * FROM training_samples_v17")
    shapes, factors, nodes = (sorted(f[k].unique()) for k in ("shape_id", "factor_mult", "node_id"))
    print("shapes/factores/nodos:", len(shapes), len(factors), len(nodes), "esperado", len(shapes)*len(factors)*len(nodes), "filas", len(f))
    dup = f.groupby(["shape_id", "factor_mult", "node_id"]).size()
    print("duplicados:", int((dup > 1).sum()), "faltantes:",
          len(pd.MultiIndex.from_product([shapes, factors, nodes]).difference(dup.index)))
    na = f.isna().sum(); print(na[na > 0].to_string())
    print(f.drop(columns=["scenario_key", "scenario_kind", "shape_id", "node_id"]).describe().T[["min", "max", "mean"]].to_string())
    checks = {
        "vol<0": (f.vol_inundacion_m3 < 0).sum(), "factor<=0": (f.factor_mult <= 0).sum(),
        "prof_max<=0": (f.prof_max <= 0).sum(), "q_pico_nodo<0": (f.q_pico_nodo < 0).sum(),
        "tpico>duracion": (f.tiempo_al_pico_h > f.duracion_horas).sum(),
        "n_in=0 & diam_in notnull": ((f.n_tuberias_in == 0) & f.diam_max_in.notna()).sum(),
        "n_in>0 & diam_in null": ((f.n_tuberias_in > 0) & f.diam_max_in.isna()).sum(),
        "q_acum_esc < q_pico_nodo": (f.q_pico_acum_escalado < f.q_pico_nodo - 1e-9).sum(),
        "inf": int(np.isinf(f.select_dtypes("number").to_numpy()).sum())}
    for k, v in checks.items(): print(f"  {k:30s} {int(v)}")
    rel = (f.q_pico_acum_escalado / f.q_pico_acum_base) / f.factor_mult
    r2 = (f.q_pico_nodo / (f.base_inflow_lps * f.factor_mult)).replace([np.inf, -np.inf], np.nan)
    print("escalado acum min/max:", rel.min(), rel.max(), "| q_pico_nodo/(qb*f) min/max:", r2.min(), r2.max())
    static = ["elev_fondo", "prof_max", "n_tuberias_in", "n_tuberias_out", "diam_max_in", "diam_max_out",
              "pendiente_max_in", "pendiente_out", "base_inflow_lps", "dist_outfall_m",
              "n_nodos_aguas_arriba", "q_pico_acum_base", "upstream_capacity_lps"]
    print("max nunique estáticos por nodo:", int(f.groupby("node_id")[static].nunique(dropna=False).max().max()))
    print("global inunda:", f.inunda.sum(), len(f), round(f.inunda.mean(), 4))
    by_f = f.groupby("factor_mult").inunda.mean(); by_s = f.groupby("shape_id").inunda.mean()
    by_n = f.groupby("node_id").inunda.mean()
    print("por factor min/max:", by_f.min(), by_f.max(), "factores sin positivos:", int((by_f == 0).sum()))
    print("por forma min/max:", by_s.idxmin(), by_s.min(), by_s.idxmax(), by_s.max())
    print("nodos nunca/siempre:", int((by_n == 0).sum()), int((by_n == 1).sum()))
    exp = (f.vol_inundacion_m3 >= THRESHOLD).astype(int)
    print(pd.crosstab(f.inunda, [f.vol_inundacion_m3 == 0, exp]).to_string(), "discrepancias:", int((f.inunda != exp).sum()))
    pos = f.vol_inundacion_m3[f.vol_inundacion_m3 > 0]
    print("vol>0 n/min/frac enteros/(0,thr):", len(pos), pos.min(), np.isclose(pos, pos.round()).mean(), ((pos > 0) & (pos < THRESHOLD)).sum())
    print("cuantiles:", f.vol_inundacion_m3[f.inunda == 1].quantile([0, .25, .5, .75, .99, 1]).round(3).tolist())
    g = f.sort_values(["shape_id", "node_id", "factor_mult"])
    d = g.groupby(["shape_id", "node_id"]).inunda.diff(); dv = g.groupby(["shape_id", "node_id"]).vol_inundacion_m3.diff()
    print("1->0:", int((d == -1).sum()), "de", int(d.notna().sum()), "| caídas vol:", int((dv < -1e-9).sum()), ">1m3:", int((dv < -1).sum()))
    print(q(c, "SELECT COUNT(*) n, SUM(peak_flooding_lps IS NULL) peak_null, SUM(flooding_duration_min IS NULL) dur_null, "
               "SUM(max_depth_m IS NULL) depth_null FROM node_results"))
    print(q(c, "SELECT (SELECT COUNT(*) FROM nodes WHERE node_pk NOT IN (SELECT node_pk FROM node_features)) nodos_sin_feat, "
               "(SELECT COUNT(*) FROM node_features f LEFT JOIN node_results r USING(run_id,node_pk) WHERE r.run_id IS NULL) feat_sin_res, "
               "(SELECT COUNT(*) FROM node_results r LEFT JOIN node_features f USING(run_id,node_pk) WHERE f.run_id IS NULL) res_sin_feat, "
               "(SELECT COUNT(*) FROM runs WHERE run_id NOT IN (SELECT run_id FROM training_run_inputs)) runs_no_pinned, "
               "(SELECT COUNT(*) FROM scenarios WHERE scenario_id NOT IN (SELECT scenario_id FROM runs)) esc_sin_run"))
    print(q(c, "SELECT SUM(n.invert_elevation_m<>f.elev_fondo) elev, SUM(n.max_depth_m<>f.prof_max) prof, "
               "SUM(n.base_inflow_lps<>f.base_inflow_lps) qb FROM node_features f JOIN nodes n USING(node_pk)"))
    c.close()
```

### D03 — Proveniencia, hashes y SQL frente a CSV (`d03_proveniencia_csv.py`)

```python
"""D03 - Proveniencia, hashes y consistencia SQL vs CSV (SOLO LECTURA, immutable=1)."""
import hashlib, json, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model")
DB40, DB19 = ROOT / "outputs/training_v17.sqlite3", ROOT / "outputs/training_v17_19shapes.sqlite3"
CSV = ROOT / "data/training/dataset_final.csv"
INP = ROOT / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
KEY = ["shape_id", "factor_mult", "node_id"]
ro = lambda p: sqlite3.connect(f"{p.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
sha = lambda b: hashlib.sha256(b).hexdigest()

for db in (DB40, DB19):
    c = ro(db); print("=" * 90, db.name)
    s_db, inp = c.execute("SELECT network_sha256, inp_bytes FROM networks").fetchone()
    print("network_sha==sha(inp_bytes):", s_db == sha(inp), "| inp_bytes==disco:", inp == INP.read_bytes())
    tr = pd.read_sql_query("SELECT * FROM training_runs", c); print(tr.T.to_string())
    print("sha(query_sql)==contract:", sha(tr.query_sql[0].encode()) == tr.training_query_contract_sha256[0])
    view_sql = c.execute("SELECT sql FROM sqlite_master WHERE name='training_samples_v17'").fetchone()[0]
    print("sha(vista):", sha(view_sql.encode())[:16], "contract:", tr.training_query_contract_sha256[0][:16])
    inc = json.loads(tr.included_run_ids_json[0])
    print("included==inputs:", sorted(inc) == [r[0] for r in c.execute("SELECT run_id FROM training_run_inputs ORDER BY run_id")])
    print(pd.read_sql_query("SELECT model_id,target,algorithm,hyperparameters_json,preprocessing_json,target_transform_json,"
                            "library_versions_json,length(model_blob) nbytes FROM trained_models", c).to_string())
    for mid, s, blob in c.execute("SELECT model_id, model_sha256, model_blob FROM trained_models"):
        print("blob ok", mid, sha(blob) == s)
    print(pd.read_sql_query("SELECT model_id, scope, metric_name, value FROM model_metrics", c).to_string())
    print(pd.read_sql_query("SELECT evaluation_id,task,hyperparameters_json,fold_id,status,fit_seconds FROM model_evaluations", c).to_string())
    vf = pd.read_sql_query("SELECT e.fold_id, s.factor_mult FROM model_evaluation_runs m JOIN model_evaluations e USING(evaluation_id) "
                           "JOIN runs r USING(run_id) JOIN scenarios s USING(scenario_id) "
                           "WHERE m.role='validation' AND e.task='classification'", c)
    print(vf.groupby("fold_id").factor_mult.apply(lambda s: sorted(set(np.round(s, 2)))).to_string())
    print("train y val solapados:", c.execute("SELECT COUNT(*) FROM model_evaluation_runs a JOIN model_evaluation_runs b ON "
                                  "a.evaluation_id=b.evaluation_id AND a.run_id=b.run_id AND a.role='train' AND b.role='validation'").fetchone()[0])
    print(pd.read_sql_query("SELECT target, COUNT(*) n, COUNT(DISTINCT run_id||'-'||node_pk) uniq FROM oof_predictions GROUP BY target", c))
    c.close()

c = ro(DB40); sql = pd.read_sql_query("SELECT * FROM training_samples_v17", c); c.close()
csv = pd.read_csv(CSV)
m = sql.assign(factor_mult=sql.factor_mult.round(6)).merge(csv.assign(factor_mult=csv.factor_mult.round(6)),
                                                           on=KEY, how="outer", suffixes=("_sql", "_csv"), indicator=True)
print("merge:", m["_merge"].value_counts().to_dict())
for cc in [k for k in csv.columns if k not in KEY and k + "_sql" in m.columns]:
    a, b = m[cc + "_sql"], m[cc + "_csv"]
    diff = ~((a.isna() & b.isna()) | np.isclose(a.astype(float), b.astype(float), rtol=0, atol=1e-9))
    if diff.sum(): print("dif", cc, int(diff.sum()))
c = ro(DB19); s19 = pd.read_sql_query("SELECT * FROM training_samples_v17", c); c.close()
mm = sql.assign(factor_mult=sql.factor_mult.round(6)).merge(s19.assign(factor_mult=s19.factor_mult.round(6)), on=KEY, suffixes=("_40", "_19"))
print("comunes:", len(mm), {k: int((~np.isclose(mm[k + "_40"], mm[k + "_19"])).sum())
                            for k in ["inunda", "vol_inundacion_m3", "duracion_horas", "tiempo_al_pico_h", "q_pico_nodo"]})
```

### D04 — Experimentos de comportamiento sobre bases sintéticas (`d04_experimentos.py`)

```python
"""D04 - Experimentos sobre BD SINTÉTICAS/COPIAS en un directorio temporal junto al script."""
import hashlib, shutil, sqlite3, sys, tempfile
from pathlib import Path

ROOT = Path(r"C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests" / "database"))
WORK = Path(tempfile.mkdtemp(prefix="d04_", dir=Path(__file__).resolve().parent))
from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database import csv_backfill
from swmm_resilience.database.csv_backfill import backfill_networks_and_runs, persist_training_run
from swmm_resilience.database.schema import create_schema
from swmm_resilience.database.training_queries import load_training_frame
from swmm_resilience.reset import reset_db
from swmm_resilience.config import load_config
from test_csv_backfill import _synthetic_dataset   # 2 formas x 6 factores x 8 nodos = 96 filas

sha_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]

def fresh(name, train=True):
    db, inp = WORK / name, WORK / "net.inp"; inp.write_text("[TITLE]\nsynthetic\n")
    conn = connect_managed_database(db); apply_migrations(conn)
    df = _synthetic_dataset(); info = backfill_networks_and_runs(conn, df, inp, "net")
    if train:
        persist_training_run(conn, df, info["run_id_by_key"], info["node_pk_by_id"], load_config(ROOT / "config.yaml"))
    return conn, db, df, inp, info

def e1():  # pinning incompleto
    conn, *_ = fresh("e1.sqlite3")
    conn.execute("UPDATE runs SET scenario_id=(SELECT scenario_id FROM runs WHERE run_id=2) WHERE run_id=1"); conn.commit()
    print("runs.scenario_id de run fijado: UPDATE ACEPTADO")
    conn.execute("UPDATE runs SET config_sha256=?, continuity_error_pct=99 WHERE run_id=1", ("f" * 64,)); conn.commit()
    conn.execute("INSERT INTO scenario_inflows VALUES (1,1,1,0,0,5.0)")
    conn.execute("UPDATE scenario_inflows SET inflow_lps=999 WHERE scenario_id=1")
    conn.execute("DELETE FROM scenario_inflows WHERE scenario_id=1"); conn.commit()
    print("scenario_inflows de escenario fijado: INSERT/UPDATE/DELETE ACEPTADOS")

def e2():  # create_schema legacy sobre v17
    conn, db, *_ = fresh("e2.sqlite3", train=False); conn.close()
    before = [r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(runs)")]
    c = sqlite3.connect(db)
    try: create_schema(c)
    except Exception as e: print("create_schema lanzó:", type(e).__name__, e)
    finally: c.close()
    c = sqlite3.connect(db)
    print("columnas añadidas a runs:", sorted(set(r[1] for r in c.execute("PRAGMA table_info(runs)")) - set(before)))
    print("tablas legacy:", [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN "
          "('network_nodes','network_links','run_inputs','link_results','run_summary','temporal_artifacts')")])

def e3():  # reset_db sobre v17
    conn, db, *_ = fresh("e3u.sqlite3", train=False); conn.close()
    n = reset_db(db); c = sqlite3.connect(db)
    print("sin pinning: borradas", n, "node_features restantes", c.execute("SELECT COUNT(*) FROM node_features").fetchone()[0],
          "violaciones FK", len(c.execute("PRAGMA foreign_key_check").fetchall()))
    conn, db2, *_ = fresh("e3p.sqlite3", train=True); conn.close()
    try: reset_db(db2)
    except Exception as e: print("con pinning:", type(e).__name__, e)

def e4():  # re-ejecución backfill
    conn, db, df, inp, _ = fresh("e4.sqlite3", train=True)
    try: backfill_networks_and_runs(conn, df, inp, "net")
    except Exception as e: print("2a ejecución:", type(e).__name__, e)

def e5():  # fallo tras commit intermedio
    conn, db, df, inp, info = fresh("e5.sqlite3", train=False)
    orig = csv_backfill.joblib.dump
    csv_backfill.joblib.dump = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fallo simulado"))
    try: persist_training_run(conn, df, info["run_id_by_key"], info["node_pk_by_id"], load_config(ROOT / "config.yaml"))
    except RuntimeError as e: print("persist lanzó:", e)
    finally: csv_backfill.joblib.dump = orig
    conn.close(); c = sqlite3.connect(db)
    print(c.execute("SELECT training_run_id, status FROM training_runs").fetchall(),
          c.execute("SELECT COUNT(*) FROM trained_models").fetchone()[0])
    try: c.execute("DELETE FROM training_runs")
    except sqlite3.DatabaseError as e: print("DELETE:", e)

def e6():  # lectura sobre copia del BD real
    copies = []
    for name in ("a", "b"):
        src = sqlite3.connect(f"{(ROOT/'outputs/training_v17_19shapes.sqlite3').as_uri()}?mode=ro&immutable=1", uri=True)
        p = WORK / f"copia_{name}.sqlite3"; dst = sqlite3.connect(p); src.backup(dst); dst.close(); src.close(); copies.append(p)
    print("backups idénticos:", sha_file(copies[0]) == sha_file(copies[1]))
    h0 = sha_file(copies[0]); fr = load_training_frame(copies[0]); fr2 = load_training_frame(copies[0])
    print("sha antes/después:", h0, sha_file(copies[0]), "| lecturas iguales:", fr.equals(fr2),
          "| sidecars:", list(WORK.glob("copia_a.sqlite3-*")))

def e7():  # migración por lectura
    cat = WORK / "cat4"; cat.mkdir()
    for f in sorted((ROOT / "swmm_resilience/database/sql").glob("00[1-4]_*.sql")): shutil.copyfile(f, cat / f.name)
    db = WORK / "e7.sqlite3"; conn = connect_managed_database(db); apply_migrations(conn, migration_dir=cat); conn.close()
    try: load_training_frame(db)
    except Exception as e: print("load_training_frame:", type(e).__name__, e)
    print("versión tras 'leer':", sqlite3.connect(db).execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0])

for f in (e1, e2, e3, e4, e5, e6, e7):
    print("###", f.__name__)
    try: f()
    except Exception as exc: print("EXCEPCIÓN:", type(exc).__name__, str(exc)[:300])
```

### D05 — Permutación silenciosa de `scenario_id` (`d05_swap.py`)

```python
"""D05 - Permutación silenciosa de scenario_id entre dos runs fijados (BD sintética)."""
import sys, tempfile
from pathlib import Path
ROOT = Path(r"C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests" / "database"))
from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.csv_backfill import backfill_networks_and_runs, persist_training_run
from swmm_resilience.database.training_queries import load_training_frame
from swmm_resilience.config import load_config
from test_csv_backfill import _synthetic_dataset
W = Path(tempfile.mkdtemp(prefix="d05_", dir=Path(__file__).resolve().parent))
inp = W / "n.inp"; inp.write_text("[TITLE]\nx\n")
conn = connect_managed_database(W / "s.sqlite3"); apply_migrations(conn)
df = _synthetic_dataset(); info = backfill_networks_and_runs(conn, df, inp, "n")
persist_training_run(conn, df, info["run_id_by_key"], info["node_pk_by_id"], load_config(ROOT / "config.yaml"))
before = load_training_frame(W / "s.sqlite3")
a, b = 3, 5
sa = conn.execute("SELECT scenario_id FROM runs WHERE run_id=?", (a,)).fetchone()[0]
sb = conn.execute("SELECT scenario_id FROM runs WHERE run_id=?", (b,)).fetchone()[0]
conn.execute("UPDATE runs SET scenario_id=? WHERE run_id=?", (sb, a))
conn.execute("UPDATE runs SET scenario_id=? WHERE run_id=?", (sa, b)); conn.commit(); conn.close()
after = load_training_frame(W / "s.sqlite3")
m = before.merge(after, on=["shape_id", "factor_mult", "node_id"], suffixes=("_antes", "_despues"))
print("filas:", len(after), "| vol cambiado misma clave:", int((m.vol_inundacion_m3_antes != m.vol_inundacion_m3_despues).sum()))
# Resultado observado: filas: 96 | vol cambiado misma clave: 6
```

### D06 — No monotonía y cuantización (`d06_detalle.py`)

```python
"""D06 - Detalle de no-monotonía, nodo con pendiente_out NULL y cuantización (SOLO LECTURA immutable=1)."""
import sqlite3
from pathlib import Path
import pandas as pd
ROOT = Path(r"C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model")
c = sqlite3.connect(f"{(ROOT/'outputs/training_v17.sqlite3').as_uri()}?mode=ro&immutable=1", uri=True)
f = pd.read_sql_query("SELECT * FROM training_samples_v17", c)
g = f.sort_values(["shape_id", "node_id", "factor_mult"]).copy()
g["prev_inunda"] = g.groupby(["shape_id", "node_id"]).inunda.shift()
g["prev_vol"] = g.groupby(["shape_id", "node_id"]).vol_inundacion_m3.shift()
bad = g[(g.prev_inunda == 1) & (g.inunda == 0)]
print("1->0:", len(bad), "nodos:", bad.node_id.nunique(), "formas:", bad.shape_id.nunique(), "vol previo:", bad.prev_vol.value_counts().to_dict())
drops = g[g.vol_inundacion_m3 < g.prev_vol - 1e-9]
print("caídas:", len(drops), "máx:", (drops.prev_vol - drops.vol_inundacion_m3).max())
print("pendiente_out NULL:", f.loc[f.pendiente_out.isna(), "node_id"].unique(), f.loc[f.pendiente_out.isna(), "diam_max_out"].unique())
v = f.vol_inundacion_m3[f.vol_inundacion_m3 > 0]
print("positivos 1..3 m3:", int(((v >= 1) & (v <= 3)).sum()), "| valores únicos:", f.vol_inundacion_m3.nunique())
```

### D07 — Manifiesto del bench frente a la base actual (`d07_snapshot.py`)

```python
"""D07 - ¿El manifest del bench corresponde a la BD actual? (SOLO LECTURA immutable=1; copia temporal)."""
import hashlib, json, sqlite3, tempfile
from pathlib import Path
ROOT = Path(r"C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model")
W = Path(tempfile.mkdtemp(prefix="d07_", dir=Path(__file__).resolve().parent))
for name, man in (("training_v17", "af3751551ca5df98"), ("training_v17_19shapes", "4a48359e7e5b53c6")):
    src = sqlite3.connect(f"{(ROOT/'outputs'/(name+'.sqlite3')).as_uri()}?mode=ro&immutable=1", uri=True)
    p = W / f"{name}.sqlite3"; dst = sqlite3.connect(p); src.backup(dst); dst.commit(); dst.close(); src.close()
    snap = hashlib.sha256(p.read_bytes()).hexdigest()
    flat = (ROOT / "outputs/bench/prepared" / man / "manifest.json").read_text(encoding="utf-8")
    print(name, snap[:16], "en manifest:", snap in flat)
# Resultado: training_v17 a88d3b996b6f0cc8 en manifest: True ; 19shapes 69f2d6c1a35e35e9 en manifest: False
```

### Comandos auxiliares

```bash
# C1 - Dependencia del hash de red respecto del EOL (D-10)
git ls-files --eol "data/networks/chico_hydro-qx1/*.inp"           # i/lf  w/crlf
git show "HEAD:data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp" | sha256sum   # 1cad5f54…
sha256sum "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"                     # ea48571b…

# C2 - Afinidad de tipos sin STRICT (D-14)
venv/Scripts/python.exe -c "import sqlite3;c=sqlite3.connect(':memory:');c.execute('create table t(v REAL NOT NULL CHECK(v>=0))');c.execute(\"insert into t values ('abc')\");c.execute('insert into t values (1e999)');print(c.execute('select v,typeof(v) from t').fetchall())"

# C3 - Equivalencia del orden run_id frente a (shape, factor, node) (D-15)
#   SELECT r.run_id, s.shape_id, s.factor_mult FROM runs r JOIN scenarios s USING(scenario_id) ORDER BY r.run_id
#   comparar con el orden de pandas sort_values(['shape_id','factor_mult'])  -> True en ambas bases

# C4 - Planes de consulta de triggers y FK hijas (D-16), sobre una conexión immutable
#   EXPLAIN QUERY PLAN SELECT 1 FROM training_run_inputs JOIN node_features ON node_features.run_id=training_run_inputs.run_id WHERE node_features.node_pk=5
#   -> SCAN node_features ;  EXPLAIN QUERY PLAN SELECT 1 FROM oof_predictions WHERE run_id=1 AND node_pk=5 -> SCAN oof_predictions

# Tests del área (subconjunto, sin caché en el repo)
venv/Scripts/python.exe -m pytest tests/database -q -p no:cacheprovider --basetemp=<scratchpad>/pytest-db   # 208 passed, 3 deselected
```
