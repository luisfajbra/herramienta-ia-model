# Revisión de la rama `fix/test-suite-contracts-2026-10-27` (2026-11-04)

Documento para el siguiente agente que trabaje en esta rama. Recoge lo que se
revisó, qué problemas aparecieron, por qué ocurrían, qué se corrigió y qué queda
pendiente. Las correcciones están en los commits `ea3a7a56` (tests de migraciones),
`813614ba` (etiqueta `usage`) y el de este documento. No se ha hecho push.

## 1. Alcance de la revisión

- Comparación: `main...HEAD` (90 commits, 559 archivos, ~133k líneas; ~121 MB
  añadidos, la mayoría `.pptx` y artefactos JSON en `docs/`).
- Se revisó a fondo el código de librería en `swmm_resilience/`: ML bench
  (`evaluate.py`, `tune.py`, `train.py`, `schemas.py`, `preprocess.py`,
  `metrics.py`, `folds.py`, `promote.py`, `frozen_transfer.py`,
  `models/mlp_family.py`), etiquetas (`extraction/labels.py`), simulación
  (`runner.py`, `batch.py`, `swmm_api_io.py`, `effective_inflow.py`,
  `timeseries_scenario.py`), calidad (`validation/scenario_quality.py`,
  `database/hydraulic_quality.py`, `database/training_queries.py`, migraciones
  009–011) y `main.py`.
- Se revisó solo superficialmente: scripts de figuras/presentaciones en
  `scripts/`, `database/label_free_b_staging.py` (salvo el punto de llamada a
  `register_transfer`) y `validation/network_admission.py`.

### Cómo correr la suite

```bash
python -m pytest -q -p no:cacheprovider --basetemp="<carpeta temporal fuera del repo>"
```

- `pytest-xdist` **no** está instalado: no usar `-n`.
- La suite completa tarda ~8 minutos.
- Usar siempre `--basetemp` fuera del repo. Ya hay más de 50 carpetas `.pytest-*`
  sin seguimiento en la raíz, y al menos un `tmp` de pytest dentro de
  `outputs/experiments/frozen_pair_rerun_20261028/.../_tmp_preparation/`.

Estado antes de las correcciones: **8 failed, 1658 passed, 1 skipped**.

## 2. Problemas corregidos

### 2.1 Ocho tests fallaban por un conteo de migraciones desactualizado

**Síntoma.** Fallaban estos tests:

- `tests/database/test_maintenance_v17.py::test_checkpoint_and_backup_creates_standalone_database`
- `tests/database/test_migrations_v17.py::test_initial_migration_creates_expected_schema`
- `tests/database/test_migrations_v17.py::test_migrations_are_idempotent`
- `tests/database/test_model_integrity_migration_v17.py::test_fresh_database_applies_all_migrations_idempotently`
- `tests/database/test_upgrade_v17.py::test_upgrade_backs_up_before_applying_005`
- `tests/database/test_upgrade_v17.py::test_upgrade_applies_all_migrations_from_a_fresh_database`
- `tests/packaging/test_wheel_migrations_v17.py::test_wheel_exposes_all_nine_migrations_and_validator`
- `tests/test_cli_sql_reads.py::test_migrate_db_is_the_explicit_schema_write_command`

Con errores del tipo `assert 11 == 9` o con tablas «extra» `run_source_artifacts`
y `run_source_artifacts_legacy_010`.

**Causa.** La rama añadió dos migraciones nuevas:

- `swmm_resilience/database/sql/010_run_source_artifacts.sql` (commit `c3790ab3`)
  crea `run_source_artifacts`.
- `swmm_resilience/database/sql/011_run_source_artifact_identity.sql` (commit
  `34e0e435`) renombra esa tabla a `run_source_artifacts_legacy_010` y crea una
  `run_source_artifacts` nueva con identidad inmutable.

Esos commits no actualizaron los tests que fijan el catálogo de migraciones. Los
tests seguían esperando 9 migraciones (hasta `009_run_hydraulic_quality`). **El
código de las migraciones está bien; los tests estaban desactualizados.**

**Solución aplicada** (solo en tests):

| Archivo | Cambio |
|---|---|
| `tests/database/test_maintenance_v17.py` | `COUNT(*) FROM schema_migrations` 9 → 11 |
| `tests/database/test_migrations_v17.py` | conteo 9 → 11 (2 sitios); `EXPECTED_TABLES` añade `run_source_artifacts` y `run_source_artifacts_legacy_010` |
| `tests/database/test_model_integrity_migration_v17.py` | lista esperada añade `(10, "run_source_artifacts")` y `(11, "run_source_artifact_identity")` |
| `tests/database/test_upgrade_v17.py` | `range(1, 10)` → `range(1, 12)` (2 sitios) |
| `tests/packaging/test_wheel_migrations_v17.py` | lista de `.sql` del wheel añade 010 y 011; el test se renombra a `test_wheel_exposes_all_migrations_and_validator` |
| `tests/test_cli_sql_reads.py` | `MAX(version)` 9 → 11 |

`EXPECTED_INDEXES` no cambió: las migraciones 010 y 011 solo crean tablas y
triggers, no índices.

**Cómo evitarlo la próxima vez.** Al añadir una migración `0NN_*.sql`, actualizar
en el mismo commit todos los sitios que fijan el catálogo. Para encontrarlos:

```bash
grep -rnE "schema_migrations|range\(1, [0-9]+\)|_v17_initial.sql|EXPECTED_TABLES" tests/
```

### 2.2 `registration.json` de transferencias reales decía `synthetic_only_unbound_sources`

**Síntoma.** Toda preregistración creada por
`swmm_resilience/ml/bench/frozen_transfer.py::register_transfer` guardaba
`"usage": "synthetic_only_unbound_sources"`, también la del flujo real
(`database/label_free_b_staging.py::register_and_predict_staged_transfer`), que
es el que produjo las evaluaciones de Chico Norte.

**Causa.** La etiqueta estaba fija en el diccionario de `registration`. Venía del
commit `a6e5832e`, cuando la función solo se usaba en tests sintéticos; después
se conectó al flujo real sin parametrizarla.

**Qué NO estaba mal** (aclaración respecto de la revisión inicial):

- El puntaje real, `score_transfer` → `verified_scores.json`, ya escribe
  `"usage": "verified_staged_sources"`. `scripts/figures_part2/make_chico_norte_transfer_figures.py`
  lo exige. Las cifras reportadas no se ven afectadas.
- `_score_synthetic_transfer` → `synthetic_scores.json` es solo para tests y su
  etiqueta sintética es correcta.

**Solución aplicada.**

- `register_transfer(..., *, usage='synthetic_only_unbound_sources')` acepta un
  argumento `usage` validado contra `REGISTRATION_USAGES =
  ('synthetic_only_unbound_sources', 'verified_staged_sources')`. Un valor
  desconocido lanza `ValueError` antes de crear ningún directorio.
- El valor por defecto sigue siendo sintético, a propósito: una llamada directa
  nunca se presenta como verificada.
- `register_and_predict_staged_transfer` pasa `usage="verified_staged_sources"`,
  porque ya verificó el paquete de fuentes B antes de registrar.
- Tests de regresión:
  - `tests/database/test_label_free_b_staging.py::test_staged_registration_reconstructs_plan_from_sqlite_then_predicts`
    comprueba que el flujo staged pasa `usage="verified_staged_sources"`. El mock
    `fake_register` ahora acepta `**kwargs`.
  - `tests/ml/bench/test_frozen_transfer_experiment.py::test_predictions_sealed_before_labels_and_immutable`
    comprueba que el valor por defecto es sintético.
  - `tests/ml/bench/test_frozen_transfer_experiment.py::test_register_transfer_rejects_unknown_usage`
    (nuevo).

**Artefactos ya existentes: NO editarlos.** Todas las preregistraciones creadas
antes del commit `813614ba` conservan la etiqueta antigua. Son carpetas locales
sin seguimiento en git; a 2026-11-04 hay al menos 10 bajo `outputs/` (ninguna
en `docs/`). Por ejemplo:

- `outputs/bench/multinetwork/20261030T005341Z-813323e3ef75/A_to_B_frozen/registration.json`
- `outputs/bench/multinetwork/20261030T011406Z-813323e3ef75/A_to_B_frozen/registration.json`
- `outputs/experiments/chico_norte_novel_hydrograph_eval_20261102/run_20261103T034108Z/predictions/{mlp,xgboost}/.../A_to_B_frozen/registration.json`
- `outputs/experiments/piloto_seleccion_swmm_20261103/selected_predictions/{mlp,xgboost}/.../A_to_B_frozen/registration.json`
- `outputs/experiments/piloto_seleccion_inundado_20261103/selected_predictions/.../A_to_B_frozen/registration.json`

Para listarlas todas:
`grep -rl synthetic_only_unbound_sources outputs --include=registration.json`.

Cada `registration.json` está sellado por `registration_seal.json`, y su SHA-256
se encadena en `prediction_seal.json` y `verified_scores.json`. Editarlos rompería
la verificación. La etiqueta antigua es solo cosmética: la fuente de verdad sobre
la verificación es `verified_scores.json["usage"] == "verified_staged_sources"`.
Si un revisor pregunta, basta con explicarlo; no hace falta regenerar nada.

### Verificación tras las correcciones

- Tests afectados (migraciones, transferencia, staging): todos pasan.
- Suite completa: 1668 passed, 0 failed (ver la sección 5).

## 3. Problemas detectados y NO corregidos (decisión pendiente del usuario)

### 3.1 La documentación del filtro de continuidad no coincide con el código

- **Dónde.** `swmm_resilience/database/training_queries.py`
  (`load_training_samples`, docstring de `load_training_frame`) y
  `swmm_resilience/ml/bench/preprocess.py` (docstring de `prepare_dataset`).
- **Qué pasa.** Los docstrings dicen que con `max_continuity_error_pct=None` «no
  se aplica filtro de continuidad». Pero `quality_filter` en
  `load_training_samples` **siempre** exige una fila en `run_hydraulic_quality`
  con `eligible = 1` y `continuity_error_pct IS NOT NULL`.
- **Consecuencia.** Los runs históricos sin evaluación de calidad (la migración
  009 no hace backfill a propósito) quedan fuera del entrenamiento. Con una base
  antigua, el frame de entrenamiento puede salir vacío o reducido. Parece
  intencional (commit `8a9f933b fix: validate only QA-eligible training runs`).
- **Solución sugerida.** Corregir solo los docstrings para que describan el
  comportamiento real; no cambiar la lógica.

### 3.2 `preprocess` falla con un error poco claro en bases sin migrar

- **Dónde.** `swmm_resilience/ml/bench/preprocess.py::_source_provenance`, en la
  consulta `SELECT ... FROM run_hydraulic_quality`.
- **Qué pasa.** Si la base no tiene la migración 009, sqlite lanza
  `OperationalError: no such table` en vez de pedir `--migrate-db`.
- **Solución sugerida.** Comprobar antes la versión de `schema_migrations` y dar
  un mensaje claro. Prioridad baja.

### 3.3 Cambios de comportamiento a propósito (no son errores, pero conviene conocerlos)

- `simulation/swmm_api_io.py::get_node_inflow_profiles` y
  `simulation/batch.py::_base_inflow_scenario` usan
  `effective_inflow.build_effective_inflow`, que **falla** si `FLOW_UNITS` no es
  `LPS`, si un INFLOW tiene patrón o si una serie temporal se comparte entre
  nodos. Chico Sur y Chico Norte cumplen esas condiciones; una red en CMS o CFS
  ya no corre.
- En modo `steady`, el runner ahora escala solo el `baseline` del INFLOW
  (`at_scaled_baseline`); en `timeseries` escala solo la componente de la serie
  (`at_scaled_component`). Antes se multiplicaba todo.
- `extraction/labels.py` es ahora *fail-closed*: un `.rpt` ausente o malformado
  lanza `FloodingReportError` en vez de producir etiquetas a cero. En `main.py`,
  `_extract_labels_for_quality` convierte ese error en un escenario no elegible
  (`flooding_report_invalid`), que la puerta de calidad excluye.

### 3.4 Peso del repositorio y basura local

- La rama añade unos 121 MB, sobre todo 8 `.pptx` de 6 a 15 MB en la raíz y
  `docs/superpowers/reports/artifacts/2026-10-30-chico-norte/*-verified_scores.json`
  (~1,6 MB cada uno). Valorar Git LFS o sacarlos del repo antes de fusionar.
- En la raíz hay muchas carpetas sin seguimiento: `.pytest-*`, `tmp*`,
  `_ptest*`, `frozen-runner-test-*`, `tmp_review_task5c*`. Se pueden borrar o
  añadir a `.gitignore`. No se tocaron porque el usuario no lo pidió.

## 4. Revisado y correcto (no reabrir sin motivo)

- **Parser del `.rpt`** (`extraction/labels.py`, `swmm_api_io.read_node_flooding_summary`):
  elige la columna de volumen total aunque exista la de encharcamiento, convierte
  10^6 L → m³ (×1000), rechaza nodos duplicados o volúmenes no finitos y exige la
  marca de fin de sección. Si `swmm_api` falla, cae al parser de texto estricto,
  que deja constancia en el manifiesto (`parser: strict_text_v1`).
- **Evaluación anidada LeaveOneNetworkOut** (`ml/bench/evaluate.py::evaluate_nested_networks`):
  el tuning recibe una vista (`_fold_training_view`) que solo contiene el outer
  train, sin fuga de la red de prueba.
- **Ensamble de 5 semillas del MLP**: entrenamiento (`SeedEnsembleRegressor`),
  evaluación (`evaluate_mlp_seeds`, anidada) y transferencia promedian el volumen
  en m³ después de `expm1` y del recorte a 0, y la probabilidad por media
  aritmética con umbral 0,5. Es coherente en los tres sitios.
- **`run_hydraulic_quality`**: `run_id` es `PRIMARY KEY` (una decisión por run)
  y hay triggers que impiden UPDATE y DELETE. El `EXISTS (... eligible = 1)` de
  `training_queries` no puede elegir una decisión antigua.
- **Nombres de manifiestos en `batch.run_batch`**: `<scenario_id>.inp`,
  `.drainage_state.json` e `.initial_condition.json` comparten el mismo stem, que
  es lo que `main.py` espera al leerlos.

## 5. Estado final

- Archivos modificados (commits `ea3a7a56` y `813614ba`, sin push):
  - `swmm_resilience/ml/bench/frozen_transfer.py`
  - `swmm_resilience/database/label_free_b_staging.py`
  - Los 6 archivos de tests de la sección 2.1
  - `tests/database/test_label_free_b_staging.py`
  - `tests/ml/bench/test_frozen_transfer_experiment.py`
  - Este documento
- Suite completa tras las correcciones (2026-11-04): **1668 passed, 0 failed,
  3 deselected** en ~7 minutos. Antes: 8 failed, 1658 passed.
