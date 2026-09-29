# Integridad y validación por forma — Plan de implementación

> **Para agentes:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development` o `superpowers:executing-plans` para ejecutar este plan tarea por tarea. Los pasos usan casillas de verificación.

**Objetivo:** hacer que el banco evalúe formas de hidrograma no vistas sin reinterpretar los protocolos históricos, y evitar que el CLI promueva modelos automáticamente.

**Arquitectura:** se declara un catálogo de protocolos con su columna de agrupación. `LeaveOneShapeOut` usa `shape_id`; `LOSO` y `GroupKFold5` conservan `factor_mult`. El `PreparedDataset` persiste esa semántica y rehúsa cachés incompatibles. La regla inclusiva de inundación reutiliza `flood_label`; el benchmark solo genera resultados hasta que exista la promoción transaccional de una fase posterior.

**Tecnologías:** Python 3.11, pandas, scikit-learn, pytest y SQLite existente.

**Spec:** `docs/superpowers/specs/2026-09-27-cierre-auditoria-surrogate-hidrogramas-design.md`

## Restricciones globales

- Trabajar en `fix/cierre-auditoria-hidrogramas-20260927`; no limpiar, sobrescribir artefactos históricos, promover modelos ni ejecutar el benchmark completo.
- Aplicar TDD: cada prueba nueva debe fallar por la ausencia del comportamiento antes de modificar producción.
- Conservar `LOSO` y `GroupKFold5` como protocolos legacy agrupados por `factor_mult`.
- `LeaveOneShapeOut` debe agrupar exclusivamente por `shape_id`; este identificador nunca entra en `X`.
- Los manifiestos nuevos usan SHA-256. Un `PreparedDataset` con semántica de protocolo distinta debe rechazarse de forma explícita.
- La regla canónica es `volumen_m3 >= threshold_m3` cuando el umbral es positivo; no se reetiquetan datos correctos.
- `--bench` y `--bench-evaluate` no pueden escribir `outputs/models/`; esta fase no implementa la promoción transaccional completa.
- No hacer commits, push ni cambios a `outputs/models/` durante este plan.

---

## Mapa de archivos

- `swmm_resilience/ml/bench/schemas.py`: catálogo de protocolos y columnas persistidas de los folds.
- `swmm_resilience/ml/bench/folds.py`: división pura de sklearn y anotación de grupo retenido.
- `swmm_resilience/ml/bench/preprocess.py`: selección de columna de grupo, descriptor y validación semántica de caché.
- `swmm_resilience/ml/bench/quality.py`: uso de la regla canónica para detectar desacuerdos.
- `swmm_resilience/config.py` y `config.yaml`: protocolo principal por forma y promoción desactivada por defecto.
- `main.py`: separar preparación/evaluación de promoción explícita.
- `tests/ml/bench/test_folds.py`, `test_preprocess.py`, `test_quality.py`, `test_schemas.py`, `test_bench_config.py` y `test_bench_cli.py`: regresiones unitarias y de CLI.

## Interfaces que produce la fase

```python
# schemas.py
PROTOCOL_GROUP_COLUMNS = {
    "LOSO": "factor_mult",
    "GroupKFold5": "factor_mult",
    "LeaveOneShapeOut": "shape_id",
}
FOLD_COLUMNS = ("protocol", "fold_id", "sample_idx", "split", "group_column", "held_out_groups")

# folds.py
def group_column_for(protocol: str) -> str: ...
def build_folds(groups: pd.Series, protocol: str) -> pd.DataFrame: ...

# preprocess.py
def validate_prepared_semantics(prepared: PreparedDataset, protocols: Sequence[str]) -> None: ...
```

`held_out_groups` será un JSON canónico de una lista de valores convertidos a texto y ordenados. Para `LeaveOneShapeOut` contiene exactamente un `shape_id`; para `GroupKFold5` puede contener más de uno. El manifiesto guardará `protocol_definitions` con el mismo mapeo. La identidad usará una huella canónica del frame de entrenamiento realmente consumido; el checksum físico del snapshot SQLite capturado se persistirá por separado como evidencia, sin convertir su distribución de páginas en la identidad de caché.

### Task 1: Declarar protocolos y folds por forma

**Archivos:**

- Modificar: `swmm_resilience/ml/bench/schemas.py`, `swmm_resilience/ml/bench/folds.py`
- Modificar: `tests/ml/bench/test_folds.py`, `tests/ml/bench/test_schemas.py`

**Consume:** `KEY_COLUMNS` ya contiene `factor_mult` y `shape_id`.

**Produce:** `PROTOCOLS` con `LeaveOneShapeOut`, `PROTOCOL_GROUP_COLUMNS`, `group_column_for()` y folds con procedencia explícita.

- [x] **Paso 1: escribir la prueba que describe el aislamiento temporal.**

```python
def test_leave_one_shape_out_holds_every_factor_of_one_shape_out():
    keys = pd.DataFrame({
        "shape_id": ["a"] * 4 + ["b"] * 4,
        "factor_mult": [1.0, 1.0, 2.0, 2.0] * 2,
    })
    folds = build_folds(keys["shape_id"], "LeaveOneShapeOut")
    for _, fold in folds.groupby("fold_id"):
        train = fold.loc[fold.split == "train", "sample_idx"]
        test = fold.loc[fold.split == "test", "sample_idx"]
        assert not set(keys.shape_id.iloc[train]) & set(keys.shape_id.iloc[test])
        assert keys.shape_id.iloc[test].nunique() == 1
        assert set(keys.factor_mult.iloc[test]) == {1.0, 2.0}
        assert json.loads(fold.held_out_groups.iloc[0]) == [keys.shape_id.iloc[test[0]]]
```

- [x] **Paso 2: ejecutar la prueba y comprobar que falla porque el protocolo aún no existe.**

Ejecutar: `venv\\Scripts\\python.exe -m pytest tests/ml/bench/test_folds.py -q`

Esperado: fallo que nombre `LeaveOneShapeOut` como protocolo desconocido o la ausencia de las columnas de procedencia.

- [x] **Paso 3: implementar el catálogo y anotar cada fold.**

```python
PROTOCOL_GROUP_COLUMNS = {
    "LOSO": "factor_mult",
    "GroupKFold5": "factor_mult",
    "LeaveOneShapeOut": "shape_id",
}

def group_column_for(protocol: str) -> str:
    try:
        return PROTOCOL_GROUP_COLUMNS[protocol]
    except KeyError as exc:
        raise ValueError(f"Protocolo desconocido: {protocol!r}. Opciones: {PROTOCOLS}") from exc
```

En `build_folds`, validar que `groups.name == group_column_for(protocol)`, serializar los grupos de test de cada fold con `json.dumps(sorted(map(str, ...)), ensure_ascii=False)`, y repetir `group_column`/`held_out_groups` en las filas train y test del fold. Mantener los splitters actuales: `LeaveOneGroupOut` para LOSO y LeaveOneShapeOut; `GroupKFold(5)` para GroupKFold5.

- [x] **Paso 4: ejecutar pruebas focalizadas.**

Ejecutar: `venv\\Scripts\\python.exe -m pytest tests/ml/bench/test_folds.py tests/ml/bench/test_schemas.py -q`

Esperado: todas pasan, incluidos los tests legacy de 25 folds por factor y los nuevos de 2 folds por forma.

### Task 2: Preparación reproducible y rechazo de caché semánticamente incorrecta

**Archivos:**

- Modificar: `swmm_resilience/ml/bench/preprocess.py`, `swmm_resilience/ml/bench/schemas.py`
- Modificar: `tests/ml/bench/test_preprocess.py`, `tests/ml/bench/test_bench_cli.py`

**Consume:** `group_column_for()` y `PROTOCOL_GROUP_COLUMNS` de la tarea 1.

**Produce:** `prepare_dataset()` construye cada protocolo contra su propia columna; el manifiesto y `prep_id` describen los grupos; `resolve_prepared()` rechaza manifiestos antiguos o incompatibles.

- [x] **Paso 1: escribir pruebas de preparación por forma y de incompatibilidad.**

```python
def test_prepare_dataset_records_each_protocol_group_column(sql_training_db, tmp_path):
    prepared = prepare_dataset(
        sql_training_db, protocols=("LOSO", "LeaveOneShapeOut"),
        flood_threshold_m3=1.0, output_dir=tmp_path / "prepared",
    )
    assert prepared.manifest["protocol_definitions"] == {
        "LOSO": {"group_column": "factor_mult"},
        "LeaveOneShapeOut": {"group_column": "shape_id"},
    }
    shape_rows = prepared.folds.query("protocol == 'LeaveOneShapeOut'")
    assert set(shape_rows.group_column) == {"shape_id"}

def test_resolve_prepared_rejects_manifest_without_protocol_definitions(prepared, tmp_path):
    manifest_path = tmp_path / "prepared" / prepared.prep_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("protocol_definitions")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="semántica.*protocol"):
        resolve_prepared(tmp_path / "prepared")
```

- [x] **Paso 2: ejecutar estas pruebas y comprobar el fallo esperado.**

Ejecutar: `venv\\Scripts\\python.exe -m pytest tests/ml/bench/test_preprocess.py tests/ml/bench/test_bench_cli.py -q`

Esperado: los tests nuevos fallan porque `build_all_folds()` recibe una sola serie de `factor_mult` y no existe validación del manifiesto.

- [x] **Paso 3: implementar un constructor de folds por protocolo y manifiesto verificable.**

```python
def build_all_folds(keys: pd.DataFrame, protocols: Sequence[str]) -> pd.DataFrame:
    return pd.concat(
        [build_folds(keys[group_column_for(protocol)], protocol) for protocol in protocols],
        ignore_index=True,
    )

def protocol_definitions(protocols: Sequence[str]) -> dict[str, dict[str, str]]:
    return {protocol: {"group_column": group_column_for(protocol)} for protocol in protocols}
```

Capturar la fuente mediante una conexión SQLite de solo lectura y `Connection.backup()` hacia un temporal; las migraciones y el cargador se ejecutarán solo sobre esa copia. Guardar `source_capture_method`, `source_snapshot_sha256` (checksum de la copia antes de migrar) y una `training_frame_sha256` canónica del frame ordenado que realmente se consume. Incluir la huella lógica, contrato, parámetros y `protocol_definitions` en `prep_id`; no usar el checksum físico del archivo principal como identidad. Verificar en `PreparedDataset.load()` la identidad del manifiesto, la semántica de protocolo y que cada fold coincide exactamente con los grupos y particiones reconstruidos desde `keys`. `resolve_prepared()` debe aceptar un parámetro `protocols` y llamar esa validación; `main._run_bench()` le pasará `tuple(config.bench.protocols)` en train/evaluate. No eliminar ni modificar cachés antiguos: rechazarlos con un mensaje que indique ejecutar `--bench-prepare`.

- [x] **Paso 4: ejecutar el conjunto de preparación y CLI.**

Ejecutar: `venv\\Scripts\\python.exe -m pytest tests/ml/bench/test_preprocess.py tests/ml/bench/test_bench_cli.py tests/ml/bench/test_schemas.py -q`

Esperado: pasa y ningún test crea archivos fuera de su `tmp_path`.

### Task 3: Unificar la auditoría de umbral inclusivo

**Archivos:**

- Modificar: `swmm_resilience/ml/bench/quality.py`
- Modificar: `tests/ml/bench/test_quality.py`

**Consume:** `swmm_resilience.extraction.labels.flood_label(volumes, threshold_m3)`.

**Produce:** `threshold_disagreements` compara la etiqueta persistida contra la misma regla de extracción.

- [x] **Paso 1: escribir una regresión para el volumen exactamente en el umbral.**

```python
def test_report_does_not_flag_volume_equal_to_positive_threshold():
    keys, X, _, _ = _frames()
    y_reg = pd.Series([1.0] + [0.0] * (len(keys) - 1), name="vol_inundacion_m3")
    y_clf = pd.Series([1] + [0] * (len(keys) - 1), name="inunda")
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)
    assert report["threshold_disagreements"] == 0
```

- [x] **Paso 2: ejecutar la prueba y comprobar que falla con un desacuerdo.**

Ejecutar: `venv\\Scripts\\python.exe -m pytest tests/ml/bench/test_quality.py -q`

Esperado: fallo porque el reporte actual usa `y_reg > flood_threshold_m3`.

- [x] **Paso 3: reemplazar la comparación duplicada por la función canónica.**

```python
from ...extraction.labels import flood_label

expected_labels = flood_label(y_reg, flood_threshold_m3)
disagreements = int((expected_labels != y_clf.astype(int)).sum())
```

No cambiar `y_clf` ni `y_reg`; este módulo solo informa calidad. Mantener el comportamiento `threshold <= 0` definido por `flood_label`.

- [x] **Paso 4: ejecutar regresiones de labels y calidad.**

Ejecutar: `venv\\Scripts\\python.exe -m pytest tests/test_labels.py tests/ml/bench/test_quality.py tests/ml/bench/test_stage1_imports.py -q`

Esperado: pasa sin introducir imports de sklearn, torch o xgboost en la etapa 1.

### Task 4: Desactivar promoción implícita y seleccionar LeaveOneShapeOut como protocolo principal

**Archivos:**

- Modificar: `swmm_resilience/config.py`, `config.yaml`, `main.py`
- Modificar: `tests/ml/bench/test_bench_config.py`, `tests/ml/bench/test_bench_cli.py`

**Consume:** el catálogo de protocolos de la tarea 1 y los datasets semánticamente válidos de la tarea 2.

**Produce:** la configuración usa `LeaveOneShapeOut` primero y `promote: none`; `--bench` prepara/entrena/evalúa/reporta, pero no toca `outputs/models`; `--bench-promote` exige una familia explícita distinta de `none` y conserva la promoción legacy aislada hasta su reemplazo posterior.

- [x] **Paso 1: escribir las pruebas de no promoción.**

```python
def test_full_bench_run_writes_ranking_without_promoting(sql_training_db, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _make_config(
        inp_path=tmp_path / "network.inp", db_path=sql_training_db,
        families=["xgboost"], promote="none", protocols=("LeaveOneShapeOut",),
    )
    main._run_bench(_make_args(bench=True, models="xgboost"), config)
    assert (tmp_path / "outputs/bench/reports/ranking.csv").exists()
    assert not (tmp_path / "outputs/models").exists()

def test_bench_promote_rejects_none(sql_training_db, tmp_path):
    config = _make_config(inp_path=tmp_path / "network.inp", db_path=sql_training_db,
                          families=["xgboost"], promote="none")
    with pytest.raises(SystemExit, match="familia explícita"):
        main._run_bench(_make_args(bench_promote=True), config)
```

- [x] **Paso 2: ejecutar las pruebas y comprobar que fallan porque `--bench` promociona con `auto`.**

Ejecutar: `venv\\Scripts\\python.exe -m pytest tests/ml/bench/test_bench_config.py tests/ml/bench/test_bench_cli.py -q`

Esperado: falla al no reconocer `LeaveOneShapeOut` o al encontrar `outputs/models` tras `--bench`.

- [x] **Paso 3: ajustar configuración y orquestación.**

En `config.yaml`, establecer `protocols: ["LeaveOneShapeOut", "LOSO", "GroupKFold5"]`, cambiar los desempates a `classifier.csi` y `end_to_end.mean_abs_total_volume_error_m3` solo cuando esas métricas existan en la fase D; en esta fase conservar los desempates actuales para no declarar métricas inexistentes. Establecer `promote: "none"`.

En `main._run_bench`, definir `run_all = args.bench`, mantener la evaluación dentro de `run_all`, y entrar a promoción solo si `args.bench_promote` es verdadero. Si `config.bench.promote in {"none", "auto"}`, abortar `--bench-promote` con un mensaje que exija una familia explícita. No añadir un camino que seleccione una familia a partir del ranking.

- [x] **Paso 4: ejecutar regresiones de configuración y CLI.**

Ejecutar: `venv\\Scripts\\python.exe -m pytest tests/ml/bench/test_bench_config.py tests/ml/bench/test_bench_cli.py -q`

Esperado: pasa; el full-bench reducido solo deja resultados bajo `outputs/bench` en el `tmp_path`.

### Task 5: Verificación de la fase A

**Archivos:**

- No crear archivos de producción adicionales.
- Modificar solo pruebas/documentación si una prueba existente contradice la semántica nueva de forma.

**Consume:** las cuatro tareas anteriores.

**Produce:** evidencia repetible de que la fase A no filtró `shape_id`, no alteró etiquetas y no promovió modelos.

- [x] **Paso 1: ejecutar la batería del banco.**

Ejecutar: `venv\\Scripts\\python.exe -m pytest tests/ml/bench -q`

Esperado: salida con código 0. Si falla, abrir una prueba de regresión que reproduzca la causa antes de cambiar producción.

- [x] **Paso 2: ejecutar las regresiones de la etiqueta y carga de configuración.**

Ejecutar: `venv\\Scripts\\python.exe -m pytest tests/test_labels.py tests/ml/bench/test_bench_config.py -q`

Esperado: salida con código 0.

- [x] **Paso 3: ejecutar un smoke de preparación aislado.**

Ejecutar: `venv\\Scripts\\python.exe main.py --bench-prepare --config config.yaml`

Esperado: crea un `PreparedDataset` nuevo bajo `outputs/bench/prepared/` con `LeaveOneShapeOut` y no escribe en `outputs/models/`. No ejecutar `--bench`, `--bench-train` ni `--bench-evaluate` contra la base real en esta fase.

- [x] **Paso 4: revisar el diff y el estado de Git.**

Ejecutar: `git diff --check` y `git status --short`.

Esperado: no hay errores de espacios y las modificaciones pertenecen a esta fase. No hacer commit: el usuario pidió preservar el control de integración.

## Autorrevisión del plan

- Cobertura: C2 se cubre con tareas 1–2; el defecto `>`/`>=` de M5/MO3 con la tarea 3; la promoción automática con la tarea 4. Features, drenaje, parser ponding, métricas y promoción transaccional pertenecen a fases posteriores y no se simulan aquí.
- Consistencia: `LeaveOneShapeOut`, `PROTOCOL_GROUP_COLUMNS`, `group_column_for()` y `protocol_definitions` son los nombres usados por todas las tareas.
- Sin placeholders: los valores, rutas, tests y comandos están definidos; los cambios hidráulicos se excluyen de forma explícita para evitar implementarlos sin el piloto físico exigido por la spec.
