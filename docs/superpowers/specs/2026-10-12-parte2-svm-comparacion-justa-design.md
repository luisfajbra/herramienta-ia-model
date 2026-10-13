# Parte 2 — SVM aproximado y comparación justa XGBoost / MLP / SVM — diseño

**Fecha:** 2026-10-12
**Rama:** `fix/cierre-auditoria-hidrogramas-20260927` (worktree `optuna-tuning-familia-ganadora`).
**Precede:** `docs/superpowers/specs/2026-10-11-correcciones-auditoria-design.md` (Parte 1, cerrada).
**Decisiones del autor (2026-10-12):** (1) etiqueta: **ruta A** (no se re-simula; la base de datos no se toca ni se migra); (2) XGBoost y MLP se presentan siempre; el SVM entra como tercer candidato con su resultado; (3) esta semana: SVM corregido o su alternativa de vectores de soporte + corridas con datos reales + actualización de la presentación del 2026-10-13; (4) hacer solo entrenamientos y cosas poco dañinas o demoradas.

## 1. Objetivo
Dejar el SVM en condiciones comparables con XGBoost y MLP y obtener resultados reproducibles de los tres sobre el mismo dato (`tabular_v4_21`, `prep_id ca706e27654e0cf5`, 154.080 filas), con el mismo presupuesto de afinación y el mismo protocolo de evaluación (LeaveOneShapeOut como cifra principal), documentados para la investigación.

## 2. Estado de partida
- `svm_family.py` (commit WIP `1ecca50`): Nystroem (RBF) + `SGDClassifier(loss="log_loss")` + `SGDRegressor(loss="epsilon_insensitive")`. El regresor es un SVR lineal sobre el mapeo de kernel; el clasificador es regresión logística, **no** una máquina de vectores de soporte. El benchmark del 2026-10-05 lo dejó 3.º: RMSE 155,2 m³, F1 0,843 (contrato `v3_17`, sin afinar).
- `tune.py` ya tiene `SEARCH_SPACES` para `xgboost`, `mlp` y `svm`, y un `run_study` (GroupKFold(5) por `shape_id` interno, objetivo `end_to_end.rmse_vol_todos_nodos`). Etiqueta sus resultados como `estimacion_exploratoria_condicionada`.
- La CLI (`--bench-tune`, `--bench-evaluate`) solo prepara/lee el contrato `tabular_v3_17`; el contrato vigente `tabular_v4_21` ya está preparado en `outputs/bench/prepared_v4_21/ca706e27654e0cf5` y se usa desde scripts.
- Hardware: 64 CPU, sin GPU, torch CPU (32 hilos).

## 3. Decisiones de diseño

### 3.1 SVM: "SVM aproximado con mapeo de Nystroem"
- Clasificador: `Nystroem(kernel="rbf")` → `CalibratedClassifierCV(SGDClassifier(loss="hinge", class_weight=…), method="sigmoid", cv=3)`. Es un SVM lineal de margen (pérdida *hinge*) sobre el kernel RBF aproximado, con calibración de Platt para obtener `predict_proba` (requerido por el evaluador para el AUC). Equivale, con kernel aproximado, a `SVC(probability=True)` que usaba la versión exacta.
- Regresor: sin cambios — `SGDRegressor(loss="epsilon_insensitive")` sobre el mismo mapeo (SVR lineal sobre Nystroem), con el objetivo en `log1p`.
- El nombre interno de la familia sigue siendo `svm` (compatibilidad con el registro y el espacio de Optuna); el descriptor de preprocesado declara `svm_aproximado_nystroem` y la documentación lo llama "SVM aproximado (Nystroem + SVM lineal)". No se presenta como un SVM de kernel exacto.
- Hiperparámetros afinables: `gamma`, `n_components`, `alpha` (clasificador y regresor) y `epsilon` (regresor) — el espacio existente; se añade `C` equivalente vía `alpha` (SGD). No se amplía el espacio más allá de eso.
- Alternativa descartada: `SVC`/`SVR` exactos (O(n²)–O(n³); la corrida del 2026-10-05 acumuló más de 1 día y 15 h de CPU sin terminar).

### 3.2 Comparación justa
- Mismo dato (`prep_id ca706e27654e0cf5`), misma semilla (`ML_RANDOM_STATE`), mismo objetivo (`end_to_end.rmse_vol_todos_nodos`), **mismo número de trials Optuna** para las tres familias, mismo CV interno (GroupKFold(5) por forma).
- Presupuesto: N trials iguales para las tres, fijado tras medir el costo de un trial por familia en esta máquina (ver §5). Si el MLP no cabe en el tiempo disponible con 200 épocas, se reduce `epochs` y/o se sube `batch_size` **en las tres evaluaciones del MLP por igual** (afinación y evaluación final) y se declara.
- Sesgo declarado: los hiperparámetros se eligen con todas las formas (`estimacion_exploratoria_condicionada`), así que las cifras de LeaveOneShapeOut posteriores son optimistas **por igual** para las tres familias; sirven para comparar, no como confirmación independiente. Una afinación anidada por forma (×40) queda como trabajo futuro.
- XGBoost: además del resultado afinado se conserva el resultado con parámetros por defecto ya medido (`datos/runs/metrics_xgboost.json`), para mostrar el efecto de la afinación.

### 3.3 Evaluación y reporte
- Protocolo principal: LeaveOneShapeOut (40 folds) para el modelo afinado de cada familia, más el SVM con parámetros por defecto (barato). No se repite LOSO/GroupKFold5 para estas corridas (miden interpolación entre factores; ya documentado) salvo que sobre tiempo.
- Referencias sin aprendizaje: se reutilizan las ya calculadas (misma `prep_id`).
- Tablas con el mismo formato y procedencia que `docs/auditoria/correcciones/datos/` (`informe_evaluacion.build_tables`), en un directorio nuevo `docs/auditoria/correcciones/datos_parte2/`: comparación, matriz de confusión, banda de 1–2 m³ y por forma, más el resumen de cada estudio (mejores parámetros, trials, tiempos, curva de convergencia) y `provenance.json`.
- Documento `docs/auditoria/correcciones/06-seleccion-de-modelos.md` con "Texto para la tesis" y la regla de presentación: XGBoost y MLP siempre; el SVM como tercer candidato con su resultado.

### 3.4 Ejecución
- Script resumible `scripts/ejecutar_parte2_v4_21.py`: etapas por familia (estudio → mejores parámetros → evaluación LeaveOneShapeOut → OOF y métricas), cada etapa se persiste al terminar y se salta si ya existe. Orden: svm, xgboost, mlp (el MLP es el cuello de botella). No escribe en `outputs/models`, `outputs/bench/reports`, `outputs/bench/prepared*`, la base SQLite ni `data/`.
- Sin migraciones ni re-simulación. No se promueve ningún modelo.

## 4. Fuera de alcance
Re-simulación/regeneración del dataset (ruta B); afinación anidada; ampliar el espacio de búsqueda; GroupKFold5/LOSO para los modelos afinados; promoción de un modelo a `outputs/models`; temporal LSTM/CNN.

## 5. Plan de ejecución (compacto, por tarea)
1. **SVM aproximado** (código, TDD): `svm_family.py` con el clasificador hinge + Platt; pruebas de contrato (predict_proba en [0,1], `class_weight`, determinismo, parámetros desconocidos fallan, descriptor); espacio de Optuna sin cambios.
2. **Medición de costos** (spike, solo lectura): tiempo de un ajuste y de un trial por familia en esta máquina sobre `prepared_v4_21`; propuesta de N y de `epochs/batch_size` del MLP.
3. **Script resumible** (código, TDD en lo puro): etapas, persistencia, procedencia, tablas.
4. **Corrida real** (en segundo plano), vigilada por etapas.
5. **Documento 06** y **presentación** (actualización con lo ya existente y luego con resultados).
