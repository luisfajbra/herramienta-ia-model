# Seguimiento de los 111 hallazgos de la auditoría del 10 de octubre

Fecha: 2026-10-25. Worktree confirmado: `optuna-tuning-familia-ganadora`, commit `78e551a`. Véase el [informe nuevo](2026-10-25-revision-generalizacion.md) para evidencias, nuevos hallazgos y decisión sobre entrenamiento entre redes.

**Criterio:** corregido significa que el mecanismo concreto se comprobó en el alcance escrito; parcial significa que hay mejora pero no cierre completo; abierto confirmado se apoya en lectura del mecanismo vigente y/o reproducción; no revalidado significa que no hay verificación nueva suficiente. No revalidado no significa corregido ni demuestra que siga fallando. Los tests aprobados no cierran automáticamente un hallazgo.

Los IDs D de esta matriz corresponden al anexo de datos/SQL, no a las 34 discrepancias documentales que reutilizan el prefijo D en el anexo 04.

| Estado | Cantidad |
|---|---:|
| Corregido en el alcance indicado | 10 |
| Parcial | 28 |
| Abierto confirmado | 34 |
| No revalidado | 39 |
| **Total** | **111** |

La auditoría de 2026-10-10 se hizo sobre otro checkout. Por tanto, esta matriz compara estados; no atribuye todas las correcciones a commits posteriores a esa fecha.

| ID | Estado | Tema | Verificación o límite actual |
|---|---|---|---|
| H-01 | Abierto confirmado | InitFlow heredado | 160 conductos con caudal inicial; seis controles nuevos. N-10 del informe. |
| H-02 | Corregido en el alcance indicado | Falta de volumen en las entradas | v4_21 incorpora volumen local y tributario; verificación de columnas y OOF. No cierra M-02. |
| H-03 | Parcial | Uniformidad espacial y desfases | Variables tributarias por nodo añadidas; catálogo de entrenamiento aún uniforme/sincrónico. No hay validación espacial nueva. |
| H-04 | Corregido en el alcance indicado | Continuidad no persistida | Ahora hay valores en 1000/1000 runs, 37 excluidos. El criterio de admisión completo sigue pendiente en N-08. |
| H-05 | Parcial | Rampa de aporte durante el drenaje | Cierre explícito en formas adicionales/CSV; pruebas y tres controles originales coinciden con SQL. Rutas y admisión no unificadas. |
| H-06 | Parcial | Resolución/semántica de inundación | Resolución sigue en 1 m³; estudio posterior aclaró el umbral efectivo. Decisión de semántica pendiente; no se declara ruido. |
| H-07 | Abierto confirmado | Parser textual equivocado | Seis RPT nuevos: fallback suma cero pese a inundación; N-09. |
| H-08 | Abierto confirmado | Capacidad aguas arriba | Sigue siendo upstream_capacity_lps; 54891 filas sin capacidad en el preparado. N-10. |
| H-09 | Abierto confirmado | Pendiente del enlace al outfall | Extractor solo usa cotas de junctions; 963 nulos en pendiente_out en el preparado. |
| H-10 | Abierto confirmado | Horizonte distinto para forma base | run_batch llama al runner simple; no aplica el mismo control de cola. No se re-simuló base extendida en esta revisión. |
| H-11 | Abierto confirmado | Unidades y semántica de INFLOWS | Pico ignora escala y basal; entrenamiento sin preflight de unidades. N-06. |
| H-12 | Abierto confirmado | Dominio topológico limitado | Solo CONDUITS y suma por ancestros; reproducción adicional de pérdida de tuberías paralelas. N-05. |
| H-13 | No revalidado | Duraciones/estadísticas legacy | No se reejecutó la cadena legacy ni se validaron sus estadísticas temporales. |
| H-14 | Abierto confirmado | Redondeo temporal a minutos | _hours_to_hhmm sigue redondeando; impacto numérico antiguo no vuelto a medir. |
| M-01 | Corregido en el alcance indicado | Protocolos y ausencia de referencias | Referencias presentes, interpretación por protocolo, LeaveOneShapeOut principal; tests del banco aprobados. |
| M-02 | Parcial | Generalización del volumen por forma | OOF ajustados recalculados; persiste concentración del error y no hay prueba de otras redes. N-01/N-04. |
| M-03 | Corregido en el alcance indicado | Agregación de clasificación del banco | Pooled y dispersión implementados; 154080 predicciones únicas por modelo. Cierre limitado al banco. |
| M-04 | Parcial | Ranking e incertidumbre | Existe RMSE pooled y bootstrap documentado, pero config/tuning optimizan media por fold. Bootstrap antiguo no reejecutado. |
| M-05 | Parcial | Subajuste y sesgo de retransformación | Tuning mejora desempeño medido; log1p/expm1 permanece. No se aisló el sesgo de retransformación con nuevo experimento. |
| M-06 | Parcial | Comparación con parámetros heredados | 20 trials por familia en parte 2 para XGBoost/MLP/SVM aproximado; no comparación optimizada de las cinco familias originales. |
| M-07 | Parcial | Sobrescritura/promoción | SHA actuales correctos y guarda de contrato; otras rutas y metadata siguen incompletas. N-02/N-11. |
| M-08 | Corregido en el alcance indicado | OOF no persistidos | OOF y procedencia disponibles y usados en recálculo independiente. |
| M-09 | Abierto confirmado | Umbral y probabilidades | Continúa umbral 0.5 sin calibración validada de XGBoost/MLP; pesos de clase no convierten scores en riesgo calibrado. |
| M-10 | Parcial | Métricas insuficientes | Confusión, especificidad y desglose por forma añadidos; PR-AUC/MCC/balanced accuracy no en núcleo del banco. |
| M-11 | Parcial | Independencia/incertidumbre | Hay bootstrap por forma documentado, pero una red y una semilla; no prueba entre redes. |
| M-12 | Abierto confirmado | Ablación/SHAP de contrato legacy | feature_analysis conserva 17 variables, LOSO por factor y TreeExplainer; no representa al MLP v4_21. |
| M-13 | Parcial | Evaluación en muestra | Avisos y nuevas figuras OOF, sin revisión de todas las rutas ni regeneración de figuras históricas. |
| M-14 | Abierto confirmado | Nulos estructurales imputados | Mediana sin indicador estructural; 57 nodos de cabecera sin capacidad. N-10. |
| M-15 | Abierto confirmado | Tuning no anidado | run_study usa todas las formas; marcado exploratorio, no corregido metodológicamente. N-04. |
| M-16 | No revalidado | Surrogate temporal | LSTM/CNN fuera de la comprobación empírica de esta revisión centrada en MLP/XGBoost. |
| M-17 | Parcial | Cobertura de regresiones | 1035 pruebas seleccionadas pasan; no se repitió campaña de mutaciones. Nuevas reproducciones hallan vacíos. |
| M-18 | No revalidado | Nombres y restos legacy | No se realizó nuevo inventario de huérfanos del paquete completo. |
| M-19 | Parcial | Resolución de targets | Ver H-06; estudio actualizado, representación cuantizada permanece. |
| M-20 | Corregido en el alcance indicado | Validador del frame de inferencia | predict_network y ScenarioPredictor llaman TABULAR_V4_21.validate_frame; tests seleccionados aprobados. |
| M-21 | Abierto confirmado | Procedencia SQL legacy | persist_training_run conserva metadata vacía y evaluación por factor; N-07 añade brecha de v4_21. |
| M-22 | Parcial | Determinismo de MLP | Semillas y limitación de hilos en workers; determinismo entre máquinas/semillas no demostrado por esta revisión. |
| D-01 | Abierto confirmado | Runs referenciados mutables | UPDATE de continuidad en copia en memoria aceptado; trigger aún solo protege status/network_id. N-07. |
| D-02 | Parcial | Evidencia SWMM ausente | Continuidad corregida; backfill aún marca COMPLETE y usa tiempos de importación, sin cadena completa de evidencia. |
| D-03 | Abierto confirmado | Metadata de modelos SQL vacía | persist_training_run conserva JSON vacíos y contrato hash de literal SQL; ruta legacy no corregida. |
| D-04 | Parcial | Cuantización del volumen | 46236/46242 positivos enteros; 1902 volúmenes de 1. Se conserva y se aclara la limitación; ver H-06. |
| D-05 | Abierto confirmado | Migraciones al leer | load_training_frame todavía aplica apply_migrations. Banco protege origen con copia, otros lectores no. |
| D-06 | Abierto confirmado | Persistencia no atómica/idempotente | backfill y persistencia conservan commits parciales; función declara no idempotencia de escenarios/runs. |
| D-07 | Abierto confirmado | Bloqueo de workflow incompleto | No se usa WorkflowLock en el flujo --persist-sql inspeccionado. No se hizo nueva prueba concurrente. |
| D-08 | No revalidado | Reset legacy sobre v17 | No se ejecutaron operaciones de borrado ni nueva reproducción de reset. |
| D-09 | No revalidado | Dos esquemas homónimos | Coexistencia presente; no se reejecutó create_schema sobre copia v17 para confirmar sus efectos. |
| D-10 | Abierto confirmado | Hash de INP dependiente de bytes | csv_backfill sigue usando read_bytes/sha256; no hay normalización de identidad semántica. |
| D-11 | Parcial | Manifest de datos | Nuevos manifiestos y hashes en prepared/reportes/versionados de correcciones; no se verificó una restauración completa desde cero. |
| D-12 | Abierto confirmado | Hash de evento incompleto | config_payload contiene nombre/factor/duración/pico, no las ordenadas del hidrograma. N-03. |
| D-13 | Abierto confirmado | Generación y SQL desacoplados | CLI general mantiene CSV y persistencia manual; v4_21 se calcula pero descarta en generación general. N-02. |
| D-14 | No revalidado | Tipado SQL no estricto | No se repitieron inserciones de texto/infinitos en todas las tablas. |
| D-15 | Abierto confirmado | Orden canónico dependiente del run_id | Persiste SORT_KEY run_id,node_id y suposición de equivalencia en main; riesgo con orden distinto de importación. |
| D-16 | No revalidado | Índices de FK/pinning | No se reejecutaron benchmarks de consultas ni auditoría completa de índices. |
| D-17 | No revalidado | Fingerprint del upgrade | No se probó carrera entre respaldo y migración ni recuperación completa. |
| D-18 | Abierto confirmado | Fallo de lectura convertido en cero | fillna(0.0) y fallback sin señal de invalidez permanecen; N-09. |
| D-19 | No revalidado | Atomicidad de persistencia legacy | No se ejecutó el pipeline desktop/legacy con fallos inyectados. |
| D-20 | Abierto confirmado | Métricas SQL agrupadas por factor | persist_training_run sigue usando GroupKFold sobre factor_mult; no generalización espacial. |
| D-21 | Abierto confirmado | Pendiente de 108C ausente | 963 nulos en preparado; extractor omite cota outfall, H-09. |
| D-22 | Parcial | Cobertura SQL | Pruebas seleccionadas de SQL aprobadas; nueva reproducción demuestra falta de pinning v4_21 y run mutable. |
| T-01 | Corregido en el alcance indicado | Fase A sin commit | En el worktree confirmado los cambios funcionales están registrados; el checkout principal sigue distinto. |
| T-02 | Parcial | Divergencia de checkouts | Worktree identificado y confirmado por autor; principal no integrado. No se modificaron ramas. |
| T-03 | Parcial | Mutaciones no detectadas | No se repitieron las 26 mutaciones; cobertura nueva no prueba cierre de esas ocho por sí sola. |
| T-04 | Abierto confirmado | --help ejecuta entrenamiento legacy | ml/train.py sigue llamando main sin argparse; inspeccionado sin ejecutar ese comando. |
| T-05 | Abierto confirmado | FLOW_UNITS omitido pasa guarda | Condición sigue siendo flow_units is not None; N-06. |
| T-06 | Corregido en el alcance indicado | Frame de inferencia no validado | Validador v4_21 incorporado en las dos rutas activas; no implica coherencia de manifiestos. |
| T-07 | Parcial | Documentación principal desactualizada | Informes recientes buenos; README conserva descripción CSV/sin SQLite/15 features y flujo antiguo. |
| T-08 | No revalidado | INP de validación versionados duplicados | No se repitió inventario ni comparación de artefactos históricos. |
| T-09 | Corregido en el alcance indicado | Config obsoleta de siete formas | config_7shapes.yaml eliminado en worktree respecto a 6e8579b. |
| T-10 | Parcial | Deriva de entorno | optuna ya declarado; reproduce_results exige 3.13 mientras evidencia usa 3.12.10. |
| T-11 | No revalidado | Clasificación documental de scenario_predict | Módulo activo revisado; no se cerró el inventario/documento antiguo de código basura. |
| T-12 | Abierto confirmado | Tuning contradice evaluación externa estricta | No anidado, reconocido como exploratorio. N-04. |
| T-13 | No revalidado | Presentaciones pesadas versionadas | No se hizo nuevo análisis de tamaño/historial para limpieza. |
| T-14 | No revalidado | build obsoleto | No se revalidó distribución instalada ni caché de build. |
| T-15 | No revalidado | Restos temporales | No se hizo limpieza ni auditoría de temporales previos. |
| T-16 | No revalidado | Gitignore inconsistente | No se ejecutó nueva auditoría integral de reglas de seguimiento. |
| T-17 | No revalidado | Diagrama antiguo | No se verificó todo el material de diagramación/presentación. |
| T-18 | Abierto confirmado | Errores de prepared no uniformes | _run_bench captura ValueError, no FileNotFoundError de resolve_prepared. |
| T-19 | Parcial | OOF y tiempos | OOF recuperables y recalculados; no cierre de temporización por inferencia en todas las rutas. |
| V-01 | Parcial | Figuras en muestra sin nota | Notas en curvas/factor y pruebas; figuras OOF nuevas. No se regeneró ni revisó todo el histórico. |
| V-02 | Abierto confirmado | Etiquetas de nodos no únicas | format_node_label sigue quitando sufijos: 2C y 2I se muestran como 2. |
| V-03 | No revalidado | Herramientas legacy sobre BD v17 | No se reejecutó GUI/runner legacy contra v17 para caracterizar sus mutaciones. |
| V-04 | Parcial | Procedencia no verificada en inferencia | predict_network verifica hashes, ScenarioPredictor no. Par activo distinto del afinado; N-11. |
| V-05 | Abierto confirmado | EDA con targets legacy | eda.py sigue usando ML_TARGET_* y get_feature_columns legacy; no se actualizó al contrato tabular activo. |
| V-06 | No revalidado | Tabla y mapa GUI con modelos distintos | GUI no ejecutada; no se declara cerrado. |
| V-07 | Corregido en el alcance indicado | halve_timeseries con efectos al importar | Guard main/función/control de sobrescritura; pruebas de scripts aprobadas. |
| V-08 | Abierto confirmado | Reproducción exige otro Python | reproduce_results.ps1 exige 3.13 y acaba invocando pipeline general v17; no reproduce parte 2. |
| V-09 | No revalidado | Etiquetas de familia fijas en gráficos | No se hizo revisión visual completa de todas las figuras con MLP. |
| V-10 | No revalidado | Visor de BD inexistente | No se ejecutó el botón de GUI ni se auditó su fallback. |
| V-11 | No revalidado | Filtrado por red/status en visualización | No se reprodujo con BD multirred; los consumidores deben revisarse al incorporar network_id. |
| V-12 | Parcial | Métricas de paridad no condicionadas | Parte 2 declara RMSE pooled y oracle; no se revisó cada figura histórica ni su rotulado. |
| V-13 | Abierto confirmado | Resiliencia como proporción de nodos secos | analysis/resilience.py conserva esa definición; no mide recuperación temporal ni servicio hidráulico completo. |
| V-14 | No revalidado | Nodos apilados en mapas | No se volvió a medir o inspeccionar solapamiento en renderizados. |
| V-15 | No revalidado | Ceros en eje logarítmico | No se inspeccionaron todos los renderizados de curvas. |
| V-16 | No revalidado | Excepciones silenciadas en soporte | No se inyectaron fallos en GUI, EDA y runner para cerrar el hallazgo. |
| V-17 | No revalidado | Plan EDA con completed fijos | No se ejecutó EDA con PCA/profiling omitidos o fallidos. |
| V-18 | No revalidado | Entradas vacías y dominios de métricas | No se repitió toda la batería adversaria de analysis/model_comparison. |
| V-19 | No revalidado | Lecturas crean BD legacy | No se abrieron rutas de BD inexistentes mediante runner para evitar artefactos fuera de auditoría. |
| V-20 | No revalidado | Hilos de GUI | No se ejecutó GUI ni prueba de concurrencia de Tk. |
| V-21 | No revalidado | Coma decimal/NaN en GUI | No se revalidaron entradas del parser de interfaz. |
| V-22 | Parcial | Cobertura de visualización | Pruebas de procedencia/notas incluidas y aprobadas; no equivalen a inspección visual integral. |
| V-23 | Parcial | Dependencias de reproducción | optuna y python-pptx añadidos; pyproject sigue sin dependencias runtime. Entorno limpio no reconstruido. |
| V-24 | No revalidado | Grafo Obsidian | No se regeneró ni revisó su cobertura de arquitectura. |
| V-25 | Abierto confirmado | GUI no corresponde al banco | GUI sigue ligada al stack legacy; no se considera entrada de entrenamiento multirred validada. |
| V-26 | No revalidado | Título Qx1 y escala de hidrograma | No se inspeccionó la figura resultante en diferentes redes/factores de inflow. |
| V-27 | No revalidado | Abrir imágenes en Windows | No se ejecutó la apertura de imágenes desde GUI/runner. |
| V-28 | No revalidado | Opciones de runner | No se revalidaron combinaciones de flags legacy. |
| V-29 | No revalidado | Código duplicado de gráficos | No se hizo inventario de alcance/huérfanos ni refactorización. |
| V-30 | No revalidado | CRS y unidades de coordenadas | No se conoce CRS; no se infiere por el rango numérico de coordenadas. |
| V-31 | No revalidado | Deserialización joblib | Se usaron artefactos locales del proyecto; no se hizo auditoría de seguridad de archivos externos. |
| V-32 | No revalidado | Empaquetado con efectos laterales | Suite de packaging y reconstrucción de wheel no ejecutadas. |
| V-33 | No revalidado | Detalles de plotting/encoding/imports | No se probaron INP cp1252 ni el conjunto completo de fallos secundarios. |
| V-34 | No revalidado | Entorno Tcl/Tk | Pruebas GUI no ejecutadas en esta revisión; no se hereda el resultado antiguo como verificación actual. |
