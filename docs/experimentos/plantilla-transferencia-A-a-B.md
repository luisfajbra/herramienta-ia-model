# Plantilla: transferencia congelada A → B

**Estado: protocolo exclusivamente sintético. La función pública `score_transfer` bloquea incondicionalmente el cálculo real.** No ejecutar con B real hasta completar la admisión hidráulica y la vinculación de procedencia descrita abajo. Tras usar las etiquetas de B para decisiones, B deja de ser un test externo para esas decisiones.

## Antes de simular o predecir

Registrar el par entrenado únicamente con A, su `prep_id`, corridas incluidas, contrato y hash, objetivo `macro_network_rmse`, parámetros seleccionados, semillas, estudio y hash, umbral de inundación en m³, umbral de clasificación 0,5 y bandas 24/108/344 m³. Registrar INP y hash de B, política de dominio, ID de red y corrida, componentes y hash de cada evento, y política hidráulica con hash. La ruta es `outputs/bench/multinetwork/<UTC>-<prep_id[:12]>/A_to_B_frozen/`.

La metadata del par debe contener `trial_metadata` con `selection=inner_tuning`, `study_path` y `study_sha256`. El estudio debe ser el `study.json` guardado por `save_tune_result`. El runner verifica hash, `prep_id`, familia, objetivo, dirección, parámetros, valor del mejor trial y redes de los trials completos. Las corridas A se derivan del manifiesto del par menos las excluidas. Esto comprueba consistencia, pero no demuestra cómo se generó el estudio. **Pendiente:** vincular automáticamente el dataset A, folds interiores y estudio desde el pipeline real; una declaración manual no reemplaza esa evidencia.

## Secuencia sintética de la API

1. `register_transfer(candidate_dir, inp_path, domain_policy, planned_runs, volume_bands_m3, output_root)` recibe solo planes. Cada registro incluye `network_id`, `run_id`, `event_series_components`, `event_series_sha256`, `hydraulic_policy`, `hydraulic_policy_sha256` y `expected_node_ids`. No recibe resultados ni etiquetas. Escribe el registro y su hash. Para MLP exige el ensemble persistido de cinco semillas con la agregación, el límite mínimo de volumen y el umbral 0,5 aprobados; una pareja de una sola semilla se rechaza.
2. `predict_transfer(directory, keys, features, runs)` recibe claves, features ordenadas y calidad por corrida. Comprueba política, INP, par y registro; excluye y reporta corridas no elegibles. Exige exactamente los nodos esperados declarados y un `shape_id` no vacío y consistente para cada corrida elegible. Si existe un rechazo de dominio o ninguna corrida elegible, no predice. Si pasa, escribe y sella `inputs.csv`, `predictions.csv` y el informe de exclusiones cuando exista. No recibe etiquetas.
3. `score_transfer(directory, labels)` está bloqueada por código y no produce métricas. Solo las pruebas sintéticas usan el helper privado `_score_synthetic_transfer`, que verifica sellos y crea `synthetic_scores.json` identificado como sintético. Esos resultados no son evidencia para una red real. La forma del hidrograma es un estrato descriptivo y no un holdout de familia física.

## Barrera pendiente para uso real

Implementar un manifiesto de admisión generado y validado internamente que vincule el INP y los archivos efectivos de eventos, la corrida y su reporte SWMM, la decisión QA, las claves y la cobertura nodal esperada, las features y las etiquetas con sus hashes. Antes de predecir, obtener de ese artefacto una vista B sin etiquetas; solo después del sello, cargar las etiquetas del mismo artefacto verificado. `PreparedDataset.load` verifica estructura y contrato, pero no autentica actualmente el contenido parquet contra los reportes. Por eso no basta para habilitar el score. La lista `expected_node_ids` del prototipo es declarada y no demuestra cobertura real. Las copias de archivo y hash en una misma carpeta no constituyen una firma independiente.

Las políticas de continuidad y drenaje son provisionales. Dos redes no estiman el desempeño sobre una población de redes futuras. Este flujo no entrena ni promueve modelos y no selecciona modelos usando B.
