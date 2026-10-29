# Frozen A/B SWMM rerun execution log

## Frozen run decisions

- Execute the pre-registered 1,000 A and 975 B scenarios with the existing synthetic nodal hydrographs.
- Round source timestamps to the nearest minute, as approved, and preserve nodal peaks.
- Keep B frozen as the transfer test; do not use B targets for fitting or tuning.
- Keep the 5% continuity limit and configured drainage/coverage tolerances provisional. Record ineligible cases and exclude them from training.
- Preserve every run root; do not overwrite partial generations.

## Execution interruptions and recovery

The first execution root, `outputs/experiments/frozen_pair_rerun_20261028/20261028T184001Z`, stopped after 464 A scenarios because Windows denied the atomic checkpoint replacement. Its partial artifacts remain preserved and are not eligible for staging.

The second root, `outputs/experiments/frozen_pair_rerun_20261028/20261028T191042Z`, stopped after the checkpoint JSON had reached 487 completed scenarios while its `.tmp` held 488. No simulator process remained, both files were readable, neither was read-only, and promoting the temporary checkpoint with `os.replace` then succeeded. This confirms a transient Windows file-lock/sharing condition at checkpoint publication rather than a SWMM or hydraulic QA error.

The runner now retries checkpoint replacement after bounded delays when it receives `PermissionError`. A regression test injects two transient permission failures and verifies the third atomic replacement succeeds. The focused runner suite passes (9 tests). The 488-scenario root is preserved as an interrupted generation; a new, cleanly anchored pair run is required for the authoritative A/B manifests.

## Authoritative execution

The current authoritative root is `outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z`, anchored before simulation at commit `4efc53a`. A completed all 1,000 scenarios with no technical errors. Recalculated QA currently excludes 134 runs (100 not drained under the frozen retry policy, 35 above the provisional 5% continuity limit, and one with both reasons); all 1,000 have finite coverage. The remaining 866 are provisionally eligible for A training, subject to successful manifest staging.

B remains in progress and sealed from model selection. Its initial checkpoint was 63/975; at 2026-10-29 03:25 UTC it had 352/975 results and 116 QA exclusions. These are operational counts, not final B results.

## Staging corrections found by the real A manifest

The first A staging attempt exposed a representational difference in the INP parser/writer: an empty optional `[TAGS]` section can be present in the source and omitted from the effective INP. Since the parsed section is empty, it carries no hydraulic objects or values. The hydraulic descriptor comparison now ignores only empty sections; a regression test covers this exact equivalence.

A second attempt exposed the distinct frozen event contracts. Synthetic shape events contain every source `FLOW` node with positive base peak. The `base` event preserves every source `FLOW` node, including 13 whose peak is zero. The verifier and feature constructor now accept exactly either set and reject any other node set. A regression fixture exercises a base event with a zero-peak `FLOW` node. The two staging suites pass: 43 tests.

The first two failed staging databases and their diagnostic artifacts are preserved and must not be reused. A third, fresh SQLite staging is underway with unique v3 manifest names. Until it completes and its final manifest is verified, A is not yet a prepared training dataset. No A model fitting or hyperparameter search has started; B targets have not been read.

## Model-selection boundary for the sealed A-to-B test

Because B is the frozen external transfer test, its labels cannot be used in `LeaveOneNetworkOut` tuning. That evaluator would train one direction with B labels when scoring A and would compromise the approved A-only-to-B claim. The approved training budget was estimated for five inner folds, so the A-only studies will use `GroupKFold(5)` grouped by `shape_id`, with 50 trials, `macro_network_rmse`, and the configured 200 MLP epochs. With only network A in these studies, the macro objective equals A's RMSE; the metric name and one-network limitation will remain explicit in metadata. The five-seed MLP ensemble is fitted only after its A-only study selects parameters. B is scored once after prediction hashes are sealed. This estimates transfer to this one held-out network, not performance over a population of unseen networks.

The `--bench-tune` CLI previously omitted both the configured tuning objective and the fold protocol when it called `run_study`, silently selecting the function's default `mean_inner_fold_rmse`. A regression test reproduced the omission. The CLI now passes the configured objective (falling back to the legacy mean-fold objective only when configuration omits one) and explicitly records `GroupKFold`; the focused CLI tuning tests pass.

## A staging and preparation completed

The third fresh A staging completed successfully. It records all 1,000 `COMPLETE` runs; 866 are eligible and 134 are excluded (100 `drainage_not_drained`, 35 `continuity_limit_exceeded`, with one run in both reason groups). The SQLite contains 138,560 eligible feature rows and exactly 138,560 target rows. No target is stored for the excluded runs. The final source manifest SHA-256 is `4bf0c50f61b2287ab8510369e46f26495393104d0fe670ac6ab444df28c055a6`; the verified label-free source manifest is `6b7e7ece058a9e91e708224fe48339afe039a506d84f8327ae53dc4eaa3b42e5`.

PreparedDataset `813323e3ef75b752` contains those 138,560 rows from A and the `tabular_v4_21` contract. Its `GroupKFold5` split integrity was reloaded and checked: all five folds have zero `shape_id` overlap between train and validation. The frozen A protocol has 40 groups (39 synthetic forms plus `base`); QA left no eligible run for `base`, `long_moderate`, `shape_2h`, or `shape_5h`. Ten other groups have between 16 and 24 eligible factors rather than all 25. The training data therefore represent 36/40 groups, with no manual repair or target substitution.

Selection is preregistered in `A/model_selection_Aonly_813323e3ef75b752/selection_preregistration_v2.json` (SHA-256 `6d5b729e33bd11622c2aba70063facdef4e9997c2dffb2aee7dcc8e54b64fd77`). The earlier v1 file is preserved but superseded because its `excluded_run_ids` field held exclusion records rather than integer IDs; the corrected v2 was sealed before either study began. XGBoost completed 50 A-only trials with GroupKFold(5) and `macro_network_rmse`. The best result was 73.78205679320975 at trial 13. Its study JSON is `A/model_selection_Aonly_813323e3ef75b752/studies/xgboost/813323e3ef75b752_20261028T224207Z/study.json` (SHA-256 `5b3ace9ece0223b86c538c63e56ccaff60f6ff37de8e0221021adca57764cd93`). The fitted candidate at `A/candidates/xgboost` records the study and preregistration hashes; model SHA-256 values are classifier `a1a6e504cca9a4553e93edeed3340da48c70b5b74167716c4bc96946f376046f` and regressor `045526a91a4a99a89bde1f473bfc4e67eefa950c5baacf835dcc595c5ecc1b7f`. Candidate provenance reload verification passed. The 50-trial MLP search is running over the same A-only `PreparedDataset`, GroupKFold(5), seed 42, `macro_network_rmse`, 200 epochs, and batch size 256. Trials 0–9 have completed with values 359.0459, 129.1288, 208.1512, 152.8342, 201.8404, 123.2106, 96.6671, 166.5368, 129.0509, and 174.2521; trial 6 is the current best. Trial wall times have ranged from about 22 to 40 minutes. These are inner-search observations, not selected-model estimates. Neither study uses B targets.

## Source-bound A-to-B scoring gate

Task 5 previously exposed only a synthetic scoring helper; the public scorer rejected every real run. `score_transfer` now accepts only a verified B staging and SQLite connection, verifies the frozen source manifest, artifact links, candidate, prediction and input seals, and feature rows, and replays the registered candidate to require identical predictions before reading targets. It then re-extracts labels in memory from each eligible run's exact hash-verified RPT. It checks the report hash against hydraulic QA and the flood threshold, requires exact prediction/label run-node coverage, and writes aggregate overall, by-shape, and by-scenario metrics without saving B label rows or inserting them into SQLite. Callers cannot pass an arbitrary label frame. After adding the replay check, the label-free staging and frozen-transfer suites passed (55 tests).

## Weekly presentation prepared (2026-10-28; initial English version superseded)

Created `Presentacion_MLSMFP_28-10_avances_auditoria_transferencia.pptx` using the 2026-10-20 weekly deck's 16:9 theme and visual style. It reports A's 866 eligible runs and 138,560 rows; the XGBoost A-only search result (73.78205679320975 m³, Optuna trial 13); MLP and B as still running; and the frozen transfer/scoring boundary. The operational B snapshot when generated was 321 QA results, 114 exclusions and 540 pending scenarios. This count is time-dependent, not an evaluation metric.

The deck does not reuse the prior benchmark's parity plots or flood maps as current findings. It explains that current B visuals must be generated only after MLP completion, prediction sealing and verified B scoring. The generated deck passed ZIP integrity and `python-pptx` reopen checks (9 slides). SHA-256: `f357d3e60a34676d4f411c187ea465b4144d6ac4cce7861db26beaa8613129a1`. Rebuild command: `venv/Scripts/python.exe scripts/build_weekly_2026_10_28_deck.py`.

## Spanish style revision (2026-10-28)

The first presentation was open in PowerPoint, so it was left intact and a revised file was generated as `Presentacion_MLSMFP_28-10_avances_auditoria_transferencia_v2.pptx`. The revision translates the deck into Spanish, samples the navy/blue/gold palette from the 20-10 slides, and redistributes the content across a summary, QA, evaluation design, XGBoost result, model/B status, visual-readiness, scope and traceability slides. It passed ZIP integrity and `python-pptx` reopen checks (9 slides). The B progress snapshot in this version is 324 QA results, 114 exclusions and 537 pending scenarios; it is operational and time-dependent. SHA-256: `b72f494918c2b88c8da3a9eaa0a68f437ce29ff78e72a9ac4a77e0b7c4d0e078`.

## Visualización de resultados actuales (2026-10-28)

En respuesta a la necesidad de mostrar resultados visibles, se generó `Presentacion_MLSMFP_28-10_avances_auditoria_transferencia_con_mapas.pptx`, que supersede a v2 para la presentación semanal. Conserva el idioma y la paleta del formato anterior y agrega dos diapositivas visuales. La topología de B se trazó desde el INP congelado de Chico Norte: 523 nodos con coordenadas y 522 conducciones. Esa figura es únicamente geométrica; no implica que B ya tenga evaluación hidráulica o predicciones ML.

Los mapas de A representan volúmenes de inundación de SWMM para corridas del conjunto A admitidas por QA, no predicciones. Se eligieron determinísticamente entre 711 corridas inundadas: una corrida mediana por volumen total (`run_id=439`, `shape_five_pulse_8h5`, factor 2,8; 9.518 m³ en 52 nodos) y la de máximo volumen total (`run_id=700`, `shape_quasi_uniform_block`, factor 5,0; 183.407 m³ en 93 nodos). Los tres PNG fuente están bajo `outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z/presentation_visuals/`.

Al regenerar la presentación, el estado operativo de B era 339 resultados QA, 115 excluidos y 521 pendientes (corte aproximado: 22:04 hora de Bogotá, 2026-10-28). Esos conteos cambian durante la ejecución y no son métricas de transferencia. Los resultados de paridad y los mapas de predicción de B siguen pendientes de completar MLP, sellar predicciones y ejecutar el scoring externo verificado.

La presentación se verificó con `python-pptx` (11 diapositivas), validación ZIP (102 entradas, sin errores), y una comprobación de límites que no encontró elementos fuera del lienzo. La compilación del script generador también pasó. Una primera comprobación visual detectó que el mapa de topología excedía el borde inferior; se redujo su ancho y se regeneró antes de la verificación final. SHA-256 del PPTX: `a50e9c37ce9dff40d55d001b77ded94b91124d6552b78b3f46df0e35336102ba`. SHA-256 de los mapas, en orden topología B, mediana A y máximo A: `68a41346beb7d232f5c93c2c2764655de58858b43e5431bbcd7158fc10c65823`, `59970dd527269e446f2ffef6ec45ad176e75e23fbb0f70794349edfaa34653a1`, `3a5ed25e44505205f2816d83cfb1adf299db748404f59d3a325c17b2e04d60c`.


## Comparaci?n retrospectiva y entrada sint?tica de B (2026-10-28)

Se prepar? una revisi?n posterior como `Presentacion_MLSMFP_28-10_avances_auditoria_transferencia_comparacion_modelo_hidrograma.pptx`. Los dos escenarios de A tienen ahora una diapositiva cada uno, con mapas SWMM y de predicci?n del XGBoost previamente promovido el 2026-10-06. El artefacto del modelo se verific? contra `outputs/models/promotion.json` (contrato `tabular_v4_21`; SHA-256 del clasificador `a7bb8f8f08f86deba0a99dfc8009e516e4e0a5843050601335a2ea8cd8a3589f`, regresor `162154754b305b8d911bb1dd9c9b4ac09a738abac8cf9088aa950f8da363cb62`) y se aplic? a las features de los mismos dos escenarios admitidos de A. Los vol?menes totales comparados son: mediano, SWMM 9.518 m? frente al modelo 9.278 m? (52 nodos con volumen ?1 m? en ambos); m?ximo, SWMM 183.407 m? frente al modelo 188.628 m? (93 y 91 nodos, respectivamente).

Estos pares son **ilustrativos y retrospectivos**: el modelo antiguo no representa al candidato XGBoost de la b?squeda actual, ni sus predicciones se generaron bajo el protocolo OOF congelado. No se interpretan como confiabilidad, exactitud independiente ni transferencia entre redes. La diapositiva explicita que el porcentaje de confiabilidad se medir? despu?s de sellar predicciones y verificar etiquetas de B; no se fij? ni invent? un porcentaje esperado.

Se agreg? una figura de un hidrograma sint?tico de entrada en Chico Norte: nodo `268I`, de mayor pico base entre los nodos FLOW del INP (54,08 L/s), forma `shape_five_pulse_8h5` de 8,5 h y factor 2,8, con pico aplicado 151,4 L/s. El INP contiene 522 nodos FLOW, de los que 521 tienen pico positivo. La campa?a congelada contempla 39 formas ? 25 factores (0,2?5,0, paso 0,2) = 975 escenarios sint?ticos de caudal por nodo. No es lluvia ni transformaci?n lluvia?escorrent?a. Al generar la diapositiva, B registraba 344 resultados aptos, 115 excluidos por QA y 516 pendientes; corte operacional cambiante, no m?trica de modelo.

La versi?n final qued? en 13 diapositivas. `python-pptx` pudo recargarla, el ZIP tuvo 109 entradas y cero errores, ninguna forma excedi? los l?mites del lienzo, se verificaron los textos clave (975 escenarios, mapas emparejados, cautelas metodol?gicas y ausencia de porcentaje no respaldado) y pas? `py_compile`. SHA-256 del PPTX: `748e173f9db22d8ca16a21d6e075ae36a3d67cfc1f0ff972e746b817b35279af`. Las figuras fuente quedaron en `outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z/presentation_visuals_modelo_hidrograma/`.

## Corte operacional de la ejecuci?n (2026-10-29 04:14 UTC)

Se confirma la convenci?n aprobada: redondear al minuto m?s cercano; el desplazamiento de cada punto queda limitado a 30 s. En este corte, B tiene 393/975 corridas registradas, con 117 exclusiones QA, y contin?a en ejecuci?n. El estudio MLP lleva 11/50 trials A-only completados; el mejor sigue siendo el trial 6 (`macro_network_rmse=96.6671`). El trial 11 contin?a ejecut?ndose desde hace m?s tiempo que el rango previo de 22–40 minutos. Su proceso sigue activo y consume CPU, por lo que a?n no se declara fallido ni se interrumpe. No se han consultado targets de B.

## Corte operacional posterior (2026-10-29 04:20 UTC)

B tiene 398/975 corridas registradas y 117 exclusiones QA, y sigue en ejecuci?n. MLP complet? 12/50 trials A-only: el trial 11 obtuvo `macro_network_rmse=135.6435`; el mejor permanece en el trial 6 con `96.6671`. El trial 12 ya est? en ejecuci?n. Optuna emiti? las advertencias conocidas de codificaci?n de hiperpar?metros categ?ricos con tuplas de capas; el estudio en memoria sigui? y el trial termin? normalmente. No se han consultado targets de B.

## Corte de tuning y simulaci?n (2026-10-29 04:43 UTC)

B tiene 413/975 corridas registradas y 118 exclusiones QA; sigue ejecut?ndose. MLP complet? 13/50 trials A-only. El trial 12 obtuvo `macro_network_rmse=1635.6768`; el mejor sigue siendo el trial 6 con `96.6671`, por lo que el valor del trial 12 solo queda como observaci?n interna de b?squeda. El trial 13 est? en ejecuci?n. Las advertencias de Optuna sobre opciones categ?ricas de tipo tupla volvieron a aparecer; no impidieron guardar el resultado del trial. No se han consultado targets de B.

## Corte operacional posterior (2026-10-29 05:14 UTC)

B tiene 445/975 corridas registradas, con 128 exclusiones QA, y sigue ejecut?ndose. MLP complet? 14/50 trials A-only. El trial 13 obtuvo `macro_network_rmse=121.7167`; el mejor contin?a siendo el trial 6 con `96.6671`. El trial 14 est? en ejecuci?n. Se mantienen las advertencias de Optuna acerca de opciones categ?ricas representadas con tuplas; los trials terminan y se registran normalmente en el estudio en memoria. No se han consultado targets de B.

## Corte operacional posterior (2026-10-29 05:44 UTC)

B tiene 470/975 corridas registradas, con 129 exclusiones QA, y sigue ejecut?ndose. MLP complet? 15/50 trials A-only. El trial 14 obtuvo `macro_network_rmse=120.9003`; el mejor sigue en el trial 6 con `96.6671`. El trial 15 est? en ejecuci?n. Las advertencias de Optuna sobre opciones categ?ricas de tipo tupla persisten, sin impedir la finalizaci?n ni el registro manual de los trials. No se han consultado targets de B.

## Corte operacional posterior (2026-10-29 06:12 UTC)

B tiene 495/975 corridas registradas y 131 exclusiones QA; sigue en ejecuci?n. MLP complet? 16/50 trials A-only. El trial 15 obtuvo `macro_network_rmse=138.7742`; el mejor permanece en el trial 6 con `96.6671`. El trial 16 ya est? en ejecuci?n. Las advertencias de Optuna sobre las opciones categ?ricas de tipo tupla contin?an, sin impedir la finalizaci?n de los trials. No se han consultado targets de B.

## Corte operacional posterior (2026-10-29 06:39 UTC)

B tiene 526/975 corridas registradas, con 132 exclusiones QA, y sigue en ejecuci?n. MLP complet? 17/50 trials A-only. El trial 16 obtuvo `macro_network_rmse=143.9044`; el mejor permanece en el trial 6 con `96.6671`. El trial 17 est? en ejecuci?n. Contin?an las advertencias de Optuna relacionadas con las opciones categ?ricas de tipo tupla; el estudio en memoria sigue registrando trials completados. No se han consultado targets de B.

## Corte operacional posterior (2026-10-29 07:10 UTC)

B tiene 539/975 corridas registradas y 132 exclusiones QA; sigue en ejecuci?n. MLP complet? 18/50 trials A-only. El trial 17 obtuvo `macro_network_rmse=144.3201`; el mejor sigue siendo el trial 6 con `96.6671`. El trial 18 est? en ejecuci?n. Las advertencias de Optuna sobre las opciones categ?ricas de tipo tupla contin?an; los trials siguen concluyendo y registrando resultados. No se han consultado targets de B.

## Corte operacional posterior (2026-10-29 07:40 UTC)

B tiene 559/975 corridas registradas y 134 exclusiones QA; sigue en ejecuci?n. MLP complet? 19/50 trials A-only. El trial 18 obtuvo `macro_network_rmse=162.4768`; el mejor permanece en el trial 6 con `96.6671`. El trial 19 ya est? en ejecuci?n. Las advertencias de Optuna relacionadas con tuplas en las opciones categ?ricas contin?an sin detener el estudio en memoria. No se han consultado targets de B.

## Corte operacional posterior (2026-10-29 08:11 UTC)

B tiene 587/975 corridas registradas y 136 exclusiones QA; sigue ejecut?ndose. MLP complet? 20/50 trials A-only. El trial 19 obtuvo `macro_network_rmse=116.9490`; el mejor permanece en el trial 6 con `96.6671`. El trial 20 est? en ejecuci?n. Persisten las advertencias de Optuna acerca de opciones categ?ricas de tipo tupla, sin impedir la finalizaci?n de los trials. No se han consultado targets de B.

## Corte operacional posterior (2026-10-29 08:42 UTC)

B tiene 625/975 corridas registradas y 146 exclusiones QA; sigue en ejecuci?n. MLP complet? 21/50 trials A-only. El trial 20 obtuvo `macro_network_rmse=123.9830`; el mejor permanece en el trial 6 con `96.6671`. El trial 21 est? en ejecuci?n. Las advertencias de Optuna sobre opciones categ?ricas de tipo tupla persisten sin impedir la finalizaci?n de los trials. No se han consultado targets de B.

## Corte operacional posterior (2026-10-29 09:12 UTC)

B tiene 654/975 corridas registradas y 148 exclusiones QA; sigue en ejecuci?n. MLP complet? 22/50 trials A-only. El trial 21 obtuvo `macro_network_rmse=129.5188`; el mejor permanece en el trial 6 con `96.6671`. El trial 22 est? en ejecuci?n. Contin?an las advertencias de Optuna sobre codificaci?n de opciones categ?ricas con tuplas; los trials se completan y registran con normalidad. No se han consultado targets de B.

## Corte operacional posterior (2026-10-29 09:41 UTC)

B tiene 673/975 corridas registradas y 148 exclusiones QA; sigue en ejecuci?n. MLP complet? 23/50 trials A-only. El trial 22 obtuvo `macro_network_rmse=154.7311`; el mejor permanece en el trial 6 con `96.6671`. El trial 23 est? en ejecuci?n. Persisten las advertencias de Optuna acerca de opciones categ?ricas con tuplas; los resultados se siguen guardando al completar cada trial. No se han consultado targets de B.

## Corte operacional posterior (2026-10-29 10:12 UTC)

B tiene 692/975 corridas registradas y 148 exclusiones QA; sigue en ejecuci?n. MLP complet? 24/50 trials A-only. El trial 23 obtuvo `macro_network_rmse=124.7401`; el mejor permanece en el trial 6 con `96.6671`. El trial 24 est? en ejecuci?n. Persisten las advertencias de Optuna sobre opciones categ?ricas representadas como tuplas; no han impedido el cierre de los trials. No se han consultado targets de B.

## Corte operacional posterior (2026-10-29 10:44 UTC)

B tiene 725/975 corridas registradas y 151 exclusiones QA; sigue en ejecuci?n. MLP complet? 25/50 trials A-only. El trial 24 obtuvo `macro_network_rmse=115.3383`, segundo mejor hasta ahora; el trial 6 sigue primero con `96.6671`. El trial 25 est? en ejecuci?n. Persisten las advertencias de Optuna sobre opciones categ?ricas tipo tupla, sin impedir finalizar los trials. No se han consultado targets de B.

## Corte operacional posterior (2026-10-29 11:31 UTC)

B tiene 771/975 corridas registradas y 155 exclusiones QA; el lote continúa en ejecución. MLP completó 25/50 trials A-only. El trial 25 obtuvo `macro_network_rmse=128.1566`; el trial 6 sigue primero con `96.6671`. El trial 26 continúa en ejecución. Se mantienen las advertencias de Optuna sobre opciones categóricas tipo tupla, sin fallos de trials observados. No se han consultado targets de B. Se conserva el redondeo aprobado al minuto más cercano (desplazamiento máximo: 30 s).

## Corte operacional posterior (2026-10-29 12:01 UTC)

B tiene 802/975 corridas registradas y 159 exclusiones QA; continúa en ejecución. MLP completó 26/50 trials A-only. El trial 26 obtuvo `macro_network_rmse=127.4849`; el trial 6 continúa como mejor con `96.6671`. El trial 27 sigue ejecutándose. Se mantienen las advertencias de Optuna acerca de opciones categóricas tipo tupla, sin error de estudio observado. No se han consultado targets de B. Se conserva el redondeo aprobado al minuto más cercano, con desplazamiento máximo de 30 s.


## Corte para la presentaci?n semanal (2026-10-29 12:09 UTC)

Al regenerar la presentaci?n, B ten?a 805/975 resultados registrados, 159 exclusiones QA y 11 escenarios pendientes. El ?ltimo corte documentado de MLP sigue siendo 12:01 UTC: 26/50 ensayos terminados, trial 27 en ejecuci?n y mejor resultado interno en trial 6 (`macro_network_rmse=96.6671`). El conjunto B contin?a sellado para el scoring externo. La presentaci?n y el guion estiman 10?22 horas desde las 07:09 de Bogot? hasta completar el scoring, con reserva de hasta 24 horas; es un rango de planificaci?n basado en la duraci?n observada, no una promesa. Se crearon `Presentacion_MLSMFP_29-10_avances_auditoria_transferencia.pptx`, `Guion_presentador_MLSMFP_29-10-2026.docx` y su fuente editable `Guion_presentador_MLSMFP_29-10-2026.md`.

## Corte operacional posterior (2026-10-29 13:16 UTC)

B tiene 842/975 corridas registradas y 161 exclusiones QA; el lote sigue en ejecución. MLP completó 29/50 trials A-only. El trial 29 obtuvo `macro_network_rmse=127.5997`; el trial 6 continúa como mejor con `96.6671`. El trial 30 acaba de iniciar. Siguen apareciendo las advertencias de Optuna sobre opciones categóricas de tipo tupla; no han impedido completar los trials observados. No se han consultado targets de B. Se mantiene el redondeo aprobado al minuto más cercano (desplazamiento máximo de 30 s).


## Presentaci?n: nombres de las redes y corte actualizado (2026-10-29 14:00 UTC / 09:00 Bogot?)

A petici?n del usuario, en los materiales de exposici?n se nombran las redes como Chico Sur (entrenamiento) y Chico Norte (evaluaci?n externa); A/B se conservan solo cuando forman parte de rutas o identificadores t?cnicos internos. El corte de `batch_progress.json` registra 871 escenarios con resultado, 709 elegibles, 162 excluidos por QA y 104 pendientes de 975. Las filas `hydraulic_qa_ineligible` tambi?n aparecen en `results`; por eso el conteo no debe sumar resultados y fallas como categor?as independientes. Se corrigi? el generador para deduplicar por `scenario_key` y contar QA sobre la intersecci?n de claves. El ?ltimo contador persistido de MLP sigue en 26/50 a las 07:01 (trial 27 en curso); el proceso permanece activo, pero no se ha confirmado un contador actualizado. La ETA anterior de 10?22 horas qued? retirada por estar basada en un corte vencido. Presentaci?n y guion reflejan estos nombres y las limitaciones de estado.

## Corte operacional posterior (2026-10-29 14:22 UTC)

B tiene 882/975 corridas registradas y 162 exclusiones QA; el lote continúa. MLP completó 31/50 trials A-only. El trial 31 obtuvo `macro_network_rmse=116.4868`; el trial 6 sigue primero con `96.6671`. El trial 32 acaba de iniciar. Las advertencias de Optuna sobre opciones categóricas tipo tupla siguen presentes, sin impedir que los trials terminen. No se han consultado targets de B. Se mantiene la convención aprobada de redondeo al minuto más cercano, con desplazamiento máximo de 30 s.


## Corte operacional y registro para la tesis (2026-10-29 14:24 UTC / 09:24 Bogotá)

**Chico Norte:** `batch_progress.json` fue escrito a las 09:22:42 Bogotá y mantiene estado `running`. Hay 883 claves de escenario con resultado de 975: 721 elegibles, 162 excluidas por QA y 92 pendientes. El conteo usa `scenario_key` único; las filas `hydraulic_qa_ineligible` se solapan con `results` y no se suman como corridas adicionales. El último artefacto SWMM observado se actualizó alrededor de las 09:22. Entre los cortes 13:16 y 14:24 UTC, el registro pasó de 842 a 883 resultados (+41 en 68 min; aproximadamente 36 escenarios/h). Si el ritmo se mantiene, los 92 restantes tomarían cerca de 2 h 33 min; reservar 3-4 h por variabilidad, extensiones de drenaje y reintentos. Es una extrapolación de planificación, no un compromiso.

**Desglose QA de Chico Norte:** 90 exclusiones registran continuidad superior al 5% y 75 estado de drenaje no aceptable; tres escenarios tienen ambos motivos. El total de escenarios excluidos únicos es 162.

**MLP:** el último corte persistido disponible es 14:22 UTC: 31/50 trials terminados, trial 32 iniciado; trial 31 marcó `macro_network_rmse=116.4868` y el mejor sigue siendo trial 6 (`96.6671`). El proceso permanece activo. Estos trials usan Chico Sur para seleccionar hiperparámetros; cada evaluación sigue el protocolo registrado de cinco folds.

**Secuencia acordada:** terminar y calificar las simulaciones de Chico Norte; terminar MLP y fijar el ensemble; sellar modelos, variables y predicciones; luego hacer una sola evaluación externa en Chico Norte y reportar métricas, paridad y mapas. No entrenar ni seleccionar con Chico Norte antes de ese scoring, porque dejaría de ser una prueba independiente. Después de registrar el resultado externo, se puede ajustar una versión final con Chico Sur + Chico Norte; para medir transferencia de esa versión hace falta una tercera red no usada durante el entrenamiento. Los umbrales hidráulicos siguen siendo provisionales y deben presentarse como tales.

**Trazabilidad para la tesis:** cada corte se añade al final de esta bitácora (sin sobrescribir cortes anteriores) e incluye hora UTC/Bogotá, fuente del estado, escenarios únicos completados/elegibles/excluidos/pendientes, motivos QA, progreso persistido de Optuna, cálculo y límites de cualquier ETA, decisiones de entrenamiento/evaluación y artefactos/manifiestos aplicables. El staging valida estructura e integridad de los RPT/OUT y registra sus hashes; no usa objetivos de Chico Norte para seleccionar ni entrenar modelos antes de la evaluación externa.

## Corte operacional posterior (2026-10-29 14:55 UTC)

B tiene 895/975 corridas registradas y 162 exclusiones QA; sigue en ejecución. MLP completó 32/50 trials A-only. El trial 32 obtuvo `macro_network_rmse=138.7855`; el trial 6 mantiene el mejor valor con `96.6671`. El trial 33 ya está en curso. Persisten advertencias de Optuna sobre opciones categóricas tipo tupla, sin impedir la finalización de los trials observados. No se han consultado targets de B. Se mantiene el redondeo aprobado al minuto más cercano, con desplazamiento máximo de 30 s.

## Corte operacional posterior (2026-10-29 15:56 UTC)

B tiene 934/975 corridas registradas y 163 exclusiones QA; el lote continúa en ejecución. MLP completó 34/50 trials A-only. El trial 34 obtuvo `macro_network_rmse=129.0895`; el trial 6 conserva el mejor valor con `96.6671`. El trial 35 acaba de iniciar. Continúan las advertencias de Optuna sobre opciones categóricas tipo tupla, sin fallos de trials registrados. No se han consultado targets de B. Se mantiene el redondeo aprobado al minuto más cercano, con desplazamiento máximo de 30 s.

## Corte operacional posterior (2026-10-29 16:27 UTC)

B tiene 944/975 corridas registradas y 163 exclusiones QA; el lote continúa. MLP completó 35/50 trials A-only. El trial 35 obtuvo `macro_network_rmse=125.7343`; el trial 6 conserva el mejor valor con `96.6671`. El trial 36 acaba de iniciar. Se mantienen las advertencias de Optuna sobre opciones categóricas tipo tupla, sin error de trial observado. No se han consultado targets de B. Se mantiene la convención aprobada de redondeo al minuto más cercano, con desplazamiento máximo de 30 s.


## Corte de estado de modelos y Chico Norte (2026-10-29 16:45 UTC / 11:45 Bogot?)

**Chico Norte:** el ?ltimo `batch_progress.json` se escribi? a las 16:45:48 UTC y conserva estado `running`. Registra 969 de 975 escenarios ?nicos; quedan 6 pendientes. Bajo el filtro congelado hay 787 elegibles y 182 excluidos: 110 exceden el 5% de continuidad, 75 no alcanzan drenaje aceptable y 3 aparecen en ambos grupos. No hay escenarios con cobertura no finita. La cuenta de exclusiones es la uni?n de motivos, no su suma simple.

**XGBoost:** el estudio A-only termin? sus 50/50 ensayos; el trial 13 obtuvo el menor `macro_network_rmse` interno registrado (73.7821 m?). El candidato XGBoost ya qued? ajustado y sellado con datos elegibles de Chico Sur. Este n?mero es de selecci?n interna por cinco folds de formas; a?n no es el resultado externo de Chico Norte.

**MLP:** el ?ltimo contador de ensayos que qued? documentado es 31/50 a las 14:22 UTC, con el trial 32 iniciado. Hay procesos Python activos, pero no se encontr? un checkpoint persistido m?s reciente que permita afirmar el contador actual. Por ello, este corte no infiere cu?ntos ensayos MLP han terminado desde entonces.

**Siguiente paso:** terminar y recalificar las 6 simulaciones pendientes; completar y guardar el estudio/ensemble MLP; despu?s sellar modelos y predicciones y realizar una sola evaluaci?n externa en los escenarios elegibles de Chico Norte. No incorporar objetivos de Chico Norte al entrenamiento ni a la selecci?n antes de esa evaluaci?n. Los umbrales hidr?ulicos, incluido el 5% de continuidad, siguen siendo provisionales.

## Cierre de regeneración B y verificación de artefactos (2026-10-29 16:55 UTC)

El lote B terminó con 975/975 escenarios registrados. El registro del lote reporta 188 escenarios únicos QA-ineligibles y 787 elegibles. La cobertura finita está completa en 975; 900 reportan estado drenado y 75 no drenado; 859 tienen continuidad de hasta 5%. Las causas registradas son 116 excedencias de continuidad y 75 casos sin drenaje; tres escenarios tienen ambas causas, por eso las causas suman 191 y los excluidos únicos son 188. La regla de continuidad del 5% sigue siendo provisional y requiere justificación hidráulica.

Verifiqué SHA-256 y existencia para los 3.901 archivos únicos enlazados desde las corridas B (INP efectivos, eventos, RPT y OUT): cero faltantes y cero discrepancias. No se consultaron targets de B. El scoring A→B permanece cerrado hasta que estén listos y congelados los candidatos de A. En este corte MLP lleva 35/50 trials y el trial 36 continúa ejecutándose; el mejor sigue siendo el trial 6 (`macro_network_rmse=96.6671`).

## Corte operacional posterior (2026-10-29 17:56 UTC)

B permanece completo: 975 escenarios, 188 exclusiones QA únicas y 787 elegibles; los 3.901 archivos únicos mantienen hashes verificados. MLP completó 38/50 trials A-only. El trial 38 obtuvo `macro_network_rmse=153.5249`; el trial 6 continúa como mejor con `96.6671`. El trial 39 está en ejecución. El scoring de B sigue cerrado y no se han consultado sus targets.


## Cierre de simulaciones de Chico Norte (2026-10-29 16:48 UTC / 11:48 Bogot?)

El `batch_progress.json` de Chico Norte fue actualizado a las 16:48:39 UTC y marca `complete`: las 975 de 975 simulaciones tienen resultado y no quedan escenarios pendientes. El QA provisional acepta 787 escenarios. Excluye 188: 116 superan el umbral provisional de continuidad del 5%, 75 no alcanzan drenaje aceptable y 3 presentan ambos motivos. No hay casos con cobertura no finita. Los 6 escenarios que faltaban en el corte anterior se completaron y quedaron excluidos por continuidad; por eso no aument? el conjunto elegible. Estos conteos son escenarios ?nicos.

XGBoost conserva el estudio 50/50 y el candidato Chico Sur ya sellado, con mejor objetivo interno de 73.7821 m? en trial 13; a?n no se ha calculado su evaluaci?n externa en Chico Norte. Se observan procesos Python activos correspondientes al trabajo del modelo, pero no hay un checkpoint MLP m?s nuevo que el de 31/50 trials al 14:22 UTC; el contador actual no est? confirmado.

El siguiente paso es completar la verificaci?n/staging de Chico Norte con los enlaces e hashes de sus artefactos, y preparar el scoring externo del candidato XGBoost sin recalibrarlo con etiquetas de Chico Norte. En paralelo, completar y persistir los 50 trials y el ensemble MLP; despu?s evaluar ese candidato con el mismo conjunto elegible congelado. Mantener separados los resultados externos por modelo y no usar los objetivos de Chico Norte para seleccionar hiperpar?metros.

## Corte operacional posterior (2026-10-29 19:01 UTC)

B permanece completo y verificado: 975 escenarios, 188 exclusiones QA únicas, 787 elegibles y 3.901 artefactos únicos sin faltantes ni discrepancias SHA-256. MLP completó 40/50 trials A-only. El trial 40 obtuvo `macro_network_rmse=123.1903`; el trial 6 mantiene el mejor valor con `96.6671`. El trial 41 está en curso. El scoring A→B sigue sellado; no se han consultado targets de B.

## Corte operacional posterior (2026-10-29 19:33 UTC)

B permanece completo y verificado: 975 escenarios, 188 exclusiones QA únicas, 787 elegibles y 3.901 artefactos únicos verificados por SHA-256. MLP completó 41/50 trials A-only. El trial 41 obtuvo `macro_network_rmse=124.8785`; el trial 6 continúa como mejor con `96.6671`. El trial 42 ya está en ejecución. B permanece sellado y sus targets no se han consultado.

## Corte operacional posterior (2026-10-29 20:05 UTC)

B permanece completo y verificado: 975 escenarios, 188 exclusiones QA únicas, 787 elegibles y 3.901 artefactos únicos sin faltantes ni discrepancias de SHA-256. MLP completó 42/50 trials A-only. El trial 42 obtuvo `macro_network_rmse=101.3328`, segundo mejor; el trial 6 conserva el primer lugar con `96.6671`. El trial 43 está en curso. El scoring de B sigue sellado y sus targets no se han consultado.

## Corte operacional posterior (2026-10-29 20:36 UTC)

B permanece completo y verificado: 975 escenarios, 188 exclusiones QA únicas, 787 elegibles y 3.901 artefactos únicos sin faltantes ni discrepancias SHA-256. MLP completó 43/50 trials A-only. El trial 43 obtuvo `macro_network_rmse=106.0186`; el trial 6 sigue primero con `96.6671` y el trial 42 sigue segundo con `101.3328`. El trial 44 está en curso. No se han consultado targets de B y el scoring sigue cerrado.

## Corte operacional posterior (2026-10-29 21:09 UTC)

B sigue completo y verificado: 975 escenarios, 188 exclusiones QA únicas, 787 elegibles y 3.901 artefactos únicos con hashes coincidentes. MLP completó 44/50 trials A-only. El trial 44 obtuvo `macro_network_rmse=123.6570`; el trial 6 conserva el mejor valor con `96.6671`. El trial 45 está en curso. El scoring A→B permanece sellado y no se han consultado targets de B.

## Corte operacional posterior (2026-10-29 21:39 UTC)

B continúa completo y verificado, con 975 escenarios registrados, 188 exclusiones QA únicas, 787 elegibles y 3.901 artefactos con hashes verificados. El ensayo MLP 45 sigue ejecutándose; el proceso conserva uso activo de CPU y memoria. Desde el corte de las 21:09 UTC no se ha recibido resultado nuevo, por lo que el contador confirmado permanece en 44/50 y el mejor trial confirmado es el 6 (`macro_network_rmse=96.6671`). No se reinició la búsqueda ni se consultaron los targets de B; el scoring A→B permanece sellado.

## Corte operacional posterior (2026-10-29 22:12 UTC)

B sigue completo, con 975 escenarios, 188 exclusiones QA únicas, 787 elegibles y 3.901 artefactos verificados por SHA-256. MLP completó 46/50 trials A-only: el trial 45 obtuvo `macro_network_rmse=115.3338` y el trial 46 obtuvo `110.1191`. El trial 6 conserva el mejor resultado con `96.6671`; el trial 47 está en ejecución. El scoring externo A→B permanece cerrado y no se han consultado targets de B.
