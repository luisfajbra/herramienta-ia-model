# Evidencia congelada

Este paquete guarda las especificaciones, decisiones de ejecución, QA por escenario y resultados verificados del informe [Evaluación de modelos con hidrogramas nuevos en Chico Norte](../../superpowers/reports/2026-11-02-evaluacion-hidrogramas-nuevos-chico-norte.md).

`evaluation_design.json` fija hidrogramas, factores, candidatos, umbrales y hashes. `protocol.json`, `anchor_receipt.json`, `frozen_b_manifest.json` y `experiment_source_manifest.json` enlazan las simulaciones y sus fuentes. `batch_progress.json` conserva los reintentos de drenaje y los hashes de INP, RPT y OUT. Los dos archivos `*_verified_scores.json` guardan el score calculado después de verificar sellos y etiquetas obtenidas de RPT.

Los archivos completos `.inp/.rpt/.out`, CSV de predicción y SQLite siguen en `outputs/experiments/chico_norte_novel_hydrograph_eval_20261102/run_20261103T034108Z/`; el volumen de archivos de simulación es aproximadamente 199 MB. Sus SHA-256 están en los manifiestos de este paquete. Las etiquetas no se escribieron en `node_results`.

La presentación semanal asociada está en la raíz del proyecto: [`Presentacion_MLSMFP_02-11_avance_chico_norte_hidrogramas_nuevos_revision_graficas.pptx`](../../../Presentacion_MLSMFP_02-11_avance_chico_norte_hidrogramas_nuevos_revision_graficas.pptx). El [guion oral](../guion_presentacion_2026-11-02-chico-norte.md) explica las doce diapositivas nuevas y el orden actualizado y aclara el alcance de las conclusiones.
