# Comprobación de exclusiones de QA en un lote mixto

**Fecha:** 3 de noviembre de 2026. **Propósito:** comprobar que un escenario fallido no contamine los datos ni las métricas de un escenario que sí cumple la política vigente.

Se añadió `test_mixed_qa_stages_and_scores_only_eligible_run` en `tests/database/test_label_free_b_staging.py`. La prueba construye una red y reportes **sintéticos** de dos corridas con protocolo anclado y fuentes selladas. La primera declara continuidad 0,5 %; la segunda, 8,0 %. Se mantiene el límite provisional acordado de 5 %, sin introducir nuevos criterios ni tocar las simulaciones del proyecto.

La prueba ejecuta la importación real sin etiquetas, el scoring real y la unión real de datos usada por las figuras. Comprueba que:

1. La corrida de 8 % queda no elegible, con motivo `continuity_limit_exceeded`, y no se le crean features.
2. Las filas de entrada, las etiquetas leídas y las métricas corresponden únicamente a la corrida elegible; el reporte de scoring registra la exclusión y su motivo.
3. La lectura de etiquetas ocurre después de sellar las predicciones, y no se guardan etiquetas en `node_results`.
4. La unión para las figuras solo contiene el `run_id` elegible y conserva el estado y motivo del excluido en sus metadatos.

**Resultado:** la prueba pasó junto con otras cuatro comprobaciones enfocadas de exclusión, cobertura y coherencia de elegibilidad: **5 pruebas pasaron**. Se emplean lectores de reportes, observables hidráulicos y predictores sustituidos para que el resultado sea controlable: no ejecuta SWMM ni valida la física, y utiliza el mismo conjunto de predicciones sintéticas para representar los dos modelos en la unión de figuras. La prueba construye los archivos de predicción sellados; por ello **no prueba de extremo a extremo la etapa real de inferencia ni el renderizado completo de un lote mixto**. Tampoco cubre en esta integración todas las causas de rechazo ni el caso de cero elegibles. Esas verificaciones siguen pendientes como integración, aunque hay pruebas separadas para algunas de ellas.

El entorno de Windows denegó el directorio temporal creado por pytest (`WinError 5`). La prueba se ejecutó con `outputs/run_qa_mixto_tests.py`, que proporciona directorios temporales dentro de `outputs/qa_test_work`; el aviso se refiere al entorno de pruebas, no a una simulación hidráulica.
