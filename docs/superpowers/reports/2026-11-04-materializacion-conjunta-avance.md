# Avance de materialización conjunta

**Fecha:** 4 de noviembre de 2026. Se desarrolla una base supervisada nueva para Chico Sur y Chico Norte según el [diseño aprobado](../specs/2026-11-03-base-conjunta-verificable-design.md). Todavía no se ha materializado la base real ni se han entrenado modelos.

## Pruebas controladas

Se implementó un constructor que toma bases y manifiestos explícitos, crea snapshots SQLite mediante respaldo en lectura, revalida las fuentes y variables, extrae etiquetas elegibles de Chico Norte y exige que su hash coincida con el resultado externo congelado indicado por el usuario. Genera una base y un manifiesto nuevos; nunca reutiliza un directorio existente y marca el resultado `materialized_not_training_ready`. Conserva los campos auxiliares de resultados cuando existen, remapea los IDs locales y registra la relación entre los enlaces originales y los nuevos. Los escenarios excluidos mantienen sus dictámenes, pero no aportan filas de entrenamiento.

Se verificó con fuentes sintéticas que la unión admite IDs locales coincidentes en redes distintas, rechaza etiquetas incompletas o un hash incorrecto, mantiene las razones de QA y revierte la escritura cuando falla el enlace a un artefacto. Otra prueba comprobó que los cambios pendientes en el WAL de SQLite se detectan tanto en la base conjunta como en las copias de las fuentes; verificar únicamente el hash del archivo principal no basta en ese caso. Los hashes son controles de integridad, no firmas contra manipulación deliberada.

El 4 de noviembre se ejecutaron 172 pruebas de las fuentes supervisadas, la preparación, los esquemas, las restricciones de importación y las interfaces de línea de comandos: **172 pasaron**. Incluyen pruebas sintéticas y no sustituyen la verificación de la base real.

## Falta para entrenamiento

El constructor no equivale todavía a un pipeline listo para entrenar: se está probando la integración del nuevo manifiesto con el preparador multirred sin fabricar procedencia; también faltan la evaluación integral, materializar las fuentes reales y acordar las particiones y el presupuesto. El recálculo real de variables de Chico Norte terminó el 4 de noviembre: **410.814 filas elegibles** coincidieron con las reconstruidas desde sus fuentes; el resultado persistente está en `outputs/experiments/preflight_chico_norte_features_20261104.json`. La evaluación anterior y los diez escenarios de presentación permanecen separados.

Los avisos sobre escalas de color independientes y factores redondeados en títulos se anotaron en el [informe de figuras](2026-11-03-evaluacion-eventos-seleccionados.md); el investigador decidió no cambiar las figuras en esta etapa.

La integración del paquete con el preparador está en prueba: usa una copia de bytes de SQLite que debe coincidir con el hash sellado, comprueba que no haya WAL pendiente y conserva en la procedencia el manifiesto conjunto y los enlaces por corrida. Se comprobó con fuentes sintéticas que una modificación durante la captura detiene la preparación y que no se sobrescribe un directorio de salida existente. La prueba de extremo a extremo usa dos redes sintéticas y sustituye algunas entradas hidráulicas para obtener un ejemplo pequeño; ajusta versiones mínimas de MLP y XGBoost exclusivamente para comprobar la conexión y el registro de procedencia. Esos modelos no se usan como candidatos de tesis ni proporcionan métricas científicas. Los scripts `scripts/materialize_joint_dataset.py` y `scripts/prepare_joint_dataset.py` requieren fuentes, hashes y protocolos explícitos, y no entrenan ni promueven modelos.

**Inferencia futura pendiente:** `register_transfer` exige actualmente un candidato entrenado con una sola red y evidencia de selección para esa red. Un modelo conjunto de Chico Sur y Chico Norte no puede conectarse sin cambios a ese flujo de transferencia. Será necesario diseñar y probar una ruta multired antes de afirmar que se pueden hacer predicciones con ese modelo en otra red; no se debe relajar la barrera existente sin una alternativa verificable.

## Secuencia de uso y límites

1. Ejecutar `scripts/materialize_joint_dataset.py` con las rutas de las dos bases y sus manifiestos, los SHA-256 de ambos INP, el hash de etiquetas de Chico Norte y un directorio de salida nuevo. La materialización une fuentes verificadas; no selecciona particiones ni entrena.
2. Comprobar que el estado del paquete sea `materialized_not_training_ready` y conservar su manifiesto y los snapshots. Este estado no acredita por sí solo la aptitud para entrenamiento.
3. Acordar y registrar la reserva de formas de hidrograma, las agrupaciones de validación y el presupuesto antes de preparar los datos para entrenar. `scripts/prepare_joint_dataset.py` exige un protocolo explícito, pero **no decide ni impone por sí solo una reserva científica**: preparar todos los escenarios elegibles y entrenar con ellos consumiría la posible prueba reservada. `GroupKFold5` agrupa factores y no equivale a reservar formas.
4. Entrenar y evaluar únicamente según esa partición acordada. Los diez escenarios de presentación permanecen separados y, al haberse examinado, no constituyen una prueba final ciega. Para evaluar transferencia del modelo conjunto a una red no vista hace falta una tercera red y adaptar el flujo de transferencia.
