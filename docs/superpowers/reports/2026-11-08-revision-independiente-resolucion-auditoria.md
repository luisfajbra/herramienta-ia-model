# Revisión independiente de la resolución de la auditoría

> **Actualización posterior:** el respaldo 33 y una presentación con la conclusión revisada se subieron a Drive. La comprobación criptográfica por descarga remota seguía pendiente en la [actualización de respaldo](2026-11-08-respaldo-y-presentacion-tras-auditoria.md). Las observaciones de esta revisión sobre pasos pendientes reflejan el momento en que se escribió.

Fecha: 8 de noviembre de 2026. Alcance: contraste del informe `2026-11-08-resolucion-hallazgos-auditoria.md`, los cambios desde `345a7dd0` hasta `65d1e170`, las salidas locales y las pruebas específicas. Se revisó también el inventario de commits aún no presentes en la referencia remota local; no se hizo una auditoría línea por línea de toda la historia anterior. No se modificó código ni se reentrenaron modelos.

## Conclusiones

1. La corrección de interpretación de la reserva es pertinente. El `audit.json` local registra RMSE macro de 234,54 m³ para XGBoost y 184,63 m³ para MLP. Al retirar descriptivamente `shape_quasi_uniform_block`, registra 66,68 y 87,23 m³, respectivamente. Estas últimas cifras son post hoc y no reemplazan la métrica acordada ni constituyen una nueva prueba ciega. No corresponde declarar que un modelo es globalmente superior a partir de esta comparación.
2. El diagnóstico de continuidad compara poblaciones diferentes: pasar del 5 % al 2 % o al 1 % modifica las corridas que entran en la métrica. El aumento de RMSE en el subconjunto no demuestra por sí solo que endurecer la QA empeore los modelos, ni valida hidráulicamente el 5 %. El 5 % sigue siendo provisional conforme a la decisión previa.
3. La cota de volumen tributario es un diagnóstico exploratorio, no una garantía física demostrada. La cadena hidráulica incluye almacenamiento y posibles efectos de remanso; que las redes se representen como árboles no basta para demostrar esa cota. El propio informe reconoce el almacenamiento inicial y la falta de demostración, aunque el docstring de `physical_checks.py` formula la cota con demasiada seguridad. No se recomienda aplicar el recorte a inferencia sin una definición hidráulica válida y evaluación independiente. La ausencia de violaciones de SWMM en esta reserva no prueba validez universal.
4. El análisis de monotonía cuenta descensos de predicción al aumentar el factor cuando SWMM no desciende en ese paso; es una comparación diagnóstica con el simulador, no una demostración general de que la inundación nodal deba crecer monótonamente bajo cualquier condición hidráulica. Sus tolerancias difieren: 1 m³ por nodo y 10⁻⁶ m³ en el total. Los conteos dependen de esas reglas y no son porcentajes de error global.
5. `stratified_shape_draw` es una herramienta para protocolo futuro. Retener las formas de mayor volumen en desarrollo ayuda a evaluar casos dentro de cobertura, pero no sustituye una evaluación explícita de extrapolación. Una nueva partición de las mismas formas ya examinadas tampoco recupera la independencia de una prueba ciega; hace falta un conjunto independiente predefinido para conclusiones confirmatorias.
6. La cola de drenaje permanece sin corregir por decisión registrada. El informe identifica 175 corridas excluidas afectadas. Mantenerlas excluidas evita incorporarlas al conjunto actual, pero no elimina el problema del generador ni justifica relajar la QA; es necesario decidir y documentar el tratamiento antes de admitir esas formas en futuros experimentos.

## Respaldo y reproducibilidad

- Se verificó localmente `outputs/backups/33_reserva_conjunta_y_base_20261108.zip` con `verify_archive`: los 21 miembros y sus hashes coinciden con el manifiesto; el SHA-256 del ZIP es `2f1ea8857d45e5fcefa6eb2faad435b35ffb2f869f8168ea163fe78b049b4022`.
- El ZIP contiene la evaluación, las bases, la partición y el análisis de auditoría; no contiene los modelos finales ni los estudios Optuna. Según el informe existen respaldos separados en Drive, pero esta revisión no volvió a contrastar sus hashes. La subida del ZIP 33 a Drive no se verificó. Una copia en el mismo equipo no constituye respaldo frente a la pérdida del equipo.
- La afirmación del informe de que los agentes no pueden subir archivos a Drive no se verificó como limitación general: depende de las herramientas y permisos disponibles. Lo comprobado aquí es que la copia remota del paquete 33 queda pendiente de verificación.
- Limitación adicional del archivador: al recibir una base SQLite individual como `--source`, `_members` solo incluye ese archivo y no examina automáticamente un `-wal` hermano. El rechazo de WAL no vacío sí cubre los `-wal` encontrados al archivar una carpeta. Por tanto, no debe afirmarse que protege todos los usos posibles con bases activas. Este punto no invalida la comprobación interna del ZIP 33 ni demuestra que ese ZIP sea inconsistente; sí exige precaución en respaldos futuros y validación de escritores activos.
- La auditoría v1 se generó antes de las últimas protecciones de partición y apertura inmutable; el informe registra esta procedencia. Verificar y preservar también la versión congelada del código y los sellos de modelos, no solo el JSON de resultados.

## Commits y pruebas

- La rama activa es `fix/test-suite-contracts-2026-10-27`, con HEAD `65d1e170`. Respecto a la última referencia remota local hay 39 commits por subir; esa comparación no comprueba cambios remotos posteriores a la última actualización. Los nueve commits de la auditoría son `345a7dd0`, `0ebda567`, `b59cc500`, `602ff5f9`, `020af994`, `7a9a35e4`, `d158e575`, `86cea273` y `65d1e170`.
- La prueba focalizada de cinco archivos de tests recogió 35 pruebas: **35 pasaron**. El primer intento dentro del sandbox no pudo acceder a los directorios temporales de pytest; el segundo se ejecutó fuera de esa restricción y terminó correctamente, con advertencias de deprecación de dependencias. No se ejecutó la suite completa.
- Los archivos rastreados no tenían cambios al comprobar `git status --short --untracked-files=no`. Existen presentaciones y otros archivos sin rastrear, además de este informe; el respaldo de datos y los archivos ignorados no quedan incluidos automáticamente al subir commits.

## Prioridades propuestas

1. Preservar el ZIP 33 y su manifiesto en almacenamiento independiente, verificar la copia y cotejar los modelos y estudios con sus sellos existentes.
2. Corregir la presentación que afirma una ventaja general del MLP y explicar la dependencia de la forma extrema; conservar el dato oficial y rotular la exclusión como post hoc.
3. Antes de adoptar recortes, monotonía, una nueva reserva o un cambio de QA, definir la pregunta de investigación y un protocolo independiente, respetando las decisiones del investigador y el estado provisional de las condiciones iniciales.
4. Antes de nuevos lotes que usen formas sin recesión a cero, decidir el tratamiento de la cola y comprobar que las variables describen el hidrograma efectivamente simulado.
