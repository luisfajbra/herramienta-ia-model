# Verificación documental y estado del proyecto

Fecha: 8 de noviembre de 2026. El investigador solicitó revisar la documentación y preparar una guía para una persona nueva. La revisión se realizó sobre la copia local en la rama `fix/test-suite-contracts-2026-10-27`, con commit base `65d1e170` y cambios locales adicionales. El análisis posterior de resultados hidráulicos y la evaluación en una tercera red permanecen pausados por solicitud del investigador.

## Alcance y límites de la revisión

Se inventariaron automáticamente 172 archivos Markdown (171 bajo `docs/` más el README existentes antes de crear esta revisión y la guía) y se revisaron 277 enlaces locales con un análisis estático sencillo que omite bloques de código y enlaces externos. Se examinaron en detalle los documentos actuales de resultados, evaluación, modelos, protocolo, respaldo y presentación, además del README. Se contrastaron afirmaciones seleccionadas con código, configuración, metadatos, predicciones, figuras y respaldos locales.

**Esto no constituye una certificación del 100 % de todas las afirmaciones de los 172 archivos.** La revisión automática de enlaces no interpreta contexto, no verifica todos los anclajes, expresiones Markdown complejas ni URL externas. No se auditó semánticamente línea por línea cada documento histórico, ni se reejecutaron las simulaciones, el entrenamiento, la suite completa, una instalación limpia o una restauración integral desde Drive. Las fuentes antiguas se conservan como evidencia de sus etapas y deben leerse con su fecha y alcance.

## Correcciones aplicadas

1. **Contratos y objetivos:** el README mezclaba descripciones de 15, 17 y 21 variables y un objetivo histórico de caudal pico con el flujo conjunto de volumen. Se añadió una advertencia al inicio y se identificaron las secciones históricas. Los candidatos conjuntos actuales usan 21 variables, `inunda` y `vol_inundacion_m3`; el flujo principal histórico usa 17 variables.
2. **Comando de predicción:** el README sugería inferencia sin SWMM para `--predict`; `main.py` además ejecuta SWMM para comparación. Se corrigió el texto y se señaló la diferencia en la guía.
3. **Interfaz local:** las instrucciones antiguas `python app.py` y `python view_db.py` apuntaban a archivos que no existen en la raíz de esta copia. Se aclaró que existe código Tkinter en `swmm_resilience/desktop/app.py`, pero su arranque y el visor no se verificaron.
4. **Cota hidráulica:** el informe de la reserva formulaba como universal que el volumen de inundación no puede superar el volumen tributario construido. Se corrigió a comparación exploratoria sin cota universal demostrada y se matizó la explicación de extrapolación; se conservaron las cifras y la condición de análisis post hoc.
5. **Enlaces:** se corrigieron cuatro rutas relativas rotas en dos índices de figuras de la evaluación anterior. Un índice pertenece a una carpeta archivada local; su modificación no cambia las figuras.
6. **Respaldos:** se añadió una actualización a la revisión independiente anterior, que decía pendiente la subida del paquete 33. Ahora consta la subida y se conserva la advertencia de que falta cotejo criptográfico mediante descarga.
7. **Guía:** se creó `docs/GUIA_INTEGRAL_PROYECTO.md`, que distingue las rutas históricas y actuales, explica el flujo y señala qué resultados se pueden atribuir a cada modelo.

## Comprobaciones ejecutadas

- Se recalcularon RMSE desde Parquet guardados y se verificó la correspondencia uno a uno de las claves. Reserva: 106.746 pares en 321 corridas; XGBoost 234,883567 m³ global y 234,543071 m³ macro entre redes; MLP 191,877423 m³ global y 184,630902 m³ macro. Los resultados coincidieron con `scores.json`.
- Escenarios nuevos conjuntos: 5.220 pares en 10 corridas; RMSE global 51,119203 m³ para XGBoost y 27,015368 m³ para MLP. Ambos coincidieron con sus resultados registrados.
- Se verificaron los hashes de las 23 figuras de la prueba de escenarios nuevos; los 23 coincidieron con el manifiesto.
- Se verificaron localmente los ZIP y todos sus miembros frente a sus manifiestos: respaldo 33 de reserva y base, respaldo 44 de la evaluación conjunta de diez escenarios históricos y respaldo 45 de los escenarios nuevos.
- Se ejecutaron 22 pruebas focalizadas de auditoría, cobertura, diagnósticos físicos y controles de alineación y novedad: **22 pasaron**. Hubo 14 advertencias de deprecación de dependencias. No se ejecutó la suite completa.
- Se consultó `main.py --help`, `config.yaml`, `requirements.txt`, `pyproject.toml`, los contratos de variables y scripts concretos de preparación y entrenamiento. El Python observado fue 3.12.10, mientras el script histórico de reproducción exige 3.13. `pyproject.toml` declara ≥3.11 y deja las dependencias de ejecución a `requirements.txt`.
- Se corroboró que los dos candidatos conjuntos declaran las mismas 21 variables y que los scripts de scoring de eventos antiguos apuntan a modelos entrenados solo con Chico Sur. El ejecutable principal no debe describirse como una conexión automática a los candidatos conjuntos.
- Tras las correcciones y la creación de la guía y este informe, se repitió el análisis estático: **174 archivos Markdown, 294 enlaces locales y 0 destinos faltantes** dentro del alcance del verificador descrito arriba.
- Se generó `docs/Guia_integral_proyecto_redes_pluviales.docx` desde la guía Markdown mediante `scripts/generar_guia_docx.py` y la librería estándar de Python. El ZIP y los XML del DOCX se verificaron; Word abrió el documento y reportó 9 páginas y 3.861 palabras. SHA-256 del DOCX: `6caa1bf7fa295145f4aed89534c91dea0aeb991e8489e910c1e34ed27c46de08`. La revisión no incluyó inspección visual individual de cada página. La primera conversión desde HTML con Word quedó bloqueada y se cerró; el DOCX final se creó por Open XML y se comprobó por separado.
- Se compararon 218 fragmentos de texto y celdas del Markdown fuente con el contenido XML del Word: todos están presentes. Esta comprobación cubre el texto, no garantiza la apariencia de cada página ni convierte las rutas mencionadas en enlaces interactivos de Word.

## Pendientes que no se pueden presentar como verificados

- Comparar mediante descarga los SHA-256 de los respaldos de Drive, reconstruir el ZIP multipartes y realizar una prueba de restauración en otro directorio o equipo. La presencia y el tamaño de archivos remotos no demuestran identidad de bytes.
- Revisar científicamente cada afirmación de los informes históricos, fuentes y rangos hidráulicos, y completar una ejecución limpia reproducible. El límite de continuidad del 5 %, las tolerancias y el origen físico de `InitFlow` siguen pendientes de justificación.
- Algunos scripts registran rutas absolutas y contratos de hashes ligados a la versión del código; mover o reanudar requiere conservar fuentes y no debe implicar editar silenciosamente sellos.
- Las nuevas formas CSV y figuras PNG, además de `outputs/`, están ignoradas por Git. Sus copias de Drive y manifiestos se documentan en los informes de resultados, pero no quedan incluidas automáticamente en un commit.
- La [evaluación independiente de los modelos conjuntos en una tercera red](2026-11-08-resultados-escenarios-nuevos-modelos-conjuntos.md) no se ha realizado. La exploración de los resultados nuevos y la tercera red están pausadas; este trabajo fue documental y no modifica decisiones hidráulicas ni modelos.

## Respaldo de esta revisión

La guía Word, el Markdown, este informe, README y las correcciones principales se incluyeron en `outputs/backups/46_guia_y_verificacion_documental_20261108.zip`, verificado localmente con su manifiesto; SHA-256 `3274bc8b9e364cede59809cde0d1e7811ba490b67101b2e41b2a2f24be5577ee`. Se subieron a la carpeta privada de tesis el [ZIP](https://drive.google.com/file/d/1I_-hIc_HAbU1ahq9o82SysFHrO2Oouzz/view?usp=drivesdk), su [manifiesto](https://drive.google.com/file/d/1oghluv5oxui1dWRACusFCQXlzFvbF9-L/view?usp=drivesdk) y una [copia independiente de la guía Word](https://docs.google.com/document/d/1_g2nDec_PdQZqzKjUDVFfPu9LPOvIgbt/edit?usp=drivesdk&ouid=105819791257918831830&rtpof=true&sd=true). Los metadatos remotos confirman presencia, tamaño y carpeta; queda pendiente comparar hashes de descargas. Esta sección se añadió después de generar el ZIP, por lo que el informe dentro del ZIP corresponde a la versión inmediatamente anterior.
