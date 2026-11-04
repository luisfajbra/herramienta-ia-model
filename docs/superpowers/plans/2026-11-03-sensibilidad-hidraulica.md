# Plan de implementación: sensibilidad hidráulica

> **Para quien implemente:** usar `superpowers:executing-plans`; ejecutar las tareas por orden, con pruebas y revisión de cada entrega. Los pasos se marcan solo con evidencia de ejecución.

**Objetivo:** incorporar un análisis optativo y reproducible de sensibilidad del paso hidráulico y del estado inicial en el proyecto.

**Arquitectura:** un protocolo versionado describe casos y variantes; un ejecutor aislado reutiliza SWMM y QA y registra fuentes y salidas; un comparador genera tablas y figuras. Se mantiene independiente del entrenamiento y de los experimentos congelados.

**Tecnología:** Python, PySWMM/SWMM 5.2.4 del entorno documentado, pandas, pytest, JSON, archivos INP/RPT/OUT.

**Diseño:** [sensibilidad hidráulica](../specs/2026-11-03-sensibilidad-hidraulica-design.md). Leer también el [protocolo exploratorio](../reports/2026-11-03-protocolo-sensibilidad-ampliada.md) y el [informe de revisión](../reports/2026-11-03-revision-hidraulica-sensibilidad-chico-norte.md).

## Restricciones generales

- Solo paso de enrutamiento y estado inicial; las perturbaciones físicas quedan fuera.
- QA provisional sin nuevos filtros de FII/no convergencia; registrar también fallos.
- No sobrescribir resultados, alterar modelos, entrenar ni escribir en la base canónica.
- No interpretar el paso máximo como constante ni la sensibilidad como validación física.
- Rutas y hashes verificables; conservar artefactos grandes mediante respaldo separado.

## Mapa de archivos propuesto

| Archivo | Responsabilidad |
|---|---|
| `swmm_resilience/simulation/inflow_closure.py`, `frozen_pair_runner.py` | Corregir cierre compartido y congelar entrada efectiva. |
| `swmm_resilience/simulation/swmm_api_io.py` | Contrato de segundos transcurridos al leer series. |
| `swmm_resilience/simulation/sensitivity_protocol.py` | Validar protocolo y producir variantes autenticadas. |
| `swmm_resilience/simulation/sensitivity_runner.py` | Preparar, ejecutar con activación explícita, registrar y reanudar. |
| `swmm_resilience/simulation/sensitivity_report.py` | Comparación pareada y evidencia; no cambiar QA. |
| `tests/simulation/test_sensitivity_protocol.py`, `test_sensitivity_runner.py`, `test_sensitivity_report.py` | Pruebas independientes de contratos. |
| `docs/experimentos/sensibilidad-hidraulica.md` | Instrucciones, interpretación y preservación para tesis. |

### 1. Cierre y contrato temporal

**Archivos:** `inflow_closure.py`, `frozen_pair_runner.py`, `swmm_api_io.py` y pruebas existentes `test_inflow_closure.py`, `test_frozen_pair_runner.py`, `test_swmm_api_io.py`.

- [ ] Añadir una prueba con dos nodos que compartan malla y valores finales 1 y 0 L/s. Al cerrar, ambos deben compartir la misma malla extendida, el segundo debe mantener 0, la entrada debe permanecer intacta y `event_stats` debe aceptarla. Probar además que no se extienda una malla cuando ningún nodo requiera cierre.
- [ ] Ejecutar `venv\Scripts\python.exe -m pytest tests/simulation/test_inflow_closure.py tests/simulation/test_frozen_pair_runner.py -q`; confirmar que la prueba nueva falla por el problema de malla.
- [ ] Corregir la política y hacer que `prepare_pair` con cierre explícito materialice el evento cerrado antes de congelar; conservar versión de política, fuente, efectivo y hashes. Modo estricto debe seguir rechazando eventos no cerrados. Añadir prueba de extremo a extremo que compruebe lo anterior sin alterar protocolos históricos.
- [ ] Escribir una prueba para nodos y conductos con índice `DatetimeIndex` de `00:01` y `00:02` y origen explícito `00:00`; exigir `time_sec == [60, 120]`. Probar también `TimedeltaIndex`, índice numérico con unidad declarada y rechazo de índice ambiguo. Añadir el origen al contrato sin reinterpretar silenciosamente consumidores anteriores.
- [ ] Corregir la lectura temporal y ejecutar `venv\Scripts\python.exe -m pytest tests/simulation/test_swmm_api_io.py tests/simulation/test_inflow_closure.py tests/simulation/test_frozen_pair_runner.py -q`. Documentar fallos ambientales por separado; no regenerar las 75 corridas históricas en esta tarea.

### 2. Protocolo y variantes aisladas

**Archivos:** crear `sensitivity_protocol.py` y `test_sensitivity_protocol.py`; consultar `review_chico_norte_hydraulics.py` sin trasladar sus identificadores fijos de Chico Norte al módulo general.

- [ ] Definir `load_protocol(path: Path) -> dict` y `build_variants(protocol: dict, source_inp: Path, output_dir: Path) -> list[dict]`. Cada entrada identifica escenario, variante, diferencias permitidas y SHA-256 de fuente y efectivo.
- [ ] Probar una red mínima de dos conductos: la variante de paso modifica solo `ROUTING_STEP`; la inicial modifica solo ambos `InitFlow`; la fuente conserva su hash. Rechazar secciones faltantes, pasos no positivos, máximo menor que mínimo, identidades duplicadas y archivos fuente cuyo hash no coincida.
- [ ] Ejecutar `venv\Scripts\python.exe -m pytest tests/simulation/test_sensitivity_protocol.py -q` y confirmar el fallo inicial. Implementar validación, escritura exclusiva y comparación estructural del INP; ejecutar otra vez y exigir éxito. Guardar en protocolo los niveles realmente elegidos; no inferirlos del nombre.

### 3. Ejecución, QA y reanudación

**Archivos:** crear `sensitivity_runner.py` y `test_sensitivity_runner.py`; reutilizar las comprobaciones de `frozen_pair_runner.py` y el código existente de QA.

- [ ] Definir `prepare(protocol_path: Path, output_dir: Path) -> Path` y `execute(manifest_path: Path) -> None`; la primera valida y crea manifiesto sin SWMM, la segunda ejecuta en proceso aislado y escribe por corrida sus rutas, hashes, diagnóstico y estado `complete` o `failed`. La interfaz CLI `python -m swmm_resilience.simulation.sensitivity_runner --protocol PROTOCOLO --output DIRECTORIO` prepara; `python -m swmm_resilience.simulation.sensitivity_runner --manifest DIRECTORIO/manifest.json --execute` ejecuta el manifiesto preparado.
- [ ] En una fixture sintética con ejecutor falso, probar que preparar no ejecuta SWMM; que un fallo se registra y no se marca completo; que una reanudación verifica todos los hashes y no sobrescribe; que un OUT modificado bloquea el reuso; y que QA no excluye automáticamente por FII ni no convergencia.
- [ ] Ejecutar `venv\Scripts\python.exe -m pytest tests/simulation/test_sensitivity_runner.py -q`, implementar y repetir. Asegurar que el estado de cada corrida se guarda inmediatamente y que la referencia y variantes fallidas se reportan, sin escribir SQLite canónica.

### 4. Comparación e informe

**Archivos:** crear `sensitivity_report.py`, `test_sensitivity_report.py` y `docs/experimentos/sensibilidad-hidraulica.md`.

- [ ] Definir `compare(manifest_path: Path, report_dir: Path) -> Path`. Para cada pareja, verificar identidad, hashes, unidades, nodos y fechas; calcular diferencia total, máxima absoluta por nodo, número de nodos cambiados y cambios de clase al umbral vigente. Registrar diferencias de continuidad y diagnósticos sin asignar un umbral nuevo. Con referencia cero reportar diferencia absoluta y relativo no definido.
- [ ] Probar dos pares fabricados: volúmenes `[0, 2]` contra `[0, 3]` producen diferencia total 1, máxima 1 y un nodo cambiado; cero contra cero no produce infinito. Rechazar nodos faltantes, fechas distintas y fuentes adulteradas. Para pasos nativos, probar que la interpolación nunca extrapola y rechaza intervalos sin intersección.
- [ ] Ejecutar `venv\Scripts\python.exe -m pytest tests/simulation/test_sensitivity_report.py -q`, implementar y repetir. Generar tablas por corrida y gráficas individuales; mapas separados por variante, escala de volumen común y sin símbolos extra. Etiquetar las ventanas elegidas después de observar resultados. Documentar condiciones, unidades, resolución del RPT, costos, comandos y limitaciones científicas.

### 5. Prueba piloto y preservación

- [ ] Preparar una fixture de dos redes pequeñas y dos escenarios con un par de variantes; comprobar que cada red se identifica por su contenido y que una ejecución completa y reanudada genera la misma tabla y manifiesto verificable. Ejecutar `venv\Scripts\python.exe -m pytest tests/simulation/test_sensitivity_protocol.py tests/simulation/test_sensitivity_runner.py tests/simulation/test_sensitivity_report.py -q`.
- [ ] Diseñar un piloto real separado para cada red, con selección, niveles, nodos y presupuesto registrados antes de correr. Si falta una condición inicial o hay un conflicto del protocolo de cierre, registrar bloqueo y no inventar la condición. La elección de niveles adicionales al piloto de 5/1 s requiere revisión de `MINIMUM_STEP` y presupuesto.
- [ ] Publicar índice de evidencia con hash/tamaño para INP, evento, RPT, OUT y tablas. Verificar restauración desde el destino de respaldo elegido; no declarar respaldo remoto hasta comprobarlo. Actualizar el documento de decisiones y el cronograma con resultados y pendientes.

## Salida verificable

Quedan terminadas las tareas solo cuando las pruebas y un piloto controlado pasen; cada fallo o exclusión queda visible. Concluir si el flujo es reproducible y qué diferencias se observaron, sin extrapolar a validación física, entrenamiento conjunto ni red no vista.
