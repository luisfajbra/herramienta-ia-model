# Resolución de los hallazgos de la auditoría hidráulica y de ML

**Alcance.** Este documento cuenta, hallazgo por hallazgo, qué se observó, qué se decidió, qué se hizo y qué queda por hacer. Está escrito para el investigador y para el siguiente agente que retome el trabajo. Se refiere a la [auditoría del 4 al 8 de noviembre de 2026](2026-11-08-auditoria-hidraulica-ml.md) y al [plan de resolución](../plans/2026-11-08-resolver-hallazgos-auditoria.md), ejecutado en la rama `fix/test-suite-contracts-2026-10-27` (sin push). El plan se ejecutó en seis tareas, una por commit; esta documentación es la sexta.

Reglas que valen para todo el documento:

- **Todo resultado nuevo sobre la reserva es post hoc.** Se rotula `post_hoc_audit_not_model_selection`. Nunca reemplaza las métricas oficiales de `scores.json`, y la métrica principal sigue siendo `macro_network_rmse`.
- **La reserva no se reutiliza ni se vuelve a sortear.** Ya fue consultada, por lo que no sirve como prueba ciega. No hay reentrenamiento ni selección de modelos en este plan.
- **Los artefactos sellados no se escriben.** Se abren en solo lectura; las salidas nuevas van a directorios nuevos con creación exclusiva. Los scripts ya ejecutados (`scripts/*_20261104.py` y `scripts/*_20261108.py`) no se modifican.
- **Las rutas bajo `outputs/` no están versionadas** (`.gitignore` ignora `outputs/*`). Se escriben como código, no como enlaces, porque en otro clon no existen.
- Constantes de la evaluación oficial: umbral de inundación 1,0 m³; bandas de volumen 24, 108 y 344 m³; umbral de probabilidad 0,5; SHA-256 de la partición `b81fd31efa6a0b28b51e66e7bdfb6255d80de37ea25f0cb57714010bb2e892f4`.
- En las cifras, red 1 es Chico Sur y red 2 es Chico Norte. Las cifras con coma decimal salen de `outputs/experiments/auditoria_reserva_conjunta_20261108_v1/audit.json` (en adelante, `audit.json`), redondeadas.
- Agregar archivos `.py` bajo `swmm_resilience/` cambia el conjunto de hashes del contrato de los estudios Optuna (`implementation_sha256`). Los estudios ya terminaron. Cualquier intento de reanudarlos debe usar la versión congelada del código, no la actual.

## Resumen

| Hallazgo | Estado | Commits | Evidencia |
|---|---|---|---|
| 1. La ventaja de MLP depende de una sola forma | Atendido con análisis post hoc y conclusión corregida | `0ebda567`, `020af994` y el commit de documentación (`docs: report post hoc holdout sensitivity and audit status`) | `audit.json` (`by_shape_network`, `leave_one_shape_out`), [informe de la reserva](2026-11-08-evaluacion-reserva-conjunta.md) |
| 2. El sorteo dejó lo más extremo fuera del entrenamiento | Atendido con diagnóstico de cobertura y herramienta para el futuro; la reserva no se toca | `602ff5f9`, `020af994`, documentación | `audit.json` (`inflow_coverage`), sección "Sensibilidad post hoc" del informe de la reserva |
| 3. Cola de drenaje como rampa | **Fuera de alcance por decisión del investigador (8 de noviembre)**; se considera diseño de la red y se mantiene | Ninguno | Sección 4 de la auditoría |
| 4. MLP extrapola de forma exponencial | Atendido con diagnóstico físico; el recorte y el reentrenamiento quedan pendientes de decisión | `b59cc500`, `020af994`, documentación | `audit.json` (`tributary_bound`, `total_monotonicity_violations`, `node_monotonicity`) |
| 5. Límite de continuidad del 5 % | Atendido con análisis de sensibilidad; la QA no se cambia | `0ebda567`, `020af994`, documentación | `audit.json` (`continuity_sensitivity`) |
| 6. Artefactos sellados sin respaldo | Respaldo local creado y verificado; **subida a Drive pendiente** (la hace el investigador) | `7a9a35e4` | `outputs/backups/33_reserva_conjunta_y_base_20261108.zip` y `.manifest.json` |
| 7. Detalles menores de código | Punto de SQLite retirado tras revisarlo; nota de tests en el README | Commit de documentación | [README](../../../README.md#verificacion) |

Cuatro puntos requieren acción o decisión del investigador:

1. Subir el ZIP y su manifiesto a Drive y confirmarlo (hallazgo 6).
2. Aprobar o no el cambio de texto de la diapositiva que todavía afirma que MLP gana en volumen (véase "Presentación" más abajo).
3. Decidir sobre las cuatro "Decisiones pendientes" del final.
4. Si alguna vez se retoma el hallazgo 3, decidir entre las dos correcciones propuestas.

## Hallazgo 1: la ventaja de MLP depende de una sola forma

### Problema

El [informe de la reserva](2026-11-08-evaluacion-reserva-conjunta.md) concluía que "en esta reserva MLP tiene menor error de volumen". Las cifras macro son correctas: 234,54 m³ para XGBoost y 184,63 m³ para MLP (`macro_network_rmse_m3`), y se reproducen exactamente desde las predicciones selladas. Lo engañoso es la interpretación. Al excluir una sola forma, `shape_quasi_uniform_block`, el orden se invierte:

| RMSE macro (m³) | Todas las formas | Sin `shape_quasi_uniform_block` |
|---|---:|---:|
| XGBoost | 234,54 | 66,68 |
| MLP | 184,63 | 87,23 |

RMSE por forma y red (m³), de `models.<familia>.by_shape_network`:

| Forma | XGB Sur | XGB Norte | MLP Sur | MLP Norte |
|---|---:|---:|---:|---:|
| common_storm | 49,3 | 48,0 | 39,5 | 35,3 |
| rare_extreme | 2,7 | 5,8 | 4,0 | 4,4 |
| shape_delayed_abrupt_peak | 40,8 | 40,8 | 28,1 | 27,2 |
| shape_flashy_long_tail_9h5 | 20,2 | 23,1 | 25,5 | 23,1 |
| shape_front_loaded_broad | 24,1 | 21,3 | 29,9 | 28,7 |
| **shape_quasi_uniform_block** | **583,8** | **563,0** | **398,2** | **443,0** |
| shape_twin_plateau_15h | 147,0 | 130,0 | 183,2 | 203,9 |

**Por qué ocurre.** El RMSE de cada red es la raíz de un promedio de errores al cuadrado, y el macro es el promedio de los dos. Elevar al cuadrado hace que un error grande pese mucho más que varios errores moderados. `shape_quasi_uniform_block` tiene un error de 400 a 580 m³ mientras que las demás formas están entre 2 y 200 m³, y por eso concentra el 93,38 % de la suma de errores cuadrados agrupada de XGBoost y el 81,32 % de la de MLP (cifras del informe de la reserva). La métrica principal mide, casi solo, qué tan bien se predice esa forma. Por forma el resultado está repartido: MLP gana en `common_storm`, `shape_delayed_abrupt_peak` y `shape_quasi_uniform_block`; XGBoost gana en `shape_twin_plateau_15h` y `shape_front_loaded_broad`; en `rare_extreme` y `shape_flashy_long_tail_9h5` hay empate o ventaja según la red. Esto contradice la idea de que MLP "extrapola mejor" en general.

### Plan

- Corregir la redacción de la conclusión y agregar la tabla por forma y red al informe de la reserva. Solo cambia documentación.
- **No** cambiar la métrica principal acordada: eso sería un ajuste post hoc de la regla de evaluación. La tabla y la sensibilidad "sin una forma" se reportan junto a ella, rotuladas como descriptivas.
- No se elige un modelo ganador ni se reentrena.
- La diapositiva de la presentación que repite la afirmación vieja no se toca sin permiso (véase "Presentación").

### Solución aplicada

Código nuevo (funciones puras, sin tocar artefactos):

- [`swmm_resilience/ml/bench/holdout_audit.py`](../../../swmm_resilience/ml/bench/holdout_audit.py): `macro_network_rmse`, `rmse_by_shape_network` (RMSE y proporción del error cuadrado por forma y red), `leave_one_shape_out_macro` (macro excluyendo cada forma; rechaza excluir una red completa) y `continuity_subset`.
- [`scripts/auditar_reserva_conjunta.py`](../../../scripts/auditar_reserva_conjunta.py): verifica los sellos, reconstruye la tabla de predicciones, comprueba que el macro recalculado coincida con el sellado y escribe `audit.json` en un directorio nuevo.

Tests: [`tests/ml/bench/test_holdout_audit.py`](../../../tests/ml/bench/test_holdout_audit.py) (7 casos) y [`tests/test_auditar_reserva_conjunta.py`](../../../tests/test_auditar_reserva_conjunta.py) (6 casos, comunes a los hallazgos 1, 2, 4 y 5).

Para reproducirlo:

```bash
venv/Scripts/python.exe -m pytest tests/ml/bench/test_holdout_audit.py tests/test_auditar_reserva_conjunta.py -v -p no:cacheprovider --basetemp C:/Users/<usuario>/AppData/Local/Temp/pt-h1

venv/Scripts/python.exe scripts/auditar_reserva_conjunta.py \
  --evaluation outputs/experiments/evaluacion_reserva_conjunta_20261108_v1 \
  --package outputs/experiments/joint_chico_sur_norte_20261104_v1 \
  --partition outputs/experiments/particion_chico_sur_norte_20261104.json \
  --partition-sha256 b81fd31efa6a0b28b51e66e7bdfb6255d80de37ea25f0cb57714010bb2e892f4 \
  --output outputs/experiments/auditoria_reserva_conjunta_20261108_v1
```

El script falla si el directorio de salida ya existe (para repetir la corrida hay que usar un directorio nuevo). Cifras resultantes (`audit.json`, `models.<familia>`):

- `macro_network_rmse_m3`: 234,54 (XGBoost) y 184,63 (MLP), idénticos a `scores.json`.
- `leave_one_shape_out`: 66,68 y 87,23 al excluir `shape_quasi_uniform_block`. Al excluir cualquiera de las otras seis formas el macro cambia poco respecto al valor con todas: entre 244,14 y 254,66 m³ para XGBoost y entre 182,96 y 200,36 m³ para MLP.

El informe de la reserva quedó corregido: la conclusión ahora dice que la ventaja de MLP depende de una sola forma e incluye la tabla.

### Qué queda pendiente y cómo continuar

- Si se vuelve a citar el RMSE macro en la tesis o en una presentación, acompañarlo de la tabla por forma y de la cifra "sin `shape_quasi_uniform_block`".
- Aprobar el cambio de texto de la presentación (véase "Presentación").
- Menores diferidos de `holdout_audit.py` (no cambian resultados; conviene resolverlos en una revisión final):
  - El chequeo de tipo del límite de continuidad (`max_abs_pct` en `continuity_subset`) rechaza escalares de numpy que no sean `float` y acepta `True` y `False`.
  - `continuity_subset` puede dejar sin filas a una red completa sin avisar, lo que cambiaría el significado del RMSE macro. El script registra corridas y pares de cada subconjunto, y con los límites usados (2 % y 1 %) ambas redes siguen presentes.
  - `_require` no rechaza volúmenes `NaN`.
  - Faltan tests para un conjunto vacío y para el caso en que la suma de errores cuadrados es cero.
- Menores diferidos de `scripts/auditar_reserva_conjunta.py`:
  - La protección de salida solo comprueba que el directorio no esté dentro de `--evaluation` ni `--package`; no cubre `final_joint_candidates_20261108_v1` ni los JSON de partición y reserva.
  - `audit.json` se crea antes de serializar su contenido. Si la serialización fallara, quedaría un archivo parcial que bloquearía repetir la corrida hasta borrarlo a mano.
  - Un valor incorrecto de `status` se informa como "sealed scores changed", y una familia ausente lanza `KeyError` en vez de un mensaje claro.
  - No hay tests para un sello de predicciones alterado, para la comprobación de reproducción del macro (`audit_family`) ni para `main()`.

## Hallazgo 2: el sorteo dejó lo más extremo fuera del entrenamiento

### Problema

El sorteo de formas reservadas (`random.Random(42).sample`) es correcto y estaba fijado de antemano. Por azar dejó en la reserva la forma de mayor volumen de entrada de toda la población (`shape_quasi_uniform_block`) y la tercera (`shape_twin_plateau_15h`). La mayor entrada de desarrollo es `shape_sustained_plateau_14h`. Con `inflow_coverage` de `audit.json`:

| Red | Máximo de desarrollo (m³) | Máximo de la reserva (m³) | Corridas de la reserva sobre el máximo de desarrollo | `max_ratio_to_development_max` |
|---|---:|---:|---:|---:|
| 1 (Chico Sur) | 225.639 (`shape_sustained_plateau_14h`) | 262.697 (`shape_quasi_uniform_block`) | 4 | 1,164 |
| 2 (Chico Norte) | 670.283 (`shape_sustained_plateau_14h`) | 780.366 (`shape_quasi_uniform_block`) | 4 | 1,164 |

(La auditoría citaba 780.411 m³ en Chico Norte porque lo tomó del "External Inflow" del RPT; `audit.json` suma las entradas integradas de las variables. La diferencia es menor al 0,01 % y la razón es la misma.)

**Por qué ocurre.** Con solo 7 formas reservadas, el azar puede dejar en la reserva la de mayor volumen. En factor 5, esa forma introduce 16 % más agua que cualquier evento de desarrollo, y cerca del 70 % de la entrada se pierde como inundación (cifra de la auditoría), de modo que la red trabaja limitada por la capacidad de su salida. Los modelos nunca vieron un caso así. El problema no es solo que la etiqueta salga de rango: solo 18 pares nodo–corrida de la reserva superan la etiqueta máxima de desarrollo y explican el 5,8 % del error de XGBoost (cifra de la auditoría). En esa forma, XGBoost predice entre el 28 % y el 49 % del volumen total de SWMM desde el factor 1,4, donde las variables aún están en rango. Una hipótesis, sin probar, es que las variables no capturan la duración sostenida sobre la capacidad: en una red saturada, el volumen inundado es aproximadamente la entrada menos la capacidad de salida multiplicada por la duración.

Esto probablemente explica la brecha entre la validación cruzada interna y la reserva. Sin esa forma, el RMSE macro de la reserva (66,68 y 87,23 m³) queda cerca del de la selección interna (58,89 y 74,73 m³). La comparación es post hoc.

### Plan

- **No** volver a sortear ni rediseñar esta reserva: ya fue consultada y rehacerla invalidaría la cadena de sellos.
- Declarar la limitación en el informe: la reserva mide en parte extrapolación.
- Dejar construida la herramienta para que una reserva futura se diseñe revisando la cobertura antes de sortear: medir cobertura de volumen de entrada y sortear una forma por estrato de volumen, dejando siempre la mayor en desarrollo.
- No hay reentrenamiento. Diseñar una reserva nueva es una decisión pendiente (véase al final).

### Solución aplicada

- [`swmm_resilience/ml/bench/partition_coverage.py`](../../../swmm_resilience/ml/bench/partition_coverage.py):
  - `run_inflow_volumes`: suma el volumen de entrada de los nodos por corrida (y valida identidad y finitud).
  - `inflow_coverage`: por red, máximo de desarrollo, máximo de la reserva, cuántas corridas de la reserva superan el máximo de desarrollo y su razón. Rechaza corridas solapadas o desconocidas.
  - `shape_representative_volume`: volumen por forma normalizado por el máximo de su red.
  - `stratified_shape_draw`: sorteo reproducible de una forma por estrato de volumen, reservando las mayores para desarrollo. **Es solo para un diseño futuro; no se usó sobre esta reserva.**
- `scripts/auditar_reserva_conjunta.py` calcula `inflow_coverage` con los volúmenes de la base conjunta.

Tests: [`tests/ml/bench/test_partition_coverage.py`](../../../tests/ml/bench/test_partition_coverage.py) (7 casos) y los 6 del script de auditoría.

```bash
venv/Scripts/python.exe -m pytest tests/ml/bench/test_partition_coverage.py -v -p no:cacheprovider --basetemp C:/Users/<usuario>/AppData/Local/Temp/pt-h2
```

La cifra se obtiene con el mismo comando de auditoría del hallazgo 1 (`inflow_coverage` en `audit.json`): la reserva supera en 16,4 % al máximo de desarrollo en ambas redes. Está descrito en la sección "Sensibilidad post hoc" del informe de la reserva.

### Qué queda pendiente y cómo continuar

- Decidir si se diseña una reserva nueva (decisión pendiente 3). Si se hace: ejecutar `inflow_coverage` y `stratified_shape_draw` **antes** de sortear, con una semilla registrada, y registrar el procedimiento antes de ejecutarlo, como se hizo el 4 de noviembre.
- Probar la hipótesis de la duración sostenida sobre la capacidad como variable nueva. Solo vale para un reentrenamiento con una prueba independiente nueva.
- Menores diferidos de `partition_coverage.py`:
  - `stratified_shape_draw` calcula un corte negativo si `keep_top_in_development` es mayor que el número de formas; falta una verificación explícita que dé un error claro.
  - `run_inflow_volumes` no rechaza filas duplicadas (corrida, nodo), y el conteo de identidades ignora los `NaN`.
  - `inflow_coverage` y `shape_representative_volume` suponen entradas limpias (por ejemplo, un máximo de cero o un índice duplicado no se validan).
  - La semilla no se valida como entero.
  - Los datos de prueba usan volúmenes iguales en todos los nodos, así que los tests no distinguen sumar de tomar el máximo, y el test de cobertura omite algunas columnas.
- Menores diferidos del script de auditoría: los cuatro listados en el hallazgo 1.

## Hallazgo 3: la cola de drenaje es una rampa cuando el CSV no termina en cero

### Problema

En [`swmm_resilience/simulation/timeseries_scenario.py`](../../../swmm_resilience/simulation/timeseries_scenario.py) (líneas 122 a 124), al escribir cada serie de entrada se agrega un punto final de caudal cero:

```python
data = list(scenario.node_series[node_id])
if drain_down_hours > 0:
    data.append((scenario.last_time_hours + drain_down_hours, 0.0))
```

**Por qué ocurre.** SWMM interpola linealmente entre los puntos de un `[TIMESERIES]`. Si el último valor del CSV ya es cero, el punto agregado no tiene efecto y se obtiene el drenaje buscado. Si el CSV termina con caudal distinto de cero, el punto agregado crea una rampa: el caudal de entrada baja linealmente desde ese valor hasta cero a lo largo de todo `drain_down_hours`. No hay período de drenaje, sino varias horas más de entrada. El bucle de reintentos de [`batch.py`](../../../swmm_resilience/simulation/batch.py) (y el de `frozen_pair_runner.py`) alarga `drain_down_hours`, lo que solo alarga la rampa, de modo que nunca converge. `write_scenario_inp` ya emite un `warnings.warn` en este caso, pero no se detiene.

Evidencia de la auditoría (volumen del CSV frente a "External Inflow" del RPT):

| Grupo | Corridas | Exceso de entrada en SWMM frente al CSV |
|---|---:|---|
| Elegibles | 1.653 | media 0,00 %, máximo 0,18 % (redondeo) |
| Excluidas, CSV termina en cero | 147 | media 0,00 % |
| Excluidas, CSV termina > 1 % del pico | **175** | **16 % a 41 %** |

Las 175 corridas son de las formas `base` (solo en Chico Sur), `long_moderate`, `shape_2h` y `shape_5h`, en todos sus factores. Todas quedan como `drainage_not_drained`. Por ejemplo, la corrida 13 (`base`, Chico Sur) termina el CSV en 2,32 L/s a las 3:00 y el INP lleva el cero a las 13:00.

Impacto: ninguno en los resultados actuales, porque la QA excluye las 175 corridas y ninguna entra en desarrollo ni en la reserva. Se pierden cuatro formas completas (8,9 % de la población) y el motivo registrado no revela la causa. Riesgo latente: si alguien relaja la QA de drenaje, esas corridas tendrían etiquetas calculadas con 16 % a 41 % más agua que la que describen sus variables.

### Plan

**Fuera de alcance por decisión del investigador del 8 de noviembre. Se considera diseño de la red y se mantiene.** No se modifican `timeseries_scenario.py`, `batch.py` ni `frozen_pair_runner.py`, y tampoco las condiciones iniciales (`InitFlow`) de los INP. No hay ningún commit asociado.

### Solución aplicada

Ninguna, por la decisión anterior. Lo único realizado es documentar el mecanismo (arriba) y la corrección propuesta (abajo), para que el efecto sobre las 175 corridas no se pierda de vista.

### Qué queda pendiente y cómo continuar

Solo si el investigador retoma el tema. La auditoría propuso elegir una de estas opciones:

1. Agregar el cero un paso después del último punto del CSV (por ejemplo, `último + 1 s`, o el paso de reporte), para que la entrada caiga a cero y el resto del período sea drenaje real.
2. Rechazar el escenario si el CSV no termina en cero, y que el autor del CSV agregue su recesión.

Cualquiera de las dos cambia la entrada que SWMM recibe en esas formas. Requiere decisión del investigador, una nueva simulación de las 175 corridas, y debe registrarse como un cambio de protocolo. Con la opción 1, la integral de las variables y la del INP deben seguir coincidiendo; el cotejo `scripts/cotejo_fuentes_bloque_20261108.py` ya exige que la cola valga cero. Antes de citar resultados históricos, conviene comprobar si estudios anteriores a la QA de drenaje (migración 009) usaron estas formas: la auditoría no lo revisó.

## Hallazgo 4: el MLP extrapola de forma exponencial en m³

### Problema

Cifras de la auditoría y de `audit.json`:

- Mayor predicción nodal del MLP: 17.440 m³ en Chico Sur (nodo `52C`, corrida 699) y 21.193 m³ en Chico Norte (nodo `268I`, corrida 1675, donde SWMM da 6.814 m³). Las etiquetas máximas de la reserva son 7.418 y 14.928 m³. XGBoost nunca predice por encima de lo visto: sus máximos son 6.098 y 14.068 m³.
- Cota física: el volumen inundado de un nodo no puede superar el volumen que le llega de su propia entrada y de sus nodos aguas arriba (las redes son árboles). Con tolerancia de 1 m³, `tributary_bound.violations` es **0 para XGBoost y 46 para MLP**, con `max_ratio` de 2,27 (el MLP llega a 2,27 veces la cota).
- Monotonía del total: el volumen total predicho por MLP para `shape_quasi_uniform_block` en Chico Sur **baja** 3.170,65 m³ al pasar al factor 5,0 (de 185.554 a 182.383 m³) mientras SWMM sube. Es la única violación (`total_monotonicity_violations`: 1 fila para MLP, 0 para XGBoost).
- Monotonía nodal (`node_monotonicity`): la predicción baja al subir el factor, sin que SWMM baje, en 63 de 101.972 pasos para MLP y en 1.242 para XGBoost (1,2 %).

**Por qué ocurre.** El regresor aprende `log1p(volumen)`. Una red con activaciones ReLU es lineal por tramos, así que extrapola de forma aproximadamente lineal en el espacio logarítmico. Un crecimiento lineal en `log1p` equivale a un crecimiento exponencial en m³, y por eso el MLP sobrepasa a SWMM fuera del rango de entrenamiento. Los árboles de XGBoost, en cambio, predicen promedios de valores ya vistos, así que se saturan y se quedan cortos. La parte de monotonía de XGBoost (1.242 pasos) es otra causa: los saltos de los árboles. Ninguno de los dos modelos tiene garantía física de monotonía respecto al factor.

### Plan

- Diagnosticar y documentar como limitación. Los diagnósticos usan las predicciones selladas y no las cambian.
- El recorte a la cota tributaria se calcula solo para medir cuánto mejoraría el MLP. **No se adopta** en la inferencia: cambiaría las predicciones oficiales y no debe aplicarse con efecto retroactivo a la reserva consultada (decisión pendiente 1).
- No se reentrena con restricciones de monotonía (decisión pendiente 2).

### Solución aplicada

- [`swmm_resilience/ml/bench/physical_checks.py`](../../../swmm_resilience/ml/bench/physical_checks.py):
  - `tributary_bound_violations`: filas donde el volumen predicho supera la cota tributaria más la tolerancia, con su razón.
  - `cap_at_tributary_volume`: recorte elemental a la cota (solo para sensibilidad).
  - `total_monotonicity_violations`: marca los pasos donde el total predicho de la red baja y el total de SWMM no.
  - `node_monotonicity_violations`: cuenta pasos y violaciones por nodo y forma.
- Tests: [`tests/ml/bench/test_physical_checks.py`](../../../tests/ml/bench/test_physical_checks.py) (4 casos) y los 6 del script de auditoría.

```bash
venv/Scripts/python.exe -m pytest tests/ml/bench/test_physical_checks.py -v -p no:cacheprovider --basetemp C:/Users/<usuario>/AppData/Local/Temp/pt-h4
```

El comando de auditoría es el del hallazgo 1. Resultado: el recorte baja el RMSE macro de MLP de 184,63 a 174,29 m³ (`capped_macro_network_rmse_m3`); en XGBoost no hay violaciones y queda en 234,54 m³. Ese recorte está rotulado "recorte post hoc, no adoptado" en el informe de la reserva.

### Qué queda pendiente y cómo continuar

- Decisión pendiente 1 (adoptar el recorte en la inferencia de modelos futuros) y 2 (reentrenar con `monotone_constraints` de XGBoost sobre las variables de volumen, o con otra representación del volumen para el MLP). Requieren una prueba independiente nueva.
- Mientras tanto, citar las predicciones de MLP fuera del rango de entrenamiento como una limitación conocida.
- Menores diferidos de `physical_checks.py`:
  - Los tests de monotonía usan una sola forma y entradas ordenadas; un error en la clave de agrupación pasaría inadvertido.
  - `tributary_bound_violations` no valida volúmenes tributarios `NaN` o cero. La columna no admite nulos en `tabular_v4_21`, así que hoy no ocurre.
  - Si hubiera corridas duplicadas por (red, forma, factor), la monotonía se distorsionaría. La reserva tiene exactamente una corrida por factor (25 factores por forma y red), así que no ocurre.
  - `cap_at_tributary_volume` no tiene un docstring que diga que es solo para sensibilidad.
- Menores diferidos del script de auditoría: los cuatro listados en el hallazgo 1.

## Hallazgo 5: el límite de continuidad del 5 % es laxo para etiquetas pequeñas

### Problema

La QA admite corridas con error de continuidad de hasta 5 % (máximo admitido: 4,98 % en Chico Sur y 4,99 % en Chico Norte; mediana 0,40 % y 0,73 %). Hay 99 corridas elegibles en Sur y 151 en Norte por encima de 2 %. En 119 corridas elegibles con inundación (101 de desarrollo y 18 de la reserva), el volumen de error de continuidad supera la mitad del volumen inundado. Son corridas pequeñas: mediana de 66 m³ inundados, factores de 1,2 a 2,6, y 0,05 % del volumen inundado total.

**Por qué ocurre.** El error de continuidad es el desbalance de masa de SWMM. Es relativo al volumen total que pasa por la red, no al volumen inundado. Cuando la inundación es pequeña, el desbalance absoluto puede ser comparable al propio volumen etiquetado. Cerca del umbral de 1 m³ (que coincide con la resolución del RPT, 0,001 × 10⁶ L) la etiqueta "inunda / no inunda" es sensible a ese ruido. Además, las condiciones iniciales pesan: el volumen almacenado inicial es constante (855 m³ en Sur y 4.048 m³ en Norte) porque todos los conductos tienen `InitFlow > 0`, y llega a 59 % de la entrada en eventos pequeños. Eso ya estaba documentado como origen no aclarado de los caudales iniciales de Chico Norte, y aplica también a Chico Sur.

### Plan

- Medir cuánto cambian las métricas si se usa un límite más estricto, sin cambiar la QA ni la población de entrenamiento. Es una sensibilidad post hoc sobre las predicciones ya selladas.
- No endurecer la QA (decisión pendiente 4): cambiaría la población de entrenamiento y obligaría a preparar de nuevo los datos. No se tocan las condiciones iniciales (`InitFlow`).

### Solución aplicada

- `continuity_subset` en [`swmm_resilience/ml/bench/holdout_audit.py`](../../../swmm_resilience/ml/bench/holdout_audit.py): conserva las corridas con |continuidad| menor o igual al límite; rechaza corridas sin dato de continuidad y límites no positivos.
- `scripts/auditar_reserva_conjunta.py` lo aplica con límites de 2 % y 1 % y calcula las métricas completas por red (`continuity_sensitivity` en `audit.json`).
- Tests: los 7 de `test_holdout_audit.py` (dos cubren este subconjunto) y los del script. Comando: el del hallazgo 1.

Cifras (`continuity_sensitivity`):

| Límite | Corridas | Pares | RMSE macro XGBoost (m³) | RMSE macro MLP (m³) |
|---|---:|---:|---:|---:|
| Oficial (≤ 5 %) | 321 | 106.746 | 234,54 | 184,63 |
| ≤ 2 % | 283 | 93.064 | 249,96 | 196,91 |
| ≤ 1 % | 245 | 79.744 | 268,87 | 211,94 |

| Límite | Modelo | Precisión Sur / Norte | Recall Sur / Norte |
|---|---|---:|---:|
| ≤ 2 % | XGBoost | 94,3 % / 93,5 % | 98,0 % / 97,6 % |
| ≤ 2 % | MLP | 85,5 % / 86,0 % | 93,1 % / 93,9 % |
| ≤ 1 % | XGBoost | 94,4 % / 93,9 % | 98,2 % / 97,8 % |
| ≤ 1 % | MLP | 85,7 % / 86,4 % | 93,7 % / 94,5 % |

El orden entre modelos no cambia, y precisión y recall varían menos de 2 puntos porcentuales respecto a los oficiales. El RMSE macro sube al restringir; no se investigó la causa, pero el subconjunto cambia la mezcla de corridas y de formas. Está en la sección "Sensibilidad post hoc" del informe de la reserva.

### Qué queda pendiente y cómo continuar

- Si se cita la clasificación cerca del umbral de 1 m³, acompañarla de esta sensibilidad.
- Decisión pendiente 4 (endurecer la QA al 2 %): obliga a preparar de nuevo los datos y a recalcular el desarrollo.
- Aclarar el origen de los caudales iniciales (`InitFlow`) de ambas redes; hoy se mantienen como están.
- Menores diferidos de `holdout_audit.py` y del script de auditoría: los listados en el hallazgo 1. El más relevante aquí es que `continuity_subset` no avisa si un límite deja a una red sin filas; con 2 % y 1 % no ocurre.

## Hallazgo 6: los artefactos sellados no tienen respaldo

### Problema

`outputs/*` está en `.gitignore`. La base conjunta (247 MB) y la evaluación sellada de la reserva (6,7 MB: sello, predicciones, etiquetas y puntajes) no tenían respaldo documentado. Los estudios, XGBoost y MLP sí lo tienen en Drive (`27_`, `29_` y `32_`).

**Por qué importa.** La reserva ya fue consultada y no puede repetirse a ciegas. Si se perdiera `evaluacion_reserva_conjunta_20261108_v1`, sería irrecuperable en sentido científico, aunque se pudiera recalcular.

### Plan

- Crear un archivo ZIP con los artefactos sellados y un manifiesto con el SHA-256 de cada archivo, verificar que el ZIP coincida con el manifiesto y que las fuentes no hayan cambiado, y no modificar nada de lo respaldado.
- La subida a Drive **no la pueden hacer los agentes**: la hace el investigador. Hasta que la confirme, el hallazgo no se considera resuelto.

### Solución aplicada

- [`scripts/respaldar_artefactos_sellados.py`](../../../scripts/respaldar_artefactos_sellados.py):
  - `build_archive(sources, archive)`: crea el ZIP y su `.manifest.json` con modo de creación exclusiva (no sobrescribe), verifica el hash de cada miembro antes y después de copiarlo, rechaza nombres duplicados y bases SQLite con archivo `-wal` no vacío.
  - `verify_archive(archive, manifest)`: comprueba el hash del ZIP, la lista de miembros y el hash de cada uno.
- Tests: [`tests/test_respaldar_artefactos_sellados.py`](../../../tests/test_respaldar_artefactos_sellados.py) (3 casos: ida y vuelta, detección de corrupción y de discrepancias de miembros, y rechazo de salida existente, WAL no vacío y nombres duplicados).

Comando ejecutado:

```bash
mkdir -p outputs/backups && venv/Scripts/python.exe scripts/respaldar_artefactos_sellados.py \
  --source outputs/experiments/evaluacion_reserva_conjunta_20261108_v1 \
  --source outputs/experiments/joint_chico_sur_norte_20261104_v1 \
  --source outputs/experiments/particion_chico_sur_norte_20261104.json \
  --source outputs/experiments/reserva_formas_chico_sur_norte_20261104.json \
  --source outputs/experiments/auditoria_reserva_conjunta_20261108_v1 \
  --archive outputs/backups/33_reserva_conjunta_y_base_20261108.zip
```

Resultado, verificado en local:

- Archivo: `outputs/backups/33_reserva_conjunta_y_base_20261108.zip` y `outputs/backups/33_reserva_conjunta_y_base_20261108.manifest.json`.
- 21 archivos, 100.110.651 bytes (unos 95 MiB comprimidos; 265.496.386 bytes sin comprimir).
- SHA-256 del ZIP: `2f1ea8857d45e5fcefa6eb2faad435b35ffb2f869f8168ea163fe78b049b4022`.
- Incluye la evaluación sellada, la base conjunta, la partición, la reserva de formas y el `audit.json` de este plan. El hash de la partición del manifiesto coincide con la constante oficial (`b81fd31efa6a...`).

**Drive: pendiente.** El investigador debe subir el ZIP y el manifiesto a la carpeta privada de Drive, junto a `27_`, `29_` y `32_`, y confirmarlo.

### Qué queda pendiente y cómo continuar

1. Subir `33_reserva_conjunta_y_base_20261108.zip` y su `.manifest.json` a Drive. Después de subirlo, comparar el SHA-256 del archivo descargado con el de arriba (`sha256sum` en Git Bash o `certutil -hashfile <archivo> SHA256` en Windows).
2. Cuando el investigador confirme la subida, actualizar la línea del hallazgo 6 en la sección "Estado" de la [auditoría](2026-11-08-auditoria-hidraulica-ml.md) y la fila de este documento, y recién entonces considerarlo resuelto.
3. Observación sobre el contenido: el ZIP incluye archivos `-shm` y `-wal` de las tres bases SQLite (los `-wal` están vacíos, los `-shm` pesan 32 KiB). La base principal se archivó sin un checkpoint explícito; como el WAL estaba vacío, no había transacciones pendientes. Para un respaldo estrictamente consistente, confirmar que no hay escritores activos antes de archivar de nuevo.
4. Menores diferidos de `respaldar_artefactos_sellados.py`:
   - Si una escritura falla a mitad del archivado, queda un ZIP parcial en disco.
   - El hash previo y la escritura del ZIP son lecturas separadas (`verify_archive` detecta la discrepancia), y faltan tests para el cambio de una fuente, la lista de miembros y un manifiesto ya existente.
   - La protección contra WAL ignora los archivos `-journal`, y no se rechaza un `--archive` ubicado dentro de una carpeta fuente.

## Hallazgo 7: detalles menores de código

### Problema

La auditoría listó detalles de código que no cambian resultados. Este documento explica los dos que requirieron una decisión.

**Punto de SQLite (retirado).** La auditoría señaló que `scripts/prepare_joint_dataset.py:35` y `scripts/train_joint_final_20261108.py:88` usan `with sqlite3.connect(...) as db`, que en Python no cierra la conexión (solo hace commit o rollback), y que en Windows eso impediría borrar o mover el archivo mientras el proceso siga vivo. **Al revisarlo, la conexión vive dentro de una función (`prepare_joint_dataset.py`) o de un bucle dentro de `main` (`train_joint_final_20261108.py`), y CPython la cierra al liberar el objeto cuando termina la función o la iteración.** No queda abierta de forma persistente y no requiere cambios.

**Longitud de ruta de los tests en Windows.** La suite completa dio 1.706 aprobados y 1 fallido: `test_joint_builder_verifies_two_sources_and_publishes_new_dataset[False-joint]`, con `FileNotFoundError [WinError 3]`. El test copia archivos llamados `source-<sha256 de 64 caracteres>.bin`; con la carpeta temporal por defecto (unos 110 caracteres) la ruta supera el límite de 260 caracteres de Windows, porque `LongPathsEnabled=0`. Con un `--basetemp` corto pasan sus 6 casos. Es un problema del entorno, no del código.

Los demás puntos (el cotejo que integra el OUT desde la primera muestra, los dos parsers de RPT, `verify_joint_materialization` con rutas absolutas) ya estaban documentados como declarados.

### Plan

- Retirar el punto de SQLite sin cambiar los scripts: son scripts ya ejecutados y documentan lo que se corrió.
- Documentar la nota de rutas cortas en el README, en vez de cambiar tests.

### Solución aplicada

- [README](../../../README.md#verificacion), sección "Verificacion": el comando ahora lleva `--basetemp C:/Users/<usuario>/AppData/Local/Temp/pt` y una nota que explica el límite de 260 caracteres, la alternativa de habilitar rutas largas (requiere administrador) y que `pytest-xdist` no está instalado, por lo que no hay que usar `-n`.
- La suite completa de este plan se corrió con `--basetemp C:/Users/lf.fajardo/AppData/Local/Temp/pt-full`; el resultado fue `1733 passed, 1 skipped, 3 deselected, 649 warnings in 477.15s (0:07:57)`: 0 fallos, los 1.706 aprobados de antes más los 27 tests nuevos (7 + 4 + 7 + 6 + 3). El test que fallaba por la ruta larga pasa con ese `--basetemp`.

### Qué queda pendiente y cómo continuar

- Nada urgente. Si se quiere eliminar el requisito de ruta corta, habilitar `LongPathsEnabled` en Windows o acortar los nombres `source-<sha256>.bin` en los tests.
- Los demás detalles menores de la auditoría (cotejo con `time_sec` corregido, dos parsers de RPT, rutas absolutas en `verify_joint_materialization`) no cambian resultados y quedan como están.

## Presentación

La diapositiva "Hallazgos y próximos pasos" de `scripts/build_weekly_2026_11_08_joint_deck.py` (línea 182) contiene el texto `'MLP logró menor RMSE de volumen en esta reserva.'`, que todavía afirma lo que corrige el hallazgo 1. **No se editó el script ni se regeneró ningún `.pptx`**, porque el script ya fue ejecutado y regenerar el archivo binario (unos 15 MB) requiere la aprobación explícita del investigador.

Si el investigador aprueba: cambiar ese texto a `'MLP logró menor RMSE macro, pero por una sola forma; sin ella gana XGBoost.'`, regenerar la presentación y registrar el nuevo SHA-256 en [`2026-11-08-presentacion-y-diagnostico.md`](2026-11-08-presentacion-y-diagnostico.md).

## Decisiones tomadas durante la ejecución

Las líneas `Ruling:` del registro de avance, copiadas tal cual (están en inglés). Debajo de cada una, una glosa en español.

```
Ruling: execute in place on branch fix/test-suite-contracts-2026-10-27, no worktree — sealed artifacts under outputs/ are git-ignored and exist only in this checkout; a worktree could not run T4/T5 — cost if wrong: task commits land directly on the branch (reversible with git).
```

Se trabajó directamente en la rama, sin un worktree aparte, porque los artefactos sellados solo existen en este clon y un worktree no podría ejecutar las Tareas 4 y 5. Si fue un error, los commits quedaron en la rama, y se pueden revertir con git.

```
Ruling: T6 step 5 (deck text) is NOT executed; recorded as pending user decision — plan requires explicit approval and SDD runs without check-ins — cost if wrong: deck keeps the old one-shape claim until the user approves the edit.
```

No se cambió el texto de la presentación; queda pendiente de la decisión del investigador. Si fue un error, la presentación conserva la afirmación vieja hasta que se apruebe el cambio.

```
Ruling: T5 step 6 (Drive upload) cannot be done by agents; finding 6 stays "respaldo local creado, subida pendiente de confirmación" — cost if wrong: none beyond a pending manual step.
```

La subida a Drive no la pueden hacer los agentes; el hallazgo 6 queda como "respaldo local creado, subida pendiente de confirmación". No hay costo más allá de un paso manual pendiente.

```
Ruling: T6 adds docs/superpowers/reports/2026-11-08-resolucion-hallazgos-auditoria.md (problem/plan/solution per finding) — explicit user request on 2026-11-08 — cost if wrong: one extra doc file.
```

La Tarea 6 agrega este documento de resolución (problema, plan y solución por hallazgo), por pedido expreso del investigador el 8 de noviembre. Si fue un error, el costo es un archivo de documentación de más.

```
Ruling: implementers run only focused tests; the full suite runs once in Task 6 step 6 — a full run costs ~8 min per task and Tasks 1-5 only add new files — cost if wrong: a cross-module regression surfaces at Task 6 instead of at its task.
```

Cada tarea corrió solo sus tests específicos; la suite completa se corrió una vez, en el paso 6 de la Tarea 6, porque cuesta unos 8 minutos y las Tareas 1 a 5 solo agregan archivos. Si fue un error, una regresión entre módulos aparecería en la Tarea 6 y no en su tarea.

## Decisiones pendientes (no son tareas de este plan)

Cambian modelos o protocolo y requieren aprobación explícita del investigador antes de planificarse:

1. **Adoptar el recorte por volumen tributario en la inferencia** (`cap_at_tributary_volume`). Cambiaría las predicciones oficiales de los modelos futuros. No debe aplicarse con efecto retroactivo a la reserva ya consultada.
2. **Reentrenar con restricciones de monotonía** (`monotone_constraints` en XGBoost sobre las variables de volumen) o con otra representación del volumen para el MLP. Requiere una prueba independiente nueva, porque la reserva actual ya se consultó.
3. **Diseñar una reserva nueva** con `stratified_shape_draw` y `inflow_coverage` comprobados antes del sorteo, y registrar el procedimiento antes de ejecutarlo, como se hizo el 4 de noviembre.
4. **Endurecer la QA de continuidad** (por ejemplo, al 2 %). Cambia la población de entrenamiento y obliga a preparar de nuevo los datos.

Además, siguen abiertos el cambio de texto de la presentación (véase "Presentación"), la confirmación de la subida a Drive (hallazgo 6) y, solo si se retoma, la corrección de la cola de drenaje (hallazgo 3).
