# Informe de los últimos cambios del proyecto

**Propósito:** documento de estudio y apoyo para presentar los avances recientes al asesor.  
**Historial revisado:** commits hasta el 27 de septiembre de 2026.  
**Commit más reciente:** `6e8579bf` — `feat: validacion out-of-sample de hidrogramas y correccion de features de forma en inferencia`.  
**Alcance:** cambios registrados en Git. Las modificaciones locales que no se han confirmado en un commit se identifican por separado al final.

---

## 1. Resumen ejecutivo

El cambio más reciente mejora el surrogate de inundación en dos frentes relacionados:

1. **Coherencia entre entrenamiento e inferencia.** La predicción ahora utiliza la duración y el tiempo al pico del hidrograma base del archivo `.inp`, en vez de asignar cero a ambas variables.
2. **Evaluación con formas temporales no vistas.** Se agregaron 21 perfiles al catálogo de entrenamiento y siete perfiles distintos en un directorio de validación. Se añadió un comando para convertir esos perfiles normalizados en escenarios que se comparan con SWMM.

El propósito científico es evaluar si el surrogate representa distintos patrones temporales de caudal y no solo distintas escalas de una forma conocida. El commit prepara datos y herramientas para esa evaluación, pero **no incluye una tabla de resultados que demuestre la precisión alcanzada en los siete perfiles**. Para presentar conclusiones de desempeño se deben ejecutar las corridas y estudiar las métricas generadas.

En paralelo, durante septiembre se trabajó en un banco de comparación de familias de modelos, en la reproducibilidad del entrenamiento y en el uso de SQLite como fuente de datos para análisis. Estos cambios dan contexto al estado del proyecto, aunque responden a preguntas diferentes de la validación reciente por forma de hidrograma.

## 2. Contexto del problema

### 2.1 ¿Qué es el surrogate?

SWMM calcula la respuesta hidráulica de una red de drenaje para un escenario de caudales. El surrogate ML intenta aproximar parte de esa respuesta con modelos entrenados usando escenarios previamente simulados. En este proyecto se predicen, entre otros resultados, si un nodo se inunda y qué volumen de inundación tendría.

La ventaja práctica que se busca es estimar respuestas sin ejecutar una simulación hidráulica completa en cada consulta. Para justificar esa sustitución, el surrogate debe compararse con SWMM en escenarios que no hayan servido para entrenarlo.

### 2.2 ¿Por qué importa la forma temporal?

Dos hidrogramas pueden tener un pico semejante y, aun así, diferir en duración, tiempo al pico y distribución temporal del caudal. La red puede reaccionar de forma distinta porque la inundación depende de cuánto caudal llega y durante cuánto tiempo, además del pico.

Si el conjunto de entrenamiento contiene solo versiones escaladas de una misma forma, una evaluación que cambie únicamente el factor de escala no prueba generalización a formas temporales nuevas. Separar perfiles completos para validación permite plantear una prueba más directa de esa generalización.

## 3. Cambio principal: perfiles fuera de muestra

### 3.1 Datos añadidos

El commit más reciente contiene:

- **21 perfiles nuevos** en `data/hydrograph_shapes/`, destinados al catálogo de entrenamiento.
- **7 perfiles de validación** en `data/hydrograph_validation/`, separados del directorio de entrenamiento.
- Modelos serializados reentrenados en `outputs/models/classifier.joblib` y `outputs/models/regressor.joblib`.

Los CSV de forma usan dos columnas normalizadas:

- `time_h`: tiempo en horas.
- `q_norm`: caudal normalizado respecto al pico.

El mecanismo de validación transforma cada forma normalizada en series de caudal por nodo, expresadas en L/s, aplicando los inflows base del `.inp` y el factor indicado. Esas series alimentan el flujo existente de validación por lotes.

### 3.2 Protección contra contaminación entre entrenamiento y validación

Antes de evaluar, el CLI compara los nombres de las formas solicitadas con los nombres del catálogo de entrenamiento. Si encuentra una forma en ambos directorios, detiene la ejecución con un error. Esto evita etiquetar como “fuera de muestra” un perfil que también esté presente en el entrenamiento.

La separación por nombre es una salvaguarda práctica. La interpretación científica sigue dependiendo de que los perfiles se mantengan separados y de que los datos de validación no se incorporen posteriormente al entrenamiento antes de reportar esa evaluación.

### 3.3 Comando principal

Para evaluar todas las formas del directorio de validación:

```bash
python main.py --evaluate-validation-shapes data/hydrograph_validation
```

Sin `--factors`, el código elige los puntos medios entre factores de entrenamiento. En la configuración versionada, el rango va de 0.2 a 5.0 con paso 0.2; por lo tanto, los puntos medios son 0.3, 0.5, …, 4.9 (**24 factores**). Con siete formas serían hasta **168 escenarios** (siete formas por 24 factores).

Se puede reducir la corrida indicando factores concretos:

```bash
python main.py --evaluate-validation-shapes data/hydrograph_validation `
    --factors 1.15,2.5,3.75 `
    --out-dir ./validation_output
```

Para evaluar una sola forma, el directorio entregado debe contener solo ese CSV. El CLI lee todos los `*.csv` del directorio indicado.

### 3.4 Interpolación y extrapolación de duración

El comando compara la duración de cada perfil con el rango de duraciones observado en el catálogo de entrenamiento y la etiqueta como **interpolación** o **extrapolación**. Esto ayuda a interpretar si la forma evaluada cae dentro del rango temporal de entrenamiento.

La etiqueta se refiere a la duración; no equivale por sí sola a una prueba completa de extrapolación hidráulica ni garantiza que todas las demás propiedades de la forma estén cubiertas por el entrenamiento.

## 4. Corrección de las variables de forma en inferencia

### 4.1 El problema previo

El pipeline de entrenamiento registraba las variables `duracion_horas` y `tiempo_al_pico_h` asociadas a la forma. En cambio, `predict_network` asignaba `0.0` a ambas en inferencia.

Eso hacía que el modelo recibiera durante la predicción valores distintos de los que describían el hidrograma usado en entrenamiento. El commit señala que el regresor daba alta importancia a esas variables y que su omisión podía subestimar el volumen total inundado hasta en un 94 % en algunos casos. Esa cifra es la motivación reportada en el mensaje del commit; no debe presentarse como un resultado estadístico general sin revisar el experimento y sus datos originales.

### 4.2 La corrección

Se añadió `base_shape_stats()` en `swmm_resilience/ml/predict.py`. La función carga el `.inp`, busca la serie temporal disponible y obtiene duración y tiempo al pico mediante `get_shape_stats()`. Después, `predict_network()` pasa ambos valores a `compute_dynamic_features()`, de forma que la inferencia use la misma semántica básica de variables que el entrenamiento.

El flujo simplificado queda así:

```text
.inp de la red
   ├── características estáticas y topológicas
   └── serie temporal base → duración y tiempo al pico
                                  ↓
                        variables de predicción
                                  ↓
                  clasificador + regresor ML
                                  ↓
            estado de inundación y volumen por nodo
```

La predicción también conserva la verificación del hash MD5 del `.inp` frente al archivo guardado con el entrenamiento. Además, advierte cuando el factor solicitado está fuera del intervalo de entrenamiento configurado.

## 5. Robustez de la ejecución y presentación de métricas

El último commit incluye dos cambios de robustez en `main.py`:

- `_fmt_metric()` imprime `N/A` cuando una métrica es `None`, `NaN` o no se puede formatear. Esto cubre escenarios pequeños en los que métricas como F1, NSE o error porcentual pueden no estar definidas.
- `_make_stdout_unicode_safe()` intenta configurar UTF-8 en la salida estándar y de error cuando la consola lo permite. Esto evita que algunos terminales antiguos de Windows detengan la ejecución al imprimir caracteres Unicode.

También se añadió `validation_output/` a `.gitignore`, porque los resultados de validación son salidas regenerables. Los CSV fuente de validación sí se incluyeron en el repositorio.

## 6. Cambios inmediatamente anteriores

### 6.1 Reproducibilidad entre entrenamiento en memoria y lectura SQL

El commit `9d6ec4a` del 6 de septiembre ordenó el DataFrame del flujo de entrenamiento en frío por `shape_id`, `factor_mult` y `node_id` antes de entrenar y evaluar.

La causa fue que XGBoost usa submuestreo de filas (`subsample < 1`). La selección aleatoria de filas por árbol puede depender del orden en que se presentan los datos. Por eso, las mismas filas en distinto orden podían producir modelos o métricas diferentes entre el flujo que ensamblaba datos en memoria y el flujo que los leía desde SQL.

El arreglo ordena la copia en memoria que se entrega al entrenamiento. Mantiene intacto el orden del CSV persistido. El commit también agregó pruebas que comparan los resultados después del orden canónico y que muestran por qué importa el orden en el fixture empleado.

### 6.2 Banco de comparación de modelos

Varios commits del 6 de septiembre completaron una primera capa del banco (`bench`) para preparar datos, entrenar, evaluar, clasificar candidatos y exportar reportes. La configuración admite cinco familias de modelos:

| Familia | Uso general |
|---|---|
| XGBoost | Ensambles de árboles con boosting |
| Random Forest | Ensamble de árboles independientes |
| Lineal | Baselines lineales o regularizados |
| SVM | Máquinas de vectores de soporte |
| MLP | Perceptrón multicapa tabular |

Entre las mejoras del banco se registraron:

- Metadatos y procedencia reproducible para el dataset preparado.
- Validaciones de calidad, incluyendo error explícito si una variable completamente nula pudiera desaparecer silenciosamente durante la imputación.
- Folds de evaluación y puntuación por protocolo.
- Ranking con identificación del protocolo utilizado y exportación de JSON estricto.
- Validación de que el dataset preparado cubra los protocolos solicitados.
- Filtrado común de hiperparámetros para las cinco familias.
- Comandos CLI para preparar, entrenar, evaluar y promover candidatos.
- Salvaguardas para familias desconocidas y para la promoción de modelos.

La existencia de estas capacidades no demuestra que se haya ejecutado un benchmark final conjunto ni que una familia haya ganado en condiciones comparables. Para afirmarlo se deben consultar sus datos de ejecución y reportes correspondientes.

### 6.3 Lectura del conjunto de entrenamiento desde SQLite

El 3 de septiembre se documentó y completó el cambio de varios consumidores para leer el conjunto de entrenamiento desde SQLite. La intención fue unificar la fuente de lectura de flujos como entrenamiento, mapas y análisis, reduciendo dependencias del CSV legado. El commit `9d6ec4a` posterior ajustó la paridad de orden entre esa ruta y el flujo en memoria.

## 7. Cómo interpretar la evidencia disponible

Conviene separar tres tipos de afirmaciones:

1. **Cambio de implementación:** el código ahora calcula las dos variables de forma desde el `.inp` y ofrece un comando de evaluación de formas separadas.
2. **Cobertura de pruebas:** el commit agrega pruebas para la construcción de features, el CLI de validación, el formateo de métricas y perfiles de hidrograma.
3. **Resultado científico:** requiere ejecutar el experimento y revisar las salidas generadas. El commit revisado no aporta una tabla de métricas por forma que permita afirmar cuánto error obtuvo el surrogate en esa validación.

Esta distinción es importante: una prueba automatizada puede verificar que el programa aplica un contrato, pero no sustituye una corrida experimental ni una comparación cuantitativa con SWMM.

### Protocolo previo y pregunta de generalización

El documento de diseño de cierre de auditoría registra que el banco existente agrupaba sus protocolos por `factor_mult`. Esos protocolos pueden medir generalización a una intensidad apartada, manteniendo formas presentes en distintos conjuntos. No son equivalentes a dejar fuera una forma temporal completa.

La nueva colección de siete formas intenta responder esta pregunta más directamente. En la exposición se recomienda nombrar explícitamente qué perfiles se reservaron, qué factores se evaluaron y qué métricas se reportaron.

## 8. Alcance y limitaciones que se deben mencionar

- **No hay resultados nuevos incluidos en este informe.** El informe describe cambios de código e historial; no ejecuta SWMM ni calcula métricas.
- **La evaluación de siete formas es evidencia sobre perfiles reservados**, no evidencia automática de generalización a otras redes. Los modelos siguen ligados a la red y al `.inp` usados para entrenarlos.
- **El rango de factor también importa.** La configuración de entrenamiento cubre factores 0.2–5.0. El código advierte fuera de ese rango, pero una advertencia no valida la predicción extrapolada.
- **Duración dentro del rango no significa forma idéntica.** Una forma puede tener duración interpolada y aun presentar una secuencia de pulsos o una distribución temporal nueva.
- **El mensaje de commit menciona una subestimación de hasta 94 %.** Preséntala como antecedente reportado que motivó la corrección, no como desempeño universal ni como porcentaje confirmado por todas las formas.
- **Los modelos serializados cambiaron.** Si se usan para resultados, hay que registrar qué archivos de modelo, `.inp`, configuración y datos se usaron en cada evaluación.

## 9. Commits principales del periodo revisado

| Fecha | Commit | Aporte |
|---|---|---|
| 3 sep 2026 | `b622d06`, `6e6273b`, `c166afd`, `7778a46`, `7a5ce36`, `1e9d1c4` | Lectura desde SQLite para consumidores del conjunto de entrenamiento y documentación del corte. |
| 5–6 sep 2026 | `881a9db`, `cc317ef`, `f0eea7c`, `eb85c2e`, `7daa361`, `ef48872`, `71be972` y otros | Construcción incremental del banco ML: esquemas, folds, preparación, métricas, puntuación, ranking y reportes. |
| 6 sep 2026 | `be84d27` | Configuración y comandos CLI del banco, incluida promoción explícita de candidatos. |
| 6 sep 2026 | `1bd879c`, `3715551`, `a8d1e40`, `656d669`, `6f46921`, `3822be3` y otros | Correcciones de casos límite, coherencia entre familias, carga y procedencia del benchmark. |
| 6 sep 2026 | `9d6ec4a` | Orden canónico para hacer reproducibles las métricas entre rutas de entrenamiento. |
| 27 sep 2026 | `6e8579b` | Corrección de features temporales de inferencia, perfiles reservados de validación, comando de evaluación y modelos reentrenados. |

El commit más reciente modifica **50 archivos**, con aproximadamente **1.770 inserciones y 153 eliminaciones**. Entre los cambios también figura la incorporación de presentaciones `.pptx`, documentación y pruebas, y la eliminación de un documento de revisión anterior.

## 10. Guion sugerido para la exposición

> “El avance más reciente aborda un problema de coherencia entre entrenamiento e inferencia. El modelo aprendía usando duración y tiempo al pico del hidrograma, pero la inferencia asignaba cero a ambas variables. Ahora esas características se calculan a partir del hidrograma base del archivo de red. También se añadieron 21 perfiles al entrenamiento y se reservaron siete perfiles distintos para evaluación fuera de muestra. El nuevo comando expande esos perfiles a escenarios por nodo y permite compararlos contra SWMM en factores especificados. El paso pendiente para cerrar la conclusión científica es ejecutar la validación y presentar las métricas por perfil, distinguiendo interpolación y extrapolación.”

## 11. Preguntas que podría hacer el asesor

### ¿Qué se corrigió exactamente?

La inferencia ya no asigna duración y tiempo al pico iguales a cero; obtiene ambos descriptores de la serie temporal del `.inp` y los pasa al cálculo de variables del modelo.

### ¿Qué significa “fuera de muestra” aquí?

Que los siete perfiles colocados en `data/hydrograph_validation/` están separados del catálogo `data/hydrograph_shapes/` usado como entrenamiento y que el CLI comprueba que sus nombres no estén duplicados allí.

### ¿La validación demuestra que el surrogate generaliza?

No por sí sola. La incorporación del mecanismo hace posible medirlo. Para afirmarlo se deben ejecutar las comparaciones y reportar los resultados por forma y factor, además de aclarar que la red evaluada es la misma red para la que se entrenó el surrogate.

### ¿Qué diferencia hay entre variar el factor y variar la forma?

Variar el factor escala el caudal del perfil; variar la forma cambia cómo se distribuye el caudal a lo largo del tiempo. Son pruebas distintas de comportamiento del modelo.

### ¿Qué significa interpolación o extrapolación en el CLI?

La etiqueta compara la duración del perfil con el rango de duraciones del catálogo de entrenamiento. Describe cobertura de duración; no resume todas las diferencias temporales o hidráulicas entre perfiles.

### ¿Por qué importa ordenar el dataset antes de entrenar?

Porque con submuestreo de filas el orden de presentación puede afectar qué filas se usan en los árboles. El orden canónico busca que las rutas en memoria y desde SQL entrenen con la misma secuencia de muestras.

## 12. Glosario breve

| Término | Significado en este proyecto |
|---|---|
| Hidrograma | Serie de caudal a lo largo del tiempo. |
| `time_h` | Tiempo de cada punto del perfil, en horas. |
| `q_norm` | Caudal normalizado respecto al pico del perfil. |
| Factor (`factor_mult`) | Multiplicador aplicado a los caudales base. |
| Feature | Variable de entrada que recibe el modelo ML. |
| Inferencia | Uso del modelo ya entrenado para generar predicciones. |
| Out-of-sample / fuera de muestra | Evaluación en datos reservados que no se usaron al entrenar. |
| Surrogate | Modelo aproximado que busca reproducir parte de la respuesta de una simulación más costosa. |
| SWMM | Modelo/simulador hidráulico usado como referencia para comparar escenarios. |
| Fold | Partición de datos para entrenar y evaluar bajo un protocolo de validación cruzada. |
| Promoción | Operación explícita que copia un candidato seleccionado a los nombres de modelo activos. |

## 13. Archivos recomendados para profundizar

1. `AUDITORIA_SURROGATE_HIDROGRAMAS.md` — antecedentes y hallazgos de auditoría.
2. `docs/superpowers/specs/2026-09-27-cierre-auditoria-surrogate-hidrogramas-design.md` — diseño metodológico y pendientes de la auditoría.
3. `COMANDOS.md` — instrucciones del CLI, incluida la validación con perfiles externos.
4. `main.py` — argumentos y flujo de `--evaluate-validation-shapes`.
5. `swmm_resilience/ml/predict.py` — cálculo de descriptores del `.inp` y predicción de la red.
6. `swmm_resilience/simulation/hydrograph_shapes.py` — lectura, estadísticas y expansión de perfiles.
7. `tests/test_predict_shape_features.py` y `tests/test_validation_shapes_cli.py` — contratos automatizados para las funciones nuevas.

## 14. Estado local del repositorio al revisar

Además del historial, el árbol de trabajo tenía modificaciones y archivos nuevos sin commit: entre ellos cambios en `config.yaml`, `main.py`, módulos del banco ML, pruebas y documentos con fechas del 27 y 29 de septiembre.

Esos cambios no se consideran parte del último commit ni se usan como evidencia de que el trabajo esté terminado. Antes de mostrar el estado actual del repositorio como si fuera un único avance cerrado, conviene revisar y clasificar esos cambios locales.

---

**Nota de elaboración:** informe basado en `git log`, el diff del commit `6e8579b` y archivos versionados relacionados. No se ejecutaron pruebas, comandos de evaluación ni simulaciones SWMM para elaborar este documento.
