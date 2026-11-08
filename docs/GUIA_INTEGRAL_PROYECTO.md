# Guía integral del proyecto de predicción de inundaciones en redes pluviales

**Fecha de referencia: 8 de noviembre de 2026.** Esta guía explica el objetivo de la tesis, los datos, el funcionamiento hidráulico y de aprendizaje automático, las piezas de software, los experimentos y cómo orientarse en el repositorio. Describe el estado comprobado en esta copia local de la rama `fix/test-suite-contracts-2026-10-27`, commit base `65d1e170`, con cambios y archivos adicionales todavía sin confirmar en Git. Las rutas de `outputs/` son locales y no se pueden dar por presentes en un clon nuevo.

## 1. Qué problema busca resolver

El proyecto investiga si un modelo de aprendizaje automático puede aproximar la inundación que calcula SWMM para los nodos de una red de alcantarillado pluvial. El propósito es poder producir predicciones más rápidas después de entrenar, manteniendo una comparación verificable con el simulador hidráulico.

Para un escenario, la entrada física de esta etapa consiste en la descripción de la red y en **hidrogramas de caudal por nodo**. El sistema no recibe lluvia directamente ni calcula, como parte del objetivo actual, la transformación lluvia–escorrentía. La salida predictiva es, para cada nodo, una probabilidad o decisión de inundación y un volumen de inundación en m³.

SWMM proporciona la referencia simulada. El resultado es una aproximación a SWMM, no una validación directa contra inundaciones observadas. Los modelos no sustituyen automáticamente una revisión hidráulica ni están demostrados para cualquier red.

## 2. Idea general del funcionamiento

1. Leer un archivo SWMM `.inp`: nodos, conducciones, geometría, series de entrada y opciones del modelo.
2. Definir un escenario: forma temporal de cada hidrograma y factor de escala. Para los lotes sintéticos de Chico Norte, cada forma normalizada se multiplica por el pico de referencia del nodo y por el factor seleccionado.
3. Generar un INP efectivo por escenario y ejecutar SWMM. Guardar los insumos, el reporte `.rpt`, la salida `.out`, la configuración, los hashes y el estado de cada corrida.
4. Aplicar la política provisional de calidad hidráulica; las corridas que no cumplen se excluyen de las etapas que la exigen y se registran con su motivo.
5. Calcular variables de geometría, conectividad e hidrogramas para cada par nodo–escenario. Extraer del RPT el volumen de inundación de referencia y derivar la etiqueta binaria.
6. Preparar los datos preservando la identidad de la red, nodo, escenario y corrida; separar formas de desarrollo y formas reservadas.
7. Ajustar los modelos de clasificación y regresión con validación agrupada, seleccionar hiperparámetros y guardar candidatos y su procedencia.
8. Comparar predicciones y SWMM mediante errores, clasificación, paridades, mapas y curvas por factor. Registrar qué datos participaron en cada fase.

Esquema conceptual: `red + hidrogramas → SWMM → QA → variables y etiquetas → partición → entrenamiento → predicciones → evaluación y figuras`.

## 3. Conceptos básicos

- **Red:** nodos y conducciones representados en un INP; en esta investigación las redes principales son Chico Sur y Chico Norte.
- **Nodo o junction:** punto de la red para el que se construye una fila de variables y se predice la inundación.
- **Hidrograma:** serie temporal de caudal que entra en un nodo. Una forma normalizada define la evolución relativa en el tiempo; por sí sola no es un registro de lluvia.
- **Factor:** multiplicador de intensidad usado al crear un escenario. `factor_mult` se guarda como metadato y afecta las variables calculadas, pero no es una de las 21 entradas directas de los modelos conjuntos.
- **Escenario / corrida:** combinación de red, entradas y configuración que se simula en SWMM. Una corrida produce resultados para muchos nodos; sus filas no son redes independientes.
- **QA:** criterios de control para decidir si una corrida se considera elegible bajo una política de software. Cumplir el QA provisional no demuestra por sí solo que la red esté validada físicamente.
- **Desarrollo:** corridas usadas para seleccionar y entrenar. **Reserva:** formas separadas del desarrollo para evaluar los candidatos; si sus resultados ya se examinaron, no se puede volver a tratar la misma reserva como prueba ciega.

## 4. Estructura del repositorio

| Ruta | Función principal |
|---|---|
| `main.py` | Interfaz de línea de comandos y flujo histórico principal; también expone modos de banco y selección de eventos. |
| `config.yaml` | Red predeterminada, factores, QA, modelos y parámetros de visualización y banco. |
| `swmm_resilience/simulation/` | Generación de escenarios, escritura de INP, simulación y gestión de lotes. |
| `swmm_resilience/extraction/` | Variables estáticas, topología, características de hidrogramas y lectura de etiquetas de inundación. |
| `swmm_resilience/database/` | SQLite, migraciones, calidad, enlaces a artefactos, procedencia y materialización conjunta. |
| `swmm_resilience/dataset/` | Ensamblaje, validación y selección de datos. |
| `swmm_resilience/ml/` | Contratos de variables, entrenamiento, inferencia, evaluación y banco de modelos. |
| `swmm_resilience/evaluation/` | Flujos de eventos seleccionados y evaluación congelada. |
| `swmm_resilience/visualization/` | Gráficos, mapas y lectura geométrica de redes. |
| `swmm_resilience/analysis/` | Análisis de curvas y comparaciones por factor. |
| `swmm_resilience/desktop/` | Código de una interfaz Tkinter; no es el flujo conjunto documentado aquí. Su lanzamiento no se verificó en esta revisión. |
| `swmm_resilience/ml/temporal/` | Componentes experimentales temporales; no son los modelos conjuntos tabulares usados en los resultados recientes. |
| `data/networks/` | Redes y sus archivos INP. |
| `data/hydrograph_shapes/` | Formas normalizadas usadas en campañas de escenarios. |
| `data/hydrograph_validation/` | Formas y protocolos de validación; algunas series están ignoradas por Git y deben respaldarse. |
| `scripts/` | Ejecuciones especializadas, entrenamiento conjunto, evaluación, figuras y verificaciones. |
| `docs/superpowers/reports/` | Informes, resultados y decisiones con fecha y alcance. |
| `docs/superpowers/plans/` y `specs/` | Planes y diseños; algunos son históricos y no implican que todos los ítems estén terminados. |
| `tests/` | Pruebas de contratos, integración y componentes. |
| `outputs/` | Bases, estudios, modelos, simulaciones y resultados locales; se ignora en gran parte por Git. |

El repositorio tiene más de un flujo y conserva código histórico. La presencia de un módulo o comando no demuestra que se haya integrado con los candidatos conjuntos o verificado para redes arbitrarias.

## 5. Entrada hidráulica y simulaciones

### 5.1 Archivo INP y unidades

El INP de SWMM describe geometría y conectividad, cotas, secciones, rugosidad, condiciones iniciales, entrada de caudal y opciones de simulación. La configuración actual de `config.yaml` apunta por defecto a Chico Sur y a factores de 0,2 a 5,0 en pasos de 0,2. Las pruebas de Chico Norte usan su INP explícito y sus propios protocolos. Los contratos de las variables de la evaluación reciente usan principalmente m, m³, L/s y horas; no se debe presuponer compatibilidad de unidades al incorporar otra red.

El proyecto utiliza `swmm-api` para manipular archivos SWMM y PySWMM / SWMM Toolkit para ejecutar el motor. Los escenarios recientes se congelan con su INP efectivo, serie de evento, RPT, OUT, hashes SHA-256 y protocolo. Los identificadores y hashes sirven para detectar cambios y sostener la trazabilidad; no validan por sí mismos el modelo hidráulico.

### 5.2 Tiempo y cierre de los eventos

La convención aprobada para la regeneración conjunta redondea los tiempos de las formas al minuto. Antes de las simulaciones nuevas se fijan las formas, los factores y sus hashes. El drenaje configurado empieza con seis horas y permite hasta dos extensiones de dos horas. Las tolerancias de aporte, inundación, flujo residual y cambio de almacenamiento aparecen en `config.yaml` y deben interpretarse junto a la política del experimento concreto.

**Limitación conocida:** si una serie termina con caudal distinto de cero, el flujo histórico que agrega un cero lejano puede generar una rampa de entrada durante el supuesto drenaje. La auditoría identificó 175 corridas excluidas afectadas. El problema permanece documentado y no debe relajarse el QA para introducirlas sin decidir y volver a simular su tratamiento.

### 5.3 Calidad de cada corrida

Para las evaluaciones recientes se mantiene provisionalmente: continuidad absoluta ≤5 %, cobertura finita, estado drenado y procedencia verificable de condición inicial y reporte de inundación. La política y las razones de exclusión están en `swmm_resilience/database/hydraulic_quality.py` y en los manifiestos de las corridas. No se añadió una exclusión automática por porcentaje de pasos sin convergencia o por FII. El límite del 5 % y las tolerancias aún requieren justificación hidráulica; no son una certificación de calidad de todas las etiquetas.

Las redes conservan las condiciones iniciales registradas para los experimentos. El origen de los caudales iniciales `InitFlow` no está aclarado. El programa guarda la identidad del estado empleado, pero eso no resuelve su interpretación física.

## 6. Variables que reciben los modelos conjuntos

Los candidatos conjuntos MLP y XGBoost usan el mismo contrato ordenado `tabular_v4_21` con 21 variables por nodo–escenario:

1. `elev_fondo`: cota de fondo del nodo.
2. `prof_max`: profundidad máxima del nodo.
3. `n_tuberias_in`: número de conducciones entrantes.
4. `n_tuberias_out`: número de conducciones salientes.
5. `diam_max_in`: máximo tamaño vertical de sección entrante; es diámetro en secciones circulares.
6. `diam_max_out`: máximo tamaño vertical de sección saliente.
7. `pendiente_max_in`: máxima pendiente geométrica entrante calculable.
8. `pendiente_out`: pendiente de la primera salida calculable.
9. `base_inflow_lps`: pico del perfil de referencia del nodo en L/s.
10. `dist_outfall_m`: distancia dirigida mínima hasta una descarga alcanzable.
11. `n_nodos_aguas_arriba`: número de nodos antecesores en el grafo.
12. `q_pico_acum_base`: suma de picos de referencia del nodo y sus antecesores.
13. `upstream_capacity_lps`: suma de capacidades estimadas a tubo circular lleno de los conductos inmediatamente entrantes.
14. `q_pico_nodo`: pico del aporte del escenario en el nodo.
15. `q_pico_acum_escalado`: suma de picos nodales tributarios para el escenario.
16. `volumen_local_integrado_m3`: integral temporal de la entrada del nodo.
17. `volumen_tributario_integrado_m3`: integral del hidrograma tributario construido.
18. `duracion_efectiva_tributario_h`: tiempo por encima del 1 % del pico tributario construido.
19. `tiempo_primer_pico_tributario_h`: instante del primer máximo tributario construido.
20. `tiempo_sobre_capacidad_h`: tiempo en que el aporte tributario construido supera la capacidad de referencia.
21. `integral_exceso_capacidad_m3`: integral del exceso sobre esa capacidad de referencia.

El hidrograma tributario se construye sumando las entradas del nodo y de sus antecesores sobre una malla común. Es una aproximación de aportes; no simula desfases, almacenamiento o remanso en los conductos. Una suma de picos individuales tampoco es necesariamente el pico simultáneo de la suma. La capacidad estimada no es la capacidad hidráulica efectiva de toda la red. La lista detallada, sus unidades, justificaciones y restricciones está en [Inventario y justificación de variables](superpowers/reports/2026-11-08-inventario-variables-modelos-conjuntos.md).

## 7. Qué predicen los modelos

Cada familia usa una cascada de dos componentes:

1. El **clasificador** estima la probabilidad de inundación. La referencia `inunda` vale 1 cuando SWMM reporta un volumen ≥1 m³. En la evaluación conjunta la decisión de inundación usa probabilidad ≥0,5.
2. El **regresor** estima el volumen condicional de inundación. Trabaja con la transformación `log1p` del volumen y la salida se devuelve a m³ mediante `expm1`, con recorte inferior en cero. Si el clasificador indica no inundación, la cascada reporta volumen cero.

Ambos modelos imputan valores faltantes con la mediana aprendida durante el ajuste. MLP además estandariza; XGBoost no utiliza escalador. Los metadatos de los candidatos finales indican que no se aplica PCA.

**XGBoost** usa árboles de decisión en ensamble. El candidato final tiene clasificadores y regresores ajustados por separado. **MLP** es una red neuronal densa implementada con PyTorch; el candidato final combina cinco semillas deterministas. El ensamble promedia probabilidades y volúmenes condicionales convertidos a m³ y después aplica el umbral de clasificación. No se eligió una sola semilla mirando la reserva.

La justificación hidráulica de las variables no demuestra que todas sean importantes o causales. Los modelos pueden aprovechar relaciones estadísticas particulares de las dos redes y requieren evaluación adicional para medir transferencia.

## 8. Preparación y entrenamiento conjunto realizado

La base conjunta se materializó a partir de fuentes verificadas de Chico Sur y Chico Norte, conservando registros y exclusiones. El conjunto de 1.975 corridas comprende 1.332 de desarrollo, 321 reservadas y 322 excluidas bajo el QA utilizado. Chico Sur aportó 866 corridas elegibles y Chico Norte 787; los diez escenarios de una evaluación anterior se mantuvieron separados de ese lote principal.

Se seleccionaron siete formas reservadas mediante un sorteo reproducible previo; todos sus factores se mantuvieron fuera del desarrollo en ambas redes. Para el ajuste interno se usaron cinco particiones agrupadas por forma: una forma y sus factores permanecen juntos, evitando dividir la misma forma entre ajuste y validación de una partición. La búsqueda persistente con Optuna se fijó en 50 pruebas por familia y hasta 200 épocas por MLP. El criterio principal de búsqueda fue `macro_network_rmse`: se calcula un RMSE por red y se promedian los valores de las redes presentes.

Los estudios se completaron y a partir de ellos se entrenaron los candidatos finales en `outputs/experiments/final_joint_candidates_20261108_v1/`. Los valores seleccionados de validación interna registrados fueron 58,89 m³ para XGBoost y 74,73 m³ para MLP; **no son las métricas de la reserva**. El archivo de estado de entrenamiento dice `both_candidates_fitted_not_holdout_evaluated` porque registra esa fase; la evaluación posterior tiene su propio estado `holdout_scored`. No se debe interpretar el primer estado como ausencia de evaluación posterior.

Los scripts especializados son `scripts/materialize_joint_dataset.py`, `scripts/prepare_joint_dataset.py`, `scripts/tune_joint_checkpointed.py`, `scripts/train_joint_final_20261108.py` y `scripts/evaluate_joint_holdout_20261108.py`. Los estudios contienen contratos y hashes del código y de los insumos: agregar o modificar archivos Python puede impedir una reanudación con el código actual. Para reproducir o reanudar se necesitan la versión congelada, los estudios y todos los insumos exactos; no se deben sustituir hashes ni archivos históricos.

## 9. Evaluaciones realizadas y cómo interpretarlas

### 9.1 Transferencia anterior Chico Sur → Chico Norte

Antes del entrenamiento conjunto se evaluaron modelos ajustados solo con Chico Sur en Chico Norte. También se analizaron dos formas sintéticas anteriores con cinco factores. Esa prueba evalúa otra versión de los modelos y sus resultados no deben presentarse como si pertenecieran al modelo conjunto. Permanece preservada para la historia del proyecto.

### 9.2 Reserva de formas para modelos conjuntos

Los modelos conjuntos se evaluaron frente a SWMM en 321 corridas de siete formas reservadas en Chico Sur y Chico Norte. La métrica principal fue RMSE macro entre redes: 234,54 m³ para XGBoost y 184,63 m³ para MLP. Un análisis posterior encontró que la ventaja global de MLP depende considerablemente de una forma sostenida. Al excluirla descriptivamente el orden cambia; eso no cambia la métrica oficial, no justifica eliminarla ni constituye una prueba ciega nueva. Véase [evaluación de la reserva](superpowers/reports/2026-11-08-evaluacion-reserva-conjunta.md).

### 9.3 Escenarios nuevos con modelos conjuntos

Se diseñaron dos hidrogramas sintéticos nuevos para Chico Norte y cinco factores nuevos 1,9; 2,1; 2,5; 2,9 y 3,7. Las diez simulaciones fueron elegibles bajo el QA provisional. Sobre 5.220 pares nodo–escenario, el RMSE global fue 27,02 m³ para MLP y 51,12 m³ para XGBoost; el recall fue 94,39 % y 98,19 %, respectivamente. Estos RMSE se calculan agrupando todas las filas, no promediando diez RMSE de escenarios. El sesgo global del MLP puede ocultar errores de signo opuesto entre formas.

Los factores están dentro del intervalo del entrenamiento y evalúan interpolación. El tiempo al primer pico tributario de la forma escalonada supera el máximo de desarrollo por nodo en 2.610 pares, de modo que existe extrapolación univariada para esa variable. No se ha demostrado transferencia del modelo conjunto a una tercera red. Véase [resultados y protocolo](superpowers/reports/2026-11-08-resultados-escenarios-nuevos-modelos-conjuntos.md).

## 10. Métricas y gráficas

- **RMSE:** raíz del promedio de errores de volumen al cuadrado. Un error grande pesa más; las unidades son m³ por nodo–escenario.
- **MAE:** promedio del valor absoluto del error de volumen, en m³.
- **RMSE macro por red:** promedio simple de los RMSE calculados por separado en cada red. No es necesariamente igual al RMSE global de todos los pares, porque las redes tienen distinto número de nodos y escenarios.
- **Precisión:** proporción de nodos que SWMM clasifica inundados entre los que el modelo marca inundados.
- **Recall o detección:** proporción de inundaciones de referencia que el modelo identifica. No es un porcentaje universal de confiabilidad.
- **Sesgo de volumen:** diferencia agregada entre el total predicho y el total SWMM; errores positivos y negativos pueden compensarse.
- **Paridad:** cada punto compara volumen SWMM y predicción. Algunas figuras muestran `log10(1 + volumen)` en los ejes; los errores informados se calculan en m³ originales.
- **Mapa de inundación:** muestra la red y los volúmenes de los nodos. En la evaluación más reciente se producen mapas individuales de SWMM, MLP y XGBoost con escala común entre los tres para cada escenario. Las escalas pueden variar entre escenarios diferentes.

## 11. Base de datos y trazabilidad

SQLite conserva redes, escenarios, corridas, variables, etiquetas, QA y enlaces a archivos según el esquema y las migraciones activas. Las migraciones están en `swmm_resilience/database/sql/` y en esta copia llegan de `001` a `011`. Una corrida puede enlazar su INP efectivo, evento, RPT y OUT con sus hashes. El flujo de materialización conjunta verifica fuentes y crea una base nueva; la lectura de entrenamiento usa instantáneas y controles de identidad para reducir riesgo de mezcla o alteración.

Los archivos `.inp`, `.rpt`, `.out`, CSV, bases SQLite, modelos `joblib`, manifiestos JSON, estudios Optuna y gráficos tienen funciones distintas. Un hash verifica identidad frente a una referencia conservada, pero no garantiza por sí mismo el origen físico de los datos ni la validez científica de una decisión.

## 12. Instalación y dependencias

El entorno observado usa Python 3.12.10 en Windows. `pyproject.toml` declara Python ≥3.11 y no instala dependencias de ejecución; `requirements.txt` contiene versiones fijadas. El script histórico `scripts/reproduce_results.ps1` exige específicamente Python 3.13 y ejecuta pruebas y el flujo principal histórico; **no equivale a reproducir la campaña conjunta ni debe ejecutarse sin revisar sus efectos**. La compatibilidad completa de todas las dependencias y comandos en una instalación limpia no se comprobó en esta revisión.

Para preparar un entorno en una copia nueva, desde PowerShell y desde la raíz del proyecto, una receta general es:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --help
```

Instalar paquetes puede requerir acceso a internet y versiones compatibles con el Python usado. La instalación anterior no garantiza disponer de bases, modelos ni salidas ignoradas por Git. El `venv` observado en este equipo es distinto de `.venv` usado en los ejemplos de instalación.

Versiones observadas y funciones:

| Librería | Versión observada | Uso |
|---|---:|---|
| `pyswmm` / `swmm-toolkit` | 2.1.0 / 0.17.0 | Ejecución y acceso al motor SWMM. |
| `swmm-api` | 0.4.73 | Lectura y manipulación de archivos SWMM. |
| `numpy` / `pandas` | 2.3.5 / 2.3.3 | Cálculo numérico y tablas. |
| `scikit-learn` | 1.8.0 | Pipelines, imputación, escalado, métricas y validación. |
| `xgboost` | 3.2.0 | Modelos basados en árboles. |
| `torch` | 2.6.0 | Implementación de los MLP. |
| `optuna` | 5.0.0 | Búsqueda y registro de hiperparámetros. |
| `networkx` | 3.6.1 | Grafos y conectividad de la red. |
| `matplotlib` | 3.10.0 | Mapas, hidrogramas y métricas gráficas. |
| `pyarrow` | 22.0.0 | Archivos Parquet. |
| `python-pptx` | 1.0.2 | Generación de presentaciones. |
| `pytest` | 9.0.3 | Ejecución de pruebas. |
| `PyYAML` | 6.0.3 | Lectura de configuración YAML. |
| `joblib`, `Pillow`, `shap`, `ydata-profiling` | Versiones en `requirements.txt` | Persistencia de modelos, imágenes y análisis opcionales. |

## 13. Comandos y diferencias entre flujos

Los siguientes ejemplos describen comportamientos; no se ejecutaron simulaciones ni entrenamientos como parte de la revisión documental.

```powershell
.\venv\Scripts\python.exe main.py --help
.\venv\Scripts\python.exe main.py --inspect-event-plan data/hydrograph_validation/chico_norte_conjuntos_nuevos_20261108/seleccion_conjuntos_nueva_20261108.json
```

`--help` muestra opciones; `--inspect-event-plan` inspecciona la selección sin simular. El protocolo puede prepararse y congelarse con `--freeze-event-plan` en una carpeta nueva y ejecutarse con `--run-frozen-event-plan`. Esas etapas producen archivos y pueden tardar; deben preservarse los directorios y el ancla.

**Diferencia crucial:** `--score-frozen-event-plan` y `--evaluate-frozen-chico-norte` apuntan al flujo congelado anterior de candidatos entrenados con Chico Sur. No seleccionar ese comando suponiendo que evalúa los candidatos conjuntos. La evaluación reciente de los modelos conjuntos usó `scripts/puntuar_conjuntos_escenarios_nuevos.py`, `scripts/cobertura_conjuntos_escenarios_nuevos.py` y `scripts/figuras_conjuntos_escenarios_nuevos.py`; son scripts de un experimento específico con rutas, selecciones y salvaguardas fijadas, no una interfaz general para cualquier red.

El comando `python main.py` y `--only-ml` siguen el flujo histórico de 17 variables; no se deben usar para reemplazar modelos conjuntos sin entender su política de promoción. `main.py --predict --factor ...` carga el modelo activo de `outputs/models/` y, según el código actual, ejecuta también SWMM para comparación. Los scripts de evaluación recientes no promueven automáticamente un candidato a `outputs/models/`.

Para verificar pruebas de software se usa `python -m pytest ...` con el entorno adecuado. En este equipo algunas rutas temporales de Windows han presentado problemas de permisos o longitud. La última revisión documental no equivale a volver a ejecutar la suite completa.

## 14. Cómo se construyen los resultados y la presentación

El protocolo de escenarios nuevos se registró antes de predecir. El flujo conserva los 10 eventos, ancla y salidas de SWMM; un proceso de staging verifica fuentes y calcula variables sin escribir etiquetas nodales. Luego se guardaron y sellaron las predicciones de los modelos conjuntos; las etiquetas se reconstruyeron de los RPT enlazados y se compararon por `run_id` y `node_id`. Los detalles están en [protocolo](superpowers/reports/2026-11-08-protocolo-escenarios-nuevos-modelos-conjuntos.md) y [resultados](superpowers/reports/2026-11-08-resultados-escenarios-nuevos-modelos-conjuntos.md).

La presentación `Presentacion_MLSMFP_08-11_modelos_conjuntos_escenarios_nuevos.pptx` es una copia del formato semanal con 46 diapositivas. Conserva la sección de modelos conjuntos y añade hidrogramas, mapas separados, paridades, curvas y resultados nuevos. La versión anterior se conserva. Su construcción, hashes y revisión visual parcial están en [documentación de la presentación](superpowers/reports/2026-11-08-presentacion-modelos-conjuntos-escenarios-nuevos.md).

## 15. Respaldos y transferencia a otra persona

Gran parte de `outputs/`, archivos `.joblib`, bases SQLite, `.rpt`, `.out`, CSV y PNG está ignorada por Git. En particular, `git check-ignore` confirmó que las formas CSV y figuras de la prueba nueva y sus resultados bajo `outputs/` no se incluyen automáticamente. Un commit o push de código no garantiza que esos archivos viajen con el proyecto.

La carpeta privada de Google Drive autorizada contiene respaldos de resultados, estudios, candidatos, presentaciones y paquetes con manifiestos. El respaldo 45 de los escenarios nuevos se dividió en cinco partes por el límite de carga del conector. Se verificó localmente que, concatenadas en orden, reproducen el SHA-256 del ZIP; los metadatos remotos confirman tamaños, pero la restauración y verificación criptográfica de las descargas sigue pendiente. Las rutas absolutas almacenadas en algunos manifiestos pueden requerir un procedimiento de restauración o revisión antes de ejecutar en otro equipo; no debe modificarse un manifiesto sellado sin registrar el cambio.

Para entregar el proyecto se necesita identificar expresamente: versión del código, requisitos y Python, redes INP, formas CSV, partición, base conjunta, estudios Optuna, candidatos y sus metadatos, simulaciones y reportes QA, predicciones y figuras. Confirmar las autorizaciones de acceso y las licencias pertinentes; los enlaces privados no son automáticamente accesibles a terceros.

## 16. Riesgos y pendientes conocidos

1. El QA, sus tolerancias y el límite de continuidad son provisionales; algunos pasos SWMM sin convergencia no se excluyen automáticamente. El origen físico de `InitFlow` sigue sin aclarar.
2. La cola de drenaje de formas que no finalizan en cero puede generar entrada adicional; las corridas afectadas revisadas fueron excluidas, no corregidas.
3. La cota de volumen tributario es un diagnóstico exploratorio y el recorte no se adoptó. No se ha demostrado una ley universal aplicable a todos los estados hidráulicos.
4. La evaluación conjunta usa dos redes conocidas; la transferencia a una tercera red no vista todavía no se ha probado. La cobertura por variable individual no demuestra cobertura multivariada.
5. El código de capacidad y algunas variables de sección se apoyan en supuestos geométricos simplificados; una red con bifurcaciones o elementos distintos requiere revisión antes de extrapolar.
6. El README y parte de la documentación histórica mezclan versiones; las notas de fecha, el código y los artefactos concretos definen el alcance de cada resultado.
7. Los respaldos remotos se han comprobado en distintos grados; la comprobación de hashes de archivos descargados y un ensayo de restauración integral no están completos.
8. Los siguientes pasos de análisis de los nuevos resultados y evaluación en una tercera red están **pausados por solicitud del investigador** mientras se revisa la documentación.

## 17. Lecturas recomendadas

1. [Inventario y justificación de las 21 variables](superpowers/reports/2026-11-08-inventario-variables-modelos-conjuntos.md).
2. [Materialización y avances del entrenamiento conjunto](superpowers/reports/2026-11-04-materializacion-conjunta-avance.md).
3. [Evaluación de la reserva conjunta](superpowers/reports/2026-11-08-evaluacion-reserva-conjunta.md) y [revisión de la auditoría](superpowers/reports/2026-11-08-revision-independiente-resolucion-auditoria.md).
4. [Protocolo de escenarios nuevos](superpowers/reports/2026-11-08-protocolo-escenarios-nuevos-modelos-conjuntos.md) y [resultados](superpowers/reports/2026-11-08-resultados-escenarios-nuevos-modelos-conjuntos.md).
5. [Documentación de la presentación reciente](superpowers/reports/2026-11-08-presentacion-modelos-conjuntos-escenarios-nuevos.md).
6. [Informe de revisión documental](superpowers/reports/2026-11-08-verificacion-documentacion.md).
