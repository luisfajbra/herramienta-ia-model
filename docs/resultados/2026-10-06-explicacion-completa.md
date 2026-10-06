# Explicación completa: de Fase C2 al hallazgo de features tributarias

Este documento explica, de principio a fin y sin asumir que lo recordás de
memoria, todo lo que se hizo entre el cierre de Fase C2 y el hallazgo de que
las features tributarias mejoran el modelo. La idea es que puedas releerlo
en cualquier momento y entender (o explicarle a alguien más) qué se hizo,
por qué, y qué significa cada número.

Hay un informe técnico más corto en
`docs/resultados/2026-10-05-benchmark-fase-c2.md` — este documento es la
versión larga, pensada para entender el razonamiento completo, no solo los
resultados.

---

## 1. De dónde venimos

Antes de esta sesión, el proyecto tenía construidas (en sesiones previas)
varias piezas de infraestructura — una búsqueda automática de
hiperparámetros con Optuna, una política de cierre de hidrogramas (Fase
B1), un contrato de features tributarias nuevo (Fase B2), y una
clasificación del estado de drenaje de cada simulación (Fase C1) — pero
**ninguna de esas piezas se había corrido todavía contra la red SWMM real**.
Todo se había probado solo con datos sintéticos de prueba (fixtures), que
sirven para verificar que el código no tiene bugs, pero no dicen nada sobre
si las ideas funcionan con datos de verdad.

Esta sesión cerró dos piezas que faltaban del diseño original (Fase C2), y
después corrió **todo el pipeline por primera vez de punta a punta contra
la red real**.

## 2. Fase C2 — qué construimos y por qué

El diseño original del proyecto (el "spec" — el documento que define qué
debe hacer el sistema) exigía dos cosas que todavía no existían:

### 2.1. Exclusión por mala continuidad de SWMM

Cuando SWMM simula una red de alcantarillado, internamente verifica que el
agua que entra sea (aproximadamente) igual al agua que sale más la que se
queda almacenada — eso se llama **continuidad**. Si SWMM mismo reporta un
error de continuidad alto (por ejemplo, un 12% de descuadre), significa que
**ni SWMM confía en su propio resultado** para esa simulación puntual. Usar
esa simulación como "la verdad" para entrenar o evaluar un modelo de
Machine Learning sería entrenar contra un dato que ya sabemos que es poco
confiable.

Antes de esta sesión, el código **medía** ese error de continuidad pero
solo mostraba una advertencia (un *warning*) — nunca excluía nada
realmente. Fase C2 hizo que esa medición pasara a tener consecuencia real:
si una simulación supera 5% de error de continuidad, **se saca del
entrenamiento y del ranking de modelos**, y queda registrada en un reporte
separado (`continuity_excluded.csv`) para que no se pierda, solo que no
cuenta como dato de entrenamiento.

**Por qué importa esto en la práctica:** cuando corrimos el pipeline real,
encontramos que **37 de 1000 simulaciones (3.7%)** superaban ese umbral —
mayormente en formas de hidrograma muy cortas y con factores bajos de
intensidad (ej. `shape_ultra_short_impulse` a factor 0.2, con 12.38% de
error). Sin este filtro, esas 37 simulaciones ruidosas habrían contaminado
el entrenamiento sin que nadie lo notara.

### 2.2. Reintento con extensión de drenaje

Una simulación SWMM necesita correr lo suficiente para que toda el agua
termine de drenar de la red (si la cortás muy pronto, el volumen de
inundación queda subestimado porque todavía hay agua "en camino"). El
proyecto ya tenía (de una fase anterior, C1) una forma de **clasificar** si
una simulación terminó bien drenada (`drained`) o no (`not_drained`), pero
no hacía nada si el resultado era `not_drained` — simplemente lo aceptaba
tal cual.

Fase C2 agregó un **bucle de reintento**: si una simulación no drena bien,
se vuelve a correr con más tiempo de drenaje (un incremento configurable),
hasta un número máximo de intentos (también configurable) — nunca
indefinidamente, eso lo exigía el spec explícitamente. Cada intento queda
registrado (cuánto tiempo de drenaje se usó, qué resultado dio), así que si
algo sale mal, se puede auditar exactamente qué pasó.

**En la corrida real:** ninguna de las 1000 simulaciones necesitó
reintento — todas drenaron bien al primer intento. Esto no significa que
el código esté de más (sigue siendo la protección correcta si algún día
cambia la red o los escenarios), simplemente que esta red particular, con
esta configuración, no tuvo problemas de drenaje.

### 2.3. Un hallazgo Critical que casi se nos escapa

Durante el desarrollo de Fase C2, el proceso de revisión de código (donde
un revisor independiente examina todo el trabajo antes de cerrarlo)
encontró algo importante: **el filtro de continuidad que acabábamos de
construir nunca se conectaba a nada real.** El código que excluía
simulaciones malas existía y funcionaba perfectamente en aislamiento, pero
ningún camino de producción (ni el banco de 5 familias, ni el
entrenamiento del modelo legado) lo llamaba — era código correcto que nadie
usaba. Se corrigió antes de cerrar la fase, conectándolo a los dos lugares
donde realmente se entrena un modelo.

Esto es un ejemplo de por qué vale la pena revisar el trabajo integrado, no
solo pieza por pieza: cada pieza individual estaba bien, pero el conjunto
tenía un agujero que solo se ve cuando preguntás "¿y quién llama a esto en
la práctica?".

## 3. La primera corrida real

Con Fase C2 cerrada, corrimos el pipeline completo por primera vez:

1. **Generar el dataset real**: simular 39 formas de hidrograma × 25
   factores de intensidad (0.2 a 5.0) + la forma base = **1000 simulaciones
   SWMM**, cada una con 160 nodos = **160,000 filas** de datos.
2. **Persistir en SQLite** (`--persist-sql`): guardar esas 160,000 filas en
   una base de datos, aplicando ya el filtro de continuidad, y entrenar el
   modelo XGBoost "legado" (el que se usa en producción hoy).
3. **Banco de 5 familias** (`--bench`): entrenar y comparar 5 algoritmos de
   Machine Learning distintos sobre los mismos datos, de forma justa (mismos
   folds de validación, mismo umbral de clasificación, misma semilla
   aleatoria).
4. **Optuna**: buscar automáticamente mejores hiperparámetros para la
   familia ganadora.

Todo esto corrió sin errores de código — la única sorpresa fue de tiempo
(ver sección 4).

## 4. El problema de SVM (y por qué lo dejamos pendiente)

El banco de 5 familias incluye SVM (Support Vector Machine) con kernel
RBF. SVM con ese kernel tiene una propiedad matemática importante: su
costo computacional crece como el cuadrado o el cubo de la cantidad de
filas de entrenamiento (en la jerga, "O(n²) a O(n³)"). Con ~150,000 filas,
eso lo vuelve extremadamente lento — la primera vez que lo dejamos correr,
acumuló **más de un día y medio de tiempo de CPU activo sin terminar**, y
lo cancelamos.

**Esto no es un bug.** Es una limitación conocida y documentada de cómo
funciona SVM con kernel RBF a esta escala de datos. La solución no es
"arreglarlo" sino **configurarlo distinto** para este volumen: por ejemplo,
usar un kernel lineal en vez de RBF, entrenar sobre una muestra en vez de
todos los datos, o usar una variante aproximada de SVM pensada para
datasets grandes. Ninguna de esas opciones se implementó todavía — quedó
en la lista de pendientes.

Por eso el benchmark que sí corrimos tiene **4 familias, no 5** (XGBoost,
Random Forest, Lineal, MLP) — SVM quedó deliberadamente afuera de esta
ronda, no se perdió por accidente.

## 5. El benchmark de 4 familias — resultados y qué significan

| Familia | RMSE volumen | F1 clasificador | NSE regresor |
|---|---|---|---|
| **XGBoost** (ganador) | **78.04** | 0.979 | 0.815 |
| MLP | 99.61 | 0.977 | 0.631 |
| Random Forest | 185.46 | 0.890 | 0.277 |
| Lineal | 201.18 | 0.817 | 0.071 |

**Qué es cada columna:**

- **RMSE volumen** (`rmse_vol_todos_nodos`): el error típico, en metros
  cúbicos, entre el volumen de inundación que predijo el modelo y el que
  realmente ocurrió en SWMM, promediado sobre todos los nodos de la red.
  **Menor es mejor.** Es la métrica principal — la que define quién gana.
- **F1 clasificador**: qué tan bien el modelo adivina *si* un nodo se va a
  inundar o no (sin importar cuánto). Va de 0 a 1, **mayor es mejor**. Los
  cuatro modelos son razonablemente buenos en esto (todos por encima de
  0.8) — la diferencia real entre ellos está en el volumen, no en el
  "¿se inunda sí o no?".
- **NSE regresor** (Nash-Sutcliffe Efficiency): qué tan bien el modelo
  predice la *cantidad* de inundación, solo en los nodos que de verdad se
  inundaron. Va de -∞ a 1, **1 es perfecto**, 0 significa "tan bueno como
  predecir siempre el promedio". XGBoost (0.815) es claramente el mejor
  acá; Random Forest (0.277) y Lineal (0.071) apenas superan ese piso
  trivial.

**Cómo se evaluó:** con el protocolo `LeaveOneShapeOut` — para cada una de
las 39 formas de hidrograma, se entrena el modelo con las otras 38 formas y
se predice sobre la que quedó afuera, repitiendo 39 veces. Esto simula
honestamente "¿qué tan bien predeciría el modelo una tormenta que nunca vio
antes?" — es más exigente (y más realista) que simplemente medir qué tan
bien el modelo recuerda los datos con los que se entrenó.

**Conclusión:** XGBoost gana con margen claro (22% mejor RMSE que el
segundo lugar). No fue un empate reñido.

### 5.1. El caso MLP — investigado y cerrado

Durante la evaluación de MLP (una red neuronal densa simple) aparecieron 70
advertencias de `overflow en exp`. Esto generaba la sospecha de que algo
estaba roto. Se investigó con una verificación numérica directa (no solo
leyendo el código): el resultado es que **era cosmético, no un error
real**. La fórmula para calcular la probabilidad de inundación
(`1/(1+e^-x)`, la función sigmoide) puede desbordar numéricamente cuando la
red está muy segura de su predicción, pero el resultado final sigue
siendo matemáticamente correcto (satura a 0.0 exactamente, nunca da un
valor sin sentido). El F1 alto de MLP (0.977) ya lo confirmaba
indirectamente — si estuviera corrompiendo resultados, el F1 no sería tan
bueno.

**Importante:** el NSE más débil de MLP (0.631, comparado con 0.815 de
XGBoost) no tiene nada que ver con ese overflow — viene de una parte
completamente distinta del modelo (el regresor, que ni siquiera usa esa
fórmula). Es simplemente un hecho real: una red densa simple ajusta peor la
parte de "cuánto se inunda" que un modelo de árboles como XGBoost, en este
problema particular. No hay nada roto ahí, es una limitación real de esa
arquitectura para este tipo de dato.

Se corrigió la fórmula de todos modos (es un cambio de una línea, sin
riesgo, solo para que el log quede limpio) — no cambió ningún resultado.

## 6. Optuna sobre XGBoost — qué encontró y por qué no es definitivo

Con XGBoost confirmado como ganador, se corrió una búsqueda automática de
hiperparámetros (Optuna) — 50 combinaciones distintas de parámetros,
evaluando cada una con su propio esquema de validación interno
(`GroupKFold(5)` por forma: en vez de dejar 1 forma afuera a la vez como en
el ranking, deja **8 formas afuera a la vez**, repetido 5 veces).

**Mejor resultado encontrado** (trial #47 de 50):

| Hiperparámetro | Valor por defecto | Mejor encontrado |
|---|---|---|
| Clasificador: profundidad / tasa de aprendizaje / # árboles | 6 / 0.05 / 200 | 5 / 0.136 / 313 |
| Regresor: profundidad / tasa de aprendizaje / # árboles | 6 / 0.05 / 200 | 8 / 0.199 / 421 |

**Por qué el número de este estudio (108.84) no se compara directamente con
el 78.04 del ranking:** son dos validaciones distintas, con distinto nivel
de dificultad. El ranking deja afuera 1 forma a la vez (el modelo entrena
con 39 de 40 formas — mucha información disponible). Optuna deja afuera ~8
formas a la vez (el modelo entrena con solo ~32 de 40 — bastante menos
información, y tiene que generalizar a un grupo más grande y diverso de
formas nuevas a la vez). Es normal y esperado que el número de Optuna sea
más alto (peor) — no significa que el modelo empeoró, significa que se lo
evaluó con una vara más exigente a propósito, para que la búsqueda de
hiperparámetros no se sobreajuste a la validación más "amable" del
ranking.

El propio estudio se etiqueta a sí mismo como
`estimacion_exploratoria_condicionada` — el diseño original (el spec) exige
esa etiqueta porque no hubo una forma completamente separada reservada
como "examen final" nunca vista ni por el ranking ni por Optuna. Es una
búsqueda exploratoria legítima, pero no es la palabra final sobre qué
hiperparámetros usar en producción.

## 7. Mapas de paridad por familia — qué son y cómo se hicieron

El pedido del asesor de tesis era generalizar los gráficos de "paridad"
(SWMM real vs. predicción del modelo) a las 5 familias, no solo XGBoost.
Al investigar el código se encontró que era más simple de lo esperado: los
textos "XGBoost" en los ejes y leyendas eran solo palabras fijas en el
código (strings hardcodeados), no una limitación real — las funciones de
gráfico ya recibían los datos de forma genérica.

Lo que sí había que decidir era **qué predicciones usar**. Había dos
caminos:

- **Ajuste sobre el 100% de los datos**: usar el modelo ya entrenado con
  todo el dataset y predecir sobre esos mismos datos. Es rápido, pero es un
  resultado "optimista" — el modelo ya vio esos datos durante el
  entrenamiento, así que naturalmente le va bien.
- **Out-of-fold (OOF)**: usar las mismas predicciones que generaron la
  tabla de ranking de la sección 5 — donde cada forma fue predicha por un
  modelo que *nunca la vio* durante el entrenamiento. Es más lento de
  generar (hay que re-evaluar cada familia), pero es la comparación
  honesta, consistente con los números ya reportados.

Se eligió **OOF**, por consistencia. El resultado son gráficos de paridad
(dispersión SWMM vs. predicho) para XGBoost, Random Forest, Lineal y MLP,
guardados en `resultados_archivados/2026-10-05-benchmark-fase-c2/paridad/`.

## 8. El hallazgo de features tributarias (`tabular_v4_21`)

Este es el hallazgo más importante de esta sesión.

### 8.1. Qué son las features tributarias

En una sesión anterior (Fase B2) se diseñaron 6 columnas nuevas de datos
("features") que describen algo que las 17 features actuales no capturan
bien: **cuánta agua llega a un nodo desde los nodos que drenan hacia él
("aguas arriba"), no solo del nodo mismo.** Las 6 son:

1. `volumen_local_integrado_m3` — volumen que entra directamente al nodo.
2. `volumen_tributario_integrado_m3` — volumen que llega desde aguas arriba.
3. `duracion_efectiva_tributario_h` — cuánto dura el aporte tributario.
4. `tiempo_primer_pico_tributario_h` — cuándo llega el primer pico tributario.
5. `tiempo_sobre_capacidad_h` — cuánto tiempo el caudal tributario supera la
   capacidad de la tubería aguas abajo.
6. `integral_exceso_capacidad_m3` — cuánto volumen excede esa capacidad,
   acumulado en el tiempo.

Estas features ya estaban programadas y probadas con datos sintéticos
desde Fase B2, pero **nunca se habían usado para entrenar un modelo real**
— quedaron deliberadamente sin conectar hasta decidir si de verdad valía
la pena adoptarlas (adoptarlas formalmente significa cambiar el "contrato"
de datos del proyecto, de la versión actual `tabular_v3_17` —17 features—
a una nueva `tabular_v4_21` —17+6 = 23 features—, lo cual tiene costos de
migración).

### 8.2. Cómo se validó si valía la pena

En vez de hacer la migración completa a ciegas (que implica cambiar cómo
se genera, persiste y entrena todo el pipeline), se hizo un experimento
rápido y aislado:

1. Se tomaron las 150,240 filas del dataset real ya generado (excluyendo
   la forma base, por simplicidad técnica, y excluyendo las 37 corridas de
   mala continuidad, igual que el benchmark real).
2. Se recalcularon las 6 features tributarias para cada una de esas filas,
   usando la misma función que ya existía desde Fase B2.
3. Se entrenó XGBoost **dos veces**, con exactamente las mismas filas y el
   mismo esquema de validación (`LeaveOneGroupOut` por forma): una vez con
   las 17 features de siempre, otra vez con las 17+6=23.
4. Se comparó el RMSE de las dos corridas.

### 8.3. El resultado

| Conjunto de features | RMSE end-to-end |
|---|---|
| 17 features actuales (`tabular_v3_17`) | 81.95 |
| 17+6 features tributarias (`tabular_v4_21`) | **59.07** |

**Mejora: 27.9%.** No es una mejora marginal ni ruido estadístico — es una
diferencia grande y consistente. Esto responde la pregunta que quedaba
abierta desde Fase B2: **sí, vale la pena migrar el contrato de datos a
`tabular_v4_21`.**

*(Nota: el 81.95 de esta validación no es el mismo número que el 78.04 del
ranking oficial de la sección 5 — son conjuntos de datos levemente
distintos, porque esta validación excluye la forma base y usa
`LeaveOneGroupOut` en vez de `LeaveOneShapeOut`. Lo que importa de este
experimento no es el valor absoluto, sino la comparación relativa entre
las dos corridas, que usaron exactamente las mismas filas y el mismo
esquema.)*

### 8.4. Qué falta para adoptarlo de verdad

Este experimento fue deliberadamente pequeño y aislado — **no modifica
nada del pipeline real.** Para adoptar `tabular_v4_21` en producción
faltaría:

1. Fusionar las 6 columnas tributarias al dataset persistido de verdad
   (`dataset_final.csv` y la base SQLite), no solo calcularlas al vuelo
   para un experimento.
2. Re-generar el dataset completo con las columnas nuevas incluidas.
3. Volver a correr el banco de 5 familias (o al menos XGBoost) sobre el
   contrato nuevo, con el protocolo oficial (`LeaveOneShapeOut`, no
   `LeaveOneGroupOut`), para confirmar la mejora con la metodología
   definitiva.
4. Reconectar `ScenarioPredictor` (el módulo que genera predicciones en
   producción, fuera del banco de pruebas) para que también use las
   features nuevas — hoy las calcula pero las descarta, por la misma
   decisión de Fase B2 de no mergearlas hasta decidir.

Esto quedó como el primer punto de la lista de pendientes, ahora con
evidencia real detrás en vez de ser una apuesta.

## 9. Qué se entregó al final de esta sesión

- **Informe técnico corto**: `docs/resultados/2026-10-05-benchmark-fase-c2.md`
- **Este documento** (la explicación larga): `docs/resultados/2026-10-06-explicacion-completa.md`
- **Presentación actualizada**: `Presentacion_MLSMFP_05-10_resultados_fase_c2.pptx`
  (37 slides: las 30 originales intactas + 7 nuevas con todo lo de este
  documento, incluyendo el hallazgo de features tributarias)
- **Mapas y datos crudos archivados**: `resultados_archivados/2026-10-05-benchmark-fase-c2/`
  (mapas de paridad por familia, ranking, estudio completo de Optuna,
  reporte de corridas excluidas por continuidad)
- **Scripts reutilizables** (`scripts/`):
  - `generar_paridad_por_familia.py` — regenera los gráficos de paridad OOF
    para cualquier combinación de familias.
  - `armar_presentacion_resultados.py` — reconstruye la presentación desde
    cero con los datos actuales (nunca sobreescribe el PPT original de
    15-09).
  - `validar_features_tributarias.py` — repite el experimento de la
    sección 8 (útil si se quiere re-validar después de algún cambio).

## 10. Glosario rápido

- **RMSE** (Root Mean Squared Error): error típico entre predicción y
  realidad, en las mismas unidades que el dato (aquí, m³). Menor es mejor.
- **F1**: balance entre precisión y sensibilidad de una clasificación
  binaria (¿se inunda o no?). Entre 0 y 1, mayor es mejor.
- **NSE** (Nash-Sutcliffe Efficiency): qué tan bien un modelo predice una
  cantidad continua, comparado contra predecir siempre el promedio. 1 es
  perfecto, 0 es "tan bueno como el promedio", negativo es peor que eso.
- **OOF** (Out-Of-Fold): predicciones hechas por un modelo sobre datos que
  *no* usó para entrenarse — la forma honesta de medir qué tan bien
  generalizaría el modelo a datos nuevos.
- **LeaveOneShapeOut / LeaveOneGroupOut**: esquema de validación donde se
  deja afuera un grupo (una forma de hidrograma) a la vez, se entrena con
  el resto, y se predice sobre el que quedó afuera. Se repite para cada
  grupo.
- **GroupKFold(5)**: parecido, pero en vez de dejar 1 grupo afuera a la
  vez, divide todos los grupos en 5 partes y deja una parte entera (varios
  grupos juntos) afuera en cada repetición.
- **Continuidad (SWMM)**: verificación interna de SWMM de que el balance
  de agua (entra = sale + almacenado) cuadra razonablemente. Un error alto
  significa que la propia simulación no es confiable.
- **OOF vs. in-sample**: in-sample es "predecir sobre datos que el modelo
  ya vio" (optimista); OOF es "predecir sobre datos que nunca vio"
  (honesto).
