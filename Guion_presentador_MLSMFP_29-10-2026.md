# Guion para presentar los avances del proyecto

**Auditoría hidráulica, modelos y transferencia A → B**
Presentación asociada: `Presentacion_MLSMFP_29-10_avances_auditoria_transferencia.pptx`
Duración sugerida: 9 minutos, más preguntas.
Corte operativo: 29 de octubre de 2026, aproximadamente 07:09, hora de Bogotá.

## Antes de exponer

Esta copia de la presentación refleja **805 resultados de B registrados, 159 exclusiones QA y 11 escenarios pendientes** (corte aproximado de las 07:09). El último corte documentado para MLP es de las 07:01: **26 de 50 ensayos completados**, con el trial 27 en curso; el mejor valor interno está en el trial 6 (96,667 m³). Ambos procesos avanzan, así que actualiza estas cifras si la exposición ocurre más tarde.

No anticipes un porcentaje de confiabilidad. Solo podremos reportarlo después de congelar las predicciones A→B, verificar los artefactos y abrir las etiquetas de B.

## Guion diapositiva por diapositiva

### 1. Portada · 20 s

> Buenos días. Voy a resumir qué quedó corregido y trazable, qué resultados tenemos para la red A y cómo estamos preparando la prueba de transferencia en la red B, Chico Norte. La idea central es separar los resultados internos de la evaluación externa, que todavía está en curso.

### 2. Resumen · 40 s

> Reejecutamos y calificamos hidráulicamente las simulaciones de A. Con esos datos completamos la búsqueda de XGBoost y tenemos un candidato ajustado. En paralelo avanzan las simulaciones de B y la búsqueda de MLP. Todavía no hay una métrica de desempeño externo: primero deben terminar los modelos y quedar selladas sus predicciones.

### 3. QA de A · 50 s

> La base se reconstruyó desde el INP y los hidrogramas actuales. De mil corridas, 866 pasaron los controles provisionales y entraron al conjunto elegible; 134 se excluyeron. Los motivos incluyen 100 casos que no drenaron y 35 que superaron el 5 por ciento de error de continuidad, con un caso que presenta ambos motivos. El conjunto tiene 138.560 filas, 160 nodos y 21 variables. El umbral de inundación es un metro cúbico. Los criterios hidráulicos siguen documentados como provisionales.

### 4. Diseño A→B · 40 s

> La selección de hiperparámetros usa solo A y separa por forma de hidrograma con GroupKFold de cinco partes. B se reserva como prueba externa: no debe usarse para elegir parámetros, umbrales ni modelos. Esta prueba nos dirá cómo se transfiere el modelo a esta red concreta; por sí sola no demuestra que funcionará en todas las redes.

### 5. Topología de Chico Norte · 35 s

> Aquí está la geometría del INP de B: 523 nodos con coordenadas y 522 conducciones. Es un mapa de la red, no de inundación. Sirve para entender el tamaño y la disposición de la red que estamos incorporando.

### 6. Hidrograma y campaña · 50 s

> El gráfico muestra una entrada sintética de ejemplo en el nodo 268I. La forma dura 8,5 horas; el pico nodal del INP es 54,08 litros por segundo y, con el factor 2,8, el pico aplicado es 151,4 litros por segundo. La campaña contempla 39 formas por 25 factores, 975 escenarios en total, con factores de 0,2 a 5,0. Son caudales sintéticos por nodo; no son lluvia observada ni incluyen la transformación lluvia–escorrentía.

### 7. Ejemplo mediano de A · 45 s

> A la izquierda está el resultado SWMM para el escenario inundado mediano por volumen total; a la derecha, la predicción de un XGBoost promovido anteriormente. En este ejemplo los totales son 9.518 y 9.278 metros cúbicos, con 52 nodos inundados según cada mapa. La comparación es retrospectiva e ilustrativa. No es una predicción OOF ni una prueba de transferencia.

### 8. Mayor volumen de A · 45 s

> Este es el escenario con mayor volumen total de A: SWMM obtiene 183.407 metros cúbicos y el modelo anterior 188.628; los mapas marcan 93 y 91 nodos inundados. Lo mostramos para contrastar una distribución espacial concreta. No debemos convertir uno o dos casos seleccionados en un porcentaje de confiabilidad.

### 9. Resultado interno de XGBoost · 50 s

> La búsqueda de XGBoost completó 50 ensayos Optuna. El mejor valor objetivo fue 73,782 metros cúbicos, en el trial 13, con GroupKFold de cinco partes. Es una estimación interna sobre A y formas separadas por fold. No es el error que esperamos en B ni una garantía de generalización.

### 10. Estado de MLP y B · 55 s

> Al corte de esta presentación, B registra 805 resultados, tiene 159 escenarios excluidos por QA y 11 pendientes. MLP lleva 26 de 50 ensayos; el trial 27 estaba corriendo al último corte documentado. El mejor trial de MLP hasta ahora es el 6, con 96,667 metros cúbicos, pero esa búsqueda aún no termina. Los conteos de B son operativos, no resultados de predicción.

### 11. Qué falta para los mapas de predicción B · 40 s

> El orden es terminar MLP, ajustar el ensemble de cinco semillas, congelar los candidatos y sus hashes, verificar los enlaces y la cobertura de B, y entonces ejecutar el scoring externo. Solo después se abren las etiquetas de B y se producen gráficos de paridad y mapas de predicción. No reportamos confiabilidad antes de ese punto.

### 12. Afirmaciones y límites · 45 s

> Lo que podemos afirmar hoy es que A está reconstruida y filtrada, y que existe un candidato XGBoost con trazabilidad. La transferencia a B sigue sin medirse. Además, los escenarios son sintéticos y los umbrales hidráulicos siguen siendo provisionales. El paso siguiente permitirá evaluar una red no vista; para afirmar generalización amplia tendremos que probar redes adicionales.

### 13. Cierre y trazabilidad · 25 s

> Cada cifra está ligada a una preparación, una configuración y artefactos con hashes. En una frase: cerramos la base de entrenamiento A y estamos terminando B como prueba externa; después evaluamos una sola vez y decidimos si conviene entrenar una nueva versión con A y B.

## Estado y tiempo esperado

La estimación usa el ritmo observado, no una duración garantizada. Los 25 ensayos MLP anteriores tardaron aproximadamente 22–40 minutos cada uno. Al corte quedaban 24 ensayos, incluido el que estaba en curso.

- **Escenarios B pendientes:** 13; aproximadamente 15–60 minutos si mantienen el ritmo actual. Las extensiones de drenaje o los reintentos pueden alargarlo.
- **Búsqueda MLP restante:** 24 ensayos; aproximadamente 9–16 horas según el rango observado por ensayo.
- **Ajuste y verificación del ensemble de cinco semillas:** reservar 0,5–1,5 horas. Esta fase aún no tiene una medición en esta corrida.
- **Staging, verificación de hashes/cobertura y scoring B:** reservar 1–3 horas una vez que ambos modelos estén sellados. Parte del staging puede hacerse mientras termina MLP.

**Estimación total para completar la evaluación externa: 10–22 horas desde el corte de las 07:09; reservar hasta 24 horas** si hay reintentos, bloqueos de archivos o correcciones de manifiestos. El rango apunta a la tarde/noche del 29 de octubre y, en el extremo alto, a la mañana del 30. El tiempo para una eventual versión entrenada con A+B no está incluido.

## Siguientes pasos, en orden

1. Completar los 11 escenarios B pendientes y cerrar QA con el umbral de continuidad del 5 por ciento todavía provisional.
2. Completar los 50 trials MLP con A únicamente; seleccionar parámetros con el protocolo registrado y ajustar el ensemble de cinco semillas.
3. Verificar que las corridas B aptas tengan INP, hidrogramas, RPT/OUT, hashes, variables y etiquetas enlazados por `run_id`. Mantener cerradas las etiquetas durante preparación y predicción.
4. Sellar los candidatos y sus predicciones para B; ejecutar la evaluación externa una sola vez.
5. Reportar métricas generales y por forma/escenario; producir paridad y mapas SWMM versus modelo para B.
6. Solo después del scoring A→B, decidir si se crea una versión entrenada con A+B. Esa nueva versión ya no se evaluará independientemente en B; para medir generalización se necesitará una red C no usada en el entrenamiento.

## Respuestas breves a preguntas probables

**¿El 73,78 m³ significa que el modelo se equivoca eso en Chico Norte?**
No. Es el mejor objetivo interno de XGBoost durante GroupKFold sobre A. El error en B sigue pendiente.

**¿Los 975 escenarios son 975 tormentas observadas?**
No. Son escenarios sintéticos formados por 39 hidrogramas normalizados y 25 factores de caudal. No modelan lluvia ni escorrentía.

**¿Por qué no entrenar ya con B?**
Porque primero necesitamos mantener B como prueba externa. Si sus etiquetas entran al entrenamiento antes del scoring, deja de ser una verificación independiente.

**¿Podemos decir que el sistema será confiable en otras redes?**
Todavía no. B medirá transferencia a una red. Para estimar comportamiento en una población de redes se necesitan más redes no vistas y justificar los criterios hidráulicos provisionales.

**¿Cuándo podremos mostrar paridad y mapas nuevos?**
Después de terminar MLP, sellar predicciones y completar el scoring verificado de B. El plazo de planificación es aproximadamente 10–22 horas desde el corte, sujeto a la ejecución real.

## Nota de método

La comparación espacial de las diapositivas 7–8 usa un XGBoost anterior aplicado retrospectivamente a ejemplos de A. Está rotulada como ilustrativa. No debe usarse para afirmar confiabilidad, exactitud independiente ni transferencia. Los valores internos de XGBoost/MLP tampoco son métricas de B.
