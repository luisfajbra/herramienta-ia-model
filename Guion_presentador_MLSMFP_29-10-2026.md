# Guion para presentar los avances del proyecto

**Auditoría hidráulica, modelos y transferencia de Chico Sur a Chico Norte**
Presentación asociada: `Presentacion_MLSMFP_29-10_avances_auditoria_transferencia.pptx`
Duración sugerida: 9 minutos, más preguntas.
Corte operativo: 29 de octubre de 2026, aproximadamente 09:00, hora de Bogotá.

## Antes de exponer

Esta copia de la presentación refleja **871 simulaciones completadas para Chico Norte: 709 elegibles, 162 excluidas por QA y 104 pendientes** (corte aproximado de las 09:00). Para MLP, el último contador persistido es de las 07:01: **26 de 50 ensayos completados**, con el trial 27 en curso; el proceso sigue activo, pero aún no pude confirmar un contador más reciente. El mejor valor registrado está en el trial 6 (96,667 m³). Actualiza el estado de MLP antes de exponer.

No anticipes un porcentaje de confiabilidad. Solo podremos reportarlo después de congelar las predicciones de transferencia de Chico Sur a Chico Norte, verificar los artefactos y abrir las etiquetas de Chico Norte.

## Guion diapositiva por diapositiva

### 1. Portada · 20 s

> Buenos días. Voy a resumir qué quedó corregido y trazable, qué resultados tenemos para Chico Sur y cómo estamos preparando la prueba de transferencia en Chico Norte. La idea central es separar los resultados internos de la evaluación externa, que todavía está en curso.

### 2. Resumen · 40 s

> Reejecutamos y calificamos hidráulicamente las simulaciones de Chico Sur. Con esos datos completamos la búsqueda de XGBoost y tenemos un candidato ajustado. En paralelo avanzan las simulaciones de Chico Norte y la búsqueda de MLP. Todavía no hay una métrica de desempeño externo: primero deben terminar los modelos y quedar selladas sus predicciones.

### 3. QA de Chico Sur · 50 s

> La base se reconstruyó desde el INP y los hidrogramas actuales. De mil corridas, 866 pasaron los controles provisionales y entraron al conjunto elegible; 134 se excluyeron. Los motivos incluyen 100 casos que no drenaron y 35 que superaron el 5 por ciento de error de continuidad, con un caso que presenta ambos motivos. El conjunto tiene 138.560 filas, 160 nodos y 21 variables. El umbral de inundación es un metro cúbico. Los criterios hidráulicos siguen documentados como provisionales.

### 4. Diseño de transferencia Chico Sur → Chico Norte · 40 s

> La selección de hiperparámetros usa solo Chico Sur y separa por forma de hidrograma con GroupKFold de cinco partes. Chico Norte se reserva como prueba externa: no debe usarse para elegir parámetros, umbrales ni modelos. Esta prueba nos dirá cómo se transfiere el modelo a esta red concreta; por sí sola no demuestra que funcionará en todas las redes.

### 5. Topología de Chico Norte · 35 s

> Aquí está la geometría del INP de Chico Norte: 523 nodos con coordenadas y 522 conducciones. Es un mapa de la red, no de inundación. Sirve para entender el tamaño y la disposición de la red que estamos incorporando.

### 6. Hidrograma y campaña · 50 s

> El gráfico muestra una entrada sintética de ejemplo en el nodo 268I. La forma dura 8,5 horas; el pico nodal del INP es 54,08 litros por segundo y, con el factor 2,8, el pico aplicado es 151,4 litros por segundo. La campaña contempla 39 formas por 25 factores, 975 escenarios en total, con factores de 0,2 a 5,0. Son caudales sintéticos por nodo; no son lluvia observada ni incluyen la transformación lluvia–escorrentía.

### 7. Ejemplo mediano de Chico Sur · 45 s

> A la izquierda está el resultado SWMM para el escenario inundado mediano por volumen total; a la derecha, la predicción de un XGBoost promovido anteriormente. En este ejemplo los totales son 9.518 y 9.278 metros cúbicos, con 52 nodos inundados según cada mapa. La comparación es retrospectiva e ilustrativa. No es una predicción OOF ni una prueba de transferencia.

### 8. Mayor volumen de Chico Sur · 45 s

> Este es el escenario con mayor volumen total de Chico Sur: SWMM obtiene 183.407 metros cúbicos y el modelo anterior 188.628; los mapas marcan 93 y 91 nodos inundados. Lo mostramos para contrastar una distribución espacial concreta. No debemos convertir uno o dos casos seleccionados en un porcentaje de confiabilidad.

### 9. Resultado interno de XGBoost · 50 s

> La búsqueda de XGBoost completó 50 ensayos Optuna. El mejor valor objetivo fue 73,782 metros cúbicos, en el trial 13, con GroupKFold de cinco partes. Es una estimación interna sobre Chico Sur y formas separadas por fold. No es el error que esperamos en Chico Norte ni una garantía de generalización.

### 10. Estado de MLP y Chico Norte · 55 s

> Al corte de esta presentación, Chico Norte registra 805 resultados, tiene 159 escenarios excluidos por QA y 11 pendientes. MLP lleva 26 de 50 ensayos; el trial 27 estaba corriendo al último corte documentado. El mejor trial de MLP hasta ahora es el 6, con 96,667 metros cúbicos, pero esa búsqueda aún no termina. Los conteos de simulaciones de Chico Norte son operativos, no resultados de predicción.

### 11. Qué falta para los mapas de predicción de Chico Norte · 40 s

> El orden es terminar MLP, ajustar el ensemble de cinco semillas, congelar los candidatos y sus hashes, verificar los enlaces y la cobertura de Chico Norte, y entonces ejecutar la evaluación externa. Solo después se abren las etiquetas de Chico Norte y se producen gráficos de paridad y mapas de predicción. No reportamos confiabilidad antes de ese punto.

### 12. Afirmaciones y límites · 45 s

> Lo que podemos afirmar hoy es que Chico Sur está reconstruida y filtrada, y que existe un candidato XGBoost con trazabilidad. La transferencia a Chico Norte sigue sin medirse. Además, los escenarios son sintéticos y los umbrales hidráulicos siguen siendo provisionales. El paso siguiente permitirá evaluar una red no vista; para afirmar generalización amplia tendremos que probar redes adicionales.

### 13. Cierre y trazabilidad · 25 s

> Cada cifra está ligada a una preparación, una configuración y artefactos con hashes. En una frase: cerramos la base de entrenamiento de Chico Sur y estamos terminando Chico Norte como prueba externa; después evaluamos una sola vez y decidimos si conviene entrenar una nueva versión con ambas redes.

## Estado y estimación de tiempo

La estimación anterior de 10–22 horas se calculó a las 07:09 y ya no sirve como plazo vigente. A las 09:00 faltaban 104 escenarios de Chico Norte; el proceso seguía escribiendo resultados. El proceso MLP también seguía activo, pero el último contador guardado es de las 07:01, por lo que no puedo confirmar cuántos ensayos de los 50 faltan ahora.

- **Escenarios de Chico Norte pendientes:** 104. Las extensiones de drenaje o los reintentos pueden alterar el ritmo.
- **Búsqueda MLP:** el último contador persistido mostraba 26/50, con el trial 27 en curso. Los ensayos previos tardaron aproximadamente 22–40 minutos cada uno, pero no uses una ETA total hasta verificar el contador actual.
- **Ajuste y verificación del ensemble de cinco semillas:** reservar 0,5–1,5 horas. Esta fase aún no tiene una medición en esta corrida.
- **Preparación, verificación de hashes/cobertura y evaluación de Chico Norte:** reservar 1–3 horas una vez que ambos modelos estén sellados. Parte de la preparación puede hacerse mientras termina MLP.

**No presentes una fecha de terminación todavía.** Actualiza el contador de Optuna y el avance de simulaciones de Chico Norte antes de comunicar una nueva ETA. El tiempo para una eventual versión entrenada con ambas redes tampoco está incluido.

## Siguientes pasos, en orden

1. Completar los 104 escenarios pendientes de Chico Norte y cerrar QA con el umbral de continuidad del 5 por ciento todavía provisional.
2. Completar los 50 trials MLP con Chico Sur únicamente; seleccionar parámetros con el protocolo registrado y ajustar el ensemble de cinco semillas.
3. Verificar que las corridas aptas de Chico Norte tengan INP, hidrogramas, RPT/OUT, hashes, variables y etiquetas enlazados por `run_id`. Mantener cerradas las etiquetas durante preparación y predicción.
4. Sellar los candidatos y sus predicciones para Chico Norte; ejecutar la evaluación externa una sola vez.
5. Reportar métricas generales y por forma/escenario; producir paridad y mapas SWMM versus modelo para Chico Norte.
6. Solo después de la evaluación de transferencia a Chico Norte, decidir si se crea una versión entrenada con ambas redes. Esa nueva versión ya no se evaluará independientemente en Chico Norte; para medir generalización se necesitará una tercera red no usada en el entrenamiento.

## Respuestas breves a preguntas probables

**¿El 73,78 m³ significa que el modelo se equivoca eso en Chico Norte?**
No. Es el mejor objetivo interno de XGBoost durante GroupKFold sobre Chico Sur. El error en Chico Norte sigue pendiente.

**¿Los 975 escenarios son 975 tormentas observadas?**
No. Son escenarios sintéticos formados por 39 hidrogramas normalizados y 25 factores de caudal. No modelan lluvia ni escorrentía.

**¿Por qué no entrenar ya con Chico Norte?**
Porque primero necesitamos mantener Chico Norte como prueba externa. Si sus etiquetas entran al entrenamiento antes de la evaluación, deja de ser una verificación independiente.

**¿Podemos decir que el sistema será confiable en otras redes?**
Todavía no. Chico Norte medirá la transferencia a una red. Para estimar comportamiento en una población de redes se necesitan más redes no vistas y justificar los criterios hidráulicos provisionales.

**¿Cuándo podremos mostrar paridad y mapas nuevos?**
Después de terminar MLP, sellar predicciones y completar la evaluación verificada de Chico Norte. La ETA necesita actualizarse con los contadores vigentes de ambos procesos.

## Nota de método

La comparación espacial de las diapositivas 7–8 usa un XGBoost anterior aplicado retrospectivamente a ejemplos de Chico Sur. Está rotulada como ilustrativa. No debe usarse para afirmar confiabilidad, exactitud independiente ni transferencia. Los valores internos de XGBoost/MLP tampoco son métricas de Chico Norte.
