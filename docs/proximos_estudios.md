# Próximos estudios

Este documento reúne líneas de investigación que pueden aportar valor al
proyecto, pero que quedan fuera del alcance de la consolidación actual del
pipeline SQLite y del contrato tabular de 17 features. Su inclusión no implica
un compromiso de implementación.

## Estimación de incertidumbre con GLUE

La metodología *Generalized Likelihood Uncertainty Estimation* (GLUE) puede
emplearse para estudiar cómo la incertidumbre de los parámetros de un modelo
hidráulico se transmite a sus resultados. En este proyecto permitiría evaluar
si diferentes combinaciones plausibles de parámetros SWMM producen respuestas
hidráulicas igualmente aceptables, aunque no exista una única calibración
óptima. Este fenómeno se conoce como equifinalidad.

### Posible pregunta de investigación

¿Cómo cambia la estimación de inundación nodal y volumen total cuando se
consideran múltiples parametrizaciones plausibles de SWMM, en lugar de una
sola configuración determinista?

### Aplicación potencial

Un estudio posterior podría:

1. Seleccionar parámetros SWMM cuya incertidumbre tenga justificación física,
   por ejemplo rugosidad, infiltración, pérdidas o condiciones iniciales.
2. Definir rangos o distribuciones de muestreo respaldados por mediciones,
   bibliografía o criterios de ingeniería.
3. Ejecutar un conjunto amplio de simulaciones.
4. Comparar cada simulación con observaciones mediante una medida de ajuste
   declarada previamente.
5. Clasificar como conductuales las parametrizaciones que superen un umbral
   explícito.
6. Construir intervalos de resultados ponderados por su medida de ajuste.
7. Evaluar la sensibilidad de las conclusiones al número de muestras, la
   medida de ajuste y el umbral de aceptación.

El surrogate podría reducir el costo de explorar muchas combinaciones, pero no
debería sustituir completamente a SWMM dentro de este análisis. Sería necesario
validar una muestra con SWMM e incorporar el error propio del surrogate para no
confundir incertidumbre hidráulica con error de aproximación del modelo de
machine learning.

### Precauciones metodológicas

GLUE incluye decisiones subjetivas que pueden modificar de forma importante
los intervalos obtenidos:

- parámetros considerados inciertos;
- distribuciones o rangos de muestreo;
- medida de ajuste usada como likelihood informal;
- umbral que separa simulaciones conductuales y no conductuales;
- cantidad y estrategia de muestreo.

La metodología también ha sido cuestionada por no constituir necesariamente
una inferencia bayesiana coherente cuando emplea likelihoods informales. Por
esta razón, un estudio serio debería justificar todas las decisiones, presentar
análisis de sensibilidad y comparar los resultados con una alternativa formal
cuando sea viable.

Referencias de partida:

- Mantovan, P. y Todini, E. (2006), [*Hydrological forecasting uncertainty
  assessment: Incoherence of the GLUE methodology*](https://doi.org/10.1016/j.jhydrol.2006.04.046).
- Beven, K. J., Smith, P. J. y Freer, J. E. (2008), [*So just why would a
  modeller choose to be incoherent?*](https://doi.org/10.1016/j.jhydrol.2008.02.007).
- Jin, X. et al. (2010), [*Parameter and modeling uncertainty simulated by
  GLUE and a formal Bayesian method for a conceptual hydrological
  model*](https://doi.org/10.1016/j.jhydrol.2009.12.028).
- Stedinger, J. R. et al. (2008), [*Appraisal of the generalized likelihood
  uncertainty estimation (GLUE) method*](https://doi.org/10.1029/2008WR006822).

### Relación con la incertidumbre del surrogate

GLUE está orientado principalmente a incertidumbre paramétrica y estructural
del modelo hidráulico. No reemplaza una evaluación específica de la
incertidumbre predictiva del surrogate. Para esta última podrían estudiarse,
por separado:

- bootstrap o ensembles;
- predicción conformal;
- regresión cuantílica;
- calibración de probabilidades de clasificación;
- propagación conjunta del error SWMM y del error del surrogate.

## Recomendación

GLUE puede ser una recomendación final valiosa si el trabajo busca discutir
calibración, equifinalidad o incertidumbre de parámetros SWMM. No es un
requisito para consolidar el pipeline ni para validar inicialmente el surrogate
de 17 features. Se recomienda abordarlo como un estudio independiente después
de estabilizar la persistencia SQLite, la trazabilidad de escenarios y la
evaluación reproducible de los modelos.
