# Guion para el avance del 30 de octubre de 2026

**Presentación:** `Presentacion_MLSMFP_30-10_avance_chico_norte_y_modelos.pptx`  
**Duración sugerida:** 8–9 minutos, más preguntas.  
**Idea central:** Chico Norte ya fue incorporada, simulada y preparada como prueba externa; el desempeño de los modelos en esa red todavía no se ha medido.

## Antes de exponer

- XGBoost obtuvo un mejor objetivo interno de `73,782 m³` (trial 13); MLP obtuvo `86,420 m³` (trial 48). Son resultados de selección interna en Chico Sur, con GroupKFold de cinco folds por forma y 50 ensayos por modelo. No son métricas de Chico Norte ni una estimación de desempeño en una población de redes.
- El ensemble final MLP de cinco semillas (42–46) y el candidato XGBoost están ajustados y su procedencia fue verificada.
- En Chico Norte terminaron las 975 simulaciones: 787 son elegibles y 188 se excluyeron con QA provisional. Se verificaron 3.901 artefactos enlazados con hashes.
- Sí hay mapas topológicos de Chico Norte y mapas de inundación SWMM de Chico Sur. Las comparaciones con XGBoost de las diapositivas 7–8 usan un modelo anterior y son retrospectivas. Todavía no hay paridad del candidato actual ni mapas de sus predicciones para Chico Norte.
- La regla de continuidad del 5% y los criterios de drenaje siguen siendo provisionales. La evaluación externa real permanece cerrada hasta congelar explícitamente el dominio estructural y sellar predicciones fuente-enlazadas.

## Guion por diapositiva

### 1. Portada

> Buenos días. Presento el avance de la auditoría hidráulica y del ajuste de modelos, con foco en la incorporación de Chico Norte. Ya completamos sus simulaciones de caudal sintético; el trabajo ahora es preparar una evaluación de transferencia que conserve a Chico Norte como red externa.

### 2. Resumen

> Chico Sur aporta 138.560 filas de 866 corridas elegibles. XGBoost y MLP completaron la búsqueda interna, y el ensemble MLP de cinco semillas ya está ajustado. Chico Norte tiene 975 escenarios simulados y clasificados por QA. Todavía no reportamos un resultado de transferencia: falta aprobar el dominio de aplicación y sellar las predicciones antes de consultar sus etiquetas.

### 3. QA de Chico Sur

> Reconstruimos y recalificamos las mil corridas de Chico Sur. El filtro provisional dejó 866 elegibles y excluyó 134; 100 casos no alcanzaron el drenaje requerido y 35 superaron el cinco por ciento de continuidad, con un caso que comparte ambos motivos. El conjunto tiene 160 nodos y 21 variables. El umbral de inundación usado es un metro cúbico. El cinco por ciento y las tolerancias de drenaje todavía requieren justificación hidráulica.

### 4. Diseño de transferencia

> La selección se hizo solo con Chico Sur y separó las formas de hidrograma en cinco folds. Chico Norte se mantiene fuera del ajuste y de la selección. Los manifiestos enlazan las fuentes hidráulicas, las variables y los resultados por corrida. La prueba automatizada del scorer se hizo con fixtures sintéticos; eso verifica el control de procedencia, pero no equivale a haber medido Chico Norte.

### 5. Topología de Chico Norte

> El mapa muestra la topología leída del INP: 523 nodos con coordenadas y 522 conducciones. Es una vista de la geometría de la red, no un mapa de inundación ni una predicción.

### 6. Campaña de simulación

> Ejecutamos 975 escenarios, combinando 39 formas de hidrograma con 25 factores entre 0,2 y 5,0. Las entradas son caudales sintéticos en nodos, escalados desde los picos del INP; no son lluvias observadas y no incluyen transformación lluvia–escorrentía. Los tiempos se redondearon al minuto más cercano, como se aprobó, con un desplazamiento máximo de 30 segundos.

### 7–8. Mapas de Chico Sur

> Estas diapositivas muestran mapas espaciales de dos escenarios de Chico Sur. La referencia es el resultado de SWMM; la segunda imagen reproduce una predicción retrospectiva de un XGBoost anterior. En el escenario mediano, los totales son 9.518 y 9.278 metros cúbicos; en el de máximo volumen, 183.407 y 188.628. Son ejemplos ilustrativos, no validación del XGBoost y MLP recién ajustados ni prueba de transferencia.

### 9. Selección interna

> Bajo el mismo objetivo y protocolo interno, el mejor ensayo de XGBoost obtuvo 73,782 metros cúbicos y el de MLP 86,420. Ambos completaron 50 ensayos Optuna usando cinco folds agrupados por forma, solo con Chico Sur. El valor bajo favorece a XGBoost en esta búsqueda, pero la selección puede optimizarse al conjunto A y no anticipa el error en Chico Norte. MLP ya se ajustó como ensemble de cinco semillas.

### 10. Estado de Chico Norte

> Chico Norte está completa: 975 de 975 escenarios. El QA provisional dejó 787 elegibles y 188 excluidos. Se verificaron hashes para 3.901 archivos enlazados, incluidos los artefactos de simulación. Estos conteos describen la campaña y su calidad; no describen el desempeño de los modelos.

### 11. Qué hay y qué falta para la transferencia

> Ya tenemos la topología de Chico Norte y mapas SWMM de inundación para Chico Sur. Las comparaciones con el modelo anterior siguen siendo retrospectivas. No tenemos todavía paridad del candidato actual ni mapas de predicción de Chico Norte. El siguiente paso es congelar la política de compatibilidad, registrar los modelos y sus fuentes, generar y sellar predicciones, y recién entonces abrir los reportes elegibles para calcular métricas y producir nuevos mapas.

### 12–13. Cierre y trazabilidad

> El resultado del periodo es una campaña A/B trazable y dos candidatos A-only listos para la prueba externa. Chico Norte todavía no entra al entrenamiento: primero debe medir la transferencia. Los criterios de QA permanecen provisionales. Si la evaluación externa es satisfactoria, podremos decidir si entrenamos una versión combinada; para evaluar la generalización de esa versión necesitaremos una tercera red independiente.

## Respuestas breves a preguntas probables

**¿73,78 m³ significa que XGBoost se equivoca eso en Chico Norte?**  
No. Es el mejor objetivo interno de la búsqueda en Chico Sur; el error externo de Chico Norte todavía no está medido.

**¿Tenemos mapas de inundación?**  
Sí: mapas de SWMM para escenarios de Chico Sur. También existe la topología de Chico Norte. No hay mapas de predicción de los candidatos actuales para Chico Norte.

**¿Las diapositivas 7–8 comparan los modelos actuales?**  
No. Muestran una comparación retrospectiva con el XGBoost promovido anteriormente; está rotulada como ilustrativa.

**¿Por qué no entrenamos ya con ambas redes?**  
Porque Chico Norte es la prueba externa acordada. Incorporar sus etiquetas antes de medir la transferencia contaminaría esa evaluación. Después se puede entrenar una versión conjunta, que requerirá una tercera red para volver a probar transferencia.

**¿Qué significa que 787 escenarios sean elegibles?**  
Que pasan los filtros operativos actuales de cobertura, continuidad y drenaje. Parte de los umbrales aún es provisional, así que debe presentarse como QA aplicado, no como estándar hidráulico validado.

**¿Los 975 escenarios son tormentas reales?**  
No. Son combinaciones sintéticas de formas normalizadas de caudal nodal y factores de escala.
