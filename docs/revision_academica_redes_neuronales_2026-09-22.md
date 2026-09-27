**Revisión académica: redes neuronales como sustitutos hidráulicos para inundación por nodo**

Fecha de consulta: 22 de septiembre de 2026.

Alcance acordado: predecir ocurrencia y volumen total de inundación por nodo; primero en una misma red ante hidrogramas nuevos, después estudiar transferencia. Se dispone de una red, generación de escenarios sintéticos, etiquetas de SWMM y GPU. Interesan tanto el estado del arte como una futura implementación. Se aceptan fuentes en inglés, revistas, congresos y tesis.

**Conclusión de la búsqueda**

Sí existen antecedentes neuronales próximos. Sin embargo, hay que distinguir tres objetivos: predecir directamente un volumen por evento, reconstruir estados hidráulicos temporales y calcular después el desbordamiento, o emular solamente una parte de un simulador. No son experimentos equivalentes.

La búsqueda encontró artículos publicados en Water Research, Water Resources Research y Journal of Water Resources Planning and Management, con registros editoriales o institucionales verificables. TU Delft, Technical University of Denmark, University of Manchester y City University of Hong Kong aparecen en las fuentes seleccionadas. La selección considera correspondencia del objetivo, metodología y reproducibilidad; no atribuye cuartiles o factores de impacto sin verificar año y categoría.

Esta es una revisión dirigida, no una revisión sistemática exhaustiva. No permite afirmar que una combinación específica de modelos constituya por sí sola una novedad científica.

**Trabajos prioritarios y qué aportan**

1. **Zhang, Z., Tian, W., Lu, C., Liao, Z. y Yuan, Z. (2024). Graph neural network-based surrogate modelling for real-time hydraulic prediction of urban drainage networks. Water Research, 263, 122142.**

   Es el antecedente prioritario para conectar redes neuronales, SWMM y detección de inundación. Usa estados hidráulicos recientes, escorrentía futura y condiciones de control; incorpora relaciones entre nodos y tuberías y restricciones físicas. Su salida es temporal, por lo que no reproduce directamente la tabla nodo–evento de este proyecto.

   La versión de autor consultada describe una pérdida combinada para estados de nodos, estados de aristas y clasificación de inundación; el clasificador restringe la estimación de desbordamiento para evitar atribuir inundación a simples errores de balance. Ese detalle se consultó en un preprint de un trabajo cuya publicación revisada está verificada; debe contrastarse con la versión editorial antes de replicar cifras o ecuaciones.

   [Artículo/DOI](https://doi.org/10.1016/j.watres.2024.122142) · [Registro institucional](https://scholars.cityu.edu.hk/en/publications/graph-neural-network-based-surrogate-modelling-for-real-time-hydr/) · [Versión de autor](https://arxiv.org/pdf/2404.10324) · [Código de los autores](https://github.com/Zhiyu014/GNN-UDS).

2. **Palmitessa, R., Grum, M., Engsig-Karup, A. P. y Löwe, R. (2022). Accelerating hydrodynamic simulations of urban drainage systems with physics-guided machine learning. Water Research, 223, 118972.**

   DTU. Usa redes residuales y restricciones físicas para reproducir niveles, caudales y desbordamientos. Evalúa volumen excedente por nodo y evento: es especialmente pertinente para tu variable objetivo. El simulador de referencia es **MIKE 1D**, no SWMM.

   Preparan dos conjuntos de lluvias; uno refuerza eventos intensos. Entrenan con ventanas temporales y evalúan eventos independientes. Utilizan Adam, hasta 2.000 épocas, tasa decreciente de 0,001 a 0,0001 y parada tras 500 épocas sin mejorar el MSE de validación. Comparan tamaños de red y ventanas con cinco inicializaciones; seleccionan seis capas de 100 neuronas y ventanas de 60 minutos. Son resultados de ese experimento, no parámetros óptimos universales.

   Aporta una justificación experimental para estudiar cobertura de extremos, restricciones físicas y variabilidad entre semillas.

   [Artículo/DOI](https://doi.org/10.1016/j.watres.2022.118972) · [Texto publicado en DTU](https://backend.orbit.dtu.dk/ws/portalfiles/portal/282620979/1_s2.0_S0043135422009198_main.pdf), secciones 2.3–2.6 y 3.1–3.2.

3. **Garzón, A., Kapelan, Z., Langeveld, J. y Taormina, R. (2024). Transferable and data efficient metamodeling of storm water system nodal depths using auto-regressive graph neural networks. Water Research, 266, 122396.**

   TU Delft. GNN autorregresiva de SWMM para cargas hidráulicas. Organiza geometría, topología, escorrentía y estados en ventanas; aplica min–max y balancea ventanas secas/húmedas. Usa 160 eventos: 100 de entrenamiento, 30 de validación y 30 de prueba. Estos últimos corresponden a lluvias reales, pero las respuestas hidráulicas siguen siendo simuladas.

   Entrena con AdamW, MSE y 100 épocas; conserva los pesos de menor pérdida de validación y aumenta la dificultad temporal durante el entrenamiento. Estudia eficiencia al variar eventos y semillas. La transferencia presentada a otra parte de la misma red es **sin reajuste**, no fine-tuning.

   No predice directamente el volumen desbordado; el propio artículo reconoce esa limitación. Es una referencia metodológica y dispone de datos y código públicos.

   [Artículo/DOI](https://doi.org/10.1016/j.watres.2024.122396) · [Texto publicado](https://repository.tudelft.nl/file/File_82fef3ff-9a5f-4275-bd8f-6a4eb06c463d) · [Código](https://github.com/alextremo0205/SWMM_GNN_Repository_Paper_version).

4. **Garzón, A., Kapelan, Z., Langeveld, J. y Taormina, R. (2026). Evaluation of graph neural networks for urban drainage metamodeling: Key components and transferability analysis. Water Research, 290, 125079.**

   TU Delft. Compara tipo de capa de grafo, profundidad y horizonte de predicción en dos redes. Predice cargas hidráulicas y caudales, no directamente volumen total de inundación.

   Su configuración común incluye AdamW, tasa 0,001, pérdida Huber, dimensión oculta 32, lotes de 20 eventos y 25 épocas; selecciona pesos mediante validación. Los lotes de eventos no equivalen a lotes de filas tabulares.

   Para transferencia, compara reutilización sin ajuste, reajuste y entrenamiento conjunto. Al cambiar de tarea, prueba actualizar solo el decodificador o todos los pesos. La transferencia sin ajuste entre redes distintas funciona mal; adaptar con datos de destino recupera desempeño. Es útil para tu segunda etapa, sin asumir que unas pocas redes prueben generalización universal.

   [Artículo/DOI](https://doi.org/10.1016/j.watres.2025.125079) · [Texto publicado](https://repository.tudelft.nl/file/File_1cf09147-d8d4-4be3-8b42-f2e508ab64a6) · [Código](https://github.com/alextremo0205/SWMM_GNN_Component_Evaluation_and_Transferability_Analysis). Se cita 2026 por el volumen publicado; el DOI contiene 2025.

5. **Brazão, A. J. C., Silva, C. M., Costa, M. E. L., Koide, S. y Silva, G. B. L. (2026). Evaluation of Artificial Neural Network Surrogate Models for Detention-Based Flood Control in Urban Areas. Journal of Water Resources Planning and Management, 152(11).**

   Artículo de ASCE publicado en línea el 3 de septiembre de 2026. Entrena MLP con simulaciones SWMM para volumen total de inundación y caudal máximo, considerando características de lluvia y sistemas de detención. Compara tamaño de dataset y arquitectura; aumentar de 20.000 a 50.000 muestras aporta mejoras marginales y profundizar puede producir sobreajuste.

   Es cercano a una regresión por escenario, pero el resumen consultado no demuestra salida de volumen para cada nodo. Por eso se clasifica como antecedente parcial. Solo se verificó el resumen editorial: normalización, particiones y optimizador quedan pendientes de acceso al texto completo; no deben inventarse a partir del resumen.

   [Artículo y resumen editorial](https://doi.org/10.1061/JWRMD5.WRENG-7598).

6. **Seyedashraf, O., Bottacin-Busolin, A. y Harou, J. J. (2021). A Disaggregation–Emulation Approach for Optimization of Large Urban Drainage Systems. Water Resources Research, 57, e2020WR029098.**

   University of Manchester y UCL. Compara MLP y GRNN para representar condiciones hidráulicas en interfaces de una red dividida. Genera 2.000 configuraciones con SWMM, separa 70/15/15 para entrenamiento, validación y prueba y explora arquitecturas MLP. Después integra el emulador con simulación y optimización.

   Es útil para entender generación de datos, ajuste de arquitectura y uso posterior del modelo. Sigue simulando una parte de la red: no sustituye todo SWMM ni equivale a un predictor directo de inundación en cada nodo.

   [Artículo/DOI](https://doi.org/10.1029/2020WR029098) · [Texto publicado en Manchester](https://pure.manchester.ac.uk/ws/portalfiles/portal/197742710/2020WR029098.pdf).

7. **Garzón, A., Kapelan, Z., Langeveld, J. y Taormina, R. (2022). Machine Learning-Based Surrogate Modeling for Urban Water Networks: Review and Future Research Directions. Water Resources Research, 58, e2021WR031808.**

   Revisión de 31 trabajos sobre sustitutos de modelos de redes de agua. Recomendada para estructurar el estado del arte, diferenciar propósitos y justificar preguntas sobre dimensionalidad, interpretabilidad y transferencia. No aporta un experimento propio que debas replicar.

   [Artículo de revisión](https://doi.org/10.1029/2021WR031808).

**Un comparador muy cercano que no es neuronal**

Aderyani, F. R., Jafarzadegan, K. y Moradkhani, H. (2025), *A surrogate machine learning modeling approach for enhancing the efficiency of urban flood modeling at metropolitan scales*, Sustainable Cities and Society, 123, 106277. Predice duración, pico y volumen por pozo mediante agrupación hidráulica y Random Forest. Es pertinente para comparar el objetivo y la construcción de datos, pero no debe presentarse como evidencia de redes neuronales. [Artículo](https://doi.org/10.1016/j.scs.2025.106277) · [Repositorio de los autores](https://github.com/FRAderyani/Metropolitan-SurrogateModel).

**Qué significa entrenamiento y qué significa ajuste**

Entrenar estima los pesos de una arquitectura fijada mediante una función de pérdida. Ajustar hiperparámetros compara decisiones externas a esos pesos: capas, neuronas, tasa de aprendizaje, regularización, tamaño de lote o duración del entrenamiento. Fine-tuning, en sentido estricto, parte de pesos previamente entrenados y los adapta. Tu prioridad es la segunda operación; no requiere disponer de un modelo preentrenado. Los experimentos de transferencia de la referencia 4 corresponden a una etapa posterior.

**Lectura del proyecto local**

Se revisaron config.yaml, ml/contracts.py y los módulos ml/bench/models/mlp_family.py, ml/bench/preprocess.py y ml/bench/folds.py.

El banco ya contiene una MLP con 17 entradas, capas ocultas 64 y 32, ReLU, dropout 0,1, Adam, tasa 0,001, lotes de 256 y 200 épocas. Tiene clasificador y regresor separados; imputa por mediana y estandariza. La pérdida de clasificación admite ponderación positiva; el regresor usa MSE sobre una etiqueta que debe recibir transformada en log1p. El bucle inspeccionado usa épocas fijas, sin selección interna de pesos por validación. Esto describe código y configuración, no acredita resultados de un entrenamiento ejecutado en esta revisión.

El preparador del banco construye particiones usando factor_mult. Eso evalúa factores no vistos, pero por sí solo no prueba generalización a familias de hidrogramas desconocidas. Ambas preguntas merecen pruebas separadas. El wrapper MLP inspeccionado trabaja en CPU; disponer de GPU no significa que esa implementación ya la utilice.

**Propuesta para tu experimento — recomendaciones propias, no una receta atribuida a los artículos**

| Etapa | Decisión propuesta | Evidencia a guardar |
|---|---|---|
| 1. Objetivo | Una muestra nodo × evento; etiquetas ocurrencia y volumen total en m³ | Definición de umbral, unidades y configuración de ponding |
| 2. Escenarios | Variar pico, duración, volumen entrante, posición y número de picos; cuando proceda, desfases entre entradas | Identificador de familia, parámetros y semilla del generador |
| 3. Simulación | Obtener etiquetas con SWMM y revisar continuidad, inestabilidades y tiempo de vaciado | Red, versión de SWMM y controles de calidad por corrida |
| 4. Particiones | Separar eventos completos; reservar familias de forma y extremos para pruebas específicas | Listas fijas de entrenamiento, validación y prueba |
| 5. Preparación | Ajustar imputación, escalado y cualquier selección usando solo entrenamiento | Transformadores persistidos dentro de cada partición |
| 6. Entrenamiento | Comparar MLP y XGBoost con las mismas entradas; mantener clasificación y regresión inicialmente | Pérdidas, métricas, semillas y pesos seleccionados |
| 7. Ajuste | Buscar hiperparámetros con validación interna agrupada | Presupuesto y registro completo de intentos |
| 8. Evaluación | Abrir la prueba reservada al terminar las decisiones | Errores por nodo/evento y rendimiento del sistema completo |
| 9. Uso | Guardar modelo y preprocesamiento, inferir nuevos eventos y producir mapas/curvas | Dominio de uso, tiempos y comparación con SWMM |

Tener escenarios sintéticos ilimitados elimina una restricción de acceso, pero no elimina el costo de simular ni garantiza diversidad física. Conviene medir curvas de aprendizaje por **número de eventos y familias independientes**, además del número de filas. Miles de nodos de una corrida no equivalen a miles de eventos independientes.

Para una comparación inicial controlada conservaría las 17 entradas. Después estudiaría una ampliación que describa mejor la forma temporal: volumen entrante, duración sobre umbrales, asimetría o múltiples picos. Dos hidrogramas pueden compartir pico, duración y tiempo al pico sin producir idéntica respuesta. Si las entradas no los distinguen, aumentar la complejidad neuronal no recupera esa información.

Mantendría al principio dos modelos: clasificador de inundación y regresor condicional de volumen. Para entrenar el segundo, usaría exclusivamente las filas inundadas del conjunto de entrenamiento. Evaluaría el sistema combinado sobre todos los nodos para que los falsos negativos también cuenten. Una red compartida con dos salidas sería una comparación adicional, no una mejora garantizada.

**Espacio inicial de búsqueda para la MLP**

Estos rangos son una propuesta que debe adaptarse al tamaño efectivo del dataset y al costo observado; no son valores copiados de un artículo ni óptimos demostrados.

| Hiperparámetro | Espacio inicial |
|---|---|
| Capas ocultas | 1–3 |
| Anchos | 32, 64, 128, 256; incluir 64–32 como referencia |
| Tasa de aprendizaje | 10⁻⁴ a 3×10⁻³, escala logarítmica |
| Dropout | 0–0,3 |
| Weight decay | 0 y 10⁻⁶ a 10⁻² |
| Tamaño de lote | 128, 256, 512 |
| Pérdida de regresión | MSE o Huber sobre log1p(volumen); evaluar siempre también en m³ |
| Selección de época | Mejor resultado de validación, con parada temprana |

El código actual no expone todas estas alternativas: incorporarlas sería una tarea de implementación posterior. El ajuste necesita validación interna separada de los folds utilizados para estimar desempeño final. Se puede usar búsqueda aleatoria u optimización bayesiana; lo esencial es que todas las decisiones, incluido el umbral del clasificador, se tomen sin mirar la prueba final. Repetiría los mejores candidatos con varias semillas y daría a XGBoost un presupuesto comparable.

Como criterio principal conservaría RMSE de volumen del sistema completo, en m³, acompañado de recall y precisión de inundación, F1/CSI, PR-AUC, MAE y sesgo del volumen total por evento. También reportaría errores cerca del umbral, en eventos extremos y por familia. Evitaría MAPE como única métrica porque hay volúmenes nulos o muy pequeños. Si se ponderan clases, comprobaría calibración antes de interpretar la salida como probabilidad física de inundación.

**Qué hacer después del entrenamiento**

Una vez elegidos arquitectura e hiperparámetros, entrenar el modelo final con los datos de desarrollo siguiendo una regla de selección de época predefinida; evaluar una vez en la prueba reservada; persistir modelo, transformadores, unidades y contrato de entradas. Medir tiempo de extracción de variables más inferencia, y distinguirlo del costo previo de generar etiquetas y ajustar modelos. Frente a familias fuera del dominio validado, contrastar con SWMM antes de extender las conclusiones.

Para transferencia, conseguir varias redes y reservar al menos una como destino. Comparar entrenamiento desde cero, aplicación sin ajuste y reajuste con exactamente los mismos presupuestos de eventos del destino. La evaluación final debe usar eventos de destino diferentes de los de adaptación. Si hay suficientes redes, repetir dejando fuera una red cada vez; con pocas redes, las conclusiones serán de casos estudiados.

Una GNN sería una segunda línea porque permite representar explícitamente conectividad y atributos de tuberías. Para tu objetivo hay dos diseños posibles: salida directa de dos variables por nodo/evento, o predicción temporal del desbordamiento seguida de integración. La primera es una adaptación a investigar, no un resultado ya demostrado por los artículos de cargas hidráulicas. La segunda exige series de etiquetas y una definición consistente de caudal de inundación e integración temporal. **Un buen ajuste de niveles no demuestra por sí solo un buen ajuste de volumen desbordado.**

**Orden sugerido de lectura**

Empezar con Zhang 2024 y Palmitessa 2022 para entender inundación y volúmenes; seguir con Garzón 2024 para preparación y entrenamiento reproducible; consultar Brazão 2026 para MLP por escenario si se consigue texto completo; usar Garzón 2022 para organizar antecedentes y Garzón 2026 al diseñar la transferencia. Seyedashraf 2021 ayuda a explicar el uso posterior del sustituto en optimización.

No se modificó ni entrenó el sistema durante esta revisión. Quedan por comprobar los detalles no accesibles del artículo ASCE y, antes de replicar Zhang, las diferencias entre su preprint y la versión editorial.
