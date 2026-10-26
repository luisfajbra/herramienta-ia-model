# Justificación de las variables tributarias en `tabular_v4_21`

## Propósito

Este documento justifica la inclusión de seis variables que resumen el hidrograma tributario de cada nodo de la red. El objetivo no es reemplazar una simulación hidráulica completa, sino entregar al modelo tabular información física que un único caudal pico no contiene.

En términos simples, para estimar si un nodo se inundará no basta preguntar «¿cuál fue el caudal máximo?». También importa cuánta agua llegó, cuánto tiempo llegó, cuándo llegó y cuánto excedió la capacidad de la red.

## Definición de caudal tributario

Para un nodo, `Q_tributario(t)` es la suma temporal del caudal del propio nodo y de los nodos únicos que aportan aguas arriba. Las series se interpolan sobre una misma malla temporal antes de sumarlas.

Esto evita una aproximación incorrecta: sumar los máximos individuales. Dos máximos que ocurren en instantes diferentes no necesariamente producen el mismo máximo ni la misma sobrecarga en el nodo receptor.

## Las seis variables

| Variable | Qué representa | Por qué ayuda a explicar la inundación |
|---|---|---|
| Volumen local integrado, m³ | Agua aportada directamente en el nodo. | Separa un nodo que produce mucha carga local de uno cuya carga proviene sobre todo de aguas arriba. |
| Volumen tributario integrado, m³ | Agua total recibida: aporte local más aporte de los ancestros. | Mide la carga hidráulica acumulada que debe conducir el sistema, incluso si el pico no es muy alto. |
| Duración efectiva tributaria | Tiempo durante el cual `Q_tributario` supera el 1 % de su propio pico. | Distingue un pulso corto de una carga sostenida. Una red puede soportar un pico breve y fallar ante un caudal moderado que persiste. |
| Tiempo al primer pico tributario | Instante del primer máximo de `Q_tributario`; se fija en cero si la serie es nula. | Representa la sincronización temporal de los aportes. El mismo volumen puede llegar antes, después o coincidir con otros aportes. |
| Tiempo sobre `upstream_capacity_lps` | Duración durante la cual el caudal tributario supera la capacidad agregada aguas arriba. | Indica por cuánto tiempo la demanda hidráulica supera la capacidad disponible. |
| Volumen de excedencia, m³ | Integral de `max(Q_tributario - capacidad, 0)`. | Cuantifica la gravedad acumulada de la sobrecarga: no solo si se excedió la capacidad, sino cuánto volumen no pudo ser transportado bajo ese límite. |

## Relación con la hidráulica de redes urbanas

SWMM y la literatura de drenaje urbano reportan variables como volumen de entrada, momento del caudal máximo, horas de surcharge, horas de inundación y volumen total inundado. Esto respalda usar descriptores que incorporen magnitud, duración, instante y excedencia, en vez de depender únicamente de un máximo de caudal.

La condición de *surcharge* aparece cuando la red deja de disponer de volumen libre suficiente o cuando el nivel en un nodo supera la corona del conducto más alto. Por ello, medir tiempo y volumen por encima de la capacidad es una forma interpretable de representar el mecanismo que precede a una posible inundación.

La inclusión de estas variables debe entenderse como una hipótesis física verificable: se debe evaluar su aporte mediante una ablación, comparando el desempeño del modelo con y sin ellas bajo los mismos folds de validación. Si no mejoran la predicción o dañan la generalización a hidrogramas no vistos, no deben mantenerse solo por plausibilidad hidráulica.

## Texto sugerido para la tesis

> Se incorporaron descriptores tributarios por nodo para resumir la carga hidráulica efectiva recibida desde el propio nodo y sus contribuyentes aguas arriba. Las variables representan magnitud acumulada, persistencia, sincronización temporal y excedencia respecto de la capacidad de transporte. Esta información no está contenida en un caudal pico aislado: eventos con el mismo máximo pueden diferir en volumen, duración, tiempo de llegada y sobrecarga acumulada, produciendo respuestas hidráulicas distintas. La selección se fundamenta en las variables hidráulicas reportadas por SWMM —volumen, instante de máximo, surcharge, duración y volumen de inundación— y se validará empíricamente mediante estudios de ablación.

## Referencias y enlaces

1. **U.S. Environmental Protection Agency.** *Storm Water Management Model (SWMM).* Página oficial, software y manuales de referencia. Explica el uso de SWMM para analizar y diseñar sistemas de drenaje urbano.  
   <https://www.epa.gov/water-research/storm-water-management-model-swmm>

2. **Rossman, L. A. (2022).** *Storm Water Management Model User’s Manual, Version 5.2.* U.S. Environmental Protection Agency. El manual documenta resúmenes de nodos con caudal máximo, instante de máximo, volumen total de entrada, horas de surcharge, horas de inundación y volumen total inundado.  
   <https://www.epa.gov/sites/production/files/2019-02/documents/epaswmm5_1_manual_master_8-2-15.pdf>

3. **Rossman, L. A. (2017).** *Storm Water Management Model Reference Manual, Volume II: Hydraulics.* U.S. Environmental Protection Agency. Define las condiciones de surcharge, flooding y el comportamiento hidráulico de nodos y conductos.  
   <https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100S9AS.TXT>

4. **Aderyani, O. A., et al. (2025).** *A surrogate machine learning modeling approach for enhancing the efficiency of urban flood modeling at metropolitan scales.* Sustainable Cities and Society, 123, 106277. El estudio desarrolla modelos surrogate para variables de surcharge por pozo, incluidas duración, pico y volumen, y destaca la relevancia de profundidad y duración para la predicción.  
   <https://www.sciencedirect.com/science/article/pii/S2210670725001544>

5. **Zahura, F. T., Goodall, J. L., Sadler, J. M., Shen, Y., Morsy, M. M., & Behl, M. (2020).** *Training Machine Learning Surrogate Models From a High-Fidelity Physics-Based Model: Application for Real-Time Street-Scale Flood Prediction in an Urban Coastal Community.* Water Resources Research, 56, e2019WR027038. Es un antecedente de uso de modelos surrogate entrenados con simulaciones hidráulicas físicas para predicción de inundación urbana.  
   <https://doi.org/10.1029/2019WR027038>

## Limitación explícita

Estas referencias justifican las variables como descriptores hidráulicamente razonables, pero no prueban por sí solas que las seis mejoren este caso de estudio. Esa afirmación exige resultados del experimento de ablación y validación sobre hidrogramas y formas no vistas.
