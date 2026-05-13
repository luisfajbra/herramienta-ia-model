## Revision detallada de `delta_inflow_lps` y escenarios parciales

### 1. Problema critico: corrupcion semantica de `delta_inflow_lps`

#### Que representa cada columna

- `runs.inflow_multiplier`
  Es un control de corrida a nivel global. Ejemplo: `0.5`, `1.0`, `1.5`, `3.0`.

- `node_results.delta_inflow_lps`
  Debe representar el caudal adicional aplicado a un nodo especifico, en `L/s`.
  No es un factor. No es metadata global. Es una magnitud hidraulica por nodo.

#### Por que el valor positivo en nodos con `base_inflow_lps = 0` no deberia pasar

Si la corrida usa un multiplicador sobre el inflow embebido, el caudal agregado al nodo se calcula asi:

`delta_nodo = inflow_base_nodo * (multiplicador - 1)`

Entonces, si un nodo tiene `inflow_base_nodo = 0`, su delta real tambien debe ser `0`.

Ejemplo:

- Nodo `2C`
- `base_inflow_lps = 0`
- `inflow_multiplier = 1.5`

Resultado correcto:

- `delta_inflow_lps = 0 * (1.5 - 1) = 0`

Si en la BD aparece:

- `delta_inflow_lps = 1.5`

ese dato esta mal por dos razones:

1. Tiene unidades equivocadas: `1.5` ahi es un factor, no un caudal en `L/s`.
2. Cambia el significado de la columna: deja de ser una entrada fisica por nodo y pasa a ser una copia degradada de metadata del run.

#### Donde nace el error

Hoy el riesgo esta en `swmm_resilience/database/schema.py`, dentro de `create_schema()`, cuando se hace este backfill:

- si `delta_inflow_lps` es `NULL` o `0`
- entonces se reemplaza por `runs.delta_inflow_lps`

Ese backfill es inseguro por tres motivos:

1. `0` es un valor valido, no un faltante.
2. `runs.delta_inflow_lps` hoy se usa como si fuera un valor de escenario global, no necesariamente un delta real por nodo.
3. Aunque el nombre coincida, la semantica no coincide entre tabla de corridas y tabla de resultados por nodo.

#### Por que es importante corregirlo

- Rompe auditoria: ya no podemos confiar en que `delta_inflow_lps` diga cuanto caudal extra recibio cada nodo.
- Rompe analisis hidraulico: nodos sin aporte real parecen perturbados.
- Rompe trazabilidad: dos columnas con nombres parecidos terminan diciendo cosas distintas.
- Puede contaminar ML futuro: si algun flujo usa `delta_inflow_lps` como feature, el modelo aprendera una mezcla de unidades y significados.

#### Señal observable en la BD actual

Ya se detectaron filas donde:

- `network_nodes.base_inflow_lps = 0`
- `node_results.delta_inflow_lps > 0`

Eso es evidencia directa de corrupcion semantica, no solo una sospecha teorica.

### 2. Plan recomendado para corregir `delta_inflow_lps`

#### Objetivo

Separar definitivamente:

- el control global de corrida
- del caudal adicional real por nodo

#### Plan tecnico

1. Dejar `runs.inflow_multiplier` como el valor canonico del escenario global.
2. Dejar `node_results.delta_inflow_lps` solo para el delta real por nodo en `L/s`.
3. Eliminar el backfill que pisa ceros validos.
4. No volver a copiar `runs.delta_inflow_lps` hacia tablas por nodo si la semantica no coincide.
5. Agregar una rutina de diagnostico para identificar filas historicas contaminadas.
6. Evaluar una reparacion historica:
   - si el run puede reconstruirse con datos confiables, recalcular `delta_inflow_lps`
   - si no, marcar esas filas como no confiables o regenerar la BD desde simulaciones limpias

#### Como deberia quedar la migracion

El criterio seguro es:

- solo backfill de campos realmente faltantes
- nunca tratar `0` como faltante
- nunca copiar valores entre columnas con semantica diferente aunque el nombre se parezca

#### Riesgo operativo

Si esta correccion no se hace, cada apertura de una BD legacy puede seguir deformando datos historicos sin que el usuario lo note.

### 3. Problema alto para revision: escenarios parciales en el dataset de ML

#### Que pasa hoy

Cuando una corrida aplica el cambio solo a un subconjunto de nodos:

- `node_results.delta_inflow_lps` si conserva el delta por nodo
- pero el CSV exportado para ML privilegia `runs.inflow_multiplier`
- y el pipeline de features actual excluye `delta_inflow_lps`

#### Por que esto importa

En un escenario parcial, dos nodos del mismo run pueden tener realidades distintas:

- nodo A si fue perturbado
- nodo B no fue perturbado

Pero si el dataset solo expone el multiplicador global del run, ambos quedan descritos igual desde el punto de vista del modelo.

Eso produce una representacion incompleta del experimento:

- el modelo no sabe cuales nodos recibieron el cambio
- la variable de entrada deja de corresponder exactamente al target observado

#### Impacto

- Dificulta interpretar el dataset.
- Hace mas riesgoso entrenar modelos con escenarios parciales mezclados.
- Puede inducir ruido o sesgo en entrenamiento e inferencia.

#### Estado acordado en esta revision

Este punto queda documentado para revision detallada posterior.
No se cambia su manejo en esta iteracion.

### 4. Recomendacion practica

Antes de volver a usar `delta_inflow_lps` para analisis o ML:

1. Congelar la logica de migracion insegura.
2. Etiquetar o aislar las BDs posiblemente contaminadas.
3. Decidir si se reconstruye historial o si se parte desde una BD limpia nueva.
