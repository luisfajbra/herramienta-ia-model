# Estado de la recalificación de la red A

> **Actualización 2026-10-26:** Las decisiones y el estado descritos más abajo eran correctos antes de la instrucción posterior del usuario de fijar `455C` en cero y reanudar A. La nueva recalificación completa de A está documentada en [2026-10-26-recalificacion-chico-hydro-qx1.md](2026-10-26-recalificacion-chico-hydro-qx1.md). La tabla nueva produjo 963 escenarios elegibles y 37 excluidos; los runs históricos permanecen intactos. La nota sobre `455C` aplica al INP de Chico Norte B, no al lote A.

## Decisiones confirmadas

- Reejecutar las 1.000 combinaciones como una versión nueva y reproducible, reconstruida desde el INP y los CSV actuales; conservar intactos los runs históricos.
- Usar provisionalmente el límite de continuidad de 5% y los valores ya configurados para cierre/reintentos y tolerancias. Estos umbrales todavía requieren justificación hidráulica.
- No se debe asumir el origen de las condiciones iniciales ni del hidrograma negativo del nodo `455C`.
- El usuario pidió detener la recalificación hasta aclarar el origen de `455C`. No se ejecutará ninguna simulación ni se generará una base de entrenamiento durante esa pausa.

## Trabajo realizado

- Se añadió una migración que registra una evaluación hidráulica inmutable por corrida. Corridas sin continuidad medida se pueden registrar como `unverified` e inelegibles; el loader de entrenamiento requiere una evaluación `eligible` y continuidad finita.
- Se cerraron hallazgos de política de admisión, entradas persistentes no contempladas, orden estable de errores, RPT incompleto, volúmenes de inundación ambiguos y clasificación de drenaje con datos ausentes.
- La forma base ya usa las curvas propias por nodo, evalúa el flujo efectivo y aplica cierre, vaciado y reintentos. La ruta anterior corría la base sin drain-down ni control de drenaje.
- El primer piloto de SWMM no llegó a iniciar: SWMM rechazó el INP generado porque el escritor serializaba el patrón opcional ausente como el literal `NONE`. Corregí el escritor para omitirlo y añadí una prueba sobre un INP real. No repetí el piloto ni ejecuté otra simulación.
- El manifiesto hidráulico ahora está protegido también en SQLite: una corrida no puede quedar `eligible` mediante un `INSERT` directo sin procedencia válida del estado inicial y del RPT, ni si el manifiesto contradice continuidad, cobertura o drenaje. Se actualizaron los fixtures sintéticos que alimentan las pruebas del cargador.
- Los escenarios base y los hidrogramas por forma ahora generan un manifiesto de condiciones iniciales desde el INP fuente. Incluye SHA-256 de bytes, valores iniciales de caudales de conducciones y profundidades de nodos, un identificador de política de registro y un hash canónico del estado. La rutina conserva el INP y no interpreta esos valores como válidos o inválidos.
- Suite enfocada actual: 238 pruebas pasaron.
- La auditoría de la ruta de recalificación confirmó que el lote actual puede ejecutar las 975 combinaciones de las 39 formas, pero la forma `base` usa otro camino sin evaluación de drenaje. No existe todavía una tubería completa para ejecutar, manifestar e ingerir las 1.000 corridas con QA común.
- El backfill existente no escribe hidrogramas efectivos ni manifiestos de calidad, y no es idempotente por escenario. La siguiente implementación debe usar un snapshot/DB nueva y selección explícita de sus nuevos `run_id`.

## Decisión hidráulica pendiente para Chico Norte (red B)

En el INP, las 61 muestras de `455C` son negativas (`−0,226` a `−0,011 L/s`). `apply_shape` omite nodos cuyo flujo base no es positivo; por eso `455C` no recibe los 39 perfiles escalados. La forma base conserva la serie del INP. Además, el esquema de entrenamiento exige flujos no negativos. No se debe poner esa serie en cero, escalarla, admitirla como válida ni entrenar con ella sin resolver su significado.

El usuario pidió pausar la recalificación hasta aclarar el origen de 455C. Por ello quedan detenidas las simulaciones de recalificación; no se han creado datos nuevos de entrenamiento ni se ha escrito ninguna base nueva. La admisión de Chico Norte también queda en pausa. Las 1.000 corridas nuevas de A aún requieren una tubería completa de snapshot/QA/ingestión; cualquier reanudación debe esperar la aclaración solicitada. Los arreglos independientes de software sí continúan con fixtures y pruebas unitarias, sin usar la red B.

## Verificación

Suite enfocada — base de datos, calidad, cargador, admisión, drenaje, topología, características, condiciones iniciales, hidrogramas y etiquetas:

Resultado más reciente: `267 passed, 31 warnings` en 25,70 s. Las advertencias son de dependencias, caché de pytest y fixtures que ejercitan series con recesión incompleta. Las pruebas de procedencia inicial, escenarios base/por forma y factor único sustituyen PySWMM por mocks; no ejecutan el motor SWMM.

La ejecución completa más reciente terminó con `1270 passed, 68 failed, 50 errors, 3 deselected`. Muchas pruebas que escriben mediante `tempfile` o el backend de build fallaron con `WinError 5` en este entorno Windows restringido. La expectativa del catálogo de migraciones obsoleta se actualizó y su suite de base de datos pasó dentro de las 259 pruebas enfocadas. El test de wheel aún no se pudo validar porque el backend también recibe `WinError 5` al crear su directorio temporal; el wheel no se declara comprobado.
