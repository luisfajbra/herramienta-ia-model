# Preparación del entrenamiento conjunto de Chico Sur y Chico Norte

**Fecha:** 3 de noviembre de 2026. El investigador autorizó el uso provisional de Chico Norte para desarrollo con los `InitFlow` originales y el QA acordado. Esta decisión y sus límites están registrados en la [especificación de generalización](../specs/2026-10-25-generalizacion-entre-redes-design.md). No significa que la red esté validada físicamente.

## Inventario comprobado en lectura

Se abrieron en modo de solo lectura las dos bases de las simulaciones congeladas. La base supervisada de Chico Sur contiene 1.000 corridas con dictamen de QA y 138.560 filas de features y objetivos. La base de Chico Norte contiene 975 corridas con dictamen, 410.814 filas de features y **cero objetivos almacenados**. En el informe de transferencia figuran 787 corridas elegibles y 188 excluidas para Chico Norte; las 410.814 filas corresponden a 787 × 522 nodos. El scoring anterior obtuvo objetivos de los RPT verificados sin insertarlos en esa base.

Fuentes locales revisadas:

- `outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z/A/A_training_verified_v3.sqlite3`.
- `outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z/B/B_verified_label_free_v1.sqlite3`.

En los protocolos de las dos redes aparecen 39 identificadores de forma comunes y una forma adicional `base` en Chico Sur. Los recibos de anclaje incluyen los mismos hashes de los CSV de esas 39 formas para ambas redes; los 39 archivos actuales coincidieron con esos hashes en la revisión del 3 de noviembre. Esto no sustituye volver a comprobarlos al construir la base ni descarta otras relaciones físicas entre formas.

Ambas bases son individuales. El preparador ML admite varias redes en una base, pero la ruta supervisada inspeccionada exige una base nueva y está específicamente restringida a Chico Sur. La base de Chico Norte se construyó sin etiquetas para proteger la evaluación congelada. **Por tanto, no existe todavía, a partir de este inventario, una base conjunta supervisada y verificada lista para entrenar.** No se debe copiar o combinar bases a mano: hay que mantener la identidad compuesta por red, corrida y nodo, las decisiones QA, las fuentes y los hashes; los `network_id` enteros locales pueden coincidir en bases distintas.

## Siguiente preparación

1. Usar los lotes principales elegibles y conservar separados los diez escenarios de la presentación, según decisión explícita del investigador. Esos diez resultados ya se observaron: conservarlos separados no los convierte en una prueba ciega independiente.
2. Diseñar y probar una materialización supervisada multired que verifique anclas, INP, eventos, RPT/OUT y QA; extraiga objetivos únicamente de corridas elegibles y genere una base nueva y manifiestos auditables sin alterar los artefactos anteriores.
3. Definir y registrar particiones por identidad de forma y sus factores, con grupos disjuntos entre ajuste y evaluación; comprobar que la misma forma usada en ambas redes no se filtre entre particiones. El análisis de ambas redes conocidas no demuestra transferencia a una red nueva.
4. Comprobar contrato de variables, cobertura por red y clase, registros de exclusiones, recursos y compatibilidad de los comandos antes de iniciar búsqueda de hiperparámetros o entrenamiento. Conservar `macro_network_rmse` y los demás criterios ya aprobados; cualquier decisión nueva sobre particiones, inclusión de lotes o presupuesto requiere acuerdo.
5. Reservar una tercera red sin uso para ajuste o selección si se busca medir transferencia independiente a una red no vista. No hay una tercera red incorporada ni evaluada en esta entrega.

No se modificaron las bases, no se extrajeron nuevas etiquetas para entrenamiento y no se entrenaron modelos durante este inventario.

## Verificación parcial de fuentes reales

El 3 de noviembre se revalidaron en modo de solo lectura los enlaces de archivos, el protocolo anclado y el QA de las **975 corridas de Chico Norte** mediante el verificador existente de staging: **787 elegibles y 188 excluidas**. El SHA-256 de bytes de `experiment_source_manifest.json` fue `68bed0e283ddafbfada42743ae49f28d4611aa1585069a8d2f6b19815313e37b`, coincidente con el informe anterior de transferencia. Esta comprobación no recomputó las features ni volvió a extraer etiquetas, de manera que no demuestra por sí sola que la base esté lista para entrenamiento. No se verificó con ello la equivalencia entre otras formas que podrían ser físicamente relacionadas.

También se verificaron los enlaces y QA de las **1.000 corridas de Chico Sur**, con **866 elegibles y 134 excluidas**. Para las elegibles se reprodujeron desde los RPT las **138.560 etiquetas** y coincidieron con las almacenadas en su base supervisada. El SHA-256 de bytes del manifiesto supervisado fue `4bf0c50f61b2287ab8510369e46f26495393104d0fe670ac6ab444df28c055a6`. Esta comprobación no recalculó las features ni constituye una validación hidráulica o una evaluación del modelo. Ambas verificaciones reales precedieron a la mejora posterior que abre una transacción de lectura estable; no se afirma con ellas protección frente a modificaciones simultáneas durante aquella ejecución.

Se comprobó que ambos INP tienen hashes distintos y los manifiestos declaran el contrato `tabular_v4_21` con el mismo hash de descriptor vigente. Una planificación de identidades en lectura confirmó que ambas bases usan localmente `network_id=1`: el mapeo propuesto los distingue y asigna claves sin colisiones a 160 y 522 nodos, y a 1.000 y 975 escenarios/corridas, respectivamente. El mapeo se calculó en memoria; no se ha escrito la base conjunta. Diez pruebas enfocadas pasaron, incluidas colisiones, incompatibilidad del contrato y presencia indebida de etiquetas en el flujo label-free.

Una consulta adicional de integridad relacional en lectura encontró cero infracciones de clave foránea en ambas bases y confirmó que todas sus filas `node_features_v4_21` pertenecen a corridas marcadas elegibles: 138.560 en Chico Sur y 410.814 en Chico Norte. Esa consulta no compara los valores de las variables contra las fuentes; los recálculos completos se iniciaron por separado y deben terminar antes de afirmarlo.

La distribución QA por forma cambia la población utilizable para particionar: de las 39 formas comunes, **35 tienen al menos una corrida elegible en cada red**. Las formas `long_moderate`, `shape_2h` y `shape_5h` no tienen ninguna corrida elegible en ninguna de las dos redes; `shape_ultra_short_impulse` tiene 16 elegibles en Chico Sur y ninguna en Chico Norte. La forma `base`, exclusiva de Chico Sur, tiene cero elegibles. Estos conteos describen los lotes y la regla QA vigente, no una medida de representatividad física. Se corrigió la propuesta inicial de reservar formas sobre el total nominal de 39 y se consultó al investigador si reservar 7 de las 35 formas disponibles en ambas redes.

La reconstrucción de variables de Chico Sur finalizó: sus **138.560 filas elegibles** coincidieron con las calculadas a partir de los eventos y la red enlazados, con tolerancias relativas y absolutas de `1e-12` y valores ausentes equivalentes. Por separado, la extracción en memoria de los RPT elegibles de Chico Norte produjo **410.814 etiquetas**, omitió las **188 corridas excluidas** y generó el SHA-256 canónico `e52cd199f1a0de11e98461007ed3313e80bb895a851c7ed6b9863b0a0489529f`, igual al registrado en los dos scores externos congelados. No se escribieron etiquetas en la base original. Estos cotejos no sustituyen la validación hidráulica; el recálculo completo de variables de Chico Norte seguía en curso al redactar esta nota.
