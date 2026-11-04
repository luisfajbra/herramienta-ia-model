# Plan de implementación: base conjunta verificable

> Para ejecutar: usar `superpowers:executing-plans` y pruebas de regresión por tarea.

**Objetivo:** crear una fuente supervisada nueva, íntegra y auditable para desarrollo con los lotes elegibles de Chico Sur y Chico Norte.

**Diseño aprobado:** [base conjunta verificable](../specs/2026-11-03-base-conjunta-verificable-design.md). **Tecnología:** Python, SQLite, pytest, parser SWMM y contrato `tabular_v4_21` ya existentes.

**Restricciones:** originales de solo lectura; sin sobrescribir ni promover modelos; conservar los diez escenarios de la presentación separados; QA existente y etiquetas desde RPT enlazado; sin inferir objetivos ausentes ni inventar manifiestos. Las formas compartidas y sus factores deben agruparse; las particiones concretas y presupuestos se acuerdan antes de entrenar.

## 1. Contrato de procedencia y preflight

- [ ] Definir entrada explícita de ambas bases, manifiestos, protocolos, anclas y destino nuevo; distinguir identidad de red por INP y claves originales de las nuevas.
- [ ] Añadir pruebas sintéticas que detecten anclas o archivos alterados, identidad duplicada, contratos incompatibles, destino existente y selección sin elegibles; verificar fallos antes de cambiar código.
- [ ] Implementar preflight de solo lectura que inspeccione sellos y fuentes por corrida y genere dictamen descriptivo; no convertir una huella en firma de autenticidad.
- [ ] Probar que una falla no modifica originales ni publica un resultado completo.

## 2. Materialización de etiquetas y claves

- [ ] Probar dos redes sintéticas con IDs locales y nodos coincidentes, QA mixto y etiquetas vinculadas al RPT; verificar exclusiones, cobertura, remapeo y recuentos.
- [ ] Escribir base nueva con las dos redes, su inventario, escenarios, dictámenes, fuentes y features; extraer Chico Norte solo de RPT elegibles verificados. Conservar mapeo auditable de IDs y SHA de cada fuente.
- [ ] Comparar las etiquetas de Chico Norte con el sello y los recuentos de la evaluación ya congelada; confirmar cobertura y resultados de Chico Sur. Al fallar, no publicar la salida como completa.

## 3. Compatibilidad con preparación y particiones

- [ ] Resolver por prueba la discrepancia de procedencia: el preparador multired `swmm_resilience/ml/bench/preprocess.py::_source_provenance` exige metadatos de datasets y eventos del backfill, mientras los escenarios congelados contienen otros campos y vínculos por corrida. No fabricar metadatos de datasets inexistentes.
- [ ] Extender el preparador para validar explícitamente la procedencia verificada de la nueva base, con pruebas de alteración, sin debilitar las exigencias de otras fuentes. Preparar ambas redes y verificar contrato, filas y exclusiones.
- [ ] Validar identidad de formas con los hashes de ancla y evitar cruces entre una forma y todos sus factores y redes. Reportar duplicados o equivalencias no resueltas; no fijar particiones concretas sin acuerdo.

## 4. Validación y documentación

- [ ] Ejecutar integración sintética completa: fuente, QA, etiquetas, base nueva y preparador. Probar también ausencia de elegibles, caída intermedia y origen alterado.
- [ ] Ejecutar preflight real en lectura; solo materializar el lote aprobado cuando todas las barreras y pruebas pasen. Registrar duración, recuentos, hashes y advertencias.
- [ ] Actualizar informes y decisiones; conservar resultados originales, demostrar que fuentes no fueron alteradas y respaldar los artefactos nuevos conforme a la autorización existente.
- [ ] Antes de buscar hiperparámetros o entrenar, consultar o confirmar las decisiones científicas pendientes: particiones concretas, presupuesto y criterios de aceptación.

**Hallazgo de planificación:** los `config_json` de las bases congeladas no incluyen el `source_manifest` esperado por el preparador multirred actual. La prueba con datos sintéticos de backfill no demuestra que las bases reales sean compatibles. La tarea 3 es necesaria antes de declarar que la base está lista para entrenar.

**Avance inicial, 3 de noviembre:** se implementaron verificadores parciales de solo lectura que revalidan fuentes, ancla y QA de la base sin etiquetas, y que además comparan etiquetas almacenadas de la base supervisada con sus RPT. Dos pruebas sintéticas verifican la detección de alteraciones; las pruebas enfocadas pasaron. Estos verificadores no recomputan features, no verifican todavía ambos orígenes como una pareja y no construyen la base conjunta. La comprobación parcial real de Chico Norte terminó para 975 corridas: 787 elegibles y 188 excluidas. En Chico Sur confirmó 866 elegibles, 134 excluidas y coincidencia de 138.560 etiquetas con sus RPT. Los límites y hashes están en el [informe](../reports/2026-11-03-preparacion-entrenamiento-conjunto.md).

**Avance posterior:** se añadió un verificador de variables de Chico Norte que reconstruye por corrida las features del contrato desde los archivos enlazados. Las pruebas sintéticas detectan una feature alterada y una fila huérfana. Se inició su ejecución sobre la fuente real; no se debe afirmar que verificó el conjunto completo hasta que termine. La comprobación comenzó con una versión previa al control adicional de filas huérfanas, que deberá verificarse aparte o mediante una nueva ejecución. La verificación equivalente de features de Chico Sur y la materialización conjunta siguen pendientes.

**Siguiente actualización:** se implementó y probó también el verificador de variables de Chico Sur; la ejecución real de ambas comprobaciones está en curso. La regresión focalizada de identidad, contratos, QA, etiquetas y features terminó con **13 pruebas aprobadas**. La base conjunta todavía no existe, y continúan pendientes la materialización, su integración con el preparador y las decisiones de partición y presupuesto.

**Extracción de objetivos:** se añadió una función que lee en memoria los reportes enlazados de Chico Norte, comprueba su hash y cobertura, omite y enumera las corridas no elegibles y no modifica la base. Su prueba con QA mixto pasó y una regresión ampliada terminó con **14 pruebas aprobadas**. Se inició la comparación de la huella de etiquetas reales con la evaluación externa congelada; aún no se ha confirmado el resultado ni se ha persistido una base supervisada de Chico Norte.

**Cierre parcial de comprobaciones reales:** coincidieron las 138.560 filas de variables reconstruidas de Chico Sur y la huella de las 410.814 etiquetas elegibles de Chico Norte con el score congelado. La reconstrucción de variables de Chico Norte sigue en curso. La salida supervisada conjunta, su compatibilidad con el preparador y las particiones siguen pendientes; no hubo entrenamiento.
