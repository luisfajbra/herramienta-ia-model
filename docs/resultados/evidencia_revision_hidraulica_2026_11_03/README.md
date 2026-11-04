# Evidencia de la revisión hidráulica de Chico Norte

El [informe de revisión](../../superpowers/reports/2026-11-03-revision-hidraulica-sensibilidad-chico-norte.md) interpreta estas tablas y figuras, describe las limitaciones y diferencia la QA provisional de la validación hidráulica.

## Tablas

- [Diagnósticos de los 985 RPT congelados](diagnosticos_reportes_congelados.csv): identidad de escenario, elegibilidad histórica, hash del RPT, continuidad, no convergencia, paso y FII.
- [Finales de los diez eventos nuevos](cierre_eventos_nuevos.csv): número de series y caudal terminal máximo, después de verificar cada hash de evento.
- [Cierre de los 75 escenarios históricos](cierre_lote_historico.csv): finales de CSV congelados y resultados correspondientes del lote exploratorio independiente.
- [Resumen de las seis sensibilidades](sensibilidad/sensibilidad_resumen.csv): continuidad, no convergencia, FII, inundación y diferencias por nodo frente a cada referencia.
- [Comparación de series](sensibilidad/sensibilidad_series_resumen.csv): diferencias para T463, T507, T508, T92, 523C y 396I en las fechas de reporte de un minuto.
- [QA de las seis variantes](sensibilidad/qa_sensibilidad.csv), [balance de volumen SWMM](sensibilidad/balance_sensibilidad.csv) y [cambios de INP verificados](sensibilidad/cambios_entrada_verificados.csv). La QA es diagnóstica: para InitFlow cero se verifica la variante contra sí misma, sin certificar el estado del INP original.
- [Extensión de factor bajo: resumen](factor_bajo/sensibilidad_resumen.csv), [QA](factor_bajo/qa_sensibilidad.csv), [balance](factor_bajo/balance_sensibilidad.csv) y [series](factor_bajo/sensibilidad_series_resumen.csv). Conserva el mismo método para pulsos dobles con factor 0,7.
- [Comparaciones de pasos internos](pasos_nativos/comparaciones_interpoladas_1s.csv): dos ventanas, seis elementos y dos variantes frente al control, interpoladas a segundos comunes sin extrapolación. Los datos nativos se preservan comprimidos por corrida en [datos de pasos internos](pasos_nativos/datos/).

## Figuras por elemento

Cada gráfica contiene las tres variantes de un único caso y un único elemento. Las líneas usan resultados OUT a intervalos exactos de un minuto; no permiten descartar fluctuaciones más rápidas.

| Elemento | Histórico crítico, factor 1,4 | Meseta nueva, factor 4,3 |
|---|---|---|
| Caudal T507 | [Ver](sensibilidad/graficas_sensibilidad/historico_no_convergencia_T507_flow.png) | [Ver](sensibilidad/graficas_sensibilidad/meseta_nueva_factor_4_3_T507_flow.png) |
| Caudal T508 | [Ver](sensibilidad/graficas_sensibilidad/historico_no_convergencia_T508_flow.png) | [Ver](sensibilidad/graficas_sensibilidad/meseta_nueva_factor_4_3_T508_flow.png) |
| Caudal T463 | [Ver](sensibilidad/graficas_sensibilidad/historico_no_convergencia_T463_flow.png) | [Ver](sensibilidad/graficas_sensibilidad/meseta_nueva_factor_4_3_T463_flow.png) |
| Caudal T92 | [Ver](sensibilidad/graficas_sensibilidad/historico_no_convergencia_T92_flow.png) | [Ver](sensibilidad/graficas_sensibilidad/meseta_nueva_factor_4_3_T92_flow.png) |
| Profundidad outfall 523C | [Ver](sensibilidad/graficas_sensibilidad/historico_no_convergencia_523C_depth.png) | [Ver](sensibilidad/graficas_sensibilidad/meseta_nueva_factor_4_3_523C_depth.png) |
| Profundidad nodo 396I | [Ver](sensibilidad/graficas_sensibilidad/historico_no_convergencia_396I_depth.png) | [Ver](sensibilidad/graficas_sensibilidad/meseta_nueva_factor_4_3_396I_depth.png) |

Las seis gráficas del factor bajo están en [figuras de pulsos con factor 0,7](factor_bajo/graficas_sensibilidad/); por ejemplo, [T508](factor_bajo/graficas_sensibilidad/pulsos_nuevos_factor_0_7_T508_flow.png) y [523C](factor_bajo/graficas_sensibilidad/pulsos_nuevos_factor_0_7_523C_depth.png).

La inspección de pasos internos tiene 24 gráficas separadas por caso, ventana y elemento. Como ejemplos: [T508 durante la ventana crítica de la meseta](pasos_nativos/graficas/meseta_nueva_factor_4_3_ventana_critica_T508.png), [396I en la misma ventana](pasos_nativos/graficas/meseta_nueva_factor_4_3_ventana_critica_396I.png) y [T508 al inicio](pasos_nativos/graficas/meseta_nueva_factor_4_3_inicio_T508.png). Las líneas unen las salidas de pasos internos irregulares; la tabla de diferencias interpola sobre una malla común de un segundo.

## Artefactos y resguardo

La carpeta [artefactos de sensibilidad](sensibilidad/artefactos_sensibilidad/) contiene el diseño, los resultados, los seis INP, sus seis RPT, las tablas nodales de inundación y las series seleccionadas de un minuto. [El manifiesto de sensibilidad](sensibilidad/manifiesto_sensibilidad.json) registra hashes de los archivos generados hasta su creación. [El manifiesto de evidencia](manifest.json) registra hashes de esta carpeta completa y de los scripts utilizados.

Los seis OUT completos —aproximadamente 178,3 MiB— permanecen en `outputs/experiments/chico_norte_hydraulic_sensitivity_20261103T130312Z/`. **`outputs/` no se guarda mediante Git**: respaldar esa carpeta y las fuentes congeladas por separado si se quiere conservar el conjunto completo. Las figuras y tablas son resultados de diagnóstico, no datos incorporados al entrenamiento.

Los tres OUT completos del factor bajo están en `outputs/experiments/chico_norte_hydraulic_sensitivity_20261103T132030Z/`; [sus INP, RPT y series seleccionadas](factor_bajo/artefactos_sensibilidad/) se conservan en esta carpeta de evidencia.

Las seis repeticiones de pasos internos están en `outputs/experiments/chico_norte_native_windows_20261103T132556Z/` y sus OUT suman otros **178,3 MiB**. El intento previo, interrumpido por redondeo de marcas de tiempo, se encuentra en `chico_norte_native_windows_20261103T132158Z/` y no se usa en las tablas finales. En esta evidencia se conservan los seis RPT y los registros comprimidos de las dos ventanas.

## Reproducción

Desde la raíz del repositorio, con las fuentes congeladas y sus manifiestos disponibles y la misma versión del motor y dependencias indicadas en el informe:

1. Ejecutar `python -m scripts.review_chico_norte_hydraulics` para crear un nuevo directorio de experimento y las seis simulaciones. Nunca sobrescribe una ejecución previa.
2. Ejecutar `python -m scripts.summarize_chico_norte_hydraulic_sensitivity <directorio-del-experimento> <directorio-nuevo-de-evidencia>` para verificar hashes y regenerar las series, tablas y figuras de sensibilidad.
3. Ejecutar `python -m scripts.inventory_chico_norte_hydraulic_reports <directorio-nuevo-de-inventario> <directorio-del-experimento>` para reconstruir el inventario de RPT, los cierres, la QA, el balance y la comparación de INP. El inventario comprobado para esta revisión se encuentra bajo `outputs/experiments/chico_norte_hydraulic_inventory_check_20261103_v3/`.

Para repetir la extensión de factor bajo, usar `python -m scripts.run_chico_norte_low_flow_sensitivity` y aplicar los pasos 2 y 3 al nuevo directorio generado.

Para repetir los pasos internos de las seis variantes originales, ejecutar `python -m scripts.review_chico_norte_native_steps`, seguido por `python -m scripts.summarize_chico_norte_native_steps <directorio-nuevo-de-pasos> <directorio-nuevo-de-evidencia>`. El primero utiliza los INP verificados de `chico_norte_hydraulic_sensitivity_20261103T130312Z` y compara sus RPT y volúmenes nodales con cada control. Los tiempos se obtienen de los días decimales devueltos por `swmm_step`, no de la fecha redondeada de `sim.current_time`.

En Windows se usó `venv\Scripts\python.exe`. Las rutas y hashes de los INP, eventos, RPT y OUT fuente están en `sensibilidad/artefactos_sensibilidad/design.json`; los hashes de los seis juegos de salida están en `results.json`. Los RPT registran valores redondeados y sus tablas nodales dan volúmenes con resolución de 1 m³. Los resultados de control coincidieron con sus RPT congelados en indicadores y volúmenes nodales a esa resolución.
