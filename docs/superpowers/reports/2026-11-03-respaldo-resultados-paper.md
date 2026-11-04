# Respaldo de resultados para el artículo

**Identidad:** `chico_sur_a_chico_norte_diez_escenarios_sinteticos_20261103`. Se trata de la evaluación de modelos entrenados con Chico Sur sobre dos hidrogramas sintéticos y cinco factores de Chico Norte; no se entrenó con Chico Norte. El [informe científico](2026-11-02-evaluacion-hidrogramas-nuevos-chico-norte.md) contiene métricas y límites.

## Ubicación y contenido

Se preparó una copia local en `outputs/respaldos/paper_chico_norte_20261103` y se creó una [carpeta privada de respaldo en Google Drive](https://drive.google.com/drive/folders/1chsofbBXPKxzUNbwYir5wmgw4K6tSk0G), con autorización expresa del investigador para subir los modelos, hidrogramas, simulaciones SWMM, predicciones, código y documentación. El repositorio GitHub configurado es público; esta copia no se subió allí.

| Archivo | Contenido | SHA-256 del ZIP completo |
|---|---|---|
| `01_evaluacion_original.zip` | Evaluación original, eventos, SWMM, predicciones y resultados. | `7e28fd92935efda5a5cac8f1364b0eb0906183f170e2843ccb0fa02179a72345` |
| `02_repeticion_verificada.zip` | Repetición realizada desde el comando principal. En Drive está dividida en cinco partes. | `942de75009740b30ef48c11a0040d44ad4d2e987d522ce260a1b1def79b1885f` |
| `03_modelos_y_politica.zip` | Candidatos, ensemble y registros de transferencia. En Drive está dividido en tres partes. | `036d82117722d6240e78c64eda35de14ad495b7d87d17bce75baaab54c6b5e1e` |
| `04_documentacion_codigo_entradas.zip` | Presentación, figuras, código, entradas y documentación disponibles al crear el paquete. | `092a698710b1bd04e54eea579e7e45f82f954f46d8b71cb07e4c464acb68fe23` |
| `05_estudios_seleccion_optuna.zip` | Estudios de selección de hiperparámetros. Subido tras autorización específica. | `75968f5ac78ad552fd4a68aa6f681be08ca91387b9e87b58769182737fac61f3` |
| `06_avance_seleccion_eventos.zip` | Respaldo incremental: código, documentación y dos pilotos de selección, con anclas y figuras. | `0ab7be57116abf9e084d8ed789f99ebabbb74ca89455fd8b806456a30d7f378e` |
| `07_qa_y_base_conjunta_avance.zip` | Respaldo incremental del diseño aprobado, pruebas QA y primeros verificadores; refleja el estado de los archivos al empaquetarlo. | `9f79a4c6f15909898716eee34d488f9f3f702aa852f69656bdc3f97fe34bd3d5` |
| `08_qa_y_base_conjunta_actualizado.zip` | Actualización del avance de pruebas y verificadores antes del mapeo de identidades. | `47bc680ebb14246c16975d385e7ea526922f96337c20872713ddcb93f5f587c4` |
| `09_qa_y_claves_multired.zip` | Avance que incorpora el mapeo de identidades en memoria. | `bb734a32b26cd90ab27402d3985fa0f5d46afd5bf07855774cf7c44650d707df` |
| `10_verificacion_features_en_curso.zip` | Código, pruebas y documentación cuando se iniciaron los recálculos de variables. | `79aff6489ceafa7baaf1e8e282bbbdf96463c4e75819120af3a1f18a18362991` |

También se subieron `MANIFIESTO_RESPALDO.json`, `PARTES_RECONSTRUCCION.json`, `LEEME_RESPALDO.md` y `RESTAURAR_RESPALDO.py`. Los ZIP tienen su propio manifiesto con ruta, tamaño y SHA-256 de cada archivo.

El paquete `06` tiene 86 archivos y se creó después de `01` a `05`; su hash consta en `MANIFIESTO_AVANCE_EVENTOS.json`, subido a la misma carpeta. Se verificaron localmente CRC, hashes por archivo y estabilidad de las fuentes durante la creación. Es un complemento y no está incluido en el programa de restauración inicial: debe descargarse y verificarse su SHA-256 por separado. Los documentos que lo integran reflejan el estado anterior a esta anotación.

El paquete incremental `07` y `MANIFIESTO_AVANCE_QA.json` se subieron a la misma carpeta privada. Se verificaron localmente los hashes por archivo y el CRC. No incluye automáticamente cambios realizados después de su creación ni una base conjunta. Debe descargarse y verificarse por separado; el SHA-256 de la descarga remota sigue sin comprobarse.

## Verificación y límites

Se comprobó localmente que cada fuente mantuviera su SHA-256 durante el empaquetado y que todos los ZIP pasaran la verificación CRC. El programa de restauración reconstruyó localmente los ZIP divididos y comprobó el SHA-256 de las partes y de los cinco ZIP completos. En Drive se verificaron nombres, tamaños y estado no compartido de los paquetes y partes subidos; también se solicitó la descarga del primer ZIP y el servicio devolvió una referencia de descarga completa. **No se ha verificado el SHA-256 de una descarga remota:** para esa comprobación el usuario debe descargar todos los archivos y ejecutar `RESTAURAR_RESPALDO.py` como indica el LEEME.

El paquete no contiene todo el histórico de simulaciones y entrenamiento de Chico Sur ni los `.out` completos de los experimentos de sensibilidad. No equivale a un respaldo integral del proyecto ni elimina la necesidad de otra copia. Los documentos y el código creados después del empaquetado no están incluidos automáticamente. Algunos manifiestos conservan rutas absolutas originales; restaurar los archivos no garantiza por sí solo su ejecución desde otro directorio.

La revisión automática rechazó inicialmente la transferencia a Drive por falta de un destino y contenido explícitos. Tras la autorización del investigador se subieron los paquetes anteriores. Luego rechazó por separado `05_estudios_seleccion_optuna.zip` porque los estudios de Optuna no se nombraron expresamente en esa autorización; el investigador autorizó incluirlos y se subió el archivo.
