# Respaldo y presentación tras la auditoría

Fecha: 8 de noviembre de 2026. El investigador autorizó respaldar el paquete y corregir la conclusión de la presentación tras la revisión independiente.

## Respaldo

Se subieron el [ZIP de la reserva y base conjunta](https://drive.google.com/file/d/18OzbKH16A_BoBPRg44DgmO8Mx-99GJlb/view?usp=drivesdk) y su [manifiesto](https://drive.google.com/file/d/1MFH5oYRelqSxAjtOauR33g25aK4-gqV6/view?usp=drivesdk) a la [carpeta privada de la tesis](https://drive.google.com/drive/folders/1chsofbBXPKxzUNbwYir5wmgw4K6tSk0G). Los metadatos de Drive confirman los nombres, la carpeta, el estado no compartido y los tamaños de 100.110.651 y 4.225 bytes, respectivamente. El conector permitió solicitar ambas descargas y devolvió referencias de archivo con esos tamaños.

El ZIP local se verificó con su manifiesto, incluidos los hashes de sus 21 miembros. Su SHA-256 local es `2f1ea8857d45e5fcefa6eb2faad435b35ffb2f869f8168ea163fe78b049b4022`; el SHA-256 del manifiesto local es `85879fb896029fbb81a472d8ba8409e627055eebad5f1631bac9744f08e81a49`. **La comparación criptográfica de la copia descargada de Drive queda pendiente**: los metadatos normalizados del conector no devolvieron los hashes y el intento de traer al sistema local la referencia de descarga falló por conexión. La presencia y el tamaño remotos no prueban identidad byte por byte.

El ZIP no contiene los modelos ni los estudios Optuna. Se confirmó en el listado de la misma carpeta la presencia de los paquetes `27`, `29` y las partes y manifiesto `32`; no se realizó una restauración ni se cotejaron sus hashes remotos con los sellos. Se observó también el paquete `36_evaluacion_reserva_conjunta_20261108.zip`, sin comparar su contenido con el paquete 33.

## Presentación

Se creó `Presentacion_MLSMFP_08-11_entrenamiento_conjunto_revision_auditoria.pptx` como copia de la presentación de 60 diapositivas. En la diapositiva 56 se conservó el resultado de menor RMSE del MLP en la reserva completa y se aclaró que, al excluir un hidrograma sostenido, XGBoost tiene menor RMSE en un análisis posterior. No se alteraron las métricas ni se usó ese análisis para seleccionar un modelo. La copia anterior se conserva.

La copia nueva se abrió con `python-pptx`, se comprobó que contiene 60 diapositivas y que el ZIP del PPTX no tiene entradas duplicadas ni errores CRC, y se volvió a leer el texto de la diapositiva 56. SHA-256 local: `cff5510b3543d01b69a8ce61dbfe31f706379b75d12817ef55d94bec6f43ced3`. Se subió una [copia a Drive](https://docs.google.com/presentation/d/1U_AFbJOBKXvsa8AtZAjJD3-WCyYa5iV1/edit?usp=drivesdk&ouid=105819791257918831830&rtpof=true&sd=true); sus metadatos indican 23.323.768 bytes y la carpeta privada. No se completó una inspección visual renderizada de la diapositiva ni una comprobación del hash remoto.

Para reproducir la copia, con la presentación original presente y la salida aún inexistente: `venv/Scripts/python.exe scripts/corregir_conclusion_presentacion_20261108.py`. El script comprueba la presentación de entrada y no sobrescribe una copia existente.
