# Regenerar el conjunto supervisado A desde un manifiesto congelado

`stage_frozen_a_supervised` recibe una base SQLite migrada, vacía y exclusiva;
un protocolo v2 anclado antes de simular; el manifiesto completo con el orden
esperado de escenarios; y la ruta al `config.yaml` cuya clave
`dataset.flood_threshold_m3` gobierna la etiqueta de inundación.
El loader oficial de configuración fija la red A: la ruta y SHA-256 del INP
protocolizado deben coincidir exactamente con `config.network.inp_path`, el
nombre/rol sellado debe ser `A`, y las claves deben coincidir en orden con las
25 corridas base más las 39 formas por los mismos 25 factores aprobados.

El stager v2 compartido verifica la red y su hash, el ancla externa, el conjunto
y orden de claves, los cinco roles de artefactos por corrida, sus rutas y hashes,
la correspondencia exacta INP efectivo/evento y la calidad hidráulica recalculada
desde INP/RPT/OUT. Registra corrida, escenario, calidad y vínculos de fuentes
para todas las corridas, incluidas las excluidas por QA. Solo guarda features
v4_21 y extrae etiquetas del RPT de la misma corrida cuando QA la declara elegible.
Volumen se guarda en m³; `inunda` usa el umbral del config. No se fabrican
`peak_flooding_lps`, duración ni profundidades, porque el manifiesto no los aporta.

## Recuperación ante interrupciones

El registro de fuentes/calidad/features y las etiquetas se confirman en dos fases.
El archivo `a_training_source_manifest.json` se publica de forma exclusiva
dentro de la transacción que inserta las etiquetas y antes de confirmar esa
transacción. Si falla la escritura, la transacción revierte todas las etiquetas.
Si el proceso cae después de publicar el archivo pero antes del `COMMIT`, puede
quedar un manifiesto huérfano junto a una base sin etiquetas; la validación de
cardinalidad del loader rechaza esa base porque las corridas elegibles no tienen
resultados. Tanto esa base como cualquier manifiesto huérfano se conservan para
diagnóstico; no se reintenta sobre ellos ni se añaden filas a mano. La
recuperación requiere un directorio de experimento y una base SQLite nuevos y
vacíos.

Las corridas `unverified`/inelegibles permanecen en el registro de calidad y
artefactos, llevan `run_id` y `reason_codes` en el manifiesto final y no tienen
features ni filas de `node_results`. Antes de entrenar, el loader debe limitarse
explícitamente a `eligible_run_ids` del manifiesto A verificado.
