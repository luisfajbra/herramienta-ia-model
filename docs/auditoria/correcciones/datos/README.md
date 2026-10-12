# Datos de evaluación de las correcciones de la auditoría (M-01, M-02, M-03)

Estas tablas son la fuente de las cifras que cita el informe de la tesis sobre
la evaluación de XGBoost y de las referencias sin aprendizaje. Todo número
citado debe salir de aquí y poder rastrearse con `tables_provenance.json`.

## Qué se evaluó

- **Modelo:** XGBoost con los hiperparámetros **por defecto de `config.yaml`**
  (`bench.families.xgboost`: 200 árboles, profundidad 6, tasa 0,05,
  submuestreo 0,8, igual en el clasificador y en el regresor). **No** es el
  modelo ajustado con Optuna: estas tablas miden protocolos y métricas.
- **Referencias sin aprendizaje:** `baseline_interpolacion_factor`
  (interpola el volumen de la misma forma y nodo entre factores vecinos) y
  `baseline_forma_mas_cercana` (copia el volumen del mismo nodo y factor en la
  forma de entrenamiento más parecida por duración y tiempo al pico).
- **Datos:** contrato `tabular_v4_21` (21 features), PreparedDataset
  `ca706e27654e0cf5`. Son 154 080 filas: 963 de los 1000 runs (40 formas × 25
  factores) × 160 nodos. El filtro de continuidad (|error| ≤ 5 %) deja fuera
  37 celdas (forma, factor). Una fila está inundada si su volumen es ≥ 1 m³.

  > **Nota de corrección (2026-10-11, spike D-04):** el volumen es el del `.rpt`, con resolución de 1 m³. En la muestra re-simulada (una red, 13 escenarios con desborde), "≥ 1 m³" es consistente con un umbral efectivo de ≈ 0,5 m³ de volumen continuo, acotado a (0,4988; 0,5002] m³. Ver `../03-resolucion-de-la-etiqueta.md`.

- **Protocolos:**
  - `LOSO` y `GroupKFold5` agrupan por factor de intensidad, así que miden
    **interpolación entre factores**: todas las formas siguen en el
    entrenamiento. Con esta malla, `GroupKFold5` nunca retiene dos factores
    contiguos, y la referencia de interpolación da exactamente las mismas
    predicciones que en `LOSO`.
  - `LeaveOneShapeOut` retiene una forma de hidrograma completa: mide la
    generalización a **formas no vistas**. Es el protocolo principal.

## Archivos

Las métricas "agregadas" se calculan sobre todas las filas de test del
protocolo; "media por fold" es el promedio de los folds.

- `comparacion_protocolos.csv`: una fila por (modelo, protocolo). Columnas:
  - `modelo`, `protocolo`, `que_mide`;
  - `f1_agregado`, `recall_agregado`, `precision_agregada`;
  - `f1_media_por_fold_con_ceros`: la definición anterior, que cuenta como 0 los folds sin positivos;
  - `folds_f1_indefinido`;
  - `rmse_vol_agregado_m3`, `rmse_vol_media_por_fold_m3`: volumen enrutado por el clasificador, en todos los nodos;
  - `error_vol_total_pct`;
  - `nse_regresor_oracle`: el regresor evaluado solo en los nodos realmente inundados.
- `matriz_confusion.csv`: columnas `modelo`, `protocolo`, `tn`, `fp`, `fn`, `tp`. En cada fila, tn + fp + fn + tp = 154 080.
- `banda_limitrofe.csv`: columnas `modelo`, `protocolo`, `fn_total`, `fn_en_banda_1_2m3` (volumen real en [1, 2) m³), `fn_fuera_de_banda` y `fn_con_vol_ge_100m3` (volumen real ≥ 100 m³). Clasifica los falsos negativos por volumen real.
- `por_forma.csv`: una fila por (modelo, protocolo, forma). Columnas: `modelo`, `protocolo`, `forma`, `n_rows`, `n_positives`, `f1`, `rmse_vol`, `vol_real_m3`, `vol_pred_m3` y `error_vol_pct`.
- `tables_provenance.json`: la procedencia de los cuatro CSV. Registra:
  - el comando exacto, el commit de git y si el árbol versionado estaba sucio;
  - el SHA-256 de cada OOF leído y de cada CSV escrito;
  - n_rows, n_runs, la malla completa y las celdas faltantes.
- `runs/metrics_<modelo>.json`: las métricas completas que devuelve
  `score_predictions` para cada modelo o referencia. Incluyen `classifier`,
  `confusion_matrix`, `classifier_per_fold`,
  `classifier_legacy_mean_zero_fill`, `regressor_oracle`, `end_to_end`,
  `end_to_end_pooled`, `by_factor` y `by_shape`, separadas por protocolo.
- `runs/provenance.json`: la procedencia de la corrida. Registra:
  - prep_id, contrato, features, filas, umbral y filtro de continuidad;
  - los hiperparámetros, el commit, las versiones y los segundos de cada etapa;
  - las filas de respaldo de cada referencia (`baseline_fallback`).

## Cómo reproducir

```
# 1. Corrida completa (prepara el dataset, evalúa XGBoost y las referencias,
#    escribe los OOF en outputs/bench/oof/<prep_id>/ y runs/*.json). ~6 min.
venv/Scripts/python.exe scripts/ejecutar_evaluacion_v4_21.py

# 2. Tablas a partir de los OOF guardados (sin reentrenar). ~1,5 min.
venv/Scripts/python.exe scripts/informe_evaluacion.py --prep-id ca706e27654e0cf5 --prepared-dir outputs/bench/prepared_v4_21 --oof-root outputs/bench/oof --out docs/auditoria/correcciones/datos --flood-threshold 1.0
```

## Nota de trazabilidad

Los archivos de `runs/` (métricas y `provenance.json`) salieron de
`scripts/ejecutar_evaluacion_v4_21.py` en el commit `4a2a87c`. Esa corrida
también escribió los CSV. Después, los cuatro CSV se regeneraron con
`scripts/informe_evaluacion.py` a partir de los mismos OOF, sin reentrenar, en
el commit que registra `tables_provenance.json`. Esa versión ordena las filas
por (modelo, protocolo, forma) y escribe con fin de línea LF. Los valores son
idénticos a los de la corrida original: se comprobó con pandas, comparando
valor a valor.

La corrida del paso 1 también escribe los CSV, pero no
`tables_provenance.json`. Por eso, si se repite, hay que ejecutar después el
paso 2 para que las tablas y su procedencia vuelvan a coincidir.
