# Paridad global para la presentación del 30 de octubre

La presentación usa una única gráfica de paridad global, con XGBoost en azul y el ensemble MLP de cinco semillas en naranja. Incluye 410.814 pares corrida–nodo de 787 corridas elegibles. Se reportan RMSE, MAE, R² descriptivo y `n`; cada punto compara el volumen predicho con el volumen de SWMM. La línea `y = x` representa igualdad entre ambos.

Se retiró la anotación «Ambos cero». Su significado era la proporción de pares en que tanto SWMM como el modelo daban un volumen de 0 m³. Aunque describe coincidencias en nodos secos, no es una métrica de error y podía confundirse con confiabilidad. También se eliminó la lámina condicional de nodos inundados para mantener una sola comparación global y reducir información secundaria.

La figura usa los datos verificados del score externo congelado; este cambio es solo de presentación y no modifica modelos, predicciones, etiquetas, candidatos ni métricas. El manifiesto registra el hash de la figura y del archivo `.pptx`.

Para regenerar desde la raíz del repositorio:

```powershell
.\venv\Scripts\python.exe scripts/build_2026_10_30_dark_format.py
```
