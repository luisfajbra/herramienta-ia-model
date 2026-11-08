# Presentación: modelos conjuntos y escenarios nuevos

Fecha: 8 de noviembre de 2026. El investigador pidió actualizar la presentación semanal con los resultados de la evaluación de escenarios nuevos y documentar la procedencia. El archivo generado es `Presentacion_MLSMFP_08-11_modelos_conjuntos_escenarios_nuevos.pptx`; SHA-256 `c672ee7ee9adcb59b5f0905036d88868316d05822d7a9617d45d270a77aa5f82`.

## Procedencia y estructura

Se utilizó como base la copia `Presentacion_MLSMFP_08-11_entrenamiento_conjunto_revision_auditoria.pptx`, SHA-256 `cff5510b3543d01b69a8ce61dbfe31f706379b75d12817ef55d94bec6f43ced3`. La nueva copia conserva el formato institucional, portada, resumen y las 19 diapositivas relativas a los modelos conjuntos y su evaluación reservada. Se retiraron de esta copia las antiguas diapositivas 3–41, que presentaban la etapa con modelos entrenados solo con Chico Sur, porque el investigador pidió mostrar resultados de los modelos conjuntos. El archivo anterior permanece disponible y conserva ese historial. El resumen se actualizó para diferenciar la evaluación de escenarios nuevos de la reserva anterior.

La nueva presentación tiene **46 diapositivas**:

| Diapositivas | Contenido |
|---|---|
| 1–2 | Portada y resumen actualizado. |
| 3–21 | Entrenamiento conjunto, resultados de la reserva y revisión de fuentes previamente documentados. |
| 22–23 | Hidrogramas nuevos, uno por diapositiva. |
| 24–41 | Mapas individuales para cada combinación SWMM/MLP/XGBoost, dos hidrogramas y factores 1,9; 2,5 y 3,7. Los tres mapas de un mismo caso comparten escala y muestran solo la inundación sobre la red. |
| 42–43 | Paridad global de MLP y XGBoost, una por modelo. |
| 44 | Volumen total frente al factor para los dos hidrogramas. |
| 45–46 | Resultados globales, alcance y próximos pasos. |

## Cómo interpretar los resultados

Las dos formas y los factores 1,9; 2,1; 2,5; 2,9 y 3,7 fueron fijados antes de calcular las predicciones. Las diez simulaciones de Chico Norte pasaron el QA provisional. La muestra comprende 522 nodos por escenario, es decir 5.220 pares. Los resultados son frente a SWMM, no a observaciones de campo, y Chico Norte ya participó en el entrenamiento de los modelos conjuntos.

El RMSE mostrado es **global sobre los 5.220 pares**, no el promedio aritmético de los RMSE individuales de diez simulaciones. Sus valores son 27,02 m³ para MLP y 51,12 m³ para XGBoost. El porcentaje de detección es el recall calculado sobre los 1.766 pares donde SWMM reportó inundación ≥1 m³: 94,39 % para MLP y 98,19 % para XGBoost. El sesgo total de volumen se calcula comparando sumas de volumen: +0,03 % para MLP y +24,41 % para XGBoost. Un sesgo agregado cercano a cero puede ocultar errores que se compensan.

Los factores son nuevos y se encuentran dentro del intervalo de entrenamiento; evalúan interpolación de factor. En el hidrograma de ascenso escalonado, el tiempo al pico tributario es 9 h y superó el máximo de desarrollo por nodo para los 522 nodos en sus cinco factores. Esto demuestra extrapolación univariada para esa variable, no transferencia a una red desconocida. Las decisiones, las limitaciones hidráulicas, el detalle por factor y la procedencia completa se encuentran en el [protocolo](2026-11-08-protocolo-escenarios-nuevos-modelos-conjuntos.md) y el [informe de evaluación](2026-11-08-resultados-escenarios-nuevos-modelos-conjuntos.md).

## Reproducción y comprobaciones

Con el archivo de origen, las figuras y las salidas de la evaluación disponibles, ejecutar desde la raíz:

```powershell
venv/Scripts/python.exe scripts/build_weekly_2026_11_08_escenarios_nuevos.py
```

El script no sobrescribe una salida existente. Comprueba el hash de la presentación de origen, el sello de las métricas y los hashes de las figuras; verifica 46 diapositivas, CRC y ausencia de entradas ZIP duplicadas. Se compararon los bytes de las 23 imágenes insertadas en las diapositivas 22–44 con sus archivos de origen y se verificó su orden.

PowerPoint abrió la copia generada y se exportaron diapositivas representativas de resumen, hidrograma, mapa, paridad, resultados e interpretación para una revisión visual. Durante la generación se detectaron nombres internos de diapositivas duplicados al quitar la sección histórica; se renumeraron las partes antes de añadir diapositivas y se repitieron las comprobaciones. También se retiró un marcador de posición vacío que aparecía como borde en el resumen. La revisión visual fue selectiva, no una inspección individual de las 46 diapositivas.

## Respaldo

La [presentación final](https://docs.google.com/presentation/d/12uoquOS9K_vY8dweV-Wyvrn6Kp55I8F2/edit?usp=drivesdk&ouid=105819791257918831830&rtpof=true&sd=true), este informe y el [script de generación](https://drive.google.com/file/d/1icZccgkKpixtTIhxTAsDYJMJU2MFFenQ/view?usp=drivesdk) se copiaron a la carpeta privada de la tesis en Drive. Los metadatos de la presentación confirman 18.667.568 bytes, el mismo tamaño que el archivo local, y la carpeta prevista. La comparación de hash de una descarga remota queda pendiente; los metadatos por sí solos no la prueban.
