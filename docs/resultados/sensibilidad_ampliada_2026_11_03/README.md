# Evidencia: sensibilidad hidráulica ampliada de Chico Norte

El [informe](../../superpowers/reports/2026-11-03-sensibilidad-hidraulica-ampliada.md) explica resultados y limitaciones. El [protocolo](../../superpowers/reports/2026-11-03-protocolo-sensibilidad-ampliada.md) identifica la selección de casos y variantes.

- `comparacion_diez_escenarios.csv`: treinta variantes y diferencias contra la referencia.
- `cambios_clase_por_nodo.csv`: pares nodo–escenario que cruzan el umbral de 1 m³.
- Seis PNG: volumen nodal total, pasos sin convergencia y continuidad para cada hidrograma.
- `qa/sensibilidad/`: QA, balance y campos de INP verificados en las 24 variantes nuevas.
- `variantes/`: INP, RPT, volúmenes nodales, series de seis ubicaciones a intervalos de un minuto y figuras por caso; incluye su manifiesto de hashes.
- `manifest.json`: hashes de la tabla y figuras agregadas, fuente de los diez casos y tolerancia numérica usada. `manifest_completo.json` inventaría la evidencia de esta carpeta y los códigos de método; no se incluye a sí mismo.

Se conservan las tres configuraciones de cada uno de los dos casos previos en `outputs/experiments/chico_norte_hydraulic_sensitivity_20261103T130312Z` y `outputs/experiments/chico_norte_hydraulic_sensitivity_20261103T132030Z`. Las 24 nuevas están en `outputs/experiments/chico_norte_hydraulic_sensitivity_20261103T134733Z`. Los OUT completos se encuentran en esos directorios y están ignorados por Git; guardar solo esta carpeta de evidencia no los respalda. Los archivos históricos fuente también deben conservarse para revalidar sus hashes.

Para reproducir la tabla consolidada en un directorio nuevo, desde la raíz del proyecto:

```powershell
.\venv\Scripts\python.exe -m scripts.summarize_chico_norte_extended_sensitivity SALIDA_NUEVA outputs/experiments/chico_norte_hydraulic_sensitivity_20261103T130312Z outputs/experiments/chico_norte_hydraulic_sensitivity_20261103T132030Z outputs/experiments/chico_norte_hydraulic_sensitivity_20261103T134733Z
```

El comando rechaza rutas ya existentes, casos faltantes, fuentes modificadas o referencias que no coincidan con los RPT congelados dentro de la tolerancia numérica documentada. Las salidas son exploratorias y no se han incorporado a entrenamiento.
