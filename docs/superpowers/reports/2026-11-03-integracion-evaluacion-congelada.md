# Integración de la evaluación congelada de Chico Norte

**Fecha:** 3 de noviembre de 2026. **Objetivo:** permitir ejecutar desde `main.py` la evaluación documentada de candidatos entrenados con Chico Sur sobre los diez escenarios sintéticos de Chico Norte y generar las figuras aprobadas. La integración es optativa, no cambia el comando predeterminado ni reentrena modelos.

## Comandos y alcance

Desde la raíz del repositorio, inspeccionar los insumos sin ejecutar SWMM:

```powershell
.\venv\Scripts\python.exe main.py --evaluate-frozen-chico-norte
```

Para ejecutar, indicar una carpeta que no exista:

```powershell
.\venv\Scripts\python.exe main.py --evaluate-frozen-chico-norte --execute-evaluation --evaluation-output outputs/experiments/MI_EVALUACION_NUEVA
```

Se usan exclusivamente la red Chico Norte, los dos hidrogramas y cinco factores del estudio de la presentación, el XGBoost congelado y el ensemble MLP de cinco semillas. No se puede seleccionar otra red, otros hidrogramas ni otros factores con esta opción; esa ampliación es trabajo pendiente y requiere verificar contratos, política de dominio y admisión. Los factores están dentro del rango del entrenamiento, de modo que no representan extrapolación en factor.

La ejecución crea `evaluation/` con SWMM, una base temporal, fuentes, predicciones y métricas; `figuras_base/` con las figuras generadas por el evaluador anterior; y `figuras/` con las figuras aprobadas: hidrogramas separados, mapas de inundación independientes para SWMM, MLP y XGBoost, paridad global por modelo y la curva por factor. Los mapas de `figuras/` no incluyen símbolos de errores de clasificación. Las escalas de color son independientes y deben leerse en cada barra. El archivo `estado_integracion.json` registra si la integración terminó o falló.

El flujo conserva la QA provisional vigente (cobertura finita, continuidad absoluta ≤5 % y drenaje); FII y no convergencia no son exclusiones automáticas. El generador de figuras de presentación actual exige que los diez escenarios estén completos y sean elegibles. Si alguno se excluye, el experimento puede contener la evaluación y QA disponibles, pero la integración marca error al generar ese conjunto de figuras; no se presenta un conjunto parcial como completo. Preparar visualizaciones para poblaciones parcialmente excluidas queda pendiente.

## Prueba real

Se ejecutó el comando integrado en una carpeta nueva, `outputs/experiments/main_frozen_chico_norte_20261103`, sin sustituir el experimento anterior. Los **10 escenarios** terminaron elegibles bajo el QA provisional. La base temporal registró **0 filas en `node_results`** después de evaluar; no se entrenó ni promovió ningún modelo ni se ingresaron etiquetas a la base canónica.

Las métricas del conjunto coincidieron con la [evaluación original](2026-11-02-evaluacion-hidrogramas-nuevos-chico-norte.md) para RMSE, MAE, recall, precisión, F1, sesgo porcentual y matriz de confusión, con tolerancia numérica de `1e-9` absoluta y `1e-10` relativa. Los valores de RMSE agrupado fueron **186,26 m³ para MLP** y **186,18 m³ para XGBoost**, sobre 5.220 pares nodo–escenario. Estos valores describen únicamente ese conjunto y no son porcentajes de confiabilidad.

La carpeta `figuras/` contiene **23 imágenes**, entre ellas 18 mapas independientes; se comprobaron sus SHA-256 contra `presentation_figures_manifest.json`. Los sellos de `evaluation_completion.json`, resultados, diseño y hashes de los hidrogramas también se verificaron. La comprobación adicional de esos sellos antes y después de crear las figuras se añadió al comando mientras transcurría la prueba real; por ello **en esa ejecución concreta se aplicó y confirmó por separado después de terminar**, y las pruebas del código cubren su ejecución automática en futuras corridas. Esta precisión evita presentar la prueba como si ya hubiera ejecutado esa salvaguarda en línea.

## Pruebas y límites

Las 10 pruebas del nuevo comando y flujo comprobaron inspección sin simulación, bandera explícita, exclusión de otros modos, rechazo de carpetas existentes, estado ante fallo y detección de cambios de hidrograma. También pasaron **37 pruebas** de comandos existentes con un fixture temporal dentro del proyecto. La ejecución estándar de esas pruebas falló antes por permisos `WinError 5` del directorio temporal de Windows; la repetición usó temporalmente un directorio controlado del proyecto sin modificar el código de producción. Se ejecutaron además las tres pruebas del diseño de los hidrogramas.

La integración no incorpora Chico Norte al entrenamiento, no cambia la QA, no resuelve la condición inicial de la red y no demuestra generalización a una tercera red. Los `.out`, las bases y las figuras de la prueba real están en `outputs/`, ignorado por Git. Los hashes permiten detectar alteraciones, pero no sustituyen un respaldo externo.

**Siguiente etapa:** diseñar la selección explícita de red, hidrogramas y factores, con verificación de identidad, parámetros hidráulicos, alcance de dominio, exclusiones y figuras de escenarios elegibles. Posteriormente, una vez admitida Chico Norte para ese uso, seguir el plan del modelo compartido y su evaluación por eventos. La tercera red debe mantenerse independiente para una prueba futura de transferencia entre redes.
