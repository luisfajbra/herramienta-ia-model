# Integración de evaluación congelada al comando principal

**Alcance autorizado:** hacer accesible desde `main.py` la evaluación verificada de MLP y XGBoost frente a SWMM, con artefactos separados, mapas y paridades en el formato aprobado. Primera entrega: el diseño ya congelado de Chico Norte y sus diez escenarios; no se interpreta como selección genérica de cualquier red o hidrograma.

**Seguridad:** inspección sin simulación por defecto; ejecutar solo con bandera explícita y directorio nuevo; no entrenar, promover ni escribir en la base canónica. Verificar fuentes, predicciones y reportes con la infraestructura existente. Ante escenarios excluidos no fabricar gráficas: registrar e informar la limitación del generador actual.

1. Añadir pruebas del comando: ayuda, preinspección sin ejecución, ejecución explícita en ruta requerida, rechazo de modos incompatibles y rutas existentes.
2. Conectar `main.py` a una interfaz optativa que invoque la evaluación existente y genere las figuras aprobadas en directorios bajo una ejecución nueva. Conservar la trazabilidad y las salvaguardas de la implementación existente.
3. Comprobar con pruebas, preinspección y una ejecución real aislada; contrastar métricas, integridad, figuras y ausencia de escritura en la base canónica. Documentar comandos y alcance real.
4. Etapa posterior: seleccionar redes, hidrogramas y factores mediante un protocolo explícito, validando admisión hidráulica, compatibilidad de dominio y QA. Esta capacidad no se declara realizada por integrar el diseño congelado.

**Estado del 3 de noviembre de 2026:** primera entrega implementada y probada con diez simulaciones reales, candidatos congelados y 23 figuras. Ver [informe y límites](../reports/2026-11-03-integracion-evaluacion-congelada.md). La selección general y el manejo de figuras con escenarios excluidos permanecen pendientes.
