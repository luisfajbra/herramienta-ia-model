# Plan: búsqueda conjunta Optuna recuperable

**Objetivo:** ejecutar una búsqueda secuencial con almacenamiento persistente, sin acceder a las formas reservadas y con configuración verificable antes de reanudar. La selección usa `macro_network_rmse`, cinco grupos por forma y el presupuesto aprobado. No entrenar modelos reales hasta completar las pruebas y estimar el coste.

**Decisión:** el investigador aprobó guardar cada prueba y una regla reproducible de semilla por prueba, aunque la secuencia de candidatos cambie frente al flujo anterior. Se debe conservar intacta la API anterior.

## Componentes

- Nuevo módulo de búsqueda persistente: estudiar con SQLite de Optuna, candidato secuencial, semilla por número de trial y objetivo existente; bloqueo de un solo ejecutor liberado por el sistema al terminar el proceso.
- Contrato inmutable para reanudar: huellas de todos los componentes del conjunto preparado y de la partición, identidad de la familia, protocolo, versión de bibliotecas, espacio, objetivo, semilla, parámetros fijos y límite de pruebas. Rechazar una discrepancia antes de cualquier trial.
- Tratamiento de interrupción: registrar el estado interrumpido como fallido, sin considerar su métrica como completa; detenerse al alcanzar el presupuesto de intentos y reportar cuántos finalizaron. No afirmar que un estudio interrumpido reproduce una trayectoria continua sin interrupciones; la regla reproduce una trayectoria dada su historia registrada.
- Interfaz explícita que relea el conjunto validado, verifique la reserva y nunca ejecute entrenamiento con sus corridas. Separar resultados, logs y manifiestos de los estudios anteriores.

## Tareas

1. Pruebas que simulen dos ejecuciones, cambio de contrato, trial interrumpido y segundo ejecutor concurrente; demostrar el fallo antes de implementar cada control.
2. Implementar persistencia de un ejecutor, recuperación explícita y configuración inmutable. Verificar pruebas y que una interrupción no produzca un resultado completo falso.
3. Conectar únicamente el conjunto de desarrollo aprobado y el objetivo existente; comprobar en cada fold que las formas se agrupan y que las redes estén representadas. Registrar versiones y parámetros.
4. Correr un ensayo cronometrado de coste sobre desarrollo, claramente identificado y conservado; estimar el presupuesto sin evaluar la reserva. Acordar el inicio o continuar conforme a la autorización vigente, sin presentar el ensayo como resultado final.
5. Documentar y respaldar código, pruebas, manifiestos y estudio persistente. Entrenar y evaluar por separado; la prueba de formas reservadas se realiza solo una vez fijados los modelos y no demuestra transferencia a una red nueva.
