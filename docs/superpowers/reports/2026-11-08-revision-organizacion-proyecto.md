# Revisión de organización del proyecto

Fecha: 8 de noviembre de 2026. Alcance: inventario y recomendaciones, sin mover ni eliminar archivos. La revisión se hizo sobre `fix/test-suite-contracts-2026-10-27`, después de respaldar el commit `73604979` en GitHub.

## Inventario y límites

- Git rastrea 1.122 archivos: 574 bajo `docs/`, 173 bajo `tests/`, 138 bajo `swmm_resilience/`, 60 bajo `data/`, 59 bajo `scripts/` y 62 en la raíz. Las demás rutas completan el total.
- Hay 33 presentaciones rastreadas en la raíz y 54 scripts Python rastreados bajo `scripts/`. Algunas presentaciones son versiones de trabajo y otras son evidencia histórica: no se determinó su equivalencia ni se propone borrar ninguna.
- Se identificaron 180 carpetas locales en la raíz con nombres `.pytest-*` y nueve con nombres que comienzan por `tmp` o `.tmp`. Son candidatas a inspección como temporales, no se comprobó su contenido; algunas producen `Permission denied` y no se puede asegurar que sean prescindibles.
- Git ignora gran parte de las bases, salidas, modelos y artefactos grandes. Algunos archivos bajo `outputs/` y `validation_output/` sí están rastreados por excepciones o por haber sido añadidos previamente. Que algo esté ignorado no demuestra que sea innecesario ni que esté respaldado. La presentación terminada en `.building.pptx` permanece local y sin rastrear como probable archivo de construcción, sin verificar su contenido.
- Estas cifras describen el estado del repositorio en la fecha indicada. No constituyen una revisión exhaustiva del contenido de todos los archivos ni una certificación de que algún archivo esté obsoleto.

## Riesgos principales

1. **Confusión entre flujos.** `main.py` y sus modos históricos no son equivalentes a la evaluación reciente de los modelos conjuntos. La [guía integral](../../GUIA_INTEGRAL_PROYECTO.md) distingue el contrato histórico de 17 variables, el de 21 variables y los comandos de cada etapa. Hay que mantener visible esta distinción al reorganizar.
2. **Reproducibilidad de Optuna.** `scripts/tune_joint_checkpointed.py` calcula SHA-256 de los archivos de entrada, de scripts concretos y de los Python bajo `swmm_resilience/`, con rutas absolutas como claves. Mover, agregar o cambiar archivos puede alterar el contrato usado para verificar o reanudar un estudio. Se debe conservar el commit histórico y probar cualquier migración sobre copias; no modificar hashes históricos para forzar compatibilidad.
3. **Dependencias por rutas.** Existen scripts fechados, presentaciones y manifiestos con rutas a entradas o salidas concretas. Una búsqueda textual ayuda a encontrar referencias, pero no demuestra por sí sola que sea seguro mover o borrar un archivo. Los resultados antiguos siguen siendo evidencia científica.
4. **Respaldo incompleto si solo se mira GitHub.** Los archivos pesados excluidos de Git requieren su respaldo externo. Antes de cualquier limpieza, cotejar un inventario de rutas, tamaños y hashes con los paquetes de Drive y comprobar su restauración; no dar por restaurables archivos solo porque se haya completado una subida.
5. **Archivos temporales inaccesibles.** Las denegaciones de permisos impiden inspeccionar todo el contenido local. Evitar borrados recursivos hasta identificar propiedad, contenido y posible uso; no cambiar permisos masivamente como primera medida.

## Organización propuesta

La estructura es orientativa; se implementaría gradualmente y con pruebas, no como un traslado masivo:

```text
README.md                       entrada breve y actual
swmm_resilience/                biblioteca y contratos del programa
scripts/                        puntos de entrada, organizados por función
  training/                     preparación, búsqueda y entrenamiento
  evaluation/                   evaluación y diagnóstico
  visualization/                mapas, paridades y presentaciones
  historical/                   scripts de campañas cerradas identificadas
data/                           redes y entradas versionadas con procedencia
docs/
  guia/                         guía de uso y arquitectura vigente
  investigacion/                metodología, decisiones y resultados
  presentaciones/               entregas por fecha e índice de versión
  historico/                    documentación anterior identificada
tests/                          pruebas de contratos y flujos
outputs/                        salidas locales; política de respaldo explícita
```

Las carpetas propuestas no son todavía rutas reales. Los nombres y límites se ajustarán tras un inventario de dependencias y una comprobación de los comandos usados. Conservar en Git los datos livianos necesarios y en almacenamiento externo los grandes, con manifiestos y referencias estables.

## Secuencia recomendada

1. **Inventario y clasificación:** listar cada script y documento con finalidad, fecha o campaña, entradas, salidas, referencias y estado propuesto: vigente, histórico, generado o pendiente de aclarar. Separar las presentaciones definitivas de sus borradores sin destruir versiones.
2. **Respaldo verificable:** contrastar las fuentes y artefactos locales con los respaldos, registrar hashes y probar la restauración de una muestra representativa; identificar expresamente lo que falte. La rama respaldada permanece como referencia.
3. **Definir interfaz vigente:** documentar de forma corta cómo ejecutar cada flujo y qué modelos usa. Los modelos conjuntos requieren una integración específica antes de atribuirlos al comando histórico.
4. **Ordenar documentación y presentaciones:** crear índices y rutas estables, actualizar enlaces y generadores, y comprobar apertura de las presentaciones. Mantener trazabilidad de versiones y evitar duplicaciones innecesarias en futuras entregas.
5. **Clasificar scripts antes de moverlos:** revisar imports, rutas, comandos, manifiestos y pruebas. Preservar los scripts de campañas selladas; para los flujos vigentes, usar migraciones con pruebas y, cuando corresponda, compatibilidad para rutas antiguas.
6. **Temporales y generados:** inspeccionar en una fase aparte, revisar permisos y respaldo; solo retirar los identificados con seguridad como regenerables y sin procesos activos.
7. **Validar:** comprobar enlaces, comandos y pruebas pertinentes, verificar hashes y comparar una reproducción representativa antes y después. Registrar los cambios y sus efectos en la documentación de tesis.

No se realizó ninguna limpieza ni se inició entrenamiento, evaluación o modificación del flujo durante esta revisión.
