# Configuración del servidor MCP de Perplexity

Fecha: 8 de noviembre de 2026.

Durante la pausa solicitada para una revisión independiente de la tesis se registró el servidor MCP de Perplexity en Codex. El comando facilitado inicialmente correspondía a Claude Code, cuyo ejecutable no estaba disponible en el entorno. Se utilizó el equivalente de Codex:

```powershell
codex mcp add perplexity --url https://api.perplexity.ai/mcp
```

El servidor quedó registrado como `perplexity`, habilitado y con la dirección indicada. La configuración inició un flujo de autorización OAuth, pero la persona usuaria informó que no pudo completarlo. En la última comprobación, `codex mcp list` indicaba autenticación `Unknown`; por tanto, **la conexión autenticada y su funcionamiento no se han verificado**. No se guardaron credenciales ni enlaces temporales de autorización en este informe.

Queda pendiente identificar el impedimento en la autorización, completarla y comprobar el acceso. La disponibilidad del servidor en una sesión de Codex ya iniciada también deberá verificarse; podría requerir una sesión nueva. Este registro no modifica los modelos, las simulaciones ni los resultados de la tesis.

La persona usuaria indicó posteriormente que aún no cuenta con una licencia y solicitó esperar hasta conseguirla. Se deja la autorización pendiente hasta entonces; después se deberá iniciar un nuevo proceso de autenticación y comprobar la conexión.
