# Manual de usuario

## Inicio
Ejecute `python main.py gui` para abrir la aplicación o `python main.py` para el menú de consola. En Windows, `python scripts/crear_acceso_directo.py` crea un acceso directo en el escritorio.

## Configuración
Desde **Configuración** puede:
- Importar un historial propio en CSV o Excel, o volver al dataset de demostración.
- Cambiar la fuente del padrón SUNAT (página, URL directa del ZIP o archivo local) y elegir padrones adicionales.
- Guardar y probar la conexión a MongoDB.

## Fases
1. Inspeccionar datos: filas, columnas, fechas, nulos y duplicados.
2. Procesamiento distribuido: resumen por proveedor en bloques.
3. Scraping SUNAT: descarga o reutiliza una descarga reciente del padrón.
4. Cargar MongoDB: actualiza los proveedores sin duplicarlos.
5. Preparar datos: integra histórico y SUNAT y crea variables.
6. Entrenar: compara modelos, calibra y define el umbral.
7. Predecir: calcula la probabilidad de incidencia de los pendientes.
8. Reportes: actualiza métricas, gráficos y documentación.

## Dashboard de riesgo
Muestra indicadores, gráficos interactivos y la tabla de comprobantes prioritarios. Un clic en un gráfico filtra la tabla y un doble clic en una fila abre la ficha del comprobante. El botón **Ocultar nombres** enmascara razones sociales y domicilios.

## Tiempos aproximados
La inspección y la predicción tardan segundos; el scraping, la preparación y el entrenamiento pueden tardar varios minutos.

## Problemas externos
Si SUNAT no responde, se reutiliza la última descarga válida. Si MongoDB no conecta, el pipeline usa el respaldo local.
