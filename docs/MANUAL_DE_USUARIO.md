# Manual de usuario

## Uso básico
1. Clone o descargue el repositorio.
2. Instale los requisitos siguiendo el manual de instalación.
3. Ejecute `python main.py` para la consola o `python main.py gui` para la interfaz gráfica.
4. Mantenga conexión a internet para SUNAT y MongoDB.

## Opciones del menú
1. Inspeccionar datos: revisa filas, columnas, fechas, nulos y duplicados.
2. Procesamiento distribuido: resume el historico por chunks con logica MapReduce.
3. Ejecutar scraping SUNAT: descarga o reutiliza el padrón reducido del día.
4. Cargar MongoDB: actualiza proveedores en Atlas sin duplicarlos.
5. Preparar datos y crear variables: integra histórico con SUNAT y crea datasets.
6. Entrenar, optimizar umbral y calibrar: evalúa modelos y selecciona el mejor.
7. Predecir comprobantes pendientes: genera probabilidades de incidencia.
8. Mostrar gráficos y resultados: abre el resumen visual final.
9. Ejecutar proceso completo: ejecuta todo el flujo.

## Qué son los chunks
Los chunks son bloques de lectura. Se usan para procesar archivos grandes sin cargar todo en memoria.

## Tiempos aproximados
Inspección y predicción suelen tardar segundos. Scraping, MongoDB, preparación y entrenamiento pueden tardar varios minutos según internet, equipo y tamaño de datos.

## Proceso recomendado para la demostración
1. Inspeccionar datos.
2. Ejecutar procesamiento distribuido.
3. Ejecutar scraping SUNAT.
4. Cargar MongoDB.
5. Preparar datos.
6. Entrenar modelos, optimizar umbral y calibrar.
7. Predecir pendientes.
8. Mostrar gráficos y resultados.

## Problemas externos
Si SUNAT no responde, use la descarga previa. Si MongoDB no conecta, puede que la IP no esté autorizada en Atlas. El programa no muestra credenciales.
