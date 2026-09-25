# FactuRisk SUNAT

Proyecto académico aplicado al contexto de una empresa peruana de facturación electrónica para estimar el riesgo de incidencia en comprobantes electrónicos pre-registrados.

## Problema
La empresa gestiona comprobantes electrónicos con estados finales como `Aceptada`, `Observada` y `Rechazada`, además de estados pendientes. Revisar todos los pendientes con el mismo nivel de prioridad consume tiempo. El proyecto busca ordenar esa revisión usando datos históricos, información tributaria actual de SUNAT y Machine Learning.

## Objetivo general
Preparar un flujo reproducible que inspeccione datos, actualice información tributaria, integre MongoDB Atlas, entrene modelos y prediga el riesgo de incidencia de comprobantes pendientes.

## Objetivos específicos
- Inspeccionar la base histórica sin modificar el archivo original.
- Obtener del padrón reducido SUNAT la situación actual de proveedores.
- Guardar la información SUNAT en MongoDB Atlas con upsert por RUC.
- Crear datasets de entrenamiento y predicción.
- Entrenar y comparar modelos con validación temporal.
- Optimizar el umbral de clasificación.
- Generar predicciones, métricas, gráficos y una ventana final de resultados.

## Fuentes
- Datos históricos: Dataset empresarial en `data/raw/comprobantes.parquet`.
- Fuente externa: SUNAT - Padrón Reducido.
- Base NoSQL: MongoDB Atlas, base `facturisk`, colección `proveedores_sunat`.

## Flujo textual
Datos históricos
→ scraping SUNAT
→ CSV temporal
→ MongoDB Atlas
→ integración por RUC
→ preparación de datos
→ creación de variables
→ entrenamiento
→ optimización del umbral
→ predicción
→ gráficos y reporte final

## Estructura principal
- `src/`: scripts del pipeline.
- `data/raw/`: CSV histórico y descarga SUNAT.
- `data/processed/`: datasets procesados y predicciones.
- `data/models/`: modelo serializado con joblib.
- `outputs/`: métricas, reportes y gráficos.
- `docs/`: documentación.
- `config/`: configuración portable.
- `logs/`: registros de ejecución.

## Comandos disponibles
```powershell
python .\main.py inspect
python .\main.py distribuido
python .\main.py scraping
python .\main.py mongodb
python .\main.py preparar
python .\main.py entrenar
python .\main.py predecir
python .\main.py resumen
python .\main.py documentar
python .\main.py todo
```

## Menú
1. Inspeccionar datos
2. Procesamiento distribuido
3. Ejecutar scraping SUNAT
4. Cargar MongoDB
5. Preparar datos y crear variables
6. Entrenar, optimizar umbral y calibrar
7. Predecir comprobantes pendientes
8. Mostrar gráficos y resultados
9. Ejecutar proceso completo
0. Salir

## Resultados actuales
- Registros históricos: 150000
- Registros de entrenamiento: 100000
- Registros pendientes: 50000
- Modelo seleccionado actualmente: HistGradientBoostingClassifier (B_con_sunat)
- Umbral seleccionado: 0.1
- Recall clase 1: 73.31%
- F1 clase 1: 21.21%
- PR-AUC: 0,1223
- Falsos positivos: 8792
- Falsos negativos: 453
- Confiabilidad: Limitada

## Uso recomendado
El modelo debe usarse para priorizar revisiones manuales de comprobantes pendientes. No debe rechazar comprobantes automáticamente.

## Ejecución
Para ejecutar el proyecto utilice `python main.py` o `python main.py gui` para abrir la interfaz gráfica.
