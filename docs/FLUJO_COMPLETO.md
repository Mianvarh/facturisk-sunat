# Flujo completo

`Ejecutar proceso completo` recorre todas las fases en orden y se detiene si alguna falla.

1. Inspección: dataset histórico → `outputs/inspeccion_dataset.txt`.
2. Procesamiento distribuido: dataset histórico → `mapreduce_resumen_proveedores.csv`.
3. Scraping: RUC únicos → `proveedores_sunat.csv` y `padrones_sunat.csv`.
4. MongoDB: datos SUNAT → colección actualizada por upsert.
5. Preparación: histórico y SUNAT → `dataset_modelo.csv` y `dataset_pendientes.csv`.
6. Entrenamiento: dataset modelo → métricas, gráficos y modelo guardado.
7. Predicción: pendientes y modelo → `predicciones_pendientes.csv`.
8. Reportes: métricas reales → documentación actualizada.

Si SUNAT o MongoDB no están disponibles, el pipeline continúa con los respaldos locales.
