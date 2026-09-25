# Flujo completo

Al elegir `Ejecutar proceso completo`, el sistema valida archivos iniciales, descarga o reutiliza SUNAT, carga MongoDB, prepara datos, crea variables, entrena, compara modelos, optimiza umbral, calibra cuando corresponde, predice pendientes, genera gráficos, muestra resumen final y actualiza documentación dinámica.

## Fases
1. Inspección: entrada dataset histórico (comprobantes.parquet); salida `outputs/inspeccion_dataset.txt`.
2. Scraping: entrada RUC únicos; salida `proveedores_sunat.csv` y reporte JSON.
3. MongoDB: entrada proveedores SUNAT; salida colección actualizada por upsert.
4. Preparación: entrada histórico y MongoDB; salida datasets modelo y pendientes.
5. Entrenamiento: entrada dataset modelo; salida métricas, gráficos y joblib.
6. Predicción: entrada pendientes y joblib; salida predicciones.
7. Resultados: entrada outputs existentes; salida ventana, resumen y guion.
8. Documentación dinámica: entrada métricas reales; salida docs actualizados.

Si falla SUNAT o MongoDB, puede continuarse con datos existentes solo cuando los archivos locales necesarios ya existen.
