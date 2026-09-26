# Interpretación de resultados

## Variable objetivo
`Incidencia_Aceptacion` vale 0 cuando el comprobante fue aceptado y 1 cuando fue observado o rechazado.

## Desbalance
Distribución actual: {'0': 91585, '1': 8415}. El porcentaje de incidencias reportado es 8.415%.

## Por qué accuracy no basta
Si hay pocas incidencias, un modelo puede acertar muchos aceptados y aun así fallar en detectar incidencias. Por eso se priorizan recall, F1 y PR-AUC de la clase 1.

## Métricas actuales
- Accuracy: 57.64%
- Balanced accuracy: 61.48%
- Precision clase 1: 12.44%
- Recall clase 1: 66.12%
- F1 clase 1: 20.94%
- PR-AUC: 0,1243
- ROC-AUC: 0,6484
- Especificidad: 56.85%
- MCC: 0,1286
- Falsos positivos: 7898
- Falsos negativos: 575
- Umbral: 0.11
- Calibración: isotonic

## Matriz de confusión actual
`[[10405, 7898], [575, 1122]]`

## Gráficos
La matriz de calor Pearson muestra la relacion lineal entre variables candidatas y la incidencia. La matriz de confusion muestra errores y aciertos con el umbral optimizado. La curva Precision-Recall muestra el intercambio entre detectar incidencias y generar alertas. La comparacion de modelos resume el rendimiento interno. El aporte SUNAT permite comparar experimentos con y sin variables tributarias.

## Conclusión
La confiabilidad actual es Limitada. El modelo puede ayudar a priorizar revisiones, pero no debe rechazar comprobantes automáticamente cuando sus métricas y falsas alertas no sean suficientes.
