# Interpretación de resultados

## Variable objetivo
`Incidencia_Aceptacion` vale 0 cuando el comprobante fue aceptado y 1 cuando fue observado o rechazado.

## Desbalance
Distribución actual: {'0': 91585, '1': 8415}. El porcentaje de incidencias reportado es 8.415%.

## Por qué accuracy no basta
Si hay pocas incidencias, un modelo puede acertar muchos aceptados y aun así fallar en detectar incidencias. Por eso se priorizan recall, F1 y PR-AUC de la clase 1.

## Métricas actuales
- Accuracy: 53.77%
- Balanced accuracy: 62.63%
- Precision clase 1: 12.40%
- Recall clase 1: 73.31%
- F1 clase 1: 21.21%
- PR-AUC: 0,1223
- ROC-AUC: 0,6469
- Especificidad: 51.96%
- MCC: 0,1408
- Falsos positivos: 8792
- Falsos negativos: 453
- Umbral: 0.1
- Calibración: isotonic

## Matriz de confusión actual
`[[9511, 8792], [453, 1244]]`

## Gráficos
La matriz de calor Pearson muestra la relacion lineal entre variables candidatas y la incidencia. La matriz de confusion muestra errores y aciertos con el umbral optimizado. La curva Precision-Recall muestra el intercambio entre detectar incidencias y generar alertas. La comparacion de modelos resume el rendimiento interno. El aporte SUNAT permite comparar experimentos con y sin variables tributarias.

## Conclusión honesta
La confiabilidad actual es Limitada. El modelo puede ayudar a priorizar revisiones, pero no debe rechazar comprobantes automáticamente cuando sus métricas y falsas alertas no sean suficientes.
