# Guía de exposición

## Exposición breve de 3 minutos
El proyecto resuelve la priorización de comprobantes electrónicos pre-registrados que podrían terminar observados o rechazados. Se parte de una base histórica, se corrigen columnas de RUC y razón social, se integran datos actuales del padrón reducido SUNAT y se guardan en MongoDB Atlas.

La variable adicional más importante desde SUNAT es la situación tributaria actual, creada a partir de estado RUC activo y condición habido. MongoDB permite mantener esa información externa actualizada y consultable sin depender solo del CSV local.

El modelo seleccionado actual es XGBClassifier con experimento B_con_sunat y umbral 0.11. El recall de incidencias es 66.12%, el F1 es 20.94% y la PR-AUC es 0,1243. La conclusión es que sirve para priorización manual, no para rechazo automático.

## Exposición completa de 7 minutos
El problema consiste en revisar comprobantes pendientes de aceptación. La solución construye un pipeline completo: inspección, scraping SUNAT, MongoDB Atlas, preparación, variables históricas sin fuga, entrenamiento con validación temporal, calibración, optimización de umbral y predicción.

El tratamiento incluyo normalizacion de RUC a 11 digitos, conversion de fechas, creacion de plazos, variables de calendario, transformacion logaritmica del importe, matriz Pearson de variables candidatas y variables historicas calculadas solo con informacion previa. Esto evita fuga de datos porque no se usan columnas que revelan el resultado, como `Estado_Aceptacion`, flags o `Fecha_Pago`.

SUNAT aporta estado tributario, condición de domicilio y domicilio fiscal. MongoDB se utiliza como capa NoSQL para almacenar la foto tributaria actual y actualizarla por upsert sin duplicar RUC.

La validación temporal usa datos más antiguos para entrenar y más recientes para probar. Este criterio es más realista que un corte aleatorio. El umbral se optimiza para mejorar F1 manteniendo recall mínimo, porque el objetivo es detectar incidencias.

Los retos principales fueron el desbalance de clases, el tamaño del padrón SUNAT, la curación de columnas con nombres invertidos y la necesidad de no exponer credenciales. Las mejoras futuras incluyen más variables operativas, monitoreo de deriva, validación con periodos posteriores y revisión del costo real de falsos positivos.
