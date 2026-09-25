# Documentación técnica

## Lenguaje y librerías
El proyecto está desarrollado en Python. Usa pandas y numpy para datos, requests y BeautifulSoup para scraping, pymongo para MongoDB Atlas, scikit-learn para modelos, joblib para serialización, matplotlib/seaborn para gráficos, Pillow para la ventana de resultados y python-dotenv para configuración.

## Organización por módulos
- `src/inspect_data.py`: inspecciona el dataset histórico y genera `outputs/inspeccion_dataset.txt`.
- `src/distributed_processing.py`: implementa procesamiento MapReduce local por chunks y genera reportes de resumen.
- `src/scrape_sunat.py`: descarga el padrón reducido SUNAT, detecta formato y filtra por RUC.
- `src/load_mongodb.py`: carga `proveedores_sunat.csv` en MongoDB con índice único y upsert.
- `src/prepare_dataset.py`: combina histórico con SUNAT desde MongoDB y crea datasets.
- `src/feature_engineering.py`: crea variables históricas sin fuga de datos.
- `src/train_model.py`: entrena, compara, calibra, optimiza umbral y guarda modelo.
- `src/predict_pending.py`: predice comprobantes pendientes usando el umbral guardado.
- `src/show_final_results.py`: muestra resumen final y ventana Tkinter.
- `src/preflight_check.py`: valida archivos, permisos y servicios externos.
- `src/generate_documentation_data.py`: actualiza documentación dinámica.
- `src/paths.py`: resuelve rutas en desarrollo y en PyInstaller.

## Entradas y salidas
Cada módulo lee archivos de `data/` y escribe resultados en `outputs/`, `data/processed/` o `data/models/`. El dataset histórico no se modifica.

## Flujo entre módulos
La fase distribuida produce `data/processed/mapreduce_resumen_proveedores.csv`; la fase SUNAT produce `data/processed/proveedores_sunat.csv`; MongoDB lo almacena por RUC; preparación consulta MongoDB o respaldo local e integra por `RUC_Proveedor`; entrenamiento usa `dataset_modelo.csv`; predicción usa `dataset_pendientes.csv` y el joblib activo.

## Manejo técnico
El scraping trabaja con reintentos, timeout, User-Agent, validación ZIP y lectura por chunks. MongoDB usa `UpdateOne` con `upsert=True`. El entrenamiento usa `Pipeline` y `ColumnTransformer` para evitar fuga de datos. La validación es temporal. El modelo se guarda con joblib junto con features, umbral y cortes de riesgo.

## PyInstaller
La versión portable usa modo `onedir`. Los archivos modificables viven al lado del EXE y no dentro de `_internal`.
