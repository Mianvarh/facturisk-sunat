# Documentación técnica

## Lenguaje y librerías
Python 3.11 o superior. pandas, numpy y pyarrow para datos; requests y BeautifulSoup para el scraping; pymongo para MongoDB; scikit-learn (y opcionalmente XGBoost, LightGBM y CatBoost) para los modelos; joblib para serializar; matplotlib y seaborn para gráficos; Tkinter y Pillow para la aplicación de escritorio.

## Módulos
- `src/datos.py`: acceso al dataset activo (demostración o importado) y a los datos SUNAT.
- `src/importar_datos.py`: importa historiales CSV o Excel y detecta sus columnas.
- `src/configuracion.py`: ajustes locales (dataset, fuentes SUNAT y MongoDB).
- `src/inspect_data.py`: reporte de calidad del dataset.
- `src/distributed_processing.py`: resumen por proveedor con lógica MapReduce por bloques.
- `src/scrape_sunat.py`: descarga y filtra el padrón reducido SUNAT.
- `src/fuentes_externas.py`: padrones SUNAT adicionales como variables 0/1.
- `src/load_mongodb.py`: carga los datos SUNAT en MongoDB con índice único y upsert.
- `src/prepare_dataset.py`: integra el histórico con SUNAT y separa definitivos y pendientes.
- `src/feature_engineering.py`: variables históricas por proveedor sin fuga de información.
- `src/train_model.py`: comparación de modelos, calibración y umbral.
- `src/predict_pending.py`: probabilidad y nivel de riesgo de cada pendiente.
- `src/resumen_resultados.py`: resumen de resultados en consola.
- `src/generate_documentation_data.py`: actualiza esta documentación con las métricas reales.
- `src/gui_app.py`, `src/dashboard.py`, `src/settings_view.py`: aplicación de escritorio.
- `src/theme.py`, `src/ui_widgets.py`: identidad visual y componentes de la interfaz.

## Entradas y salidas
Cada fase lee de `data/` y escribe en `data/processed/`, `data/models/` u `outputs/`. El dataset histórico nunca se modifica.

## Flujo entre módulos
El scraping produce `proveedores_sunat.csv` y `padrones_sunat.csv`; MongoDB los almacena por RUC; la preparación consulta MongoDB o el respaldo local e integra por `RUC_Proveedor`; el entrenamiento usa `dataset_modelo.csv` y la predicción `dataset_pendientes.csv` con el modelo guardado.

## Decisiones técnicas
El scraping usa reintentos, timeout, validación del ZIP y lectura por bloques. MongoDB usa `UpdateOne` con `upsert=True`. El entrenamiento usa `Pipeline` y `ColumnTransformer` para evitar fuga de datos y validación temporal. El modelo se guarda con sus variables, umbral y cortes de riesgo.
