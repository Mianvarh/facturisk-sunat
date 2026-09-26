"""Generate project documentation from the current real artifacts."""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from paths import ensure_directories, get_application_root

PROJECT_ROOT = get_application_root()
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON if present."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv_shape(path: Path) -> dict[str, Any]:
    """Return shape and column information for a CSV without exposing data."""

    if not path.exists():
        return {"existe": False, "filas": 0, "columnas": 0, "nombres": []}
    df = pd.read_csv(path, sep=",", encoding="utf-8-sig", nrows=5, low_memory=False)
    rows = sum(1 for _ in path.open("r", encoding="utf-8-sig", errors="ignore")) - 1
    return {"existe": True, "filas": max(rows, 0), "columnas": len(df.columns), "nombres": list(df.columns)}


def csv_sample(path: Path, sep: str = ",") -> pd.DataFrame:
    """Read a small CSV sample."""

    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path).head(20)
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig", nrows=20, low_memory=False)


def file_size(path: Path) -> str:
    """Human-readable file size."""

    if not path.exists():
        return "No existe"
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def safe_metric(metrics: dict[str, Any], key: str, default: Any = "N/D") -> Any:
    """Read a test metric from the current metrics file."""

    return metrics.get("metricas_prueba", {}).get(key, default)


def pct(value: Any) -> str:
    """Format a numeric value as percentage."""

    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/D"


def num(value: Any) -> str:
    """Format a number for Spanish documentation."""

    try:
        return f"{float(value):,.4f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return str(value)


def collect_summary() -> dict[str, Any]:
    """Collect real project values from generated artifacts."""

    ensure_directories()
    metrics = read_json(OUTPUTS_DIR / "metricas_modelo.json")
    prep = read_json(OUTPUTS_DIR / "reporte_preparacion.json")
    scraping = read_json(OUTPUTS_DIR / "reporte_scraping.json")
    distributed = read_json(OUTPUTS_DIR / "reporte_procesamiento_distribuido.json")
    modelo_shape = read_csv_shape(PROCESSED_DIR / "dataset_modelo.csv")
    pendientes_shape = read_csv_shape(PROCESSED_DIR / "dataset_pendientes.csv")
    predicciones_shape = read_csv_shape(PROCESSED_DIR / "predicciones_pendientes.csv")
    predicciones = csv_sample(PROCESSED_DIR / "predicciones_pendientes.csv")

    risk_distribution: dict[str, int] = {}
    prediction_distribution: dict[str, int] = {}
    if not predicciones.empty and (PROCESSED_DIR / "predicciones_pendientes.csv").exists():
        full_pred = pd.read_csv(PROCESSED_DIR / "predicciones_pendientes.csv", sep=",", encoding="utf-8-sig", low_memory=False)
        if "Nivel_Riesgo" in full_pred:
            risk_distribution = full_pred["Nivel_Riesgo"].value_counts(dropna=False).to_dict()
        if "Prediccion_Texto" in full_pred:
            prediction_distribution = full_pred["Prediccion_Texto"].value_counts(dropna=False).to_dict()

    aporte = {}
    aporte_path = OUTPUTS_DIR / "comparacion_aporte_sunat.csv"
    if aporte_path.exists():
        aporte_df = pd.read_csv(aporte_path)
        aporte = {
            "filas": len(aporte_df),
            "columnas": list(aporte_df.columns),
            "mejor_pr_auc": float(aporte_df["pr_auc"].max()) if "pr_auc" in aporte_df else None,
            "mejor_f1": float(aporte_df["f1_clase_1"].max()) if "f1_clase_1" in aporte_df else None,
        }

    graph_files = sorted(path.name for path in OUTPUTS_DIR.glob("*.png"))
    raw_csvs = sorted(path.name for path in RAW_DIR.glob("*.parquet"))
    generated_files = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in [
            OUTPUTS_DIR / "metricas_modelo.json",
            OUTPUTS_DIR / "reporte_modelo.txt",
            OUTPUTS_DIR / "comparacion_modelos.csv",
            OUTPUTS_DIR / "comparacion_umbrales.csv",
            OUTPUTS_DIR / "reporte_procesamiento_distribuido.json",
            OUTPUTS_DIR / "reporte_procesamiento_distribuido.txt",
            PROCESSED_DIR / "mapreduce_resumen_proveedores.csv",
            PROCESSED_DIR / "predicciones_pendientes.csv",
            MODELS_DIR / "modelo_incidencias.joblib",
        ]
        if path.exists()
    )

    return {
        "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
        "sistema": platform.platform(),
        "registros_historicos": prep.get("filas_historicas"),
        "registros_entrenamiento": modelo_shape["filas"],
        "registros_pendientes": pendientes_shape["filas"],
        "predicciones_pendientes": predicciones_shape["filas"],
        "cobertura_sunat_filas": prep.get("coincidencias_sunat"),
        "no_encontrados_sunat_filas": prep.get("no_encontrados"),
        "ruc_buscados": scraping.get("total_ruc_buscados"),
        "ruc_encontrados": scraping.get("total_encontrados"),
        "ruc_no_encontrados": scraping.get("total_no_encontrados"),
        "modelo": metrics.get("mejor_modelo"),
        "experimento": metrics.get("experimento"),
        "umbral": metrics.get("umbral"),
        "razon_umbral": metrics.get("razon_umbral"),
        "metricas": metrics.get("metricas_prueba", {}),
        "matriz_confusion": safe_metric(metrics, "matriz_confusion"),
        "distribucion_clases": prep.get("distribucion_incidencia_aceptacion"),
        "porcentaje_incidencias": prep.get("porcentaje_incidencias"),
        "distribucion_riesgos": risk_distribution,
        "distribucion_predicciones": prediction_distribution,
        "aporte_sunat": aporte,
        "procesamiento_distribuido": distributed,
        "confiabilidad": metrics.get("confiabilidad"),
        "calibracion": metrics.get("calibracion", {}),
        "features": metrics.get("features", []),
        "columnas_prohibidas_modelo": metrics.get("columnas_prohibidas_modelo", []),
        "archivos_generados": generated_files,
        "graficos": graph_files,
        "csv_historicos": raw_csvs,
        "tamano_modelo": file_size(MODELS_DIR / "modelo_incidencias.joblib"),
    }


# Columns that can hold names or addresses of companies or people; never shown as examples.
PRIVATE_COLUMNS = {"Proveedor", "Ruc Proveedor", "Cliente", "Razon_Social_SUNAT", "Razon_Social_Proveedor", "Domicilio_Fiscal"}


def table_columns(path: Path, sep: str, origin: str, model_features: list[str], forbidden: list[str]) -> str:
    """Build a Markdown dictionary table from a real CSV sample."""

    sample = csv_sample(path, sep=sep)
    if sample.empty:
        return "Archivo no disponible.\n"
    rows = [
        "| Columna | Tipo detectado | Ejemplo | Origen | Tratamiento | Uso en modelo | Motivo de exclusión |",
        "|---|---:|---|---|---|---|---|",
    ]
    for column in sample.columns:
        if column in PRIVATE_COLUMNS:
            example_text = ""
        else:
            example = sample[column].dropna().astype(str).head(1)
            example_text = example.iloc[0][:70].replace("|", "/") if len(example) else ""
        used = "Sí" if column in model_features else "No"
        reason = "Fuga de información o resultado conocido" if column in forbidden else ""
        if column == "Incidencia_Aceptacion":
            reason = "Variable objetivo"
        treatment = "Normalizada/derivada" if column in model_features or column.startswith(("mes_", "anio_", "trimestre_", "dia_", "log_", "plazo_", "SUNAT")) else "Conservada para trazabilidad"
        rows.append(f"| `{column}` | `{sample[column].dtype}` | {example_text} | {origin} | {treatment} | {used} | {reason} |")
    return "\n".join(rows) + "\n"


def write_docs(summary: dict[str, Any]) -> None:
    """Write all requested Markdown documentation files."""

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = summary["metricas"]
    mc = summary["matriz_confusion"]
    docs: dict[str, str] = {}

    docs["DOCUMENTACION_TECNICA.md"] = """# Documentación técnica

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
"""

    docs["MANUAL_DE_USUARIO.md"] = """# Manual de usuario

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
"""

    docs["MANUAL_DE_INSTALACION_DESARROLLO.md"] = """# Manual de instalación para desarrollo

## Requisitos
Python 3.11 o superior. En Windows se recomienda el instalador de python.org.

## Entorno virtual
```powershell
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

## Configuración
La conexión a MongoDB se configura desde la aplicación (Configuración) o con un archivo `.env` creado a partir de `.env.example`. Ninguno de los dos se sube al repositorio.

## Ejecución
```powershell
python main.py
python main.py gui
python main.py todo
```

## Pruebas
```powershell
python -m pytest
```
"""

    docs["INTERPRETACION_RESULTADOS.md"] = f"""# Interpretación de resultados

## Variable objetivo
`Incidencia_Aceptacion` vale 0 cuando el comprobante fue aceptado y 1 cuando fue observado o rechazado.

## Desbalance
Distribución actual: {summary['distribucion_clases']}. El porcentaje de incidencias reportado es {summary['porcentaje_incidencias']}%.

## Por qué accuracy no basta
Si hay pocas incidencias, un modelo puede acertar muchos aceptados y aun así fallar en detectar incidencias. Por eso se priorizan recall, F1 y PR-AUC de la clase 1.

## Métricas actuales
- Accuracy: {pct(metrics.get('accuracy'))}
- Balanced accuracy: {pct(metrics.get('balanced_accuracy'))}
- Precision clase 1: {pct(metrics.get('precision_clase_1'))}
- Recall clase 1: {pct(metrics.get('recall_clase_1'))}
- F1 clase 1: {pct(metrics.get('f1_clase_1'))}
- PR-AUC: {num(metrics.get('pr_auc'))}
- ROC-AUC: {num(metrics.get('roc_auc'))}
- Especificidad: {pct(metrics.get('specificity'))}
- MCC: {num(metrics.get('mcc'))}
- Falsos positivos: {metrics.get('falsos_positivos')}
- Falsos negativos: {metrics.get('falsos_negativos')}
- Umbral: {summary['umbral']}
- Calibración: {summary.get('calibracion', {}).get('calibracion_seleccionada', 'N/D')}

## Matriz de confusión actual
`{mc}`

## Gráficos
La matriz de calor Pearson muestra la relacion lineal entre variables candidatas y la incidencia. La matriz de confusion muestra errores y aciertos con el umbral optimizado. La curva Precision-Recall muestra el intercambio entre detectar incidencias y generar alertas. La comparacion de modelos resume el rendimiento interno. El aporte SUNAT permite comparar experimentos con y sin variables tributarias.

## Conclusión
La confiabilidad actual es {summary['confiabilidad']}. El modelo puede ayudar a priorizar revisiones, pero no debe rechazar comprobantes automáticamente cuando sus métricas y falsas alertas no sean suficientes.
"""

    docs["DICCIONARIO_DE_DATOS.md"] = "# Diccionario de datos\n\n## Variables históricas\n\n"
    raw_dataset = RAW_DIR / "comprobantes.parquet"
    if raw_dataset.exists():
        docs["DICCIONARIO_DE_DATOS.md"] += table_columns(raw_dataset, ";", "Dataset histórico", summary["features"], summary["columnas_prohibidas_modelo"])
    docs["DICCIONARIO_DE_DATOS.md"] += "\n## Dataset de entrenamiento\n\n"
    docs["DICCIONARIO_DE_DATOS.md"] += table_columns(PROCESSED_DIR / "dataset_modelo.csv", ",", "Preparación", summary["features"], summary["columnas_prohibidas_modelo"])
    docs["DICCIONARIO_DE_DATOS.md"] += "\n## Columnas excluidas por fuga\n\n"
    docs["DICCIONARIO_DE_DATOS.md"] += "\n".join(f"- `{col}`: revela directa o indirectamente el resultado." for col in summary["columnas_prohibidas_modelo"]) + "\n"

    docs["ARQUITECTURA_DEL_PROYECTO.md"] = """# Arquitectura del proyecto

## Flujo general
```mermaid
flowchart TD
  A["Dataset histórico (comprobantes.parquet)"] --> B["Inspección"]
  A --> C["Scraping SUNAT"]
  C --> D["CSV SUNAT filtrado"]
  D --> E["MongoDB Atlas"]
  E --> F["Preparación por RUC"]
  F --> G["Feature engineering"]
  G --> H["Entrenamiento temporal"]
  H --> I["Optimización de umbral"]
  I --> J["Predicción pendientes"]
  J --> K["Reportes y dashboard"]
```

## Componentes
```mermaid
flowchart LR
  U["Usuario Windows"] --> M["main.py / EXE"]
  M --> S["Scripts src"]
  S --> L["Archivos locales"]
  S --> N["SUNAT"]
  S --> DB["MongoDB Atlas"]
  S --> ML["Modelo ML joblib"]
```

## Entrenamiento
```mermaid
flowchart TD
  A["dataset_modelo.csv"] --> B["Orden temporal"]
  B --> C["Train / validación / prueba"]
  C --> D["Pipelines sklearn"]
  D --> E["Comparación interna"]
  E --> F["Calibración"]
  F --> G["Umbral optimizado"]
  G --> H["modelo_incidencias.joblib"]
```

## Predicción
```mermaid
flowchart TD
  A["dataset_pendientes.csv"] --> B["Modelo joblib"]
  B --> C["predict_proba"]
  C --> D["Umbral optimizado"]
  D --> E["predicciones_pendientes.csv"]
```

## Empaquetado portable
```mermaid
flowchart TD
  A["Código fuente"] --> B["PyInstaller onedir"]
  B --> C["FactuRisk_SUNAT.exe"]
  C --> D["Carpeta portable dist/FactuRisk_SUNAT"]
```
"""

    docs["SEGURIDAD_Y_CREDENCIALES.md"] = """# Seguridad y credenciales

- Las credenciales de MongoDB se guardan localmente en `config/configuracion.json` o `.env`; ambos están excluidos de Git.
- La URI se muestra enmascarada en la aplicación y nunca se imprime en consola, logs ni reportes.
- Use un usuario de MongoDB con permisos solo sobre la base del proyecto.
- En MongoDB Atlas, autorice solo las IP necesarias en Network Access.
- Los datos importados (`data/raw/local/`) y los resultados (`outputs/`, `data/processed/`) no se versionan, porque pueden contener razones sociales y domicilios.
"""

    docs["ERRORES_FRECUENTES.md"] = """# Errores frecuentes

- MongoDB no conecta: revisar internet, URI privada y lista de IP en Atlas.
- IP no autorizada: agregar temporalmente la IP del equipo en Network Access.
- SUNAT no responde: esperar o reutilizar la descarga válida existente.
- Dataset no encontrado: verificar `data/raw/comprobantes.parquet`.
- Error de formato: verificar la integridad de `data/raw/comprobantes.parquet`.
- Columnas faltantes: verificar el esquema de `data/raw/comprobantes.parquet`.
- Archivo bloqueado por Excel: cerrar Excel y volver a ejecutar.
- Modelo no encontrado: ejecutar entrenamiento o restaurar `data/models/modelo_incidencias.joblib`.
- Antivirus o SmartScreen bloquea el EXE: descomprimir en carpeta confiable y permitir ejecución.
- Rutas con espacios o tildes: el proyecto usa rutas portables, pero evite mover solo el EXE.
- Permisos de escritura: ejecutar desde una carpeta donde el usuario pueda escribir.
- Memoria insuficiente: el padrón SUNAT se procesa por chunks.
- PyInstaller hidden imports: revisar `build_windows_gui.spec`.
- EXE separado de `_internal`: asegurarse de mantener la estructura generada por PyInstaller.
"""

    docs["FLUJO_COMPLETO.md"] = """# Flujo completo

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
"""

    for name, content in docs.items():
        (DOCS_DIR / name).write_text(content.strip() + "\n", encoding="utf-8")


def guardar_resumen(summary: dict[str, Any]) -> None:
    """Persist the consolidated JSON summary."""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "resumen_proyecto.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def actualizar_documentacion_resultados() -> dict[str, Any]:
    """Main function used by the menu and CLI."""

    summary = collect_summary()
    guardar_resumen(summary)
    write_docs(summary)
    print("Documentación actualizada correctamente.")
    print(f"Resumen consolidado: {OUTPUTS_DIR / 'resumen_proyecto.json'}")
    print(f"Documentos: {DOCS_DIR}")
    return summary


if __name__ == "__main__":
    actualizar_documentacion_resultados()

