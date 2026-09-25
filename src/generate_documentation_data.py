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
        if column in ("Proveedor", "Ruc Proveedor", "Cliente", "Razon_Social_SUNAT") and sample[column].dtype == "object":
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

    docs["README_PROYECTO.md"] = f"""# FactuRisk SUNAT

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
python .\\main.py inspect
python .\\main.py distribuido
python .\\main.py scraping
python .\\main.py mongodb
python .\\main.py preparar
python .\\main.py entrenar
python .\\main.py predecir
python .\\main.py resumen
python .\\main.py documentar
python .\\main.py todo
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
- Registros históricos: {summary['registros_historicos']}
- Registros de entrenamiento: {summary['registros_entrenamiento']}
- Registros pendientes: {summary['registros_pendientes']}
- Modelo seleccionado actualmente: {summary['modelo']} ({summary['experimento']})
- Umbral seleccionado: {summary['umbral']}
- Recall clase 1: {pct(metrics.get('recall_clase_1'))}
- F1 clase 1: {pct(metrics.get('f1_clase_1'))}
- PR-AUC: {num(metrics.get('pr_auc'))}
- Falsos positivos: {metrics.get('falsos_positivos')}
- Falsos negativos: {metrics.get('falsos_negativos')}
- Confiabilidad: {summary['confiabilidad']}

## Uso recomendado
El modelo debe usarse para priorizar revisiones manuales de comprobantes pendientes. No debe rechazar comprobantes automáticamente.

## Ejecución
Para ejecutar el proyecto utilice `python main.py` o `python main.py gui` para abrir la interfaz gráfica.
"""

    docs["DOCUMENTACION_TECNICA.md"] = f"""# Documentación técnica

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
"""

    docs["MANUAL_DE_USUARIO.md"] = """# Manual de usuario

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
"""

    docs["MANUAL_DE_INSTALACION_DESARROLLO.md"] = """# Manual de instalación para desarrollo

## Requisitos
Python 3.11 o superior recomendado.

## Entorno virtual
```powershell
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Configuración
Cree `.env` desde `.env.example` con las variables `MONGODB_URI`, `MONGODB_DATABASE` y `MONGODB_COLLECTION_SUNAT`. No suba credenciales a repositorios públicos.

## Ejecución
```powershell
python .\\main.py
python .\\main.py gui
python .\\main.py todo
```

## Compilación
```powershell
.\\build_windows_gui.bat
```

## Pruebas sugeridas
Ejecutar `py_compile`, `preflight_check.py`, `show_final_results.py --no-gui` y una prueba del portable en una carpeta aislada.
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

## Conclusión honesta
La confiabilidad actual es {summary['confiabilidad']}. El modelo puede ayudar a priorizar revisiones, pero no debe rechazar comprobantes automáticamente cuando sus métricas y falsas alertas no sean suficientes.
"""

    docs["GUIA_DE_EXPOSICION.md"] = f"""# Guía de exposición

## Exposición breve de 3 minutos
El proyecto resuelve la priorización de comprobantes electrónicos pre-registrados que podrían terminar observados o rechazados. Se parte de una base histórica, se corrigen columnas de RUC y razón social, se integran datos actuales del padrón reducido SUNAT y se guardan en MongoDB Atlas.

La variable adicional más importante desde SUNAT es la situación tributaria actual, creada a partir de estado RUC activo y condición habido. MongoDB permite mantener esa información externa actualizada y consultable sin depender solo del CSV local.

El modelo seleccionado actual es {summary['modelo']} con experimento {summary['experimento']} y umbral {summary['umbral']}. El recall de incidencias es {pct(metrics.get('recall_clase_1'))}, el F1 es {pct(metrics.get('f1_clase_1'))} y la PR-AUC es {num(metrics.get('pr_auc'))}. La conclusión es que sirve para priorización manual, no para rechazo automático.

## Exposición completa de 7 minutos
El problema consiste en revisar comprobantes pendientes de aceptación. La solución construye un pipeline completo: inspección, scraping SUNAT, MongoDB Atlas, preparación, variables históricas sin fuga, entrenamiento con validación temporal, calibración, optimización de umbral y predicción.

El tratamiento incluyo normalizacion de RUC a 11 digitos, conversion de fechas, creacion de plazos, variables de calendario, transformacion logaritmica del importe, matriz Pearson de variables candidatas y variables historicas calculadas solo con informacion previa. Esto evita fuga de datos porque no se usan columnas que revelan el resultado, como `Estado_Aceptacion`, flags o `Fecha_Pago`.

SUNAT aporta estado tributario, condición de domicilio y domicilio fiscal. MongoDB se utiliza como capa NoSQL para almacenar la foto tributaria actual y actualizarla por upsert sin duplicar RUC.

La validación temporal usa datos más antiguos para entrenar y más recientes para probar. Este criterio es más realista que un corte aleatorio. El umbral se optimiza para mejorar F1 manteniendo recall mínimo, porque el objetivo es detectar incidencias.

Los retos principales fueron el desbalance de clases, el tamaño del padrón SUNAT, la curación de columnas con nombres invertidos y la necesidad de no exponer credenciales. Las mejoras futuras incluyen más variables operativas, monitoreo de deriva, validación con periodos posteriores y revisión del costo real de falsos positivos.
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
  J --> K["Reportes y ventana final"]
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

La versión privada puede incluir una cuenta MongoDB de demostración en `config/configuracion.json`. Esa cuenta no debe ser administradora y debe limitarse a la base `facturisk`.

La contraseña no debe imprimirse en consola, logs, reportes ni documentación. Las credenciales no deben compartirse públicamente. Después de la prueba se recomienda cambiar o eliminar el usuario.

MongoDB Atlas puede requerir autorización de IP. Permitir `0.0.0.0/0` facilita pruebas, pero implica riesgo; es preferible una lista temporal y limitada.
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
- Error de Pillow: reinstalar dependencias o recompilar portable.
- Ventana gráfica no abre: usar la salida de consola y revisar logs.
- Antivirus o SmartScreen bloquea el EXE: descomprimir en carpeta confiable y permitir ejecución.
- Rutas con espacios o tildes: el proyecto usa rutas portables, pero evite mover solo el EXE.
- Permisos de escritura: ejecutar desde una carpeta donde el usuario pueda escribir.
- Memoria insuficiente: el padrón SUNAT se procesa por chunks.
- PyInstaller hidden imports: revisar `build_windows_gui.spec`.
- EXE separado de `_internal`: asegurarse de mantener la estructura generada por PyInstaller.
"""

    docs["FLUJO_COMPLETO.md"] = """# Flujo completo

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
"""

    docs["HISTORIAL_DE_CAMBIOS.md"] = f"""# Historial de cambios

## {datetime.now().date().isoformat()}
- Se documentó el proyecto completo.
- Se incorporó archivo de contexto para otra IA.
- Se agregó soporte de rutas portables.
- Se creó verificación inicial.
- Se mantuvo el modelo activo actual: {summary['modelo']} ({summary['experimento']}).
- Se conserva la comparación interna de modelos durante el entrenamiento.
- Se documentó la incorporación de variables SUNAT.
- Se documentó la creación de variables históricas sin fuga.
- Se documentó la optimización de umbral y calibración.
- Se preparó configuración para versión portable.

Nota: no se inventan fechas antiguas ni métricas; los valores se leen desde los outputs actuales.
"""

    for name, content in docs.items():
        (DOCS_DIR / name).write_text(content.strip() + "\n", encoding="utf-8")


def write_ai_context(summary: dict[str, Any]) -> None:
    """Write the standalone context file for another AI."""

    metrics = summary["metricas"]
    content = f"""# Contexto completo para IA

## Resumen general
`FactuRisk SUNAT` predice incidencias de aceptación en comprobantes electrónicos pre-registrados usando histórico empresarial, SUNAT, MongoDB Atlas y Machine Learning.

## Problema de negocio
Priorizar revisión preventiva de comprobantes pendientes para una empresa peruana de facturación electrónica.

## Objetivo del modelo
Predecir `Incidencia_Aceptacion`: clase 0 aceptado, clase 1 observado o rechazado.

## Datos
- Registros históricos: {summary['registros_historicos']}
- Registros entrenamiento: {summary['registros_entrenamiento']}
- Registros pendientes: {summary['registros_pendientes']}
- Desbalance: {summary['distribucion_clases']}
- Porcentaje de incidencias: {summary['porcentaje_incidencias']}%

## SUNAT y MongoDB
SUNAT aporta RUC, razón social, estado RUC, condición de domicilio, ubigeo y domicilio fiscal. MongoDB Atlas almacena la foto tributaria actual en `facturisk.proveedores_sunat` con upsert por RUC.

## Estructura
- `main.py`: menú y orquestación.
- `src/inspect_data.py`: inspección.
- `src/scrape_sunat.py`: SUNAT.
- `src/load_mongodb.py`: carga Atlas.
- `src/prepare_dataset.py`: integración y datasets.
- `src/feature_engineering.py`: variables históricas sin fuga.
- `src/train_model.py`: entrenamiento, comparación, calibración y umbral.
- `src/predict_pending.py`: predicción de pendientes.
- `src/show_final_results.py`: consola y GUI.
- `src/preflight_check.py`: verificación inicial.
- `src/generate_documentation_data.py`: documentación dinámica.

## Flujo de ejecución
Inspección → scraping SUNAT → MongoDB → preparación → feature engineering → entrenamiento temporal → optimización de umbral → calibración → predicción → resultados.

## Modelo actual
- Modelo seleccionado: {summary['modelo']}
- Experimento: {summary['experimento']}
- Umbral: {summary['umbral']}
- Razón del umbral: {summary['razon_umbral']}
- Calibración: {summary.get('calibracion', {}).get('calibracion_seleccionada', 'N/D')}

## Métricas reales actuales
- Accuracy: {metrics.get('accuracy')}
- Balanced accuracy: {metrics.get('balanced_accuracy')}
- Precision clase 1: {metrics.get('precision_clase_1')}
- Recall clase 1: {metrics.get('recall_clase_1')}
- F1 clase 1: {metrics.get('f1_clase_1')}
- PR-AUC: {metrics.get('pr_auc')}
- ROC-AUC: {metrics.get('roc_auc')}
- Especificidad: {metrics.get('specificity')}
- MCC: {metrics.get('mcc')}
- Falsos positivos: {metrics.get('falsos_positivos')}
- Falsos negativos: {metrics.get('falsos_negativos')}
- Matriz de confusión: {summary['matriz_confusion']}
- Confiabilidad: {summary['confiabilidad']}

## Variables utilizadas
{chr(10).join(f'- `{feature}`' for feature in summary['features'])}

## Variables excluidas
{chr(10).join(f'- `{column}`' for column in summary['columnas_prohibidas_modelo'])}

## Ingeniería de características
Incluye variables de calendario, `log_importe`, `plazo_dias`, matriz Pearson para analisis de variables candidatas y variables historicas por proveedor calculadas sin mirar el resultado futuro.

## Aporte SUNAT
Resumen disponible: {summary['aporte_sunat']}. Comparar siempre con `outputs/comparacion_aporte_sunat.csv`.

## Resultados de pendientes
- Total predicciones: {summary['predicciones_pendientes']}
- Distribución de riesgo: {summary['distribucion_riesgos']}
- Distribución de predicciones: {summary['distribucion_predicciones']}

## Gráficos generados
{chr(10).join(f'- `outputs/{graph}`' for graph in summary['graficos'])}

## Interpretación
El modelo sirve para priorización; no debe usarse como decisión automática de rechazo.

## Credenciales
La configuración puede existir en `.env` o `config/configuracion.json`. No mostrar URI, usuario ni contraseña en consola, logs, reportes o documentación.

## Resultado del entrenamiento
El archivo `data/models/modelo_incidencias.joblib` contiene el mejor modelo seleccionado por el entrenamiento vigente, junto con sus variables y umbral operativo.

## Estado actual
Proyecto funcional con modelo activo guardado, predicciones existentes, gráficos, documentación y preparación portable.

## INSTRUCCIONES PARA LA IA QUE CONTINÚE EL PROYECTO
- No inventar columnas.
- Inspeccionar archivos reales antes de modificar código.
- No afirmar alta confiabilidad solo por accuracy.
- Mantener validación temporal.
- Evitar fuga de datos.
- Conservar el modelo seleccionado y validar futuros entrenamientos con metricas comparables.
- Comprobar consistencia entre métricas y gráficos.
- No mostrar credenciales.
- Realizar respaldos.
- Ejecutar pruebas reales.
- Documentar cada cambio.

## COMANDOS PRINCIPALES
```powershell
python .\\main.py
python .\\main.py todo
python .\\main.py documentar
.\\build_windows_gui.bat
```

## ARCHIVOS CRÍTICOS
- `data/raw/comprobantes.parquet`
- `data/models/modelo_incidencias.joblib`
- `outputs/metricas_modelo.json`
- `data/processed/dataset_modelo.csv`
- `data/processed/dataset_pendientes.csv`
- `data/processed/predicciones_pendientes.csv`
- `.env`
- `config/configuracion.json`
"""
    (PROJECT_ROOT / "CONTEXTO_COMPLETO_PARA_IA.md").write_text(content.strip() + "\n", encoding="utf-8")


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
    write_ai_context(summary)
    print("Documentación actualizada correctamente.")
    print(f"Resumen consolidado: {OUTPUTS_DIR / 'resumen_proyecto.json'}")
    print(f"Documentos: {DOCS_DIR}")
    print(f"Contexto para IA: {PROJECT_ROOT / 'CONTEXTO_COMPLETO_PARA_IA.md'}")
    return summary


if __name__ == "__main__":
    actualizar_documentacion_resultados()

