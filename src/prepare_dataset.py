"""Prepare ML and pending-prediction datasets from CSV history and SUNAT MongoDB data."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from configuracion import PADRON_COLUMNS, cargar_mongo_config
from fuentes_externas import cargar_variables_padrones
from datos import SUNAT_BACKUP_CSV, SUNAT_SNAPSHOT_PATH, leer_comprobantes, normalizar_ruc
from feature_engineering import HISTORICAL_FEATURES, crear_variables_historicas_sin_fuga
from paths import ensure_directories, get_application_root
from theme import PALETTE, apply_chart_style, diverging_cmap


apply_chart_style()

PROJECT_ROOT = get_application_root()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATE_COLUMNS = ["Fecha_Creacion", "Fecha_Emision", "Fecha_Vencimiento", "Fecha_Pago"]
DEFINITIVE_STATES = {"Aceptada", "Observada", "Rechazada"}
PENDING_STATES = {"Pendiente", "Borrador", "Pendiente / Borrador"}

COLUMNAS_PROHIBIDAS_MODELO = [
    "Estado_Aceptacion",
    "Flag_Aceptada",
    "Flag_Observada",
    "Flag_Rechazada",
    "Estado",
    "Estado_Grupo",
    "Estado_Emision",
    "Fecha_Pago",
]

VARIABLES_CANDIDATAS_CORRELACION = [
    "Tipo_Comprobante",
    "Moneda",
    "Importe_Total",
    "log_importe",
    "mes_emision",
    "anio_emision",
    "trimestre_emision",
    "dia_semana_emision",
    "plazo_dias",
    "Estado_RUC",
    "Condicion_Domicilio",
    "Situacion_Tributaria_Actual",
    "SUNAT_Encontrado",
    *PADRON_COLUMNS,
    *HISTORICAL_FEATURES,
]

TARGET_COLUMN = "Incidencia_Aceptacion"

SUNAT_FIELDS = [
    "RUC",
    "Razon_Social_SUNAT",
    "Estado_RUC",
    "Condicion_Domicilio",
    "Ubigeo",
    "Domicilio_Fiscal",
    "Situacion_Tributaria_Actual",
    "Fecha_Consulta",
    "Fuente",
]
# The public snapshot omits names and addresses; the model only needs these fields.
SUNAT_REQUIRED_FIELDS = ["RUC", "Estado_RUC", "Condicion_Domicilio", "Situacion_Tributaria_Actual"]


def configurar_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def leer_historico() -> pd.DataFrame:
    """Read the typed enterprise history and validate required columns."""

    logging.info("Leyendo historico empresarial.")
    df = leer_comprobantes()
    required = ["RUC_Proveedor", "Importe_Total", "Estado_Aceptacion", *DATE_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing)}")

    df["RUC_Proveedor"] = df["RUC_Proveedor"].map(normalizar_ruc).astype("string")
    for column in DATE_COLUMNS:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    df["Importe_Total"] = pd.to_numeric(df["Importe_Total"], errors="coerce")
    return df


def cargar_config_mongodb() -> tuple[str, str, str] | None:
    """MongoDB settings from the panel or .env; None when not configured."""

    return cargar_mongo_config()


def normalizar_sunat(df_sunat: pd.DataFrame, origen: str) -> pd.DataFrame:
    """Validate SUNAT fields, add optional ones and deduplicate by RUC."""

    missing = [field for field in SUNAT_REQUIRED_FIELDS if field not in df_sunat.columns]
    if missing:
        raise ValueError(f"Faltan campos SUNAT en {origen}: {', '.join(missing)}")
    df_sunat = df_sunat.copy()
    for field in SUNAT_FIELDS:
        if field not in df_sunat.columns:
            df_sunat[field] = pd.NA
    df_sunat = df_sunat[SUNAT_FIELDS]
    df_sunat["RUC"] = df_sunat["RUC"].map(normalizar_ruc).astype("string")
    df_sunat["Situacion_Tributaria_Actual"] = pd.to_numeric(df_sunat["Situacion_Tributaria_Actual"], errors="coerce")
    df_sunat = df_sunat.dropna(subset=["RUC"]).drop_duplicates(subset=["RUC"], keep="last")
    logging.info("Registros SUNAT cargados desde %s: %s", origen, len(df_sunat))
    return df_sunat


def cargar_sunat_desde_mongodb(config: tuple[str, str, str]) -> pd.DataFrame | None:
    """Fetch SUNAT supplier data from MongoDB; None when unavailable or empty."""

    uri, database, collection_name = config
    client: MongoClient | None = None
    try:
        logging.info("Conectando a MongoDB. Base=%s Coleccion=%s", database, collection_name)
        client = MongoClient(uri, serverSelectionTimeoutMS=30_000)
        client.admin.command("ping")
        docs = list(client[database][collection_name].find({}, {"_id": 0}))
    except PyMongoError as exc:
        logging.warning("MongoDB no esta disponible: %s", exc)
        return None
    finally:
        if client is not None:
            client.close()
    if not docs:
        logging.warning("La coleccion %s no contiene documentos.", collection_name)
        return None
    return normalizar_sunat(pd.DataFrame(docs), "MongoDB")


def cargar_sunat() -> pd.DataFrame:
    """Load SUNAT data: MongoDB first, then the local scraping backup, then the public snapshot."""

    config = cargar_config_mongodb()
    if config is None:
        logging.info("MongoDB no configurado; se usaran los respaldos locales.")
    else:
        df_sunat = cargar_sunat_desde_mongodb(config)
        if df_sunat is not None:
            return df_sunat

    if SUNAT_BACKUP_CSV.exists():
        logging.info("Usando respaldo local generado por scraping: %s", SUNAT_BACKUP_CSV)
        backup = pd.read_csv(SUNAT_BACKUP_CSV, sep=",", encoding="utf-8-sig", dtype={"RUC": "string"}, low_memory=False)
        return normalizar_sunat(backup, "respaldo local")
    if SUNAT_SNAPSHOT_PATH.exists():
        logging.info("Usando snapshot SUNAT incluido en el repositorio: %s", SUNAT_SNAPSHOT_PATH)
        return normalizar_sunat(pd.read_parquet(SUNAT_SNAPSHOT_PATH), "snapshot")
    raise FileNotFoundError(
        "No hay datos SUNAT disponibles. Ejecute la fase de scraping o configure MongoDB."
    )


def agregar_padrones(df: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 membership flags of the extra SUNAT registries (0 when unavailable)."""

    padrones = cargar_variables_padrones()
    if padrones is None:
        logging.info("Sin padrones SUNAT adicionales: las variables se completan con 0.")
        for column in PADRON_COLUMNS:
            df[column] = 0
        return df
    merged = df.merge(padrones.rename(columns={"RUC": "_RUC_padron"}), how="left", left_on="RUC_Proveedor", right_on="_RUC_padron")
    merged = merged.drop(columns=["_RUC_padron"])
    for column in PADRON_COLUMNS:
        merged[column] = merged[column].fillna(0).astype(int)
    logging.info("Padrones SUNAT agregados: %s", {column: int(merged[column].sum()) for column in PADRON_COLUMNS})
    return merged


def agregar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features and the target column."""

    df = df.copy()
    df["plazo_dias"] = (df["Fecha_Vencimiento"] - df["Fecha_Emision"]).dt.days
    df["mes_emision"] = df["Fecha_Emision"].dt.month
    df["anio_emision"] = df["Fecha_Emision"].dt.year
    df["trimestre_emision"] = df["Fecha_Emision"].dt.quarter
    df["dia_semana_emision"] = df["Fecha_Emision"].dt.dayofweek
    df["log_importe"] = np.log1p(df["Importe_Total"].abs())

    df["Incidencia_Aceptacion"] = pd.NA
    df.loc[df["Estado_Aceptacion"] == "Aceptada", "Incidencia_Aceptacion"] = 0
    df.loc[df["Estado_Aceptacion"].isin(["Observada", "Rechazada"]), "Incidencia_Aceptacion"] = 1
    return df


def combinar_historico_sunat(df_historico: pd.DataFrame, df_sunat: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Merge historical records with SUNAT data and return merge validation stats."""

    filas_antes = len(df_historico)
    df = df_historico.merge(
        df_sunat,
        how="left",
        left_on="RUC_Proveedor",
        right_on="RUC",
        indicator=True,
        validate="many_to_one",
    )
    filas_despues = len(df)
    if filas_antes != filas_despues:
        raise RuntimeError(f"El merge cambio la cantidad de filas: antes={filas_antes}, despues={filas_despues}")

    df["SUNAT_Encontrado"] = (df["_merge"] == "both").astype(int)
    coincidencias = int(df["SUNAT_Encontrado"].sum())
    no_encontrados = int((df["SUNAT_Encontrado"] == 0).sum())

    df["Situacion_Tributaria_Actual"] = df["Situacion_Tributaria_Actual"].fillna(-1).astype(int)
    df["Estado_RUC"] = df["Estado_RUC"].fillna("NO_ENCONTRADO")
    df["Condicion_Domicilio"] = df["Condicion_Domicilio"].fillna("NO_ENCONTRADO")
    df = df.drop(columns=["_merge"])

    stats = {
        "filas_antes_merge": filas_antes,
        "filas_despues_merge": filas_despues,
        "coincidencias_sunat_filas": coincidencias,
        "no_encontrados_sunat_filas": no_encontrados,
        "ruc_con_coincidencia": int(df.loc[df["SUNAT_Encontrado"] == 1, "RUC_Proveedor"].nunique()),
        "ruc_sin_coincidencia": int(df.loc[df["SUNAT_Encontrado"] == 0, "RUC_Proveedor"].nunique()),
    }
    return df, stats


def serializar_json(value: Any) -> Any:
    """Convert numpy/pandas/datetime values into JSON-serializable values."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def crear_reporte(
    df_completo: pd.DataFrame,
    dataset_modelo: pd.DataFrame,
    dataset_pendientes: pd.DataFrame,
    merge_stats: dict[str, int],
) -> dict[str, Any]:
    """Build preparation report with counts, target distribution and relevant nulls."""

    target_distribution = (
        dataset_modelo["Incidencia_Aceptacion"].astype("Int64").value_counts(dropna=False).sort_index()
    )
    relevant_null_columns = [
        "RUC_Proveedor",
        "Importe_Total",
        "Fecha_Emision",
        "Fecha_Vencimiento",
        "plazo_dias",
        "Situacion_Tributaria_Actual",
        "Estado_RUC",
        "Condicion_Domicilio",
        "SUNAT_Encontrado",
        "Incidencia_Aceptacion",
    ]
    nulls = {
        column: int(df_completo[column].isna().sum())
        for column in relevant_null_columns
        if column in df_completo.columns
    }
    incidencia_rate = (
        float((dataset_modelo["Incidencia_Aceptacion"].astype("Int64") == 1).mean() * 100)
        if len(dataset_modelo)
        else 0.0
    )

    return {
        "filas_historicas": int(len(df_completo)),
        "filas_con_estado_definitivo": int(len(dataset_modelo)),
        "filas_pendientes": int(len(dataset_pendientes)),
        "coincidencias_sunat": merge_stats["coincidencias_sunat_filas"],
        "no_encontrados": merge_stats["no_encontrados_sunat_filas"],
        "ruc_con_coincidencia": merge_stats["ruc_con_coincidencia"],
        "ruc_sin_coincidencia": merge_stats["ruc_sin_coincidencia"],
        "distribucion_incidencia_aceptacion": {
            str(key): int(value) for key, value in target_distribution.items()
        },
        "porcentaje_incidencias": round(incidencia_rate, 4),
        "valores_nulos_relevantes": nulls,
        "columnas_prohibidas_modelo": COLUMNAS_PROHIBIDAS_MODELO,
        "variables_historicas_sin_fuga": HISTORICAL_FEATURES,
    }


def generar_matriz_correlacion_pearson(dataset_modelo: pd.DataFrame) -> dict[str, Any]:
    """Generate Pearson correlation artifacts for candidate predictors and the target."""

    if dataset_modelo.empty:
        logging.warning("No se genero matriz de correlacion: dataset_modelo esta vacio.")
        return {
            "generado": False,
            "motivo": "dataset_modelo vacio",
        }

    available_columns = [
        column
        for column in VARIABLES_CANDIDATAS_CORRELACION
        if column in dataset_modelo.columns and column not in COLUMNAS_PROHIBIDAS_MODELO
    ]
    if TARGET_COLUMN not in dataset_modelo.columns:
        raise ValueError(f"No existe la variable objetivo requerida: {TARGET_COLUMN}")

    analysis_df = dataset_modelo[available_columns + [TARGET_COLUMN]].copy()
    categorical_columns = [
        column
        for column in analysis_df.columns
        if column != TARGET_COLUMN and (analysis_df[column].dtype == "object" or str(analysis_df[column].dtype) == "string")
    ]
    encoded_df = pd.get_dummies(
        analysis_df,
        columns=categorical_columns,
        dummy_na=True,
        drop_first=False,
        dtype=float,
    )
    encoded_df = encoded_df.apply(pd.to_numeric, errors="coerce")
    encoded_df = encoded_df.dropna(axis=1, how="all")

    if TARGET_COLUMN not in encoded_df.columns:
        raise ValueError("La variable objetivo se perdio durante la codificacion para correlacion.")

    correlations = encoded_df.corr(method="pearson", numeric_only=True)
    target_correlations = (
        correlations[TARGET_COLUMN]
        .drop(labels=[TARGET_COLUMN], errors="ignore")
        .dropna()
        .sort_values(key=lambda series: series.abs(), ascending=False)
    )
    top_variables = target_correlations.head(20)
    selected_columns = [TARGET_COLUMN, *top_variables.index.tolist()]
    selected_corr = correlations.loc[selected_columns, selected_columns]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUTS_DIR / "correlacion_pearson_variables.csv"
    png_path = OUTPUTS_DIR / "matriz_correlacion_pearson.png"

    pd.DataFrame(
        {
            "variable": target_correlations.index,
            "correlacion_pearson_con_incidencia": target_correlations.values,
            "correlacion_absoluta": target_correlations.abs().values,
        }
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(14, 10))
    heatmap = sns.heatmap(
        selected_corr,
        cmap=diverging_cmap(),
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"color": PALETTE["text"]},
        linewidths=0.4,
        linecolor=PALETTE["background"],
        cbar_kws={"label": "Correlacion Pearson"},
    )
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(colors=PALETTE["muted"])
    colorbar.set_label("Correlacion Pearson", color=PALETTE["muted"])
    plt.title("Matriz de calor Pearson - variables candidatas vs incidencia", fontsize=14, pad=16)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()

    logging.info("Matriz Pearson guardada: %s", png_path)
    logging.info("Correlaciones Pearson guardadas: %s", csv_path)
    return {
        "generado": True,
        "grafico": str(png_path),
        "tabla": str(csv_path),
        "variables_evaluadas": int(len(target_correlations)),
        "top_10_correlaciones": {
            str(variable): round(float(value), 6)
            for variable, value in target_correlations.head(10).items()
        },
    }


def guardar_outputs(
    dataset_modelo: pd.DataFrame,
    dataset_pendientes: pd.DataFrame,
    reporte: dict[str, Any],
) -> None:
    """Save prepared datasets and preparation report."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    modelo_path = PROCESSED_DIR / "dataset_modelo.csv"
    pendientes_path = PROCESSED_DIR / "dataset_pendientes.csv"
    reporte_path = OUTPUTS_DIR / "reporte_preparacion.json"

    dataset_modelo.to_csv(modelo_path, index=False, encoding="utf-8-sig")
    dataset_pendientes.to_csv(pendientes_path, index=False, encoding="utf-8-sig")
    reporte_path.write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2, default=serializar_json),
        encoding="utf-8",
    )

    logging.info("Dataset modelo guardado: %s", modelo_path)
    logging.info("Dataset pendientes guardado: %s", pendientes_path)
    logging.info("Reporte preparacion guardado: %s", reporte_path)


def preparar_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare model and pending datasets by combining history with SUNAT MongoDB data."""

    ensure_directories()
    configurar_logging()
    df_historico = leer_historico()
    df_sunat = cargar_sunat()
    df_completo, merge_stats = combinar_historico_sunat(df_historico, df_sunat)
    df_completo = agregar_padrones(df_completo)
    df_completo = agregar_variables(df_completo)
    df_completo, leakage_report = crear_variables_historicas_sin_fuga(df_completo)
    logging.info("Control de fuga feature engineering: %s", leakage_report["resultado_control_fuga"])

    definitive_mask = df_completo["Estado_Aceptacion"].isin(DEFINITIVE_STATES)
    pending_mask = df_completo["Estado_Aceptacion"].isin(PENDING_STATES)

    dataset_modelo = df_completo.loc[definitive_mask].copy()
    dataset_modelo["Incidencia_Aceptacion"] = dataset_modelo["Incidencia_Aceptacion"].astype(int)
    dataset_pendientes = df_completo.loc[pending_mask].copy()

    reporte = crear_reporte(df_completo, dataset_modelo, dataset_pendientes, merge_stats)
    reporte["matriz_correlacion_pearson"] = generar_matriz_correlacion_pearson(dataset_modelo)
    guardar_outputs(dataset_modelo, dataset_pendientes, reporte)

    print("\nResumen preparacion dataset")
    print(f"Filas historicas: {len(df_completo)}")
    print(f"Filas con estado definitivo: {len(dataset_modelo)}")
    print(f"Filas pendientes: {len(dataset_pendientes)}")
    print(f"Coincidencias SUNAT en filas: {merge_stats['coincidencias_sunat_filas']}")
    print(f"No encontrados SUNAT en filas: {merge_stats['no_encontrados_sunat_filas']}")
    print(f"Columnas prohibidas modelo: {COLUMNAS_PROHIBIDAS_MODELO}")
    print("Matriz Pearson: outputs/matriz_correlacion_pearson.png")
    print("Correlaciones Pearson: outputs/correlacion_pearson_variables.csv")

    return dataset_modelo, dataset_pendientes


if __name__ == "__main__":
    preparar_dataset()
