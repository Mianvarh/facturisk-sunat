"""Load filtered SUNAT supplier data into MongoDB Atlas."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, PyMongoError

from datos import SUNAT_BACKUP_CSV, SUNAT_SNAPSHOT_PATH, normalizar_ruc
from paths import ensure_directories, get_application_root

PROJECT_ROOT = get_application_root()

REQUIRED_FIELDS = [
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

BULK_BATCH_SIZE = 1_000


@dataclass(frozen=True)
class MongoConfig:
    """MongoDB connection settings loaded from environment variables."""

    uri: str
    database: str
    collection: str


@dataclass
class LoadStats:
    """Counters collected during the MongoDB load."""

    registros_leidos: int = 0
    insertados: int = 0
    actualizados: int = 0
    sin_cambios: int = 0
    errores: int = 0


def configurar_logging() -> None:
    """Configure readable console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cargar_configuracion() -> MongoConfig | None:
    """Load MongoDB configuration without exposing credentials; None when not configured."""

    load_dotenv(PROJECT_ROOT / ".env")
    uri = os.getenv("MONGODB_URI")
    database = os.getenv("MONGODB_DATABASE")
    collection = os.getenv("MONGODB_COLLECTION_SUNAT")

    config_path = PROJECT_ROOT / "config" / "configuracion.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8-sig") as file:
            config_data = json.load(file)
        uri = uri or config_data.get("mongodb_uri")
        database = database or config_data.get("mongodb_database")
        collection = collection or config_data.get("mongodb_collection")

    if not (uri and database and collection):
        return None
    return MongoConfig(uri=uri, database=database, collection=collection)


def leer_proveedores_sunat(
    backup_path: Path = SUNAT_BACKUP_CSV,
    snapshot_path: Path = SUNAT_SNAPSHOT_PATH,
) -> pd.DataFrame:
    """Read the SUNAT scraping backup, or the public snapshot when no scraping was run."""

    if backup_path.exists():
        logging.info("Leyendo respaldo local SUNAT: %s", backup_path)
        df = pd.read_csv(backup_path, encoding="utf-8-sig", dtype="string")
    elif snapshot_path.exists():
        logging.info("Sin respaldo de scraping; usando snapshot SUNAT: %s", snapshot_path)
        df = pd.read_parquet(snapshot_path).astype("string")
    else:
        raise FileNotFoundError(f"No existe el respaldo SUNAT: {backup_path}. Ejecute primero el scraping.")

    for field in REQUIRED_FIELDS:
        if field not in df.columns:
            df[field] = pd.NA
    return df[REQUIRED_FIELDS].copy()


def none_if_missing(value: object) -> Any:
    """Convert pandas missing values and blank strings to None."""

    if pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def convertir_fecha(value: object) -> datetime | None:
    """Convert Fecha_Consulta to a Python datetime."""

    cleaned = none_if_missing(value)
    if cleaned is None:
        return None
    parsed = pd.to_datetime(cleaned, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Fecha_Consulta no valida: {value}")
    return parsed.to_pydatetime()


def convertir_situacion(value: object) -> int | None:
    """Convert Situacion_Tributaria_Actual to int."""

    cleaned = none_if_missing(value)
    if cleaned is None:
        return None
    try:
        return int(cleaned)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Situacion_Tributaria_Actual no valida: {value}") from exc


def preparar_documentos(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Normalize CSV rows into MongoDB-ready documents."""

    documentos: list[dict[str, Any]] = []
    invalid_rucs = 0

    for row in df.to_dict(orient="records"):
        ruc = normalizar_ruc(row["RUC"])
        if ruc is None:
            invalid_rucs += 1
            continue

        documento = {
            "RUC": ruc,
            "Razon_Social_SUNAT": none_if_missing(row["Razon_Social_SUNAT"]),
            "Estado_RUC": none_if_missing(row["Estado_RUC"]),
            "Condicion_Domicilio": none_if_missing(row["Condicion_Domicilio"]),
            "Ubigeo": none_if_missing(row["Ubigeo"]),
            "Domicilio_Fiscal": none_if_missing(row["Domicilio_Fiscal"]),
            "Situacion_Tributaria_Actual": convertir_situacion(row["Situacion_Tributaria_Actual"]),
            "Fecha_Consulta": convertir_fecha(row["Fecha_Consulta"]),
            "Fuente": none_if_missing(row["Fuente"]),
        }
        documentos.append(documento)

    if invalid_rucs:
        logging.warning("Se omitieron %s filas con RUC invalido.", invalid_rucs)

    return documentos


def conectar_mongodb(config: MongoConfig) -> MongoClient:
    """Create a MongoDB client and run ping before writing."""

    logging.info(
        "Conectando a MongoDB Atlas. Base=%s Coleccion=%s",
        config.database,
        config.collection,
    )
    client: MongoClient = MongoClient(config.uri, serverSelectionTimeoutMS=30_000)
    client.admin.command("ping")
    logging.info("Ping a MongoDB exitoso.")
    return client


def obtener_coleccion(client: MongoClient, config: MongoConfig) -> Collection:
    """Return the configured collection and ensure a unique RUC index."""

    collection = client[config.database][config.collection]
    collection.create_index([("RUC", ASCENDING)], unique=True, name="idx_unique_ruc")
    logging.info("Indice unico sobre RUC verificado/creado.")
    return collection


def ejecutar_bulk_upsert(collection: Collection, documentos: list[dict[str, Any]]) -> LoadStats:
    """Insert or update documents using bulk_write with UpdateOne/upsert=True."""

    stats = LoadStats(registros_leidos=len(documentos))
    if not documentos:
        logging.warning("No hay documentos validos para cargar.")
        return stats

    for start in range(0, len(documentos), BULK_BATCH_SIZE):
        batch = documentos[start : start + BULK_BATCH_SIZE]
        operaciones = [
            UpdateOne({"RUC": doc["RUC"]}, {"$set": doc}, upsert=True)
            for doc in batch
        ]
        try:
            result = collection.bulk_write(operaciones, ordered=False)
        except BulkWriteError as exc:
            write_errors = exc.details.get("writeErrors", [])
            stats.errores += len(write_errors)
            logging.error("Errores en bulk_write: %s", len(write_errors))
            continue

        stats.insertados += int(result.upserted_count)
        stats.actualizados += int(result.modified_count)
        stats.sin_cambios += int(result.matched_count - result.modified_count)

    return stats


def mostrar_documentos_ejemplo(collection: Collection) -> None:
    """Log five example documents without credentials."""

    logging.info("Cinco documentos de ejemplo:")
    projection = {"_id": 0}
    for document in collection.find({}, projection).sort("RUC", ASCENDING).limit(5):
        printable = {
            key: (value.isoformat() if isinstance(value, datetime) else value)
            for key, value in document.items()
        }
        print(printable)


def cargar_proveedores_mongodb() -> LoadStats:
    """Load SUNAT suppliers into MongoDB Atlas with idempotent upserts."""

    ensure_directories()
    configurar_logging()
    config = cargar_configuracion()
    if config is None:
        print("MongoDB no configurado (.env o config/configuracion.json): se omite la carga.")
        print("El pipeline continuara con el respaldo local de SUNAT.")
        return LoadStats()
    df = leer_proveedores_sunat()
    documentos = preparar_documentos(df)

    client: MongoClient | None = None
    try:
        client = conectar_mongodb(config)
        collection = obtener_coleccion(client, config)
        stats = ejecutar_bulk_upsert(collection, documentos)
        mostrar_documentos_ejemplo(collection)
    except PyMongoError as exc:
        raise RuntimeError(f"Error de MongoDB: {exc}") from exc
    finally:
        if client is not None:
            client.close()

    print("\nResumen de carga MongoDB")
    print(f"Registros leidos: {len(df)}")
    print(f"Insertados: {stats.insertados}")
    print(f"Actualizados: {stats.actualizados}")
    print(f"Sin cambios: {stats.sin_cambios}")
    print(f"Errores: {stats.errores}")
    return stats


if __name__ == "__main__":
    cargar_proveedores_mongodb()
