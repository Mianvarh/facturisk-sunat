"""Shared access to the public invoice dataset and the SUNAT snapshot."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from paths import get_application_root

PROJECT_ROOT = get_application_root()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COMPROBANTES_PATH = RAW_DIR / "comprobantes.parquet"
SUNAT_BACKUP_CSV = PROCESSED_DIR / "proveedores_sunat.csv"
SUNAT_SNAPSHOT_PATH = RAW_DIR / "proveedores_sunat_snapshot.parquet"

DATE_COLUMNS = ["Fecha_Creacion", "Fecha_Emision", "Fecha_Vencimiento", "Fecha_Pago"]


def normalizar_ruc(value: object) -> str | None:
    """Normalize a RUC-like value as an 11-digit string."""

    if pd.isna(value):
        return None
    digits = re.sub(r"\D", "", str(value).strip())
    if 0 < len(digits) <= 11:
        return digits.zfill(11)
    return None


def _validar_dataset(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el dataset de comprobantes: {path}. "
            "Genere el archivo con scripts/convertir_dataset.py."
        )


def leer_comprobantes(columns: list[str] | None = None, path: Path = COMPROBANTES_PATH) -> pd.DataFrame:
    """Read the typed invoice history from Parquet."""

    _validar_dataset(path)
    return pd.read_parquet(path, columns=columns)


def iterar_lotes_comprobantes(batch_size: int, path: Path = COMPROBANTES_PATH) -> Iterator[pd.DataFrame]:
    """Yield the invoice history in fixed-size batches without loading it all at once."""

    _validar_dataset(path)
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def leer_rucs_proveedores(path: Path = COMPROBANTES_PATH) -> set[str]:
    """Return the unique normalized supplier RUCs in the invoice history."""

    df = leer_comprobantes(columns=["RUC_Proveedor"], path=path)
    return set(df["RUC_Proveedor"].map(normalizar_ruc).dropna())
