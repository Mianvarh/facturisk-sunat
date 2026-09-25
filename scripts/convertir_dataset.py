"""Convierte el CSV historico original al dataset publico en Parquet.

Se ejecuta una sola vez sobre el archivo original del proyecto academico:

    python scripts/convertir_dataset.py "ruta/al/Datos PreRegistro ALINEADO.csv"

Cambios aplicados para la version publica:
- Se eliminan la razon social del proveedor y el nombre del cliente.
- Se corrige el desalineamiento de columnas del CSV original: la columna
  "Proveedor" contenia el RUC y "Ruc Proveedor" contenia la razon social.
- Se tipan fechas, importes y flags, y se normalizan nombres de columnas a ASCII.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "comprobantes.parquet"

COLUMNAS_ELIMINADAS = ["Ruc Proveedor", "Cliente"]
RENOMBRES = {
    "Proveedor": "RUC_Proveedor",
    "Año": "Anio",
    "Año_Mes": "Anio_Mes",
    "Orden_Año_Mes": "Orden_Anio_Mes",
}
DATE_COLUMNS = ["Fecha_Creacion", "Fecha_Emision", "Fecha_Vencimiento", "Fecha_Pago"]
FLAG_COLUMNS = [
    "Flag_Factura",
    "Flag_Emitida",
    "Flag_Aceptada",
    "Flag_Observada",
    "Flag_Rechazada",
    "Flag_Pendiente_Pago",
]


def normalizar_ruc(value: object) -> str | None:
    """Normalize a RUC-like value as an 11-digit string."""

    if pd.isna(value):
        return None
    digits = re.sub(r"\D", "", str(value).strip())
    if 0 < len(digits) <= 11:
        return digits.zfill(11)
    return None


def convertir(csv_path: Path, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Read the original CSV and write the anonymized, typed Parquet dataset."""

    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", dtype=str, low_memory=False)
    df = df.drop(columns=COLUMNAS_ELIMINADAS).rename(columns=RENOMBRES)

    df["RUC_Proveedor"] = df["RUC_Proveedor"].map(normalizar_ruc).astype("string")
    for column in DATE_COLUMNS:
        df[column] = pd.to_datetime(df[column], dayfirst=True, errors="coerce")
    df["Importe_Total"] = pd.to_numeric(df["Importe_Total"], errors="coerce")
    for column in FLAG_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int8")
    for column in ["Anio", "Mes_Num", "Orden_Anio_Mes"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int32")
    text_columns = df.columns.difference([*DATE_COLUMNS, "Importe_Total", *FLAG_COLUMNS, "Anio", "Mes_Num", "Orden_Anio_Mes"])
    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="zstd")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Convierte el CSV historico original a Parquet anonimizado.")
    parser.add_argument("csv_path", type=Path, help="Ruta al CSV original separado por punto y coma.")
    args = parser.parse_args()
    df = convertir(args.csv_path)
    print(f"Filas: {len(df)} | Columnas: {len(df.columns)}")
    print(f"RUC unicos: {df['RUC_Proveedor'].nunique()}")
    print(f"Archivo generado: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
