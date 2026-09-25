"""Inspect the enterprise invoice dataset before building the ML pipeline.

This script only reads the dataset. Derived columns are computed in memory for
inspection purposes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from datos import COMPROBANTES_PATH, leer_comprobantes
from paths import ensure_directories, get_application_root

PROJECT_ROOT = get_application_root()
DEFAULT_DATASET_PATH = COMPROBANTES_PATH
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "inspeccion_dataset.txt"

DATE_COLUMNS = [
    "Fecha_Creacion",
    "Fecha_Emision",
    "Fecha_Vencimiento",
    "Fecha_Pago",
]


def read_dataset(dataset_path: Path) -> pd.DataFrame:
    """Read the typed Parquet dataset."""

    return leer_comprobantes(path=dataset_path)


def append_section(lines: list[str], title: str) -> None:
    """Append a visual section separator to the report."""

    lines.extend(["", "=" * 80, title, "=" * 80])


def format_series(series: pd.Series) -> list[str]:
    """Format a pandas Series into aligned text lines."""

    return [f"{index}: {value}" for index, value in series.items()]


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Faltan columnas requeridas: {missing}")


def inspect_dataset(df_original: pd.DataFrame, dataset_path: Path) -> str:
    """Create a text report with all requested dataset checks."""

    validate_required_columns(
        df_original,
        [
            "RUC_Proveedor",
            "Estado_Aceptacion",
            "Importe_Total",
            "ID_Comprobante",
            *DATE_COLUMNS,
        ],
    )

    lines: list[str] = []

    append_section(lines, "Archivo")
    lines.append(f"Ruta dataset: {dataset_path}")
    lines.append("Formato: Parquet (tipos de datos preservados)")

    append_section(lines, "Cantidad de filas y columnas")
    rows, columns = df_original.shape
    lines.append(f"Filas: {rows}")
    lines.append(f"Columnas: {columns}")

    append_section(lines, "Nombres y tipos de columnas")
    for column, dtype in df_original.dtypes.items():
        lines.append(f"{column}: {dtype}")

    df = df_original.copy()

    append_section(lines, "Validacion de RUC_Proveedor")
    ruc_series = df["RUC_Proveedor"].astype("string").str.strip()
    ruc_valid_mask = ruc_series.str.fullmatch(r"\d{11}").fillna(False)
    invalid_ruc_count = int((~ruc_valid_mask).sum())
    valid_ruc_count = int(ruc_valid_mask.sum())
    unique_ruc_count = int(ruc_series[ruc_valid_mask].nunique(dropna=True))
    lines.append(f"RUC validos con 11 digitos: {valid_ruc_count}")
    lines.append(f"RUC invalidos o vacios: {invalid_ruc_count}")
    lines.append(f"Cantidad de RUC unicos validos: {unique_ruc_count}")
    if invalid_ruc_count:
        invalid_examples = ruc_series[~ruc_valid_mask].dropna().head(20).tolist()
        lines.append(f"Ejemplos de RUC invalidos: {invalid_examples}")

    append_section(lines, "Distribucion exacta de Estado_Aceptacion")
    state_distribution = df["Estado_Aceptacion"].value_counts(dropna=False)
    lines.extend(format_series(state_distribution))

    append_section(lines, "Valores nulos por columna")
    null_counts = df.isna().sum()
    lines.extend(format_series(null_counts))

    append_section(lines, "Conversion y rango de fechas")
    parsed_dates: dict[str, pd.Series] = {}
    impossible_date_rows: dict[str, int] = {}
    for column in DATE_COLUMNS:
        parsed = pd.to_datetime(df[column], errors="coerce")
        parsed_dates[column] = parsed
        non_null_original = df[column].notna()
        unparsable_count = int((non_null_original & parsed.isna()).sum())
        impossible_date_rows[column] = unparsable_count

        if parsed.notna().any():
            min_date = parsed.min().date().isoformat()
            max_date = parsed.max().date().isoformat()
        else:
            min_date = "sin fechas validas"
            max_date = "sin fechas validas"

        lines.append(f"{column}:")
        lines.append(f"  Minimo: {min_date}")
        lines.append(f"  Maximo: {max_date}")
        lines.append(f"  Valores no convertibles: {unparsable_count}")

    append_section(lines, "Estadisticas de Importe_Total")
    amount = pd.to_numeric(df["Importe_Total"], errors="coerce")
    amount_stats = amount.describe()
    lines.extend(format_series(amount_stats))
    lines.append(f"Valores no numericos en Importe_Total: {int((df['Importe_Total'].notna() & amount.isna()).sum())}")

    append_section(lines, "Calculo de plazo_dias")
    df["Fecha_Emision_parseada"] = parsed_dates["Fecha_Emision"]
    df["Fecha_Vencimiento_parseada"] = parsed_dates["Fecha_Vencimiento"]
    df["plazo_dias"] = (
        df["Fecha_Vencimiento_parseada"] - df["Fecha_Emision_parseada"]
    ).dt.days
    plazo_stats = df["plazo_dias"].describe()
    lines.extend(format_series(plazo_stats))
    lines.append(f"plazo_dias nulos: {int(df['plazo_dias'].isna().sum())}")

    append_section(lines, "Deteccion de problemas de calidad")
    negative_amounts = df[amount < 0]
    duplicate_ids = df[df["ID_Comprobante"].duplicated(keep=False)]
    invalid_date_total = sum(impossible_date_rows.values())
    negative_plazo = df[df["plazo_dias"].notna() & (df["plazo_dias"] < 0)]

    min_reasonable_date = pd.Timestamp("1990-01-01")
    max_reasonable_date = pd.Timestamp("2100-12-31")
    out_of_range_counts = {
        column: int(((parsed < min_reasonable_date) | (parsed > max_reasonable_date)).sum())
        for column, parsed in parsed_dates.items()
    }
    out_of_range_total = sum(out_of_range_counts.values())

    lines.append(f"Importes negativos: {len(negative_amounts)}")
    if len(negative_amounts):
        lines.append(
            "Ejemplos ID_Comprobante con importe negativo: "
            f"{negative_amounts['ID_Comprobante'].head(20).tolist()}"
        )

    lines.append(f"Fechas no convertibles: {invalid_date_total}")
    for column, count in impossible_date_rows.items():
        lines.append(f"  {column}: {count}")

    lines.append(f"Fechas fuera de rango 1990-01-01 a 2100-12-31: {out_of_range_total}")
    for column, count in out_of_range_counts.items():
        lines.append(f"  {column}: {count}")

    lines.append(f"Registros con Fecha_Vencimiento menor que Fecha_Emision: {len(negative_plazo)}")
    if len(negative_plazo):
        lines.append(
            "Ejemplos ID_Comprobante con plazo negativo: "
            f"{negative_plazo['ID_Comprobante'].head(20).tolist()}"
        )

    duplicated_unique_ids = int(duplicate_ids["ID_Comprobante"].nunique(dropna=True))
    lines.append(f"Filas con ID_Comprobante duplicado: {len(duplicate_ids)}")
    lines.append(f"ID_Comprobante duplicados unicos: {duplicated_unique_ids}")
    if len(duplicate_ids):
        duplicate_examples = (
            duplicate_ids["ID_Comprobante"]
            .value_counts(dropna=False)
            .head(20)
        )
        lines.append("Top ID_Comprobante duplicados:")
        lines.extend(f"  {index}: {value}" for index, value in duplicate_examples.items())

    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Inspecciona el dataset empresarial.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the inspection and write the text report."""

    ensure_directories()
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    report_path = args.report_path.expanduser().resolve()

    df = read_dataset(dataset_path)
    report = inspect_dataset(df, dataset_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Reporte guardado en: {report_path}")


if __name__ == "__main__":
    main()
