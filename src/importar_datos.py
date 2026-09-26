"""Import invoice history from a user CSV or Excel file into the pipeline format.

The file is read with automatic delimiter/encoding detection, columns are
matched by name and by content (e.g. the column that holds 11-digit RUCs), and
the result is saved as a local Parquet dataset (excluded from Git) that becomes
the active dataset of the pipeline.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from configuracion import actualizar_ajustes
from datos import (
    COMPROBANTES_PATH,
    DATE_COLUMNS,
    IMPORTED_DATASET_PATH,
    PROJECT_ROOT,
    normalizar_ruc,
)

REQUIRED = ["RUC_Proveedor", "Fecha_Emision", "Importe_Total", "Estado_Aceptacion"]
OPTIONAL_DEFAULTS = {
    "Tipo_Comprobante": "No especificado",
    "Moneda": "No especificado",
}

# Canonical column → accepted header aliases (compared after normalization).
ALIASES: dict[str, list[str]] = {
    "ID_Comprobante": ["id comprobante", "comprobante", "numero comprobante", "nro comprobante", "serie numero", "id", "documento"],
    "RUC_Proveedor": ["ruc proveedor", "ruc", "ruc emisor", "proveedor ruc", "nro ruc", "numero ruc", "proveedor"],
    "Razon_Social_Proveedor": ["razon social", "razon social proveedor", "nombre proveedor", "proveedor nombre", "ruc proveedor", "proveedor", "nombre", "emisor"],
    "Tipo_Comprobante": ["tipo comprobante", "tipo documento", "tipo", "tipo doc"],
    "Fecha_Creacion": ["fecha creacion", "fecha registro", "fecha carga"],
    "Fecha_Emision": ["fecha emision", "fecha", "fecha documento", "emision"],
    "Fecha_Vencimiento": ["fecha vencimiento", "vencimiento", "fecha vence"],
    "Fecha_Pago": ["fecha pago", "pago"],
    "Moneda": ["moneda", "divisa", "currency"],
    "Importe_Total": ["importe total", "importe", "monto", "monto total", "total", "valor"],
    "Estado_Aceptacion": ["estado aceptacion", "estado sunat", "estado cpe", "resultado", "estado"],
}

STATE_MAP = {
    "aceptada": "Aceptada",
    "aceptado": "Aceptada",
    "accepted": "Aceptada",
    "observada": "Observada",
    "observado": "Observada",
    "aceptada con observaciones": "Observada",
    "rechazada": "Rechazada",
    "rechazado": "Rechazada",
    "rejected": "Rechazada",
    "anulada": "Rechazada",
    "pendiente": "Pendiente / Borrador",
    "borrador": "Pendiente / Borrador",
    "pendiente / borrador": "Pendiente / Borrador",
    "en proceso": "Pendiente / Borrador",
    "por enviar": "Pendiente / Borrador",
}


@dataclass
class ImportReport:
    """What the import detected, for display in the panel."""

    archivo: str
    filas: int = 0
    columnas_detectadas: dict[str, str] = field(default_factory=dict)
    estados: dict[str, int] = field(default_factory=dict)
    advertencias: list[str] = field(default_factory=list)
    rango_fechas: tuple[str, str] | None = None
    rucs_unicos: int = 0
    con_razon_social: bool = False


def _normalize(text: object) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def leer_archivo(path: Path) -> pd.DataFrame:
    """Read CSV (any common delimiter/encoding) or Excel as text columns."""

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, dtype=str)
    if suffix == ".xls":
        raise ValueError("El formato .xls antiguo no está soportado: guarde el archivo como .xlsx o .csv.")
    if suffix not in {".csv", ".txt"}:
        raise ValueError(f"Formato no soportado: {suffix}. Use CSV o Excel (.xlsx).")
    raw = path.read_bytes()[:200_000]
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            sample = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=";,|\t").delimiter
    except csv.Error:
        delimiter = max(";,|\t", key=sample.count)
    return pd.read_csv(path, sep=delimiter, encoding=encoding, dtype=str, low_memory=False)


def _ruc_share(series: pd.Series) -> float:
    values = series.dropna().astype(str).str.strip().head(2000)
    if values.empty:
        return 0.0
    return float(values.str.replace(r"\D", "", regex=True).str.fullmatch(r"\d{11}").mean())


def _text_share(series: pd.Series) -> float:
    values = series.dropna().astype(str).str.strip().head(2000)
    if values.empty:
        return 0.0
    return float(values.str.contains(r"[A-Za-zÁÉÍÓÚÑ]{3,}").mean())


def detectar_columnas(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical columns to the file headers, by alias and by content."""

    normalized = {column: _normalize(column) for column in df.columns}
    mapping: dict[str, str] = {}

    # RUC: the column with most 11-digit values among RUC/provider-like headers (or any column).
    ruc_candidates = [c for c, n in normalized.items() if any(alias == n or alias in n for alias in ("ruc", "proveedor", "emisor"))]
    ruc_candidates = ruc_candidates or list(df.columns)
    shares = {column: _ruc_share(df[column]) for column in ruc_candidates}
    best = max(shares, key=shares.get) if shares else None
    if best is not None and shares[best] >= 0.6:
        mapping["RUC_Proveedor"] = best

    # Supplier name: a mostly-text column among provider/name headers, not the RUC one.
    name_candidates = [
        c
        for c, n in normalized.items()
        if c != mapping.get("RUC_Proveedor") and any(alias in n for alias in ("razon", "nombre", "proveedor", "emisor"))
    ]
    name_candidates = [c for c in name_candidates if _text_share(df[c]) >= 0.6 and _ruc_share(df[c]) < 0.2]
    if name_candidates:
        mapping["Razon_Social_Proveedor"] = name_candidates[0]

    used = set(mapping.values())
    for canonical, aliases in ALIASES.items():
        if canonical in mapping:
            continue
        exact = [c for c, n in normalized.items() if c not in used and n in aliases]
        partial = [c for c, n in normalized.items() if c not in used and any(alias in n for alias in aliases if len(alias) > 4)]
        chosen = (exact or partial or [None])[0]
        if canonical == "Estado_Aceptacion" and chosen is None:
            continue
        if chosen is not None:
            mapping[canonical] = chosen
            used.add(chosen)
    # Prefer an explicit acceptance-status column over a generic "Estado".
    for column, n in normalized.items():
        if "aceptacion" in n or "estado sunat" in n:
            mapping["Estado_Aceptacion"] = column
            break
    return mapping


def normalizar_estado(value: object) -> str:
    if pd.isna(value):
        return "Pendiente / Borrador"
    text = _normalize(value)
    for key, label in STATE_MAP.items():
        if text == _normalize(key):
            return label
    for key, label in STATE_MAP.items():
        if _normalize(key) in text:
            return label
    return str(value).strip()


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    if parsed.notna().mean() < 0.5:
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def _parse_amount(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(r"[^\d,.\-]", "", regex=True)
    # "1.234,56" → "1234.56"; "1,234.56" → "1234.56"
    comma_decimal = text.str.contains(r",\d{1,2}$")
    text = text.where(~comma_decimal, text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    text = text.where(comma_decimal, text.str.replace(",", "", regex=False))
    return pd.to_numeric(text, errors="coerce")


def transformar(df: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Build the pipeline dataset from the detected columns."""

    missing = [column for column in REQUIRED if column not in mapping]
    if missing:
        raise ValueError(
            "No se encontraron columnas obligatorias: "
            + ", ".join(missing)
            + ". Renombre las columnas del archivo (por ejemplo 'RUC Proveedor', 'Fecha Emision', 'Importe Total', 'Estado Aceptacion')."
        )
    warnings: list[str] = []
    result = pd.DataFrame(index=df.index)
    for canonical, source in mapping.items():
        result[canonical] = df[source]

    result["RUC_Proveedor"] = result["RUC_Proveedor"].map(normalizar_ruc).astype("string")
    invalid = int(result["RUC_Proveedor"].isna().sum())
    if invalid:
        warnings.append(f"{invalid:,} filas sin RUC válido fueron descartadas.")
        result = result.dropna(subset=["RUC_Proveedor"])

    for column in DATE_COLUMNS:
        result[column] = _parse_dates(result[column]) if column in result else pd.NaT
    if "Fecha_Vencimiento" not in mapping:
        warnings.append("Sin fecha de vencimiento: se usará la fecha de emisión (plazo 0 días).")
        result["Fecha_Vencimiento"] = result["Fecha_Emision"]
    bad_dates = int(result["Fecha_Emision"].isna().sum())
    if bad_dates:
        warnings.append(f"{bad_dates:,} filas con fecha de emisión no reconocida.")

    result["Importe_Total"] = _parse_amount(result["Importe_Total"])
    result["Estado_Aceptacion"] = result["Estado_Aceptacion"].map(normalizar_estado).astype("string")
    for column, default in OPTIONAL_DEFAULTS.items():
        if column not in result:
            result[column] = default
            warnings.append(f"Sin columna '{column}': se completó con '{default}'.")
        result[column] = result[column].fillna(default).astype("string").str.strip()
    if "ID_Comprobante" not in result:
        result["ID_Comprobante"] = [f"IMP-{index:08d}" for index in range(len(result))]
        warnings.append("Sin identificador de comprobante: se generaron IDs correlativos.")
    result["ID_Comprobante"] = result["ID_Comprobante"].astype("string")
    if "Razon_Social_Proveedor" in result:
        result["Razon_Social_Proveedor"] = result["Razon_Social_Proveedor"].astype("string").str.strip().str.strip("'\"")

    known = {"Aceptada", "Observada", "Rechazada", "Pendiente / Borrador"}
    unknown = sorted(set(result["Estado_Aceptacion"].dropna()) - known)
    if unknown:
        warnings.append("Estados no reconocidos (se tratan como no definitivos): " + ", ".join(unknown[:6]))
    definitive = result["Estado_Aceptacion"].isin(["Aceptada", "Observada", "Rechazada"]).sum()
    incidents = result["Estado_Aceptacion"].isin(["Observada", "Rechazada"]).sum()
    if definitive < 200 or incidents < 20:
        warnings.append("Hay pocos comprobantes con estado definitivo o pocas incidencias: el modelo puede no ser confiable.")
    return result.reset_index(drop=True), warnings


def importar(path: Path, output_path: Path = IMPORTED_DATASET_PATH) -> ImportReport:
    """Read, validate, convert and activate a user dataset."""

    path = Path(path)
    df = leer_archivo(path)
    mapping = detectar_columnas(df)
    result, warnings = transformar(df, mapping)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False, compression="zstd")
    relative = output_path.relative_to(PROJECT_ROOT) if output_path.is_relative_to(PROJECT_ROOT) else output_path
    actualizar_ajustes(dataset={"ruta": str(relative), "origen": "importado", "archivo_original": path.name, "importado_en": datetime.now().isoformat(timespec="seconds")})

    dates = result["Fecha_Emision"].dropna()
    return ImportReport(
        archivo=path.name,
        filas=len(result),
        columnas_detectadas={canonical: str(source) for canonical, source in mapping.items()},
        estados=result["Estado_Aceptacion"].value_counts().to_dict(),
        advertencias=warnings,
        rango_fechas=(f"{dates.min():%d/%m/%Y}", f"{dates.max():%d/%m/%Y}") if not dates.empty else None,
        rucs_unicos=int(result["RUC_Proveedor"].nunique()),
        con_razon_social="Razon_Social_Proveedor" in result,
    )


def usar_dataset_demo() -> None:
    """Switch back to the public demo dataset shipped with the repository."""

    actualizar_ajustes(dataset={"ruta": None, "origen": "demo", "archivo_original": None})


def resumen_dataset_activo() -> dict[str, object]:
    """Small description of the active dataset for the settings panel."""

    from configuracion import cargar_ajustes
    from datos import leer_comprobantes, ruta_comprobantes

    settings = cargar_ajustes()["dataset"]
    path = ruta_comprobantes()
    try:
        df = leer_comprobantes(columns=["RUC_Proveedor", "Fecha_Emision"], path=path)
        dates = pd.to_datetime(df["Fecha_Emision"], errors="coerce").dropna()
        es_demo = path == COMPROBANTES_PATH
        return {
            "origen": "Demostración" if es_demo else "Importado",
            "es_demo": es_demo,
            "archivo": settings.get("archivo_original") or path.name,
            "filas": len(df),
            "rucs": int(df["RUC_Proveedor"].nunique()),
            "desde": f"{dates.min():%d/%m/%Y}" if not dates.empty else "—",
            "hasta": f"{dates.max():%d/%m/%Y}" if not dates.empty else "—",
        }
    except (OSError, ValueError) as exc:
        return {"origen": "Error", "es_demo": True, "archivo": path.name, "filas": 0, "rucs": 0, "desde": "—", "hasta": str(exc)}
