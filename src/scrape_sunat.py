"""Download and filter SUNAT reduced registry data for enterprise suppliers."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes

from datos import COMPROBANTES_PATH, leer_rucs_proveedores
from paths import ensure_directories, get_application_root

PROJECT_ROOT = get_application_root()
SUNAT_DIR = PROJECT_ROOT / "data" / "raw" / "sunat"
SUNAT_EXTRACT_DIR = SUNAT_DIR / "extraido"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SUNAT_PAGE_URL = "https://www.sunat.gob.pe/descargaPRR/mrc137_padron_reducido.html"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
REQUEST_TIMEOUT = 60
RETRIES = 3
CHUNK_SIZE = 200_000
SUNAT_ZIP_MAX_AGE_DAYS = 7

FUENTE_SUNAT = "SUNAT - Padrón Reducido"

EXPECTED_POSITIONAL_COLUMNS = [
    "RUC",
    "Razon_Social_SUNAT",
    "Estado_RUC",
    "Condicion_Domicilio",
    "Ubigeo",
    "Tipo_Via",
    "Nombre_Via",
    "Codigo_Zona",
    "Tipo_Zona",
    "Numero",
    "Interior",
    "Lote",
    "Departamento",
    "Manzana",
    "Kilometro",
]


@dataclass(frozen=True)
class PadronFormat:
    """Detected format metadata for the SUNAT registry file."""

    file_path: Path
    encoding: str
    separator: str
    column_count: int
    has_header: bool


def configurar_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalizar_ruc(value: object) -> str | None:
    """Normalize a RUC value as an 11-digit string."""

    if pd.isna(value):
        return None
    digits = re.sub(r"\D", "", str(value).strip())
    if len(digits) == 11:
        return digits
    if 0 < len(digits) < 11:
        padded = digits.zfill(11)
        return padded if len(padded) == 11 else None
    return None


def leer_rucs_empresariales(dataset_path: Path = COMPROBANTES_PATH) -> set[str]:
    """Read unique supplier RUCs from the enterprise dataset."""

    logging.info("Leyendo RUC unicos del dataset empresarial: %s", dataset_path)
    rucs = leer_rucs_proveedores(dataset_path)
    if not rucs:
        raise ValueError("No se encontraron RUC validos en la columna RUC_Proveedor.")
    logging.info("RUC unicos normalizados: %s", len(rucs))
    return rucs


def request_with_retries(url: str, *, stream: bool = False) -> requests.Response:
    """GET a URL with timeout, retries, User-Agent and status validation."""

    headers = {"User-Agent": USER_AGENT}
    last_error: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
                stream=stream,
            )
            if response.status_code != 200:
                raise requests.HTTPError(
                    f"Status HTTP {response.status_code} para {url}",
                    response=response,
                )
            return response
        except requests.RequestException as exc:
            last_error = exc
            logging.warning("Intento %s/%s fallido para %s: %s", attempt, RETRIES, url, exc)
            if attempt < RETRIES:
                time.sleep(2 * attempt)

    raise RuntimeError(f"No se pudo descargar {url}") from last_error


def encontrar_url_zip(page_url: str = SUNAT_PAGE_URL) -> str:
    """Find the current ZIP link in the official SUNAT page."""

    logging.info("Descargando HTML SUNAT: %s", page_url)
    response = request_with_retries(page_url)
    soup = BeautifulSoup(response.text, "html.parser")

    candidates: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = str(tag["href"]).strip()
        text = tag.get_text(" ", strip=True)
        joined = urljoin(page_url, href)
        candidate_text = f"{href} {text}".lower()
        if ".zip" in joined.lower() or ".zip" in candidate_text:
            candidates.append(joined)

    if not candidates:
        for match in re.findall(r"""['"]?([^'"\s<>]+\.zip)['"]?""", response.text, flags=re.I):
            candidates.append(urljoin(page_url, match))

    if not candidates:
        raise RuntimeError("No se encontro un enlace ZIP en la pagina de SUNAT.")

    padron_candidates = [
        url for url in candidates if any(term in url.lower() for term in ["padron", "reduc"])
    ]
    selected = padron_candidates[0] if padron_candidates else candidates[0]
    logging.info("ZIP SUNAT detectado: %s", selected)
    return selected


def zip_descargado_hoy(zip_path: Path) -> bool:
    """Return True when the ZIP exists and was modified today."""

    if not zip_path.exists():
        return False
    modified_date = datetime.fromtimestamp(zip_path.stat().st_mtime).date()
    return modified_date == datetime.now().date()


def zip_vigente(zip_path: Path, max_age_days: float) -> bool:
    """Return True when a valid ZIP is younger than the configured age."""

    if max_age_days < 0:
        raise ValueError("max_age_days debe ser mayor o igual a cero.")
    if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
        return False
    age_seconds = datetime.now().timestamp() - zip_path.stat().st_mtime
    return age_seconds <= max_age_days * 24 * 60 * 60


def descargar_zip(
    zip_url: str,
    output_dir: Path = SUNAT_DIR,
    *,
    force_download: bool = False,
    max_age_days: float = SUNAT_ZIP_MAX_AGE_DAYS,
) -> Path:
    """Download the SUNAT ZIP and validate that it is a real ZIP file."""

    if max_age_days < 0:
        raise ValueError("max_age_days debe ser mayor o igual a cero.")
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_name = Path(zip_url.split("?")[0]).name or "padron_reducido_sunat.zip"
    zip_path = output_dir / zip_name

    if not force_download and zip_vigente(zip_path, max_age_days):
        logging.info("Reutilizando ZIP SUNAT de menos de %.1f dias: %s", max_age_days, zip_path)
        return zip_path

    if zip_path.exists() and not zipfile.is_zipfile(zip_path):
        logging.warning("El ZIP SUNAT existente no es valido; se descargara de nuevo: %s", zip_path)

    logging.info("Descargando ZIP SUNAT en: %s", zip_path)
    response = request_with_retries(zip_url, stream=True)
    with zip_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)

    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile(f"El archivo descargado no es un ZIP valido: {zip_path}")

    logging.info("ZIP validado correctamente: %s", zip_path)
    return zip_path


def descomprimir_zip(zip_path: Path, extract_dir: Path = SUNAT_EXTRACT_DIR) -> list[Path]:
    """Extract the ZIP file and return extracted files."""

    extract_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Descomprimiendo ZIP en: %s", extract_dir)
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(extract_dir)

    files = [path for path in extract_dir.rglob("*") if path.is_file()]
    if not files:
        raise FileNotFoundError("El ZIP no contiene archivos extraidos.")
    return files


def detectar_archivo_principal(files: Iterable[Path]) -> Path:
    """Detect the main registry file, preferring the largest text-like file."""

    text_extensions = {".txt", ".csv", ".dat"}
    candidates = [path for path in files if path.suffix.lower() in text_extensions]
    if not candidates:
        candidates = list(files)
    main_file = max(candidates, key=lambda path: path.stat().st_size)
    logging.info("Archivo principal detectado: %s", main_file)
    return main_file


def obtener_archivos_extraidos(zip_path: Path, extract_dir: Path = SUNAT_EXTRACT_DIR) -> list[Path]:
    """Reuse extracted files when the current main file is newer than the ZIP."""

    existing_files = [path for path in extract_dir.rglob("*") if path.is_file()] if extract_dir.exists() else []
    if existing_files:
        main_file = detectar_archivo_principal(existing_files)
        if main_file.stat().st_mtime > zip_path.stat().st_mtime:
            logging.info("Reutilizando archivos SUNAT extraidos: %s", extract_dir)
            return existing_files

    return descomprimir_zip(zip_path, extract_dir)


def detectar_encoding(file_path: Path, sample_size: int = 2_000_000) -> str:
    """Detect file encoding from a byte sample."""

    with file_path.open("rb") as file:
        sample = file.read(sample_size)
    expected_terms = ["RAZON", "RAZÓN", "CONDICION", "CONDICIÓN", "RUC"]
    candidate_scores: list[tuple[int, str]] = []
    for candidate in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        decoded = sample.decode(candidate, errors="replace")
        replacement_count = decoded.count("\ufffd")
        header = decoded.splitlines()[0].upper() if decoded.splitlines() else ""
        term_score = sum(1 for term in expected_terms if term in header)
        candidate_scores.append((term_score * 1000 - replacement_count, candidate))

    best_score, best_candidate = max(candidate_scores, key=lambda item: item[0])
    if best_score > 0:
        logging.info("Encoding detectado por encabezado: %s", best_candidate)
        return best_candidate

    result = from_bytes(sample).best()
    if result is None or not result.encoding:
        logging.warning("No se pudo detectar encoding; usando latin-1.")
        return "latin-1"
    logging.info("Encoding detectado: %s", result.encoding)
    return result.encoding


def detectar_separador_y_columnas(file_path: Path, encoding: str) -> tuple[str, int]:
    """Detect delimiter and number of columns from a file sample."""

    with file_path.open("rb") as file:
        sample_text = file.read(200_000).decode(encoding, errors="replace")
    lines = [line for line in sample_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("El archivo principal SUNAT esta vacio.")

    separators = ["|", ";", "\t", ","]
    best_separator = "|"
    best_columns = 0
    best_score = -1
    for separator in separators:
        counts = [len(line.split(separator)) for line in lines[:100]]
        score = max(set(counts), key=counts.count)
        if score > best_score:
            best_score = score
            best_separator = separator
            best_columns = score

    logging.info("Separador detectado: %r con %s columnas", best_separator, best_columns)
    return best_separator, best_columns


def normalizar_nombre_columna(column: object) -> str:
    """Normalize a column name for matching."""

    text = str(column).strip().upper()
    replacements = {
        "Á": "A",
        "É": "E",
        "Í": "I",
        "Ó": "O",
        "Ú": "U",
        "Ñ": "N",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"[^A-Z0-9]+", "_", text).strip("_")


def detectar_header(file_path: Path, encoding: str, separator: str) -> bool:
    """Detect whether the first row looks like a header."""

    first_line = file_path.open("r", encoding=encoding, errors="replace").readline()
    first_values = [value.strip() for value in first_line.split(separator)]
    normalized = [normalizar_nombre_columna(value) for value in first_values]
    header_terms = {"RUC", "RAZON_SOCIAL", "ESTADO", "CONDICION", "UBIGEO"}
    has_header = any(value in header_terms or "RAZON" in value for value in normalized)
    logging.info("Encabezado detectado: %s", has_header)
    return has_header


def detectar_formato_padron(file_path: Path) -> PadronFormat:
    """Detect encoding, separator, number of columns and header presence."""

    encoding = detectar_encoding(file_path)
    separator, column_count = detectar_separador_y_columnas(file_path, encoding)
    has_header = detectar_header(file_path, encoding, separator)
    return PadronFormat(
        file_path=file_path,
        encoding=encoding,
        separator=separator,
        column_count=column_count,
        has_header=has_header,
    )


def build_column_names(column_count: int) -> list[str]:
    """Build column names for a headerless SUNAT file."""

    if column_count <= len(EXPECTED_POSITIONAL_COLUMNS):
        return EXPECTED_POSITIONAL_COLUMNS[:column_count]
    extra_columns = [f"Columna_{index}" for index in range(len(EXPECTED_POSITIONAL_COLUMNS), column_count)]
    return [*EXPECTED_POSITIONAL_COLUMNS, *extra_columns]


def find_column(columns: Iterable[object], candidates: list[str]) -> object | None:
    """Find a column by normalized-name candidates."""

    normalized_map = {normalizar_nombre_columna(column): column for column in columns}
    for candidate in candidates:
        normalized_candidate = normalizar_nombre_columna(candidate)
        if normalized_candidate in normalized_map:
            return normalized_map[normalized_candidate]

    for normalized, original in normalized_map.items():
        if any(normalizar_nombre_columna(candidate) in normalized for candidate in candidates):
            return original

    return None


def domicilio_from_chunk(chunk: pd.DataFrame) -> pd.Series:
    """Create Domicilio_Fiscal from available address columns."""

    direct = find_column(chunk.columns, ["Domicilio_Fiscal", "Domicilio Fiscal", "Direccion", "Direccion Fiscal"])
    if direct is not None:
        return chunk[direct].astype("string").fillna("").str.strip()

    address_candidates = [
        ["Tipo_Via", "Tipo de Via", "Tipo de Vía"],
        ["Nombre_Via", "Nombre de Via", "Nombre de Vía"],
        ["Numero", "Número", "Nro"],
        ["Interior"],
        ["Lote"],
        ["Departamento"],
        ["Manzana"],
        ["Kilometro", "Kilómetro"],
        ["Tipo_Zona", "Tipo de Zona"],
        ["Codigo_Zona", "Código de Zona", "Codigo de Zona"],
    ]
    address_columns = []
    for candidates in address_candidates:
        column = find_column(chunk.columns, candidates)
        if column is not None:
            address_columns.append(column)

    if not address_columns:
        return pd.Series([""] * len(chunk), index=chunk.index, dtype="string")

    address_frame = chunk[address_columns].fillna("").astype(str)
    address_frame = address_frame.map(lambda value: "" if value.strip() == "-" else value.strip())
    address = address_frame.agg(" ".join, axis=1)
    return address.str.replace(r"\s+", " ", regex=True).str.strip()


def normalizar_texto(value: object) -> str:
    """Normalize a text value for output and comparisons."""

    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def transformar_chunk_sunat(chunk: pd.DataFrame, rucs_buscados: set[str]) -> pd.DataFrame:
    """Filter and normalize one SUNAT chunk."""

    ruc_column = find_column(chunk.columns, ["RUC"])
    if ruc_column is None:
        raise ValueError("No se pudo identificar la columna RUC en el padron SUNAT.")

    chunk = chunk.copy()
    chunk["RUC_NORMALIZADO"] = chunk[ruc_column].map(normalizar_ruc)
    chunk = chunk[chunk["RUC_NORMALIZADO"].isin(rucs_buscados)]
    if chunk.empty:
        return pd.DataFrame()

    razon_column = find_column(chunk.columns, ["Razon_Social_SUNAT", "Razon Social", "Nombre o Razon Social"])
    estado_column = find_column(chunk.columns, ["Estado_RUC", "Estado", "Estado del Contribuyente"])
    condicion_column = find_column(chunk.columns, ["Condicion_Domicilio", "Condicion de Domicilio", "Condicion"])
    ubigeo_column = find_column(chunk.columns, ["Ubigeo"])

    required = {
        "Razon_Social_SUNAT": razon_column,
        "Estado_RUC": estado_column,
        "Condicion_Domicilio": condicion_column,
        "Ubigeo": ubigeo_column,
    }
    missing = [name for name, column in required.items() if column is None]
    if missing:
        raise ValueError(f"No se pudieron identificar columnas SUNAT: {', '.join(missing)}")

    result = pd.DataFrame(
        {
            "RUC": chunk["RUC_NORMALIZADO"],
            "Razon_Social_SUNAT": chunk[razon_column].fillna("").astype("string").str.strip(),
            "Estado_RUC": chunk[estado_column].map(normalizar_texto),
            "Condicion_Domicilio": chunk[condicion_column].map(normalizar_texto),
            "Ubigeo": chunk[ubigeo_column].fillna("").astype("string").str.strip(),
            "Domicilio_Fiscal": domicilio_from_chunk(chunk),
        }
    )
    result["Situacion_Tributaria_Actual"] = (
        ~(
            (result["Estado_RUC"] == "ACTIVO")
            & (result["Condicion_Domicilio"] == "HABIDO")
        )
    ).astype(int)
    return result.drop_duplicates(subset=["RUC"], keep="last")


def leer_y_filtrar_padron(format_info: PadronFormat, rucs_buscados: set[str]) -> pd.DataFrame:
    """Read the SUNAT registry by chunks and keep only enterprise supplier RUCs."""

    logging.info("Leyendo padron por chunks sin cargarlo completo en memoria.")
    read_kwargs = {
        "filepath_or_buffer": format_info.file_path,
        "sep": format_info.separator,
        "encoding": format_info.encoding,
        "dtype": "string",
        "chunksize": CHUNK_SIZE,
        "on_bad_lines": "skip",
        "low_memory": False,
    }

    if format_info.has_header:
        reader = pd.read_csv(**read_kwargs)
    else:
        reader = pd.read_csv(
            **read_kwargs,
            header=None,
            names=build_column_names(format_info.column_count),
        )

    matches: list[pd.DataFrame] = []
    encontrados: set[str] = set()
    for index, chunk in enumerate(reader, start=1):
        filtered = transformar_chunk_sunat(chunk, rucs_buscados - encontrados)
        if not filtered.empty:
            matches.append(filtered)
            encontrados.update(filtered["RUC"].dropna().tolist())
        logging.info("Chunk %s procesado. RUC encontrados acumulados: %s", index, len(encontrados))
        if encontrados == rucs_buscados:
            logging.info("Todos los RUC fueron encontrados; fin de lectura anticipado.")
            break

    if not matches:
        return pd.DataFrame(
            columns=[
                "RUC",
                "Razon_Social_SUNAT",
                "Estado_RUC",
                "Condicion_Domicilio",
                "Ubigeo",
                "Domicilio_Fiscal",
                "Situacion_Tributaria_Actual",
            ]
        )

    return pd.concat(matches, ignore_index=True).drop_duplicates(subset=["RUC"], keep="last")


def guardar_resultados(
    proveedores: pd.DataFrame,
    rucs_buscados: set[str],
    fecha_consulta: str,
    zip_path: Path,
) -> None:
    """Save filtered providers, missing RUCs and scraping report."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    proveedores = proveedores.copy()
    proveedores["Fecha_Consulta"] = fecha_consulta
    proveedores["Fuente"] = FUENTE_SUNAT
    proveedores = proveedores[
        [
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
    ].sort_values("RUC")

    found_rucs = set(proveedores["RUC"].dropna().tolist())
    missing_rucs = sorted(rucs_buscados - found_rucs)
    no_encontrados = pd.DataFrame({"RUC": missing_rucs})

    proveedores_path = PROCESSED_DIR / "proveedores_sunat.csv"
    no_encontrados_path = PROCESSED_DIR / "ruc_no_encontrados.csv"
    reporte_path = OUTPUTS_DIR / "reporte_scraping.json"

    proveedores.to_csv(proveedores_path, index=False, encoding="utf-8-sig")
    no_encontrados.to_csv(no_encontrados_path, index=False, encoding="utf-8-sig")

    report = {
        "total_ruc_buscados": len(rucs_buscados),
        "total_encontrados": len(found_rucs),
        "total_no_encontrados": len(missing_rucs),
        "regulares": int((proveedores["Situacion_Tributaria_Actual"] == 0).sum()),
        "irregulares": int((proveedores["Situacion_Tributaria_Actual"] == 1).sum()),
        "fecha_consulta": fecha_consulta,
        "ruta_zip_descargado": str(zip_path),
    }
    reporte_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info("Archivo proveedores SUNAT guardado: %s", proveedores_path)
    logging.info("Archivo RUC no encontrados guardado: %s", no_encontrados_path)
    logging.info("Reporte scraping guardado: %s", reporte_path)


def obtener_datos_sunat(
    *,
    force_download: bool = False,
    max_age_days: float = SUNAT_ZIP_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """Download, extract, parse and filter SUNAT data for enterprise RUCs."""

    ensure_directories()
    configurar_logging()
    fecha_consulta = datetime.now().date().isoformat()

    rucs_buscados = leer_rucs_empresariales()
    zip_url = encontrar_url_zip()
    zip_path = descargar_zip(zip_url, force_download=force_download, max_age_days=max_age_days)
    extracted_files = obtener_archivos_extraidos(zip_path)
    main_file = detectar_archivo_principal(extracted_files)
    format_info = detectar_formato_padron(main_file)

    logging.info(
        "Formato detectado: archivo=%s encoding=%s separador=%r columnas=%s header=%s",
        format_info.file_path,
        format_info.encoding,
        format_info.separator,
        format_info.column_count,
        format_info.has_header,
    )
    proveedores = leer_y_filtrar_padron(format_info, rucs_buscados)
    guardar_resultados(proveedores, rucs_buscados, fecha_consulta, zip_path)

    logging.info(
        "Proceso terminado. Buscados=%s Encontrados=%s No encontrados=%s",
        len(rucs_buscados),
        proveedores["RUC"].nunique() if "RUC" in proveedores.columns else 0,
        len(rucs_buscados - set(proveedores.get("RUC", pd.Series(dtype='string')).dropna().tolist())),
    )
    return proveedores


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description="Descarga y filtra el padron reducido SUNAT.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Vuelve a descargar el ZIP SUNAT aunque exista una copia vigente.",
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=SUNAT_ZIP_MAX_AGE_DAYS,
        help=f"Edad maxima del ZIP reutilizable en dias (por defecto: {SUNAT_ZIP_MAX_AGE_DAYS}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    obtener_datos_sunat(force_download=args.force_download, max_age_days=args.max_age_days)
