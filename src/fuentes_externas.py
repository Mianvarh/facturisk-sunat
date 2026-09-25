"""Extra external variables from official SUNAT registries (padrones).

Each registry is a ZIP with a pipe-separated TXT (records separated by CR):
Ruc|Nombre/Razon|A partir del|Resolucion|. Membership of each supplier RUC in
a registry becomes a 0/1 feature for the model.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from configuracion import PADRON_COLUMNS, PADRONES_SUNAT, cargar_ajustes
from datos import PADRONES_BACKUP_CSV, PADRONES_SNAPSHOT_PATH, PROJECT_ROOT, normalizar_ruc

PADRONES_DIR = PROJECT_ROOT / "data" / "raw" / "sunat" / "padrones"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FactuRisk-SUNAT"
TIMEOUT = 60


def _zip_path(key: str) -> Path:
    return PADRONES_DIR / f"{key}.zip"


def _recent(path: Path, max_age_days: float) -> bool:
    return path.exists() and zipfile.is_zipfile(path) and (time.time() - path.stat().st_mtime) <= max_age_days * 86_400


def descargar_padron(key: str, *, max_age_days: float = 7, force: bool = False) -> Path:
    """Download one registry ZIP, reusing a recent valid copy."""

    info = PADRONES_SUNAT[key]
    path = _zip_path(key)
    if not force and _recent(path, max_age_days):
        logging.info("Reutilizando padrón %s: %s", info["nombre"], path)
        return path
    PADRONES_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Descargando padrón %s", info["nombre"])
    response = requests.get(info["zip_url"], headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    if not zipfile.is_zipfile(io.BytesIO(response.content)):
        raise ValueError(f"La descarga de {info['nombre']} no es un ZIP válido.")
    path.write_bytes(response.content)
    return path


def leer_padron(zip_path: Path) -> pd.DataFrame:
    """Parse a registry ZIP into RUC, name and start date."""

    with zipfile.ZipFile(zip_path) as archive:
        name = next(member for member in archive.namelist() if member.lower().endswith(".txt"))
        text = archive.read(name).decode("latin-1")
    rows = []
    for line in text.replace("\r\n", "\r").replace("\n", "\r").split("\r")[1:]:
        parts = line.split("|")
        if len(parts) >= 3 and parts[0].strip():
            rows.append((parts[0].strip(), parts[1].strip().strip('"'), parts[2].strip()))
    frame = pd.DataFrame(rows, columns=["RUC", "Razon_Social", "Desde"])
    frame["RUC"] = frame["RUC"].map(normalizar_ruc)
    return frame.dropna(subset=["RUC"]).drop_duplicates("RUC")


def construir_variables_padrones(rucs: set[str], *, force: bool = False) -> pd.DataFrame:
    """Download the enabled registries and flag each supplier RUC."""

    settings = cargar_ajustes()["sunat"]
    enabled = [key for key, active in settings["padrones"].items() if active and key in PADRONES_SUNAT]
    result = pd.DataFrame({"RUC": sorted(rucs)})
    for key in PADRONES_SUNAT:
        result[PADRONES_SUNAT[key]["columna"]] = 0
    report: dict[str, int] = {}
    for key in enabled:
        info = PADRONES_SUNAT[key]
        try:
            padron = leer_padron(descargar_padron(key, max_age_days=float(settings.get("max_age_days", 7)), force=force))
        except (requests.RequestException, ValueError, zipfile.BadZipFile, StopIteration) as exc:
            logging.warning("No se pudo obtener %s: %s", info["nombre"], exc)
            continue
        members = set(padron["RUC"])
        result[info["columna"]] = result["RUC"].isin(members).astype(int)
        report[info["nombre"]] = int(result[info["columna"]].sum())
        logging.info("%s: %s proveedores del dataset figuran en el padrón", info["nombre"], report[info["nombre"]])
    result["Fecha_Consulta_Padrones"] = datetime.now().date().isoformat()
    PADRONES_BACKUP_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(PADRONES_BACKUP_CSV, index=False, encoding="utf-8-sig")
    print("\nPadrones SUNAT adicionales")
    if not enabled:
        print("Ningún padrón adicional habilitado en Configuración.")
    for name, count in report.items():
        print(f"  {name}: {count:,} proveedores del dataset")
    return result


def cargar_variables_padrones() -> pd.DataFrame | None:
    """Local registry flags, else the bundled snapshot; None when neither exists."""

    if PADRONES_BACKUP_CSV.exists():
        frame = pd.read_csv(PADRONES_BACKUP_CSV, encoding="utf-8-sig", dtype={"RUC": "string"})
    elif PADRONES_SNAPSHOT_PATH.exists():
        frame = pd.read_parquet(PADRONES_SNAPSHOT_PATH)
    else:
        return None
    frame["RUC"] = frame["RUC"].map(normalizar_ruc)
    for column in PADRON_COLUMNS:
        if column not in frame:
            frame[column] = 0
    return frame[["RUC", *PADRON_COLUMNS]].dropna(subset=["RUC"]).drop_duplicates("RUC")
