"""Local settings edited from the desktop panel (never committed to Git).

- config/app_settings.json: active dataset and external SUNAT sources.
- config/configuracion.json: MongoDB connection (same keys as before).

MongoDB precedence: panel settings, then the .env file.
"""

from __future__ import annotations

import contextlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from paths import get_application_root

PROJECT_ROOT = get_application_root()
CONFIG_DIR = PROJECT_ROOT / "config"
APP_SETTINGS_PATH = CONFIG_DIR / "app_settings.json"
MONGO_SETTINGS_PATH = CONFIG_DIR / "configuracion.json"

SUNAT_PADRON_PAGE = "https://www.sunat.gob.pe/descargaPRR/mrc137_padron_reducido.html"

# Official SUNAT registries usable as extra external variables.
PADRONES_SUNAT: dict[str, dict[str, str]] = {
    "agentes_retencion": {
        "nombre": "Agentes de Retención del IGV",
        "descripcion": "Empresas designadas por SUNAT para retener IGV a sus proveedores.",
        "zip_url": "https://www.sunat.gob.pe/descarga/AgentRet/AgenRet_TXT.zip",
        "columna": "Es_Agente_Retencion",
    },
    "buenos_contribuyentes": {
        "nombre": "Buenos Contribuyentes",
        "descripcion": "Contribuyentes con cumplimiento tributario oportuno.",
        "zip_url": "https://www.sunat.gob.pe/descarga/BueCont/BueCont_TXT.zip",
        "columna": "Es_Buen_Contribuyente",
    },
    "agentes_percepcion": {
        "nombre": "Agentes de Percepción (combustibles)",
        "descripcion": "Agentes de percepción del IGV en la venta de combustibles.",
        "zip_url": "https://www.sunat.gob.pe/descarga/AgentRet/AgenPerc_TXT.zip",
        "columna": "Es_Agente_Percepcion",
    },
    "percepcion_venta_interna": {
        "nombre": "Agentes de Percepción (venta interna)",
        "descripcion": "Agentes de percepción del IGV en la venta interna de bienes.",
        "zip_url": "https://www.sunat.gob.pe/descarga/AgentRet/AgenPercVI_TXT.zip",
        "columna": "Es_Agente_Percepcion_VI",
    },
}
PADRON_COLUMNS = [info["columna"] for info in PADRONES_SUNAT.values()]

DEFAULT_SETTINGS: dict[str, Any] = {
    "dataset": {"ruta": None, "origen": "demo", "archivo_original": None},
    "sunat": {
        "pagina_padron": SUNAT_PADRON_PAGE,
        "zip_url": "",
        "archivo_local": "",
        "max_age_days": 7,
        "padrones": {key: key != "agentes_percepcion" for key in PADRONES_SUNAT},
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def cargar_ajustes() -> dict[str, Any]:
    """Settings with defaults for every missing key."""

    if APP_SETTINGS_PATH.exists():
        try:
            stored = json.loads(APP_SETTINGS_PATH.read_text(encoding="utf-8"))
            return _merge(DEFAULT_SETTINGS, stored)
        except (OSError, json.JSONDecodeError):
            pass
    return deepcopy(DEFAULT_SETTINGS)


def guardar_ajustes(settings: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    APP_SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def actualizar_ajustes(**sections: dict[str, Any]) -> dict[str, Any]:
    """Merge the given sections into the stored settings and save them."""

    settings = _merge(cargar_ajustes(), sections)
    guardar_ajustes(settings)
    return settings


# ------------------------------------------------------------------ MongoDB


def cargar_mongo_config() -> tuple[str, str, str] | None:
    """(uri, database, collection) from the panel settings, then .env; None when incomplete."""

    uri = database = collection = None
    usar = True
    if MONGO_SETTINGS_PATH.exists():
        try:
            data = json.loads(MONGO_SETTINGS_PATH.read_text(encoding="utf-8-sig"))
            uri = data.get("mongodb_uri") or None
            database = data.get("mongodb_database") or None
            collection = data.get("mongodb_collection") or None
            usar = bool(data.get("usar_mongodb", True))
        except (OSError, json.JSONDecodeError):
            pass
    if not usar:
        return None
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    uri = uri or os.getenv("MONGODB_URI")
    database = database or os.getenv("MONGODB_DATABASE")
    collection = collection or os.getenv("MONGODB_COLLECTION_SUNAT")
    if not (uri and database and collection):
        return None
    return str(uri), str(database), str(collection)


def leer_mongo_panel() -> dict[str, Any]:
    """Raw MongoDB values stored by the panel (for editing)."""

    defaults = {"mongodb_uri": "", "mongodb_database": "facturisk", "mongodb_collection": "proveedores_sunat", "usar_mongodb": True}
    if MONGO_SETTINGS_PATH.exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            defaults.update(json.loads(MONGO_SETTINGS_PATH.read_text(encoding="utf-8-sig")))
    return defaults


def guardar_mongo_panel(uri: str, database: str, collection: str, usar: bool = True) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = leer_mongo_panel()
    data.update({"mongodb_uri": uri.strip(), "mongodb_database": database.strip(), "mongodb_collection": collection.strip(), "usar_mongodb": usar})
    MONGO_SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def borrar_mongo_panel() -> None:
    if MONGO_SETTINGS_PATH.exists():
        MONGO_SETTINGS_PATH.unlink()


def probar_mongo(uri: str, timeout_ms: int = 8000) -> tuple[bool, str]:
    """Ping a MongoDB URI; returns (ok, message) without exposing credentials."""

    try:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError
    except ImportError:
        return False, "pymongo no está instalado."
    client = None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        client.admin.command("ping")
        return True, "Conexión exitosa."
    except PyMongoError as exc:
        text = str(exc)
        if "auth" in text.lower():
            return False, "Autenticación fallida: revisa usuario y contraseña."
        if "timed out" in text.lower() or "timeout" in text.lower():
            return False, "Tiempo agotado: revisa la URI, tu red o la IP autorizada en Atlas."
        return False, "No se pudo conectar con MongoDB."
    except Exception:  # invalid URI formats raise ConfigurationError/ValueError
        return False, "La URI no tiene un formato válido."
    finally:
        if client is not None:
            client.close()


def ocultar_uri(uri: str) -> str:
    """URI with the password masked, for display."""

    if "@" not in uri or "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    credentials, host = rest.split("@", 1)
    user = credentials.split(":", 1)[0]
    return f"{scheme}://{user}:••••@{host}"


def ruta_en_proyecto(path: str | Path | None) -> Path | None:
    if not path:
        return None
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
