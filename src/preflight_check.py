"""Initial checks for the portable FactuRisk SUNAT application."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from paths import ensure_directories, get_application_root


PROJECT_ROOT = get_application_root()
SUNAT_URL = "https://www.sunat.gob.pe/descargaPRR/mrc137_padron_reducido.html"


def _load_config() -> dict[str, Any]:
    """Load non-printed configuration values from .env and portable JSON."""

    load_dotenv(PROJECT_ROOT / ".env")
    config = {
        "mongodb_uri": os.getenv("MONGODB_URI"),
        "mongodb_database": os.getenv("MONGODB_DATABASE"),
        "mongodb_collection": os.getenv("MONGODB_COLLECTION_SUNAT"),
        "usar_mongodb": True,
        "permitir_scraping": True,
    }
    config_path = PROJECT_ROOT / "config" / "configuracion.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8-sig") as file:
            file_config = json.load(file)
        for key, value in file_config.items():
            config[key] = value if value not in ("", None) else config.get(key)
    return config


def _ok_file(path: Path) -> str:
    return "OK" if path.exists() else "No encontrado"


def _can_write(path: Path) -> str:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return "OK"
    except OSError:
        return "Sin permiso"


def _internet_available() -> str:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5).close()
        return "OK"
    except OSError:
        return "No disponible"


def _sunat_available() -> str:
    try:
        response = requests.get(SUNAT_URL, timeout=10)
        return "OK" if response.status_code == 200 else f"No disponible ({response.status_code})"
    except requests.RequestException:
        return "No disponible"


def _mongodb_available(config: dict[str, Any]) -> str:
    if not config.get("mongodb_uri") or not config.get("usar_mongodb", True):
        return "No configurado"
    try:
        from pymongo import MongoClient

        client = MongoClient(str(config["mongodb_uri"]), serverSelectionTimeoutMS=8_000)
        client.admin.command("ping")
        client.close()
        return "OK"
    except Exception:
        return (
            "No disponible. No se pudo conectar con MongoDB Atlas. "
            "La aplicación está configurada, pero la dirección IP de este equipo podría no estar autorizada."
        )


def ejecutar_preflight(mostrar: bool = True) -> dict[str, str]:
    """Run portable readiness checks and optionally print the result."""

    ensure_directories()
    config = _load_config()
    checks = {
        "Sistema operativo": "OK" if platform.system().lower() == "windows" else platform.system(),
        "Dataset historico": _ok_file(PROJECT_ROOT / "data" / "raw" / "comprobantes.parquet"),
        "Modelo actual": _ok_file(PROJECT_ROOT / "data" / "models" / "modelo_incidencias.joblib"),
        "Datos procesados": _ok_file(PROJECT_ROOT / "data" / "processed" / "dataset_modelo.csv"),
        "Predicciones existentes": _ok_file(PROJECT_ROOT / "data" / "processed" / "predicciones_pendientes.csv"),
        "Configuracion MongoDB": "OK" if config.get("mongodb_uri") else "No configurado (se usaran respaldos locales)",
        "Carpetas de escritura": _can_write(PROJECT_ROOT / "outputs"),
        "Espacio disponible": f"{shutil.disk_usage(PROJECT_ROOT).free // (1024 ** 3)} GB libres",
        "Internet": _internet_available(),
        "MongoDB Atlas": _mongodb_available(config),
        "SUNAT": _sunat_available(),
    }
    if mostrar:
        print("\n========================================")
        print("VERIFICACIÓN INICIAL")
        print("========================================")
        for name, status in checks.items():
            print(f"{name}: {status}")
        print("\nOpciones disponibles con datos locales: inspección, entrenamiento, predicción y resultados.")
        if checks["MongoDB Atlas"] != "OK":
            print("MongoDB no está disponible; el pipeline usará el respaldo local o el snapshot SUNAT incluido.")
        if checks["SUNAT"] != "OK":
            print("SUNAT no está disponible; puede reutilizar la última descarga válida si existe.")
    return checks


if __name__ == "__main__":
    ejecutar_preflight(mostrar=True)
