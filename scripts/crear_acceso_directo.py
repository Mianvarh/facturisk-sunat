"""Crea un acceso directo de FactuRisk SUNAT en el escritorio de Windows.

El acceso directo abre la interfaz con pythonw.exe (sin ventana de consola) y
usa el ícono del producto en lugar del logo de Python:

    python scripts/crear_acceso_directo.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ICON = PROJECT_ROOT / "assets" / "facturisk.ico"


def pythonw_path() -> Path:
    """pythonw.exe next to the running interpreter (falls back to python.exe)."""

    candidate = Path(sys.executable).with_name("pythonw.exe")
    return candidate if candidate.exists() else Path(sys.executable)


def desktop_path() -> Path:
    """User desktop folder, including OneDrive-redirected desktops."""

    output = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(output.stdout.strip())


def crear_acceso_directo() -> Path:
    if sys.platform != "win32":
        raise SystemExit("El acceso directo solo está disponible en Windows.")
    shortcut = desktop_path() / "FactuRisk SUNAT.lnk"
    script = f"""
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut('{shortcut}')
$link.TargetPath = '{pythonw_path()}'
$link.Arguments = '"{PROJECT_ROOT / "src" / "gui_app.py"}"'
$link.WorkingDirectory = '{PROJECT_ROOT}'
$link.IconLocation = '{ICON},0'
$link.Description = 'FactuRisk SUNAT - Predicción de incidencias en comprobantes'
$link.Save()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
    return shortcut


if __name__ == "__main__":
    print(f"Acceso directo creado: {crear_acceso_directo()}")
