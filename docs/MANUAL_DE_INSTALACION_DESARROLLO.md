# Manual de instalación para desarrollo

## Requisitos
Python 3.11 o superior recomendado.

## Entorno virtual
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuración
Cree `.env` desde `.env.example` con las variables `MONGODB_URI`, `MONGODB_DATABASE` y `MONGODB_COLLECTION_SUNAT`. No suba credenciales a repositorios públicos.

## Ejecución
```powershell
python .\main.py
python .\main.py gui
python .\main.py todo
```

## Compilación
```powershell
.\build_windows_gui.bat
```

## Pruebas sugeridas
Ejecutar `py_compile`, `preflight_check.py`, `show_final_results.py --no-gui` y una prueba del portable en una carpeta aislada.
