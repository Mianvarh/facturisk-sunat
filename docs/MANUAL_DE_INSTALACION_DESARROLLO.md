# Manual de instalación para desarrollo

## Requisitos
Python 3.11 o superior. En Windows se recomienda el instalador de python.org.

## Entorno virtual
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

## Configuración
La conexión a MongoDB se configura desde la aplicación (Configuración) o con un archivo `.env` creado a partir de `.env.example`. Ninguno de los dos se sube al repositorio.

## Ejecución
```powershell
python main.py
python main.py gui
python main.py todo
```

## Pruebas
```powershell
python -m pytest
```
