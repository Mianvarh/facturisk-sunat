# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


ROOT = Path.cwd()

datas = []
if (ROOT / "assets").exists():
    datas.append((str(ROOT / "assets"), "assets"))

hiddenimports = [
    "datos",
    "paths",
    "gui_app",
    "inspect_data",
    "distributed_processing",
    "scrape_sunat",
    "load_mongodb",
    "prepare_dataset",
    "feature_engineering",
    "train_model",
    "predict_pending",
    "show_final_results",
    "preflight_check",
    "generate_documentation_data",
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "joblib",
    "matplotlib",
    "PIL",
    "pymongo",
    "requests",
    "bs4",
    "charset_normalizer",
    "dotenv",
    "certifi",
    "openpyxl",
    "tkinter",
]

for package_name in ["sklearn", "pymongo", "charset_normalizer"]:
    hiddenimports += collect_submodules(package_name)

for optional_name in ["catboost"]:
    try:
        hiddenimports += collect_submodules(optional_name)
    except Exception:
        pass

datas += collect_data_files("sklearn")
datas += collect_data_files("matplotlib")


a = Analysis(
    ["src/gui_app.py"],
    pathex=[str(ROOT / "src"), str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["__pycache__", "catboost_info", "backups", ".venv", "venv", "xgboost", "lightgbm"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FactuRisk_SUNAT",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FactuRisk_SUNAT",
)
