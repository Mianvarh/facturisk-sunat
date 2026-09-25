from __future__ import annotations

import zipfile

import pandas as pd

import fuentes_externas
from configuracion import PADRON_COLUMNS


def test_leer_padron_parsea_zip_sunat_con_cr_latin1_y_ruc_unicos(tmp_path) -> None:
    zip_path = tmp_path / "padron.zip"
    contents = (
        'Ruc|Nombre/Razon|A partir del|Resolucion|\r'
        '20123456789|"Compañía Uno SAC"|01/01/2025|R-1|\r'
        '20-12345678-9|"Nombre duplicado"|02/01/2025|R-2|\r'
        '20654321098|Servicios Dos SRL|03/01/2025|R-3|'
    ).encode("latin-1")
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("padron.txt", contents)

    result = fuentes_externas.leer_padron(zip_path)

    assert result["RUC"].tolist() == ["20123456789", "20654321098"]
    assert result["Razon_Social"].tolist() == ["Compañía Uno SAC", "Servicios Dos SRL"]
    assert result.columns.tolist() == ["RUC", "Razon_Social", "Desde"]


def test_cargar_variables_padrones_sin_backup_ni_snapshot_devuelve_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fuentes_externas, "PADRONES_BACKUP_CSV", tmp_path / "missing.csv")
    monkeypatch.setattr(fuentes_externas, "PADRONES_SNAPSHOT_PATH", tmp_path / "missing.parquet")

    assert fuentes_externas.cargar_variables_padrones() is None


def test_cargar_variables_padrones_completa_columnas_ausentes_con_cero(monkeypatch, tmp_path) -> None:
    backup_path = tmp_path / "padrones.csv"
    pd.DataFrame({"RUC": ["20-12345678-9"], PADRON_COLUMNS[0]: [1]}).to_csv(backup_path, index=False)
    monkeypatch.setattr(fuentes_externas, "PADRONES_BACKUP_CSV", backup_path)
    monkeypatch.setattr(fuentes_externas, "PADRONES_SNAPSHOT_PATH", tmp_path / "missing.parquet")

    result = fuentes_externas.cargar_variables_padrones()

    assert result is not None
    assert result["RUC"].tolist() == ["20123456789"]
    assert result[PADRON_COLUMNS[0]].tolist() == [1]
    for column in PADRON_COLUMNS[1:]:
        assert result[column].tolist() == [0]
