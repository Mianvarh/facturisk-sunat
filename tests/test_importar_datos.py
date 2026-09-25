from __future__ import annotations

import json

import pandas as pd
import pytest

import configuracion
from importar_datos import (
    _parse_amount,
    detectar_columnas,
    importar,
    normalizar_estado,
    transformar,
)


def _parchear_ajustes_temporales(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(configuracion, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(configuracion, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(configuracion, "APP_SETTINGS_PATH", tmp_path / "config" / "app.json")
    monkeypatch.setattr(configuracion, "MONGO_SETTINGS_PATH", tmp_path / "config" / "mongo.json")


def test_detectar_columnas_resuelve_columnas_de_proveedor_cruzadas() -> None:
    frame = pd.DataFrame(
        {
            "Proveedor": ["20123456789", "20654321098"],
            "Ruc Proveedor": ["Comercial Uno SAC", "Servicios Dos SRL"],
        }
    )

    result = detectar_columnas(frame)

    assert result["RUC_Proveedor"] == "Proveedor"
    assert result["Razon_Social_Proveedor"] == "Ruc Proveedor"


def test_detectar_columnas_acepta_encabezados_alternativos() -> None:
    frame = pd.DataFrame(
        {"RUC": ["20123456789"], "Fecha": ["2025-01-02"], "Monto": ["100"], "Estado SUNAT": ["ACEPTADO"]}
    )

    result = detectar_columnas(frame)

    assert result["RUC_Proveedor"] == "RUC"
    assert result["Fecha_Emision"] == "Fecha"
    assert result["Importe_Total"] == "Monto"
    assert result["Estado_Aceptacion"] == "Estado SUNAT"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ACEPTADO", "Aceptada"),
        ("Rechazado", "Rechazada"),
        ("aceptada con observaciones", "Observada"),
        (float("nan"), "Pendiente / Borrador"),
    ],
)
def test_normalizar_estado_mapea_estados_conocidos(value, expected: str) -> None:
    assert normalizar_estado(value) == expected


def test_parse_amount_admite_separadores_regionales_y_moneda() -> None:
    result = _parse_amount(pd.Series(["1.234,56", "1,234.56", "S/ 500"]))

    assert result.tolist() == [1234.56, 1234.56, 500.0]


def test_transformar_indica_columnas_obligatorias_ausentes() -> None:
    with pytest.raises(ValueError, match="No se encontraron columnas obligatorias: RUC_Proveedor"):
        transformar(pd.DataFrame({"Fecha": ["2025-01-01"]}), {"Fecha_Emision": "Fecha"})


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_importar_csv_o_excel_guarda_parquet_y_activa_dataset(monkeypatch, tmp_path, suffix: str) -> None:
    _parchear_ajustes_temporales(monkeypatch, tmp_path)
    source = tmp_path / f"comprobantes{suffix}"
    data = pd.DataFrame(
        {
            "RUC": ["20-12345678-9", "20123456789"],
            "Fecha": ["2025-01-02", "2025-02-03"],
            "Monto": ["1.234,56", "S/ 500"],
            "Estado SUNAT": ["ACEPTADO", "Rechazado"],
        }
    )
    if suffix == ".csv":
        data.to_csv(source, sep=";", index=False)
    else:
        data.to_excel(source, index=False)
    output_path = tmp_path / "data" / "importado.parquet"

    report = importar(source, output_path)

    assert report.filas == 2
    assert report.rucs_unicos == 1
    assert any("Sin fecha de vencimiento" in warning for warning in report.advertencias)
    result = pd.read_parquet(output_path)
    assert result["RUC_Proveedor"].tolist() == ["20123456789", "20123456789"]
    assert result["Estado_Aceptacion"].tolist() == ["Aceptada", "Rechazada"]
    settings = json.loads(configuracion.APP_SETTINGS_PATH.read_text(encoding="utf-8"))
    assert settings["dataset"]["origen"] == "importado"
    assert settings["dataset"]["ruta"] == str(output_path)
    assert settings["dataset"]["archivo_original"] == source.name
