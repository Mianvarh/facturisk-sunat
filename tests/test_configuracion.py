from __future__ import annotations

import json

import configuracion


def _parchear_rutas(monkeypatch, tmp_path) -> None:
    config_dir = tmp_path / "config"
    monkeypatch.setattr(configuracion, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(configuracion, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(configuracion, "APP_SETTINGS_PATH", config_dir / "app.json")
    monkeypatch.setattr(configuracion, "MONGO_SETTINGS_PATH", config_dir / "mongo.json")


def test_cargar_ajustes_devuelve_defaults_si_no_hay_archivo(monkeypatch, tmp_path) -> None:
    _parchear_rutas(monkeypatch, tmp_path)

    result = configuracion.cargar_ajustes()

    assert result == configuracion.DEFAULT_SETTINGS
    assert result is not configuracion.DEFAULT_SETTINGS


def test_actualizar_ajustes_combina_secciones_y_conserva_claves(monkeypatch, tmp_path) -> None:
    _parchear_rutas(monkeypatch, tmp_path)
    configuracion.APP_SETTINGS_PATH.parent.mkdir(parents=True)
    configuracion.APP_SETTINGS_PATH.write_text(
        json.dumps({"dataset": {"archivo_original": "anterior.csv"}, "sunat": {"max_age_days": 12}}),
        encoding="utf-8",
    )

    result = configuracion.actualizar_ajustes(dataset={"origen": "importado"}, sunat={"zip_url": "https://example.test/padron.zip"})

    assert result["dataset"]["archivo_original"] == "anterior.csv"
    assert result["dataset"]["origen"] == "importado"
    assert result["sunat"]["max_age_days"] == 12
    assert result["sunat"]["zip_url"] == "https://example.test/padron.zip"
    stored = json.loads(configuracion.APP_SETTINGS_PATH.read_text(encoding="utf-8"))
    assert stored == result


def test_cargar_mongo_config_prioriza_panel_sobre_entorno(monkeypatch, tmp_path) -> None:
    _parchear_rutas(monkeypatch, tmp_path)
    configuracion.MONGO_SETTINGS_PATH.parent.mkdir(parents=True)
    configuracion.MONGO_SETTINGS_PATH.write_text(
        json.dumps(
            {
                "mongodb_uri": "mongodb+srv://panel:secreto@panel.example.net",
                "mongodb_database": "panel_db",
                "mongodb_collection": "panel_collection",
                "usar_mongodb": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MONGODB_URI", "mongodb://env.example.net")
    monkeypatch.setenv("MONGODB_DATABASE", "env_db")
    monkeypatch.setenv("MONGODB_COLLECTION_SUNAT", "env_collection")

    assert configuracion.cargar_mongo_config() == (
        "mongodb+srv://panel:secreto@panel.example.net",
        "panel_db",
        "panel_collection",
    )


def test_cargar_mongo_config_devuelve_none_si_panel_desactivado_o_incompleto(monkeypatch, tmp_path) -> None:
    _parchear_rutas(monkeypatch, tmp_path)
    monkeypatch.setenv("MONGODB_URI", "mongodb://env.example.net")
    monkeypatch.setenv("MONGODB_DATABASE", "env_db")
    monkeypatch.setenv("MONGODB_COLLECTION_SUNAT", "env_collection")
    configuracion.MONGO_SETTINGS_PATH.parent.mkdir(parents=True)
    configuracion.MONGO_SETTINGS_PATH.write_text(json.dumps({"usar_mongodb": False}), encoding="utf-8")

    assert configuracion.cargar_mongo_config() is None

    configuracion.MONGO_SETTINGS_PATH.unlink()
    monkeypatch.delenv("MONGODB_COLLECTION_SUNAT")
    assert configuracion.cargar_mongo_config() is None


def test_ocultar_uri_enmascara_la_contrasena() -> None:
    result = configuracion.ocultar_uri("mongodb+srv://ana:secreto@c.mongodb.net")

    assert "secreto" not in result
    assert result.startswith("mongodb+srv://ana:")
    assert result.endswith("@c.mongodb.net")
