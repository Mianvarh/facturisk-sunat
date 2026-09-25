from __future__ import annotations

import pandas as pd

import datos
import prepare_dataset
from train_model import select_threshold, temporal_splits


def test_normalizar_ruc() -> None:
    assert datos.normalizar_ruc("20-12345678-9") == "20123456789"
    assert datos.normalizar_ruc(123456789) == "00123456789"
    assert datos.normalizar_ruc(None) is None
    assert datos.normalizar_ruc("123456789012") is None


def test_select_threshold_prefers_best_f1_among_min_recall_rows() -> None:
    table = pd.DataFrame(
        {
            "umbral": [0.20, 0.30, 0.40],
            "recall_clase_1": [0.59, 0.80, 0.80],
            "f1_clase_1": [0.90, 0.70, 0.70],
            "precision_clase_1": [0.40, 0.90, 0.80],
        }
    )

    threshold, reason = select_threshold(table)

    assert threshold == 0.30
    assert "maximiza F1" in reason


def test_select_threshold_falls_back_to_highest_f1() -> None:
    table = pd.DataFrame(
        {
            "umbral": [0.20, 0.30],
            "recall_clase_1": [0.40, 0.50],
            "f1_clase_1": [0.70, 0.60],
            "precision_clase_1": [0.50, 0.80],
        }
    )

    threshold, reason = select_threshold(table)

    assert threshold == 0.20
    assert "Ningun umbral" in reason


def test_temporal_splits_are_chronological_and_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "Fecha_Emision": pd.date_range("2024-01-01", periods=20, freq="D"),
            "Incidencia_Aceptacion": [0, 1] * 10,
        }
    )
    train, validation, test = temporal_splits(frame)

    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)
    assert len(train) + len(validation) + len(test) == len(frame)
    assert train["Fecha_Emision"].max() < validation["Fecha_Emision"].min()
    assert validation["Fecha_Emision"].max() < test["Fecha_Emision"].min()


def test_combinar_historico_sunat_preserves_rows_and_fills_missing() -> None:
    historico = pd.DataFrame(
        {
            "RUC_Proveedor": ["20100000001", "20200000002", "20100000001"],
            "Importe_Total": [10.0, 20.0, 30.0],
        }
    )
    sunat = pd.DataFrame(
        {
            "RUC": ["20100000001"],
            "Estado_RUC": ["ACTIVO"],
            "Condicion_Domicilio": ["HABIDO"],
            "Situacion_Tributaria_Actual": [0],
        }
    )

    combined, stats = prepare_dataset.combinar_historico_sunat(historico, sunat)

    assert len(combined) == len(historico)
    assert combined["SUNAT_Encontrado"].tolist() == [1, 0, 1]
    missing = combined.loc[combined["RUC_Proveedor"] == "20200000002"].iloc[0]
    assert missing["Estado_RUC"] == "NO_ENCONTRADO"
    assert missing["Condicion_Domicilio"] == "NO_ENCONTRADO"
    assert missing["Situacion_Tributaria_Actual"] == -1
    assert stats["filas_antes_merge"] == stats["filas_despues_merge"] == 3


def test_cargar_sunat_uses_snapshot_without_mongodb_or_backup(monkeypatch, tmp_path) -> None:
    snapshot_path = tmp_path / "proveedores_sunat_snapshot.parquet"
    pd.DataFrame(
        {
            "RUC": [20100000001],
            "Estado_RUC": ["ACTIVO"],
            "Condicion_Domicilio": ["HABIDO"],
            "Situacion_Tributaria_Actual": [0],
        }
    ).to_parquet(snapshot_path, index=False)

    monkeypatch.setattr(prepare_dataset, "cargar_config_mongodb", lambda: None)
    monkeypatch.setattr(prepare_dataset, "SUNAT_BACKUP_CSV", tmp_path / "missing.csv")
    monkeypatch.setattr(prepare_dataset, "SUNAT_SNAPSHOT_PATH", snapshot_path)

    loaded = prepare_dataset.cargar_sunat()

    assert loaded["RUC"].tolist() == ["20100000001"]
    assert loaded["Estado_RUC"].tolist() == ["ACTIVO"]
