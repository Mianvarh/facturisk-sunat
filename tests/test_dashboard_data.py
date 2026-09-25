from __future__ import annotations

import json

import pandas as pd
import pytest

from dashboard_data import (
    DashboardData,
    amount_vs_probability,
    kpis,
    load_dashboard_data,
    risk_heatmap,
    risk_level_counts,
    top_risk_factors,
)


def test_load_dashboard_data_missing_file_returns_none(tmp_path) -> None:
    assert load_dashboard_data(tmp_path / "missing.csv", tmp_path / "missing.json") is None


def test_load_dashboard_data_reads_predictions_and_optional_metrics(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "Probabilidad_Incidencia": [0.75],
            "Nivel_Riesgo": ["Alto"],
            "Fecha_Emision": ["2025-01-10"],
            "RUC_Proveedor": ["20100000001"],
        }
    ).to_csv(predictions_path, index=False)

    without_metrics = load_dashboard_data(predictions_path, tmp_path / "absent.json")
    assert isinstance(without_metrics, DashboardData)
    assert without_metrics.metrics == {}
    assert pd.Timestamp("2025-01-10") == without_metrics.predictions.loc[0, "Fecha_Emision"]

    metrics_path = tmp_path / "metricas_modelo.json"
    metrics_path.write_text(json.dumps({"mejor_modelo": "Modelo de prueba"}), encoding="utf-8")
    with_metrics = load_dashboard_data(predictions_path, metrics_path)
    assert isinstance(with_metrics, DashboardData)
    assert with_metrics.metrics["mejor_modelo"] == "Modelo de prueba"


def test_kpis_calculates_counts_percentages_and_model_metrics() -> None:
    data = DashboardData(
        predictions=pd.DataFrame(
            {
                "Nivel_Riesgo": ["Alto", "Alto", "Medio", "Bajo"],
                "Prediccion_Codigo": [1, 0, 1, 1],
                "Probabilidad_Incidencia": [0.8, 0.6, 0.4, 0.2],
            }
        ),
        metrics={
            "mejor_modelo": "Modelo A",
            "umbral": 0.42,
            "metricas_prueba": {"recall_clase_1": 0.75, "pr_auc": 0.61},
        },
    )

    result = kpis(data)

    assert result["pendientes"] == 4
    assert result["riesgo_alto"] == 2
    assert result["riesgo_alto_pct"] == 50.0
    assert result["marcados_revision"] == 3
    assert result["marcados_revision_pct"] == 75.0
    assert result["probabilidad_media"] == pytest.approx(0.5)
    assert result["modelo"] == "Modelo A"
    assert result["umbral"] == 0.42
    assert result["recall"] == 0.75
    assert result["pr_auc"] == 0.61


def test_risk_level_counts_keeps_business_order_and_zero_levels() -> None:
    result = risk_level_counts(pd.DataFrame({"Nivel_Riesgo": ["Alto", "Bajo", "Alto"]}))

    assert result.index.tolist() == ["Bajo", "Medio", "Alto"]
    assert result.tolist() == [1, 0, 2]


def test_top_risk_factors_groups_irregular_ruc_states_and_excludes_no_factors() -> None:
    predictions = pd.DataFrame(
        {
            "Nivel_Riesgo": ["Alto"] * 4 + ["Bajo"],
            "Razones_Principales": [
                "Estado RUC BAJA DE OFICIO",
                "Estado RUC SUSPENSION TEMPORAL",
                "Estado RUC BAJA DE OFICIO",
                "Sin factores de riesgo destacados",
                "Estado RUC INACTIVO",
            ],
        }
    )

    result = top_risk_factors(predictions)

    assert result.to_dict() == {"Estado RUC irregular": 3}
    assert "Sin factores de riesgo destacados" not in result.index


def test_risk_heatmap_uses_voucher_rows_and_only_latest_months() -> None:
    predictions = pd.DataFrame(
        {
            "Tipo_Comprobante": ["Factura", "Boleta", "Factura", "Boleta", "Factura"],
            "Fecha_Emision": pd.to_datetime(
                ["2025-01-01", "2025-01-04", "2025-02-10", "2025-03-02", "2025-03-05"]
            ),
            "Probabilidad_Incidencia": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    )

    result = risk_heatmap(predictions, max_months=2)

    assert result is not None
    assert set(result.index) == {"Factura", "Boleta"}
    assert result.columns.tolist() == ["Feb 25", "Mar 25"]
    assert "Ene 25" not in result.columns


def test_risk_heatmap_returns_none_without_valid_emission_dates() -> None:
    without_column = pd.DataFrame({"Probabilidad_Incidencia": [0.4]})
    without_dates = pd.DataFrame(
        {"Fecha_Emision": pd.to_datetime([None]), "Probabilidad_Incidencia": [0.4]}
    )

    assert risk_heatmap(without_column) is None
    assert risk_heatmap(without_dates) is None


def test_amount_vs_probability_drops_nonpositive_amounts_and_respects_sample_size() -> None:
    predictions = pd.DataFrame(
        {
            "Importe_Total": [-10, 0, 10, 20, 30, 40, 50],
            "Probabilidad_Incidencia": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "Nivel_Riesgo": ["Bajo", "Bajo", "Medio", "Medio", "Alto", "Alto", "Alto"],
        }
    )

    all_positive = amount_vs_probability(predictions, sample=10)
    sampled = amount_vs_probability(predictions, sample=3, seed=7)

    assert all_positive["Importe_Total"].tolist() == [10, 20, 30, 40, 50]
    assert len(sampled) == 3
    assert (sampled["Importe_Total"] > 0).all()
    assert sampled.columns.tolist() == ["Importe_Total", "Probabilidad_Incidencia", "Nivel_Riesgo"]
