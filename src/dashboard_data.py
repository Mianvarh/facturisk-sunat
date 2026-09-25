"""Data preparation for the risk dashboard, independent from the interface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from paths import get_application_root

PROJECT_ROOT = get_application_root()
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "predicciones_pendientes.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metricas_modelo.json"

RISK_LEVELS = ["Bajo", "Medio", "Alto"]
NO_FACTORS = "Sin factores de riesgo destacados"


@dataclass
class DashboardData:
    """Everything the dashboard needs, loaded once per refresh."""

    predictions: pd.DataFrame
    metrics: dict[str, Any]


def load_dashboard_data(predictions_path: Path = PREDICTIONS_PATH, metrics_path: Path = METRICS_PATH) -> DashboardData | None:
    """Load predictions and model metrics; None when the pipeline has not produced them yet."""

    if not predictions_path.exists():
        return None
    predictions = pd.read_csv(predictions_path, encoding="utf-8-sig", dtype={"RUC_Proveedor": "string"}, low_memory=False)
    if predictions.empty or "Probabilidad_Incidencia" not in predictions.columns:
        return None
    if "Fecha_Emision" in predictions.columns:
        predictions["Fecha_Emision"] = pd.to_datetime(predictions["Fecha_Emision"], errors="coerce")
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = {}
    return DashboardData(predictions=predictions, metrics=metrics)


def kpis(data: DashboardData) -> dict[str, Any]:
    """Headline numbers for the KPI cards."""

    pred = data.predictions
    total = len(pred)
    high = int((pred["Nivel_Riesgo"] == "Alto").sum())
    flagged = int((pred["Prediccion_Codigo"] == 1).sum()) if "Prediccion_Codigo" in pred else 0
    test = data.metrics.get("metricas_prueba", {})
    return {
        "pendientes": total,
        "riesgo_alto": high,
        "riesgo_alto_pct": high / total * 100 if total else 0.0,
        "marcados_revision": flagged,
        "marcados_revision_pct": flagged / total * 100 if total else 0.0,
        "probabilidad_media": float(pred["Probabilidad_Incidencia"].mean()) if total else 0.0,
        "modelo": data.metrics.get("mejor_modelo", "N/D"),
        "umbral": data.metrics.get("umbral"),
        "recall": test.get("recall_clase_1"),
        "pr_auc": test.get("pr_auc"),
    }


def risk_level_counts(pred: pd.DataFrame) -> pd.Series:
    """Invoices per risk level in business order."""

    return pred["Nivel_Riesgo"].value_counts().reindex(RISK_LEVELS, fill_value=0)


def high_risk_share_by_type(pred: pd.DataFrame) -> pd.DataFrame:
    """Share of high-risk invoices per voucher type, largest first."""

    grouped = pred.groupby("Tipo_Comprobante", dropna=False)
    table = pd.DataFrame(
        {
            "comprobantes": grouped.size(),
            "riesgo_alto": grouped["Nivel_Riesgo"].apply(lambda levels: int((levels == "Alto").sum())),
        }
    )
    table["porcentaje_alto"] = table["riesgo_alto"] / table["comprobantes"] * 100
    return table.sort_values("porcentaje_alto", ascending=False)


def risk_by_month(pred: pd.DataFrame) -> pd.DataFrame | None:
    """Mean incidence probability and high-risk count per emission month."""

    if "Fecha_Emision" not in pred.columns or pred["Fecha_Emision"].isna().all():
        return None
    monthly = pred.dropna(subset=["Fecha_Emision"]).copy()
    monthly["mes"] = monthly["Fecha_Emision"].dt.to_period("M").dt.to_timestamp()
    grouped = monthly.groupby("mes")
    return pd.DataFrame(
        {
            "probabilidad_media": grouped["Probabilidad_Incidencia"].mean(),
            "riesgo_alto": grouped["Nivel_Riesgo"].apply(lambda levels: int((levels == "Alto").sum())),
        }
    )


def top_suppliers(pred: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Suppliers (by RUC) with the most high-risk pending invoices."""

    high = pred[pred["Nivel_Riesgo"] == "Alto"]
    table = (
        high.groupby("RUC_Proveedor")
        .agg(riesgo_alto=("ID_Comprobante", "size"), probabilidad_media=("Probabilidad_Incidencia", "mean"))
        .sort_values(["riesgo_alto", "probabilidad_media"], ascending=False)
        .head(limit)
    )
    return table


def top_risk_factors(pred: pd.DataFrame, limit: int = 6) -> pd.Series:
    """Most frequent explanation factors among high-risk invoices."""

    high = pred.loc[pred["Nivel_Riesgo"] == "Alto", "Razones_Principales"].dropna()
    factors = high.str.split(";").explode().str.strip()
    factors = factors[(factors != "") & (factors != NO_FACTORS)]
    # Group variants such as "Estado RUC BAJA DE OFICIO" under their generic factor.
    factors = factors.str.replace(r"^(Estado RUC|Domicilio) .+$", r"\1 irregular", regex=True)
    return factors.value_counts().head(limit)


def top_invoices(pred: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """Pending invoices with the highest incidence probability."""

    columns = [
        column
        for column in ["ID_Comprobante", "RUC_Proveedor", "Tipo_Comprobante", "Importe_Total", "Probabilidad_Incidencia", "Nivel_Riesgo", "Razones_Principales"]
        if column in pred.columns
    ]
    return pred.nlargest(limit, "Probabilidad_Incidencia")[columns]
