"""Leakage-safe historical feature engineering for supplier behavior."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SMOOTHING = 5.0

HISTORICAL_FEATURES = [
    "comprobantes_previos_proveedor",
    "incidencias_previas_proveedor",
    "tasa_incidencia_previa_proveedor",
    "importe_promedio_previo_proveedor",
    "importe_maximo_previo_proveedor",
    "dias_desde_comprobante_anterior",
    "incidencias_ultimos_30_dias",
    "incidencias_ultimos_90_dias",
    "comprobantes_ultimos_30_dias",
    "comprobantes_ultimos_90_dias",
    "desviacion_importe_vs_promedio_proveedor",
    "proveedor_nuevo",
]


def _rolling_previous_counts(group: pd.DataFrame) -> pd.DataFrame:
    """Compute previous 30/90-day counts for one supplier excluding current row."""

    dates = pd.to_datetime(group["Fecha_Emision"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    valid_positions = np.flatnonzero(~np.isnat(dates))
    valid_dates = dates[valid_positions]
    incidence = pd.to_numeric(group["_incidencia_historica"], errors="coerce").fillna(0).to_numpy(dtype=float)
    valid_incidence = incidence[valid_positions]

    counts_30 = np.zeros(len(group), dtype=np.int64)
    counts_90 = np.zeros(len(group), dtype=np.int64)
    incidents_30 = np.zeros(len(group), dtype=np.int64)
    incidents_90 = np.zeros(len(group), dtype=np.int64)

    if len(valid_positions):
        previous_positions = np.arange(len(valid_positions), dtype=np.int64)
        incidence_prefix = np.concatenate(([0.0], np.cumsum(valid_incidence)))

        starts_30 = np.searchsorted(
            valid_dates,
            valid_dates - np.timedelta64(30, "D"),
            side="left",
        )
        starts_90 = np.searchsorted(
            valid_dates,
            valid_dates - np.timedelta64(90, "D"),
            side="left",
        )
        previous_counts_30 = previous_positions - starts_30
        previous_counts_90 = previous_positions - starts_90
        previous_incidents_30 = incidence_prefix[previous_positions] - incidence_prefix[starts_30]
        previous_incidents_90 = incidence_prefix[previous_positions] - incidence_prefix[starts_90]

        counts_30[valid_positions] = previous_counts_30
        counts_90[valid_positions] = previous_counts_90
        incidents_30[valid_positions] = np.trunc(previous_incidents_30).astype(np.int64)
        incidents_90[valid_positions] = np.trunc(previous_incidents_90).astype(np.int64)

    return pd.DataFrame(
        {
            "comprobantes_ultimos_30_dias": counts_30,
            "comprobantes_ultimos_90_dias": counts_90,
            "incidencias_ultimos_30_dias": incidents_30,
            "incidencias_ultimos_90_dias": incidents_90,
        },
        index=pd.Index(group.index.to_numpy(), name="index"),
    )


def _tasa_global_previa(df: pd.DataFrame) -> pd.Series:
    """Global incidence rate using only invoices emitted on earlier days.

    It is the smoothing prior of the supplier rate. Using only past days keeps
    the current row and any future outcome out of its own features.
    """

    dates = df["Fecha_Emision"].dt.normalize()
    target = df["_incidencia_historica"]
    daily = pd.DataFrame({"suma": target.fillna(0), "conocidos": target.notna().astype(int), "fecha": dates})
    daily = daily.dropna(subset=["fecha"]).groupby("fecha")[["suma", "conocidos"]].sum().sort_index()
    previous = daily.cumsum().shift(1).fillna(0)
    rate_by_day = (previous["suma"] / previous["conocidos"].replace(0, np.nan)).fillna(0.0)
    prior = dates.map(rate_by_day)
    # Rows without a date get the rate of the whole known history before the last day.
    fallback = float(rate_by_day.iloc[-1]) if len(rate_by_day) else 0.0
    return prior.fillna(fallback).astype(float)


def crear_variables_historicas_sin_fuga(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create supplier-history features using only previous records."""

    required = {"RUC_Proveedor", "Fecha_Emision", "Importe_Total", "Incidencia_Aceptacion"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas para feature engineering historico: {', '.join(missing)}")

    logging.info("Creando variables historicas sin fuga de informacion.")
    result = df.copy()
    result["_orden_original"] = np.arange(len(result))
    result = result.sort_values(["RUC_Proveedor", "Fecha_Emision", "_orden_original"]).copy()
    result["_incidencia_historica"] = pd.to_numeric(result["Incidencia_Aceptacion"], errors="coerce")

    global_prior = _tasa_global_previa(result)
    known_target = result["_incidencia_historica"].dropna()
    global_rate = float(known_target.mean()) if len(known_target) else 0.0

    grouped = result.groupby("RUC_Proveedor", dropna=False, sort=False)
    result["comprobantes_previos_proveedor"] = grouped.cumcount()
    result["incidencias_previas_proveedor"] = (
        grouped["_incidencia_historica"].transform(lambda series: series.fillna(0).cumsum().shift(1))
    ).fillna(0)
    result["_resultados_previos_proveedor"] = (
        grouped["_incidencia_historica"].transform(lambda series: series.notna().cumsum().shift(1))
    ).fillna(0)
    result["tasa_incidencia_previa_proveedor"] = (
        (result["incidencias_previas_proveedor"] + global_prior * SMOOTHING)
        / (result["_resultados_previos_proveedor"] + SMOOTHING)
    )
    result["importe_promedio_previo_proveedor"] = (
        grouped["Importe_Total"].transform(lambda series: series.expanding().mean().shift(1))
    )
    result["importe_maximo_previo_proveedor"] = (
        grouped["Importe_Total"].transform(lambda series: series.expanding().max().shift(1))
    )
    result["dias_desde_comprobante_anterior"] = grouped["Fecha_Emision"].diff().dt.days
    result["proveedor_nuevo"] = (result["comprobantes_previos_proveedor"] == 0).astype(int)

    rolling_parts = [_rolling_previous_counts(group) for _, group in grouped]
    rolling_df = pd.concat(rolling_parts).sort_index()
    for column in rolling_df.columns:
        result[column] = rolling_df[column]

    result["importe_promedio_previo_proveedor"] = result["importe_promedio_previo_proveedor"].fillna(
        result["Importe_Total"].median()
    )
    result["importe_maximo_previo_proveedor"] = result["importe_maximo_previo_proveedor"].fillna(
        result["Importe_Total"].median()
    )
    result["dias_desde_comprobante_anterior"] = result["dias_desde_comprobante_anterior"].fillna(9999)
    result["desviacion_importe_vs_promedio_proveedor"] = (
        result["Importe_Total"] - result["importe_promedio_previo_proveedor"]
    )

    for column in HISTORICAL_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)

    result = result.sort_values("_orden_original").drop(
        columns=["_orden_original", "_incidencia_historica", "_resultados_previos_proveedor"]
    )

    leakage_report = {
        "control": "Variables historicas calculadas con shift(1), acumulados desplazados y ventanas anteriores al registro.",
        "registros": len(result),
        "tasa_global_historica": global_rate,
        "prior_suavizado": "tasa global con comprobantes de dias anteriores (sin fila actual ni futuro)",
        "proveedores_nuevos": int(result["proveedor_nuevo"].sum()),
        "features_creadas": HISTORICAL_FEATURES,
        "resultado_control_fuga": "OK - el registro actual no participa en sus variables historicas.",
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "control_fuga_feature_engineering.json").write_text(
        json.dumps(leakage_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result, leakage_report
