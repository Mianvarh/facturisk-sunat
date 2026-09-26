"""Console summary of the trained model and the pending-invoice predictions."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from paths import get_application_root
from theme import model_display_name

PROJECT_ROOT = get_application_root()
METRICS_PATH = PROJECT_ROOT / "outputs" / "metricas_modelo.json"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "predicciones_pendientes.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "resumen_resultados.txt"


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/D"


def construir_resumen() -> str:
    """Model metrics and risk distribution as plain text."""

    if not METRICS_PATH.exists():
        return "Aún no hay un modelo entrenado. Ejecute las fases de entrenamiento y predicción."
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    test = metrics.get("metricas_prueba", {})
    lines = [
        "RESUMEN DE RESULTADOS",
        "=" * 40,
        f"Modelo: {model_display_name(metrics.get('mejor_modelo'))} ({metrics.get('experimento', '')})",
        f"Umbral de decisión: {float(metrics.get('umbral', 0)):.2f}",
        f"PR-AUC: {float(test.get('pr_auc', 0)):.4f}",
        f"Recall de incidencias: {_pct(test.get('recall_clase_1'))}",
        f"Precisión de incidencias: {_pct(test.get('precision_clase_1'))}",
        f"Comprobantes marcados para revisión: {float(test.get('porcentaje_revision', 0)):.1f}%",
        f"Confiabilidad: {metrics.get('confiabilidad', 'N/D')}",
    ]
    if PREDICTIONS_PATH.exists():
        levels = pd.read_csv(PREDICTIONS_PATH, usecols=["Nivel_Riesgo"], encoding="utf-8-sig")["Nivel_Riesgo"]
        counts = levels.value_counts().reindex(["Alto", "Medio", "Bajo"], fill_value=0)
        lines += ["", f"Pendientes evaluados: {len(levels):,}"]
        lines += [f"  Riesgo {level}: {count:,} ({count / max(len(levels), 1):.1%})" for level, count in counts.items()]
    lines += ["", "Detalle interactivo: python main.py gui → Dashboard de riesgo"]
    return "\n".join(lines)


def mostrar_resumen() -> str:
    summary = construir_resumen()
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return summary


if __name__ == "__main__":
    mostrar_resumen()
