"""Final console and visual presentation for the FactuRisk SUNAT project."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from paths import ensure_directories, get_application_root

PROJECT_ROOT = get_application_root()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"
VISUAL_LOG = LOGS_DIR / "resultados_visualizacion.log"

METRICS_PATH = OUTPUTS_DIR / "metricas_modelo.json"
MODEL_REPORT_PATH = OUTPUTS_DIR / "reporte_modelo.txt"
MODEL_COMPARISON_PATH = OUTPUTS_DIR / "comparacion_modelos.csv"
SUNAT_COMPARISON_PATHS = [
    OUTPUTS_DIR / "comparacion_aporte_sunat.csv",
    OUTPUTS_DIR / "comparacion_aporte_sunat_mejorada.csv",
]
PREP_REPORT_PATH = OUTPUTS_DIR / "reporte_preparacion.json"
PREDICTIONS_PATH = PROCESSED_DIR / "predicciones_pendientes.csv"
CONSOLE_RESULT_PATH = OUTPUTS_DIR / "resultado_final_consola.txt"
SCRIPT_RESULT_PATH = OUTPUTS_DIR / "guion_resultados_exposicion.txt"

GRAPH_CANDIDATES = {
    "matriz_confusion": [
        OUTPUTS_DIR / "09_matriz_confusion_optimizada.png",
        OUTPUTS_DIR / "matriz_confusion_umbral_optimizado.png",
    ],
    "comparacion_modelos": [
        OUTPUTS_DIR / "04_comparacion_modelos_pr_auc.png",
        OUTPUTS_DIR / "05_comparacion_modelos_f1.png",
    ],
    "precision_recall": [OUTPUTS_DIR / "06_precision_recall_curve.png"],
    "aporte_sunat": [OUTPUTS_DIR / "15_aporte_sunat.png"],
}

GRAPH_LABELS = {
    "matriz_confusion": "Matriz de confusión",
    "comparacion_modelos": "Comparación de modelos",
    "precision_recall": "Curva Precision-Recall",
    "aporte_sunat": "Aporte de SUNAT",
}


def setup_logging() -> None:
    """Configure visual-result logging."""

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=VISUAL_LOG,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file safely."""

    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.exception("No se pudo leer JSON %s: %s", path, exc)
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file safely."""

    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception as exc:
        logging.exception("No se pudo leer CSV %s: %s", path, exc)
        return pd.DataFrame()


def pct(value: Any) -> str:
    """Format decimal metric as percent."""

    try:
        return f"{float(value) * 100:.2f} %"
    except (TypeError, ValueError):
        return "No disponible"


def integer(value: Any) -> str:
    """Format integer-like values."""

    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "No disponible"


def get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get nested dictionary values."""

    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def locate_graphs() -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Locate important graph files."""

    found: dict[str, Path] = {}
    missing: dict[str, list[Path]] = {}
    for key, candidates in GRAPH_CANDIDATES.items():
        selected = next((path for path in candidates if path.exists()), None)
        if selected is None:
            missing[key] = candidates
        else:
            found[key] = selected
    return found, missing


def load_summary_data() -> dict[str, Any]:
    """Load all available result artifacts without recalculating anything."""

    metrics = read_json(METRICS_PATH)
    prep = read_json(PREP_REPORT_PATH)
    predictions = read_csv(PREDICTIONS_PATH)
    comparison = read_csv(MODEL_COMPARISON_PATH)
    sunat_comparison = pd.DataFrame()
    for path in SUNAT_COMPARISON_PATHS:
        sunat_comparison = read_csv(path)
        if not sunat_comparison.empty:
            break
    graphs_found, graphs_missing = locate_graphs()
    return {
        "metrics": metrics,
        "prep": prep,
        "predictions": predictions,
        "comparison": comparison,
        "sunat_comparison": sunat_comparison,
        "graphs_found": graphs_found,
        "graphs_missing": graphs_missing,
        "report_exists": MODEL_REPORT_PATH.exists(),
    }


def confidence_level(recall: Any, f1: Any) -> tuple[str, str]:
    """Classify reliability using recall and F1, not accuracy."""

    try:
        recall_value = float(recall)
        f1_value = float(f1)
    except (TypeError, ValueError):
        return "No disponible", "No hay metricas suficientes para clasificar la confiabilidad."

    if recall_value >= 0.80 and f1_value >= 0.60:
        return "BUENO", "Confiabilidad buena para apoyar decisiones, manteniendo supervision humana."
    if recall_value >= 0.60 and f1_value >= 0.35:
        return "ACEPTABLE PARA PRIORIZACION", "Utilidad aceptable para priorizar revisiones."
    if recall_value >= 0.50:
        return "LIMITADO", "Confiabilidad limitada para decisiones automaticas; util como alerta preventiva."
    return "BAJO", "Confiabilidad baja; usar solo como referencia exploratoria."


def prediction_counts(predictions: pd.DataFrame) -> dict[str, Any]:
    """Extract pending prediction counts."""

    if predictions.empty:
        return {
            "total": "No disponible",
            "accepted": "No disponible",
            "risk": "No disponible",
            "risk_low": "No disponible",
            "risk_medium": "No disponible",
            "risk_high": "No disponible",
        }
    if "Prediccion_Codigo" in predictions.columns:
        accepted = int((predictions["Prediccion_Codigo"] == 0).sum())
        risk = int((predictions["Prediccion_Codigo"] == 1).sum())
    elif "Prediccion_Texto" in predictions.columns:
        accepted = int(predictions["Prediccion_Texto"].astype(str).str.contains("aceptada", case=False, na=False).sum())
        risk = int(predictions["Prediccion_Texto"].astype(str).str.contains("incidencia", case=False, na=False).sum())
    else:
        accepted = "No disponible"
        risk = "No disponible"

    risk_counts = predictions["Nivel_Riesgo"].value_counts() if "Nivel_Riesgo" in predictions.columns else {}
    return {
        "total": len(predictions),
        "accepted": accepted,
        "risk": risk,
        "risk_low": int(risk_counts.get("Bajo", 0)) if hasattr(risk_counts, "get") else "No disponible",
        "risk_medium": int(risk_counts.get("Medio", 0)) if hasattr(risk_counts, "get") else "No disponible",
        "risk_high": int(risk_counts.get("Alto", 0)) if hasattr(risk_counts, "get") else "No disponible",
    }


def final_test_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return final test metrics using the current schema or the legacy schema."""

    return metrics.get("metricas_prueba") or metrics.get("metricas") or {}


def confusion_values(test_metrics: dict[str, Any]) -> dict[str, Any]:
    """Extract TN, FP, FN and TP from the final confusion matrix."""

    matrix = test_metrics.get("matriz_confusion")
    if not (isinstance(matrix, list) and len(matrix) == 2 and all(isinstance(row, list) and len(row) == 2 for row in matrix)):
        return {"valid": False, "matrix": None}
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    return {
        "valid": True,
        "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def validate_visual_consistency(data: dict[str, Any]) -> dict[str, Any]:
    """Validate that cards, matrix and final model metadata are consistent."""

    metrics = data["metrics"]
    test_metrics = final_test_metrics(metrics)
    confusion = confusion_values(test_metrics)
    messages: list[str] = []
    is_consistent = bool(confusion["valid"])

    if not confusion["valid"]:
        messages.append("No se encontró una matriz de confusión final válida.")
    else:
        fp_ok = int(test_metrics.get("falsos_positivos", -1)) == confusion["fp"]
        fn_ok = int(test_metrics.get("falsos_negativos", -1)) == confusion["fn"]
        recall_expected = confusion["tp"] / (confusion["tp"] + confusion["fn"]) if (confusion["tp"] + confusion["fn"]) else 0
        recall_ok = abs(float(test_metrics.get("recall_clase_1", -1)) - recall_expected) < 0.0005
        if not fp_ok:
            messages.append("Los falsos positivos de la tarjeta no coinciden con la matriz.")
        if not fn_ok:
            messages.append("Los falsos negativos de la tarjeta no coinciden con la matriz.")
        if not recall_ok:
            messages.append("El recall calculado desde la matriz no coincide con la métrica final.")
        is_consistent = fp_ok and fn_ok and recall_ok

    model_name = metrics.get("mejor_modelo") or test_metrics.get("modelo")
    if not model_name:
        is_consistent = False
        messages.append("No se encontró el nombre del modelo final seleccionado.")

    threshold = metrics.get("umbral") or metrics.get("threshold") or test_metrics.get("umbral")
    if threshold is None:
        messages.append("No se encontró el umbral final seleccionado.")

    if not is_consistent:
        logging.warning("Inconsistencia visual detectada: %s", " | ".join(messages))

    return {
        "ok": is_consistent,
        "messages": messages,
        "confusion": confusion,
        "model_name": model_name or "No disponible",
        "threshold": threshold,
    }


def graph_interpretation(key: str, data: dict[str, Any], consistency: dict[str, Any]) -> str:
    """Build a short dynamic interpretation for each graph."""

    metrics = final_test_metrics(data["metrics"])
    confusion = consistency["confusion"]
    if key == "matriz_confusion" and confusion.get("valid"):
        return (
            f"El modelo detectó {integer(confusion['tp'])} incidencias y dejó pasar "
            f"{integer(confusion['fn'])}. También generó {integer(confusion['fp'])} falsas alertas."
        )
    if key == "comparacion_modelos":
        return (
            f"{consistency['model_name']} fue seleccionado considerando métricas de la clase incidencia, "
            f"especialmente recall, F1-score y PR-AUC ({pct(metrics.get('pr_auc', metrics.get('average_precision_pr_auc')))})."
        )
    if key == "precision_recall":
        return (
            "La curva muestra la relación entre precisión y detección de incidencias. "
            "Al aumentar la detección, normalmente también aumentan las falsas alertas."
        )
    if key == "aporte_sunat":
        aporte = data.get("sunat_comparison", pd.DataFrame())
        if not aporte.empty:
            pr_cols = [column for column in aporte.columns if "pr_auc_delta_abs" in column]
            f1_cols = [column for column in aporte.columns if "f1_clase_1_delta_abs" in column]
            pr_delta = aporte[pr_cols[0]].max() if pr_cols else 0
            f1_delta = aporte[f1_cols[0]].max() if f1_cols else 0
            if pr_delta > 0 and f1_delta >= 0:
                return "Las variables de SUNAT aportaron una mejora positiva en las métricas comparadas."
            return "La comparación indica que el aporte de SUNAT fue mínimo o no mejoró de forma consistente."
        return "No hay datos suficientes para cuantificar el aporte de SUNAT."
    return "Gráfico disponible para sustentar la conclusión del modelo."


def export_summary_file(summary: str) -> None:
    """Regenerate and open the console summary file."""

    CONSOLE_RESULT_PATH.write_text(summary, encoding="utf-8")
    try:
        if os.name == "nt":
            os.startfile(CONSOLE_RESULT_PATH)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{CONSOLE_RESULT_PATH}"')
        else:
            os.system(f'xdg-open "{CONSOLE_RESULT_PATH}"')
    except Exception as exc:
        logging.exception("No se pudo abrir el resumen exportado: %s", exc)


def build_console_summary(data: dict[str, Any]) -> tuple[str, str]:
    """Build the final console summary and exposure script."""

    metrics = data["metrics"]
    prep = data["prep"]
    predictions = data["predictions"]
    counts = prediction_counts(predictions)

    test_metrics = metrics.get("metricas_prueba") or metrics.get("metricas") or {}
    model_name = metrics.get("mejor_modelo") or test_metrics.get("modelo") or "No disponible"
    experiment = metrics.get("experimento") or test_metrics.get("experimento") or "No disponible"
    threshold = metrics.get("umbral") or metrics.get("threshold") or "No disponible"

    recall = test_metrics.get("recall_clase_1")
    f1 = test_metrics.get("f1_clase_1")
    pr_auc = test_metrics.get("pr_auc", test_metrics.get("average_precision_pr_auc"))
    fp = test_metrics.get("falsos_positivos")
    fn = test_metrics.get("falsos_negativos")
    cm = test_metrics.get("matriz_confusion") or metrics.get("matriz_confusion")
    tp = "No disponible"
    if isinstance(cm, list) and len(cm) >= 2 and len(cm[1]) >= 2:
        tp = cm[1][1]

    confidence, confidence_text = confidence_level(recall, f1)
    historical = prep.get("filas_historicas", "No disponible")
    training = prep.get("filas_con_estado_definitivo", "No disponible")

    interpretation = (
        f"El modelo logro detectar aproximadamente {pct(recall)} de las incidencias reales. "
        f"Detecto {integer(tp)} incidencias, dejo sin detectar {integer(fn)} y genero "
        f"{integer(fp)} falsas alertas. Por ello, debe utilizarse como herramienta preventiva "
        "para priorizar comprobantes que requieren revision, no como mecanismo automatico de rechazo."
    )

    sunat_text = (
        "La informacion tributaria actual del proveedor permitio incorporar estado del RUC "
        "y condicion del domicilio fiscal, variables externas que no estaban en la base historica."
    )
    if isinstance(experiment, str) and "sunat" in experiment.lower():
        sunat_text += " El modelo seleccionado incluye variables de SUNAT."

    generated_files = [
        PROCESSED_DIR / "predicciones_pendientes.csv",
        OUTPUTS_DIR / "matriz_confusion.png",
        OUTPUTS_DIR / "comparacion_modelos.csv",
        OUTPUTS_DIR / "comparacion_aporte_sunat.csv",
        OUTPUTS_DIR / "reporte_modelo.txt",
    ]

    lines = [
        "=" * 70,
        "                 RESULTADO FINAL DE LA PREDICCION",
        "=" * 70,
        "",
        "PROCESO COMPLETADO CORRECTAMENTE",
        "",
        f"Registros historicos utilizados: {integer(historical)}",
        f"Registros usados para entrenamiento: {integer(training)}",
        f"Comprobantes pendientes evaluados: {integer(counts['total'])}",
        "",
        "Base NoSQL utilizada:",
        "MongoDB Atlas",
        "",
        "Informacion externa incorporada:",
        "Estado del RUC y condicion del domicilio fiscal obtenidos de SUNAT.",
        "",
        "Modelo seleccionado:",
        f"{model_name} ({experiment})",
        f"Umbral de clasificacion: {threshold if isinstance(threshold, str) else f'{float(threshold):.2f}'}",
        "",
        "RESULTADOS DEL MODELO",
        "-" * 70,
        f"Recall de incidencias: {pct(recall)}",
        f"F1-score de incidencias: {pct(f1)}",
        f"PR-AUC: {pct(pr_auc)}",
        "",
        f"Incidencias correctamente detectadas: {integer(tp)}",
        f"Incidencias no detectadas: {integer(fn)}",
        f"Falsas alertas generadas: {integer(fp)}",
        "",
        "RESULTADOS DE LOS COMPROBANTES PENDIENTES",
        "-" * 70,
        f"Probablemente aceptados: {integer(counts['accepted'])}",
        f"Con riesgo de incidencia: {integer(counts['risk'])}",
        "",
        f"Riesgo bajo: {integer(counts['risk_low'])}",
        f"Riesgo medio: {integer(counts['risk_medium'])}",
        f"Riesgo alto: {integer(counts['risk_high'])}",
        "",
        "INTERPRETACION",
        "-" * 70,
        interpretation,
        "",
        "NIVEL DE CONFIABILIDAD",
        "-" * 70,
        f"NIVEL DE CONFIABILIDAD: {confidence}",
        confidence_text,
        "",
        "PRINCIPAL APORTE DE SUNAT",
        "-" * 70,
        sunat_text,
        "",
        "ARCHIVOS GENERADOS",
        "-" * 70,
        *[f"- {path.relative_to(PROJECT_ROOT)}" if path.exists() else f"- {path.relative_to(PROJECT_ROOT)} (No disponible)" for path in generated_files],
        "",
        "=" * 70,
        "                      FIN DEL PROCESO",
        "=" * 70,
    ]
    summary = "\n".join(lines)

    script = (
        f"El modelo seleccionado fue {model_name}, incorporando la informacion tributaria obtenida de SUNAT. "
        f"El modelo logro detectar aproximadamente {pct(recall)} de las incidencias reales. "
        f"Detecto {integer(tp)} incidencias, no detecto {integer(fn)} y genero {integer(fp)} falsas alertas. "
        f"Su nivel de confiabilidad es {confidence.lower()}, por lo que su principal utilidad es priorizar "
        "los comprobantes que requieren revision antes de su aceptacion, no tomar decisiones automaticas."
    )
    return summary, script


def save_text_outputs(summary: str, script: str) -> None:
    """Persist console summary and exposure script."""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CONSOLE_RESULT_PATH.write_text(summary, encoding="utf-8")
    SCRIPT_RESULT_PATH.write_text(script, encoding="utf-8")


def can_try_gui() -> bool:
    """Return whether this process appears to have interactive GUI capability."""

    if "--no-gui" in sys.argv:
        return False
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        return False
    return True


def open_results_folder() -> None:
    """Open the outputs folder in the OS file explorer."""

    try:
        if os.name == "nt":
            os.startfile(OUTPUTS_DIR)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{OUTPUTS_DIR}"')
        else:
            os.system(f'xdg-open "{OUTPUTS_DIR}"')
    except Exception as exc:
        logging.exception("No se pudo abrir carpeta de resultados: %s", exc)


def show_gui(data: dict[str, Any], summary: str) -> None:
    """Open a Tkinter visual dashboard with the final model results."""

    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        logging.exception("Tkinter no disponible: %s", exc)
        print("No se pudo abrir la ventana gráfica porque Tkinter no está disponible.")
        return

    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("No se pudo abrir la ventana gráfica porque falta Pillow.")
        print("Ejecute: python -m pip install pillow")
        logging.exception("Pillow no esta instalado.")
        return

    metrics = data["metrics"]
    test_metrics = final_test_metrics(metrics)
    counts = prediction_counts(data["predictions"])
    consistency = validate_visual_consistency(data)
    confidence, _ = confidence_level(test_metrics.get("recall_clase_1"), test_metrics.get("f1_clase_1"))
    graphs_found = data["graphs_found"]
    if not consistency["ok"]:
        graphs_found = {key: value for key, value in graphs_found.items() if key != "matriz_confusion"}

    try:
        root = tk.Tk()
    except Exception as exc:
        logging.exception("No se pudo crear ventana Tkinter: %s", exc)
        print("No se detectó una interfaz gráfica disponible. Se mantiene el resumen en consola.")
        return

    root.title("FactuRisk SUNAT - Resultados finales del modelo")
    root.geometry("1280x820")
    root.minsize(1080, 720)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(2, weight=1)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Root.TFrame", background="#f4f6f8")
    style.configure("Panel.TFrame", background="#ffffff", relief="flat")
    style.configure("Title.TLabel", background="#f4f6f8", foreground="#1f2937", font=("Segoe UI", 20, "bold"))
    style.configure("Subtitle.TLabel", background="#f4f6f8", foreground="#4b5563", font=("Segoe UI", 11))
    style.configure("Model.TLabel", background="#f4f6f8", foreground="#1f2937", font=("Segoe UI", 10, "bold"))
    style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
    style.configure("CardTitle.TLabel", background="#ffffff", foreground="#374151", font=("Segoe UI", 9, "bold"))
    style.configure("CardValue.TLabel", background="#ffffff", foreground="#111827", font=("Segoe UI", 18, "bold"))
    style.configure("CardHelp.TLabel", background="#ffffff", foreground="#6b7280", font=("Segoe UI", 8))
    style.configure("ConclusionTitle.TLabel", background="#ffffff", foreground="#111827", font=("Segoe UI", 11, "bold"))
    style.configure("Body.TLabel", background="#ffffff", foreground="#374151", font=("Segoe UI", 9))
    style.configure("Graph.TFrame", background="#ffffff")
    style.configure("Secondary.TLabel", background="#eef2f7", foreground="#374151", font=("Segoe UI", 9))
    style.configure("TButton", font=("Segoe UI", 9))

    root_frame = ttk.Frame(root, style="Root.TFrame", padding=14)
    root_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
    root_frame.columnconfigure(0, weight=1)
    root_frame.rowconfigure(2, weight=1)

    header = ttk.Frame(root_frame, style="Root.TFrame")
    header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    header.columnconfigure(0, weight=1)
    model_name = str(consistency["model_name"])
    model_font_size = 10 if len(model_name) <= 42 else 9
    ttk.Label(header, text="RESULTADOS FINALES DEL MODELO", style="Title.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(
        header,
        text="Predicción de incidencias en comprobantes electrónicos",
        style="Subtitle.TLabel",
    ).grid(row=1, column=0, sticky="w", pady=(2, 4))
    ttk.Label(
        header,
        text=f"Modelo seleccionado: {model_name}",
        style="Model.TLabel",
        font=("Segoe UI", model_font_size, "bold"),
    ).grid(row=2, column=0, sticky="w")
    ttk.Label(
        header,
        text="Los datos visualizados corresponden al modelo final y al umbral seleccionado.",
        style="Subtitle.TLabel",
    ).grid(row=3, column=0, sticky="w", pady=(4, 0))

    summary_panel = ttk.Frame(root_frame, style="Panel.TFrame", padding=12)
    summary_panel.grid(row=1, column=0, sticky="ew", pady=(0, 10))
    summary_panel.columnconfigure(0, weight=3)
    summary_panel.columnconfigure(1, weight=2)

    cards = ttk.Frame(summary_panel, style="Panel.TFrame")
    cards.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    for column in range(3):
        cards.columnconfigure(column, weight=1, uniform="cards")
    for row in range(2):
        cards.rowconfigure(row, weight=1, uniform="cards")

    confusion = consistency["confusion"]
    tp = confusion.get("tp", "No disponible") if confusion.get("valid") else "No disponible"
    card_values = [
        ("Recall de incidencias", pct(test_metrics.get("recall_clase_1")), "Incidencias reales detectadas", "#2563eb"),
        ("F1-score", pct(test_metrics.get("f1_clase_1")), "Equilibrio entre precisión y detección", "#2563eb"),
        ("PR-AUC", pct(test_metrics.get("pr_auc", test_metrics.get("average_precision_pr_auc"))), "Rendimiento sobre la clase minoritaria", "#2563eb"),
        ("Incidencias detectadas", integer(tp), "Casos reales identificados", "#15803d"),
        ("Falsos positivos", integer(test_metrics.get("falsos_positivos")), "Alertas generadas sobre aceptados", "#c2410c"),
        ("Falsos negativos", integer(test_metrics.get("falsos_negativos")), "Incidencias que no fueron detectadas", "#b91c1c"),
    ]
    for index, (title, value, helper, color) in enumerate(card_values):
        row, column = divmod(index, 3)
        frame = ttk.Frame(cards, style="Card.TFrame", padding=10)
        frame.grid(row=row, column=column, sticky="nsew", padx=5, pady=5)
        ttk.Label(frame, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(frame, text=str(value), style="CardValue.TLabel", foreground=color).pack(anchor="w", pady=(3, 1))
        ttk.Label(frame, text=helper, style="CardHelp.TLabel", wraplength=185).pack(anchor="w")

    conclusion = ttk.Frame(summary_panel, style="Panel.TFrame", padding=8)
    conclusion.grid(row=0, column=1, sticky="nsew")
    conclusion.columnconfigure(0, weight=1)
    recall_value = float(test_metrics.get("recall_clase_1", 0) or 0)
    f1_value = float(test_metrics.get("f1_clase_1", 0) or 0)
    pr_auc_value = float(test_metrics.get("pr_auc", test_metrics.get("average_precision_pr_auc", 0)) or 0)
    limitation = (
        "El F1-score y la PR-AUC continúan siendo bajos."
        if f1_value < 0.35 or pr_auc_value < 0.20
        else "Aún requiere supervisión humana antes de tomar decisiones."
    )
    conclusion_text = (
        "Nivel de confiabilidad:\n"
        f"{confidence.title()}\n\n"
        "Interpretación:\n"
        f"El modelo detecta aproximadamente {recall_value * 100:.0f} de cada 100 incidencias reales, "
        "pero todavía genera una cantidad elevada de falsas alertas.\n\n"
        "Uso recomendado:\n"
        "Priorizar comprobantes para revisión preventiva.\n\n"
        "No recomendado:\n"
        "Rechazar comprobantes automáticamente.\n\n"
        "Limitación principal:\n"
        f"{limitation}"
    )
    ttk.Label(conclusion, text="CONCLUSIÓN SOBRE LA CONFIABILIDAD", style="ConclusionTitle.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(conclusion, text=conclusion_text, style="Body.TLabel", justify="left", wraplength=420).grid(row=1, column=0, sticky="nsew", pady=(8, 0))

    graph_panel = ttk.Frame(root_frame, style="Panel.TFrame", padding=12)
    graph_panel.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
    graph_panel.columnconfigure(0, weight=1)
    graph_panel.rowconfigure(1, weight=1)

    tab_bar = ttk.Frame(graph_panel, style="Panel.TFrame")
    tab_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    for column in range(4):
        tab_bar.columnconfigure(column, weight=1, uniform="tabs")

    image_area = ttk.Frame(graph_panel, style="Graph.TFrame")
    image_area.grid(row=1, column=0, sticky="nsew")
    image_area.columnconfigure(0, weight=1)
    image_area.rowconfigure(0, weight=1)
    image_label = ttk.Label(image_area, text="Seleccione un gráfico.", anchor="center", style="Body.TLabel")
    image_label.grid(row=0, column=0, sticky="nsew")
    interpretation_label = ttk.Label(graph_panel, text="", style="Body.TLabel", justify="center", wraplength=1100)
    interpretation_label.grid(row=2, column=0, sticky="ew", pady=(8, 0))
    image_ref: dict[str, Any] = {"photo": None}
    current_graph: dict[str, str | None] = {"key": None}

    def show_image(key: str) -> None:
        current_graph["key"] = key
        path = graphs_found.get(key)
        if path is None:
            image_label.configure(text="Gráfico no disponible para la última ejecución.", image="")
            interpretation_label.configure(text="")
            return
        try:
            image = Image.open(path)
            max_width = max(image_area.winfo_width() - 28, 760)
            max_height = max(image_area.winfo_height() - 28, 330)
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            image_ref["photo"] = photo
            image_label.configure(image=photo, text="")
            interpretation_label.configure(text=graph_interpretation(key, data, consistency))
        except Exception as exc:
            logging.exception("No se pudo mostrar imagen %s: %s", path, exc)
            image_label.configure(text=f"No se pudo abrir el gráfico:\n{path}", image="")
            interpretation_label.configure(text="")

    button_defs = [
        ("Matriz de confusión", "matriz_confusion"),
        ("Comparación de modelos", "comparacion_modelos"),
        ("Curva Precision-Recall", "precision_recall"),
        ("Aporte de SUNAT", "aporte_sunat"),
    ]
    for index, (text, key) in enumerate(button_defs):
        state = "normal" if key in graphs_found else "disabled"
        ttk.Button(tab_bar, text=text, state=state, command=lambda k=key: show_image(k)).grid(
            row=0, column=index, sticky="ew", padx=4
        )

    secondary = ttk.Frame(root_frame, style="Panel.TFrame", padding=(10, 7))
    secondary.grid(row=3, column=0, sticky="ew")
    secondary.columnconfigure(0, weight=1)
    secondary_text = (
        f"Comprobantes pendientes evaluados: {integer(counts['total'])}    |    "
        f"Riesgo alto: {integer(counts['risk_high'])}    |    "
        "Base NoSQL: MongoDB Atlas    |    Fuente externa: SUNAT"
    )
    ttk.Label(secondary, text=secondary_text, style="Secondary.TLabel").grid(row=0, column=0, sticky="w")
    actions = ttk.Frame(secondary, style="Panel.TFrame")
    actions.grid(row=0, column=1, sticky="e")
    ttk.Button(actions, text="Abrir carpeta de resultados", command=open_results_folder).pack(side="left", padx=4)
    ttk.Button(actions, text="Exportar resumen", command=lambda: export_summary_file(summary)).pack(side="left", padx=4)
    ttk.Button(actions, text="Cerrar", command=root.destroy).pack(side="left", padx=4)

    def refresh_graph(_event: object | None = None) -> None:
        if current_graph["key"]:
            show_image(str(current_graph["key"]))

    image_area.bind("<Configure>", refresh_graph)
    if "matriz_confusion" in graphs_found:
        show_image("matriz_confusion")
    elif graphs_found:
        show_image(next(iter(graphs_found)))
    elif not consistency["ok"]:
        image_label.configure(text="La matriz final no pasó la validación de consistencia y no será mostrada.")
    root.mainloop()


def mostrar_resumen_final(no_gui: bool = False, ask_gui: bool = True) -> dict[str, Any]:
    """Print final summary, save text outputs and optionally open the GUI."""

    ensure_directories()
    setup_logging()
    data = load_summary_data()
    summary, script = build_console_summary(data)
    save_text_outputs(summary, script)
    print("\n" + summary)

    found_labels = [GRAPH_LABELS[key] for key in data["graphs_found"]]
    missing_labels = [GRAPH_LABELS[key] for key in data["graphs_missing"]]
    print("\nGraficos encontrados:")
    print(", ".join(found_labels) if found_labels else "No disponible")
    print("Graficos no disponibles:")
    print(", ".join(missing_labels) if missing_labels else "Ninguno")

    if no_gui:
        return data
    if not can_try_gui():
        print("\nNo se detecto una interfaz grafica disponible. Se omite la ventana visual.")
        return data

    open_window = True
    if ask_gui and sys.stdin.isatty():
        try:
            answer = input("\n¿Desea abrir la ventana visual de resultados? [S/N]: ").strip().lower()
            open_window = answer in {"", "s", "si", "sí", "y", "yes"}
        except EOFError:
            print("\nNo se recibio respuesta interactiva. Se omite la ventana visual.")
            open_window = False
    elif ask_gui and not sys.stdin.isatty():
        open_window = False

    if open_window:
        show_gui(data, summary)
    return data


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Muestra el resumen final y graficos del proyecto.")
    parser.add_argument("--no-gui", action="store_true", help="Muestra solo el resumen en consola.")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    mostrar_resumen_final(no_gui=args.no_gui, ask_gui=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
