"""Predict acceptance-incidence risk for pending vouchers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from paths import ensure_directories, get_application_root
from theme import PALETTE, RISK_COLORS, apply_chart_style

apply_chart_style()

PROJECT_ROOT = get_application_root()
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "modelo_incidencias.joblib"
PENDING_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_pendientes.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "predicciones_pendientes.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

INPUT_COLUMNS = [
    "ID_Comprobante",
    "RUC_Proveedor",
    "Razon_Social_SUNAT",
    "Tipo_Comprobante",
    "Fecha_Emision",
    "Moneda",
    "Importe_Total",
    "Estado_RUC",
    "Condicion_Domicilio",
    "Situacion_Tributaria_Actual",
]
OUTPUT_COLUMNS = [
    *INPUT_COLUMNS,
    "Probabilidad_Incidencia",
    "Prediccion_Codigo",
    "Prediccion_Texto",
    "Nivel_Riesgo",
    "Razones_Principales",
]

PREDICTION_LABELS = {
    0: "Probablemente aceptada",
    1: "Riesgo de incidencia",
}


def configurar_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cargar_bundle_modelo(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the complete trained model bundle with joblib."""

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo entrenado: {model_path}")

    logging.info("Cargando modelo entrenado: %s", model_path)
    bundle = joblib.load(model_path)
    required_keys = {"pipeline", "features"}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise ValueError(f"El archivo joblib no contiene las claves requeridas: {', '.join(sorted(missing))}")
    return bundle


def leer_pendientes(path: Path = PENDING_DATASET_PATH) -> pd.DataFrame:
    """Read pending vouchers dataset using comma separator."""

    if not path.exists():
        raise FileNotFoundError(f"No existe dataset_pendientes.csv: {path}")

    logging.info("Leyendo comprobantes pendientes: %s", path)
    return pd.read_csv(path, sep=",", encoding="utf-8-sig", low_memory=False)


def validar_columnas(df: pd.DataFrame, features: list[str]) -> None:
    """Validate required model and output columns before prediction."""

    required = set(features + INPUT_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas para predecir: {', '.join(missing)}")


def calcular_nivel_riesgo(probability: float) -> str:
    """Map incidence probability into business risk levels."""

    if probability < calcular_nivel_riesgo.bajo_medio:
        return "Bajo"
    if probability < calcular_nivel_riesgo.medio_alto:
        return "Medio"
    return "Alto"


calcular_nivel_riesgo.bajo_medio = 0.30
calcular_nivel_riesgo.medio_alto = 0.60


def explicar_fila(row: pd.Series) -> str:
    """Create simple human-readable reasons for a pending prediction."""

    reasons: list[str] = []
    if row.get("Estado_RUC") not in [None, "ACTIVO"]:
        reasons.append(f"Estado RUC {row.get('Estado_RUC')}")
    if row.get("Condicion_Domicilio") not in [None, "HABIDO"]:
        reasons.append(f"Domicilio {row.get('Condicion_Domicilio')}")
    if row.get("Situacion_Tributaria_Actual") == 1:
        reasons.append("Situacion tributaria irregular")
    if row.get("proveedor_nuevo", 0) == 1:
        reasons.append("Proveedor sin historial previo")
    if row.get("tasa_incidencia_previa_proveedor", 0) >= 0.20:
        reasons.append("Historial previo con incidencias")
    if row.get("desviacion_importe_vs_promedio_proveedor", 0) > 0:
        reasons.append("Importe sobre promedio historico")
    return "; ".join(reasons[:4]) if reasons else "Sin factores de riesgo destacados"


def predecir_pendientes() -> pd.DataFrame:
    """Apply the trained model to pending vouchers and save predictions."""

    ensure_directories()
    configurar_logging()
    bundle = cargar_bundle_modelo()
    pipeline = bundle["pipeline"]
    features = list(bundle["features"])
    threshold = float(bundle.get("threshold", 0.50))
    risk_cuts = bundle.get("risk_cuts", {"bajo_medio": 0.30, "medio_alto": 0.60})
    calcular_nivel_riesgo.bajo_medio = float(risk_cuts.get("bajo_medio", 0.30))
    calcular_nivel_riesgo.medio_alto = float(risk_cuts.get("medio_alto", 0.60))

    df = leer_pendientes()
    validar_columnas(df, features)

    logging.info("Aplicando modelo a %s comprobantes pendientes con umbral %.2f.", len(df), threshold)
    x_pending = df[features]

    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("El pipeline cargado no soporta predict_proba.")

    probabilities = pipeline.predict_proba(x_pending)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    output = df[INPUT_COLUMNS].copy()
    output["Probabilidad_Incidencia"] = probabilities
    output["Prediccion_Codigo"] = predictions
    output["Prediccion_Texto"] = output["Prediccion_Codigo"].map(PREDICTION_LABELS)
    output["Nivel_Riesgo"] = output["Probabilidad_Incidencia"].map(calcular_nivel_riesgo)
    output["Razones_Principales"] = df.apply(explicar_fila, axis=1)
    output = output[OUTPUT_COLUMNS]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    generar_graficos_pendientes(output)
    logging.info("Predicciones guardadas en: %s", OUTPUT_PATH)

    print("\nResumen de prediccion de pendientes")
    print(f"Total de pendientes procesados: {len(output)}")
    print(f"Umbral de clasificacion usado: {threshold:.2f}")
    print(
        "Cortes de riesgo usados: "
        f"Bajo < {calcular_nivel_riesgo.bajo_medio:.4f}, "
        f"Medio < {calcular_nivel_riesgo.medio_alto:.4f}, Alto >= {calcular_nivel_riesgo.medio_alto:.4f}"
    )
    print("\nDistribucion de predicciones:")
    print(output["Prediccion_Texto"].value_counts().to_string())
    print("\nDistribucion de niveles de riesgo:")
    print(output["Nivel_Riesgo"].value_counts().reindex(["Bajo", "Medio", "Alto"], fill_value=0).to_string())
    print("\n20 comprobantes con mayor probabilidad:")
    top_20 = output.sort_values("Probabilidad_Incidencia", ascending=False).head(20)
    print(
        top_20[
            [
                "ID_Comprobante",
                "RUC_Proveedor",
                "Razon_Social_SUNAT",
                "Tipo_Comprobante",
                "Importe_Total",
                "Probabilidad_Incidencia",
                "Prediccion_Texto",
                "Nivel_Riesgo",
            ]
        ].to_string(index=False)
    )

    try:
        from show_final_results import mostrar_resumen_final

        mostrar_resumen_final(no_gui=False, ask_gui=True)
    except Exception as exc:
        logging.exception("No fue posible mostrar el resumen final: %s", exc)
        print("La prediccion termino correctamente, pero no fue posible mostrar el resumen visual.")

    return output


def generar_graficos_pendientes(output: pd.DataFrame) -> None:
    """Generate pending-risk charts from the current prediction output."""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    risk_counts = output["Nivel_Riesgo"].value_counts().reindex(["Bajo", "Medio", "Alto"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    risk_counts.plot(kind="bar", ax=ax, color=[RISK_COLORS[level] for level in risk_counts.index])
    ax.set_title("Distribucion de riesgo en comprobantes pendientes")
    ax.set_ylabel("Cantidad")
    ax.bar_label(ax.containers[0], color=PALETTE["muted"])
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "16_distribucion_riesgo_pendientes.png", dpi=160)
    plt.close(fig)

    top = output.nlargest(20, "Probabilidad_Incidencia")
    fig, ax = plt.subplots(figsize=(11, 6))
    top.plot(kind="bar", x="ID_Comprobante", y="Probabilidad_Incidencia", ax=ax, legend=False, color=PALETTE["danger"])
    ax.set_title("Top 20 comprobantes pendientes por probabilidad de incidencia")
    ax.set_ylabel("Probabilidad de incidencia")
    ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "17_top_pendientes_riesgo.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    predecir_pendientes()
