"""Enhanced ML training pipeline for acceptance-incidence prediction."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import HISTORICAL_FEATURES
from transformadores import RellenadorNAcategoricas
from paths import ensure_directories, get_application_root



PROJECT_ROOT = get_application_root()
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_modelo.csv"
PENDING_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_pendientes.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
ACTIVE_MODEL_PATH = MODELS_DIR / "modelo_incidencias.joblib"

TARGET = "Incidencia_Aceptacion"
DATE_COLUMN = "Fecha_Emision"
RANDOM_STATE = 42
MIN_RECALL_THRESHOLD = 0.60
BOOTSTRAP_ROUNDS = 120
PARALLEL_JOBS = 1 if getattr(sys, "frozen", False) else -1

SUNAT_FEATURES = [
    "Estado_RUC",
    "Condicion_Domicilio",
    "Situacion_Tributaria_Actual",
    "SUNAT_Encontrado",
]

BASE_FEATURES = [
    "Tipo_Comprobante",
    "Moneda",
    "Importe_Total",
    "log_importe",
    "mes_emision",
    "trimestre_emision",
    "dia_semana_emision",
    "plazo_dias",
    *HISTORICAL_FEATURES,
]
FEATURES_WITH_SUNAT = [*BASE_FEATURES, *SUNAT_FEATURES]
CATEGORICAL_COLUMNS = ["Tipo_Comprobante", "Moneda", "Estado_RUC", "Condicion_Domicilio"]

COLUMNAS_PROHIBIDAS_MODELO = [
    "Estado_Aceptacion",
    "Flag_Aceptada",
    "Flag_Observada",
    "Flag_Rechazada",
    "Fecha_Pago",
    "Estado",
    "Estado_Grupo",
    "Estado_Emision",
    TARGET,
]


@dataclass
class ModelResult:
    """Stores fitted model, validation metrics and final test evaluation."""

    experiment: str
    model_name: str
    estimator: Any
    features: list[str]
    threshold: float
    threshold_reason: str
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any] | None = None
    best_params: dict[str, Any] | None = None
    calibrated: bool = False
    calibration_method: str | None = None


def log_phase(name: str) -> float:
    """Print a phase header and return its start time."""

    print("\n" + "=" * 40)
    print(f"FASE: {name}")
    print("=" * 40)
    return time.perf_counter()


def end_phase(start: float, generated: str, interpretation: str) -> None:
    """Print a phase footer."""

    print(f"Duracion de la fase: {time.perf_counter() - start:.1f}s")
    print(f"Archivo generado: {generated}")
    print(f"Interpretacion: {interpretation}")
    print("FASE COMPLETADA CORRECTAMENTE")


def configurar_logging() -> None:
    """Configure logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def optional_package_status() -> dict[str, bool]:
    """Report optional ML libraries that are actually usable.

    In portable builds a package can be present but unusable because its native
    DLLs were not bundled. Import the relevant estimator classes to catch that
    case before training starts.
    """

    checks: dict[str, str] = {
        "xgboost": "from xgboost import XGBClassifier",
        "lightgbm": "from lightgbm import LGBMClassifier",
        "catboost": "from catboost import CatBoostClassifier",
        "imblearn": "import imblearn",
        "shap": "import shap",
        "optuna": "import optuna",
    }
    status: dict[str, bool] = {}
    for package, statement in checks.items():
        if importlib.util.find_spec(package) is None:
            status[package] = False
            continue
        try:
            exec(statement, {})
            status[package] = True
        except Exception as exc:
            status[package] = False
            logging.warning("Libreria opcional no utilizable: %s (%s)", package, exc)
    return status


def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder compatible with sklearn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset() -> pd.DataFrame:
    """Read prepared model dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No existe {DATASET_PATH}. Ejecuta primero prepare_dataset.py.")
    df = pd.read_csv(DATASET_PATH, sep=",", encoding="utf-8-sig", low_memory=False)
    missing = sorted(set(FEATURES_WITH_SUNAT + [TARGET, DATE_COLUMN]) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas para entrenamiento: {', '.join(missing)}")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN, TARGET]).sort_values(DATE_COLUMN).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    return df


def temporal_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create temporal train/validation/test splits."""

    test_start = int(len(df) * 0.80)
    train_validation = df.iloc[:test_start].copy()
    test = df.iloc[test_start:].copy()
    validation_start = int(len(train_validation) * 0.80)
    train = train_validation.iloc[:validation_start].copy()
    validation = train_validation.iloc[validation_start:].copy()
    return train, validation, test


def print_split_summary(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    """Show split dates, sizes and class distributions."""

    for name, data in [("Entrenamiento", train), ("Validacion", validation), ("Prueba", test)]:
        print(f"{name}: {len(data)} registros, {data.shape[1]} columnas")
        print(f"  Fechas: {data[DATE_COLUMN].min().date()} a {data[DATE_COLUMN].max().date()}")
        print("  Distribucion objetivo:")
        print(data[TARGET].value_counts().sort_index().to_string())


def build_preprocessor(features: list[str], scale_numeric: bool) -> ColumnTransformer:
    """Build preprocessing pipeline for non-native categorical models."""

    categorical = [column for column in features if column in CATEGORICAL_COLUMNS]
    numeric = [column for column in features if column not in categorical]
    categorical_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder())]
    )
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    return ColumnTransformer(
        [("cat", categorical_pipe, categorical), ("num", numeric_pipe, numeric)],
        remainder="drop",
    )


def class_ratio(y: pd.Series) -> float:
    """Return negative/positive ratio for imbalance-aware models."""

    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    return negatives / positives


def build_models(features: list[str], y_train: pd.Series) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Create base and optional models with moderate search spaces."""

    models: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "DummyClassifier": (
            Pipeline(
                [
                    ("preprocessor", build_preprocessor(features, scale_numeric=False)),
                    ("model", DummyClassifier(strategy="most_frequent")),
                ]
            ),
            {},
        ),
        "LogisticRegression": (
            Pipeline(
                [
                    ("preprocessor", build_preprocessor(features, scale_numeric=True)),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=2500,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
        ),
        "RandomForestClassifier": (
            Pipeline(
                [
                    ("preprocessor", build_preprocessor(features, scale_numeric=False)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=PARALLEL_JOBS,
                        ),
                    ),
                ]
            ),
            {
                "model__max_depth": [4, 6, 8, 12, None],
                "model__min_samples_leaf": [20, 50, 100],
                "model__max_features": ["sqrt", 0.7, None],
            },
        ),
        "HistGradientBoostingClassifier": (
            Pipeline(
                [
                    ("preprocessor", build_preprocessor(features, scale_numeric=False)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            early_stopping=True,
                        ),
                    ),
                ]
            ),
            {
                "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__l2_regularization": [0.0, 0.1, 1.0],
            },
        ),
    }

    status = optional_package_status()
    if status["xgboost"]:
        try:
            from xgboost import XGBClassifier

            models["XGBClassifier"] = (
                Pipeline(
                    [
                        ("preprocessor", build_preprocessor(features, scale_numeric=False)),
                        (
                            "model",
                            XGBClassifier(
                                n_estimators=250,
                                random_state=RANDOM_STATE,
                                eval_metric="logloss",
                                scale_pos_weight=class_ratio(y_train),
                                n_jobs=PARALLEL_JOBS,
                            ),
                        ),
                    ]
                ),
                {
                    "model__max_depth": [3, 4, 5],
                    "model__learning_rate": [0.03, 0.05, 0.1],
                    "model__subsample": [0.8, 1.0],
                },
            )
        except Exception as exc:
            logging.warning("XGBoost se omitira porque no esta utilizable: %s", exc)
    if status["lightgbm"]:
        try:
            from lightgbm import LGBMClassifier

            models["LGBMClassifier"] = (
                Pipeline(
                    [
                        ("preprocessor", build_preprocessor(features, scale_numeric=False)),
                        (
                            "model",
                            LGBMClassifier(
                                n_estimators=300,
                                class_weight="balanced",
                                random_state=RANDOM_STATE,
                                n_jobs=PARALLEL_JOBS,
                                verbose=-1,
                            ),
                        ),
                    ]
                ),
                {"model__num_leaves": [15, 31, 63], "model__learning_rate": [0.03, 0.05, 0.1]},
            )
        except Exception as exc:
            logging.warning("LightGBM se omitira porque no esta utilizable: %s", exc)
    if status["catboost"]:
        try:
            from catboost import CatBoostClassifier

            cat_features = [column for column in features if column in CATEGORICAL_COLUMNS]
            models["CatBoostClassifier"] = (
                Pipeline(
                    [
                        ("rellenador_categoricas", RellenadorNAcategoricas(cat_features)),
                        (
                            "model",
                            CatBoostClassifier(
                                iterations=250,
                                depth=5,
                                learning_rate=0.05,
                                loss_function="Logloss",
                                auto_class_weights="Balanced",
                                random_state=RANDOM_STATE,
                                verbose=False,
                                cat_features=cat_features,
                            ),
                        ),
                    ]
                ),
                {},
            )
        except Exception as exc:
            logging.warning("CatBoost se omitira porque no esta utilizable: %s", exc)
    return models


def fit_model(name: str, estimator: Any, params: dict[str, list[Any]], x_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, dict[str, Any]]:
    """Fit one model, using RandomizedSearchCV when params are available."""

    if not params or name == "DummyClassifier":
        estimator.fit(x_train, y_train)
        return estimator, {}

    n_iter = min(8, int(np.prod([len(values) for values in params.values()])))
    search = RandomizedSearchCV(
        estimator,
        params,
        n_iter=n_iter,
        scoring="average_precision",
        cv=TimeSeriesSplit(n_splits=3),
        random_state=RANDOM_STATE,
        n_jobs=PARALLEL_JOBS,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_


def predict_scores(estimator: Any, x_data: pd.DataFrame) -> np.ndarray:
    """Return class-1 probabilities."""

    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x_data)[:, 1]
    raise TypeError("El modelo no soporta predict_proba.")


def threshold_table(y_true: pd.Series, y_score: np.ndarray, model_name: str, experiment: str) -> pd.DataFrame:
    """Evaluate thresholds from 0.05 to 0.95."""

    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        y_pred = (y_score >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        rows.append(
            {
                "experimento": experiment,
                "modelo": model_name,
                "umbral": float(threshold),
                "precision_clase_1": precision_score(y_true, y_pred, zero_division=0),
                "recall_clase_1": recall_score(y_true, y_pred, zero_division=0),
                "f1_clase_1": f1_score(y_true, y_pred, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "falsos_positivos": int(cm[0, 1]),
                "falsos_negativos": int(cm[1, 0]),
                "porcentaje_revision": float(y_pred.mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame) -> tuple[float, str]:
    """Select threshold using min recall and max F1."""

    eligible = table[table["recall_clase_1"] >= MIN_RECALL_THRESHOLD]
    if eligible.empty:
        row = table.sort_values("f1_clase_1", ascending=False).iloc[0]
        return float(row["umbral"]), "Ningun umbral cumplio recall >= 0.60; se eligio el mayor F1 disponible y se documenta la limitacion."
    row = eligible.sort_values(["f1_clase_1", "precision_clase_1"], ascending=False).iloc[0]
    return float(row["umbral"]), "Se eligio el umbral que maximiza F1 entre los que cumplen recall >= 0.60."


def compute_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute final metrics for a threshold."""

    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_clase_1": precision_score(y_true, y_pred, zero_division=0),
        "recall_clase_1": recall_score(y_true, y_pred, zero_division=0),
        "f1_clase_1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
        "verdaderos_negativos": int(tn),
        "porcentaje_revision": float(y_pred.mean() * 100),
        "matriz_confusion": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def bootstrap_ci(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, list[float]]:
    """Bootstrap 95% confidence intervals for selected metrics."""

    rng = np.random.default_rng(RANDOM_STATE)
    y = y_true.to_numpy()
    values = {"recall": [], "precision": [], "f1": [], "pr_auc": []}
    for _ in range(BOOTSTRAP_ROUNDS):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        score = y_score[idx]
        pred = (score >= threshold).astype(int)
        values["recall"].append(recall_score(y[idx], pred, zero_division=0))
        values["precision"].append(precision_score(y[idx], pred, zero_division=0))
        values["f1"].append(f1_score(y[idx], pred, zero_division=0))
        values["pr_auc"].append(average_precision_score(y[idx], score))
    return {
        metric: [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]
        for metric, scores in values.items()
        if scores
    }


def evaluate_calibration(best: ModelResult, x_val: pd.DataFrame, y_val: pd.Series) -> tuple[Any, dict[str, Any]]:
    """Compare original and calibrated probabilities on validation data."""

    original_score = predict_scores(best.estimator, x_val)
    original_brier = brier_score_loss(y_val, original_score)
    rows = [{"modelo": "original", "brier_score": original_brier, "pr_auc": average_precision_score(y_val, original_score)}]
    selected_estimator = best.estimator
    selected_method = None

    for method in ["sigmoid", "isotonic"]:
        try:
            calibrated = CalibratedClassifierCV(FrozenEstimator(best.estimator), method=method)
            calibrated.fit(x_val, y_val)
            score = predict_scores(calibrated, x_val)
            brier = brier_score_loss(y_val, score)
            pr_auc = average_precision_score(y_val, score)
            rows.append({"modelo": f"calibrated_{method}", "brier_score": brier, "pr_auc": pr_auc})
            if brier < original_brier and pr_auc >= rows[0]["pr_auc"] * 0.97:
                selected_estimator = calibrated
                selected_method = method
                original_brier = brier
        except Exception as exc:  # calibration is optional and version-sensitive
            rows.append({"modelo": f"calibrated_{method}", "brier_score": np.nan, "pr_auc": np.nan, "error": str(exc)})

    calibration_report = {
        "comparacion": rows,
        "calibracion_seleccionada": selected_method,
        "criterio": "Se selecciona solo si mejora Brier sin perjudicar PR-AUC mas de 3%.",
    }
    return selected_estimator, calibration_report


def train_all(df: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame) -> tuple[list[ModelResult], pd.DataFrame, dict[str, Any]]:
    """Train all requested experiments and models."""

    status = optional_package_status()
    missing_optional = [name for name, installed in status.items() if not installed and name != "imblearn"]
    if missing_optional:
        print(f"Librerias opcionales no instaladas: {', '.join(missing_optional)}")
        print("Comando sugerido: pip install xgboost lightgbm catboost shap optuna")

    all_results: list[ModelResult] = []
    all_thresholds: list[pd.DataFrame] = []
    best_params: dict[str, Any] = {}
    experiments = {"A_sin_sunat": BASE_FEATURES, "B_con_sunat": FEATURES_WITH_SUNAT}

    for experiment, features in experiments.items():
        x_train, y_train = train[features], train[TARGET]
        x_val, y_val = validation[features], validation[TARGET]
        models = build_models(features, y_train)
        for model_name, (estimator, params) in models.items():
            logging.info("Entrenando %s - %s", experiment, model_name)
            try:
                fitted, params_found = fit_model(model_name, estimator, params, x_train, y_train)
                scores = predict_scores(fitted, x_val)
            except Exception as exc:
                logging.exception("Modelo omitido por error real: %s - %s", experiment, model_name)
                print(f"Modelo omitido: {experiment} - {model_name}. Detalle: {exc}")
                continue
            table = threshold_table(y_val, scores, model_name, experiment)
            threshold, reason = select_threshold(table)
            metrics = compute_metrics(y_val, scores, threshold)
            metrics["brier_score"] = brier_score_loss(y_val, scores)
            all_thresholds.append(table)
            best_params[f"{experiment}__{model_name}"] = params_found
            all_results.append(
                ModelResult(
                    experiment=experiment,
                    model_name=model_name,
                    estimator=fitted,
                    features=features,
                    threshold=threshold,
                    threshold_reason=reason,
                    validation_metrics=metrics,
                    best_params=params_found,
                )
            )
    if not all_results:
        raise RuntimeError("No se pudo entrenar ningun modelo candidato. Revise dependencias y dataset.")
    return all_results, pd.concat(all_thresholds, ignore_index=True), best_params


def select_best(results: list[ModelResult]) -> ModelResult:
    """Select best non-dummy model by validation PR-AUC, F1 and recall."""

    candidates = [result for result in results if result.model_name != "DummyClassifier"]
    return sorted(
        candidates,
        key=lambda result: (
            result.validation_metrics["pr_auc"],
            result.validation_metrics["f1_clase_1"],
            result.validation_metrics["recall_clase_1"],
        ),
        reverse=True,
    )[0]


def final_evaluate(result: ModelResult, test: pd.DataFrame) -> ModelResult:
    """Evaluate selected model on final test set."""

    scores = predict_scores(result.estimator, test[result.features])
    result.test_metrics = compute_metrics(test[TARGET], scores, result.threshold)
    result.test_metrics["bootstrap_ci_95"] = bootstrap_ci(test[TARGET], scores, result.threshold)
    result.test_metrics["brier_score"] = brier_score_loss(test[TARGET], scores)
    result.test_metrics["umbral"] = result.threshold
    return result


def comparison_tables(results: list[ModelResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create model comparison and SUNAT contribution tables from validation metrics."""

    rows = []
    for result in results:
        row = {
            "experimento": result.experiment,
            "modelo": result.model_name,
            "umbral_validacion": result.threshold,
            **{k: v for k, v in result.validation_metrics.items() if not isinstance(v, (list, dict, str))},
        }
        rows.append(row)
    comparison = pd.DataFrame(rows)

    aporte_rows = []
    for model_name in sorted(comparison["modelo"].unique()):
        base = comparison[(comparison["experimento"] == "A_sin_sunat") & (comparison["modelo"] == model_name)]
        sunat = comparison[(comparison["experimento"] == "B_con_sunat") & (comparison["modelo"] == model_name)]
        if base.empty or sunat.empty:
            continue
        b, s = base.iloc[0], sunat.iloc[0]
        row: dict[str, Any] = {"modelo": model_name}
        for metric in ["pr_auc", "f1_clase_1", "recall_clase_1", "precision_clase_1", "falsos_positivos", "falsos_negativos"]:
            delta = s[metric] - b[metric]
            row[f"{metric}_sin_sunat"] = b[metric]
            row[f"{metric}_con_sunat"] = s[metric]
            row[f"{metric}_delta_abs"] = delta
            row[f"{metric}_delta_pct"] = (delta / abs(b[metric]) * 100) if b[metric] else np.nan
        row["conclusion"] = "aporta" if row["pr_auc_delta_abs"] > 0.005 and row["f1_clase_1_delta_abs"] > 0 else "aporte_minimo_o_negativo"
        aporte_rows.append(row)
    return comparison, pd.DataFrame(aporte_rows)


def reliability(metrics: dict[str, Any], dummy_pr_auc: float) -> str:
    """Classify reliability without relying on accuracy."""

    if metrics["pr_auc"] <= dummy_pr_auc * 1.05 or metrics["f1_clase_1"] < 0.12:
        return "No confiable"
    if metrics["f1_clase_1"] < 0.30 or metrics["precision_clase_1"] < 0.20:
        return "Limitada"
    if metrics["f1_clase_1"] < 0.50:
        return "Aceptable para priorizacion"
    return "Buena"


def save_confusion(cm: list[list[int]], path: Path, title: str) -> None:
    """Save a confusion matrix plot."""

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], labels=["Aceptada", "Incidencia"])
    ax.set_yticks([0, 1], labels=["Aceptada", "Incidencia"])
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bar_plot(df: pd.DataFrame, x: str, y: str, path: Path, title: str, ylabel: str) -> None:
    """Save a simple bar plot."""

    fig, ax = plt.subplots(figsize=(9, 5))
    df.plot(kind="bar", x=x, y=y, ax=ax, legend=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_graphs(
    df: pd.DataFrame,
    test: pd.DataFrame,
    best: ModelResult,
    comparison: pd.DataFrame,
    threshold_df: pd.DataFrame,
    aporte: pd.DataFrame,
    calibration_report: dict[str, Any],
) -> dict[str, Any]:
    """Generate requested charts."""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    scores = predict_scores(best.estimator, test[best.features])
    pred_050 = (scores >= 0.50).astype(int)
    pred_opt = (scores >= best.threshold).astype(int)
    precision, recall, _ = precision_recall_curve(test[TARGET], scores)
    fpr, tpr, _ = roc_curve(test[TARGET], scores)

    df[TARGET].value_counts().sort_index().plot(kind="bar", title="Distribucion de clases")
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "01_distribucion_clases.png", dpi=160); plt.close()

    nulls = df[FEATURES_WITH_SUNAT + [TARGET]].isna().sum().sort_values(ascending=False).head(20)
    nulls.plot(kind="bar", title="Valores nulos principales")
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "02_calidad_datos.png", dpi=160); plt.close()

    df["SUNAT_Encontrado"].value_counts().sort_index().plot(kind="bar", title="Cobertura SUNAT")
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "03_cobertura_sunat.png", dpi=160); plt.close()

    comparison["modelo_experimento"] = comparison["modelo"] + " - " + comparison["experimento"]
    bar_plot(comparison, "modelo_experimento", "pr_auc", OUTPUTS_DIR / "04_comparacion_modelos_pr_auc.png", "Comparacion PR-AUC", "PR-AUC")
    bar_plot(comparison, "modelo_experimento", "f1_clase_1", OUTPUTS_DIR / "05_comparacion_modelos_f1.png", "Comparacion F1 clase incidencia", "F1")

    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(recall, precision); ax.set_title("Curva Precision-Recall"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "06_precision_recall_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "--"); ax.set_title("Curva ROC"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "07_roc_curve.png", dpi=160); plt.close(fig)

    save_confusion(confusion_matrix(test[TARGET], pred_050, labels=[0, 1]).tolist(), OUTPUTS_DIR / "08_matriz_confusion_050.png", "Matriz de confusion umbral 0.50")
    save_confusion(confusion_matrix(test[TARGET], pred_opt, labels=[0, 1]).tolist(), OUTPUTS_DIR / "09_matriz_confusion_optimizada.png", "Matriz de confusion umbral optimizado")
    save_confusion(confusion_matrix(test[TARGET], pred_opt, labels=[0, 1]).tolist(), OUTPUTS_DIR / "matriz_confusion_umbral_optimizado.png", "Matriz de confusion umbral optimizado")
    save_confusion(confusion_matrix(test[TARGET], pred_050, labels=[0, 1]).tolist(), OUTPUTS_DIR / "matriz_confusion.png", "Matriz de confusion umbral 0.50")

    best_thresholds = threshold_df[(threshold_df["modelo"] == best.model_name) & (threshold_df["experimento"] == best.experiment)]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(best_thresholds["umbral"], best_thresholds["f1_clase_1"], label="F1"); ax.plot(best_thresholds["umbral"], best_thresholds["recall_clase_1"], label="Recall"); ax.plot(best_thresholds["umbral"], best_thresholds["precision_clase_1"], label="Precision"); ax.legend(); ax.set_title("Metricas por umbral"); fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "10_metricas_por_umbral.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(best_thresholds["umbral"], best_thresholds["falsos_positivos"]); ax.set_title("Falsos positivos por umbral"); ax.set_xlabel("Umbral"); fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "11_falsos_positivos_por_umbral.png", dpi=160); plt.close(fig)

    prob_true, prob_pred = calibration_curve(test[TARGET], scores, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(prob_pred, prob_true, marker="o"); ax.plot([0, 1], [0, 1], "--"); ax.set_title("Curva de calibracion"); ax.set_xlabel("Probabilidad media"); ax.set_ylabel("Fraccion positiva"); fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "12_curva_calibracion.png", dpi=160); plt.close(fig)

    importance_df = variable_importance(best, test)
    importance_df.head(20).plot(kind="barh", x="variable", y="importancia", legend=False, title="Importancia de variables")
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "13_importancia_variables.png", dpi=160); plt.close()
    if importlib.util.find_spec("shap") is None:
        (OUTPUTS_DIR / "14_shap_resumen.txt").write_text("SHAP no esta instalado. Comando sugerido: pip install shap\n", encoding="utf-8")

    if not aporte.empty:
        aporte_plot = aporte[["modelo", "pr_auc_delta_abs", "f1_clase_1_delta_abs", "recall_clase_1_delta_abs"]].set_index("modelo")
        aporte_plot.plot(kind="bar", title="Aporte de variables SUNAT")
        plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "15_aporte_sunat.png", dpi=160); plt.close()

    return {"importance": importance_df, "calibration": calibration_report}


def variable_importance(best: ModelResult, test: pd.DataFrame) -> pd.DataFrame:
    """Compute permutation importance for the winning model."""

    try:
        result = permutation_importance(
            best.estimator,
            test[best.features],
            test[TARGET],
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="average_precision",
            n_jobs=PARALLEL_JOBS,
        )
        return pd.DataFrame({"variable": best.features, "importancia": result.importances_mean}).sort_values("importancia", ascending=False)
    except Exception as exc:
        logging.warning("No se pudo calcular permutation importance: %s", exc)
        return pd.DataFrame({"variable": best.features, "importancia": np.zeros(len(best.features))})


def risk_thresholds_for_pending(estimator: Any, features: list[str]) -> dict[str, float]:
    """Define risk cuts from the pending probability distribution."""

    if not PENDING_PATH.exists():
        return {"bajo_medio": 0.30, "medio_alto": 0.60, "metodo": "fallback"}
    pending = pd.read_csv(PENDING_PATH, sep=",", encoding="utf-8-sig", low_memory=False)
    scores = predict_scores(estimator, pending[features])
    return {
        "bajo_medio": float(np.quantile(scores, 0.33)),
        "medio_alto": float(np.quantile(scores, 0.85)),
        "metodo": "cortes por distribucion real de pendientes: p33 y p85",
    }


def save_reports(
    df: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    results: list[ModelResult],
    best: ModelResult,
    comparison: pd.DataFrame,
    aporte: pd.DataFrame,
    thresholds: pd.DataFrame,
    best_params: dict[str, Any],
    calibration_report: dict[str, Any],
    importance: pd.DataFrame,
) -> None:
    """Save metrics, reports and model artifacts."""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUTS_DIR / "comparacion_modelos.csv", index=False, encoding="utf-8-sig")
    aporte.to_csv(OUTPUTS_DIR / "comparacion_aporte_sunat_mejorada.csv", index=False, encoding="utf-8-sig")
    aporte.to_csv(OUTPUTS_DIR / "comparacion_aporte_sunat.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(OUTPUTS_DIR / "comparacion_umbrales.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(OUTPUTS_DIR / "importancia_variables.csv", index=False, encoding="utf-8-sig")
    (OUTPUTS_DIR / "mejores_parametros.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    dummy_pr = float(comparison[comparison["modelo"] == "DummyClassifier"]["pr_auc"].max())
    reliability_level = reliability(best.test_metrics or {}, dummy_pr)
    metrics_payload = {
        "mejor_modelo": best.model_name,
        "experimento": best.experiment,
        "umbral": best.threshold,
        "razon_umbral": best.threshold_reason,
        "metricas_prueba": best.test_metrics,
        "metricas_validacion": best.validation_metrics,
        "calibracion": calibration_report,
        "confiabilidad": reliability_level,
        "features": best.features,
        "columnas_prohibidas_modelo": COLUMNAS_PROHIBIDAS_MODELO,
    }
    (OUTPUTS_DIR / "metricas_modelo.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "REPORTE FINAL DEL MODELO",
        "=" * 80,
        f"Datos utilizados: {len(df)} registros definitivos. Train={len(train)}, Validacion={len(validation)}, Test={len(test)}.",
        f"Cobertura SUNAT en dataset: {df['SUNAT_Encontrado'].mean() * 100:.2f}%.",
        "Variable adicional: situacion tributaria actual y variables historicas sin fuga por proveedor.",
        f"Modelos comparados: {', '.join(sorted(comparison['modelo'].unique()))}.",
        f"Mejor modelo seleccionado: {best.model_name} ({best.experiment}).",
        f"Mejor umbral: {best.threshold:.2f}. {best.threshold_reason}",
        "Metricas finales:",
        json.dumps(best.test_metrics, ensure_ascii=False, indent=2, default=str),
        f"Modelo guardado: {ACTIVE_MODEL_PATH.name}.",
        "Aporte real de SUNAT:",
        aporte.to_string(index=False) if not aporte.empty else "No disponible.",
        "Principales variables:",
        importance.head(15).to_string(index=False),
        f"Falsas alertas: {(best.test_metrics or {})['falsos_positivos']}.",
        f"Incidencias no detectadas: {(best.test_metrics or {})['falsos_negativos']}.",
        "Limitaciones: clase minoritaria desbalanceada, precision baja si se prioriza recall, no interpretar asociaciones como causalidad.",
        "Uso recomendado: priorizacion operativa de revision, no decision automatica definitiva.",
        f"Conclusion sobre confiabilidad: {reliability_level}.",
    ]
    report_text = "\n".join(lines)
    (OUTPUTS_DIR / "reporte_modelo.txt").write_text(report_text, encoding="utf-8")
    (OUTPUTS_DIR / "reporte_final_modelo.txt").write_text(report_text, encoding="utf-8")
    (OUTPUTS_DIR / "resumen_ejecutivo.md").write_text(
        f"# Resumen ejecutivo\n\nMejor modelo: **{best.model_name}**.\n\nUmbral: **{best.threshold:.2f}**.\n\n"
        f"PR-AUC: **{(best.test_metrics or {})['pr_auc']:.4f}**. F1 incidencia: **{(best.test_metrics or {})['f1_clase_1']:.4f}**. "
        f"Recall incidencia: **{(best.test_metrics or {})['recall_clase_1']:.4f}**.\n\n"
        f"Confiabilidad: **{reliability_level}**.\n",
        encoding="utf-8",
    )
    (OUTPUTS_DIR / "resultados_exposicion.txt").write_text(
        f"Mejor modelo: {best.model_name}\nUmbral: {best.threshold:.2f}\n"
        f"Precision incidencia: {(best.test_metrics or {})['precision_clase_1']:.4f}\n"
        f"Recall incidencia: {(best.test_metrics or {})['recall_clase_1']:.4f}\n"
        f"F1 incidencia: {(best.test_metrics or {})['f1_clase_1']:.4f}\n"
        f"PR-AUC: {(best.test_metrics or {})['pr_auc']:.4f}\n"
        f"Falsos positivos: {(best.test_metrics or {})['falsos_positivos']}\n"
        f"Falsos negativos: {(best.test_metrics or {})['falsos_negativos']}\n"
        f"Confiabilidad: {reliability_level}\n",
        encoding="utf-8",
    )


def save_model(best: ModelResult, risk_cuts: dict[str, float]) -> None:
    """Save the selected model bundle used by the prediction phase."""

    bundle = {
        "pipeline": best.estimator,
        "features": best.features,
        "target": TARGET,
        "experiment": best.experiment,
        "model_name": best.model_name,
        "threshold": best.threshold,
        "metrics": best.test_metrics,
        "risk_cuts": risk_cuts,
        "columnas_prohibidas_modelo": COLUMNAS_PROHIBIDAS_MODELO,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ACTIVE_MODEL_PATH)


def entrenar_modelos() -> ModelResult:
    """Main enhanced training function."""

    ensure_directories()
    configurar_logging()
    start = log_phase("ENTRENAMIENTO Y COMPARACION DE MODELOS")
    df = load_dataset()
    print(f"Archivo leido: {DATASET_PATH}")
    print(f"Registros: {len(df)} | Columnas: {df.shape[1]}")
    print("Nulos importantes:")
    print(df[FEATURES_WITH_SUNAT + [TARGET]].isna().sum().sort_values(ascending=False).head(12).to_string())
    print(f"Duplicados ID_Comprobante: {df['ID_Comprobante'].duplicated().sum() if 'ID_Comprobante' in df else 'N/D'}")
    print("Distribucion objetivo:")
    print(df[TARGET].value_counts().sort_index().to_string())
    print("Transformaciones: variables historicas sin fuga, division temporal train/validacion/test, busqueda moderada, umbral y calibracion.")

    train, validation, test = temporal_splits(df)
    print_split_summary(train, validation, test)
    results, thresholds, best_params = train_all(df, train, validation)
    comparison, aporte = comparison_tables(results)
    best = select_best(results)
    calibrated_estimator, calibration_report = evaluate_calibration(best, validation[best.features], validation[TARGET])
    if calibration_report["calibracion_seleccionada"]:
        best.estimator = calibrated_estimator
        best.calibrated = True
        best.calibration_method = calibration_report["calibracion_seleccionada"]
        calibrated_scores = predict_scores(best.estimator, validation[best.features])
        calibrated_thresholds = threshold_table(
            validation[TARGET],
            calibrated_scores,
            best.model_name,
            f"{best.experiment}_calibrado_{best.calibration_method}",
        )
        thresholds = pd.concat([thresholds, calibrated_thresholds], ignore_index=True)
        best.threshold, best.threshold_reason = select_threshold(calibrated_thresholds)
        best.validation_metrics = compute_metrics(validation[TARGET], calibrated_scores, best.threshold)
        best.validation_metrics["brier_score"] = brier_score_loss(validation[TARGET], calibrated_scores)
    best = final_evaluate(best, test)
    risk_cuts = risk_thresholds_for_pending(best.estimator, best.features)
    save_model(best, risk_cuts)
    graph_info = generate_graphs(df, test, best, comparison, thresholds, aporte, calibration_report)
    save_reports(
        df,
        train,
        validation,
        test,
        results,
        best,
        comparison,
        aporte,
        thresholds,
        best_params,
        calibration_report,
        graph_info["importance"],
    )

    print("\n" + "=" * 40)
    print("RESUMEN FINAL DEL PROYECTO")
    print("=" * 40)
    print(f"Registros historicos definitivos: {len(df)}")
    print(f"Registros usados para entrenamiento: {len(train)}")
    print(f"Registros validacion: {len(validation)}")
    print(f"Registros prueba: {len(test)}")
    print(f"Cobertura SUNAT: {df['SUNAT_Encontrado'].mean() * 100:.2f}%")
    print(f"Mejor modelo seleccionado: {best.model_name} ({best.experiment})")
    print(f"Umbral seleccionado: {best.threshold:.2f}")
    print(f"Precision de incidencias: {(best.test_metrics or {})['precision_clase_1']:.4f}")
    print(f"Recall de incidencias: {(best.test_metrics or {})['recall_clase_1']:.4f}")
    print(f"F1 de incidencias: {(best.test_metrics or {})['f1_clase_1']:.4f}")
    print(f"PR-AUC: {(best.test_metrics or {})['pr_auc']:.4f}")
    print(f"Falsos positivos: {(best.test_metrics or {})['falsos_positivos']}")
    print(f"Falsos negativos: {(best.test_metrics or {})['falsos_negativos']}")
    print(f"Modelo guardado para prediccion: {ACTIVE_MODEL_PATH}")
    print("Aporte de SUNAT: revisar outputs/comparacion_aporte_sunat.csv")
    print(f"Nivel de confiabilidad: {reliability(best.test_metrics or {}, float(comparison[comparison['modelo'] == 'DummyClassifier']['pr_auc'].max()))}")
    print("Uso recomendado: priorizar revision manual, no automatizar rechazo.")
    print("Archivos principales generados: outputs/reporte_final_modelo.txt, outputs/resumen_ejecutivo.md, data/models/modelo_incidencias.joblib")

    end_phase(start, str(OUTPUTS_DIR / "reporte_final_modelo.txt"), "Entrenamiento completado con validacion temporal honesta.")
    return best


if __name__ == "__main__":
    entrenar_modelos()
