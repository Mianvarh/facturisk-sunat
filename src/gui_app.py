"""Premium desktop interface for the FactuRisk SUNAT pipeline."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

import pandas as pd
from PIL import Image, ImageTk

from paths import ensure_directories, get_application_root


PROJECT_ROOT = get_application_root()
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ASSETS_DIR = PROJECT_ROOT / "assets"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PHASE_SCRIPTS = {
    "inspect": SRC_DIR / "inspect_data.py",
    "distribuido": SRC_DIR / "distributed_processing.py",
    "scraping": SRC_DIR / "scrape_sunat.py",
    "mongodb": SRC_DIR / "load_mongodb.py",
    "preparar": SRC_DIR / "prepare_dataset.py",
    "entrenar": SRC_DIR / "train_model.py",
    "predecir": SRC_DIR / "predict_pending.py",
    "documentar": SRC_DIR / "generate_documentation_data.py",
}

PHASE_ORDER = ["inspect", "distribuido", "scraping", "mongodb", "preparar", "entrenar", "predecir", "documentar"]


@dataclass(frozen=True)
class PhaseInfo:
    """Visible description for one pipeline phase."""

    key: str
    title: str
    subtitle: str
    detail: str


PHASES = {
    "inspect": PhaseInfo(
        "inspect",
        "Inspección inicial",
        "Valida estructura, fechas, RUC, nulos y duplicados.",
        "Lee el dataset histórico sin modificarlo y genera un reporte de calidad en outputs/inspeccion_dataset.txt.",
    ),
    "distribuido": PhaseInfo(
        "distribuido",
        "Procesamiento distribuido",
        "Ejecuta una lógica MapReduce local por bloques.",
        "Divide el histórico en bloques, resume por proveedor y consolida resultados para analítica del negocio.",
    ),
    "scraping": PhaseInfo(
        "scraping",
        "Scraping SUNAT",
        "Descarga o reutiliza el padrón reducido oficial.",
        "Ubica automáticamente el ZIP vigente, valida su formato, lo descomprime y filtra los RUC del negocio.",
    ),
    "mongodb": PhaseInfo(
        "mongodb",
        "Carga en MongoDB Atlas",
        "Actualiza proveedores SUNAT sin duplicar RUC.",
        "Crea/verifica indice unico y usa upsert para mantener la coleccion proveedores_sunat.",
    ),
    "preparar": PhaseInfo(
        "preparar",
        "Preparación de datasets",
        "Une histórico, SUNAT y variables sin fuga.",
        "Genera dataset_modelo.csv y dataset_pendientes.csv para entrenamiento y predicción.",
    ),
    "entrenar": PhaseInfo(
        "entrenar",
        "Entrenamiento ML",
        "Compara modelos, calibra y optimiza umbral.",
        "Prioriza recall, F1 y PR-AUC de incidencias con validación temporal.",
    ),
    "predecir": PhaseInfo(
        "predecir",
        "Predicción de pendientes",
        "Clasifica comprobantes pendientes por nivel de riesgo.",
        "Aplica el umbral operativo guardado y genera predicciones_pendientes.csv.",
    ),
    "documentar": PhaseInfo(
        "documentar",
        "Documentación dinámica",
        "Actualiza reportes, gráficos y resumen ejecutivo.",
        "Consolida métricas, gráficos y reportes finales del proyecto.",
    ),
}

STEP_RESULTS = {
    "inspect": [
        OUTPUTS_DIR / "inspeccion_dataset.txt",
    ],
    "distribuido": [
        OUTPUTS_DIR / "reporte_procesamiento_distribuido.txt",
        OUTPUTS_DIR / "reporte_procesamiento_distribuido.json",
        PROCESSED_DIR / "mapreduce_resumen_proveedores.csv",
    ],
    "scraping": [
        OUTPUTS_DIR / "reporte_scraping.json",
        PROCESSED_DIR / "proveedores_sunat.csv",
        PROCESSED_DIR / "ruc_no_encontrados.csv",
    ],
    "mongodb": [
        OUTPUTS_DIR / "pipeline.log",
        PROCESSED_DIR / "proveedores_sunat.csv",
    ],
    "preparar": [
        OUTPUTS_DIR / "reporte_preparacion.json",
        OUTPUTS_DIR / "matriz_correlacion_pearson.png",
        OUTPUTS_DIR / "correlacion_pearson_variables.csv",
        PROCESSED_DIR / "dataset_modelo.csv",
        PROCESSED_DIR / "dataset_pendientes.csv",
    ],
    "entrenar": [
        OUTPUTS_DIR / "reporte_modelo.txt",
        OUTPUTS_DIR / "metricas_modelo.json",
        OUTPUTS_DIR / "comparacion_modelos.csv",
        OUTPUTS_DIR / "09_matriz_confusion_optimizada.png",
    ],
    "predecir": [
        PROCESSED_DIR / "predicciones_pendientes.csv",
        OUTPUTS_DIR / "resultado_final_consola.txt",
        OUTPUTS_DIR / "16_distribucion_riesgo_pendientes.png",
    ],
    "documentar": [
        OUTPUTS_DIR / "resumen_proyecto.json",
        OUTPUTS_DIR / "matriz_correlacion_pearson.png",
        OUTPUTS_DIR / "reporte_modelo.txt",
        OUTPUTS_DIR / "09_matriz_confusion_optimizada.png",
        OUTPUTS_DIR / "06_precision_recall_curve.png",
        OUTPUTS_DIR / "07_roc_curve.png",
        OUTPUTS_DIR / "16_distribucion_riesgo_pendientes.png",
    ],
}

STEP_OUTCOMES = {
    "inspect": "Busca confirmar que el dataset histórico esté listo para el análisis: estructura, tipos, RUC válidos, fechas, nulos y duplicados.",
    "distribuido": "Busca demostrar procesamiento por bloques: mapear datos por proveedor, agruparlos y consolidar indicadores de negocio.",
    "scraping": "Busca actualizar la información tributaria de proveedores desde la fuente oficial SUNAT, sin consultas individuales por RUC.",
    "mongodb": "Busca dejar la foto tributaria actual en MongoDB Atlas, usando RUC único para evitar duplicados.",
    "preparar": "Busca unir histórico y SUNAT, crear variables útiles y separar datos definitivos de comprobantes pendientes.",
    "entrenar": "Busca comparar modelos, elegir el mejor, calibrar probabilidades y definir el umbral operativo.",
    "predecir": "Busca estimar la probabilidad de incidencia para comprobantes pendientes y clasificarlos por riesgo.",
    "documentar": "Busca consolidar resultados finales, gráficos y reportes para explicar lo logrado en la demostración.",
}

GRAPH_SUMMARY = [
    (
        OUTPUTS_DIR / "matriz_correlacion_pearson.png",
        "Matriz de calor Pearson: identifica las variables candidatas con mayor relacion lineal frente a la incidencia.",
    ),
    (
        OUTPUTS_DIR / "09_matriz_confusion_optimizada.png",
        "Matriz de confusión: resume aciertos, falsas alertas e incidencias no detectadas con el umbral optimizado.",
    ),
    (
        OUTPUTS_DIR / "06_precision_recall_curve.png",
        "Curva Precision-Recall: muestra el equilibrio entre detectar incidencias y generar alertas.",
    ),
    (
        OUTPUTS_DIR / "07_roc_curve.png",
        "Curva ROC: evalúa la capacidad general del modelo para separar aceptados e incidencias.",
    ),
    (
        OUTPUTS_DIR / "04_comparacion_modelos_pr_auc.png",
        "Comparación de modelos: muestra qué algoritmo obtuvo mejor desempeño bajo métricas relevantes.",
    ),
    (
        OUTPUTS_DIR / "16_distribucion_riesgo_pendientes.png",
        "Riesgo de pendientes: muestra cómo quedaron clasificados los comprobantes pendientes.",
    ),
]


class FactuRiskPipelineApp:
    """Desktop orchestrator with a live console and guided project screens."""

    def __init__(self) -> None:
        ensure_directories()
        self.root = Tk()
        self.root.title("FactuRisk SUNAT | Plataforma de Predicción de Incidencias")
        self.root.geometry("1440x860")
        self.root.minsize(1280, 760)
        self.root.configure(bg="#F5F8FC")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        icon_path = ASSETS_DIR / "facturisk.ico"
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except Exception:
                pass

        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.current_process: subprocess.Popen[str] | None = None
        self.start_time: float | None = None
        self.running = False

        self.selected_phase = StringVar(value="todo")
        self.status_text = StringVar(value="Listo para iniciar la demostración")
        self.current_step = StringVar(value="Ejecute el Paso 01 o el proceso completo. Las fases posteriores se habilitan al completar la anterior.")
        self.elapsed_text = StringVar(value="00:00")
        self.force_scraping = BooleanVar(value=False)

        self.phase_cards: dict[str, ttk.Frame] = {}
        self.phase_status: dict[str, StringVar] = {key: StringVar(value="Pendiente") for key in PHASE_ORDER}
        self.metric_vars: dict[str, StringVar] = {}
        self.completed_phases: set[str] = set()
        self.nav_buttons: dict[str, ttk.Button] = {}
        self.icons: dict[str, ImageTk.PhotoImage] = {}
        self.logo_image: ImageTk.PhotoImage | None = None
        self.preview_image: ImageTk.PhotoImage | None = None
        self.current_graph_index = 0

        self.setup_styles()
        self.load_icons()
        self.build_layout()
        self.reset_metric_cards()
        self.update_phase_availability()
        self.update_step_results_panel("todo")
        self.root.after(120, self.process_queue)
        self.root.after(1000, self.tick)

    def setup_styles(self) -> None:
        """Configure corporate visual styles."""

        self.colors = {
            "navy": "#071D49",
            "ink": "#102033",
            "muted": "#64748B",
            "blue": "#0067B1",
            "cyan": "#00A6CE",
            "green": "#13A538",
            "lime": "#A6CE39",
            "soft": "#F5F8FC",
            "panel": "#FFFFFF",
            "line": "#DDE7F0",
            "warning": "#F59E0B",
            "danger": "#D92D20",
        }
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Root.TFrame", background=self.colors["soft"])
        style.configure("Nav.TFrame", background=self.colors["navy"])
        style.configure("Panel.TFrame", background=self.colors["panel"])
        style.configure("Card.TFrame", background=self.colors["panel"], relief="flat")
        style.configure("Title.TLabel", background=self.colors["soft"], foreground=self.colors["ink"], font=("Segoe UI Semibold", 22))
        style.configure("Subtitle.TLabel", background=self.colors["soft"], foreground=self.colors["muted"], font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background=self.colors["panel"], foreground=self.colors["ink"], font=("Segoe UI Semibold", 13))
        style.configure("Body.TLabel", background=self.colors["panel"], foreground=self.colors["muted"], font=("Segoe UI", 9))
        style.configure("Small.TLabel", background=self.colors["panel"], foreground=self.colors["muted"], font=("Segoe UI", 8))
        style.configure("NavTitle.TLabel", background=self.colors["navy"], foreground="#FFFFFF", font=("Segoe UI Semibold", 16))
        style.configure("NavText.TLabel", background=self.colors["navy"], foreground="#BFD5E8", font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=self.colors["panel"], foreground=self.colors["blue"], font=("Segoe UI Semibold", 10))
        style.configure("Step.TButton", background="#0D2A5B", foreground="#FFFFFF", borderwidth=0, focusthickness=0, font=("Segoe UI Semibold", 9), padding=(10, 7), anchor="w")
        style.map("Step.TButton", background=[("active", "#123B78"), ("disabled", "#19335F")], foreground=[("disabled", "#7E94B1")])
        style.configure("Primary.TButton", background=self.colors["blue"], foreground="#FFFFFF", borderwidth=0, focusthickness=0, font=("Segoe UI Semibold", 10), padding=(16, 10))
        style.map("Primary.TButton", background=[("active", "#004F8D"), ("disabled", "#A9B9C7")])
        style.configure("Secondary.TButton", background="#EAF3FA", foreground=self.colors["blue"], borderwidth=0, focusthickness=0, font=("Segoe UI Semibold", 9), padding=(12, 8))
        style.map("Secondary.TButton", background=[("active", "#D9EBF6")])
        style.configure("Danger.TButton", background=self.colors["danger"], foreground="#FFFFFF", borderwidth=0, focusthickness=0, font=("Segoe UI Semibold", 9), padding=(12, 8))
        style.configure("TCheckbutton", background=self.colors["panel"], foreground=self.colors["ink"], font=("Segoe UI", 9))
        style.configure("Horizontal.TProgressbar", troughcolor="#EAF0F6", background=self.colors["cyan"], bordercolor="#EAF0F6", lightcolor=self.colors["cyan"], darkcolor=self.colors["cyan"])

    def load_icons(self) -> None:
        """Load local PNG icons for navigation and metric cards."""

        icon_files = {
            "todo": "icon_todo.png",
            "inspect": "icon_01_inspect.png",
            "distribuido": "icon_02_mapreduce.png",
            "scraping": "icon_03_sunat.png",
            "mongodb": "icon_04_mongodb.png",
            "preparar": "icon_05_prepare.png",
            "entrenar": "icon_06_train.png",
            "predecir": "icon_07_predict.png",
            "documentar": "icon_08_docs.png",
            "modelo": "metric_model.png",
            "recall": "metric_recall.png",
            "f1": "metric_f1.png",
            "pendientes": "metric_pending.png",
            "riesgo_alto": "metric_high.png",
        }
        for key, filename in icon_files.items():
            path = ASSETS_DIR / filename
            if not path.exists():
                continue
            try:
                image = Image.open(path).convert("RGBA").resize((28, 28), Image.Resampling.LANCZOS)
                self.icons[key] = ImageTk.PhotoImage(image)
            except Exception:
                pass

    def build_layout(self) -> None:
        """Create all visible sections."""

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        nav = ttk.Frame(self.root, style="Nav.TFrame", width=318)
        nav.grid(row=0, column=0, sticky="nsew")
        nav.grid_propagate(False)
        nav.columnconfigure(0, weight=1)

        self.build_nav(nav)

        main = ttk.Frame(self.root, style="Root.TFrame")
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)
        main.rowconfigure(3, weight=1)

        self.build_header(main)
        self.build_metric_strip(main)
        self.build_phase_area(main)
        self.build_console(main)

    def build_nav(self, parent: ttk.Frame) -> None:
        """Build the left navigation panel."""

        logo_frame = ttk.Frame(parent, style="Nav.TFrame")
        logo_frame.grid(row=0, column=0, sticky="ew", padx=24, pady=(22, 14))
        logo_frame.columnconfigure(0, weight=1)

        logo_path = ASSETS_DIR / "logo.png"
        if logo_path.exists():
            try:
                image = Image.open(logo_path).convert("RGBA")
                image.thumbnail((118, 54))
                self.logo_image = ImageTk.PhotoImage(image)
                ttk.Label(logo_frame, image=self.logo_image, background=self.colors["navy"]).grid(row=0, column=0, sticky="w")
            except Exception:
                ttk.Label(logo_frame, text="FactuRisk", style="NavTitle.TLabel").grid(row=0, column=0, sticky="w")
        else:
            ttk.Label(logo_frame, text="FactuRisk", style="NavTitle.TLabel").grid(row=0, column=0, sticky="w")

        ttk.Label(logo_frame, text="ML SUNAT Control Tower", style="NavTitle.TLabel").grid(row=1, column=0, sticky="w", pady=(14, 2))
        ttk.Label(logo_frame, text="Predicción preventiva de incidencias", style="NavText.TLabel").grid(row=2, column=0, sticky="w")

        nav_items = [
            ("todo", "Todo el proceso", "Ejecución guiada completa"),
            ("inspect", "Paso 01 · Inspección", "Calidad inicial del CSV"),
            ("distribuido", "Paso 02 · MapReduce", "Procesamiento por bloques"),
            ("scraping", "Paso 03 · SUNAT", "Padrón reducido oficial"),
            ("mongodb", "Paso 04 · MongoDB", "Carga NoSQL por RUC"),
            ("preparar", "Paso 05 · Preparación", "Datasets y variables"),
            ("entrenar", "Paso 06 · Entrenamiento", "Modelos y umbral"),
            ("predecir", "Paso 07 · Predicción", "Pendientes por riesgo"),
            ("documentar", "Paso 08 · Reportes", "Resultados finales"),
        ]
        items_frame = ttk.Frame(parent, style="Nav.TFrame")
        items_frame.grid(row=1, column=0, sticky="ew", padx=18)
        for idx, (key, label, desc) in enumerate(nav_items):
            button = ttk.Button(
                items_frame,
                text=f"{label}\n{desc}",
                image=self.icons.get(key),
                compound="left",
                style="Step.TButton",
                command=lambda value=key: self.select_phase(value),
            )
            button.grid(row=idx, column=0, sticky="ew", pady=4)
            self.nav_buttons[key] = button

        footer = ttk.Frame(parent, style="Nav.TFrame")
        footer.grid(row=2, column=0, sticky="sew", padx=24, pady=24)
        parent.rowconfigure(2, weight=1)
        ttk.Label(footer, text="Fuente externa: SUNAT", style="NavText.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(footer, text="Base NoSQL: MongoDB Atlas", style="NavText.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(footer, text="Validación: temporal y reproducible", style="NavText.TLabel").grid(row=2, column=0, sticky="w")

    def build_header(self, parent: ttk.Frame) -> None:
        """Build top title and action buttons."""

        header = ttk.Frame(parent, style="Root.TFrame")
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(22, 12))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Plataforma de Analítica Predictiva FactuRisk SUNAT", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Inspección, SUNAT, MongoDB, procesamiento distribuido, Machine Learning y predicción de comprobantes pendientes.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        actions = ttk.Frame(header, style="Root.TFrame")
        actions.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Checkbutton(actions, text="Forzar scraping SUNAT", variable=self.force_scraping).grid(row=0, column=0, padx=(0, 10))
        self.run_button = ttk.Button(actions, text="Ejecutar paso", style="Primary.TButton", command=self.run_selected)
        self.run_button.grid(row=0, column=1, padx=4)
        self.todo_button = ttk.Button(actions, text="Ejecutar todo", style="Primary.TButton", command=lambda: self.run_phase_group("todo"))
        self.todo_button.grid(row=0, column=2, padx=4)
        self.clear_all_button = ttk.Button(actions, text="Limpiar todo", style="Danger.TButton", command=self.clean_generated_data)
        self.clear_all_button.grid(row=0, column=3, padx=4)
        self.stop_button = ttk.Button(actions, text="Detener", style="Danger.TButton", command=self.stop_process)
        self.stop_button.grid(row=0, column=4, padx=(4, 0))

    def build_metric_strip(self, parent: ttk.Frame) -> None:
        """Build top metric cards."""

        strip = ttk.Frame(parent, style="Root.TFrame")
        strip.grid(row=1, column=0, sticky="ew", padx=28, pady=(0, 14))
        for col in range(5):
            strip.columnconfigure(col, weight=1)

        metrics = [
            ("Modelo", "modelo", ""),
            ("Recall incidencias", "recall", ""),
            ("F1 incidencias", "f1", ""),
            ("Pendientes evaluados", "pendientes", ""),
            ("Riesgo alto", "riesgo_alto", ""),
        ]
        for col, (title, key, default) in enumerate(metrics):
            self.metric_vars[key] = StringVar(value=default)
            card = ttk.Frame(strip, style="Panel.TFrame", padding=(16, 12))
            card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 8, 0))
            row = ttk.Frame(card, style="Panel.TFrame")
            row.pack(anchor="w", fill="x")
            if key in self.icons:
                ttk.Label(row, image=self.icons[key], background=self.colors["panel"]).pack(side="left", padx=(0, 8))
            text_box = ttk.Frame(row, style="Panel.TFrame")
            text_box.pack(side="left", fill="x", expand=True)
            ttk.Label(text_box, text=title, style="Small.TLabel").pack(anchor="w")
            ttk.Label(text_box, textvariable=self.metric_vars[key], style="PanelTitle.TLabel").pack(anchor="w", pady=(4, 0))

    def build_phase_area(self, parent: ttk.Frame) -> None:
        """Build phase cards and explanatory panel."""

        area = ttk.Frame(parent, style="Root.TFrame")
        area.grid(row=2, column=0, sticky="nsew", padx=28, pady=(0, 14))
        area.columnconfigure(0, weight=2)
        area.columnconfigure(1, weight=1)
        area.rowconfigure(0, weight=1)

        cards = ttk.Frame(area, style="Root.TFrame")
        cards.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        for col in range(2):
            cards.columnconfigure(col, weight=1)

        for idx, key in enumerate(PHASE_ORDER):
            info = PHASES[key]
            card = ttk.Frame(cards, style="Panel.TFrame", padding=(14, 12))
            card.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=6, pady=6)
            card.columnconfigure(0, weight=1)
            title_row = ttk.Frame(card, style="Panel.TFrame")
            title_row.grid(row=0, column=0, sticky="ew")
            if key in self.icons:
                ttk.Label(title_row, image=self.icons[key], background=self.colors["panel"]).pack(side="left", padx=(0, 8))
            ttk.Label(title_row, text=f"Paso {idx + 1:02d} · {info.title}", style="PanelTitle.TLabel").pack(side="left", anchor="w")
            ttk.Label(card, text=info.subtitle, style="Body.TLabel", wraplength=390).grid(row=1, column=0, sticky="w", pady=(7, 6))
            ttk.Label(card, textvariable=self.phase_status[key], style="Status.TLabel").grid(row=2, column=0, sticky="w")
            card.bind("<Button-1>", lambda _event, phase=key: self.select_phase(phase))
            self.phase_cards[key] = card

        detail = ttk.Frame(area, style="Panel.TFrame", padding=(18, 16))
        detail.grid(row=0, column=1, sticky="nsew")
        detail.columnconfigure(0, weight=1)
        ttk.Label(detail, text="Guía de ejecución", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(detail, textvariable=self.current_step, style="Body.TLabel", wraplength=380).grid(row=1, column=0, sticky="ew", pady=(8, 18))
        ttk.Label(detail, text="Estado actual", style="Small.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Label(detail, textvariable=self.status_text, style="PanelTitle.TLabel", wraplength=380).grid(row=3, column=0, sticky="w", pady=(2, 14))
        ttk.Label(detail, text="Tiempo transcurrido", style="Small.TLabel").grid(row=4, column=0, sticky="w")
        ttk.Label(detail, textvariable=self.elapsed_text, style="PanelTitle.TLabel").grid(row=5, column=0, sticky="w", pady=(2, 18))
        self.progress = ttk.Progressbar(detail, mode="determinate", maximum=100, style="Horizontal.TProgressbar")
        self.progress.grid(row=6, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(detail, text="Resultado esperado", style="Small.TLabel").grid(row=7, column=0, sticky="w", pady=(4, 0))
        self.result_text = StringVar(value="")
        ttk.Label(detail, textvariable=self.result_text, style="Body.TLabel", wraplength=380).grid(row=8, column=0, sticky="ew", pady=(4, 10))

        self.preview_label = ttk.Label(detail, text="", style="Body.TLabel", background=self.colors["panel"])
        self.preview_label.grid(row=9, column=0, sticky="ew", pady=(0, 8))

        graph_buttons = ttk.Frame(detail, style="Panel.TFrame")
        graph_buttons.grid(row=10, column=0, sticky="ew", pady=(0, 8))
        graph_buttons.columnconfigure(0, weight=1)
        graph_buttons.columnconfigure(1, weight=1)
        self.prev_graph_button = ttk.Button(graph_buttons, text="Gráfico anterior", style="Secondary.TButton", command=lambda: self.change_graph(-1))
        self.prev_graph_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.next_graph_button = ttk.Button(graph_buttons, text="Gráfico siguiente", style="Secondary.TButton", command=lambda: self.change_graph(1))
        self.next_graph_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.open_step_button = ttk.Button(detail, text="Abrir resultado del paso", style="Secondary.TButton", command=self.open_step_result)
        self.open_step_button.grid(row=11, column=0, sticky="ew", pady=4)
        self.open_predictions_button = ttk.Button(detail, text="Abrir predicciones", style="Secondary.TButton", command=self.open_predictions)
        self.open_predictions_button.grid(row=12, column=0, sticky="ew", pady=4)
        ttk.Button(detail, text="Actualizar resultados", style="Secondary.TButton", command=self.refresh_metrics).grid(row=13, column=0, sticky="ew", pady=4)
        ttk.Button(detail, text="Limpiar todo", style="Danger.TButton", command=self.clean_generated_data).grid(row=14, column=0, sticky="ew", pady=4)

    def build_console(self, parent: ttk.Frame) -> None:
        """Build the bottom live console."""

        console_panel = ttk.Frame(parent, style="Panel.TFrame", padding=(14, 10))
        console_panel.grid(row=3, column=0, sticky="nsew", padx=28, pady=(0, 22))
        console_panel.columnconfigure(0, weight=1)
        console_panel.rowconfigure(1, weight=1)

        top = ttk.Frame(console_panel, style="Panel.TFrame")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top.columnconfigure(0, weight=1)
        ttk.Label(top, text="Consola de ejecución en tiempo real", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(top, text="Limpiar consola", style="Secondary.TButton", command=self.clear_console).grid(row=0, column=1, sticky="e")

        self.console = ScrolledText(
            console_panel,
            height=11,
            bg="#07111F",
            fg="#DBEAFE",
            insertbackground="#FFFFFF",
            relief="flat",
            font=("Consolas", 9),
            padx=12,
            pady=10,
        )
        self.console.grid(row=1, column=0, sticky="nsew")
        self.console.tag_configure("ok", foreground="#8EE6A1")
        self.console.tag_configure("warn", foreground="#FFD166")
        self.console.tag_configure("err", foreground="#FF8A80")
        self.console.tag_configure("phase", foreground="#7DD3FC")
        self.log("Sistema listo. Ejecute el Paso 01 o el proceso completo.", "ok")

    def select_phase(self, phase: str) -> None:
        """Select a phase and update explanatory text."""

        self.selected_phase.set(phase)
        if phase == "todo":
            text = (
                "Proceso completo: inspección, procesamiento distribuido, scraping SUNAT, MongoDB, "
                "preparación, entrenamiento, predicción y documentación."
            )
        else:
            info = PHASES[phase]
            text = f"{info.title}: {info.detail}\n\nObjetivo: {STEP_OUTCOMES.get(phase, '')}"
        self.current_step.set(text)
        self.status_text.set(f"Seleccionado: {phase}")
        self.update_step_results_panel(phase)

    def update_step_results_panel(self, phase: str) -> None:
        """Update contextual result text and graph preview for the selected phase."""

        if phase == "todo":
            self.result_text.set("Ejecuta todos los pasos en orden. Los indicadores superiores aparecerán al completar entrenamiento y predicción.")
            self.set_graph_controls(False)
            self.preview_label.configure(image="", text="")
            self.open_step_button.configure(text="Abrir carpeta de resultados", state="normal", command=self.open_outputs)
            self.open_predictions_button.configure(state="disabled")
            return

        result_paths = STEP_RESULTS.get(phase, [])
        existing = [path for path in result_paths if path.exists()]
        if existing:
            names = "\n".join(f"• {path.name}" for path in existing[:4])
            self.result_text.set(f"Resultados disponibles para este paso:\n{names}")
            self.open_step_button.configure(text="Abrir resultado del paso", state="normal", command=self.open_step_result)
        else:
            expected = "\n".join(f"• {path.name}" for path in result_paths[:4]) or "Este paso mostrará su avance principalmente en la consola."
            self.result_text.set(f"Aún no hay resultados generados para este paso.\nSe espera:\n{expected}")
            self.open_step_button.configure(text="Abrir carpeta de resultados", state="normal", command=self.open_outputs)

        self.open_predictions_button.configure(state="normal" if phase == "predecir" and (PROCESSED_DIR / "predicciones_pendientes.csv").exists() else "disabled")

        if phase == "documentar":
            self.current_graph_index = 0
            self.show_current_graph()
            self.set_graph_controls(True)
        else:
            self.set_graph_controls(False)
            self.preview_label.configure(image="", text="")

    def set_graph_controls(self, enabled: bool) -> None:
        """Enable or disable graph navigation buttons."""

        state = "normal" if enabled else "disabled"
        self.prev_graph_button.configure(state=state)
        self.next_graph_button.configure(state=state)

    def show_current_graph(self) -> None:
        """Show one generated graph with a short interpretation."""

        available = [(path, text) for path, text in GRAPH_SUMMARY if path.exists()]
        if not available:
            self.preview_label.configure(image="", text="Al completar el flujo se mostrarán aquí las gráficas principales del modelo.")
            return
        self.current_graph_index %= len(available)
        path, explanation = available[self.current_graph_index]
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((360, 170), Image.Resampling.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(image)
            self.preview_label.configure(
                image=self.preview_image,
                text=f"\n{explanation}",
                compound="top",
                wraplength=360,
                justify="left",
            )
        except Exception as exc:
            self.preview_label.configure(image="", text=f"No se pudo cargar el gráfico: {exc}")

    def change_graph(self, direction: int) -> None:
        """Move graph preview backward or forward."""

        self.current_graph_index += direction
        self.show_current_graph()

    def can_run_phase(self, phase: str) -> tuple[bool, str]:
        """Validate sequential execution before allowing one phase."""

        if phase == "todo":
            return True, ""
        index = PHASE_ORDER.index(phase)
        missing = [PHASES[key].title for key in PHASE_ORDER[:index] if key not in self.completed_phases]
        if missing:
            return False, f"Antes debe completar: {', '.join(missing)}."
        return True, ""

    def update_phase_availability(self) -> None:
        """Enable only the next allowed phase to avoid broken executions."""

        for key, button in self.nav_buttons.items():
            if key == "todo":
                button.configure(state="normal")
                continue
            allowed, _ = self.can_run_phase(key)
            button.configure(state="normal" if allowed else "disabled")

    def build_command(self, phase: str) -> list[str]:
        """Build a subprocess command for source and PyInstaller modes."""

        if getattr(sys, "frozen", False):
            command = [sys.executable, "--worker-phase", phase]
            if phase == "scraping" and self.force_scraping.get():
                command.append("--force-scraping")
            return command
        script = PHASE_SCRIPTS[phase]
        command = [sys.executable, str(script)]
        if phase == "scraping" and self.force_scraping.get():
            command.append("--force-download")
        if phase == "resumen":
            command.append("--no-gui")
        return command

    def run_selected(self) -> None:
        """Run the currently selected phase or full flow."""

        self.run_phase_group(self.selected_phase.get())

    def run_phase_group(self, selected: str) -> None:
        """Start a background worker for one phase or the full pipeline."""

        if self.running:
            messagebox.showinfo("Proceso en ejecución", "Ya hay una fase en ejecución.")
            return
        allowed, reason = self.can_run_phase(selected)
        if not allowed:
            messagebox.showwarning("Paso bloqueado", reason)
            self.status_text.set(reason)
            self.log(f"Paso bloqueado: {reason}", "warn")
            return
        phases = PHASE_ORDER if selected == "todo" else [selected]
        self.running = True
        self.start_time = time.perf_counter()
        self.progress.configure(value=0)
        self.set_buttons_state(False)
        if selected == "todo":
            for key in PHASE_ORDER:
                self.phase_status[key].set("Pendiente")
            self.completed_phases.clear()
            self.reset_metric_cards()
        self.worker_thread = threading.Thread(target=self.worker, args=(phases,), daemon=True)
        self.worker_thread.start()

    def worker(self, phases: list[str]) -> None:
        """Run subprocess phases and stream their output to the UI queue."""

        total = len(phases)
        for index, phase in enumerate(phases, start=1):
            if not self.running:
                break
            self.queue.put(("phase_start", phase))
            command = self.build_command(phase)
            self.queue.put(("log", (f"INICIO DE FASE: {phase}", "phase")))
            self.queue.put(("log", (f"Comando: {' '.join(command)}", "phase")))
            try:
                self.current_process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert self.current_process.stdout is not None
                for line in self.current_process.stdout:
                    self.queue.put(("log", (line.rstrip("\n"), self.tag_for_line(line))))
                code = self.current_process.wait()
            except Exception as exc:
                code = 1
                self.queue.put(("log", (f"FASE FALLIDA: {phase}. Detalle: {exc}", "err")))
            finally:
                self.current_process = None

            if code != 0:
                self.queue.put(("phase_failed", phase))
                self.queue.put(("done", False))
                return
            self.queue.put(("phase_done", (phase, index, total)))

        self.queue.put(("done", True))

    def tag_for_line(self, line: str) -> str:
        """Select a console color tag based on text content."""

        lower = line.lower()
        if "fallida" in lower or "error" in lower or "traceback" in lower:
            return "err"
        if "warning" in lower or "advertencia" in lower:
            return "warn"
        if "completada" in lower or "correctamente" in lower or "ok" in lower:
            return "ok"
        if "fase" in lower or "inicio" in lower:
            return "phase"
        return ""

    def process_queue(self) -> None:
        """Process messages from the background worker."""

        try:
            while True:
                event, payload = self.queue.get_nowait()
                if event == "log":
                    message, tag = payload
                    self.log(message, tag)
                elif event == "phase_start":
                    self.phase_status[payload].set("En ejecución")
                    self.status_text.set(f"Ejecutando: {PHASES[payload].title}")
                    self.current_step.set(PHASES[payload].detail)
                elif event == "phase_done":
                    phase, index, total = payload
                    self.completed_phases.add(phase)
                    self.phase_status[phase].set("Completada")
                    self.progress.configure(value=(index / total) * 100)
                    self.update_phase_availability()
                    if self.selected_phase.get() == phase:
                        self.update_step_results_panel(phase)
                    self.log(f"FASE COMPLETADA: {phase}", "ok")
                elif event == "phase_failed":
                    self.phase_status[payload].set("Fallida")
                    self.status_text.set(f"Fase fallida: {payload}")
                elif event == "done":
                    success = bool(payload)
                    self.running = False
                    self.set_buttons_state(True)
                    if success:
                        self.progress.configure(value=100)
                        if {"entrenar", "predecir"}.issubset(self.completed_phases):
                            self.refresh_metrics()
                        self.status_text.set("Proceso completado correctamente")
                        self.current_step.set("Los resultados fueron actualizados. Revise las métricas y archivos generados.")
                        self.log("PROCESO FINALIZADO CORRECTAMENTE", "ok")
                    else:
                        self.status_text.set("Proceso detenido por error")
                        self.log("PROCESO DETENIDO. Revise la consola.", "err")
        except queue.Empty:
            pass
        self.root.after(120, self.process_queue)

    def tick(self) -> None:
        """Update elapsed time while running."""

        if self.running and self.start_time is not None:
            elapsed = int(time.perf_counter() - self.start_time)
            minutes, seconds = divmod(elapsed, 60)
            self.elapsed_text.set(f"{minutes:02d}:{seconds:02d}")
        self.root.after(1000, self.tick)

    def log(self, message: str, tag: str = "") -> None:
        """Append a message to the live console."""

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.configure(state="normal")
        self.console.insert("end", f"[{timestamp}] {message}\n", tag)
        self.console.see("end")
        self.console.configure(state="disabled")

    def clear_console(self) -> None:
        """Clear the live console."""

        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def set_buttons_state(self, enabled: bool) -> None:
        """Enable or disable action buttons."""

        state = "normal" if enabled else "disabled"
        self.run_button.configure(state=state)
        self.todo_button.configure(state=state)
        self.clear_all_button.configure(state=state)
        if enabled:
            self.update_phase_availability()

    def stop_process(self) -> None:
        """Terminate the active subprocess if one is running."""

        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
                self.current_process.wait(timeout=5)
            self.log("Solicitud de detención enviada al proceso activo.", "warn")
        self.running = False
        self.set_buttons_state(True)

    def reset_metric_cards(self) -> None:
        """Clear result cards for a fresh demonstration."""

        for key in ["modelo", "recall", "f1", "pendientes", "riesgo_alto"]:
            if key in self.metric_vars:
                self.metric_vars[key].set("")

    def clean_generated_data(self) -> None:
        """Delete generated artifacts while preserving raw CSV and credentials."""

        if self.running:
            messagebox.showwarning("Proceso en ejecución", "Detenga el proceso antes de limpiar resultados.")
            return
        answer = messagebox.askyesno(
            "Limpiar todo",
            "Se eliminarán resultados, modelos, reportes, descargas SUNAT y datos procesados. "
            "El dataset histórico y la configuración se conservarán. ¿Desea continuar?",
        )
        if not answer:
            return
        try:
            skipped = self.safe_clean_generated_artifacts()
            self.completed_phases.clear()
            for key in PHASE_ORDER:
                self.phase_status[key].set("Pendiente")
            self.reset_metric_cards()
            self.progress.configure(value=0)
            self.elapsed_text.set("00:00")
            self.status_text.set("Resultados limpiados. Inicie desde el Paso 01.")
            self.current_step.set("Ejecute el Paso 01 o el proceso completo. Las métricas aparecerán al finalizar.")
            self.clear_console()
            self.log("Limpieza completada. Se conservaron el dataset histórico y la configuración.", "ok")
            if skipped:
                self.log("Algunos archivos estaban en uso y fueron omitidos: " + ", ".join(path.name for path in skipped), "warn")
            self.update_phase_availability()
            self.update_step_results_panel(self.selected_phase.get())
        except Exception as exc:
            messagebox.showerror("Error al limpiar", str(exc))
            self.log(f"Error al limpiar resultados: {exc}", "err")

    def safe_clean_generated_artifacts(self) -> list[Path]:
        """Safely remove generated contents inside known project folders."""

        root = PROJECT_ROOT.resolve()
        skipped: list[Path] = []
        folders = [
            PROCESSED_DIR,
            PROJECT_ROOT / "data" / "models",
            PROJECT_ROOT / "data" / "raw" / "sunat",
            OUTPUTS_DIR,
            PROJECT_ROOT / "logs",
            PROJECT_ROOT / "catboost_info",
        ]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
            resolved = folder.resolve()
            if root not in resolved.parents and resolved != root:
                raise RuntimeError(f"Ruta fuera del proyecto: {resolved}")
            for item in resolved.iterdir():
                removed = False
                for _attempt in range(6):
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        removed = True
                        break
                    except PermissionError:
                        time.sleep(0.5)
                    except FileNotFoundError:
                        removed = True
                        break
                if not removed:
                    skipped.append(item)
        return skipped

    def refresh_metrics(self) -> None:
        """Refresh top cards from generated artifacts."""

        metrics_path = OUTPUTS_DIR / "metricas_modelo.json"
        predictions_path = PROCESSED_DIR / "predicciones_pendientes.csv"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                test = metrics.get("metricas_prueba", {})
                self.metric_vars["modelo"].set(str(metrics.get("mejor_modelo", "N/D"))[:24])
                self.metric_vars["recall"].set(self.percent(test.get("recall_clase_1")))
                self.metric_vars["f1"].set(self.percent(test.get("f1_clase_1")))
            except Exception as exc:
                self.log(f"No se pudieron leer métricas: {exc}", "warn")
        if predictions_path.exists():
            try:
                pred = pd.read_csv(predictions_path, sep=",", encoding="utf-8-sig", usecols=lambda c: c in {"Nivel_Riesgo"}, low_memory=False)
                self.metric_vars["pendientes"].set(f"{len(pred):,}")
                high = int((pred["Nivel_Riesgo"] == "Alto").sum()) if "Nivel_Riesgo" in pred.columns else 0
                self.metric_vars["riesgo_alto"].set(f"{high:,}")
            except Exception as exc:
                self.log(f"No se pudieron leer predicciones: {exc}", "warn")

    @staticmethod
    def percent(value: Any) -> str:
        """Format decimal as a percent card."""

        try:
            return f"{float(value) * 100:.2f}%"
        except (TypeError, ValueError):
            return "N/D"

    def open_outputs(self) -> None:
        """Open outputs directory."""

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(OUTPUTS_DIR)

    def open_predictions(self) -> None:
        """Open predictions CSV if available."""

        path = PROCESSED_DIR / "predicciones_pendientes.csv"
        if path.exists():
            os.startfile(path)
        else:
            messagebox.showwarning("Archivo no encontrado", f"No existe: {path}")

    def open_step_result(self) -> None:
        """Open the first available artifact that belongs to the selected step."""

        phase = self.selected_phase.get()
        if phase == "todo":
            self.open_outputs()
            return
        result_paths = STEP_RESULTS.get(phase, [])
        existing = [path for path in result_paths if path.exists()]
        if existing:
            os.startfile(existing[0])
            return
        self.open_outputs()

    def close(self) -> None:
        """Close the app safely."""

        if self.running:
            answer = messagebox.askyesno("Proceso en ejecución", "Hay un proceso en ejecución. ¿Desea detenerlo y cerrar?")
            if not answer:
                return
            self.stop_process()
        self.root.destroy()

    def run(self) -> None:
        """Start the Tkinter event loop."""

        self.root.mainloop()


def lanzar_gui() -> None:
    """Launch the premium graphical interface."""

    app = FactuRiskPipelineApp()
    app.run()


if __name__ == "__main__":
    if "--worker-phase" in sys.argv:
        from main import run_worker_phase

        phase_index = sys.argv.index("--worker-phase") + 1
        if phase_index >= len(sys.argv):
            raise SystemExit("Falta la fase para --worker-phase")
        raise SystemExit(run_worker_phase(sys.argv[phase_index], force_scraping="--force-scraping" in sys.argv))
    lanzar_gui()
