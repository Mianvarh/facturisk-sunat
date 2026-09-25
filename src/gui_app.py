"""Desktop interface for the FactuRisk SUNAT pipeline: ML module and risk dashboard."""

from __future__ import annotations

import ctypes
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, StringVar, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

import pandas as pd
from PIL import ImageTk

from paths import ensure_directories, get_application_root
from theme import FONT_MONO, PALETTE, configure_styles, model_display_name
from ui_widgets import NavItem, ResponsiveGrid, ScrollableFrame, card, load_icon

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

# Below this window width the sidebar collapses to icons and header actions wrap.
COMPACT_WIDTH = 1180
# Minimum content width to show phase cards and the detail panel side by side.
WIDE_DETAIL_WIDTH = 1000


@dataclass(frozen=True)
class PhaseInfo:
    """Visible description for one pipeline phase."""

    key: str
    title: str
    subtitle: str
    detail: str
    outcome: str


PHASES = {
    "inspect": PhaseInfo(
        "inspect",
        "Inspección inicial",
        "Valida estructura, fechas, RUC, nulos y duplicados.",
        "Lee el dataset histórico sin modificarlo y genera un reporte de calidad en outputs/inspeccion_dataset.txt.",
        "Confirmar que el histórico esté listo para el análisis: estructura, tipos, RUC válidos, fechas, nulos y duplicados.",
    ),
    "distribuido": PhaseInfo(
        "distribuido",
        "Procesamiento distribuido",
        "Lógica MapReduce local por bloques.",
        "Divide el histórico en bloques, resume por proveedor y consolida resultados para analítica del negocio.",
        "Demostrar procesamiento por bloques: mapear datos por proveedor, agruparlos y consolidar indicadores.",
    ),
    "scraping": PhaseInfo(
        "scraping",
        "Scraping SUNAT",
        "Descarga o reutiliza el padrón reducido oficial.",
        "Ubica el ZIP vigente del padrón reducido, valida su formato, lo descomprime y filtra los RUC del negocio.",
        "Actualizar la información tributaria de los proveedores desde la fuente oficial, sin consultas individuales por RUC.",
    ),
    "mongodb": PhaseInfo(
        "mongodb",
        "Carga en MongoDB",
        "Upsert de proveedores SUNAT por RUC.",
        "Crea o verifica un índice único y usa upsert para mantener la colección proveedores_sunat. Sin configuración, se omite y se usa el respaldo local.",
        "Dejar la foto tributaria actual en MongoDB, usando el RUC como clave única para evitar duplicados.",
    ),
    "preparar": PhaseInfo(
        "preparar",
        "Preparación de datasets",
        "Une histórico y SUNAT con variables sin fuga.",
        "Genera dataset_modelo.csv y dataset_pendientes.csv para entrenamiento y predicción.",
        "Unir histórico y SUNAT, crear variables útiles y separar los comprobantes definitivos de los pendientes.",
    ),
    "entrenar": PhaseInfo(
        "entrenar",
        "Entrenamiento ML",
        "Compara modelos, calibra y optimiza el umbral.",
        "Prioriza recall, F1 y PR-AUC de incidencias con validación temporal.",
        "Comparar modelos, elegir el mejor, calibrar probabilidades y definir el umbral operativo.",
    ),
    "predecir": PhaseInfo(
        "predecir",
        "Predicción de pendientes",
        "Clasifica los pendientes por nivel de riesgo.",
        "Aplica el modelo y el umbral guardados y genera predicciones_pendientes.csv.",
        "Estimar la probabilidad de incidencia de cada comprobante pendiente y clasificarlo por riesgo.",
    ),
    "documentar": PhaseInfo(
        "documentar",
        "Reportes y documentación",
        "Actualiza reportes, gráficos y resumen.",
        "Consolida métricas, gráficos y reportes finales del proyecto en docs/ y outputs/.",
        "Consolidar resultados finales, gráficos y reportes para explicar lo logrado.",
    ),
}

STEP_RESULTS = {
    "inspect": [OUTPUTS_DIR / "inspeccion_dataset.txt"],
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
    "mongodb": [OUTPUTS_DIR / "pipeline.log"],
    "preparar": [
        OUTPUTS_DIR / "reporte_preparacion.json",
        OUTPUTS_DIR / "matriz_correlacion_pearson.png",
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
        OUTPUTS_DIR / "16_distribucion_riesgo_pendientes.png",
    ],
    "documentar": [OUTPUTS_DIR / "resumen_proyecto.json"],
}

# Artifact that proves a phase already ran in a previous session (None: no artifact).
PHASE_EVIDENCE: dict[str, Path | None] = {
    "inspect": OUTPUTS_DIR / "inspeccion_dataset.txt",
    "distribuido": OUTPUTS_DIR / "reporte_procesamiento_distribuido.json",
    "scraping": None,
    "mongodb": None,
    "preparar": PROCESSED_DIR / "dataset_modelo.csv",
    "entrenar": OUTPUTS_DIR / "metricas_modelo.json",
    "predecir": PROCESSED_DIR / "predicciones_pendientes.csv",
    "documentar": OUTPUTS_DIR / "resumen_proyecto.json",
}

STATUS_STYLES = {
    "Pendiente": "Pending.Status.TLabel",
    "En ejecución": "Running.Status.TLabel",
    "Completada": "Done.Status.TLabel",
    "Fallida": "Failed.Status.TLabel",
}


def enable_high_dpi() -> None:
    """Render crisp text on scaled Windows displays."""

    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


class FactuRiskApp:
    """Desktop orchestrator with a live console, guided phases and a risk dashboard."""

    def __init__(self) -> None:
        ensure_directories()
        enable_high_dpi()
        self.root = tk.Tk()
        self.root.title("FactuRisk SUNAT · Predicción de incidencias en comprobantes")
        self.root.geometry("1400x880")
        self.root.minsize(760, 560)
        self.root.configure(background=PALETTE["background"])
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        icon_path = ASSETS_DIR / "facturisk.ico"
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except tk.TclError:
                pass

        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.current_process: subprocess.Popen[str] | None = None
        self.start_time: float | None = None
        self.running = False
        self.compact: bool | None = None
        self.detail_side: bool | None = None

        self.selected_phase = StringVar(value="todo")
        self.status_text = StringVar(value="Listo para iniciar")
        self.elapsed_text = StringVar(value="00:00")
        self.force_scraping = BooleanVar(value=False)
        self.phase_status: dict[str, StringVar] = {key: StringVar(value="Pendiente") for key in PHASE_ORDER}
        self.status_labels: dict[str, list[ttk.Label]] = {key: [] for key in PHASE_ORDER}
        self.metric_vars: dict[str, StringVar] = {}
        self.completed_phases: set[str] = set()
        self.nav_items: dict[str, NavItem] = {}
        self.icons: dict[str, ImageTk.PhotoImage] = {}

        configure_styles(ttk.Style(self.root))
        self.load_icons()
        self.build_layout()
        self.restore_completed_phases()
        self.select_phase("todo")
        self.refresh_metrics()
        self.root.bind("<Configure>", self.on_resize)
        self.root.after(120, self.process_queue)
        self.root.after(1000, self.tick)

    # ------------------------------------------------------------------- setup

    def load_icons(self) -> None:
        """Load navigation (light) and card (tinted tile) icons."""

        names = [*PHASE_ORDER, "todo", "dashboard", "modelo", "recall", "f1", "pendientes", "riesgo_alto", "importe"]
        for name in names:
            nav = load_icon(ASSETS_DIR / f"nav_{name}.png", 20)
            tile = load_icon(ASSETS_DIR / f"tile_{name}.png", 38)
            if nav:
                self.icons[f"{name}_nav"] = nav
            if tile:
                self.icons[f"{name}_tile"] = tile
        logo = load_icon(ASSETS_DIR / "logo.png", 40)
        if logo:
            self.icons["logo"] = logo

    def build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.sidebar = ttk.Frame(self.root, style="Nav.TFrame")
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.build_sidebar(self.sidebar)

        self.views = ttk.Frame(self.root, style="App.TFrame")
        self.views.grid(row=0, column=1, sticky="nsew")
        self.views.columnconfigure(0, weight=1)
        self.views.rowconfigure(0, weight=1)

        self.ml_view = ttk.Frame(self.views, style="App.TFrame")
        self.ml_view.grid(row=0, column=0, sticky="nsew")
        self.build_ml_view(self.ml_view)

        from dashboard import DashboardView

        self.dashboard_view = DashboardView(self.views, self.icons, go_to_pipeline=lambda: self.select_phase("todo"))
        self.dashboard_view.grid(row=0, column=0, sticky="nsew")
        self.dashboard_view.grid_remove()

    def build_sidebar(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        brand = ttk.Frame(parent, style="Nav.TFrame", padding=(16, 20, 16, 18))
        brand.grid(row=0, column=0, sticky="ew")
        if "logo" in self.icons:
            ttk.Label(brand, image=self.icons["logo"], background=PALETTE["ink"]).grid(row=0, column=0, rowspan=2, sticky="w", padx=(2, 10))
        self.brand_title = ttk.Label(brand, text="FactuRisk", style="Brand.TLabel")
        self.brand_title.grid(row=0, column=1, sticky="sw")
        self.brand_sub = ttk.Label(brand, text="SUNAT · Riesgo de comprobantes", style="BrandSub.TLabel")
        self.brand_sub.grid(row=1, column=1, sticky="nw")

        self.nav_sections: list[ttk.Label] = []
        row = 1
        section = ttk.Label(parent, text="MACHINE LEARNING", style="NavSection.TLabel", padding=(20, 6, 0, 4))
        section.grid(row=row, column=0, sticky="ew")
        self.nav_sections.append(section)
        row += 1
        labels = {"todo": "Proceso completo"}
        labels.update({key: f"{index:02d} · {PHASES[key].title}" for index, key in enumerate(PHASE_ORDER, start=1)})
        for key in ["todo", *PHASE_ORDER]:
            item = NavItem(parent, labels[key], self.icons.get(f"{key}_nav"), command=lambda value=key: self.select_phase(value))
            item.grid(row=row, column=0, sticky="ew")
            self.nav_items[key] = item
            row += 1

        section = ttk.Label(parent, text="ANÁLISIS", style="NavSection.TLabel", padding=(20, 16, 0, 4))
        section.grid(row=row, column=0, sticky="ew")
        self.nav_sections.append(section)
        row += 1
        item = NavItem(parent, "Dashboard de riesgo", self.icons.get("dashboard_nav"), command=self.show_dashboard)
        item.grid(row=row, column=0, sticky="ew")
        self.nav_items["dashboard"] = item
        row += 1

        parent.rowconfigure(row, weight=1)
        self.nav_footer = ttk.Label(
            parent,
            text="Fuente externa: SUNAT\nNoSQL: MongoDB\nValidación temporal",
            style="BrandSub.TLabel",
            padding=(20, 0, 16, 18),
            justify="left",
        )
        self.nav_footer.grid(row=row + 1, column=0, sticky="sew")

    def build_ml_view(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        self.header = ttk.Frame(parent, style="App.TFrame", padding=(28, 22, 28, 12))
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.columnconfigure(0, weight=1)
        titles = ttk.Frame(self.header, style="App.TFrame")
        titles.grid(row=0, column=0, sticky="ew")
        ttk.Label(titles, text="Machine Learning", style="H1.TLabel").pack(anchor="w")
        self.header_subtitle = ttk.Label(
            titles,
            text="Inspección, SUNAT, MongoDB, MapReduce, entrenamiento y predicción de comprobantes pendientes.",
            style="Sub.TLabel",
        )
        self.header_subtitle.pack(anchor="w", pady=(2, 0))
        titles.bind("<Configure>", lambda event: self.header_subtitle.configure(wraplength=max(event.width, 200)))

        self.actions = ttk.Frame(self.header, style="App.TFrame")
        ttk.Checkbutton(self.actions, text="Forzar descarga SUNAT", variable=self.force_scraping).pack(side="left", padx=(0, 12))
        self.todo_button = ttk.Button(self.actions, text="Ejecutar todo", style="Primary.TButton", command=lambda: self.run_phase_group("todo"))
        self.todo_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(self.actions, text="Detener", style="Danger.TButton", command=self.stop_process, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 8))
        self.clean_button = ttk.Button(self.actions, text="Limpiar resultados", style="Secondary.TButton", command=self.clean_generated_data)
        self.clean_button.pack(side="left")
        self.actions.grid(row=0, column=1, sticky="e")

        scroll = ScrollableFrame(parent)
        scroll.grid(row=1, column=0, sticky="nsew")
        body = ttk.Frame(scroll.body, style="App.TFrame", padding=(28, 0, 28, 24))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)

        self.build_metric_strip(body)
        self.work_area = ttk.Frame(body, style="App.TFrame")
        self.work_area.grid(row=1, column=0, sticky="ew")
        self.build_phase_cards(self.work_area)
        self.build_detail_panel(self.work_area)
        self.layout_work_area(True)
        self.build_console(body)

    def build_metric_strip(self, parent: ttk.Frame) -> None:
        grid = ResponsiveGrid(parent, min_item_width=190, max_columns=5)
        grid.grid(row=0, column=0, sticky="ew", pady=(4, 6))
        metrics = [
            ("Modelo", "modelo"),
            ("Recall incidencias", "recall"),
            ("F1 incidencias", "f1"),
            ("Pendientes evaluados", "pendientes"),
            ("Riesgo alto", "riesgo_alto"),
        ]
        for title, key in metrics:
            self.metric_vars[key] = StringVar(value="—")
            box = card(grid, padding=(14, 12))
            inner = box.inner  # type: ignore[attr-defined]
            if f"{key}_tile" in self.icons:
                ttk.Label(inner, image=self.icons[f"{key}_tile"], style="CardBody.TLabel").pack(side="left", padx=(0, 12))
            text = ttk.Frame(inner, style="Card.TFrame")
            text.pack(side="left", fill="x", expand=True)
            ttk.Label(text, text=title, style="CardCaption.TLabel").pack(anchor="w")
            ttk.Label(text, textvariable=self.metric_vars[key], style="CardValue.TLabel").pack(anchor="w")
            grid.add(box)

    def build_phase_cards(self, parent: ttk.Frame) -> None:
        self.phase_grid = ResponsiveGrid(parent, min_item_width=250, max_columns=2)
        self.phase_cards: dict[str, tk.Frame] = {}
        for index, key in enumerate(PHASE_ORDER, start=1):
            info = PHASES[key]
            box = card(self.phase_grid, padding=(14, 12))
            inner = box.inner  # type: ignore[attr-defined]
            inner.columnconfigure(1, weight=1)
            if f"{key}_tile" in self.icons:
                ttk.Label(inner, image=self.icons[f"{key}_tile"], style="CardBody.TLabel").grid(row=0, column=0, rowspan=4, sticky="nw", padx=(0, 12))
            ttk.Label(inner, text=f"Paso {index:02d}", style="CardCaption.TLabel").grid(row=0, column=1, sticky="w")
            ttk.Label(inner, text=info.title, style="CardTitle.TLabel").grid(row=1, column=1, sticky="w")
            subtitle = ttk.Label(inner, text=info.subtitle, style="CardBody.TLabel", justify="left")
            subtitle.grid(row=2, column=1, sticky="w", pady=(2, 6))
            status = ttk.Label(inner, textvariable=self.phase_status[key], style="Pending.Status.TLabel")
            status.grid(row=3, column=1, sticky="w")
            self.status_labels[key].append(status)
            inner.bind("<Configure>", lambda event, label=subtitle: label.configure(wraplength=max(event.width - 96, 120)), add="+")
            for widget in (box, inner, *inner.winfo_children()):
                widget.bind("<Button-1>", lambda _event, phase=key: self.select_phase(phase), add="+")
                widget.configure(cursor="hand2")
            self.phase_cards[key] = box
            self.phase_grid.add(box)

    def build_detail_panel(self, parent: ttk.Frame) -> None:
        self.detail = card(parent, padding=(18, 16))
        inner = self.detail.inner  # type: ignore[attr-defined]
        inner.columnconfigure(0, weight=1)
        self.detail_caption = ttk.Label(inner, text="", style="CardCaption.TLabel")
        self.detail_caption.grid(row=0, column=0, sticky="w")
        self.detail_title = ttk.Label(inner, text="", style="CardTitle.TLabel")
        self.detail_title.grid(row=1, column=0, sticky="w")
        self.detail_text = ttk.Label(inner, text="", style="CardBody.TLabel", justify="left")
        self.detail_text.grid(row=2, column=0, sticky="ew", pady=(6, 12))

        status_row = ttk.Frame(inner, style="Card.TFrame")
        status_row.grid(row=3, column=0, sticky="ew")
        status_row.columnconfigure(0, weight=1)
        ttk.Label(status_row, text="Estado", style="CardCaption.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(status_row, text="Tiempo", style="CardCaption.TLabel").grid(row=0, column=1, sticky="e")
        self.status_label = ttk.Label(status_row, textvariable=self.status_text, style="Done.Status.TLabel")
        self.status_label.grid(row=1, column=0, sticky="w")
        ttk.Label(status_row, textvariable=self.elapsed_text, style="CardTitle.TLabel").grid(row=1, column=1, sticky="e")
        self.progress = ttk.Progressbar(inner, mode="determinate", maximum=100, style="Brand.Horizontal.TProgressbar")
        self.progress.grid(row=4, column=0, sticky="ew", pady=(10, 14))

        ttk.Label(inner, text="Resultados", style="CardCaption.TLabel").grid(row=5, column=0, sticky="w")
        self.result_text = ttk.Label(inner, text="", style="CardBody.TLabel", justify="left")
        self.result_text.grid(row=6, column=0, sticky="ew", pady=(2, 14))

        self.run_button = ttk.Button(inner, text="Ejecutar paso", style="Primary.TButton", command=self.run_selected)
        self.run_button.grid(row=7, column=0, sticky="ew", pady=(0, 6))
        self.open_step_button = ttk.Button(inner, text="Abrir resultado del paso", style="Secondary.TButton", command=self.open_step_result)
        self.open_step_button.grid(row=8, column=0, sticky="ew", pady=(0, 6))
        self.dashboard_button = ttk.Button(inner, text="Ver dashboard de riesgo", style="Secondary.TButton", command=self.show_dashboard)
        self.dashboard_button.grid(row=9, column=0, sticky="ew")
        inner.bind("<Configure>", self._wrap_detail)

    def _wrap_detail(self, event: tk.Event) -> None:
        width = max(event.width - 44, 160)
        self.detail_text.configure(wraplength=width)
        self.result_text.configure(wraplength=width)

    def build_console(self, parent: ttk.Frame) -> None:
        box = card(parent, padding=(14, 10))
        box.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        inner = box.inner  # type: ignore[attr-defined]
        inner.columnconfigure(0, weight=1)
        ttk.Label(inner, text="Consola en tiempo real", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(inner, text="Limpiar consola", style="Ghost.TButton", command=self.clear_console).grid(row=0, column=1, sticky="e")
        self.console = ScrolledText(
            inner,
            height=12,
            background=PALETTE["console_bg"],
            foreground=PALETTE["console_text"],
            insertbackground="#FFFFFF",
            relief="flat",
            font=(FONT_MONO, 9),
            padx=12,
            pady=10,
            wrap="word",
        )
        self.console.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.console.tag_configure("ok", foreground="#7FD6A8")
        self.console.tag_configure("warn", foreground="#F6C177")
        self.console.tag_configure("err", foreground="#F28B82")
        self.console.tag_configure("phase", foreground="#8CC8E8")
        self.log("Sistema listo. Ejecute el Paso 01 o el proceso completo.", "ok")

    # -------------------------------------------------------------- responsive

    def on_resize(self, event: tk.Event) -> None:
        if event.widget is not self.root:
            return
        compact = event.width < COMPACT_WIDTH
        if compact != self.compact:
            self.compact = compact
            for item in self.nav_items.values():
                item.set_collapsed(compact)
            for widget in (self.brand_title, self.brand_sub, self.nav_footer, *self.nav_sections):
                if compact:
                    widget.grid_remove()
                else:
                    widget.grid()
            if compact:
                self.actions.grid(row=1, column=0, columnspan=2, sticky="w", pady=(12, 0))
            else:
                self.actions.grid(row=0, column=1, columnspan=1, sticky="e", pady=0)
        sidebar_width = 58 if compact else 260
        self.layout_work_area(event.width - sidebar_width - 70 >= WIDE_DETAIL_WIDTH)

    def layout_work_area(self, side_by_side: bool) -> None:
        if side_by_side == self.detail_side:
            return
        self.detail_side = side_by_side
        if side_by_side:
            self.work_area.columnconfigure(0, weight=3, uniform="work")
            self.work_area.columnconfigure(1, weight=2, uniform="work")
            self.phase_grid.grid(row=0, column=0, sticky="new", padx=(0, 12))
            self.detail.grid(row=0, column=1, sticky="new", pady=(0, 12))
        else:
            self.work_area.columnconfigure(0, weight=1, uniform="")
            self.work_area.columnconfigure(1, weight=0, uniform="")
            self.detail.grid(row=0, column=0, sticky="ew", pady=(0, 12))
            self.phase_grid.grid(row=1, column=0, sticky="ew", padx=0)

    # ------------------------------------------------------------- navigation

    def select_phase(self, phase: str) -> None:
        self.selected_phase.set(phase)
        self.dashboard_view.grid_remove()
        self.ml_view.grid()
        for key, item in self.nav_items.items():
            item.set_selected(key == phase)
        for key, box in self.phase_cards.items():
            box.configure(background=PALETTE["primary"] if key == phase else PALETTE["line"])

        if phase == "todo":
            self.detail_caption.configure(text="PROCESO COMPLETO")
            self.detail_title.configure(text="Ejecución guiada de las 8 fases")
            self.detail_text.configure(
                text="Inspección, procesamiento distribuido, scraping SUNAT, MongoDB, preparación, entrenamiento, predicción y reportes, en orden."
            )
            self.run_button.configure(text="Ejecutar proceso completo")
        else:
            index = PHASE_ORDER.index(phase) + 1
            info = PHASES[phase]
            self.detail_caption.configure(text=f"PASO {index:02d}")
            self.detail_title.configure(text=info.title)
            self.detail_text.configure(text=f"{info.detail}\n\nObjetivo: {info.outcome}")
            self.run_button.configure(text="Ejecutar paso")
        self.update_step_results_panel(phase)

    def show_dashboard(self) -> None:
        for key, item in self.nav_items.items():
            item.set_selected(key == "dashboard")
        self.ml_view.grid_remove()
        self.dashboard_view.grid()
        self.dashboard_view.refresh()

    def update_step_results_panel(self, phase: str) -> None:
        if phase == "todo":
            done = sum(1 for key in PHASE_ORDER if key in self.completed_phases)
            self.result_text.configure(
                text=f"{done} de {len(PHASE_ORDER)} fases completadas con resultados guardados."
                if done
                else "Los indicadores superiores aparecerán al completar entrenamiento y predicción."
            )
            self.open_step_button.configure(text="Abrir carpeta de resultados")
        else:
            result_paths = STEP_RESULTS.get(phase, [])
            existing = [path for path in result_paths if path.exists()]
            if existing:
                self.result_text.configure(text="Disponibles:\n" + "\n".join(f"• {path.name}" for path in existing))
                self.open_step_button.configure(text="Abrir resultado del paso")
            else:
                expected = "\n".join(f"• {path.name}" for path in result_paths) or "Este paso muestra su avance en la consola."
                self.result_text.configure(text=f"Aún no hay resultados. Se generará:\n{expected}")
                self.open_step_button.configure(text="Abrir carpeta de resultados")
        self.dashboard_button.configure(state="normal" if (PROCESSED_DIR / "predicciones_pendientes.csv").exists() else "disabled")
        allowed, _ = self.can_run_phase(phase)
        self.run_button.configure(state="normal" if allowed and not self.running else "disabled")

    # -------------------------------------------------------------- execution

    def restore_completed_phases(self) -> None:
        """Mark consecutive phases whose artifacts already exist so the user can resume."""

        for key in PHASE_ORDER:
            evidence = PHASE_EVIDENCE[key]
            if evidence is not None and not evidence.exists():
                break
            if evidence is None:
                # Phases without their own artifact count as done once a later one is.
                later = [PHASE_EVIDENCE[k] for k in PHASE_ORDER[PHASE_ORDER.index(key) + 1 :] if PHASE_EVIDENCE[k] is not None]
                if not later or not later[0].exists():  # type: ignore[union-attr]
                    break
            self.completed_phases.add(key)
            self.set_phase_status(key, "Completada")

    def set_phase_status(self, phase: str, status: str) -> None:
        self.phase_status[phase].set(status)
        for label in self.status_labels[phase]:
            label.configure(style=STATUS_STYLES.get(status, "Pending.Status.TLabel"))

    def can_run_phase(self, phase: str) -> tuple[bool, str]:
        if phase == "todo":
            return True, ""
        index = PHASE_ORDER.index(phase)
        missing = [PHASES[key].title for key in PHASE_ORDER[:index] if key not in self.completed_phases]
        if missing:
            return False, f"Antes debe completar: {', '.join(missing)}."
        return True, ""

    def build_command(self, phase: str) -> list[str]:
        if getattr(sys, "frozen", False):
            command = [sys.executable, "--worker-phase", phase]
            if phase == "scraping" and self.force_scraping.get():
                command.append("--force-scraping")
            return command
        command = [sys.executable, str(PHASE_SCRIPTS[phase])]
        if phase == "scraping" and self.force_scraping.get():
            command.append("--force-download")
        return command

    def run_selected(self) -> None:
        self.run_phase_group(self.selected_phase.get())

    def run_phase_group(self, selected: str) -> None:
        if self.running:
            messagebox.showinfo("Proceso en ejecución", "Ya hay una fase en ejecución.")
            return
        allowed, reason = self.can_run_phase(selected)
        if not allowed:
            messagebox.showwarning("Paso bloqueado", reason)
            self.log(f"Paso bloqueado: {reason}", "warn")
            return
        phases = PHASE_ORDER if selected == "todo" else [selected]
        self.running = True
        self.start_time = time.perf_counter()
        self.progress.configure(value=0)
        self.set_buttons_state(False)
        if selected == "todo":
            self.completed_phases.clear()
            for key in PHASE_ORDER:
                self.set_phase_status(key, "Pendiente")
            self.reset_metric_cards()
        self.worker_thread = threading.Thread(target=self.worker, args=(phases,), daemon=True)
        self.worker_thread.start()

    def worker(self, phases: list[str]) -> None:
        total = len(phases)
        for index, phase in enumerate(phases, start=1):
            if not self.running:
                break
            self.queue.put(("phase_start", phase))
            command = self.build_command(phase)
            self.queue.put(("log", (f"INICIO DE FASE: {phase}", "phase")))
            try:
                self.current_process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
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

    @staticmethod
    def tag_for_line(line: str) -> str:
        lower = line.lower()
        if "fallida" in lower or "error" in lower or "traceback" in lower:
            return "err"
        if "warning" in lower or "advertencia" in lower:
            return "warn"
        if "completada" in lower or "correctamente" in lower:
            return "ok"
        if "fase" in lower or "inicio" in lower:
            return "phase"
        return ""

    def process_queue(self) -> None:
        try:
            while True:
                event, payload = self.queue.get_nowait()
                if event == "log":
                    message, tag = payload
                    self.log(message, tag)
                elif event == "phase_start":
                    self.set_phase_status(payload, "En ejecución")
                    self.status_text.set(f"Ejecutando: {PHASES[payload].title}")
                    self.status_label.configure(style="Running.Status.TLabel")
                elif event == "phase_done":
                    phase, index, total = payload
                    self.completed_phases.add(phase)
                    self.set_phase_status(phase, "Completada")
                    self.progress.configure(value=index / total * 100)
                    self.log(f"FASE COMPLETADA: {phase}", "ok")
                    if phase in {"entrenar", "predecir"}:
                        self.refresh_metrics()
                elif event == "phase_failed":
                    self.set_phase_status(payload, "Fallida")
                    self.status_text.set(f"Fase fallida: {PHASES[payload].title}")
                    self.status_label.configure(style="Failed.Status.TLabel")
                elif event == "done":
                    self.running = False
                    self.set_buttons_state(True)
                    if payload:
                        self.progress.configure(value=100)
                        self.refresh_metrics()
                        self.status_text.set("Proceso completado correctamente")
                        self.status_label.configure(style="Done.Status.TLabel")
                        self.log("PROCESO FINALIZADO CORRECTAMENTE", "ok")
                    else:
                        self.status_text.set("Proceso detenido por error")
                        self.log("PROCESO DETENIDO. Revise la consola.", "err")
                    self.update_step_results_panel(self.selected_phase.get())
        except queue.Empty:
            pass
        self.root.after(120, self.process_queue)

    def tick(self) -> None:
        if self.running and self.start_time is not None:
            minutes, seconds = divmod(int(time.perf_counter() - self.start_time), 60)
            self.elapsed_text.set(f"{minutes:02d}:{seconds:02d}")
        self.root.after(1000, self.tick)

    def log(self, message: str, tag: str = "") -> None:
        self.console.configure(state="normal")
        self.console.insert("end", f"[{datetime.now():%H:%M:%S}] {message}\n", tag)
        self.console.see("end")
        self.console.configure(state="disabled")

    def clear_console(self) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def set_buttons_state(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.todo_button.configure(state=state)
        self.clean_button.configure(state=state)
        self.stop_button.configure(state="disabled" if enabled else "normal")
        allowed, _ = self.can_run_phase(self.selected_phase.get())
        self.run_button.configure(state="normal" if enabled and allowed else "disabled")

    def stop_process(self) -> None:
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

    # ----------------------------------------------------------------- results

    def reset_metric_cards(self) -> None:
        for var in self.metric_vars.values():
            var.set("—")

    def refresh_metrics(self) -> None:
        metrics_path = OUTPUTS_DIR / "metricas_modelo.json"
        predictions_path = PROCESSED_DIR / "predicciones_pendientes.csv"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                test = metrics.get("metricas_prueba", {})
                self.metric_vars["modelo"].set(model_display_name(metrics.get("mejor_modelo", "N/D")))
                self.metric_vars["recall"].set(self.percent(test.get("recall_clase_1")))
                self.metric_vars["f1"].set(self.percent(test.get("f1_clase_1")))
            except (OSError, json.JSONDecodeError) as exc:
                self.log(f"No se pudieron leer métricas: {exc}", "warn")
        if predictions_path.exists():
            try:
                pred = pd.read_csv(predictions_path, encoding="utf-8-sig", usecols=["Nivel_Riesgo"])
                self.metric_vars["pendientes"].set(f"{len(pred):,}")
                self.metric_vars["riesgo_alto"].set(f"{int((pred['Nivel_Riesgo'] == 'Alto').sum()):,}")
            except (OSError, ValueError) as exc:
                self.log(f"No se pudieron leer predicciones: {exc}", "warn")

    @staticmethod
    def percent(value: Any) -> str:
        try:
            return f"{float(value) * 100:.2f}%"
        except (TypeError, ValueError):
            return "N/D"

    def clean_generated_data(self) -> None:
        """Delete generated artifacts while preserving the input dataset and credentials."""

        if self.running:
            messagebox.showwarning("Proceso en ejecución", "Detenga el proceso antes de limpiar resultados.")
            return
        if not messagebox.askyesno(
            "Limpiar resultados",
            "Se eliminarán resultados, modelos, reportes, descargas SUNAT y datos procesados. "
            "El dataset de entrada y la configuración se conservarán. ¿Desea continuar?",
        ):
            return
        try:
            skipped = self.safe_clean_generated_artifacts()
        except (OSError, RuntimeError) as exc:
            messagebox.showerror("Error al limpiar", str(exc))
            self.log(f"Error al limpiar resultados: {exc}", "err")
            return
        self.completed_phases.clear()
        for key in PHASE_ORDER:
            self.set_phase_status(key, "Pendiente")
        self.reset_metric_cards()
        self.progress.configure(value=0)
        self.elapsed_text.set("00:00")
        self.status_text.set("Resultados limpiados. Inicie desde el Paso 01.")
        self.clear_console()
        self.log("Limpieza completada. Se conservaron el dataset de entrada y la configuración.", "ok")
        if skipped:
            self.log("Archivos en uso omitidos: " + ", ".join(path.name for path in skipped), "warn")
        self.update_step_results_panel(self.selected_phase.get())

    def safe_clean_generated_artifacts(self) -> list[Path]:
        """Remove generated contents inside known project folders only."""

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
            if root not in resolved.parents:
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

    @staticmethod
    def open_path(path: Path) -> None:
        if sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])

    def open_outputs(self) -> None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.open_path(OUTPUTS_DIR)

    def open_step_result(self) -> None:
        phase = self.selected_phase.get()
        existing = [path for path in STEP_RESULTS.get(phase, []) if path.exists()]
        if existing:
            self.open_path(existing[0])
        else:
            self.open_outputs()

    def close(self) -> None:
        if self.running:
            if not messagebox.askyesno("Proceso en ejecución", "Hay un proceso en ejecución. ¿Desea detenerlo y cerrar?"):
                return
            self.stop_process()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def lanzar_gui() -> None:
    """Launch the graphical interface."""

    FactuRiskApp().run()


if __name__ == "__main__":
    if "--worker-phase" in sys.argv:
        from main import run_worker_phase

        phase_index = sys.argv.index("--worker-phase") + 1
        if phase_index >= len(sys.argv):
            raise SystemExit("Falta la fase para --worker-phase")
        raise SystemExit(run_worker_phase(sys.argv[phase_index], force_scraping="--force-scraping" in sys.argv))
    lanzar_gui()
