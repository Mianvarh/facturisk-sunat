"""Risk dashboard: explores the pending-invoice predictions after the ML phase."""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
from PIL import Image, ImageTk

import dashboard_data as dd
from paths import get_application_root
from theme import PALETTE, RISK_COLORS, apply_chart_style, model_display_name
from ui_widgets import ResponsiveGrid, ScrollableFrame, card

OUTPUTS_DIR = get_application_root() / "outputs"

MODEL_CHARTS = [
    ("matriz_correlacion_pearson.png", "Correlación de Pearson entre variables candidatas e incidencia."),
    ("09_matriz_confusion_optimizada.png", "Matriz de confusión con el umbral optimizado: aciertos, falsas alertas e incidencias no detectadas."),
    ("06_precision_recall_curve.png", "Curva Precision-Recall: equilibrio entre detectar incidencias y generar alertas."),
    ("07_roc_curve.png", "Curva ROC: capacidad del modelo para separar aceptadas e incidencias."),
    ("04_comparacion_modelos_pr_auc.png", "Comparación de modelos por PR-AUC, con y sin variables SUNAT."),
    ("13_importancia_variables.png", "Importancia de variables por permutación."),
    ("15_aporte_sunat.png", "Aporte de las variables SUNAT frente al modelo sin ellas."),
    ("16_distribucion_riesgo_pendientes.png", "Distribución de riesgo en los comprobantes pendientes."),
]

CHART_HEIGHT_PX = 290


def _fmt_int(value: float) -> str:
    return f"{value:,.0f}"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/D"


class DashboardView(ttk.Frame):
    """KPIs, charts, top-risk table and the model chart gallery."""

    def __init__(self, parent: tk.Misc, icons: dict[str, ImageTk.PhotoImage], go_to_pipeline: Callable[[], None]) -> None:
        super().__init__(parent, style="App.TFrame")
        apply_chart_style()
        self.icons = icons
        self.go_to_pipeline = go_to_pipeline
        self.canvases: list[FigureCanvasTkAgg] = []
        self.gallery_index = 0
        self.gallery_image: ImageTk.PhotoImage | None = None
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, style="App.TFrame", padding=(28, 22, 28, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Dashboard de riesgo", style="H1.TLabel").grid(row=0, column=0, sticky="w")
        self.subtitle = ttk.Label(header, text="", style="Sub.TLabel")
        self.subtitle.grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Button(header, text="Actualizar", style="Secondary.TButton", command=self.refresh).grid(row=0, column=1, rowspan=2, sticky="e")

        self.scroll = ScrollableFrame(self)
        self.scroll.grid(row=1, column=0, sticky="nsew")
        self.content = ttk.Frame(self.scroll.body, style="App.TFrame", padding=(28, 0, 28, 24))
        self.content.pack(fill="both", expand=True)
        self.content.columnconfigure(0, weight=1)

    # ------------------------------------------------------------------ refresh

    def refresh(self) -> None:
        for canvas in self.canvases:
            canvas.get_tk_widget().destroy()
        self.canvases.clear()
        for child in self.content.winfo_children():
            child.destroy()

        data = dd.load_dashboard_data()
        if data is None:
            self.subtitle.configure(text="Aún no hay predicciones para analizar.")
            self._empty_state()
            return

        self.subtitle.configure(
            text=f"{len(data.predictions):,} comprobantes pendientes evaluados · actualizado {datetime.now():%d/%m/%Y %H:%M}"
        )
        self._kpi_section(data)
        self._chart_section(data)
        self._table_section(data)
        self._gallery_section()

    def _empty_state(self) -> None:
        box = card(self.content, padding=(28, 28))
        box.grid(row=0, column=0, sticky="ew", pady=(8, 0))
        inner = box.inner  # type: ignore[attr-defined]
        if "dashboard_tile" in self.icons:
            ttk.Label(inner, image=self.icons["dashboard_tile"], style="CardBody.TLabel").pack(anchor="w")
        ttk.Label(inner, text="Ejecuta el pipeline para ver el dashboard", style="CardTitle.TLabel").pack(anchor="w", pady=(12, 4))
        ttk.Label(
            inner,
            text="El dashboard se construye con las predicciones de los comprobantes pendientes (Paso 07) y las métricas del modelo (Paso 06).",
            style="CardBody.TLabel",
            wraplength=560,
            justify="left",
        ).pack(anchor="w")
        ttk.Button(inner, text="Ir a Machine Learning", style="Primary.TButton", command=self.go_to_pipeline).pack(anchor="w", pady=(16, 0))

    # --------------------------------------------------------------------- KPIs

    def _kpi_section(self, data: dd.DashboardData) -> None:
        values = dd.kpis(data)
        grid = ResponsiveGrid(self.content, min_item_width=210, max_columns=4)
        grid.grid(row=0, column=0, sticky="ew", pady=(4, 4))
        threshold = values["umbral"]
        cards = [
            ("pendientes_tile", "Pendientes evaluados", _fmt_int(values["pendientes"]), f"Probabilidad media {values['probabilidad_media'] * 100:.1f}%"),
            ("riesgo_alto_tile", "Riesgo alto", _fmt_int(values["riesgo_alto"]), f"{values['riesgo_alto_pct']:.1f}% de los pendientes"),
            (
                "recall_tile",
                "Marcados para revisión",
                _fmt_int(values["marcados_revision"]),
                f"{values['marcados_revision_pct']:.1f}% superan el umbral {threshold:.2f}" if threshold is not None else "Según el umbral del modelo",
            ),
            ("modelo_tile", "Modelo en uso", model_display_name(values["modelo"]), f"Recall {_fmt_pct(values['recall'])} · PR-AUC {values['pr_auc']:.3f}" if values["pr_auc"] is not None else ""),
        ]
        for icon_key, title, value, caption in cards:
            box = card(grid)
            inner = box.inner  # type: ignore[attr-defined]
            row = ttk.Frame(inner, style="Card.TFrame")
            row.pack(fill="x")
            if icon_key in self.icons:
                ttk.Label(row, image=self.icons[icon_key], style="CardBody.TLabel").pack(side="left", padx=(0, 12))
            text = ttk.Frame(row, style="Card.TFrame")
            text.pack(side="left", fill="x", expand=True)
            ttk.Label(text, text=title, style="CardCaption.TLabel").pack(anchor="w")
            ttk.Label(text, text=value, style="CardValue.TLabel").pack(anchor="w")
            ttk.Label(inner, text=caption, style="CardBody.TLabel").pack(anchor="w", pady=(6, 0))
            grid.add(box)

    # ------------------------------------------------------------------- charts

    def _chart_card(self, parent: tk.Misc, title: str, caption: str) -> tuple[tk.Frame, Figure]:
        box = card(parent, padding=(14, 12))
        inner = box.inner  # type: ignore[attr-defined]
        ttk.Label(inner, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(inner, text=caption, style="CardBody.TLabel").pack(anchor="w", pady=(2, 6))
        figure = Figure(figsize=(5, 2.9), dpi=96, layout="constrained")
        canvas = FigureCanvasTkAgg(figure, master=inner)
        widget = canvas.get_tk_widget()
        widget.configure(height=CHART_HEIGHT_PX, highlightthickness=0, background=PALETTE["surface"])
        widget.pack(fill="x", expand=False)
        self.canvases.append(canvas)
        return box, figure

    def _chart_section(self, data: dd.DashboardData) -> None:
        pred = data.predictions
        ttk.Label(self.content, text="ANÁLISIS DE PENDIENTES", style="Section.TLabel").grid(row=1, column=0, sticky="w", pady=(14, 8))
        grid = ResponsiveGrid(self.content, min_item_width=430, max_columns=2)
        grid.grid(row=2, column=0, sticky="ew")

        counts = dd.risk_level_counts(pred)
        box, fig = self._chart_card(grid, "Comprobantes por nivel de riesgo", "Cortes calculados sobre la distribución de pendientes.")
        ax = fig.add_subplot()
        bars = ax.bar(counts.index, counts.values, color=[RISK_COLORS[level] for level in counts.index], width=0.6)
        ax.bar_label(bars, labels=[f"{value:,}" for value in counts.values], fontsize=8, color=PALETTE["muted"])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
        grid.add(box)

        box, fig = self._chart_card(grid, "Distribución de probabilidades", "Probabilidad de incidencia estimada; la línea marca el umbral operativo.")
        ax = fig.add_subplot()
        probabilities = pred["Probabilidad_Incidencia"]
        threshold = data.metrics.get("umbral")
        upper = min(1.0, max(float(probabilities.quantile(0.999)) * 1.15, (threshold or 0) * 1.5, 0.05))
        ax.hist(probabilities, bins=40, range=(0, upper), color=PALETTE["primary"], alpha=0.85)
        ax.set_xlim(0, upper)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
        if threshold is not None:
            ax.axvline(threshold, color=PALETTE["accent_dark"], linewidth=1.6, linestyle="--")
            ax.annotate(
                f"umbral {threshold:.0%}",
                xy=(threshold, 1),
                xycoords=("data", "axes fraction"),
                xytext=(6, -12),
                textcoords="offset points",
                color=PALETTE["accent_dark"],
                fontsize=8,
            )
        ax.set_xlabel("Probabilidad de incidencia")
        grid.add(box)

        by_type = dd.high_risk_share_by_type(pred)
        box, fig = self._chart_card(grid, "Riesgo alto por tipo de comprobante", "Porcentaje de comprobantes de cada tipo clasificados en riesgo alto.")
        ax = fig.add_subplot()
        ordered = by_type.sort_values("porcentaje_alto")
        ax.barh(ordered.index.astype(str), ordered["porcentaje_alto"], color=PALETTE["accent"])
        ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        grid.add(box)

        monthly = dd.risk_by_month(pred)
        if monthly is not None and len(monthly) > 1:
            box, fig = self._chart_card(grid, "Riesgo por mes de emisión", "Probabilidad media de incidencia de los pendientes emitidos cada mes.")
            ax = fig.add_subplot()
            ax.plot(monthly.index, monthly["probabilidad_media"], color=PALETTE["primary"], linewidth=2)
            ax.fill_between(monthly.index, monthly["probabilidad_media"], color=PALETTE["primary"], alpha=0.12)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            fig.autofmt_xdate()
            grid.add(box)

        suppliers = dd.top_suppliers(pred)
        if not suppliers.empty:
            box, fig = self._chart_card(grid, "Proveedores con más riesgo alto", "RUC con mayor número de comprobantes pendientes en riesgo alto.")
            ax = fig.add_subplot()
            ordered = suppliers.iloc[::-1]
            ax.barh(ordered.index.astype(str), ordered["riesgo_alto"], color=PALETTE["danger"], alpha=0.85)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="x")
            ax.grid(axis="y", visible=False)
            grid.add(box)

        factors = dd.top_risk_factors(pred)
        if not factors.empty:
            box, fig = self._chart_card(grid, "Factores de riesgo más frecuentes", "Explicaciones asociadas a los comprobantes en riesgo alto.")
            ax = fig.add_subplot()
            ordered = factors.iloc[::-1]
            ax.barh(ordered.index, ordered.values, color=PALETTE["primary"])
            ax.grid(axis="x")
            ax.grid(axis="y", visible=False)
            grid.add(box)

    # -------------------------------------------------------------------- table

    def _table_section(self, data: dd.DashboardData) -> None:
        ttk.Label(self.content, text="COMPROBANTES PRIORITARIOS", style="Section.TLabel").grid(row=3, column=0, sticky="w", pady=(14, 8))
        box = card(self.content, padding=(12, 12))
        box.grid(row=4, column=0, sticky="ew")
        inner = box.inner  # type: ignore[attr-defined]
        inner.columnconfigure(0, weight=1)
        table = dd.top_invoices(data.predictions)
        headings = {
            "ID_Comprobante": ("Comprobante", 120),
            "RUC_Proveedor": ("RUC proveedor", 110),
            "Tipo_Comprobante": ("Tipo", 130),
            "Importe_Total": ("Importe", 90),
            "Probabilidad_Incidencia": ("Probabilidad", 90),
            "Nivel_Riesgo": ("Riesgo", 70),
            "Razones_Principales": ("Factores", 320),
        }
        columns = list(table.columns)
        tree = ttk.Treeview(inner, columns=columns, show="headings", height=min(len(table), 15))
        for column in columns:
            label, width = headings.get(column, (column, 100))
            tree.heading(column, text=label, anchor="w")
            tree.column(column, width=width, minwidth=60, stretch=column == "Razones_Principales", anchor="w")
        for _, row in table.iterrows():
            values = []
            for column in columns:
                value = row[column]
                if column == "Probabilidad_Incidencia":
                    value = f"{value * 100:.1f}%"
                elif column == "Importe_Total":
                    value = f"{value:,.2f}"
                elif column == "Nivel_Riesgo":
                    value = f"● {value}"
                values.append(value)
            tree.insert("", "end", values=values)
        xscroll = ttk.Scrollbar(inner, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="ew")
        xscroll.grid(row=1, column=0, sticky="ew")

    # ------------------------------------------------------------------ gallery

    def _gallery_section(self) -> None:
        self.gallery_items = [(OUTPUTS_DIR / name, text) for name, text in MODEL_CHARTS if (OUTPUTS_DIR / name).exists()]
        if not self.gallery_items:
            return
        ttk.Label(self.content, text="GRÁFICOS DEL MODELO", style="Section.TLabel").grid(row=5, column=0, sticky="w", pady=(18, 8))
        box = card(self.content, padding=(16, 14))
        box.grid(row=6, column=0, sticky="ew")
        inner = box.inner  # type: ignore[attr-defined]
        inner.columnconfigure(1, weight=1)
        ttk.Button(inner, text="‹", width=3, style="Secondary.TButton", command=lambda: self._move_gallery(-1)).grid(row=0, column=0, sticky="w")
        self.gallery_title = ttk.Label(inner, text="", style="CardBody.TLabel", wraplength=600, justify="center", anchor="center")
        self.gallery_title.grid(row=0, column=1, sticky="ew", padx=12)
        ttk.Button(inner, text="›", width=3, style="Secondary.TButton", command=lambda: self._move_gallery(1)).grid(row=0, column=2, sticky="e")
        self.gallery_label = tk.Label(inner, background=PALETTE["surface"], borderwidth=0)
        self.gallery_label.grid(row=1, column=0, columnspan=3, pady=(12, 0))
        self.gallery_counter = ttk.Label(inner, text="", style="CardCaption.TLabel")
        self.gallery_counter.grid(row=2, column=0, columnspan=3, pady=(6, 0))
        inner.bind("<Configure>", lambda event: self._show_gallery(event.width))
        self._show_gallery(inner.winfo_width())

    def _move_gallery(self, step: int) -> None:
        self.gallery_index = (self.gallery_index + step) % len(self.gallery_items)
        self._show_gallery(self.gallery_label.master.winfo_width())

    def _show_gallery(self, width: int) -> None:
        if not self.gallery_items:
            return
        self.gallery_index %= len(self.gallery_items)
        path, text = self.gallery_items[self.gallery_index]
        max_width = max(width - 40, 320)
        image = Image.open(path).convert("RGB")
        image.thumbnail((min(max_width, 900), 460), Image.Resampling.LANCZOS)
        self.gallery_image = ImageTk.PhotoImage(image)
        self.gallery_label.configure(image=self.gallery_image)
        self.gallery_title.configure(text=text, wraplength=max(width - 140, 200))
        self.gallery_counter.configure(text=f"{self.gallery_index + 1} de {len(self.gallery_items)} · {Path(path).name}")

