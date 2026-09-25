"""Risk dashboard: explores the pending-invoice predictions after the ML phase."""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
from PIL import Image, ImageTk

import dashboard_data as dd
from paths import get_application_root
from theme import FONT, PALETTE, RISK_COLORS, apply_chart_style, model_display_name, sequential_cmap
from ui_widgets import DonutRing, PillButton, ResponsiveGrid, RoundedCard, ScrollableFrame, SegmentedControl, SegmentedMeter, StatusPill, card

OUTPUTS_DIR = get_application_root() / "outputs"

MODEL_CHARTS = [
    ("matriz_correlacion_pearson.png", "Correlación de Pearson entre variables candidatas e incidencia."),
    ("09_matriz_confusion_optimizada.png", "Matriz de confusión con el umbral optimizado: aciertos, falsas alertas e incidencias no detectadas."),
    ("06_precision_recall_curve.png", "Curva Precision-Recall: equilibrio entre detectar incidencias y generar alertas."),
    ("07_roc_curve.png", "Curva ROC: capacidad del modelo para separar aceptadas e incidencias."),
    ("04_comparacion_modelos_pr_auc.png", "Comparación de modelos por PR-AUC, con y sin variables SUNAT."),
    ("10_metricas_por_umbral.png", "Precision, recall y F1 según el umbral de decisión."),
    ("13_importancia_variables.png", "Importancia de variables por permutación."),
    ("15_aporte_sunat.png", "Aporte de las variables SUNAT frente al modelo sin ellas."),
]

CHART_HEIGHT_PX = 300


def _fmt_int(value: float) -> str:
    return f"{value:,.0f}"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/D"


def _clean_axes(ax: Any, grid_axis: str = "y") -> None:
    ax.grid(axis="both", visible=False)
    if grid_axis:
        ax.grid(axis=grid_axis, visible=True)
    ax.tick_params(length=0)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["line"])


class DashboardView(ttk.Frame):
    """KPIs, charts, top-risk table and the model chart gallery."""

    def __init__(self, parent: tk.Misc, icons: dict[str, ImageTk.PhotoImage], go_to_pipeline: Callable[[], None]) -> None:
        super().__init__(parent, style="App.TFrame")
        apply_chart_style()
        self.icons = icons
        self.go_to_pipeline = go_to_pipeline
        self.canvases: list[FigureCanvasTkAgg] = []
        self.gallery_index = 0
        self.gallery_items: list[tuple[Path, str]] = []
        self.gallery_image: ImageTk.PhotoImage | None = None
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, style="App.TFrame", padding=(28, 22, 28, 14))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        titles = ttk.Frame(header, style="App.TFrame")
        titles.grid(row=0, column=0, sticky="w")
        title_row = ttk.Frame(titles, style="App.TFrame")
        title_row.pack(anchor="w")
        ttk.Label(title_row, text="Dashboard de riesgo", style="H1.TLabel").pack(side="left")
        self.status = StatusPill(title_row, "Sin datos", PALETTE["subtle"])
        self.status.pack(side="left", padx=(14, 0), pady=(6, 0))
        self.subtitle = ttk.Label(titles, text="", style="Sub.TLabel")
        self.subtitle.pack(anchor="w", pady=(4, 0))
        PillButton(header, "↻  Actualizar", command=self.refresh, variant="secondary").grid(row=0, column=1, sticky="ne")

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
            self.status.set("Sin predicciones", PALETTE["subtle"])
            self._empty_state()
            return

        model = model_display_name(data.metrics.get("mejor_modelo", "modelo"))
        self.status.set(f"{model} · umbral {data.metrics.get('umbral', 0):.0%}", PALETTE["success"])
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
        PillButton(inner, "Ir a Machine Learning", command=self.go_to_pipeline, background=PALETTE["surface"]).pack(anchor="w", pady=(16, 0))

    # --------------------------------------------------------------------- KPIs

    def _kpi_card(self, parent: tk.Misc, icon_key: str, title: str) -> ttk.Frame:
        box = card(parent, padding=(18, 16))
        inner = box.inner  # type: ignore[attr-defined]
        head = ttk.Frame(inner, style="Card.TFrame")
        head.pack(fill="x")
        if icon_key in self.icons:
            ttk.Label(head, image=self.icons[icon_key], style="CardBody.TLabel").pack(side="left", padx=(0, 10))
        ttk.Label(head, text=title, style="CardTitle.TLabel").pack(side="left")
        return box

    def _kpi_section(self, data: dd.DashboardData) -> None:
        values = dd.kpis(data)
        grid = ResponsiveGrid(self.content, min_item_width=250, max_columns=4)
        grid.grid(row=0, column=0, sticky="ew", pady=(4, 4))

        box = self._kpi_card(grid, "pendientes_tile", "Pendientes evaluados")
        inner = box.inner  # type: ignore[attr-defined]
        ttk.Label(inner, text=_fmt_int(values["pendientes"]), style="CardValue.TLabel").pack(anchor="w", pady=(14, 0))
        ttk.Label(inner, text=f"Probabilidad media de incidencia {values['probabilidad_media'] * 100:.1f}%", style="CardBody.TLabel").pack(anchor="w")
        grid.add(box)

        box = self._kpi_card(grid, "riesgo_alto_tile", "Riesgo alto")
        inner = box.inner  # type: ignore[attr-defined]
        row = ttk.Frame(inner, style="Card.TFrame")
        row.pack(fill="x", pady=(8, 0))
        DonutRing(row, values["riesgo_alto_pct"] / 100, "de pendientes", PALETTE["danger"], size=112).pack(side="left")
        side = ttk.Frame(row, style="Card.TFrame")
        side.pack(side="left", padx=(14, 0))
        ttk.Label(side, text=_fmt_int(values["riesgo_alto"]), style="CardValue.TLabel").pack(anchor="w")
        ttk.Label(side, text="a revisar\nprimero", style="CardBody.TLabel", justify="left").pack(anchor="w")
        grid.add(box)

        box = self._kpi_card(grid, "recall_tile", "Para revisión")
        inner = box.inner  # type: ignore[attr-defined]
        ttk.Label(inner, text=f"{values['marcados_revision_pct']:.1f} %", style="CardValue.TLabel").pack(anchor="w", pady=(14, 6))
        SegmentedMeter(inner, values["marcados_revision_pct"] / 100, PALETTE["primary"]).pack(fill="x")
        scale = ttk.Frame(inner, style="Card.TFrame")
        scale.pack(fill="x", pady=(4, 0))
        ttk.Label(scale, text="0%", style="CardCaption.TLabel").pack(side="left")
        ttk.Label(scale, text="100%", style="CardCaption.TLabel").pack(side="right")
        ttk.Label(inner, text=f"{_fmt_int(values['marcados_revision'])} superan el umbral del modelo", style="CardBody.TLabel").pack(anchor="w", pady=(4, 0))
        grid.add(box)

        box = self._kpi_card(grid, "modelo_tile", "Modelo en uso")
        inner = box.inner  # type: ignore[attr-defined]
        ttk.Label(inner, text=model_display_name(values["modelo"]), style="CardValue.TLabel").pack(anchor="w", pady=(14, 6))
        pills = ttk.Frame(inner, style="Card.TFrame")
        pills.pack(anchor="w")
        StatusPill(pills, f"Recall {float(values['recall'] or 0):.0%}", PALETTE["primary"]).pack(side="left", padx=(0, 6))
        if values["pr_auc"] is not None:
            StatusPill(pills, f"PR-AUC {values['pr_auc']:.2f}", PALETTE["accent"]).pack(side="left")
        caption = ttk.Label(inner, text="Validación temporal y probabilidades calibradas", style="CardBody.TLabel", justify="left")
        caption.pack(anchor="w", fill="x", pady=(8, 0))
        inner.bind("<Configure>", lambda event: caption.configure(wraplength=max(event.width - 40, 120)), add="+")
        grid.add(box)

    # ------------------------------------------------------------------- charts

    def _chart_card(self, parent: tk.Misc, title: str, caption: str, height: int = CHART_HEIGHT_PX) -> tuple[tk.Frame, Figure]:
        box = card(parent, padding=(16, 14))
        inner = box.inner  # type: ignore[attr-defined]
        ttk.Label(inner, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(inner, text=caption, style="CardBody.TLabel").pack(anchor="w", pady=(2, 8))
        figure = Figure(figsize=(5, height / 96), dpi=96, layout="constrained")
        canvas = FigureCanvasTkAgg(figure, master=inner)
        widget = canvas.get_tk_widget()
        widget.configure(height=height, highlightthickness=0, background=PALETTE["surface"])
        widget.pack(fill="x", expand=False)
        self.canvases.append(canvas)
        return box, figure

    def _chart_section(self, data: dd.DashboardData) -> None:
        pred = data.predictions
        threshold = data.metrics.get("umbral")
        ttk.Label(self.content, text="ANÁLISIS DE PENDIENTES", style="Section.TLabel").grid(row=1, column=0, sticky="w", pady=(16, 10))
        grid = ResponsiveGrid(self.content, min_item_width=440, max_columns=2)
        grid.grid(row=2, column=0, sticky="ew")

        self._risk_levels_chart(grid, pred)
        self._quadrant_chart(grid, pred, threshold)
        self._heatmap_chart(grid, pred)
        self._monthly_chart(grid, pred)
        self._factors_chart(grid, pred)
        self._suppliers_chart(grid, pred)

    def _risk_levels_chart(self, grid: ResponsiveGrid, pred: Any) -> None:
        counts = dd.risk_level_counts(pred)
        box, fig = self._chart_card(grid, "Comprobantes por nivel de riesgo", "Cortes calculados sobre la distribución de los pendientes.")
        ax = fig.add_subplot()
        bars = ax.bar(counts.index, counts.values, color=[RISK_COLORS[level] for level in counts.index], width=0.55)
        ax.bar_label(bars, labels=[f"{value:,}" for value in counts.values], fontsize=9, color=PALETTE["text"], padding=4)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
        ax.margins(y=0.12)
        _clean_axes(ax)
        grid.add(box)

    def _quadrant_chart(self, grid: ResponsiveGrid, pred: Any, threshold: float | None) -> None:
        points = dd.amount_vs_probability(pred)
        if points.empty:
            return
        box, fig = self._chart_card(
            grid,
            "Importe vs. probabilidad de incidencia",
            "Cada punto es un pendiente (muestra). Arriba a la derecha: alto importe y alto riesgo.",
        )
        ax = fig.add_subplot()
        for level in dd.RISK_LEVELS:
            subset = points[points["Nivel_Riesgo"] == level]
            ax.scatter(subset["Importe_Total"], subset["Probabilidad_Incidencia"], s=9, alpha=0.55, color=RISK_COLORS[level], label=level, linewidths=0)
        ax.set_xscale("log")
        median_amount = float(pred.loc[pred["Importe_Total"] > 0, "Importe_Total"].median())
        ax.axvline(median_amount, color=PALETTE["subtle"], linestyle=(0, (4, 3)), linewidth=1)
        if threshold is not None:
            ax.axhline(threshold, color=PALETTE["subtle"], linestyle=(0, (4, 3)), linewidth=1)
            critical = pred[(pred["Importe_Total"] > median_amount) & (pred["Probabilidad_Incidencia"] >= threshold)]
            ax.text(
                0.98,
                0.96,
                f"{len(critical):,} críticos",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color=PALETTE["danger"],
                bbox={"boxstyle": "round,pad=0.35", "facecolor": PALETTE["danger_soft"], "edgecolor": "none"},
            )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel("Importe (escala logarítmica)")
        ax.legend(
            loc="upper left",
            fontsize=8,
            markerscale=2.5,
            ncols=3,
            scatterpoints=1,
            handletextpad=0.2,
            columnspacing=1.2,
            frameon=True,
            facecolor=PALETTE["surface"],
            edgecolor="none",
            framealpha=1,
        )
        _clean_axes(ax, grid_axis="")
        grid.add(box)

    def _heatmap_chart(self, grid: ResponsiveGrid, pred: Any) -> None:
        table = dd.risk_heatmap(pred)
        if table is None or table.empty:
            return
        box, fig = self._chart_card(grid, "Mapa de calor: tipo × mes de emisión", "Probabilidad media de incidencia en los últimos 12 meses.")
        ax = fig.add_subplot()
        image = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap=sequential_cmap(), interpolation="nearest")
        ax.set_xticks(range(len(table.columns)), labels=table.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(table.index)), labels=[str(label) for label in table.index])
        ax.set_xticks(np.arange(-0.5, len(table.columns)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(table.index)), minor=True)
        ax.grid(which="minor", color=PALETTE["surface"], linewidth=2.5, linestyle="-")
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, format=PercentFormatter(xmax=1, decimals=0))
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(colors=PALETTE["muted"], length=0, labelsize=8)
        grid.add(box)

    def _monthly_chart(self, grid: ResponsiveGrid, pred: Any) -> None:
        monthly = dd.risk_by_month(pred)
        if monthly is None or len(monthly) < 2:
            return
        box, fig = self._chart_card(grid, "Riesgo por mes de emisión", "Probabilidad media de incidencia; se marca el mes más riesgoso.")
        ax = fig.add_subplot()
        series = monthly["probabilidad_media"]
        smooth = series.rolling(3, center=True, min_periods=1).mean()
        ax.plot(series.index, series.values, color=PALETTE["subtle"], linewidth=1, alpha=0.8)
        ax.plot(smooth.index, smooth.values, color=PALETTE["primary"], linewidth=2.4)
        ax.fill_between(smooth.index, smooth.values, series.min() * 0.95, color=PALETTE["primary"], alpha=0.12)
        peak = series.idxmax()
        ax.scatter([peak], [series[peak]], s=60, color=PALETTE["background"], edgecolors=PALETTE["text"], linewidths=2, zorder=5)
        ax.annotate(
            f"{series[peak]:.1%}\n{dd.MONTHS_ES[peak.month - 1]} {peak.year}",
            xy=(peak, series[peak]),
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=PALETTE["text"],
            bbox={"boxstyle": "round,pad=0.4", "facecolor": PALETTE["surface_alt"], "edgecolor": PALETTE["line"]},
        )
        ax.set_ylim(series.min() * 0.95, series.max() * 1.12)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        _clean_axes(ax)
        fig.autofmt_xdate()
        grid.add(box)

    def _factors_chart(self, grid: ResponsiveGrid, pred: Any) -> None:
        factors = dd.top_risk_factors(pred)
        if factors.empty:
            return
        box, fig = self._chart_card(grid, "Factores de riesgo más frecuentes", "Explicaciones asociadas a los comprobantes en riesgo alto.")
        ax = fig.add_subplot()
        ordered = factors.iloc[::-1]
        positions = range(len(ordered))
        ax.hlines(positions, 0, ordered.values, color=PALETTE["line"], linewidth=2)
        ax.scatter(ordered.values, positions, s=90, color=PALETTE["danger"], zorder=3)
        for position, value in zip(positions, ordered.values):
            ax.annotate(f"{value:,}", (value, position), xytext=(9, 0), textcoords="offset points", va="center", fontsize=8, color=PALETTE["muted"])
        ax.set_yticks(list(positions), labels=ordered.index)
        ax.set_xlim(0, ordered.max() * 1.22)
        _clean_axes(ax, grid_axis="x")
        grid.add(box)

    def _suppliers_chart(self, grid: ResponsiveGrid, pred: Any) -> None:
        suppliers = dd.top_suppliers(pred)
        if suppliers.empty:
            return
        box, fig = self._chart_card(grid, "Proveedores con más riesgo alto", "RUC con más comprobantes pendientes en riesgo alto.")
        ax = fig.add_subplot()
        ordered = suppliers.iloc[::-1]
        ax.barh(ordered.index.astype(str), ordered["riesgo_alto"], color=PALETTE["accent"], height=0.55)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _clean_axes(ax, grid_axis="x")
        grid.add(box)

    # -------------------------------------------------------------------- table

    def _table_section(self, data: dd.DashboardData) -> None:
        box = card(self.content, padding=(20, 18))
        box.grid(row=4, column=0, sticky="ew", pady=(18, 0))
        inner = box.inner  # type: ignore[attr-defined]
        inner.columnconfigure(0, weight=1)

        head = ttk.Frame(inner, style="Card.TFrame")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        head.columnconfigure(1, weight=1)
        if "riesgo_alto_tile" in self.icons:
            ttk.Label(head, image=self.icons["riesgo_alto_tile"], style="CardBody.TLabel").grid(row=0, column=0, rowspan=2, padx=(0, 12))
        ttk.Label(head, text="Comprobantes prioritarios", style="CardTitle.TLabel").grid(row=0, column=1, sticky="sw")
        ttk.Label(head, text="Los 15 pendientes con mayor probabilidad de incidencia", style="CardBody.TLabel").grid(row=1, column=1, sticky="nw")
        SegmentedControl(head, ["Todos", "Alto", "Medio", "Bajo"], command=lambda level: self._fill_table(data, level), background=PALETTE["surface"]).grid(
            row=0, column=2, rowspan=2, sticky="e"
        )

        headings = {
            "ID_Comprobante": ("Comprobante", 120),
            "RUC_Proveedor": ("RUC proveedor", 110),
            "Tipo_Comprobante": ("Tipo", 130),
            "Importe_Total": ("Importe", 90),
            "Probabilidad_Incidencia": ("Probabilidad", 90),
            "Nivel_Riesgo": ("Riesgo", 80),
            "Razones_Principales": ("Factores", 320),
        }
        columns = list(dd.top_invoices(data.predictions, limit=1).columns)
        frame = RoundedCard(inner, padding=(10, 10), radius=16, fill=PALETTE["surface_alt"], background=PALETTE["surface"], inner_style="Table.TFrame")
        frame.grid(row=1, column=0, sticky="ew")
        table_area = frame.inner
        table_area.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(table_area, columns=columns, show="headings", height=15, style="Table.Treeview")
        for column in columns:
            label, width = headings.get(column, (column, 100))
            self.tree.heading(column, text=label, anchor="w")
            self.tree.column(column, width=width, minwidth=60, stretch=column == "Razones_Principales", anchor="w")
        self.tree.tag_configure("odd", background=PALETTE["surface"])
        self.tree.tag_configure("even", background=PALETTE["surface_alt"])
        xscroll = ttk.Scrollbar(table_area, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="ew")
        xscroll.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._fill_table(data, "Todos")

    def _fill_table(self, data: dd.DashboardData, level: str) -> None:
        predictions = data.predictions
        if level != "Todos":
            predictions = predictions[predictions["Nivel_Riesgo"] == level]
        table = dd.top_invoices(predictions)
        self.tree.delete(*self.tree.get_children())
        for index, (_, row) in enumerate(table.iterrows()):
            values = []
            for column in table.columns:
                value = row[column]
                if column == "Probabilidad_Incidencia":
                    value = f"{value * 100:.1f}%"
                elif column == "Importe_Total":
                    value = f"{value:,.2f}"
                elif column == "Nivel_Riesgo":
                    value = f"●  {value}"
                values.append(value)
            self.tree.insert("", "end", values=values, tags=("odd" if index % 2 else "even",))
        self.tree.configure(height=max(min(len(table), 15), 1))

    # ------------------------------------------------------------------ gallery

    def _gallery_section(self) -> None:
        self.gallery_items = [(OUTPUTS_DIR / name, text) for name, text in MODEL_CHARTS if (OUTPUTS_DIR / name).exists()]
        if not self.gallery_items:
            return
        ttk.Label(self.content, text="GRÁFICOS DEL MODELO", style="Section.TLabel").grid(row=5, column=0, sticky="w", pady=(18, 10))
        box = card(self.content, padding=(16, 14))
        box.grid(row=6, column=0, sticky="ew")
        inner = box.inner  # type: ignore[attr-defined]
        inner.columnconfigure(1, weight=1)
        PillButton(inner, "‹", command=lambda: self._move_gallery(-1), variant="secondary", background=PALETTE["surface"], padx=18).grid(row=0, column=0, sticky="w")
        self.gallery_title = ttk.Label(inner, text="", style="CardBody.TLabel", justify="center", anchor="center")
        self.gallery_title.grid(row=0, column=1, sticky="ew", padx=12)
        PillButton(inner, "›", command=lambda: self._move_gallery(1), variant="secondary", background=PALETTE["surface"], padx=18).grid(row=0, column=2, sticky="e")
        self.gallery_label = tk.Label(inner, background=PALETTE["surface"], borderwidth=0)
        self.gallery_label.grid(row=1, column=0, columnspan=3, pady=(12, 0))
        self.gallery_counter = tk.Label(inner, text="", background=PALETTE["surface"], foreground=PALETTE["muted"], font=(FONT, 8))
        self.gallery_counter.grid(row=2, column=0, columnspan=3, pady=(6, 0))
        inner.bind("<Configure>", lambda event: self._show_gallery(event.width), add="+")
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
