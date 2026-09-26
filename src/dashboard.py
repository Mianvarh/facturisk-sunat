"""Risk dashboard: interactive exploration of the pending-invoice predictions.

Charts show tooltips on hover and filter the priority table on click. When the
data is local (imported file or SUNAT scraping), supplier names, addresses and
registry flags are shown; the "hide names" switch masks them for public
screenshots.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from tkinter import StringVar, ttk
from typing import Any

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
from PIL import Image, ImageTk

import dashboard_data as dd
from configuracion import PADRONES_SUNAT
from paths import get_application_root
from theme import (
    FONT,
    PALETTE,
    RISK_COLORS,
    apply_chart_style,
    model_display_name,
    sequential_cmap,
)
from ui_widgets import (
    DonutRing,
    FitLabel,
    PillButton,
    ResponsiveGrid,
    RoundedCard,
    ScrollableFrame,
    SegmentedControl,
    SegmentedMeter,
    StatusPill,
    card,
)

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
TABLE_LIMIT = 50
PADRON_LABELS = {info["columna"]: info["nombre"] for info in PADRONES_SUNAT.values()}
HIDDEN_NAME = "•••••• (oculto)"


def _fmt_int(value: float) -> str:
    return f"{value:,.0f}"


def _clean_axes(ax: Any, grid_axis: str = "y") -> None:
    ax.grid(axis="both", visible=False)
    if grid_axis:
        ax.grid(axis=grid_axis, visible=True)
    ax.tick_params(length=0)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["line"])


def _short(text: str, length: int = 28) -> str:
    return text if len(text) <= length else text[: length - 1] + "…"


class ChartTooltip:
    """Hover tooltip and click handling for one matplotlib axes."""

    def __init__(self, canvas: FigureCanvasTkAgg, ax: Any) -> None:
        self.canvas = canvas
        self.ax = ax
        self.hover_handlers: list[Callable[[Any], tuple[tuple[float, float], str] | None]] = []
        self.click_handlers: list[Callable[[Any], bool]] = []
        self.annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=8,
            color=PALETTE["text"],
            zorder=20,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": PALETTE["surface_alt"], "edgecolor": PALETTE["line"]},
            annotation_clip=False,
        )
        self.annotation.set_visible(False)
        canvas.mpl_connect("motion_notify_event", self._on_move)
        canvas.mpl_connect("button_press_event", self._on_click)
        canvas.mpl_connect("figure_leave_event", lambda _event: self._hide())

    def _hide(self) -> None:
        if self.annotation.get_visible():
            self.annotation.set_visible(False)
            self.canvas.draw_idle()
        self.canvas.get_tk_widget().configure(cursor="")

    def _on_move(self, event: Any) -> None:
        if event.inaxes is not self.ax:
            self._hide()
            return
        for handler in self.hover_handlers:
            result = handler(event)
            if result:
                (x, y), text = result
                self.annotation.xy = (x, y)
                self.annotation.set_text(text)
                width = self.canvas.get_tk_widget().winfo_width() or 1
                right_side = event.x > width * 0.6
                self.annotation.set_position((-12, 12) if right_side else (12, 12))
                self.annotation.set_horizontalalignment("right" if right_side else "left")
                self.annotation.set_visible(True)
                self.canvas.get_tk_widget().configure(cursor="hand2" if self.click_handlers else "")
                self.canvas.draw_idle()
                return
        self._hide()

    def _on_click(self, event: Any) -> None:
        if event.inaxes is not self.ax:
            return
        for handler in self.click_handlers:
            if handler(event):
                return


class DashboardView(ttk.Frame):
    """KPIs, interactive charts, filterable priority table and the model chart gallery."""

    def __init__(self, parent: tk.Misc, icons: dict[str, ImageTk.PhotoImage], go_to_pipeline: Callable[[], None]) -> None:
        super().__init__(parent, style="App.TFrame")
        apply_chart_style()
        self.icons = icons
        self.go_to_pipeline = go_to_pipeline
        self.canvases: list[FigureCanvasTkAgg] = []
        self.tooltips: list[ChartTooltip] = []
        self.data: dd.DashboardData | None = None
        self.hide_names = os.environ.get("FACTURISK_OCULTAR_NOMBRES") == "1"
        self.filters: dict[str, Any] = {"level": "Todos", "tipo": None, "ruc": None, "factor": None}
        self.search_var = StringVar()
        self.search_var.trace_add("write", lambda *_args: self._schedule_table())
        self._table_job: str | None = None
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
        actions = ttk.Frame(header, style="App.TFrame")
        actions.grid(row=0, column=1, sticky="ne")
        self.names_button = PillButton(actions, self._names_label(), command=self._toggle_names, variant="secondary")
        self.names_button.pack(side="left", padx=(0, 8))
        PillButton(actions, "↻  Actualizar", command=self.refresh, variant="secondary").pack(side="left")

        self.scroll = ScrollableFrame(self)
        self.scroll.grid(row=1, column=0, sticky="nsew")
        self.content = ttk.Frame(self.scroll.body, style="App.TFrame", padding=(28, 0, 28, 24))
        self.content.pack(fill="both", expand=True)
        self.content.columnconfigure(0, weight=1)

    # ------------------------------------------------------------------ helpers

    def _names_label(self) -> str:
        return "Mostrar nombres" if self.hide_names else "Ocultar nombres"

    def _toggle_names(self) -> None:
        self.hide_names = not self.hide_names
        self.names_button.configure(text=self._names_label())
        self.refresh()

    def _names(self, frame: pd.DataFrame) -> pd.Series:
        names = dd.supplier_names(frame)
        if self.hide_names:
            return names.where(names.eq(""), HIDDEN_NAME)
        return names

    # ------------------------------------------------------------------ refresh

    def refresh(self) -> None:
        for canvas in self.canvases:
            canvas.get_tk_widget().destroy()
        self.canvases.clear()
        self.tooltips.clear()
        for child in self.content.winfo_children():
            child.destroy()

        self.data = dd.load_dashboard_data()
        if self.data is None:
            self.subtitle.configure(text="Aún no hay predicciones para analizar.")
            self.status.set("Sin predicciones", PALETTE["subtle"])
            self._empty_state()
            return

        data = self.data
        model = model_display_name(data.metrics.get("mejor_modelo", "modelo"))
        self.status.set(f"{model} · umbral {data.metrics.get('umbral', 0):.0%}", PALETTE["success"])
        names_state = "con razón social" if dd.has_names(data.predictions) and not self.hide_names else "sin nombres"
        self.subtitle.configure(
            text=f"{len(data.predictions):,} comprobantes pendientes evaluados · {names_state} · actualizado {datetime.now():%d/%m/%Y %H:%M}"
        )
        self._kpi_section(data)
        self._chart_section(data)
        self._table_section()
        self._gallery_section()

    def _empty_state(self) -> None:
        box = card(self.content, padding=(28, 28))
        box.grid(row=0, column=0, sticky="ew", pady=(8, 0))
        inner = box.inner
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

    def _kpi_card(self, parent: tk.Misc, icon_key: str, title: str) -> RoundedCard:
        box = card(parent, padding=(18, 16))
        head = ttk.Frame(box.inner, style="Card.TFrame")
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
        ttk.Label(box.inner, text=_fmt_int(values["pendientes"]), style="CardValue.TLabel").pack(anchor="w", pady=(14, 0))
        ttk.Label(box.inner, text=f"Probabilidad media de incidencia {values['probabilidad_media'] * 100:.1f}%", style="CardBody.TLabel").pack(anchor="w")
        grid.add(box)

        box = self._kpi_card(grid, "riesgo_alto_tile", "Riesgo alto")
        row = ttk.Frame(box.inner, style="Card.TFrame")
        row.pack(fill="x", pady=(8, 0))
        DonutRing(row, values["riesgo_alto_pct"] / 100, "de pendientes", PALETTE["danger"], size=112).pack(side="left")
        side = ttk.Frame(row, style="Card.TFrame")
        side.pack(side="left", padx=(14, 0))
        ttk.Label(side, text=_fmt_int(values["riesgo_alto"]), style="CardValue.TLabel").pack(anchor="w")
        ttk.Label(side, text="a revisar\nprimero", style="CardBody.TLabel", justify="left").pack(anchor="w")
        grid.add(box)

        box = self._kpi_card(grid, "recall_tile", "Para revisión")
        ttk.Label(box.inner, text=f"{values['marcados_revision_pct']:.1f} %", style="CardValue.TLabel").pack(anchor="w", pady=(14, 6))
        SegmentedMeter(box.inner, values["marcados_revision_pct"] / 100, PALETTE["primary"]).pack(fill="x")
        scale = ttk.Frame(box.inner, style="Card.TFrame")
        scale.pack(fill="x", pady=(4, 0))
        ttk.Label(scale, text="0%", style="CardCaption.TLabel").pack(side="left")
        ttk.Label(scale, text="100%", style="CardCaption.TLabel").pack(side="right")
        ttk.Label(box.inner, text=f"{_fmt_int(values['marcados_revision'])} superan el umbral del modelo", style="CardBody.TLabel").pack(anchor="w", pady=(4, 0))
        grid.add(box)

        box = self._kpi_card(grid, "modelo_tile", "Modelo en uso")
        FitLabel(box.inner, text=model_display_name(values["modelo"]), max_size=22, min_size=11).pack(anchor="w", fill="x", pady=(14, 6))
        pills = ttk.Frame(box.inner, style="Card.TFrame")
        pills.pack(anchor="w")
        StatusPill(pills, f"Recall {float(values['recall'] or 0):.0%}", PALETTE["primary"], background=PALETTE["surface"]).pack(side="left", padx=(0, 6))
        if values["pr_auc"] is not None:
            StatusPill(pills, f"PR-AUC {values['pr_auc']:.2f}", PALETTE["accent"], background=PALETTE["surface"]).pack(side="left")
        caption = ttk.Label(box.inner, text="Validación temporal y probabilidades calibradas", style="CardBody.TLabel", justify="left")
        caption.pack(anchor="w", fill="x", pady=(8, 0))
        box.inner.bind("<Configure>", lambda event: caption.configure(wraplength=max(event.width - 40, 120)), add="+")
        grid.add(box)

    # ------------------------------------------------------------------- charts

    def _chart_card(self, parent: tk.Misc, title: str, caption: str, height: int = CHART_HEIGHT_PX) -> tuple[RoundedCard, Figure, FigureCanvasTkAgg]:
        box = card(parent, padding=(16, 14))
        ttk.Label(box.inner, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(box.inner, text=caption, style="CardBody.TLabel").pack(anchor="w", pady=(2, 8))
        figure = Figure(figsize=(5, height / 96), dpi=96, layout="constrained")
        canvas = FigureCanvasTkAgg(figure, master=box.inner)
        widget = canvas.get_tk_widget()
        widget.configure(height=height, highlightthickness=0, background=PALETTE["surface"])
        widget.pack(fill="x", expand=False)
        self.canvases.append(canvas)
        return box, figure, canvas

    def _tooltip(self, canvas: FigureCanvasTkAgg, ax: Any) -> ChartTooltip:
        tip = ChartTooltip(canvas, ax)
        self.tooltips.append(tip)
        return tip

    @staticmethod
    def _bar_hover(bars: Any, texts: list[str]) -> Callable[[Any], tuple[tuple[float, float], str] | None]:
        def handler(event: Any) -> tuple[tuple[float, float], str] | None:
            for bar, text in zip(bars, texts, strict=True):
                if bar.contains(event)[0]:
                    return (bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height()), text
            return None

        return handler

    @staticmethod
    def _barh_hover(bars: Any, texts: list[str]) -> Callable[[Any], tuple[tuple[float, float], str] | None]:
        def handler(event: Any) -> tuple[tuple[float, float], str] | None:
            for bar, text in zip(bars, texts, strict=True):
                if bar.contains(event)[0]:
                    return (bar.get_x() + bar.get_width(), bar.get_y() + bar.get_height() / 2), text
            return None

        return handler

    @staticmethod
    def _bar_click(bars: Any, values: list[Any], action: Callable[[Any], None]) -> Callable[[Any], bool]:
        def handler(event: Any) -> bool:
            for bar, value in zip(bars, values, strict=True):
                if bar.contains(event)[0]:
                    action(value)
                    return True
            return False

        return handler

    def _chart_section(self, data: dd.DashboardData) -> None:
        pred = data.predictions
        threshold = data.metrics.get("umbral")
        head = ttk.Frame(self.content, style="App.TFrame")
        head.grid(row=1, column=0, sticky="ew", pady=(16, 10))
        ttk.Label(head, text="ANÁLISIS DE PENDIENTES", style="Section.TLabel").pack(side="left")
        ttk.Label(head, text="  ·  pasa el mouse para ver el detalle; haz clic para filtrar la tabla", style="Sub.TLabel").pack(side="left")
        grid = ResponsiveGrid(self.content, min_item_width=440, max_columns=2)
        grid.grid(row=2, column=0, sticky="ew")

        self._risk_levels_chart(grid, pred)
        self._quadrant_chart(grid, pred, threshold)
        self._heatmap_chart(grid, pred)
        self._monthly_chart(grid, pred)
        self._factors_chart(grid, pred)
        self._suppliers_chart(grid, pred)
        self._padron_chart(grid, pred)

    def _risk_levels_chart(self, grid: ResponsiveGrid, pred: pd.DataFrame) -> None:
        counts = dd.risk_level_counts(pred)
        total = max(int(counts.sum()), 1)
        box, fig, canvas = self._chart_card(grid, "Comprobantes por nivel de riesgo", "Clic en una barra para ver solo ese nivel.")
        ax = fig.add_subplot()
        bars = ax.bar(counts.index, counts.values, color=[RISK_COLORS[level] for level in counts.index], width=0.55)
        ax.bar_label(bars, labels=[f"{value:,}" for value in counts.values], fontsize=9, color=PALETTE["text"], padding=4)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
        ax.margins(y=0.12)
        _clean_axes(ax)
        tip = self._tooltip(canvas, ax)
        mean_prob = pred.groupby("Nivel_Riesgo")["Probabilidad_Incidencia"].mean()
        texts = [
            f"Riesgo {level}\n{count:,} comprobantes ({count / total:.1%})\nProbabilidad media {mean_prob.get(level, 0):.1%}"
            for level, count in counts.items()
        ]
        tip.hover_handlers.append(self._bar_hover(bars, texts))
        tip.click_handlers.append(self._bar_click(bars, list(counts.index), lambda level: self.set_filter(level=level)))
        grid.add(box)

    def _quadrant_chart(self, grid: ResponsiveGrid, pred: pd.DataFrame, threshold: float | None) -> None:
        points = dd.amount_vs_probability(pred)
        if points.empty:
            return
        box, fig, canvas = self._chart_card(
            grid,
            "Importe vs. probabilidad de incidencia",
            "Cada punto es un pendiente (muestra). Arriba a la derecha: alto importe y alto riesgo. Clic para ver su ficha.",
        )
        ax = fig.add_subplot()
        collections = []
        for level in dd.RISK_LEVELS:
            subset = points[points["Nivel_Riesgo"] == level]
            collection = ax.scatter(subset["Importe_Total"], subset["Probabilidad_Incidencia"], s=10, alpha=0.6, color=RISK_COLORS[level], label=level, linewidths=0)
            collections.append((collection, subset.index))
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
        ax.legend(loc="upper left", fontsize=8, markerscale=2.5, ncols=3, scatterpoints=1, handletextpad=0.2, columnspacing=1.2, frameon=True, facecolor=PALETTE["surface"], edgecolor="none", framealpha=1)
        _clean_axes(ax, grid_axis="")
        names = self._names(pred)
        tip = self._tooltip(canvas, ax)

        def find_point(event: Any) -> Any:
            for collection, index in collections:
                hit, info = collection.contains(event)
                if hit and len(info["ind"]):
                    return index[info["ind"][0]]
            return None

        def hover(event: Any) -> tuple[tuple[float, float], str] | None:
            row_index = find_point(event)
            if row_index is None:
                return None
            row = pred.loc[row_index]
            name = names.loc[row_index]
            lines = [str(row["ID_Comprobante"]), f"RUC {row['RUC_Proveedor']}"]
            if name:
                lines.append(_short(name, 40))
            lines += [f"Importe {row['Importe_Total']:,.2f} {row.get('Moneda', '')}".strip(), f"Probabilidad {row['Probabilidad_Incidencia']:.1%} · {row['Nivel_Riesgo']}"]
            return (row["Importe_Total"], row["Probabilidad_Incidencia"]), "\n".join(lines)

        def click(event: Any) -> bool:
            row_index = find_point(event)
            if row_index is None:
                return False
            self.show_detail(pred.loc[row_index])
            return True

        tip.hover_handlers.append(hover)
        tip.click_handlers.append(click)
        grid.add(box)

    def _heatmap_chart(self, grid: ResponsiveGrid, pred: pd.DataFrame) -> None:
        table = dd.risk_heatmap(pred)
        if table is None or table.empty:
            return
        counts = dd.risk_heatmap(pred.assign(Probabilidad_Incidencia=1.0))
        box, fig, canvas = self._chart_card(grid, "Mapa de calor: tipo × mes de emisión", "Probabilidad media de incidencia (últimos 12 meses). Clic en una fila para filtrar ese tipo.")
        ax = fig.add_subplot()
        matrix = table.to_numpy(dtype=float)
        image = ax.imshow(matrix, aspect="auto", cmap=sequential_cmap(), interpolation="nearest")
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
        tip = self._tooltip(canvas, ax)

        def cell(event: Any) -> tuple[int, int] | None:
            if event.xdata is None or event.ydata is None:
                return None
            col, row = round(event.xdata), round(event.ydata)
            if 0 <= row < matrix.shape[0] and 0 <= col < matrix.shape[1] and not np.isnan(matrix[row, col]):
                return row, col
            return None

        def hover(event: Any) -> tuple[tuple[float, float], str] | None:
            found = cell(event)
            if not found:
                return None
            row, col = found
            n = counts.iloc[row, col] if counts is not None else float("nan")
            count_text = f"\n{int(n):,} comprobantes" if not pd.isna(n) else ""
            return (col, row), f"{table.index[row]} · {table.columns[col]}\nProbabilidad media {matrix[row, col]:.1%}{count_text}"

        def click(event: Any) -> bool:
            found = cell(event)
            if not found:
                return False
            self.set_filter(tipo=str(table.index[found[0]]))
            return True

        tip.hover_handlers.append(hover)
        tip.click_handlers.append(click)
        grid.add(box)

    def _monthly_chart(self, grid: ResponsiveGrid, pred: pd.DataFrame) -> None:
        monthly = dd.risk_by_month(pred)
        if monthly is None or len(monthly) < 2:
            return
        box, fig, canvas = self._chart_card(grid, "Riesgo por mes de emisión", "Probabilidad media de incidencia; se marca el mes más riesgoso.")
        ax = fig.add_subplot()
        series = monthly["probabilidad_media"]
        smooth = series.rolling(3, center=True, min_periods=1).mean()
        ax.plot(series.index, series.values, color=PALETTE["subtle"], linewidth=1, alpha=0.8)
        ax.plot(smooth.index, smooth.values, color=PALETTE["primary"], linewidth=2.4)
        ax.fill_between(smooth.index, smooth.values, series.min() * 0.95, color=PALETTE["primary"], alpha=0.12)
        peak = series.idxmax()
        ax.scatter([peak], [series[peak]], s=60, color=PALETTE["background"], edgecolors=PALETTE["text"], linewidths=2, zorder=5)
        ax.set_ylim(series.min() * 0.95, series.max() * 1.12)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        _clean_axes(ax)
        fig.autofmt_xdate()
        marker = ax.scatter([], [], s=40, color=PALETTE["primary"], zorder=6)
        tip = self._tooltip(canvas, ax)
        x_values = matplotlib.dates.date2num(series.index.to_pydatetime())

        def hover(event: Any) -> tuple[tuple[float, float], str] | None:
            if event.xdata is None:
                return None
            position = int(np.argmin(np.abs(x_values - event.xdata)))
            month = series.index[position]
            marker.set_offsets([[x_values[position], series.iloc[position]]])
            text = (
                f"{dd.MONTHS_ES[month.month - 1]} {month.year}\nProbabilidad media {series.iloc[position]:.2%}"
                f"\n{int(monthly['riesgo_alto'].iloc[position]):,} en riesgo alto"
            )
            return (month, series.iloc[position]), text

        tip.hover_handlers.append(hover)
        grid.add(box)

    def _factors_chart(self, grid: ResponsiveGrid, pred: pd.DataFrame) -> None:
        factors = dd.top_risk_factors(pred)
        if factors.empty:
            return
        box, fig, canvas = self._chart_card(grid, "Factores de riesgo más frecuentes", "Explicaciones de los comprobantes en riesgo alto. Clic para filtrar por factor.")
        ax = fig.add_subplot()
        ordered = factors.iloc[::-1]
        positions = list(range(len(ordered)))
        ax.hlines(positions, 0, ordered.values, color=PALETTE["line"], linewidth=2)
        ax.scatter(ordered.values, positions, s=90, color=PALETTE["danger"], zorder=3)
        for position, value in zip(positions, ordered.values, strict=True):
            ax.annotate(f"{value:,}", (value, position), xytext=(9, 0), textcoords="offset points", va="center", fontsize=8, color=PALETTE["muted"])
        ax.set_yticks(positions, labels=ordered.index)
        ax.set_xlim(0, ordered.max() * 1.22)
        _clean_axes(ax, grid_axis="x")
        high_total = max(int((pred["Nivel_Riesgo"] == "Alto").sum()), 1)
        tip = self._tooltip(canvas, ax)

        def row_at(event: Any) -> int | None:
            if event.ydata is None:
                return None
            position = round(event.ydata)
            return position if 0 <= position < len(ordered) and abs(event.ydata - position) < 0.4 else None

        def hover(event: Any) -> tuple[tuple[float, float], str] | None:
            position = row_at(event)
            if position is None:
                return None
            value = int(ordered.iloc[position])
            return (value, position), f"{ordered.index[position]}\n{value:,} comprobantes ({value / high_total:.1%} del riesgo alto)"

        def click(event: Any) -> bool:
            position = row_at(event)
            if position is None:
                return False
            self.set_filter(factor=str(ordered.index[position]), level="Alto")
            return True

        tip.hover_handlers.append(hover)
        tip.click_handlers.append(click)
        grid.add(box)

    def _suppliers_chart(self, grid: ResponsiveGrid, pred: pd.DataFrame) -> None:
        high = pred[pred["Nivel_Riesgo"] == "Alto"]
        if high.empty:
            return
        names = self._names(high)
        table = (
            high.assign(_nombre=names)
            .groupby("RUC_Proveedor")
            .agg(riesgo_alto=("ID_Comprobante", "size"), probabilidad_media=("Probabilidad_Incidencia", "mean"), importe=("Importe_Total", "sum"), nombre=("_nombre", "first"))
            .sort_values(["riesgo_alto", "probabilidad_media"], ascending=False)
            .head(10)
        )
        has_names = bool(table["nombre"].ne("").any())
        caption = "Clic en una barra para ver sus comprobantes." + ("" if has_names else " (Importa tus datos o ejecuta el scraping para ver la razón social.)")
        box, fig, canvas = self._chart_card(grid, "Proveedores con más riesgo alto", caption)
        ax = fig.add_subplot()
        ordered = table.iloc[::-1]
        labels = [_short(name, 26) if name else ruc for ruc, name in zip(ordered.index.astype(str), ordered["nombre"], strict=True)]
        bars = ax.barh(labels, ordered["riesgo_alto"], color=PALETTE["accent"], height=0.55)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _clean_axes(ax, grid_axis="x")
        tip = self._tooltip(canvas, ax)
        texts = [
            f"{name or 'Razón social no disponible'}\nRUC {ruc}\n{int(row['riesgo_alto'])} en riesgo alto · prob. media {row['probabilidad_media']:.1%}\nImporte {row['importe']:,.2f}"
            for ruc, name, (_, row) in zip(ordered.index.astype(str), ordered["nombre"], ordered.iterrows(), strict=True)
        ]
        tip.hover_handlers.append(self._barh_hover(bars, texts))
        tip.click_handlers.append(self._bar_click(bars, list(ordered.index.astype(str)), lambda ruc: self.set_filter(ruc=ruc, level="Todos")))
        grid.add(box)

    def _padron_chart(self, grid: ResponsiveGrid, pred: pd.DataFrame) -> None:
        table = dd.padron_risk(pred, PADRON_LABELS)
        if table.empty:
            return
        box, fig, canvas = self._chart_card(grid, "Riesgo según padrones SUNAT", "Porcentaje en riesgo alto dentro y fuera de cada padrón (variables externas del modelo).")
        ax = fig.add_subplot()
        positions = np.arange(len(table))
        inside = ax.bar(positions - 0.18, table["dentro"], width=0.34, color=PALETTE["primary"], label="En el padrón")
        outside = ax.bar(positions + 0.18, table["fuera"], width=0.34, color=PALETTE["subtle"], label="Fuera del padrón")
        ax.set_xticks(positions, labels=[_short(label.replace("Agentes de ", "Ag. "), 22) for label in table["padron"]])
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.legend(loc="upper right", fontsize=8)
        _clean_axes(ax)
        tip = self._tooltip(canvas, ax)
        texts_in = [f"{row.padron}\nEn el padrón: {row.dentro:.1f}% en riesgo alto\n{row.comprobantes:,} comprobantes" for row in table.itertuples()]
        texts_out = [f"{row.padron}\nFuera del padrón: {row.fuera:.1f}% en riesgo alto" for row in table.itertuples()]
        tip.hover_handlers.append(self._bar_hover(inside, texts_in))
        tip.hover_handlers.append(self._bar_hover(outside, texts_out))
        grid.add(box)

    # -------------------------------------------------------------------- table

    def set_filter(self, **changes: Any) -> None:
        self.filters.update(changes)
        if "level" in changes and hasattr(self, "segmented"):
            self.segmented.selected = changes["level"]
            self.segmented._draw()
        self._fill_table()
        self.after(50, lambda: self.scroll.canvas.yview_moveto(max(0.0, self.table_card.winfo_y() / max(self.scroll.body.winfo_height(), 1) - 0.02)))

    def _clear_filters(self) -> None:
        self.filters = {"level": "Todos", "tipo": None, "ruc": None, "factor": None}
        self.search_var.set("")
        if hasattr(self, "segmented"):
            self.segmented.selected = "Todos"
            self.segmented._draw()
        self._fill_table()

    def _schedule_table(self) -> None:
        if self._table_job:
            self.after_cancel(self._table_job)
        self._table_job = self.after(250, self._fill_table)

    def _table_section(self) -> None:
        self.table_card = card(self.content, padding=(20, 18))
        self.table_card.grid(row=4, column=0, sticky="ew", pady=(18, 0))
        inner = self.table_card.inner
        inner.columnconfigure(0, weight=1)

        head = ttk.Frame(inner, style="Card.TFrame")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        head.columnconfigure(1, weight=1)
        if "riesgo_alto_tile" in self.icons:
            ttk.Label(head, image=self.icons["riesgo_alto_tile"], style="CardBody.TLabel").grid(row=0, column=0, rowspan=2, padx=(0, 12))
        ttk.Label(head, text="Comprobantes prioritarios", style="CardTitle.TLabel").grid(row=0, column=1, sticky="sw")
        ttk.Label(head, text="Ordenados por probabilidad de incidencia. Doble clic en una fila para ver su ficha completa.", style="CardBody.TLabel").grid(row=1, column=1, sticky="nw")
        self.segmented = SegmentedControl(head, ["Todos", "Alto", "Medio", "Bajo"], command=lambda level: self.set_filter(level=level), background=PALETTE["surface"])
        self.segmented.grid(row=0, column=2, rowspan=2, sticky="e")

        tools = ttk.Frame(inner, style="Card.TFrame")
        tools.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        tools.columnconfigure(1, weight=1)
        ttk.Label(tools, text="Buscar", style="FieldLabel.TLabel").grid(row=0, column=0, padx=(0, 10))
        ttk.Entry(tools, textvariable=self.search_var, style="Field.TEntry").grid(row=0, column=1, sticky="ew")
        self.filter_pill = StatusPill(tools, "Sin filtros", PALETTE["subtle"], background=PALETTE["surface"])
        self.filter_pill.grid(row=0, column=2, padx=(12, 8))
        PillButton(tools, "Limpiar filtros", command=self._clear_filters, variant="ghost", background=PALETTE["surface"], height=34).grid(row=0, column=3)
        PillButton(tools, "Exportar CSV", command=self._export, variant="secondary", background=PALETTE["surface"], height=34).grid(row=0, column=4, padx=(8, 0))

        self.table_columns = {
            "ID_Comprobante": ("Comprobante", 115),
            "RUC_Proveedor": ("RUC", 100),
            "_nombre": ("Razón social", 220),
            "Tipo_Comprobante": ("Tipo", 115),
            "Importe_Total": ("Importe", 90),
            "Probabilidad_Incidencia": ("Probabilidad", 85),
            "Nivel_Riesgo": ("Riesgo", 75),
            "Estado_RUC": ("Estado RUC", 120),
            "Razones_Principales": ("Factores", 300),
        }
        frame = RoundedCard(inner, padding=(10, 10), radius=16, fill=PALETTE["surface_alt"], background=PALETTE["surface"], inner_style="Table.TFrame")
        frame.grid(row=2, column=0, sticky="ew")
        area = frame.inner
        area.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(area, columns=list(self.table_columns), show="headings", height=14, style="Table.Treeview")
        for column, (label, width) in self.table_columns.items():
            self.tree.heading(column, text=label, anchor="w")
            self.tree.column(column, width=width, minwidth=60, stretch=column in {"_nombre", "Razones_Principales"}, anchor="w")
        self.tree.tag_configure("odd", background=PALETTE["surface"])
        self.tree.tag_configure("even", background=PALETTE["surface_alt"])
        yscroll = ttk.Scrollbar(area, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(area, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="ew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.tree.bind("<Double-1>", self._on_row_open)
        self.table_count = ttk.Label(inner, text="", style="CardBody.TLabel")
        self.table_count.grid(row=3, column=0, sticky="w", pady=(10, 0))
        self._fill_table()

    def _filtered(self) -> pd.DataFrame:
        assert self.data is not None
        return dd.filter_predictions(
            self.data.predictions,
            level=self.filters["level"],
            text=self.search_var.get(),
            tipo=self.filters["tipo"],
            ruc=self.filters["ruc"],
            factor=self.filters["factor"],
        )

    def _fill_table(self) -> None:
        self._table_job = None
        if self.data is None or not hasattr(self, "tree"):
            return
        filtered = self._filtered()
        table = filtered.nlargest(TABLE_LIMIT, "Probabilidad_Incidencia")
        names = self._names(table)
        self.tree.delete(*self.tree.get_children())
        self.row_index: dict[str, Any] = {}
        for position, (index, row) in enumerate(table.iterrows()):
            values = []
            for column in self.table_columns:
                if column == "_nombre":
                    value = names.loc[index] or "—"
                elif column == "Probabilidad_Incidencia":
                    value = f"{row[column] * 100:.1f}%"
                elif column == "Importe_Total":
                    value = f"{row[column]:,.2f}"
                elif column == "Nivel_Riesgo":
                    value = f"●  {row[column]}"
                else:
                    value = row.get(column, "")
                    value = "—" if pd.isna(value) else value
                values.append(value)
            item = self.tree.insert("", "end", values=values, tags=("odd" if position % 2 else "even",))
            self.row_index[item] = index

        active = []
        if self.filters["level"] != "Todos":
            active.append(f"Riesgo {self.filters['level']}")
        if self.filters["tipo"]:
            active.append(self.filters["tipo"])
        if self.filters["ruc"]:
            active.append(f"RUC {self.filters['ruc']}")
        if self.filters["factor"]:
            active.append(_short(self.filters["factor"], 24))
        if self.search_var.get().strip():
            active.append(f"“{_short(self.search_var.get().strip(), 16)}”")
        self.filter_pill.set(" · ".join(active) if active else "Sin filtros", PALETTE["primary"] if active else PALETTE["subtle"])
        self.table_count.configure(text=f"Mostrando {len(table):,} de {len(filtered):,} comprobantes que coinciden (máximo {TABLE_LIMIT} por vista).")

    def _on_row_open(self, _event: tk.Event) -> None:
        selection = self.tree.selection()
        if selection and self.data is not None:
            self.show_detail(self.data.predictions.loc[self.row_index[selection[0]]])

    def _export(self) -> None:
        if self.data is None:
            return
        filtered = self._filtered().sort_values("Probabilidad_Incidencia", ascending=False).copy()
        filtered.insert(2, "Razon_Social", self._names(filtered))
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUTS_DIR / f"comprobantes_filtrados_{datetime.now():%Y%m%d_%H%M%S}.csv"
        filtered.to_csv(path, index=False, encoding="utf-8-sig")
        self.table_count.configure(text=f"✓ {len(filtered):,} comprobantes exportados a outputs/{path.name}")
        if sys.platform == "win32":
            os.startfile(OUTPUTS_DIR)

    # ------------------------------------------------------------------- detail

    def show_detail(self, row: pd.Series) -> None:
        """Modal sheet with every available field of one pending invoice."""

        window = tk.Toplevel(self)
        window.title(f"Comprobante {row.get('ID_Comprobante', '')}")
        window.configure(background=PALETTE["background"])
        window.geometry("600x680")
        window.transient(self.winfo_toplevel())
        box = card(window, padding=(22, 20))
        box.pack(fill="both", expand=True, padx=16, pady=16)
        inner = box.inner
        inner.columnconfigure(1, weight=1)
        name = self._names(row.to_frame().T).iloc[0]
        ttk.Label(inner, text=str(row.get("ID_Comprobante", "")), style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(inner, text=name or "Razón social no disponible", style="CardBody.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 10))
        level = str(row.get("Nivel_Riesgo", ""))
        StatusPill(inner, f"Riesgo {level} · {float(row.get('Probabilidad_Incidencia', 0)):.1%}", RISK_COLORS.get(level, PALETTE["subtle"]), background=PALETTE["surface"]).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(0, 14)
        )
        fields = [
            ("RUC proveedor", row.get("RUC_Proveedor")),
            ("Tipo de comprobante", row.get("Tipo_Comprobante")),
            ("Fecha de emisión", row.get("Fecha_Emision")),
            ("Moneda", row.get("Moneda")),
            ("Importe", f"{float(row.get('Importe_Total', 0)):,.2f}"),
            ("Predicción", row.get("Prediccion_Texto")),
            ("Estado RUC (SUNAT)", row.get("Estado_RUC")),
            ("Condición de domicilio", row.get("Condicion_Domicilio")),
            ("Domicilio fiscal", HIDDEN_NAME if self.hide_names and not pd.isna(row.get("Domicilio_Fiscal")) else row.get("Domicilio_Fiscal")),
            ("Ubigeo", row.get("Ubigeo")),
        ]
        for column, label in PADRON_LABELS.items():
            if column in row.index:
                fields.append((label, "Sí" if row.get(column) == 1 else "No"))
        fields.append(("Factores de riesgo", row.get("Razones_Principales")))
        for position, (label, value) in enumerate(fields, start=3):
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                value = "—"
            if isinstance(value, pd.Timestamp):
                value = f"{value:%d/%m/%Y}"
            ttk.Label(inner, text=label, style="FieldLabel.TLabel").grid(row=position, column=0, sticky="nw", padx=(0, 16), pady=4)
            ttk.Label(inner, text=str(value), style="CardBody.TLabel", wraplength=280, justify="left").grid(row=position, column=1, sticky="w", pady=4)
        PillButton(inner, "Cerrar", command=window.destroy, variant="secondary", background=PALETTE["surface"]).grid(row=len(fields) + 4, column=0, columnspan=2, sticky="w", pady=(16, 0))

    # ------------------------------------------------------------------ gallery

    def _gallery_section(self) -> None:
        self.gallery_items = [(OUTPUTS_DIR / name, text) for name, text in MODEL_CHARTS if (OUTPUTS_DIR / name).exists()]
        if not self.gallery_items:
            return
        ttk.Label(self.content, text="GRÁFICOS DEL MODELO", style="Section.TLabel").grid(row=5, column=0, sticky="w", pady=(18, 10))
        box = card(self.content, padding=(16, 14))
        box.grid(row=6, column=0, sticky="ew")
        inner = box.inner
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

