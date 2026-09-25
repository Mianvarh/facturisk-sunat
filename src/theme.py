"""Visual identity of FactuRisk SUNAT: palette, bundled typography and styles."""

from __future__ import annotations

import sys
from typing import Any

from paths import get_application_root

FONTS_DIR = get_application_root() / "assets" / "fonts"

PALETTE = {
    # Surfaces: near-black with a violet tint, from darkest to lightest
    "background": "#0B0A10",
    "ink": "#0E0D14",
    "ink_hover": "#17151F",
    "ink_active": "#1F1C2B",
    "surface": "#14121C",
    "surface_alt": "#1D1A28",
    "line": "#272336",
    # Text
    "text": "#F3F1F8",
    "muted": "#9C97AE",
    "subtle": "#625D73",
    "nav_text": "#C7C2D6",
    "nav_muted": "#6E6883",
    "nav_icon": "#B3ADC4",
    # Brand
    "primary": "#A89CFF",
    "primary_hover": "#BCB2FF",
    "primary_soft": "#231F38",
    "primary_light": "#CFC8FF",
    "on_primary": "#16122B",
    "accent": "#7C5CFF",
    "accent_light": "#C4B5FD",
    "accent_soft": "#241C40",
    "danger": "#F0506E",
    "danger_soft": "#2E1822",
    "success": "#4ADE80",
    "warning": "#FBBF24",
    # Console
    "console_bg": "#09080D",
    "console_text": "#D9D5E6",
}

RISK_COLORS = {"Bajo": "#6C8CFF", "Medio": "#A78BFA", "Alto": "#F0506E"}
CHART_SERIES = ["#A89CFF", "#F0506E", "#6C8CFF", "#4ADE80", "#FBBF24", "#9C97AE"]
# Sequential scale (low → high risk) and diverging scale (negative → positive).
SEQUENTIAL_COLORS = ["#211C3F", "#3B2F7A", "#6C5CE7", "#A78BFA", "#F0506E"]
DIVERGING_COLORS = ["#6C8CFF", "#1D1A28", "#F0506E"]

# Solid icon tiles, one hue per concept (white glyph unless listed as dark).
TILE_COLORS = {
    "todo": "#7C5CFF",
    "inspect": "#3B82F6",
    "distribuido": "#14B8A6",
    "scraping": "#F97316",
    "mongodb": "#22C55E",
    "preparar": "#EAB308",
    "entrenar": "#8B5CF6",
    "predecir": "#EC4899",
    "documentar": "#64748B",
    "dashboard": "#7C5CFF",
    "modelo": "#8B5CF6",
    "recall": "#3B82F6",
    "f1": "#14B8A6",
    "pendientes": "#F97316",
    "riesgo_alto": "#EF4444",
    "importe": "#EAB308",
}
TILES_WITH_DARK_GLYPH = {"preparar", "importe"}

FALLBACK_FONT = "Segoe UI"
FONT = "Outfit"
FONT_MEDIUM = "Outfit Medium"
FONT_SEMIBOLD = "Outfit SemiBold"
FONT_MONO = "Consolas"

_fonts_loaded = False


def load_fonts() -> bool:
    """Register the bundled Outfit typeface for this process (no system install)."""

    global _fonts_loaded, FONT, FONT_MEDIUM, FONT_SEMIBOLD
    if _fonts_loaded:
        return True
    files = sorted(FONTS_DIR.glob("Outfit-*.ttf"))
    registered = False
    if files and sys.platform == "win32":
        import ctypes

        fr_private = 0x10
        registered = all(ctypes.windll.gdi32.AddFontResourceExW(str(path), fr_private, 0) for path in files)
    if files:
        try:
            from matplotlib import font_manager

            for path in files:
                font_manager.fontManager.addfont(str(path))
        except ImportError:
            pass
    if not registered:
        FONT = FONT_MEDIUM = FONT_SEMIBOLD = FALLBACK_FONT
    _fonts_loaded = registered
    return registered


def configure_styles(style: Any) -> None:
    """Register the ttk styles used by labels, checkboxes, tables and scrollbars."""

    p = PALETTE
    style.theme_use("clam")
    style.configure(".", background=p["background"], foreground=p["text"], font=(FONT, 10), bordercolor=p["line"], troughcolor=p["background"])

    style.configure("App.TFrame", background=p["background"])
    style.configure("Card.TFrame", background=p["surface"])
    style.configure("Nav.TFrame", background=p["ink"])

    style.configure("H1.TLabel", background=p["background"], foreground=p["text"], font=(FONT, 21, "bold"))
    style.configure("Sub.TLabel", background=p["background"], foreground=p["muted"], font=(FONT, 10))
    style.configure("CardTitle.TLabel", background=p["surface"], foreground=p["text"], font=(FONT_SEMIBOLD, 12))
    style.configure("CardValue.TLabel", background=p["surface"], foreground=p["text"], font=(FONT_SEMIBOLD, 22))
    style.configure("CardBody.TLabel", background=p["surface"], foreground=p["muted"], font=(FONT, 9))
    style.configure("CardCaption.TLabel", background=p["surface"], foreground=p["muted"], font=(FONT_MEDIUM, 9))
    style.configure("Section.TLabel", background=p["background"], foreground=p["muted"], font=(FONT_SEMIBOLD, 10))

    style.configure("Brand.TLabel", background=p["ink"], foreground=p["text"], font=(FONT, 15, "bold"))
    style.configure("BrandSub.TLabel", background=p["ink"], foreground=p["nav_muted"], font=(FONT, 9))
    style.configure("NavSection.TLabel", background=p["ink"], foreground=p["nav_muted"], font=(FONT_MEDIUM, 9))
    style.configure("NavCard.TFrame", background=p["ink_hover"])
    style.configure("NavCardTitle.TLabel", background=p["ink_hover"], foreground=p["text"], font=(FONT_SEMIBOLD, 10))
    style.configure("NavCardBody.TLabel", background=p["ink_hover"], foreground=p["muted"], font=(FONT, 8))

    for name, fg in [("Pending", p["muted"]), ("Running", p["warning"]), ("Done", p["success"]), ("Failed", p["danger"])]:
        style.configure(f"{name}.Status.TLabel", background=p["surface"], foreground=fg, font=(FONT_MEDIUM, 9))

    style.configure(
        "TCheckbutton",
        background=p["background"],
        foreground=p["muted"],
        font=(FONT, 9),
        indicatorbackground=p["surface_alt"],
        indicatorforeground=p["on_primary"],
    )
    style.map("TCheckbutton", background=[("active", p["background"])], foreground=[("active", p["text"])], indicatorbackground=[("selected", p["primary"])])
    style.configure(
        "Brand.Horizontal.TProgressbar",
        troughcolor=p["surface_alt"],
        background=p["primary"],
        bordercolor=p["surface_alt"],
        lightcolor=p["primary"],
        darkcolor=p["primary"],
        thickness=6,
    )
    for orient in ("Vertical", "Horizontal"):
        style.configure(
            f"{orient}.TScrollbar",
            background=p["line"],
            troughcolor=p["background"],
            bordercolor=p["background"],
            lightcolor=p["line"],
            darkcolor=p["line"],
            arrowcolor=p["muted"],
            gripcount=0,
        )
        style.map(f"{orient}.TScrollbar", background=[("active", p["subtle"])])
    style.configure("Treeview", background=p["surface"], fieldbackground=p["surface"], foreground=p["nav_text"], rowheight=34, font=(FONT, 9), borderwidth=0)
    style.configure("Treeview.Heading", background=p["surface_alt"], foreground=p["muted"], font=(FONT_SEMIBOLD, 9), relief="flat", padding=(8, 8), borderwidth=0)
    style.map("Treeview.Heading", background=[("active", p["line"])])
    style.map("Treeview", background=[("selected", p["primary_soft"])], foreground=[("selected", p["text"])])
    style.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
    style.configure("Table.TFrame", background=p["surface_alt"])
    style.configure("Table.Treeview", background=p["surface_alt"], fieldbackground=p["surface_alt"], foreground=p["nav_text"], rowheight=36, font=(FONT, 9), borderwidth=0)
    style.configure("Table.Treeview.Heading", background=p["surface_alt"], foreground=p["muted"], font=(FONT_SEMIBOLD, 9), relief="flat", padding=(8, 10), borderwidth=0)
    style.map("Table.Treeview.Heading", background=[("active", p["surface_alt"])])
    style.map("Table.Treeview", background=[("selected", p["primary_soft"])], foreground=[("selected", p["text"])])
    style.layout("Table.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])


def apply_chart_style() -> None:
    """Apply the dark product palette and typography to matplotlib charts."""

    import matplotlib as mpl
    from cycler import cycler

    load_fonts()
    mpl.rcParams.update(
        {
            "figure.facecolor": PALETTE["surface"],
            "savefig.facecolor": PALETTE["surface"],
            "axes.facecolor": PALETTE["surface"],
            "axes.edgecolor": PALETTE["line"],
            "axes.labelcolor": PALETTE["muted"],
            "axes.titlecolor": PALETTE["text"],
            "axes.titleweight": "semibold",
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": PALETTE["line"],
            "grid.linestyle": (0, (3, 3)),
            "grid.linewidth": 0.6,
            "axes.prop_cycle": cycler(color=CHART_SERIES),
            "text.color": PALETTE["text"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.family": ["Outfit", "Segoe UI", "DejaVu Sans"],
            "legend.frameon": False,
            "legend.labelcolor": PALETTE["muted"],
        }
    )


def sequential_cmap() -> Any:
    """Colormap from low risk (deep violet) to high risk (pink)."""

    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("facturisk_seq", SEQUENTIAL_COLORS)


def diverging_cmap() -> Any:
    """Colormap for signed values such as correlations."""

    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("facturisk_div", DIVERGING_COLORS)


MODEL_SHORT_NAMES = {
    "HistGradientBoostingClassifier": "HistGB",
    "RandomForestClassifier": "Random Forest",
    "LogisticRegression": "Reg. logística",
    "XGBClassifier": "XGBoost",
    "LGBMClassifier": "LightGBM",
    "CatBoostClassifier": "CatBoost",
    "DummyClassifier": "Línea base",
}


def model_display_name(name: object) -> str:
    """Short, readable model name for cards and charts."""

    return MODEL_SHORT_NAMES.get(str(name), str(name))


# Register fonts on import so every module sees the final family names.
load_fonts()
