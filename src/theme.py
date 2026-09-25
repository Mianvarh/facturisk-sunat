"""Visual identity of FactuRisk SUNAT: palette, ttk styles and chart style."""

from __future__ import annotations

from typing import Any

PALETTE = {
    # Surfaces
    "ink": "#13232E",
    "ink_hover": "#1C3240",
    "ink_active": "#24404F",
    "background": "#F3F6F5",
    "surface": "#FFFFFF",
    "line": "#DCE3E1",
    # Text
    "text": "#15232B",
    "muted": "#5B6B73",
    "nav_text": "#E6EFEC",
    "nav_muted": "#8FA5A0",
    "nav_icon": "#CFE3DD",
    # Brand
    "primary": "#0E7C66",
    "primary_hover": "#0A6653",
    "primary_soft": "#E2F2EE",
    "accent": "#F2A33A",
    "accent_dark": "#B26F12",
    "accent_soft": "#FDF1DE",
    "danger": "#D64545",
    "danger_soft": "#FBE5E5",
    # Console
    "console_bg": "#0E1A22",
    "console_text": "#D7E4E0",
}

RISK_COLORS = {"Bajo": "#2E9E6B", "Medio": "#E3A21A", "Alto": "#D64545"}
CHART_SERIES = ["#0E7C66", "#F2A33A", "#3B7EA1", "#D64545", "#7A6CC4", "#5B6B73"]

FONT = "Segoe UI"
FONT_SEMIBOLD = "Segoe UI Semibold"
FONT_MONO = "Consolas"


def configure_styles(style: Any) -> None:
    """Register every ttk style used by the desktop interface."""

    p = PALETTE
    style.theme_use("clam")
    style.configure(".", background=p["background"], foreground=p["text"], font=(FONT, 10))

    style.configure("App.TFrame", background=p["background"])
    style.configure("Card.TFrame", background=p["surface"])
    style.configure("Nav.TFrame", background=p["ink"])

    style.configure("H1.TLabel", background=p["background"], foreground=p["text"], font=(FONT_SEMIBOLD, 20))
    style.configure("Sub.TLabel", background=p["background"], foreground=p["muted"], font=(FONT, 10))
    style.configure("CardTitle.TLabel", background=p["surface"], foreground=p["text"], font=(FONT_SEMIBOLD, 12))
    style.configure("CardValue.TLabel", background=p["surface"], foreground=p["text"], font=(FONT_SEMIBOLD, 17))
    style.configure("CardBody.TLabel", background=p["surface"], foreground=p["muted"], font=(FONT, 9))
    style.configure("CardCaption.TLabel", background=p["surface"], foreground=p["muted"], font=(FONT, 8))
    style.configure("Section.TLabel", background=p["background"], foreground=p["muted"], font=(FONT_SEMIBOLD, 9))

    style.configure("Brand.TLabel", background=p["ink"], foreground="#FFFFFF", font=(FONT_SEMIBOLD, 15))
    style.configure("BrandSub.TLabel", background=p["ink"], foreground=p["nav_muted"], font=(FONT, 9))
    style.configure("NavSection.TLabel", background=p["ink"], foreground=p["nav_muted"], font=(FONT_SEMIBOLD, 8))

    for name, fg in [("Pending", p["muted"]), ("Running", p["accent_dark"]), ("Done", p["primary"]), ("Failed", p["danger"])]:
        style.configure(f"{name}.Status.TLabel", background=p["surface"], foreground=fg, font=(FONT_SEMIBOLD, 9))

    button_base = {"borderwidth": 0, "focusthickness": 0, "font": (FONT_SEMIBOLD, 10), "padding": (14, 8)}
    style.configure("Primary.TButton", background=p["primary"], foreground="#FFFFFF", **button_base)
    style.map("Primary.TButton", background=[("disabled", "#A8C7BF"), ("active", p["primary_hover"])], foreground=[("disabled", "#F3F6F5")])
    style.configure("Secondary.TButton", background=p["primary_soft"], foreground=p["primary"], **button_base)
    style.map("Secondary.TButton", background=[("disabled", "#EEF2F1"), ("active", "#CDE8E1")], foreground=[("disabled", "#9AAEA9")])
    style.configure("Danger.TButton", background=p["danger_soft"], foreground=p["danger"], **button_base)
    style.map("Danger.TButton", background=[("disabled", "#F5EEEE"), ("active", "#F6CFCF")], foreground=[("disabled", "#C9A3A3")])
    style.configure("Ghost.TButton", background=p["surface"], foreground=p["muted"], borderwidth=0, focusthickness=0, font=(FONT, 9), padding=(8, 4))
    style.map("Ghost.TButton", background=[("active", p["background"])])

    style.configure("TCheckbutton", background=p["background"], foreground=p["text"], font=(FONT, 9))
    style.map("TCheckbutton", background=[("active", p["background"])])
    style.configure(
        "Brand.Horizontal.TProgressbar",
        troughcolor=p["primary_soft"],
        background=p["primary"],
        bordercolor=p["primary_soft"],
        lightcolor=p["primary"],
        darkcolor=p["primary"],
        thickness=8,
    )
    style.configure("Treeview", background=p["surface"], fieldbackground=p["surface"], foreground=p["text"], rowheight=26, font=(FONT, 9), borderwidth=0)
    style.configure("Treeview.Heading", background=p["background"], foreground=p["muted"], font=(FONT_SEMIBOLD, 9), relief="flat", padding=(6, 6))
    style.map("Treeview", background=[("selected", p["primary_soft"])], foreground=[("selected", p["text"])])
    style.configure("Vertical.TScrollbar", background=p["line"], troughcolor=p["background"], bordercolor=p["background"], arrowcolor=p["muted"])


def apply_chart_style() -> None:
    """Apply the product palette to matplotlib charts."""

    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
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
            "grid.color": "#EEF2F1",
            "axes.prop_cycle": cycler(color=CHART_SERIES),
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "legend.frameon": False,
        }
    )


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
