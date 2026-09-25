"""Genera el logo, el favicon y los iconos de la interfaz de FactuRisk SUNAT.

Todos los assets se dibujan con Pillow a 4x y se reducen con antialiasing, por lo
que el diseño es reproducible:

    python scripts/generar_assets.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from theme import PALETTE, TILE_COLORS, TILES_WITH_DARK_GLYPH  # noqa: E402

ASSETS_DIR = PROJECT_ROOT / "assets"
SCALE = 4
ICON_SIZE = 64

Glyph = Callable[[ImageDraw.ImageDraw, float, str, float], None]


def hex_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha


def canvas(size: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (size * SCALE, size * SCALE), (0, 0, 0, 0))
    return image, ImageDraw.Draw(image)


def finish(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)


# --------------------------------------------------------------------------- logo


def draw_logo(size: int = 512) -> Image.Image:
    """Invoice sheet with a radar pulse: the product detects risky invoices."""

    image, draw = canvas(size)
    s = size * SCALE
    # Diagonal gradient tile: primary blue to violet accent.
    start, end = hex_rgba("#6D4AFF"), hex_rgba(PALETTE["primary"])
    gradient = Image.new("RGBA", (s, s))
    pixels = gradient.load()
    for y in range(0, s):
        for x in range(0, s, 4):
            t = (x + y) / (2 * s)
            color = tuple(int(start[i] + (end[i] - start[i]) * t) for i in range(4))
            for dx in range(4):
                if x + dx < s:
                    pixels[x + dx, y] = color
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.24), fill=255)
    image.paste(gradient, (0, 0), mask)

    # Invoice sheet with folded corner.
    left, top, right, bottom = s * 0.22, s * 0.16, s * 0.68, s * 0.80
    fold = s * 0.13
    sheet = [(left, top), (right - fold, top), (right, top + fold), (right, bottom), (left, bottom)]
    draw.polygon(sheet, fill=hex_rgba("#FFFFFF"))
    draw.polygon([(right - fold, top), (right - fold, top + fold), (right, top + fold)], fill=hex_rgba("#C7D2FE"))

    # Text lines of the invoice.
    line_color = hex_rgba(PALETTE["accent"])
    widths = [0.30, 0.24, 0.30, 0.16]
    for index, width in enumerate(widths):
        y = top + s * (0.14 + index * 0.095)
        draw.rounded_rectangle((left + s * 0.06, y, left + s * (0.06 + width), y + s * 0.035), radius=int(s * 0.02), fill=line_color)

    # Radar pulse in the corner: concentric arcs around an alert dot.
    cx, cy = s * 0.72, s * 0.72
    accent = hex_rgba(PALETTE["danger"])
    draw.ellipse((cx - s * 0.20, cy - s * 0.20, cx + s * 0.20, cy + s * 0.20), fill=hex_rgba(PALETTE["ink"]))
    for radius, width in [(0.155, 0.028), (0.100, 0.028)]:
        r = s * radius
        draw.arc((cx - r, cy - r, cx + r, cy + r), start=200, end=340, fill=accent, width=int(s * width))
        draw.arc((cx - r, cy - r, cx + r, cy + r), start=20, end=160, fill=accent, width=int(s * width))
    dot = s * 0.045
    draw.ellipse((cx - dot, cy - dot, cx + dot, cy + dot), fill=accent)
    return finish(image, size)


# ------------------------------------------------------------------------- glyphs
# Each glyph draws on a canvas of side `s` with stroke color `c` and width `w`.


def g_play(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    d.ellipse((s * 0.14, s * 0.14, s * 0.86, s * 0.86), outline=c, width=int(w))
    d.polygon([(s * 0.42, s * 0.34), (s * 0.42, s * 0.66), (s * 0.68, s * 0.50)], fill=c)


def g_search(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    d.ellipse((s * 0.16, s * 0.16, s * 0.62, s * 0.62), outline=c, width=int(w))
    d.line((s * 0.56, s * 0.56, s * 0.82, s * 0.82), fill=c, width=int(w * 1.3))


def g_nodes(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    points = [(0.24, 0.26), (0.76, 0.26), (0.50, 0.76)]
    for a in range(3):
        for b in range(a + 1, 3):
            d.line((s * points[a][0], s * points[a][1], s * points[b][0], s * points[b][1]), fill=c, width=int(w * 0.8))
    for x, y in points:
        r = s * 0.12
        d.ellipse((s * x - r, s * y - r, s * x + r, s * y + r), fill=c)


def g_download(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    d.line((s * 0.50, s * 0.14, s * 0.50, s * 0.60), fill=c, width=int(w))
    d.line((s * 0.32, s * 0.44, s * 0.50, s * 0.62, s * 0.68, s * 0.44), fill=c, width=int(w), joint="curve")
    d.line((s * 0.18, s * 0.66, s * 0.18, s * 0.84, s * 0.82, s * 0.84, s * 0.82, s * 0.66), fill=c, width=int(w), joint="curve")


def g_database(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    left, right = s * 0.22, s * 0.78
    for top in (0.14, 0.40, 0.62):
        d.arc((left, s * (top + 0.02), right, s * (top + 0.20)), start=0, end=180, fill=c, width=int(w))
    d.ellipse((left, s * 0.14, right, s * 0.32), outline=c, width=int(w))
    d.line((left, s * 0.23, left, s * 0.73), fill=c, width=int(w))
    d.line((right, s * 0.23, right, s * 0.73), fill=c, width=int(w))


def g_sliders(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    for y, knob in [(0.26, 0.64), (0.50, 0.34), (0.74, 0.56)]:
        d.line((s * 0.16, s * y, s * 0.84, s * y), fill=c, width=int(w * 0.8))
        r = s * 0.09
        d.ellipse((s * knob - r, s * y - r, s * knob + r, s * y + r), fill=c)


def g_network(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    layers = [[0.30, 0.70], [0.20, 0.50, 0.80], [0.50]]
    xs = [0.18, 0.50, 0.82]
    for li in range(2):
        for ya in layers[li]:
            for yb in layers[li + 1]:
                d.line((s * xs[li], s * ya, s * xs[li + 1], s * yb), fill=c, width=int(w * 0.55))
    for li, ys in enumerate(layers):
        for y in ys:
            r = s * 0.085
            d.ellipse((s * xs[li] - r, s * y - r, s * xs[li] + r, s * y + r), fill=c)


def g_radar(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    for r in (0.36, 0.22):
        d.arc((s * (0.5 - r), s * (0.5 - r), s * (0.5 + r), s * (0.5 + r)), start=200, end=340, fill=c, width=int(w))
        d.arc((s * (0.5 - r), s * (0.5 - r), s * (0.5 + r), s * (0.5 + r)), start=20, end=160, fill=c, width=int(w))
    r = s * 0.08
    d.ellipse((s * 0.5 - r, s * 0.5 - r, s * 0.5 + r, s * 0.5 + r), fill=c)


def g_document(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    d.line(
        (s * 0.62, s * 0.14, s * 0.24, s * 0.14, s * 0.24, s * 0.86, s * 0.76, s * 0.86, s * 0.76, s * 0.28, s * 0.62, s * 0.14, s * 0.62, s * 0.28, s * 0.76, s * 0.28),
        fill=c,
        width=int(w),
        joint="curve",
    )
    for y in (0.46, 0.60, 0.74):
        d.line((s * 0.36, s * y, s * 0.64, s * y), fill=c, width=int(w * 0.8))


def g_dashboard(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    for x0, y0, x1, y1 in [(0.14, 0.14, 0.46, 0.54), (0.54, 0.14, 0.86, 0.38), (0.14, 0.62, 0.46, 0.86), (0.54, 0.46, 0.86, 0.86)]:
        d.rounded_rectangle((s * x0, s * y0, s * x1, s * y1), radius=int(s * 0.06), outline=c, width=int(w))


def g_cube(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    top = [(0.50, 0.14), (0.84, 0.32), (0.50, 0.50), (0.16, 0.32)]
    d.polygon([(s * x, s * y) for x, y in top], outline=c, width=int(w))
    d.line((s * 0.16, s * 0.32, s * 0.16, s * 0.68, s * 0.50, s * 0.86, s * 0.84, s * 0.68, s * 0.84, s * 0.32), fill=c, width=int(w), joint="curve")
    d.line((s * 0.50, s * 0.50, s * 0.50, s * 0.86), fill=c, width=int(w))


def g_target(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    for r in (0.36, 0.22):
        d.ellipse((s * (0.5 - r), s * (0.5 - r), s * (0.5 + r), s * (0.5 + r)), outline=c, width=int(w))
    r = s * 0.07
    d.ellipse((s * 0.5 - r, s * 0.5 - r, s * 0.5 + r, s * 0.5 + r), fill=c)


def g_gauge(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    d.arc((s * 0.14, s * 0.22, s * 0.86, s * 0.94), start=180, end=360, fill=c, width=int(w))
    angle = math.radians(-50)
    d.line((s * 0.50, s * 0.58, s * (0.50 + 0.26 * math.cos(angle)), s * (0.58 + 0.26 * math.sin(angle))), fill=c, width=int(w))
    r = s * 0.06
    d.ellipse((s * 0.5 - r, s * 0.58 - r, s * 0.5 + r, s * 0.58 + r), fill=c)


def g_clock(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    d.ellipse((s * 0.14, s * 0.14, s * 0.86, s * 0.86), outline=c, width=int(w))
    d.line((s * 0.50, s * 0.30, s * 0.50, s * 0.50, s * 0.66, s * 0.60), fill=c, width=int(w), joint="curve")


def g_alert(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    d.line((s * 0.50, s * 0.14, s * 0.88, s * 0.82, s * 0.12, s * 0.82, s * 0.50, s * 0.14), fill=c, width=int(w), joint="curve")
    d.line((s * 0.50, s * 0.38, s * 0.50, s * 0.60), fill=c, width=int(w))
    r = s * 0.045
    d.ellipse((s * 0.5 - r, s * 0.70 - r, s * 0.5 + r, s * 0.70 + r), fill=c)


def g_coins(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    for y in (0.20, 0.42, 0.64):
        d.ellipse((s * 0.20, s * y, s * 0.80, s * (y + 0.18)), outline=c, width=int(w))


ICONS: dict[str, Glyph] = {
    "todo": g_play,
    "inspect": g_search,
    "distribuido": g_nodes,
    "scraping": g_download,
    "mongodb": g_database,
    "preparar": g_sliders,
    "entrenar": g_network,
    "predecir": g_radar,
    "documentar": g_document,
    "dashboard": g_dashboard,
    "modelo": g_cube,
    "recall": g_target,
    "f1": g_gauge,
    "pendientes": g_clock,
    "riesgo_alto": g_alert,
    "importe": g_coins,
}


def draw_icon(glyph: Glyph, color: str, size: int = ICON_SIZE, tile: str | None = None) -> Image.Image:
    """Draw a line icon, optionally centered on a rounded solid tile with a soft top highlight."""

    image, draw = canvas(size)
    s = size * SCALE
    if tile:
        draw.rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.30), fill=hex_rgba(tile))
        highlight = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        ImageDraw.Draw(highlight).rounded_rectangle((0, 0, s - 1, s * 0.55), radius=int(s * 0.30), fill=(255, 255, 255, 28))
        mask = Image.new("L", (s, s), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.30), fill=255)
        image.paste(Image.alpha_composite(image, highlight), (0, 0), mask)
        inner, inner_draw = canvas(size)
        glyph(inner_draw, s, color, s * 0.10)
        inner = inner.resize((int(s * 0.58), int(s * 0.58)), Image.Resampling.LANCZOS)
        offset = int(s * 0.21)
        image.alpha_composite(inner, (offset, offset))
    else:
        glyph(draw, s, color, s * 0.075)
    return finish(image, size)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    logo = draw_logo(512)
    logo.save(ASSETS_DIR / "logo.png")
    logo.save(ASSETS_DIR / "facturisk.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    draw_logo(64).save(ASSETS_DIR / "logo_64.png")

    for name, glyph in ICONS.items():
        draw_icon(glyph, PALETTE["nav_icon"]).save(ASSETS_DIR / f"nav_{name}.png")
        glyph_color = "#1A1405" if name in TILES_WITH_DARK_GLYPH else "#FFFFFF"
        draw_icon(glyph, glyph_color, tile=TILE_COLORS[name]).save(ASSETS_DIR / f"tile_{name}.png")
    print(f"Assets generados en {ASSETS_DIR}")


if __name__ == "__main__":
    main()
