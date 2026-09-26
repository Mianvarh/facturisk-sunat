"""Genera el logo, el favicon y los iconos de la interfaz de FactuRisk SUNAT.

Todos los assets se dibujan con Pillow a 4x y se reducen con antialiasing, por lo
que el diseño es reproducible:

    python scripts/generar_assets.py
"""

from __future__ import annotations

import itertools
import math
import sys
from collections.abc import Callable
from pathlib import Path

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


# Line-art logo variants: "arcos" (default), "factura" and "fr".
LOGO_VARIANT = "arcos"
TILE_DARK = "#15121F"
TILE_BORDER = "#2C2740"
STROKE_MAIN = "#B8ADFF"
STROKE_SOFT = "#6F5FD6"


def _stroke_width(size: int, s: int, fine: float = 0.052) -> float:
    """Fine strokes for large logos, thicker ones so small icons stay legible."""

    if size <= 24:
        return s * 0.105
    if size <= 48:
        return s * 0.078
    return s * fine


def _quad(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], steps: int = 40) -> list[tuple[float, float]]:
    return [
        ((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0], (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1])
        for t in (i / steps for i in range(steps + 1))
    ]


def _arc(center: tuple[float, float], radius: float, start_deg: float, end_deg: float, steps: int = 40) -> list[tuple[float, float]]:
    return [
        (center[0] + radius * math.cos(math.radians(a)), center[1] + radius * math.sin(math.radians(a)))
        for a in (start_deg + (end_deg - start_deg) * i / steps for i in range(steps + 1))
    ]


def _path(draw: ImageDraw.ImageDraw, s: int, points: list[tuple[float, float]], width: float, color: str) -> None:
    """Smooth round-capped stroke: stamps dense circles along the polyline (no joint artifacts)."""

    scaled = [(x * s, y * s) for x, y in points]
    r = width / 2
    step = max(width * 0.12, 1.0)
    fill = hex_rgba(color)
    for (x0, y0), (x1, y1) in itertools.pairwise(scaled):
        length = math.hypot(x1 - x0, y1 - y0)
        count = max(int(length / step), 1)
        for index in range(count + 1):
            t = index / count
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def _line_tile(s: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Sober near-black tile with a hairline border and a very faint violet glow."""

    image = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    base = Image.new("RGBA", (s, s), hex_rgba(TILE_DARK))
    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    for step in range(30, 0, -1):
        r = s * 0.95 * step / 30
        glow_draw.ellipse((s - r, s - r, s + r, s + r), fill=(124, 92, 255, min(2 + (30 - step), 34)))
    base = Image.alpha_composite(base, glow)
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.24), fill=255)
    image.paste(base, (0, 0), mask)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.24), outline=hex_rgba(TILE_BORDER), width=max(int(s * 0.012), 2))
    return image, draw


def _dot(draw: ImageDraw.ImageDraw, s: int, x: float, y: float, radius: float) -> None:
    r = s * radius
    draw.ellipse((x * s - r, y * s - r, x * s + r, y * s + r), fill=hex_rgba(PALETTE["danger"]))


def logo_arcos(size: int) -> Image.Image:
    """F drawn with two parallel curved strokes (nested quarter arcs)."""

    s = size * SCALE
    image, draw = _line_tile(s)
    w = _stroke_width(size, s)
    outer = [(0.30, 0.78)] + _arc((0.52, 0.46), 0.22, 180, 270) + [(0.74, 0.24)]
    inner = [(0.44, 0.78)] + _arc((0.58, 0.58), 0.14, 180, 270) + [(0.70, 0.44)]
    _path(draw, s, outer, w, STROKE_MAIN)
    _path(draw, s, inner, w, STROKE_SOFT)
    _dot(draw, s, 0.64, 0.70, 0.052 if size > 48 else 0.075)
    return finish(image, size)


def logo_factura(size: int) -> Image.Image:
    """Outlined invoice with a curved folded corner and a rising trend ending in the risk dot."""

    s = size * SCALE
    image, draw = _line_tile(s)
    w = _stroke_width(size, s)
    left, top, right, bottom, fold, r = 0.27, 0.19, 0.73, 0.81, 0.14, 0.06
    outline = (
        [(right - fold, top), (left + r, top)]
        + _arc((left + r, top + r), r, 270, 180)
        + [(left, bottom - r)]
        + _arc((left + r, bottom - r), r, 180, 90)
        + [(right - r, bottom)]
        + _arc((right - r, bottom - r), r, 90, 0)
        + [(right, top + fold)]
        + _quad((right, top + fold), (right - fold * 0.2, top + fold * 0.2), (right - fold, top))
    )
    _path(draw, s, outline, w, STROKE_MAIN)
    if size > 32:
        _path(draw, s, [(0.36, 0.33), (0.54, 0.33)], w * 0.8, STROKE_SOFT)
        _path(draw, s, [(0.36, 0.42), (0.48, 0.42)], w * 0.8, STROKE_SOFT)
    trend = _quad((0.36, 0.70), (0.47, 0.70), (0.57, 0.55))
    _path(draw, s, trend, w, STROKE_SOFT)
    _dot(draw, s, 0.59, 0.53, 0.05 if size > 48 else 0.075)
    return finish(image, size)


def logo_fr(size: int) -> Image.Image:
    """F and R in flowing strokes: the F middle bar becomes the R bowl, the leg is a swoosh."""

    s = size * SCALE
    image, draw = _line_tile(s)
    w = _stroke_width(size, s)
    f_stroke = [(0.28, 0.79)] + _arc((0.42, 0.37), 0.14, 180, 270) + [(0.72, 0.23)]
    r_stroke = (
        [(0.28, 0.51), (0.52, 0.51)]
        + _arc((0.52, 0.595), 0.085, 270, 450)
        + [(0.44, 0.68)]
    )
    leg = _quad((0.50, 0.68), (0.60, 0.70), (0.70, 0.80))
    _path(draw, s, f_stroke, w, STROKE_MAIN)
    _path(draw, s, r_stroke, w, STROKE_SOFT)
    _path(draw, s, leg, w, STROKE_SOFT)
    _dot(draw, s, 0.70, 0.23, 0.05 if size > 48 else 0.075)
    return finish(image, size)


LOGO_VARIANTS = {"arcos": logo_arcos, "factura": logo_factura, "fr": logo_fr}


def draw_logo(size: int = 512) -> Image.Image:
    return LOGO_VARIANTS[LOGO_VARIANT](size)


def draw_logo_small(size: int) -> Image.Image:
    """Stroke width already adapts to small sizes."""

    return draw_logo(size)


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


def g_gear(d: ImageDraw.ImageDraw, s: float, c: str, w: float) -> None:
    cx = cy = s * 0.5
    teeth = 8
    points = []
    for index in range(teeth * 2):
        angle = math.pi * index / teeth
        radius = s * (0.40 if index % 2 == 0 else 0.31)
        for offset in (-0.16, 0.16):
            a = angle + offset * math.pi / teeth
            points.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    d.polygon(points, outline=c, width=int(w))
    r = s * 0.12
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=c, width=int(w))


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
    "config": g_gear,
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
    draw_logo(64).save(ASSETS_DIR / "logo_64.png")
    draw_logo_small(128).save(ASSETS_DIR / "logo_small.png")
    # Windows icon: the simplified FR mark for small sizes, the full logo from 48 px.
    frames = [draw_logo_small(size) for size in (16, 24, 32)] + [draw_logo(size) for size in (48, 64, 128, 256)]
    frames[-1].save(ASSETS_DIR / "facturisk.ico", format="ICO", sizes=[frame.size for frame in frames], append_images=frames[:-1])

    for name, glyph in ICONS.items():
        draw_icon(glyph, PALETTE["nav_icon"]).save(ASSETS_DIR / f"nav_{name}.png")
        glyph_color = "#1A1405" if name in TILES_WITH_DARK_GLYPH else "#FFFFFF"
        draw_icon(glyph, glyph_color, tile=TILE_COLORS[name]).save(ASSETS_DIR / f"tile_{name}.png")
    print(f"Assets generados en {ASSETS_DIR}")


if __name__ == "__main__":
    main()
