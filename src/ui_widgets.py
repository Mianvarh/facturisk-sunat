"""Reusable Tkinter widgets: rounded surfaces, pill buttons and responsive containers.

Tk has no rounded corners and the Windows canvas draws curves without
antialiasing, so curved parts are rendered with Pillow at 3x and scaled down.
Cards use a 9-slice: four antialiased corner images plus flat rectangles.
"""

from __future__ import annotations

import tkinter as tk
import tkinter.font
from functools import lru_cache
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageTk

from theme import FONT, FONT_MEDIUM, FONT_SEMIBOLD, PALETTE

SUPERSAMPLE = 3
NAV_WIDTH = 252
NAV_COLLAPSED_WIDTH = 66


def _rgba(color: str) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), 255


def rounded_image(width: int, height: int, radius: int, fill: str, background: str, border: str | None = None) -> Image.Image:
    """Antialiased rounded rectangle composited over the parent background."""

    width, height = max(width, 2), max(height, 2)
    radius = min(radius, width // 2, height // 2)
    s = SUPERSAMPLE
    image = Image.new("RGBA", (width * s, height * s), _rgba(background))
    draw = ImageDraw.Draw(image)
    if border:
        draw.rounded_rectangle((0, 0, width * s - 1, height * s - 1), radius=radius * s, fill=_rgba(border))
        draw.rounded_rectangle((s, s, width * s - 1 - s, height * s - 1 - s), radius=max(radius - 1, 0) * s, fill=_rgba(fill))
    else:
        draw.rounded_rectangle((0, 0, width * s - 1, height * s - 1), radius=radius * s, fill=_rgba(fill))
    return image.resize((width, height), Image.Resampling.LANCZOS)


@lru_cache(maxsize=256)
def _corner_images(radius: int, fill: str, border: str, background: str) -> tuple[Image.Image, ...]:
    full = rounded_image(radius * 2 + 2, radius * 2 + 2, radius, fill, background, border)
    size = radius
    width = full.width
    return (
        full.crop((0, 0, size, size)),
        full.crop((width - size, 0, width, size)),
        full.crop((0, width - size, size, width)),
        full.crop((width - size, width - size, width, width)),
    )


class ScrollableFrame(ttk.Frame):
    """Vertical scroll container whose body always matches the visible width."""

    def __init__(self, parent: tk.Misc, style: str = "App.TFrame") -> None:
        super().__init__(parent, style=style)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=PALETTE["background"])
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.body = ttk.Frame(self.canvas, style=style)
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    def _on_body_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._toggle_scrollbar()

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)
        self._toggle_scrollbar()

    def _toggle_scrollbar(self) -> None:
        if self.body.winfo_reqheight() > self.canvas.winfo_height():
            self.scrollbar.grid()
        else:
            self.scrollbar.grid_remove()
            self.canvas.yview_moveto(0)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if not self.winfo_ismapped() or not self.scrollbar.winfo_ismapped():
            return
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None and widget is not self:
            if isinstance(widget, (tk.Text, ttk.Treeview)):
                return
            widget = widget.master
        if widget is self:
            self.canvas.yview_scroll(int(-event.delta / 120), "units")


class ResponsiveGrid(ttk.Frame):
    """Lays out its items in as many equal columns as the width allows."""

    def __init__(self, parent: tk.Misc, min_item_width: int, max_columns: int, gap: int = 14, style: str = "App.TFrame") -> None:
        super().__init__(parent, style=style)
        self.min_item_width = min_item_width
        self.max_columns = max_columns
        self.gap = gap
        self.items: list[tk.Widget] = []
        self._columns = 0
        self.bind("<Configure>", self._on_configure)

    def add(self, widget: tk.Widget) -> None:
        self.items.append(widget)
        self._columns = 0
        self._layout(max(self.winfo_width(), 1))

    def _on_configure(self, event: tk.Event) -> None:
        self._layout(event.width)

    def _layout(self, width: int) -> None:
        if width <= 1:
            width = self.min_item_width * self.max_columns
        columns = max(1, min(self.max_columns, (width + self.gap) // (self.min_item_width + self.gap), len(self.items) or 1))
        if columns == self._columns:
            return
        for index in range(max(self._columns, columns, self.max_columns)):
            self.columnconfigure(index, weight=0, uniform="")
        for index in range(columns):
            self.columnconfigure(index, weight=1, uniform="cell")
        for index, widget in enumerate(self.items):
            row, column = divmod(index, columns)
            widget.grid(
                row=row,
                column=column,
                sticky="nsew",
                padx=(0 if column == 0 else self.gap // 2, 0 if column == columns - 1 else self.gap // 2),
                pady=(0, self.gap),
            )
        self._columns = columns


def load_icon(path: Path, size: int) -> ImageTk.PhotoImage | None:
    """Load a PNG asset scaled with antialiasing; None when missing."""

    if not path.exists():
        return None
    image = Image.open(path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


class RoundedCard(tk.Canvas):
    """Rounded surface that hosts a ttk frame (`.inner`) and grows to fit it."""

    def __init__(
        self,
        parent: tk.Misc,
        padding: tuple[int, int] | tuple[int, int, int, int] = (18, 16),
        radius: int = 20,
        fill: str | None = None,
        border: str | None = None,
        background: str | None = None,
        inner_style: str = "Card.TFrame",
    ) -> None:
        self.bg_color = background or PALETTE["background"]
        super().__init__(parent, highlightthickness=0, borderwidth=0, background=self.bg_color, height=40, width=40)
        self.radius = radius
        self.fill = fill or PALETTE["surface"]
        self.border = border or PALETTE["line"]
        self._corners: list[ImageTk.PhotoImage] = []
        self._size = (0, 0)
        # The inner frame is a plain rectangle: keep it inside the corner curves.
        self.inset = max(1, round(radius * 0.3))
        padding = tuple(max(0, value - self.inset + 1) for value in padding)
        self.inner = ttk.Frame(self, style=inner_style, padding=padding)
        self._window = self.create_window(self.inset, self.inset, window=self.inner, anchor="nw")
        tag = f"RoundedCardInner{id(self)}"
        self.inner.bindtags((tag, *self.inner.bindtags()))
        self.inner.bind_class(tag, "<Configure>", self._on_inner_configure)
        self.bind_class(f"RoundedCard{id(self)}", "<Configure>", self._on_configure)
        self.bindtags((f"RoundedCard{id(self)}", *self.bindtags()))

    def _on_inner_configure(self, _event: tk.Event) -> None:
        wanted = self.inner.winfo_reqheight() + self.inset * 2
        if int(self.cget("height")) != wanted:
            self.configure(height=wanted)

    def _on_configure(self, event: tk.Event) -> None:
        self.itemconfigure(self._window, width=max(event.width - self.inset * 2, 1))
        self._size = (event.width, event.height)
        self._draw()

    def set_border(self, color: str) -> None:
        self.border = color
        self._draw()

    def _draw(self) -> None:
        width, height = self._size
        if width < 4 or height < 4:
            return
        self.delete("surface")
        r = min(self.radius, width // 2, height // 2)
        fill, border = self.fill, self.border
        self.create_rectangle(r, 0, width - r, height, fill=fill, outline="", tags="surface")
        self.create_rectangle(0, r, width, height - r, fill=fill, outline="", tags="surface")
        self.create_line(r, 0, width - r, 0, fill=border, tags="surface")
        self.create_line(r, height - 1, width - r, height - 1, fill=border, tags="surface")
        self.create_line(0, r, 0, height - r, fill=border, tags="surface")
        self.create_line(width - 1, r, width - 1, height - r, fill=border, tags="surface")
        corners = _corner_images(r, fill, border, self.bg_color)
        self._corners = [ImageTk.PhotoImage(image) for image in corners]
        for image, (x, y) in zip(self._corners, [(0, 0), (width - r, 0), (0, height - r), (width - r, height - r)]):
            self.create_image(x, y, image=image, anchor="nw", tags="surface")
        self.tag_lower("surface")


def card(parent: tk.Misc, padding: tuple[int, int] = (18, 16), background: str | None = None, radius: int = 20) -> RoundedCard:
    """Rounded card with the standard surface colors."""

    return RoundedCard(parent, padding=padding, background=background, radius=radius)


BUTTON_VARIANTS = {
    "primary": ("primary", "primary_hover", "on_primary"),
    "secondary": ("surface_alt", "line", "text"),
    "danger": ("danger_soft", "#3D1E2A", "danger"),
    "ghost": (None, "surface_alt", "muted"),
}


class PillButton(tk.Canvas):
    """Fully rounded button with hover and disabled states (ttk.Button-like API)."""

    def __init__(
        self,
        parent: tk.Misc,
        text: str,
        command: Callable[[], None] | None = None,
        variant: str = "primary",
        background: str | None = None,
        height: int = 38,
        padx: int = 20,
        font: tuple[Any, ...] | None = None,
        state: str = "normal",
    ) -> None:
        self.parent_bg = background or PALETTE["background"]
        self.font = font or (FONT_SEMIBOLD, 10)
        self.text = text
        self.command = command
        self.variant = variant
        self.state = state
        self.hover = False
        self.padx = padx
        width = tk.font.Font(font=self.font).measure(text) + padx * 2
        super().__init__(parent, width=width, height=height, highlightthickness=0, borderwidth=0, background=self.parent_bg, cursor="hand2")
        self._image: ImageTk.PhotoImage | None = None
        self.bind("<Configure>", lambda _e: self._draw())
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        self.bind("<ButtonRelease-1>", self._on_click)

    def _set_hover(self, hover: bool) -> None:
        self.hover = hover
        self._draw()

    def _on_click(self, _event: tk.Event) -> None:
        if self.state != "disabled" and self.command:
            self.command()

    def configure(self, cnf: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        redraw = False
        for key in ("text", "state", "command", "variant"):
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))
                redraw = True
        if "text" in (cnf or {}):
            self.text = cnf.pop("text")
            redraw = True
        result = super().configure(cnf, **kwargs) if (cnf or kwargs) else None
        if redraw:
            self.configure_width()
            self._draw()
        return result

    config = configure

    def configure_width(self) -> None:
        tk.Canvas.configure(self, width=tk.font.Font(font=self.font).measure(self.text) + self.padx * 2)

    def _draw(self) -> None:
        width, height = self.winfo_width(), self.winfo_height()
        if width < 4:
            width, height = int(self.cget("width")), int(self.cget("height"))
        base_key, hover_key, fg_key = BUTTON_VARIANTS[self.variant]
        if self.state == "disabled":
            fill, fg = PALETTE["surface_alt"], PALETTE["subtle"]
            if self.variant == "ghost":
                fill = None
            self.configure_cursor("arrow")
        else:
            key = hover_key if self.hover else base_key
            fill = PALETTE.get(key, key) if key else None
            fg = PALETTE.get(fg_key, fg_key)
            self.configure_cursor("hand2")
        self.delete("all")
        if fill:
            self._image = ImageTk.PhotoImage(rounded_image(width, height, height // 2, fill, self.parent_bg))
            self.create_image(0, 0, image=self._image, anchor="nw")
        self.create_text(width / 2, height / 2, text=self.text, fill=fg, font=self.font)

    def configure_cursor(self, cursor: str) -> None:
        tk.Canvas.configure(self, cursor=cursor)


class NavItem(tk.Canvas):
    """Sidebar entry drawn as a rounded pill when hovered or selected."""

    def __init__(self, parent: tk.Misc, text: str, icon: ImageTk.PhotoImage | None, command: Callable[[], None]) -> None:
        super().__init__(parent, width=NAV_WIDTH, height=42, highlightthickness=0, borderwidth=0, background=PALETTE["ink"], cursor="hand2")
        self.text = text
        self.icon = icon
        self.command = command
        self.selected = False
        self.hover = False
        self.collapsed = False
        self._image: ImageTk.PhotoImage | None = None
        self.bind("<Configure>", lambda _e: self._draw())
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        self.bind("<ButtonRelease-1>", lambda _e: self.command())

    def _set_hover(self, hover: bool) -> None:
        self.hover = hover
        self._draw()

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self._draw()

    def set_collapsed(self, collapsed: bool) -> None:
        self.collapsed = collapsed
        self.configure(width=NAV_COLLAPSED_WIDTH if collapsed else NAV_WIDTH)
        self._draw()

    def _draw(self) -> None:
        width, height = self.winfo_width(), self.winfo_height()
        if width < 10:
            return
        self.delete("all")
        margin = 10
        if self.selected or self.hover:
            fill = PALETTE["ink_active"] if self.selected else PALETTE["ink_hover"]
            self._image = ImageTk.PhotoImage(rounded_image(width - margin * 2, height - 6, 12, fill, PALETTE["ink"]))
            self.create_image(margin, 3, image=self._image, anchor="nw")
        icon_x = width / 2 if self.collapsed else margin + 22
        if self.icon:
            self.create_image(icon_x, height / 2, image=self.icon)
        if not self.collapsed:
            self.create_text(
                margin + 44,
                height / 2,
                text=self.text,
                anchor="w",
                fill=PALETTE["text"] if self.selected else PALETTE["nav_text"],
                font=(FONT_MEDIUM if self.selected else FONT, 10),
            )


class SegmentedControl(tk.Canvas):
    """Pill-shaped group of options where one is selected, like tab filters."""

    def __init__(self, parent: tk.Misc, options: list[str], command: Callable[[str], None], background: str | None = None) -> None:
        self.parent_bg = background or PALETTE["surface"]
        self.options = options
        self.selected = options[0]
        self.command = command
        self.font = (FONT_MEDIUM, 9)
        measure = tk.font.Font(font=self.font).measure
        self.widths = [measure(option) + 32 for option in options]
        super().__init__(parent, width=sum(self.widths) + 8, height=38, highlightthickness=0, borderwidth=0, background=self.parent_bg, cursor="hand2")
        self._images: list[ImageTk.PhotoImage] = []
        self.bind("<ButtonRelease-1>", self._on_click)
        self.after_idle(self._draw)

    def _on_click(self, event: tk.Event) -> None:
        x = 4
        for option, width in zip(self.options, self.widths):
            if x <= event.x < x + width:
                if option != self.selected:
                    self.selected = option
                    self._draw()
                    self.command(option)
                return
            x += width

    def _draw(self) -> None:
        self.delete("all")
        self._images = [ImageTk.PhotoImage(rounded_image(sum(self.widths) + 8, 38, 19, PALETTE["surface_alt"], self.parent_bg))]
        self.create_image(0, 0, image=self._images[0], anchor="nw")
        x = 4
        for option, width in zip(self.options, self.widths):
            selected = option == self.selected
            if selected:
                pill = ImageTk.PhotoImage(rounded_image(width, 30, 15, PALETTE["primary"], PALETTE["surface_alt"]))
                self._images.append(pill)
                self.create_image(x, 4, image=pill, anchor="nw")
            self.create_text(x + width / 2, 19, text=option, font=self.font, fill=PALETTE["on_primary"] if selected else PALETTE["muted"])
            x += width


class DonutRing(tk.Canvas):
    """Antialiased ring gauge with a large centered value."""

    def __init__(self, parent: tk.Misc, fraction: float, label: str, color: str, size: int = 132) -> None:
        super().__init__(parent, width=size, height=size, highlightthickness=0, background=PALETTE["surface"])
        s = SUPERSAMPLE
        image = Image.new("RGBA", (size * s, size * s), _rgba(PALETTE["surface"]))
        draw = ImageDraw.Draw(image)
        pad, width = 8 * s, 11 * s
        box = (pad, pad, size * s - pad, size * s - pad)
        draw.ellipse(box, outline=_rgba(PALETTE["surface_alt"]), width=width)
        fraction = max(0.0, min(1.0, fraction))
        if fraction > 0:
            draw.arc(box, start=-90, end=-90 + 360 * fraction, fill=_rgba(color), width=width)
        self._image = ImageTk.PhotoImage(image.resize((size, size), Image.Resampling.LANCZOS))
        self.create_image(0, 0, image=self._image, anchor="nw")
        self.create_text(size / 2, size / 2 - 7, text=f"{fraction * 100:.1f}%", fill=PALETTE["text"], font=(FONT_SEMIBOLD, 16))
        self.create_text(size / 2, size / 2 + 16, text=label, fill=PALETTE["muted"], font=(FONT, 8))


class SegmentedMeter(tk.Canvas):
    """Row of rounded bars filled up to a fraction, like a battery meter."""

    def __init__(self, parent: tk.Misc, fraction: float, color: str, segments: int = 26, height: int = 24) -> None:
        super().__init__(parent, height=height, highlightthickness=0, background=PALETTE["surface"])
        self.fraction = max(0.0, min(1.0, fraction))
        self.color = color
        self.segments = segments
        self._image: ImageTk.PhotoImage | None = None
        self.bind("<Configure>", self._draw)

    def _draw(self, event: tk.Event) -> None:
        s = SUPERSAMPLE
        width, height = max(event.width, 10), event.height
        image = Image.new("RGBA", (width * s, height * s), _rgba(PALETTE["surface"]))
        draw = ImageDraw.Draw(image)
        gap = 4 * s
        bar = (width * s - gap * (self.segments - 1)) / self.segments
        filled = round(self.fraction * self.segments)
        for index in range(self.segments):
            x0 = index * (bar + gap)
            color = self.color if index < filled else PALETTE["surface_alt"]
            draw.rounded_rectangle((x0, 0, x0 + bar, height * s - 1), radius=int(bar / 2), fill=_rgba(color))
        self._image = ImageTk.PhotoImage(image.resize((width, height), Image.Resampling.LANCZOS))
        self.delete("all")
        self.create_image(0, 0, image=self._image, anchor="nw")


class StatusPill(tk.Canvas):
    """Rounded status label with a colored dot."""

    def __init__(self, parent: tk.Misc, text: str, color: str, background: str | None = None, fill: str | None = None) -> None:
        self.parent_bg = background or PALETTE["background"]
        self.fill = fill or PALETTE["surface_alt"]
        self.font = (FONT_MEDIUM, 9)
        super().__init__(parent, height=28, width=10, highlightthickness=0, borderwidth=0, background=self.parent_bg)
        self._image: ImageTk.PhotoImage | None = None
        self.set(text, color)

    def set(self, text: str, color: str) -> None:
        width = tk.font.Font(font=self.font).measure(text) + 38
        self.configure(width=width)
        self.delete("all")
        self._image = ImageTk.PhotoImage(rounded_image(width, 28, 14, self.fill, self.parent_bg))
        self.create_image(0, 0, image=self._image, anchor="nw")
        self.create_oval(12, 11, 18, 17, fill=color, outline=color)
        self.create_text(26, 14, text=text, anchor="w", fill=PALETTE["nav_text"], font=self.font)


class HeroBanner(tk.Canvas):
    """Wide rounded banner with a violet glow, a headline and an action."""

    def __init__(self, parent: tk.Misc, title: str, body: str, action_text: str, command: Callable[[], None], art: Image.Image | None = None) -> None:
        super().__init__(parent, height=170, highlightthickness=0, borderwidth=0, background=PALETTE["background"])
        self.title = title
        self.body = body
        self.art = art
        self._image: ImageTk.PhotoImage | None = None
        self._pending: str | None = None
        self.button = PillButton(self, action_text, command=command, variant="primary", background="#15121F")
        self.bind("<Configure>", self._schedule)

    def _schedule(self, _event: tk.Event) -> None:
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(60, self._draw)

    def _background(self, width: int, height: int, show_art: bool) -> Image.Image:
        """Rounded banner background with a violet glow and optional artwork."""

        s = 2
        base = Image.new("RGBA", (width * s, height * s), _rgba("#15121F"))
        glow = Image.new("RGBA", (width * s, height * s), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow)
        for step in range(24, 0, -1):
            radius = step * width * s / 40
            alpha = int(7 + (24 - step) * 1.4)
            cx, cy = width * s * 0.72, height * s * 1.05
            glow_draw.ellipse((cx - radius, cy - radius * 0.6, cx + radius, cy + radius * 0.6), fill=(124, 92, 255, alpha))
        base = Image.alpha_composite(base, glow)
        if show_art:
            art_size = int(min(height, 190) * s * 0.78)
            art = self.art.resize((art_size, art_size), Image.Resampling.LANCZOS)
            base.alpha_composite(art, (int(width * s - art_size - height * s * 0.18), int(height * s * 0.11)))
        mask = Image.new("L", base.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, base.width - 1, base.height - 1), radius=22 * s, fill=255)
        border = Image.new("RGBA", base.size, _rgba(PALETTE["line"]))
        framed = Image.new("RGBA", base.size, _rgba(PALETTE["background"]))
        framed.paste(border, (0, 0), mask)
        inner_mask = Image.new("L", base.size, 0)
        ImageDraw.Draw(inner_mask).rounded_rectangle((s, s, base.width - 1 - s, base.height - 1 - s), radius=21 * s, fill=255)
        framed.paste(base, (0, 0), inner_mask)
        return framed.resize((width, height), Image.Resampling.LANCZOS)

    def _draw(self) -> None:
        self._pending = None
        width, height = self.winfo_width(), self.winfo_height()
        if width < 20:
            return
        self.delete("all")
        show_art = self.art is not None and width > 620
        text_width = max(width - (min(height, 190) * 1.15 if show_art else 0) - 70, 200)
        title = self.create_text(30, 28, text=self.title, anchor="nw", fill=PALETTE["text"], font=(FONT_SEMIBOLD, 20), width=text_width)
        title_bottom = self.bbox(title)[3]
        body = self.create_text(30, title_bottom + 8, text=self.body, anchor="nw", fill=PALETTE["muted"], font=(FONT, 10), width=min(text_width, 560))
        body_bottom = self.bbox(body)[3]
        wanted = int(body_bottom + 18 + int(self.button.cget("height")) + 26)
        if abs(wanted - height) > 2:
            self.configure(height=wanted)
            return
        framed = self._background(width, height, show_art)
        self._image = ImageTk.PhotoImage(framed)
        bg_item = self.create_image(0, 0, image=self._image, anchor="nw")
        self.tag_lower(bg_item)
        # The pill corners blend with the banner pixels right behind the button.
        red, green, blue, _ = framed.getpixel((40, height - 44))
        self.button.parent_bg = f"#{red:02X}{green:02X}{blue:02X}"
        tk.Canvas.configure(self.button, background=self.button.parent_bg)
        self.button._draw()
        self.create_window(30, height - 26, window=self.button, anchor="sw")


class FitLabel(ttk.Label):
    """Label whose font shrinks until the text fits the width of its container."""

    def __init__(
        self,
        parent: tk.Misc,
        textvariable: tk.StringVar | None = None,
        text: str = "",
        max_size: int = 22,
        min_size: int = 11,
        family: str | None = None,
        style: str = "CardValue.TLabel",
    ) -> None:
        self.fit_font = tk.font.Font(family=family or FONT_SEMIBOLD, size=max_size)
        self.max_size = max_size
        self.min_size = min_size
        options: dict[str, Any] = {"style": style, "font": self.fit_font}
        if textvariable is not None:
            options["textvariable"] = textvariable
            textvariable.trace_add("write", lambda *_args: self.after_idle(self._fit))
        else:
            options["text"] = text
        super().__init__(parent, **options)
        parent.bind("<Configure>", lambda _event: self._fit(), add="+")

    def _fit(self) -> None:
        available = self.master.winfo_width() - 4
        if available <= 4:
            return
        text = self.cget("text") if not str(self.cget("textvariable")) else self.getvar(str(self.cget("textvariable")))
        size = self.max_size
        self.fit_font.configure(size=size)
        while size > self.min_size and self.fit_font.measure(text) > available:
            size -= 1
            self.fit_font.configure(size=size)
