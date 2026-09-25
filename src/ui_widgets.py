"""Reusable Tkinter widgets that keep the interface usable at any window size."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable

from PIL import Image, ImageTk

from theme import FONT, FONT_SEMIBOLD, PALETTE


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
        needs_scroll = self.body.winfo_reqheight() > self.canvas.winfo_height()
        if needs_scroll:
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

    def __init__(self, parent: tk.Misc, min_item_width: int, max_columns: int, gap: int = 12, style: str = "App.TFrame") -> None:
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

    def clear(self) -> None:
        for widget in self.items:
            widget.destroy()
        self.items.clear()
        self._columns = 0

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


class NavItem(tk.Frame):
    """Sidebar entry with icon, label, hover/selected states and collapsed mode."""

    def __init__(self, parent: tk.Misc, text: str, icon: ImageTk.PhotoImage | None, command: Callable[[], None]) -> None:
        super().__init__(parent, background=PALETTE["ink"], cursor="hand2")
        self.command = command
        self.selected = False
        self.enabled = True
        self.marker = tk.Frame(self, width=3, background=PALETTE["ink"])
        self.marker.pack(side="left", fill="y")
        self.icon_label = tk.Label(self, image=icon, background=PALETTE["ink"], borderwidth=0)
        self.icon_label.pack(side="left", padx=(14, 10), pady=8)
        self.text_label = tk.Label(
            self,
            text=text,
            anchor="w",
            background=PALETTE["ink"],
            foreground=PALETTE["nav_text"],
            font=(FONT, 10),
        )
        self.text_label.pack(side="left", fill="x", expand=True, padx=(0, 12))
        for widget in (self, self.icon_label, self.text_label, self.marker):
            widget.bind("<Button-1>", self._on_click)
            widget.bind("<Enter>", lambda _e: self._paint(hover=True))
            widget.bind("<Leave>", lambda _e: self._paint(hover=False))

    def _on_click(self, _event: tk.Event) -> None:
        if self.enabled:
            self.command()

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self._paint(hover=False)

    def set_collapsed(self, collapsed: bool) -> None:
        if collapsed:
            self.text_label.pack_forget()
            self.icon_label.pack_configure(padx=(17, 17))
        else:
            self.icon_label.pack_configure(padx=(14, 10))
            self.text_label.pack(side="left", fill="x", expand=True, padx=(0, 12))

    def _paint(self, hover: bool) -> None:
        if self.selected:
            background = PALETTE["ink_active"]
        elif hover:
            background = PALETTE["ink_hover"]
        else:
            background = PALETTE["ink"]
        self.configure(background=background)
        self.icon_label.configure(background=background)
        self.text_label.configure(
            background=background,
            foreground="#FFFFFF" if self.selected else PALETTE["nav_text"],
            font=(FONT_SEMIBOLD if self.selected else FONT, 10),
        )
        self.marker.configure(background=PALETTE["accent"] if self.selected else background)


def card(parent: tk.Misc, padding: tuple[int, int] = (16, 14)) -> ttk.Frame:
    """White surface with a subtle border."""

    outer = tk.Frame(parent, background=PALETTE["line"])
    inner = ttk.Frame(outer, style="Card.TFrame", padding=padding)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    outer.inner = inner  # type: ignore[attr-defined]
    return outer  # type: ignore[return-value]
