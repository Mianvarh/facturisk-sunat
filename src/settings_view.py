"""Settings panel: historical data import, SUNAT sources and MongoDB credentials."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import BooleanVar, StringVar, filedialog, messagebox, ttk
from typing import Callable

import pandas as pd
from PIL import ImageTk

import configuracion as cfg
from datos import PADRONES_BACKUP_CSV
from theme import PALETTE
from ui_widgets import PillButton, ScrollableFrame, StatusPill, card

SURFACE = PALETTE["surface"]


class SettingsView(ttk.Frame):
    """Three cards: data import, external SUNAT sources and MongoDB connection."""

    def __init__(
        self,
        parent: tk.Misc,
        icons: dict[str, ImageTk.PhotoImage],
        on_dataset_changed: Callable[[], None],
        on_sources_changed: Callable[[], None],
    ) -> None:
        super().__init__(parent, style="App.TFrame")
        self.icons = icons
        self.on_dataset_changed = on_dataset_changed
        self.on_sources_changed = on_sources_changed
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, style="App.TFrame", padding=(28, 22, 28, 14))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Configuración", style="H1.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Datos históricos, fuentes externas y conexión a MongoDB. Todo se guarda solo en este equipo y nunca se sube a GitHub.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        scroll = ScrollableFrame(self)
        scroll.grid(row=1, column=0, sticky="nsew")
        self.content = ttk.Frame(scroll.body, style="App.TFrame", padding=(28, 0, 28, 24))
        self.content.pack(fill="both", expand=True)
        self.content.columnconfigure(0, weight=1)

        self._build_dataset_card()
        self._build_sunat_card()
        self._build_mongo_card()

    # ----------------------------------------------------------------- helpers

    def _card(self, row: int, icon: str, title: str, subtitle: str) -> ttk.Frame:
        box = card(self.content, padding=(22, 20))
        box.grid(row=row, column=0, sticky="ew", pady=(0, 16))
        inner = box.inner
        inner.columnconfigure(0, weight=1)
        head = ttk.Frame(inner, style="Card.TFrame")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        head.columnconfigure(1, weight=1)
        if f"{icon}_tile" in self.icons:
            ttk.Label(head, image=self.icons[f"{icon}_tile"], style="CardBody.TLabel").grid(row=0, column=0, rowspan=2, padx=(0, 14))
        ttk.Label(head, text=title, style="CardTitle.TLabel").grid(row=0, column=1, sticky="sw")
        subtitle_label = ttk.Label(head, text=subtitle, style="CardBody.TLabel", justify="left")
        subtitle_label.grid(row=1, column=1, sticky="nw")
        head.bind("<Configure>", lambda event: subtitle_label.configure(wraplength=max(event.width - 80, 200)), add="+")
        body = ttk.Frame(inner, style="Card.TFrame")
        body.grid(row=1, column=0, sticky="ew")
        body.columnconfigure(1, weight=1)
        return body

    @staticmethod
    def _field(parent: ttk.Frame, row: int, label: str, variable: StringVar, show: str | None = None) -> ttk.Entry:
        ttk.Label(parent, text=label, style="FieldLabel.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 16), pady=6)
        entry = ttk.Entry(parent, textvariable=variable, style="Field.TEntry", show=show or "")
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        return entry

    # ------------------------------------------------------------------ dataset

    def _build_dataset_card(self) -> None:
        body = self._card(
            0,
            "documentar",
            "Datos históricos",
            "Carga tu propio historial de comprobantes en CSV o Excel. Las columnas se detectan automáticamente: "
            "RUC del proveedor, fecha de emisión, importe y estado de aceptación (obligatorias); tipo, moneda, "
            "vencimiento y razón social (opcionales).",
        )
        self.dataset_info = ttk.Frame(body, style="Card.TFrame")
        self.dataset_info.grid(row=0, column=0, columnspan=2, sticky="ew")
        buttons = ttk.Frame(body, style="Card.TFrame")
        buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(16, 0))
        self.import_button = PillButton(buttons, "Importar CSV o Excel…", command=self._import_file, background=SURFACE)
        self.import_button.pack(side="left", padx=(0, 10))
        PillButton(buttons, "Usar dataset de demostración", command=self._use_demo, variant="secondary", background=SURFACE).pack(side="left")
        self.import_result = ttk.Label(body, text="", style="CardBody.TLabel", justify="left")
        self.import_result.grid(row=2, column=0, columnspan=2, sticky="w", pady=(14, 0))
        self._refresh_dataset_info()

    def _refresh_dataset_info(self) -> None:
        from importar_datos import resumen_dataset_activo

        for child in self.dataset_info.winfo_children():
            child.destroy()
        info = resumen_dataset_activo()
        StatusPill(self.dataset_info, f"Dataset activo: {info['origen']}", PALETTE["primary"] if info["es_demo"] else PALETTE["success"], background=SURFACE).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 12)
        )
        items = [
            ("Archivo", str(info["archivo"])),
            ("Comprobantes", f"{int(info['filas']):,}"),
            ("Proveedores", f"{int(info['rucs']):,}"),
            ("Periodo", f"{info['desde']} – {info['hasta']}"),
        ]
        for column, (label, value) in enumerate(items):
            self.dataset_info.columnconfigure(column, weight=1)
            ttk.Label(self.dataset_info, text=label, style="FieldLabel.TLabel").grid(row=1, column=column, sticky="w")
            ttk.Label(self.dataset_info, text=value, style="CardTitle.TLabel").grid(row=2, column=column, sticky="w", padx=(0, 12))

    def _import_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Selecciona el historial de comprobantes",
            filetypes=[("CSV o Excel", "*.csv *.txt *.xlsx *.xlsm"), ("CSV", "*.csv *.txt"), ("Excel", "*.xlsx *.xlsm")],
        )
        if not path:
            return
        self.import_button.configure(state="disabled", text="Importando…")
        self.import_result.configure(text=f"Leyendo {Path(path).name}…", foreground=PALETTE["muted"])

        def work() -> None:
            from importar_datos import importar

            try:
                report = importar(Path(path))
                self.after(0, lambda: self._import_done(report, None))
            except Exception as exc:  # shown to the user as a readable error
                self.after(0, lambda error=exc: self._import_done(None, error))

        threading.Thread(target=work, daemon=True).start()

    def _import_done(self, report: object, error: Exception | None) -> None:
        self.import_button.configure(state="normal", text="Importar CSV o Excel…")
        if error is not None:
            self.import_result.configure(text=f"No se pudo importar: {error}", foreground=PALETTE["danger"])
            return
        mapping = "\n".join(f"   {canonical}  ←  {source}" for canonical, source in report.columnas_detectadas.items())
        warnings = "\n".join(f"   • {text}" for text in report.advertencias) or "   Sin advertencias."
        states = ", ".join(f"{state}: {count:,}" for state, count in report.estados.items())
        self.import_result.configure(
            foreground=PALETTE["nav_text"],
            text=(
                f"✓ {report.filas:,} comprobantes importados de {report.archivo} ({report.rucs_unicos:,} proveedores).\n\n"
                f"Columnas detectadas:\n{mapping}\n\nEstados: {states}\n\nAdvertencias:\n{warnings}\n\n"
                "Ejecuta el pipeline desde el Paso 01 para entrenar con estos datos."
            ),
        )
        self._refresh_dataset_info()
        self.on_dataset_changed()

    def _use_demo(self) -> None:
        from importar_datos import usar_dataset_demo

        usar_dataset_demo()
        self.import_result.configure(text="Se activó el dataset público de demostración. Ejecuta el pipeline desde el Paso 01.", foreground=PALETTE["nav_text"])
        self._refresh_dataset_info()
        self.on_dataset_changed()

    # -------------------------------------------------------------------- SUNAT

    def _build_sunat_card(self) -> None:
        settings = cfg.cargar_ajustes()["sunat"]
        body = self._card(
            1,
            "scraping",
            "Fuente externa SUNAT",
            "De dónde se obtiene la variable externa. Por defecto se descarga el padrón reducido oficial; puedes indicar "
            "otra página, un enlace directo al ZIP o un archivo ya descargado, y sumar padrones oficiales adicionales como variables del modelo.",
        )
        self.var_page = StringVar(value=settings.get("pagina_padron", cfg.SUNAT_PADRON_PAGE))
        self.var_zip = StringVar(value=settings.get("zip_url", ""))
        self.var_local = StringVar(value=settings.get("archivo_local", ""))
        self.var_age = StringVar(value=str(settings.get("max_age_days", 7)))
        self._field(body, 0, "Página del padrón reducido", self.var_page)
        self._field(body, 1, "URL directa del ZIP (opcional)", self.var_zip)
        self._field(body, 2, "Archivo local ZIP o TXT (opcional)", self.var_local)
        PillButton(body, "Examinar…", command=self._browse_padron, variant="secondary", background=SURFACE, height=34).grid(row=2, column=2, padx=(10, 0))
        ttk.Label(body, text="Reutilizar descargas de hasta (días)", style="FieldLabel.TLabel").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Spinbox(body, from_=0, to=90, textvariable=self.var_age, width=6, style="Field.TSpinbox").grid(row=3, column=1, sticky="w", pady=6)

        ttk.Label(body, text="PADRONES ADICIONALES (VARIABLES EXTRA DEL MODELO)", style="FieldLabel.TLabel").grid(row=4, column=0, columnspan=3, sticky="w", pady=(18, 6))
        coverage = self._padron_coverage()
        self.padron_vars: dict[str, BooleanVar] = {}
        for index, (key, info) in enumerate(cfg.PADRONES_SUNAT.items()):
            var = BooleanVar(value=bool(settings["padrones"].get(key, False)))
            self.padron_vars[key] = var
            row = ttk.Frame(body, style="Card.TFrame")
            row.grid(row=5 + index, column=0, columnspan=3, sticky="ew", pady=3)
            ttk.Checkbutton(row, text=info["nombre"], variable=var, style="Card.TCheckbutton").pack(side="left")
            extra = f" · {coverage[info['columna']]:,} proveedores del dataset" if info["columna"] in coverage else ""
            ttk.Label(row, text=f"  {info['descripcion']}{extra}", style="CardBody.TLabel").pack(side="left")

        buttons = ttk.Frame(body, style="Card.TFrame")
        buttons.grid(row=10, column=0, columnspan=3, sticky="w", pady=(16, 0))
        PillButton(buttons, "Guardar fuentes", command=self._save_sunat, background=SURFACE).pack(side="left", padx=(0, 10))
        PillButton(buttons, "Restablecer valores oficiales", command=self._reset_sunat, variant="secondary", background=SURFACE).pack(side="left")
        self.sunat_status = ttk.Label(body, text="", style="CardBody.TLabel")
        self.sunat_status.grid(row=11, column=0, columnspan=3, sticky="w", pady=(10, 0))

    @staticmethod
    def _padron_coverage() -> dict[str, int]:
        if not PADRONES_BACKUP_CSV.exists():
            return {}
        try:
            frame = pd.read_csv(PADRONES_BACKUP_CSV, encoding="utf-8-sig", usecols=lambda column: column in cfg.PADRON_COLUMNS)
        except (OSError, ValueError):
            return {}
        return {column: int(frame[column].sum()) for column in frame.columns}

    def _browse_padron(self) -> None:
        path = filedialog.askopenfilename(title="Padrón SUNAT descargado", filetypes=[("Padrón SUNAT", "*.zip *.txt"), ("Todos", "*.*")])
        if path:
            self.var_local.set(path)

    def _save_sunat(self) -> None:
        try:
            age = max(0.0, float(self.var_age.get().replace(",", ".")))
        except ValueError:
            self.sunat_status.configure(text="La cantidad de días debe ser un número.", foreground=PALETTE["danger"])
            return
        local = self.var_local.get().strip()
        if local and not Path(local).exists():
            self.sunat_status.configure(text="El archivo local indicado no existe.", foreground=PALETTE["danger"])
            return
        page = self.var_page.get().strip() or cfg.SUNAT_PADRON_PAGE
        if not page.startswith(("http://", "https://")):
            self.sunat_status.configure(text="La página del padrón debe empezar con http:// o https://.", foreground=PALETTE["danger"])
            return
        cfg.actualizar_ajustes(
            sunat={
                "pagina_padron": page,
                "zip_url": self.var_zip.get().strip(),
                "archivo_local": local,
                "max_age_days": age,
                "padrones": {key: var.get() for key, var in self.padron_vars.items()},
            }
        )
        self.sunat_status.configure(text="✓ Fuentes guardadas. Se aplicarán al ejecutar el Paso 03 (Scraping SUNAT).", foreground=PALETTE["success"])
        self.on_sources_changed()

    def _reset_sunat(self) -> None:
        defaults = cfg.DEFAULT_SETTINGS["sunat"]
        self.var_page.set(defaults["pagina_padron"])
        self.var_zip.set("")
        self.var_local.set("")
        self.var_age.set(str(defaults["max_age_days"]))
        for key, var in self.padron_vars.items():
            var.set(defaults["padrones"][key])
        self._save_sunat()

    # ------------------------------------------------------------------ MongoDB

    def _build_mongo_card(self) -> None:
        stored = cfg.leer_mongo_panel()
        body = self._card(
            2,
            "mongodb",
            "MongoDB",
            "Base NoSQL donde se suben los datos SUNAT limpios después del scraping (Paso 04). Puede ser MongoDB Atlas o un "
            "MongoDB local (docker compose up -d). Las credenciales se guardan en config/configuracion.json, excluido de Git, "
            "y tienen prioridad sobre el archivo .env.",
        )
        self.var_use_mongo = BooleanVar(value=bool(stored.get("usar_mongodb", True)))
        self.var_uri = StringVar(value=stored.get("mongodb_uri", ""))
        self.var_db = StringVar(value=stored.get("mongodb_database", "facturisk"))
        self.var_collection = StringVar(value=stored.get("mongodb_collection", "proveedores_sunat"))
        ttk.Checkbutton(body, text="Usar MongoDB en el pipeline", variable=self.var_use_mongo, style="Card.TCheckbutton").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.uri_entry = self._field(body, 1, "URI de conexión", self.var_uri, show="•")
        self.show_uri = False
        PillButton(body, "Mostrar", command=self._toggle_uri, variant="secondary", background=SURFACE, height=34).grid(row=1, column=2, padx=(10, 0))
        self._field(body, 2, "Base de datos", self.var_db)
        self._field(body, 3, "Colección", self.var_collection)
        ttk.Label(body, text="Ejemplo Atlas: mongodb+srv://usuario:contraseña@cluster.mongodb.net  ·  Local: mongodb://localhost:27017", style="CardBody.TLabel").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )
        buttons = ttk.Frame(body, style="Card.TFrame")
        buttons.grid(row=5, column=0, columnspan=3, sticky="w", pady=(16, 0))
        self.test_button = PillButton(buttons, "Probar conexión", command=self._test_mongo, variant="secondary", background=SURFACE)
        self.test_button.pack(side="left", padx=(0, 10))
        PillButton(buttons, "Guardar credenciales", command=self._save_mongo, background=SURFACE).pack(side="left", padx=(0, 10))
        PillButton(buttons, "Eliminar credenciales", command=self._delete_mongo, variant="danger", background=SURFACE).pack(side="left")
        self.mongo_status = StatusPill(body, self._mongo_state_text(), PALETTE["subtle"], background=SURFACE)
        self.mongo_status.grid(row=6, column=0, columnspan=3, sticky="w", pady=(14, 0))

    @staticmethod
    def _mongo_state_text() -> str:
        config = cfg.cargar_mongo_config()
        if config is None:
            return "Sin configurar: el pipeline usa el respaldo local"
        return f"Configurado: {cfg.ocultar_uri(config[0])} · {config[1]}.{config[2]}"

    def _toggle_uri(self) -> None:
        self.show_uri = not self.show_uri
        self.uri_entry.configure(show="" if self.show_uri else "•")

    def _test_mongo(self) -> None:
        uri = self.var_uri.get().strip()
        if not uri:
            self.mongo_status.set("Ingresa una URI para probar la conexión", PALETTE["warning"])
            return
        self.test_button.configure(state="disabled", text="Probando…")
        self.mongo_status.set("Conectando…", PALETTE["warning"])

        def work() -> None:
            ok, message = cfg.probar_mongo(uri)
            self.after(0, lambda: self._test_done(ok, message))

        threading.Thread(target=work, daemon=True).start()

    def _test_done(self, ok: bool, message: str) -> None:
        self.test_button.configure(state="normal", text="Probar conexión")
        self.mongo_status.set(message, PALETTE["success"] if ok else PALETTE["danger"])

    def _save_mongo(self) -> None:
        if self.var_use_mongo.get() and not (self.var_uri.get().strip() and self.var_db.get().strip() and self.var_collection.get().strip()):
            self.mongo_status.set("Completa URI, base de datos y colección", PALETTE["warning"])
            return
        cfg.guardar_mongo_panel(self.var_uri.get(), self.var_db.get(), self.var_collection.get(), self.var_use_mongo.get())
        self.mongo_status.set("✓ Guardado · " + self._mongo_state_text(), PALETTE["success"])
        self.on_sources_changed()

    def _delete_mongo(self) -> None:
        if not messagebox.askyesno("Eliminar credenciales", "¿Eliminar la conexión a MongoDB guardada en este equipo?"):
            return
        cfg.borrar_mongo_panel()
        self.var_uri.set("")
        self.mongo_status.set(self._mongo_state_text(), PALETTE["subtle"])
        self.on_sources_changed()
