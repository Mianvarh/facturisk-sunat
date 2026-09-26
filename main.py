"""Pipeline entry point for FactuRisk SUNAT."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import ensure_directories, get_application_root  # noqa: E402

PROJECT_ROOT = get_application_root()
SRC_DIR = PROJECT_ROOT / "src"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PIPELINE_LOG = OUTPUTS_DIR / "pipeline.log"

PHASE_SCRIPTS = {
    "gui": PROJECT_ROOT / "src" / "gui_app.py",
    "inspect": PROJECT_ROOT / "src" / "inspect_data.py",
    "distribuido": PROJECT_ROOT / "src" / "distributed_processing.py",
    "scraping": PROJECT_ROOT / "src" / "scrape_sunat.py",
    "mongodb": PROJECT_ROOT / "src" / "load_mongodb.py",
    "preparar": PROJECT_ROOT / "src" / "prepare_dataset.py",
    "entrenar": PROJECT_ROOT / "src" / "train_model.py",
    "predecir": PROJECT_ROOT / "src" / "predict_pending.py",
    "resumen": PROJECT_ROOT / "src" / "resumen_resultados.py",
    "documentar": PROJECT_ROOT / "src" / "generate_documentation_data.py",
}

TODO_ORDER = ["inspect", "distribuido", "scraping", "mongodb", "preparar", "entrenar", "predecir", "documentar"]

FINAL_ARTIFACTS = [
    PROJECT_ROOT / "outputs" / "reporte_procesamiento_distribuido.txt",
    PROJECT_ROOT / "outputs" / "09_matriz_confusion_optimizada.png",
    PROJECT_ROOT / "outputs" / "reporte_modelo.txt",
    PROJECT_ROOT / "data" / "processed" / "predicciones_pendientes.csv",
]


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as a compact human-readable string."""

    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes}m {remainder:.1f}s"
    if minutes:
        return f"{minutes}m {remainder:.1f}s"
    return f"{remainder:.1f}s"


def write_message(message: str, log_file) -> None:
    """Print and persist one pipeline message."""

    # pythonw.exe (desktop shortcut) runs without a console: sys.stdout is None.
    if sys.stdout is not None:
        console_encoding = sys.stdout.encoding or "utf-8"
        safe_message = message.encode(console_encoding, errors="replace").decode(
            console_encoding,
            errors="replace",
        )
        print(safe_message)
    log_file.write(message + "\n")
    log_file.flush()


def build_command(phase: str, *, force_scraping: bool) -> list[str]:
    """Build the subprocess command for a phase."""

    if getattr(sys, "frozen", False):
        command = [sys.executable, "--worker-phase", phase]
        if phase == "scraping" and force_scraping:
            command.append("--force-scraping")
        return command

    script = PHASE_SCRIPTS[phase]
    if not script.exists():
        raise FileNotFoundError(f"No existe el script de la fase {phase}: {script}")

    command = [sys.executable, str(script)]
    if phase == "scraping" and force_scraping:
        command.append("--force-download")
    return command


def run_phase(phase: str, *, force_scraping: bool, log_file) -> int:
    """Run one pipeline phase from the project root and stream logs."""

    command = build_command(phase, force_scraping=force_scraping)
    start = time.perf_counter()
    write_message("", log_file)
    write_message("=" * 80, log_file)
    write_message(f"INICIO DE FASE: {phase}", log_file)
    write_message(f"Comando: {' '.join(command)}", log_file)
    write_message(f"Hora inicio: {datetime.now().isoformat(timespec='seconds')}", log_file)
    write_message("=" * 80, log_file)

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        write_message(line, log_file)

    return_code = process.wait()
    elapsed = time.perf_counter() - start

    if return_code == 0:
        write_message(f"FASE COMPLETADA: {phase} | Tiempo: {format_duration(elapsed)}", log_file)
    else:
        write_message(
            f"FASE FALLIDA: {phase} | Codigo: {return_code} | Tiempo: {format_duration(elapsed)}",
            log_file,
        )
    return return_code


def phases_to_run(selected_phase: str) -> list[str]:
    """Return the list of phases requested by the user."""

    if selected_phase == "todo":
        return TODO_ORDER
    return [selected_phase]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Ejecuta fases del proyecto FactuRisk SUNAT.")
    parser.add_argument(
        "--worker-phase",
        choices=[
            "gui",
            "inspect",
            "distribuido",
            "scraping",
            "mongodb",
            "preparar",
            "entrenar",
            "predecir",
            "resumen",
            "documentar",
        ],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "phase",
        choices=[
            "gui",
            "inspect",
            "distribuido",
            "scraping",
            "mongodb",
            "preparar",
            "entrenar",
            "predecir",
            "resumen",
            "documentar",
            "todo",
        ],
        nargs="?",
        help="Fase a ejecutar.",
    )
    parser.add_argument(
        "--force-scraping",
        action="store_true",
        help="Fuerza volver a descargar el ZIP de SUNAT durante la fase scraping.",
    )
    return parser.parse_args()


def run_selected_phase(phase: str, force_scraping: bool) -> int:
    """Run one CLI/menu selection through the common subprocess pipeline."""

    selected_phases = phases_to_run(phase)
    total_start = time.perf_counter()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with PIPELINE_LOG.open("a", encoding="utf-8") as log_file:
        write_message("", log_file)
        write_message("#" * 80, log_file)
        phase_label = "proceso completo" if phase == "todo" else phase
        write_message(f"EJECUCION PIPELINE: {phase_label}", log_file)
        write_message(f"Proyecto: {PROJECT_ROOT}", log_file)
        write_message(f"Inicio total: {datetime.now().isoformat(timespec='seconds')}", log_file)
        write_message(f"Force scraping: {force_scraping}", log_file)
        write_message("#" * 80, log_file)

        for current_phase in selected_phases:
            return_code = run_phase(current_phase, force_scraping=force_scraping, log_file=log_file)
            if return_code != 0:
                write_message(f"PIPELINE DETENIDO | Tiempo total: {format_duration(time.perf_counter() - total_start)}", log_file)
                return return_code

        write_message("", log_file)
        completion_text = "PROCESO COMPLETO FINALIZADO" if phase == "todo" else "PIPELINE COMPLETADO"
        write_message(f"{completion_text} | Tiempo total: {format_duration(time.perf_counter() - total_start)}", log_file)
        write_message("Rutas finales:", log_file)
        for artifact in FINAL_ARTIFACTS:
            write_message(f"  {artifact}", log_file)
        write_message(f"Log completo: {PIPELINE_LOG}", log_file)

        if phase == "todo":
            write_message("", log_file)
            run_phase("resumen", force_scraping=False, log_file=log_file)
    return 0


MENU_OPTIONS = [
    ("1", "Inspeccionar datos", "inspect"),
    ("2", "Procesamiento distribuido", "distribuido"),
    ("3", "Ejecutar scraping SUNAT", "scraping"),
    ("4", "Cargar MongoDB", "mongodb"),
    ("5", "Preparar datos y crear variables", "preparar"),
    ("6", "Entrenar, optimizar umbral y calibrar", "entrenar"),
    ("7", "Predecir comprobantes pendientes", "predecir"),
    ("8", "Mostrar resumen de resultados", "resumen"),
    ("9", "Ejecutar proceso completo", "todo"),
    ("10", "Abrir la interfaz gráfica", "gui"),
]


def menu_interactivo() -> int:
    """Interactive terminal menu."""

    options = {key: phase for key, _label, phase in MENU_OPTIONS}
    while True:
        print("\n" + "=" * 40)
        print("MENU FACTURISK SUNAT")
        print("=" * 40)
        for key, label, _phase in MENU_OPTIONS:
            print(f"{key}. {label}")
        print("0. Salir")
        try:
            choice = input("Seleccione una opción: ").strip()
        except EOFError:
            print("Entrada no interactiva detectada. Saliendo del menú.")
            return 0
        if choice == "0":
            return 0
        if choice in options:
            run_selected_phase(options[choice], force_scraping=False)
        else:
            print("Opción no válida.")
        try:
            input("Presione Enter para volver al menú.")
        except EOFError:
            return 0


def run_worker_phase(phase: str, *, force_scraping: bool = False) -> int:
    """Run one phase inside the frozen executable worker process."""

    try:
        ensure_directories()
        worker_argv = [sys.argv[0]]
        if phase == "scraping" and force_scraping:
            worker_argv.append("--force-download")
        sys.argv = worker_argv
        if phase == "gui":
            from gui_app import lanzar_gui

            lanzar_gui()
        elif phase == "inspect":
            from inspect_data import main as inspect_main

            inspect_main()
        elif phase == "distribuido":
            from distributed_processing import ejecutar_procesamiento_distribuido

            ejecutar_procesamiento_distribuido()
        elif phase == "scraping":
            from scrape_sunat import obtener_datos_sunat

            obtener_datos_sunat(force_download=force_scraping)
        elif phase == "mongodb":
            from load_mongodb import cargar_proveedores_mongodb

            cargar_proveedores_mongodb()
        elif phase == "preparar":
            from prepare_dataset import preparar_dataset

            preparar_dataset()
        elif phase == "entrenar":
            from train_model import entrenar_modelos

            entrenar_modelos()
        elif phase == "predecir":
            from predict_pending import predecir_pendientes

            predecir_pendientes()
        elif phase == "resumen":
            from resumen_resultados import mostrar_resumen

            mostrar_resumen()
        elif phase == "documentar":
            from generate_documentation_data import actualizar_documentacion_resultados

            actualizar_documentacion_resultados()
        else:
            raise ValueError(f"Fase no reconocida: {phase}")
        return 0
    except Exception as exc:
        print(f"FASE FALLIDA: {phase}. Revise logs. Detalle: {exc}")
        return 1


def main() -> int:
    """Run the requested project phase or the full pipeline."""

    ensure_directories()
    args = parse_args()
    if args.worker_phase:
        return run_worker_phase(args.worker_phase, force_scraping=args.force_scraping)
    if args.phase is None:
        try:
            from preflight_check import ejecutar_preflight

            ejecutar_preflight(mostrar=True)
        except Exception as exc:
            print(f"No se pudo completar la verificacion inicial: {exc}")
        return menu_interactivo()
    return run_selected_phase(args.phase, args.force_scraping)


if __name__ == "__main__":
    raise SystemExit(main())

