"""MapReduce-style distributed processing evidence for the FactuRisk SUNAT dataset.

This module processes the historical CSV by chunks and separates the work into
map, shuffle and reduce stages. It is designed to be reproducible in a local
academic environment while keeping the same logic that can be moved to Spark:
each chunk is mapped independently, partial aggregates are shuffled by RUC, and
the reducer consolidates provider-level indicators.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datos import iterar_lotes_comprobantes, normalizar_ruc, ruta_comprobantes
from paths import ensure_directories, get_application_root

PROJECT_ROOT = get_application_root()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CHUNK_SIZE = 25_000
STATE_ACCEPTED = "Aceptada"
STATE_OBSERVED = "Observada"
STATE_REJECTED = "Rechazada"
PENDING_STATES = {"Pendiente", "Borrador", "Pendiente / Borrador"}


def configurar_logging() -> None:
    """Configure console logging for the processing phase."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def limpiar_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Type-normalize the fields used by the MapReduce phase."""

    required = ["RUC_Proveedor", "Estado_Aceptacion", "Importe_Total", "Fecha_Emision"]
    missing_required = [column for column in required if column not in chunk.columns]
    if missing_required:
        raise ValueError(f"Faltan columnas para procesamiento distribuido: {', '.join(missing_required)}")

    chunk = chunk.copy()
    chunk["RUC_Proveedor"] = chunk["RUC_Proveedor"].map(normalizar_ruc).astype("string")
    chunk["Estado_Aceptacion"] = chunk["Estado_Aceptacion"].astype("string").fillna("SIN_ESTADO")
    chunk["Importe_Total"] = pd.to_numeric(chunk["Importe_Total"], errors="coerce")
    chunk["Fecha_Emision"] = pd.to_datetime(chunk["Fecha_Emision"], errors="coerce")
    return chunk


def mapear_chunk(chunk: pd.DataFrame, chunk_number: int) -> tuple[pd.DataFrame, Counter[str]]:
    """Map one chunk into partial provider indicators and duplicate counters."""

    chunk = limpiar_chunk(chunk)
    valid = chunk.dropna(subset=["RUC_Proveedor"]).copy()
    duplicate_counter: Counter[str] = Counter()
    if "ID_Comprobante" in valid.columns:
        ids = valid["ID_Comprobante"].dropna().astype(str)
        duplicate_counter.update(ids.tolist())

    valid["registros"] = 1
    valid["aceptadas"] = (valid["Estado_Aceptacion"] == STATE_ACCEPTED).astype(int)
    valid["observadas"] = (valid["Estado_Aceptacion"] == STATE_OBSERVED).astype(int)
    valid["rechazadas"] = (valid["Estado_Aceptacion"] == STATE_REJECTED).astype(int)
    valid["pendientes"] = valid["Estado_Aceptacion"].isin(PENDING_STATES).astype(int)
    valid["importe_total_suma"] = valid["Importe_Total"].fillna(0)
    valid["importe_validos"] = valid["Importe_Total"].notna().astype(int)

    grouped = (
        valid.groupby("RUC_Proveedor", dropna=False)
        .agg(
            registros=("registros", "sum"),
            aceptadas=("aceptadas", "sum"),
            observadas=("observadas", "sum"),
            rechazadas=("rechazadas", "sum"),
            pendientes=("pendientes", "sum"),
            importe_total_suma=("importe_total_suma", "sum"),
            importe_validos=("importe_validos", "sum"),
            fecha_emision_min=("Fecha_Emision", "min"),
            fecha_emision_max=("Fecha_Emision", "max"),
        )
        .reset_index()
    )
    grouped["chunk_origen"] = chunk_number
    return grouped, duplicate_counter


def reducir_parciales(partials: list[pd.DataFrame]) -> pd.DataFrame:
    """Reduce shuffled partial aggregations into one provider-level table."""

    if not partials:
        raise ValueError("No se generaron parciales para reducir.")

    shuffled = pd.concat(partials, ignore_index=True)
    reduced = (
        shuffled.groupby("RUC_Proveedor", dropna=False)
        .agg(
            registros=("registros", "sum"),
            aceptadas=("aceptadas", "sum"),
            observadas=("observadas", "sum"),
            rechazadas=("rechazadas", "sum"),
            pendientes=("pendientes", "sum"),
            importe_total_suma=("importe_total_suma", "sum"),
            importe_validos=("importe_validos", "sum"),
            fecha_emision_min=("fecha_emision_min", "min"),
            fecha_emision_max=("fecha_emision_max", "max"),
        )
        .reset_index()
    )
    reduced["incidencias_definitivas"] = reduced["observadas"] + reduced["rechazadas"]
    definitive = reduced["aceptadas"] + reduced["incidencias_definitivas"]
    reduced["tasa_incidencia_definitiva"] = np.where(
        definitive > 0,
        reduced["incidencias_definitivas"] / definitive,
        np.nan,
    )
    reduced["importe_promedio"] = np.where(
        reduced["importe_validos"] > 0,
        reduced["importe_total_suma"] / reduced["importe_validos"],
        np.nan,
    )
    return reduced.sort_values(["tasa_incidencia_definitiva", "registros"], ascending=[False, False])


def construir_reporte(
    csv_path: Path,
    resumen: pd.DataFrame,
    total_filas: int,
    chunks_procesados: int,
    duplicate_counter: Counter[str],
) -> dict[str, Any]:
    """Build the JSON report required as distributed processing evidence."""

    duplicated_ids = sum(1 for count in duplicate_counter.values() if count > 1)
    total_duplicate_rows = sum(count - 1 for count in duplicate_counter.values() if count > 1)
    states_totals = {
        "aceptadas": int(resumen["aceptadas"].sum()),
        "observadas": int(resumen["observadas"].sum()),
        "rechazadas": int(resumen["rechazadas"].sum()),
        "pendientes": int(resumen["pendientes"].sum()),
    }
    top_risk = resumen.head(10)[
        ["RUC_Proveedor", "registros", "incidencias_definitivas", "tasa_incidencia_definitiva"]
    ].copy()

    return {
        "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
        "archivo_historico": str(csv_path),
        "motor": "MapReduce local por chunks",
        "equivalencia_spark": (
            "map=procesamiento independiente por particion; "
            "shuffle=agrupacion por RUC_Proveedor; reduce=consolidacion de indicadores por proveedor"
        ),
        "chunk_size": CHUNK_SIZE,
        "chunks_procesados": chunks_procesados,
        "filas_procesadas": total_filas,
        "proveedores_unicos": int(resumen["RUC_Proveedor"].nunique()),
        "distribucion_estados_agregada": states_totals,
        "id_comprobante_duplicados": int(duplicated_ids),
        "filas_duplicadas_por_id_comprobante": int(total_duplicate_rows),
        "salida_resumen_proveedores": str(PROCESSED_DIR / "mapreduce_resumen_proveedores.csv"),
        "top_proveedores_riesgo": top_risk.to_dict(orient="records"),
        "uso_del_reporte": {
            "procesamiento_distribuido": "Resumen de la orquestacion del procesamiento por bloques.",
            "analitica_negocio": "Base agregada para analisis de calidad, incidencias y variables historicas sin fuga.",
        },
    }


def guardar_reporte_texto(report: dict[str, Any]) -> None:
    """Save a readable TXT explanation for the distributed processing phase."""

    lines = [
        "REPORTE DE PROCESAMIENTO DISTRIBUIDO - FACTURISK SUNAT",
        "=" * 70,
        f"Fecha de ejecucion: {report['fecha_ejecucion']}",
        f"Archivo historico: {report['archivo_historico']}",
        f"Motor usado: {report['motor']}",
        "",
        "Logica MapReduce implementada:",
        "1. MAP: cada bloque del CSV se limpia, normaliza y resume por RUC.",
        "2. SHUFFLE: los parciales se agrupan usando RUC_Proveedor como clave.",
        "3. REDUCE: se consolidan indicadores de aceptacion, incidencia, importes y fechas.",
        "",
        "Equivalencia con Spark:",
        report["equivalencia_spark"],
        "",
        "Resultados principales:",
        f"- Bloques procesados: {report['chunks_procesados']}",
        f"- Filas procesadas: {report['filas_procesadas']}",
        f"- Proveedores unicos: {report['proveedores_unicos']}",
        f"- ID_Comprobante duplicados: {report['id_comprobante_duplicados']}",
        f"- Filas duplicadas por ID_Comprobante: {report['filas_duplicadas_por_id_comprobante']}",
        "",
        "Distribucion agregada de estados:",
        json.dumps(report["distribucion_estados_agregada"], ensure_ascii=False, indent=2),
        "",
        "Este reporte resume la fase de procesamiento distribuido del proyecto.",
    ]
    (OUTPUTS_DIR / "reporte_procesamiento_distribuido.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def ejecutar_procesamiento_distribuido() -> pd.DataFrame:
    """Run the complete MapReduce-style phase and persist its artifacts."""

    ensure_directories()
    configurar_logging()
    dataset_path = ruta_comprobantes()
    logging.info("Iniciando procesamiento distribuido local: %s", dataset_path)

    partials: list[pd.DataFrame] = []
    duplicate_counter: Counter[str] = Counter()
    total_filas = 0
    chunks_procesados = 0

    for chunks_procesados, chunk in enumerate(iterar_lotes_comprobantes(CHUNK_SIZE), start=1):
        total_filas += len(chunk)
        partial, partial_duplicates = mapear_chunk(chunk, chunks_procesados)
        partials.append(partial)
        duplicate_counter.update(partial_duplicates)
        logging.info("Chunk %s procesado: %s filas", chunks_procesados, len(chunk))

    resumen = reducir_parciales(partials)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    resumen_path = PROCESSED_DIR / "mapreduce_resumen_proveedores.csv"
    reporte_path = OUTPUTS_DIR / "reporte_procesamiento_distribuido.json"
    resumen.to_csv(resumen_path, index=False, encoding="utf-8-sig")

    report = construir_reporte(dataset_path, resumen, total_filas, chunks_procesados, duplicate_counter)
    reporte_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    guardar_reporte_texto(report)

    print("\nResumen procesamiento distribuido")
    print(f"Filas procesadas: {total_filas}")
    print(f"Chunks procesados: {chunks_procesados}")
    print(f"Proveedores unicos: {resumen['RUC_Proveedor'].nunique()}")
    print(f"Reporte JSON: {reporte_path}")
    print(f"Resumen por proveedor: {resumen_path}")
    return resumen


if __name__ == "__main__":
    ejecutar_procesamiento_distribuido()
