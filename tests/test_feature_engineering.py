from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import feature_engineering
from feature_engineering import HISTORICAL_FEATURES, _rolling_previous_counts, crear_variables_historicas_sin_fuga


def _rolling_previous_counts_reference(group: pd.DataFrame) -> pd.DataFrame:
    """Reference implementation kept from the original row-wise algorithm."""

    records_30: deque[tuple[pd.Timestamp, float]] = deque()
    records_90: deque[tuple[pd.Timestamp, float]] = deque()
    rows: list[dict[str, object]] = []

    for index, row in group.iterrows():
        current_date = row["Fecha_Emision"]
        if pd.isna(current_date):
            rows.append(
                {
                    "index": index,
                    "comprobantes_ultimos_30_dias": 0,
                    "comprobantes_ultimos_90_dias": 0,
                    "incidencias_ultimos_30_dias": 0,
                    "incidencias_ultimos_90_dias": 0,
                }
            )
            continue

        limit_30 = current_date - pd.Timedelta(days=30)
        limit_90 = current_date - pd.Timedelta(days=90)
        while records_30 and records_30[0][0] < limit_30:
            records_30.popleft()
        while records_90 and records_90[0][0] < limit_90:
            records_90.popleft()

        rows.append(
            {
                "index": index,
                "comprobantes_ultimos_30_dias": len(records_30),
                "comprobantes_ultimos_90_dias": len(records_90),
                "incidencias_ultimos_30_dias": int(sum(value for _, value in records_30)),
                "incidencias_ultimos_90_dias": int(sum(value for _, value in records_90)),
            }
        )

        incidence = row["_incidencia_historica"]
        incidence_value = 0.0 if pd.isna(incidence) else float(incidence)
        records_30.append((current_date, incidence_value))
        records_90.append((current_date, incidence_value))

    return pd.DataFrame(rows).set_index("index")


def test_rolling_previous_counts_matches_reference_with_duplicate_and_nat_dates() -> None:
    """The vectorized windows preserve the original predecessor semantics."""

    rng = np.random.default_rng(20260925)
    row_count = 240
    frame = pd.DataFrame(
        {
            "RUC_Proveedor": rng.choice(["20100000001", "20200000002", "20300000003"], row_count),
            "Fecha_Emision": pd.Timestamp("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 180, row_count), unit="D"),
            "_incidencia_historica": rng.choice([0.0, 1.0, np.nan], row_count),
            "_orden_original": np.arange(row_count),
        }
    )
    frame.loc[[0, 1, 2], "RUC_Proveedor"] = "20100000001"
    frame.loc[[0, 1, 2], "Fecha_Emision"] = pd.Timestamp("2024-01-15")
    frame.loc[[7, 38, 119, 200], "Fecha_Emision"] = pd.NaT
    ordered = frame.sort_values(["RUC_Proveedor", "Fecha_Emision", "_orden_original"])

    actual = pd.concat(
        [_rolling_previous_counts(group) for _, group in ordered.groupby("RUC_Proveedor", sort=False)],
    ).sort_index()
    expected = pd.concat(
        [_rolling_previous_counts_reference(group) for _, group in ordered.groupby("RUC_Proveedor", sort=False)],
    ).sort_index()

    assert_frame_equal(actual, expected)


def test_historical_features_do_not_use_current_target(monkeypatch, tmp_path) -> None:
    """Changing a row's target does not change that row's historical features."""

    monkeypatch.setattr(feature_engineering, "OUTPUTS_DIR", tmp_path)
    source = pd.DataFrame(
        {
            "RUC_Proveedor": ["20100000001", "20100000001", "20100000001", "20200000002"],
            "Fecha_Emision": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-02-15", "2024-01-05"]),
            "Importe_Total": [100.0, 200.0, 150.0, 80.0],
            "Incidencia_Aceptacion": [0.0, 1.0, 0.0, 1.0],
        }
    )
    changed = source.copy()
    changed.loc[2, "Incidencia_Aceptacion"] = 1.0

    original_result, _ = crear_variables_historicas_sin_fuga(source)
    changed_result, _ = crear_variables_historicas_sin_fuga(changed)

    assert_frame_equal(
        original_result.loc[[2], HISTORICAL_FEATURES],
        changed_result.loc[[2], HISTORICAL_FEATURES],
    )


def test_historical_features_ignore_future_targets(monkeypatch, tmp_path) -> None:
    """Outcomes of later invoices (any supplier) never change earlier features."""

    monkeypatch.setattr(feature_engineering, "OUTPUTS_DIR", tmp_path)
    source = pd.DataFrame(
        {
            "RUC_Proveedor": ["20100000001", "20200000002", "20100000001", "20200000002", "20300000003"],
            "Fecha_Emision": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-03-01", "2024-03-02"]),
            "Importe_Total": [100.0, 50.0, 120.0, 60.0, 90.0],
            "Incidencia_Aceptacion": [1.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    changed = source.copy()
    changed.loc[[3, 4], "Incidencia_Aceptacion"] = 1.0

    original_result, _ = crear_variables_historicas_sin_fuga(source)
    changed_result, _ = crear_variables_historicas_sin_fuga(changed)

    earlier = [0, 1, 2, 3]
    assert_frame_equal(
        original_result.loc[earlier, HISTORICAL_FEATURES],
        changed_result.loc[earlier, HISTORICAL_FEATURES],
    )
