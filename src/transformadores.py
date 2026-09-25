"""Custom sklearn transformers saved inside the model bundle.

They live in their own module so joblib can unpickle the model from any entry
point (training runs as a script, prediction imports the bundle later).
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RellenadorNAcategoricas(BaseEstimator, TransformerMixin):
    """Fill missing values only in the categorical columns used by CatBoost."""

    def __init__(self, categorical_columns: list[str] | tuple[str, ...]):
        self.categorical_columns = categorical_columns

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "RellenadorNAcategoricas":
        """Record categorical columns that are present in the input frame."""

        if not isinstance(x, pd.DataFrame):
            raise TypeError("RellenadorNAcategoricas requiere un DataFrame de pandas.")
        self.columns_ = [column for column in self.categorical_columns if column in x.columns]
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with categorical missing values represented as ``NA``."""

        if not isinstance(x, pd.DataFrame):
            raise TypeError("RellenadorNAcategoricas requiere un DataFrame de pandas.")
        result = x.copy()
        for column in self.columns_:
            result[column] = result[column].astype(object).where(result[column].notna(), "NA")
        return result
