from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cvf.features import CATEGORICAL, feature_columns

OTHER = "__other__"


@dataclass
class Preprocessor:
    """Fixes column order and categorical levels learned on the training set.

    Rare levels and levels unseen at training time are mapped to one bucket,
    so the model behaves deterministically on new countries.
    """

    min_level_count: int = 50
    features: list[str] = field(default_factory=feature_columns)
    levels: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        for col in CATEGORICAL:
            counts = df[col].value_counts()
            self.levels[col] = sorted(counts[counts >= self.min_level_count].index) + [OTHER]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.features) - set(df.columns)
        if missing:
            raise KeyError(f"Missing feature columns: {sorted(missing)}")
        X = df[self.features].copy()
        for col in CATEGORICAL:
            known = X[col].where(X[col].isin(self.levels[col]), OTHER)
            X[col] = pd.Categorical(known, categories=self.levels[col])
        numeric = [c for c in self.features if c not in CATEGORICAL]
        X[numeric] = X[numeric].astype(np.float64)
        return X
