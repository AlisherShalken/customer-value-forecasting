"""Reference models the GBMs have to beat to justify their complexity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from cvf.features import TARGET_CLASS

RFM_COLUMNS = ["recency_days", "n_orders_365d", "revenue_365d", "n_orders", "tenure_days"]


def _log1p_clipped(X: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(X, 0, None))


class RFMLogit:
    """Logistic regression on log-scaled recency/frequency/monetary features."""

    name = "rfm_logit"

    def __init__(self) -> None:
        self.pipeline = make_pipeline(
            FunctionTransformer(_log1p_clipped),
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )

    def fit(self, df: pd.DataFrame) -> RFMLogit:
        self.pipeline.fit(df[RFM_COLUMNS].to_numpy(), df[TARGET_CLASS].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(df[RFM_COLUMNS].to_numpy())[:, 1]


class RunRateRevenue:
    """Next-quarter revenue = average quarterly revenue over the last year."""

    name = "run_rate"

    def fit(self, df: pd.DataFrame) -> RunRateRevenue:
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["revenue_365d"].to_numpy() / 4.0


class LastQuarterRevenue:
    """Next-quarter revenue = revenue of the previous 90 days."""

    name = "last_quarter"

    def fit(self, df: pd.DataFrame) -> LastQuarterRevenue:
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["revenue_90d"].to_numpy()
