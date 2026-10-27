"""Population stability of features between training data and a scoring batch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cvf.features import CATEGORICAL

N_BINS = 10
EPS = 1e-4
# Calendar features shift by construction between any two snapshots.
NOT_MONITORED = {"snapshot_month"}


def reference_profile(df: pd.DataFrame, features: list[str]) -> dict[str, dict]:
    """Bin edges and bin shares of the training distribution, stored with the model."""
    profile: dict[str, dict] = {}
    for col in features:
        if col in NOT_MONITORED:
            continue
        if col in CATEGORICAL:
            shares = df[col].fillna("__missing__").value_counts(normalize=True)
            profile[col] = {"kind": "categorical", "shares": shares.to_dict()}
            continue
        values = df[col].to_numpy(dtype=float)
        finite = values[~np.isnan(values)]
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, N_BINS + 1))[1:-1])
        profile[col] = {
            "kind": "numeric",
            "edges": edges.tolist(),
            "shares": _numeric_shares(values, edges).tolist(),
        }
    return profile


def _numeric_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # Bin len(edges)+1 holds missing values so a jump in nulls is also caught.
    bins = np.where(np.isnan(values), len(edges) + 1, np.searchsorted(edges, values, side="right"))
    return np.bincount(bins, minlength=len(edges) + 2) / len(values)


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(profile: dict[str, dict], df: pd.DataFrame,
                 warn: float = 0.10, alert: float = 0.25) -> pd.DataFrame:
    rows = []
    for col, ref in profile.items():
        if ref["kind"] == "categorical":
            current = df[col].fillna("__missing__").value_counts(normalize=True)
            levels = sorted(set(ref["shares"]) | set(current.index))
            e = np.array([ref["shares"].get(k, 0.0) for k in levels])
            a = np.array([current.get(k, 0.0) for k in levels])
        else:
            e = np.asarray(ref["shares"])
            a = _numeric_shares(df[col].to_numpy(dtype=float), np.asarray(ref["edges"]))
        value = psi(e, a)
        status = "alert" if value >= alert else "warn" if value >= warn else "ok"
        rows.append({"feature": col, "psi": value, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
