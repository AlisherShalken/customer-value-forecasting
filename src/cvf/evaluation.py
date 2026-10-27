"""Model quality metrics, including ranking metrics used by the campaign team."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    mean_tweedie_deviance,
    roc_auc_score,
)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean |observed rate - mean prediction| over equal-frequency bins."""
    bins = pd.qcut(p, q=n_bins, labels=False, duplicates="drop")
    frame = pd.DataFrame({"y": y, "p": p, "bin": bins})
    grouped = frame.groupby("bin").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((grouped["n"] * (grouped["y"] - grouped["p"]).abs()).sum() / len(frame))


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = pd.qcut(p, q=n_bins, labels=False, duplicates="drop")
    return (pd.DataFrame({"y": y, "p": p, "bin": bins})
            .groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
            .reset_index(drop=True))


def top_k_mask(score: np.ndarray, share: float) -> np.ndarray:
    k = max(1, int(round(len(score) * share)))
    idx = np.argsort(-score, kind="stable")[:k]
    mask = np.zeros(len(score), dtype=bool)
    mask[idx] = True
    return mask


def lift_at(y: np.ndarray, score: np.ndarray, share: float) -> float:
    base = y.mean()
    return float(y[top_k_mask(score, share)].mean() / base) if base > 0 else float("nan")


def capture_at(value: np.ndarray, score: np.ndarray, share: float) -> float:
    """Share of total realised value held by the top `share` of customers by score."""
    total = value.sum()
    return float(value[top_k_mask(score, share)].sum() / total) if total > 0 else float("nan")


def normalized_gini(value: np.ndarray, score: np.ndarray) -> float:
    """Gini of value ordered by score, normalised by the Gini of a perfect ordering."""

    def gini(order_by: np.ndarray) -> float:
        order = np.argsort(-order_by, kind="stable")
        cum = np.cumsum(value[order]) / value.sum()
        return float(cum.mean() - (np.arange(1, len(value) + 1) / len(value)).mean())

    return gini(score) / gini(value)


def classification_metrics(y: np.ndarray, p: np.ndarray,
                           top_k_shares: list[float]) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "base_rate": float(y.mean()),
        "mean_pred": float(p.mean()),
    }
    for s in top_k_shares:
        out[f"lift@{int(s * 100)}%"] = lift_at(y, p, s)
    return out


def regression_metrics(y: np.ndarray, pred: np.ndarray,
                       top_k_shares: list[float]) -> dict[str, float]:
    pred = np.clip(pred, 0, None)
    out = {
        "mae": mean_absolute_error(y, pred),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "tweedie_dev_1.4": mean_tweedie_deviance(y, np.clip(pred, 1e-3, None), power=1.4),
        "spearman": float(spearmanr(y, pred).statistic),
        "norm_gini": normalized_gini(y, pred),
        "total_ratio": float(pred.sum() / y.sum()),
    }
    for s in top_k_shares:
        out[f"capture@{int(s * 100)}%"] = capture_at(y, pred, s)
    return out
