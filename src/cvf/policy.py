"""Turn model scores into a contact list for the retention/upsell campaign.

Economics per contacted customer:
    incremental margin = assumed_uplift * expected_revenue * gross_margin
    profit             = incremental margin - contact_cost
A customer is worth contacting when expected_revenue exceeds
contact_cost / (assumed_uplift * gross_margin).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cvf.config import PolicyConfig


def break_even_revenue(policy: PolicyConfig) -> float:
    return policy.contact_cost / (policy.assumed_uplift * policy.gross_margin)


def select_contacts(expected_revenue: np.ndarray, policy: PolicyConfig) -> np.ndarray:
    return np.asarray(expected_revenue) > break_even_revenue(policy)


def profit_curve(expected_revenue: np.ndarray, realised_revenue: np.ndarray | None,
                 policy: PolicyConfig, n_points: int = 100) -> pd.DataFrame:
    """Cumulative campaign profit when contacting the top-k customers by expected revenue.

    `expected_profit` uses model predictions; `realised_profit` replaces them
    with the revenue the customers actually generated (backtest only).
    """
    order = np.argsort(-np.asarray(expected_revenue), kind="stable")
    k_values = np.unique(np.linspace(1, len(order), n_points).astype(int))
    rate = policy.assumed_uplift * policy.gross_margin

    cum_expected = np.cumsum(np.asarray(expected_revenue)[order])
    rows = {
        "contacted": k_values,
        "share": k_values / len(order),
        "expected_profit": rate * cum_expected[k_values - 1] - policy.contact_cost * k_values,
    }
    if realised_revenue is not None:
        cum_real = np.cumsum(np.asarray(realised_revenue)[order])
        rows["realised_profit"] = rate * cum_real[k_values - 1] - policy.contact_cost * k_values
    return pd.DataFrame(rows)


def high_value_threshold(revenue_365d: np.ndarray) -> float:
    """Top-quartile trailing-year revenue; fixed at training time and stored with the model."""
    return float(np.quantile(revenue_365d, 0.75))


def segment(p_purchase: np.ndarray, revenue_365d: np.ndarray,
            high_value_revenue: float) -> np.ndarray:
    """CRM segments combining historical value and forward-looking purchase risk."""
    high_value = np.asarray(revenue_365d) >= high_value_revenue
    likely = np.asarray(p_purchase) >= 0.5
    return np.select(
        [high_value & likely, high_value & ~likely, ~high_value & likely],
        ["core", "at_risk_high_value", "active"],
        default="dormant",
    )
