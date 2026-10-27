"""Batch and single-customer scoring with reason codes."""

from __future__ import annotations

import json
import logging
from datetime import date

import duckdb
import numpy as np
import pandas as pd

from cvf.config import Settings
from cvf.features import features_as_of
from cvf.monitoring import drift_report
from cvf.policy import segment, select_contacts
from cvf.registry import ModelBundle

log = logging.getLogger(__name__)

N_REASONS = 3


def reason_codes(contrib: pd.DataFrame, features: pd.DataFrame,
                 n: int = N_REASONS) -> list[list[dict]]:
    """Top drivers per row: feature, its value and SHAP contribution (log-odds)."""
    values = contrib.drop(columns="bias")
    order = np.argsort(-values.abs().to_numpy(), axis=1)[:, :n]
    out = []
    for i, cols in enumerate(order):
        row = []
        for j in cols:
            name = values.columns[j]
            raw = features.iloc[i][name]
            row.append({
                "feature": name,
                "value": None if pd.isna(raw) else (raw if isinstance(raw, str) else float(raw)),
                "impact": round(float(values.iat[i, j]), 4),
            })
        out.append(row)
    return out


def score_frame(features: pd.DataFrame, bundle: ModelBundle,
                settings: Settings) -> pd.DataFrame:
    """Scores for any set of customers, from a single API request to the full base."""
    propensity, revenue = bundle.models["propensity"], bundle.models["revenue"]
    p = propensity.predict(features)
    ev = revenue.predict(features)
    scores = features[["snapshot_date", "customer_id"]].copy()
    scores["p_purchase_90d"] = p
    scores["expected_revenue_90d"] = ev
    scores["segment"] = segment(p, features["revenue_365d"].to_numpy(),
                                bundle.metadata["high_value_revenue_365d"])
    scores["contact"] = select_contacts(ev, settings.policy)
    scores["drivers"] = [json.dumps(r) for r in
                         reason_codes(propensity.contributions(features), features)]
    scores["model_version"] = bundle.version
    return scores


def score_batch(con: duckdb.DuckDBPyConnection, bundle: ModelBundle, settings: Settings,
                as_of: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the active base as of `as_of` and upsert into mart.customer_scores."""
    features = features_as_of(con, settings, as_of)
    if features.empty:
        raise ValueError(f"No active customers as of {as_of}")
    scores = score_frame(features, bundle, settings)
    # Decile 1 = highest expected revenue within this scoring run.
    rank = scores["expected_revenue_90d"].rank(method="first", ascending=False)
    scores["value_decile"] = pd.qcut(rank, 10, labels=False).astype(int) + 1
    drift = drift_report(bundle.metadata["reference_profile"], features,
                         settings.monitoring.psi_warn, settings.monitoring.psi_alert)

    con.execute("CREATE SCHEMA IF NOT EXISTS mart")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS mart.customer_scores (
            snapshot_date DATE, customer_id BIGINT, p_purchase_90d DOUBLE,
            expected_revenue_90d DOUBLE, value_decile INTEGER, segment VARCHAR,
            contact BOOLEAN, drivers VARCHAR, model_version VARCHAR,
            PRIMARY KEY (snapshot_date, customer_id)
        )
        """
    )
    con.execute("DELETE FROM mart.customer_scores WHERE snapshot_date = ?", [as_of])
    con.register("scores_df", scores)
    con.execute(
        """
        INSERT INTO mart.customer_scores
        SELECT snapshot_date, customer_id, p_purchase_90d, expected_revenue_90d, value_decile,
               segment, contact, drivers, model_version
        FROM scores_df
        """
    )
    con.unregister("scores_df")

    alerts = drift[drift["status"] == "alert"]
    if not alerts.empty:
        log.warning("Feature drift alert as of %s: %s", as_of, ", ".join(alerts["feature"]))
    log.info("Scored %d customers as of %s; %d selected for contact",
             len(scores), as_of, int(scores["contact"].sum()))
    return scores, drift
