"""Feature store and label construction on top of the staging layer."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import duckdb
import pandas as pd

from cvf.config import Settings
from cvf.db import render_sql, run_script

log = logging.getLogger(__name__)

WINDOWS = (30, 90, 180, 365)
KEYS = ["snapshot_date", "customer_id"]
CATEGORICAL = ["country"]
TARGET_CLASS = "purchased"
TARGET_REG = "revenue"


def feature_columns() -> list[str]:
    windowed = [f"{kind}_{w}d" for w in WINDOWS for kind in ("n_orders", "revenue")]
    return [
        "country",
        "snapshot_month",
        "recency_days",
        "tenure_days",
        "n_orders",
        *windowed,
        "n_orders_same_period_ly",
        "revenue_same_period_ly",
        "gross_revenue",
        "cancel_rate",
        "n_cancellations",
        "avg_order_value",
        "max_order_value",
        "avg_products_per_order",
        "avg_units_per_order",
        "active_months_365d",
        "n_purchase_days",
        "mean_gap_days",
        "std_gap_days",
        "overdue_ratio",
        "revenue_share_90d",
        "n_distinct_products_365d",
        "avg_unit_price_365d",
        "median_line_quantity_365d",
    ]


def snapshot_dates(first: date, last: date) -> list[date]:
    """Month starts from `first` to `last` inclusive."""
    out, d = [], first.replace(day=1)
    while d <= last:
        if d >= first:
            out.append(d)
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return out


def build_staging(con: duckdb.DuckDBPyConnection) -> None:
    run_script(con, "staging.sql")


def features_sql(settings: Settings, single_customer: bool = False,
                 snapshots: str = "feat.snapshots") -> str:
    return render_sql(
        "features.sql",
        snapshots=snapshots,
        windows=WINDOWS,
        horizon_days=settings.snapshots.horizon_days,
        active_window_days=settings.snapshots.active_window_days,
        single_customer=single_customer,
    )


def build_feature_store(con: duckdb.DuckDBPyConnection, settings: Settings) -> pd.DataFrame:
    """Materialise feat.customer_features and feat.labels for configured snapshots."""
    cfg = settings.snapshots
    dates = snapshot_dates(cfg.first, cfg.last)

    (data_end,) = con.execute("SELECT max(invoice_ts) FROM stg.orders").fetchone()
    last_label_end = dates[-1] + timedelta(days=cfg.horizon_days)
    if last_label_end > data_end.date() + timedelta(days=1):
        raise ValueError(
            f"Label window of snapshot {dates[-1]} ends {last_label_end}, after the last "
            f"available transaction ({data_end:%Y-%m-%d}). Move snapshots.last earlier."
        )

    con.execute("CREATE SCHEMA IF NOT EXISTS feat")
    con.execute("CREATE OR REPLACE TABLE feat.snapshots (snapshot_date DATE)")
    con.executemany("INSERT INTO feat.snapshots VALUES (?)", [(d,) for d in dates])

    con.execute(
        "CREATE OR REPLACE TABLE feat.customer_features AS "
        + features_sql(settings)
    )
    con.execute(
        "CREATE OR REPLACE TABLE feat.labels AS "
        + render_sql("labels.sql", population="feat.customer_features",
                     horizon_days=cfg.horizon_days)
    )
    summary = con.execute(
        """
        SELECT f.snapshot_date,
               count(*)                 AS customers,
               avg(l.purchased::INT)    AS purchase_rate,
               sum(l.revenue)           AS revenue
        FROM feat.customer_features f
        JOIN feat.labels l USING (snapshot_date, customer_id)
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    log.info("Feature store built:\n%s", summary.to_string(index=False))
    return summary


def features_as_of(con: duckdb.DuckDBPyConnection, settings: Settings, as_of: date,
                   customer_id: int | None = None) -> pd.DataFrame:
    """Compute features for an arbitrary date, optionally for one customer only."""
    sql = features_sql(
        settings,
        single_customer=customer_id is not None,
        snapshots="(SELECT CAST($as_of AS DATE) AS snapshot_date)",
    )
    params: dict[str, object] = {"as_of": as_of}
    if customer_id is not None:
        params["customer_id"] = customer_id
    df = con.execute(sql, params).df()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    return df


def load_dataset(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT f.*, l.purchased, l.revenue
        FROM feat.customer_features f
        JOIN feat.labels l USING (snapshot_date, customer_id)
        ORDER BY snapshot_date, customer_id
        """
    ).df()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    df[TARGET_CLASS] = df[TARGET_CLASS].astype(int)
    return df
