"""Staging rules, point-in-time features and label windows on hand-built data."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from cvf.db import render_sql
from cvf.features import build_staging, features_as_of
from tests.conftest import line, load_raw, make_settings

SNAPSHOT = date(2011, 6, 1)


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


def build(con, rows):
    load_raw(con, rows)
    build_staging(con)


def history_rows():
    return [
        # customer 1: three orders, one partial cancellation
        line("1001", 1, "2011-01-10 10:00", 100.0),
        line("1002", 1, "2011-03-12 10:00", 200.0),
        line("1003", 1, "2011-05-11 10:00", 50.0),
        line("1003", 1, "2011-05-11 10:00", 30.0, stock_code="85123A"),
        line("C1004", 1, "2011-05-20 10:00", -40.0, quantity=-1),
        # customer 2: last purchase more than a year before the snapshot
        line("2001", 2, "2010-05-01 10:00", 500.0),
    ]


def test_staging_rules(con):
    rows = [
        line("1", 1, "2010-12-05 09:00", 10.0, sheet="Year 2009-2010"),  # sheet overlap
        line("1", 1, "2010-12-05 09:00", 10.0),
        line("2", 1, "2011-01-01 09:00", 10.0),
        line("2", 1, "2011-01-01 09:00", 10.0),                          # exact duplicate
        line("3", None, "2011-01-01 09:00", 10.0),                       # no customer
        line("4", 1, "2011-01-01 09:00", 10.0, stock_code="POST"),       # postage
        line("5", 1, "2011-01-01 09:00", 10.0, stock_code="M"),          # manual adjustment
        line("6", 1, "2011-01-01 09:00", 10.0, stock_code="DCGS0058"),   # web-shop product
        {**line("7", 1, "2011-01-01 09:00", 10.0), "unit_price": 0.0},   # stock adjustment
    ]
    build(con, rows)
    kept = con.execute("SELECT invoice FROM stg.order_lines ORDER BY invoice").df()
    assert kept["invoice"].tolist() == ["1", "2", "6"]


def test_features_hand_computed(con, settings):
    build(con, history_rows())
    f = features_as_of(con, settings, SNAPSHOT).set_index("customer_id")

    assert list(f.index) == [1], "customer 2 is outside the active window"
    c = f.loc[1]
    assert c["recency_days"] == (SNAPSHOT - date(2011, 5, 11)).days
    assert c["tenure_days"] == (SNAPSHOT - date(2011, 1, 10)).days
    assert c["n_orders"] == 3
    assert c["n_orders_30d"] == 1 and c["revenue_30d"] == pytest.approx(80.0)
    assert c["revenue_90d"] == pytest.approx(280.0)
    assert c["gross_revenue"] == pytest.approx(380.0)
    assert c["n_cancellations"] == 1
    assert c["cancel_rate"] == pytest.approx(40.0 / 380.0)
    assert c["mean_gap_days"] == pytest.approx((61 + 60) / 2)
    assert c["overdue_ratio"] == pytest.approx(21 / 60.5)
    assert c["n_distinct_products_365d"] == 2
    assert c["active_months_365d"] == 3


def test_features_ignore_future_data(con, settings):
    build(con, history_rows())
    before = features_as_of(con, settings, SNAPSHOT)

    future = [
        line("9001", 1, "2011-06-01 00:00", 999.0),     # exactly at the snapshot
        line("9002", 1, "2011-08-01 12:00", 999.0),
        line("C9003", 1, "2011-07-01 12:00", -50.0, quantity=-1),
        line("9004", 3, "2011-06-15 12:00", 999.0),     # customer appearing later
    ]
    build(con, history_rows() + future)
    after = features_as_of(con, settings, SNAPSHOT)

    pd.testing.assert_frame_equal(before, after)


def test_single_customer_matches_batch(con, settings):
    build(con, history_rows())
    batch = features_as_of(con, settings, SNAPSHOT)
    single = features_as_of(con, settings, SNAPSHOT, customer_id=1)
    pd.testing.assert_frame_equal(batch[batch["customer_id"] == 1].reset_index(drop=True), single)


def test_label_window_boundaries(con):
    rows = [
        line("1", 1, "2011-05-01 10:00", 10.0),
        line("2", 1, "2011-06-01 00:00", 100.0),                    # first instant: in
        line("C3", 1, "2011-06-10 00:00", -30.0, quantity=-1),
        line("4", 1, "2011-08-30 00:00", 1000.0),                   # snapshot + 90d: out
        line("5", 2, "2011-05-01 10:00", 10.0),
        line("C6", 2, "2011-06-05 00:00", -10.0, quantity=-1),      # only a cancellation
    ]
    build(con, rows)
    con.execute("CREATE TABLE pop AS SELECT * FROM (VALUES (DATE '2011-06-01', 1::BIGINT),"
                " (DATE '2011-06-01', 2::BIGINT)) t(snapshot_date, customer_id)")
    labels = con.execute(render_sql("labels.sql", population="pop", horizon_days=90)).df()
    labels = labels.set_index("customer_id")

    assert bool(labels.loc[1, "purchased"]) and labels.loc[1, "revenue"] == pytest.approx(70.0)
    assert not bool(labels.loc[2, "purchased"]) and labels.loc[2, "revenue"] == 0.0
