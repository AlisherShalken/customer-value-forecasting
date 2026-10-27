from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from cvf.config import PolicyConfig
from cvf.evaluation import (
    capture_at,
    expected_calibration_error,
    lift_at,
    normalized_gini,
)
from cvf.features import snapshot_dates
from cvf.monitoring import drift_report, psi, reference_profile
from cvf.policy import break_even_revenue, profit_curve, segment, select_contacts
from cvf.splits import make_split

POLICY = PolicyConfig(contact_cost=10.0, gross_margin=0.3, assumed_uplift=0.1)


def test_snapshot_dates_month_starts():
    assert snapshot_dates(date(2010, 11, 1), date(2011, 2, 1)) == [
        date(2010, 11, 1), date(2010, 12, 1), date(2011, 1, 1), date(2011, 2, 1)]


def test_split_purges_overlapping_label_windows():
    dates = snapshot_dates(date(2010, 1, 1), date(2011, 9, 1))
    split = make_split(dates, cutoff=date(2011, 9, 1), horizon_days=90)

    assert all(d + timedelta(days=90) <= split.cutoff for d in split.train)
    assert all(d + timedelta(days=90) <= split.valid[0] for d in split.fit)
    assert split.valid == [date(2011, 6, 1)]
    assert split.fit[-1] == date(2011, 3, 1)
    assert split.test == [date(2011, 9, 1)]


def test_split_requires_history():
    with pytest.raises(ValueError):
        make_split([date(2011, 1, 1), date(2011, 2, 1)], date(2011, 4, 1), 90)


def test_ece_near_zero_when_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = rng.uniform(size=p.size) < p
    assert expected_calibration_error(y.astype(int), p) < 0.01
    assert expected_calibration_error(y.astype(int), np.clip(p + 0.2, 0, 1)) > 0.1


def test_ranking_metrics():
    value = np.array([0, 0, 10, 0, 90], dtype=float)
    perfect = value.copy()
    assert normalized_gini(value, perfect) == pytest.approx(1.0)
    assert capture_at(value, perfect, 0.2) == pytest.approx(0.9)
    assert capture_at(value, -perfect, 0.4) == pytest.approx(0.0)
    y = (value > 0).astype(int)
    assert lift_at(y, perfect, 0.4) == pytest.approx(1.0 / 0.4)


def test_policy_threshold_and_curve():
    assert break_even_revenue(POLICY) == pytest.approx(10.0 / 0.03)
    ev = np.array([1000.0, 500.0, 100.0])
    assert select_contacts(ev, POLICY).tolist() == [True, True, False]

    curve = profit_curve(ev, realised_revenue=ev, policy=POLICY, n_points=3)
    expected = [0.03 * 1000 - 10, 0.03 * 1500 - 20, 0.03 * 1600 - 30]
    assert curve["expected_profit"].tolist() == pytest.approx(expected)
    assert curve["realised_profit"].tolist() == pytest.approx(expected)


def test_segments():
    seg = segment(np.array([0.9, 0.1, 0.9, 0.1]), np.array([500, 500, 10, 10]), 100.0)
    assert seg.tolist() == ["core", "at_risk_high_value", "active", "dormant"]


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.normal(size=5000), "country": ["UK"] * 5000})
    same = pd.DataFrame({"x": rng.normal(size=5000), "country": ["UK"] * 5000})
    shifted = pd.DataFrame({"x": rng.normal(1.0, 1.0, size=5000),
                            "country": ["UK"] * 2500 + ["DE"] * 2500})
    profile = reference_profile(ref, ["x", "country"])

    assert psi(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0
    assert set(drift_report(profile, same)["status"]) == {"ok"}
    assert set(drift_report(profile, shifted)["status"]) == {"alert"}
