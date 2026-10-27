"""Full pipeline on synthetic transactions: build -> train -> score -> API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cvf import pipeline
from cvf.db import connect
from tests.conftest import load_raw, make_settings, synthetic_transactions

FAST_GBM = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 7,
            "min_child_samples": 20, "n_estimators": 60}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    settings = make_settings(
        tmp,
        **{
            "snapshots.first": "2020-07-01",
            "snapshots.last": "2021-09-01",
            "evaluation.backtest_cutoffs": ["2021-08-01", "2021-09-01"],
            "evaluation.holdout_cutoff": "2021-09-01",
            "training.early_stopping_rounds": 10,
            "training.propensity": FAST_GBM,
            "training.revenue": {**FAST_GBM, "objective": "tweedie",
                                 "tweedie_variance_power": 1.4},
        },
    )
    with connect(settings.data.warehouse) as con:
        load_raw(con, synthetic_transactions())
    pipeline.build(settings)
    bundle = pipeline.train(settings)
    scores, drift = pipeline.score(settings)
    return settings, bundle, scores, drift


def test_training_registers_model_and_report(trained):
    settings, bundle, _, _ = trained
    assert (settings.artifacts_dir / "models" / "LATEST").read_text() == bundle.version
    assert 0.5 < bundle.metadata["holdout_metrics"]["propensity"]["roc_auc"] <= 1.0
    report = (settings.reports_dir / "model_report.md").read_text(encoding="utf-8")
    assert "Campaign policy on holdout" in report
    assert len(list((settings.reports_dir / "figures").glob("*.png"))) == 5


def test_batch_scores_are_valid(trained):
    _, _, scores, drift = trained
    assert scores["customer_id"].is_unique
    assert scores["p_purchase_90d"].between(0, 1).all()
    assert (scores["expected_revenue_90d"] >= 0).all()
    assert set(scores["value_decile"]) == set(range(1, 11))
    assert not drift.empty


def test_api_matches_batch(trained, monkeypatch):
    settings, bundle, scores, _ = trained
    from cvf.api import app as api_module

    monkeypatch.setattr(api_module, "get_settings", lambda: settings)
    row = scores.sort_values("expected_revenue_90d", ascending=False).iloc[0]

    with TestClient(api_module.app) as client:
        assert client.get("/health").json()["model_version"] == bundle.version

        online = client.get(f"/customers/{row['customer_id']}/score").json()
        assert online["p_purchase_90d"] == pytest.approx(row["p_purchase_90d"])
        assert online["expected_revenue_90d"] == pytest.approx(row["expected_revenue_90d"])
        assert len(online["drivers"]) == 3

        top = client.get("/scores", params={"limit": 5}).json()
        assert top[0]["customer_id"] == row["customer_id"]

        assert client.get("/customers/1/score").status_code == 404
        assert client.get("/scores", params={"as_of": "2099-01-01"}).status_code == 422
