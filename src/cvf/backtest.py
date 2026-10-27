"""Rolling-origin backtest of the production training procedure and baselines."""

from __future__ import annotations

import logging

import pandas as pd

from cvf.config import Settings
from cvf.evaluation import classification_metrics, regression_metrics
from cvf.features import TARGET_CLASS, TARGET_REG
from cvf.models.baselines import LastQuarterRevenue, RFMLogit, RunRateRevenue
from cvf.models.gbm import GBMModel, Task
from cvf.training import fit_models, split_for

log = logging.getLogger(__name__)


def run_backtest(df: pd.DataFrame, settings: Settings
                 ) -> tuple[pd.DataFrame, pd.DataFrame, dict[Task, GBMModel]]:
    """Returns metrics in long format, out-of-time predictions for every cutoff
    and the models fitted for the last cutoff (used for holdout explanations)."""
    shares = settings.evaluation.top_k_shares
    metric_rows, predictions = [], []

    for cutoff in settings.evaluation.backtest_cutoffs:
        split = split_for(df, settings, cutoff)
        train, test = split.frame(df, "train"), split.frame(df, "test")
        if test.empty:
            raise ValueError(f"No snapshot at backtest cutoff {cutoff}")

        models = fit_models(df, split, settings)
        preds = {
            ("propensity", "lightgbm"): models["propensity"].predict(test),
            ("propensity", RFMLogit.name): RFMLogit().fit(train).predict(test),
            ("revenue", "lightgbm"): models["revenue"].predict(test),
            ("revenue", RunRateRevenue.name): RunRateRevenue().predict(test),
            ("revenue", LastQuarterRevenue.name): LastQuarterRevenue().predict(test),
        }

        for (task, model), pred in preds.items():
            if task == "propensity":
                metrics = classification_metrics(test[TARGET_CLASS].to_numpy(), pred, shares)
            else:
                metrics = regression_metrics(test[TARGET_REG].to_numpy(), pred, shares)
            metric_rows += [
                {"cutoff": cutoff, "task": task, "model": model, "metric": k, "value": v}
                for k, v in metrics.items()
            ]

        fold = test[["snapshot_date", "customer_id", TARGET_CLASS, TARGET_REG,
                     "revenue_365d"]].copy()
        fold["p_purchase"] = preds[("propensity", "lightgbm")]
        fold["expected_revenue"] = preds[("revenue", "lightgbm")]
        fold["p_purchase_raw"] = models["propensity"].predict_raw(test)
        fold["p_rfm_logit"] = preds[("propensity", RFMLogit.name)]
        fold["revenue_run_rate"] = preds[("revenue", RunRateRevenue.name)]
        fold["revenue_last_quarter"] = preds[("revenue", LastQuarterRevenue.name)]
        predictions.append(fold)
        log.info("Backtest cutoff %s: train=%d rows (%d snapshots), test=%d rows",
                 cutoff, len(train), len(split.train), len(test))

    return pd.DataFrame(metric_rows), pd.concat(predictions, ignore_index=True), models


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of each metric across backtest cutoffs."""
    return (metrics.groupby(["task", "metric", "model"])["value"]
            .agg(["mean", "std", "min", "max"])
            .reset_index())
