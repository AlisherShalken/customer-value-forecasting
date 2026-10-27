"""Ablations behind the modelling decisions; writes reports/experiments.md.

Every variant is evaluated with the same rolling-origin protocol as the
production backtest, so numbers are comparable with reports/model_report.md.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from cvf.config import Settings
from cvf.evaluation import classification_metrics, regression_metrics
from cvf.features import CATEGORICAL
from cvf.models.gbm import PlattCalibrator, train_gbm
from cvf.models.preprocess import Preprocessor
from cvf.training import model_params, split_for

log = logging.getLogger(__name__)

# Best revenue parameters found by `cvf tune` with tune_tasks: [revenue]. They
# won on the inner folds but not in the backtest, so production keeps defaults.
OPTUNA_REVENUE = {
    "objective": "tweedie", "tweedie_variance_power": 1.206, "learning_rate": 0.0327,
    "num_leaves": 85, "min_child_samples": 21, "feature_fraction": 0.499,
    "bagging_fraction": 0.926, "bagging_freq": 1, "lambda_l2": 0.0128,
    "min_split_gain": 0.981, "n_estimators": 3000,
}

PROPENSITY_COLS = ["roc_auc", "log_loss", "brier", "ece"]
REVENUE_COLS = ["norm_gini", "spearman", "capture@10%", "mae", "rmse", "total_ratio"]


def _folds(df: pd.DataFrame, settings: Settings):
    for cutoff in settings.evaluation.backtest_cutoffs:
        split = split_for(df, settings, cutoff)
        yield split, *(split.frame(df, p) for p in ("fit", "valid", "train", "test"))


def calibration_variants(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    rows = []
    params = model_params(settings, "propensity")
    for split, fit, valid, train, test in _folds(df, settings):
        model = train_gbm("propensity", fit, valid, params, seed=settings.training.seed,
                          early_stopping_rounds=settings.training.early_stopping_rounds,
                          refit_df=train)
        # Calibrators are fitted on predictions of the early-stopped model.
        early = train_gbm("propensity", fit, valid, params, seed=settings.training.seed,
                          early_stopping_rounds=settings.training.early_stopping_rounds)
        p_valid, y_valid = early.predict_raw(valid), valid["purchased"].to_numpy()
        raw = model.predict_raw(test)
        variants = {
            "raw": raw,
            "isotonic": IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                        .fit(p_valid, y_valid).predict(raw),
            "platt (production)": PlattCalibrator().fit(p_valid, y_valid).predict(raw),
        }
        for name, p in variants.items():
            m = classification_metrics(test["purchased"].to_numpy(), p, [0.1])
            rows.append({"cutoff": split.cutoff, "variant": name, **m})
    return pd.DataFrame(rows)


def _lgb_regression(params: dict, fit: pd.DataFrame, valid: pd.DataFrame,
                    train: pd.DataFrame, settings: Settings, target: Callable,
                    offset: Callable | None = None) -> Callable[[pd.DataFrame], np.ndarray]:
    """Early-stop on valid, refit on train; returns a raw-score predictor."""
    params = dict(params)
    n_estimators = params.pop("n_estimators", 3000)
    metric = {"tweedie": "tweedie", "gamma": "gamma", "regression": "l2"}[params["objective"]]
    params |= {"metric": metric, "seed": settings.training.seed, "verbosity": -1}
    prep = Preprocessor().fit(train)

    def dataset(d: pd.DataFrame, ref=None) -> lgb.Dataset:
        return lgb.Dataset(prep.transform(d), target(d), categorical_feature=CATEGORICAL,
                           init_score=None if offset is None else offset(d), reference=ref)

    dfit = dataset(fit)
    early = lgb.train(params, dfit, n_estimators, valid_sets=[dataset(valid, dfit)],
                      callbacks=[lgb.early_stopping(settings.training.early_stopping_rounds,
                                                    verbose=False)])
    booster = lgb.train(params, dataset(train), max(early.best_iteration, 1))
    return lambda d: booster.predict(prep.transform(d), raw_score=True) + (
        0 if offset is None else offset(d))


def revenue_variants(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    rows = []
    default = dict(settings.training.revenue)
    gamma = {k: v for k, v in default.items() if k != "tweedie_variance_power"}
    gamma["objective"] = "gamma"

    def run_rate_log(d: pd.DataFrame) -> np.ndarray:
        return np.log(np.maximum(d["revenue_365d"].to_numpy() / 4, 50.0))

    buyers = lambda d: d[(d["purchased"] == 1) & (d["revenue"] > 0)]  # noqa: E731
    revenue = lambda d: d["revenue"].to_numpy()  # noqa: E731

    for split, fit, valid, train, test in _folds(df, settings):
        prop = train_gbm("propensity", fit, valid, model_params(settings, "propensity"),
                         seed=settings.training.seed, refit_df=train,
                         early_stopping_rounds=settings.training.early_stopping_rounds)
        spend = _lgb_regression(gamma, buyers(fit), buyers(valid), buyers(train),
                                settings, revenue)
        variants = {
            "run-rate baseline": test["revenue_365d"].to_numpy() / 4,
            "tweedie p=1.4 (production)": np.exp(
                _lgb_regression(default, fit, valid, train, settings, revenue)(test)),
            "tweedie, Optuna inner-CV params": np.exp(
                _lgb_regression(OPTUNA_REVENUE, fit, valid, train, settings, revenue)(test)),
            "hurdle: P(buy) x gamma spend": prop.predict(test) * np.exp(spend(test)),
            "tweedie with run-rate offset": np.exp(
                _lgb_regression(default, fit, valid, train, settings, revenue,
                                offset=run_rate_log)(test)),
        }
        for name, pred in variants.items():
            m = regression_metrics(test["revenue"].to_numpy(), pred, [0.1])
            rows.append({"cutoff": split.cutoff, "variant": name, **m})
    return pd.DataFrame(rows)


def window_variants(df: pd.DataFrame, settings: Settings,
                    starts: tuple[date, ...] = (date(2010, 6, 1), date(2010, 12, 1))
                    ) -> pd.DataFrame:
    rows = []
    for start in starts:
        subset = df[df["snapshot_date"] >= start]
        for split, fit, valid, train, test in _folds(subset, settings):
            for task, cols in (("propensity", "purchased"), ("revenue", "revenue")):
                model = train_gbm(task, fit, valid, model_params(settings, task),
                                  seed=settings.training.seed, refit_df=train,
                                  early_stopping_rounds=settings.training.early_stopping_rounds)
                pred = model.predict(test)
                y = test[cols].to_numpy()
                m = (classification_metrics(y, pred, [0.1]) if task == "propensity"
                     else regression_metrics(y, pred, [0.1]))
                rows.append({"cutoff": split.cutoff, "variant": f"snapshots from {start}",
                             "task": task, **m})
    return pd.DataFrame(rows)


def _table(results: pd.DataFrame, cols: list[str]) -> str:
    return results.groupby("variant", sort=False)[cols].mean().round(4).to_markdown()


def run_experiments(df: pd.DataFrame, settings: Settings) -> str:
    calibration = calibration_variants(df, settings)
    revenue = revenue_variants(df, settings)
    window = window_variants(df, settings)
    text = "\n".join([
        "# Experiments",
        "",
        "Means over backtest cutoffs "
        f"{', '.join(map(str, settings.evaluation.backtest_cutoffs))}; "
        "same purged rolling-origin protocol as the production backtest.",
        "",
        "## Probability calibration",
        "",
        _table(calibration, PROPENSITY_COLS),
        "",
        "## Revenue model",
        "",
        _table(revenue, REVENUE_COLS),
        "",
        "## Training window start",
        "",
        "Propensity:",
        "",
        _table(window[window["task"] == "propensity"], PROPENSITY_COLS),
        "",
        "Revenue:",
        "",
        _table(window[window["task"] == "revenue"], REVENUE_COLS),
        "",
    ])
    path = settings.reports_dir / "experiments.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
