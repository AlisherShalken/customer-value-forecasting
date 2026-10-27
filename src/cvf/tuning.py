"""Hyper-parameter search with Optuna on purged inner folds.

Inner validation snapshots all have label windows that close before the first
backtest cutoff, so tuning never sees data used to report test metrics.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss, mean_tweedie_deviance

from cvf.config import Settings
from cvf.models.gbm import Task, train_gbm
from cvf.splits import eligible_snapshots
from cvf.training import TASKS, tuned_params_path

log = logging.getLogger(__name__)

N_INNER_FOLDS = 3
REFERENCE_TWEEDIE_POWER = 1.4  # fixed so trials with different powers are comparable


def inner_folds(df: pd.DataFrame, settings: Settings) -> list[tuple[list[date], date]]:
    horizon = settings.snapshots.horizon_days
    dates = sorted(df["snapshot_date"].unique())
    first_cutoff = min(settings.evaluation.backtest_cutoffs)
    candidates = eligible_snapshots(dates, first_cutoff, horizon)[-N_INNER_FOLDS:]
    folds = [(eligible_snapshots(dates, v, horizon), v) for v in candidates]
    return [(fit, v) for fit, v in folds if fit]


def search_space(trial: optuna.Trial, task: Task) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "binary" if task == "propensity" else "tweedie",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "n_estimators": 3000,
    }
    if task == "revenue":
        params["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
    return params


def _score(task: Task, y: np.ndarray, pred: np.ndarray) -> float:
    if task == "propensity":
        return log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6))
    return mean_tweedie_deviance(y, np.clip(pred, 1e-3, None), power=REFERENCE_TWEEDIE_POWER)


def tune_task(df: pd.DataFrame, settings: Settings, task: Task,
              n_trials: int) -> optuna.Study:
    folds = inner_folds(df, settings)
    target = "purchased" if task == "propensity" else "revenue"

    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial, task)
        scores, trees = [], []
        for fit_dates, valid_date in folds:
            fit_df = df[df["snapshot_date"].isin(fit_dates)]
            valid_df = df[df["snapshot_date"] == valid_date]
            model = train_gbm(task, fit_df, valid_df, params, seed=settings.training.seed,
                              early_stopping_rounds=settings.training.early_stopping_rounds)
            pred = model.booster.predict(model.preprocessor.transform(valid_df),
                                         num_iteration=model.best_iteration)
            scores.append(_score(task, valid_df[target].to_numpy(), pred))
            trees.append(model.best_iteration)
        trial.set_user_attr("mean_trees", float(np.mean(trees)))
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=settings.training.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=f"{task}-tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("%s: best inner-CV score %.4f with %s", task, study.best_value, study.best_params)
    return study


def baseline_score(df: pd.DataFrame, settings: Settings, task: Task) -> float:
    """Inner-CV score of the configured defaults, to decide whether tuning helped."""
    folds = inner_folds(df, settings)
    target = "purchased" if task == "propensity" else "revenue"
    scores = []
    for fit_dates, valid_date in folds:
        fit_df = df[df["snapshot_date"].isin(fit_dates)]
        valid_df = df[df["snapshot_date"] == valid_date]
        model = train_gbm(task, fit_df, valid_df, getattr(settings.training, task),
                          seed=settings.training.seed,
                          early_stopping_rounds=settings.training.early_stopping_rounds)
        pred = model.booster.predict(model.preprocessor.transform(valid_df),
                                     num_iteration=model.best_iteration)
        scores.append(_score(task, valid_df[target].to_numpy(), pred))
    return float(np.mean(scores))


def run_tuning(df: pd.DataFrame, settings: Settings,
               n_trials: int | None = None) -> dict[str, dict[str, Any]]:
    """Tune both tasks; keep tuned params only where they beat the defaults."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = n_trials or settings.training.n_trials
    result: dict[str, dict[str, Any]] = {}
    summary: dict[str, dict[str, float]] = {}
    for task in TASKS:
        if task not in settings.training.tune_tasks:
            continue
        default = baseline_score(df, settings, task)
        study = tune_task(df, settings, task, n_trials)
        summary[task] = {"default": default, "tuned": study.best_value}
        if study.best_value < default:
            result[task] = {**study.best_params, "bagging_freq": 1, "n_estimators": 3000}
        log.info("%s inner-CV: default %.4f, tuned %.4f", task, default, study.best_value)

    path = tuned_params_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (settings.reports_dir / "tuning_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return result
