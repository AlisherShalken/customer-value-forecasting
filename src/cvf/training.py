"""Shared training entry points used by the backtest, tuning and production fit."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from cvf.config import Settings
from cvf.models.gbm import GBMModel, Task, train_gbm
from cvf.splits import TimeSplit, make_split

log = logging.getLogger(__name__)

TASKS: tuple[Task, ...] = ("propensity", "revenue")


def tuned_params_path(settings: Settings) -> Path:
    return settings.training.tuned_params


def model_params(settings: Settings, task: Task) -> dict[str, Any]:
    """Config defaults overridden by the latest Optuna result, if one exists."""
    params = dict(getattr(settings.training, task))
    path = tuned_params_path(settings)
    if path.exists():
        params |= json.loads(path.read_text(encoding="utf-8")).get(task, {})
    return params


def split_for(df: pd.DataFrame, settings: Settings, cutoff: date) -> TimeSplit:
    dates = sorted(df["snapshot_date"].unique())
    return make_split(dates, cutoff, settings.snapshots.horizon_days)


def production_cutoff(df: pd.DataFrame, settings: Settings) -> date:
    """Latest cutoff for which every snapshot's label window is complete."""
    last = max(df["snapshot_date"])
    return last + timedelta(days=settings.snapshots.horizon_days)


def fit_models(df: pd.DataFrame, split: TimeSplit, settings: Settings,
               params: dict[Task, dict[str, Any]] | None = None) -> dict[Task, GBMModel]:
    models = {}
    for task in TASKS:
        models[task] = train_gbm(
            task,
            fit_df=split.frame(df, "fit"),
            valid_df=split.frame(df, "valid"),
            params=(params or {}).get(task) or model_params(settings, task),
            seed=settings.training.seed,
            early_stopping_rounds=settings.training.early_stopping_rounds,
            refit_df=split.frame(df, "train"),
        )
        log.info("%s @ %s: %d trees, valid %s=%.4f", task, split.cutoff,
                 models[task].best_iteration, task, models[task].valid_score)
    return models
