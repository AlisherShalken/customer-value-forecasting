"""LightGBM models for purchase propensity and 90-day revenue.

Training protocol (identical in backtest and production):
  1. fit on `fit` snapshots, early-stop on the purged `valid` snapshot;
  2. propensity only: fit a Platt calibrator (logistic regression on the
     model's log-odds) on validation predictions. Platt rescales log-odds, so
     the month effect learned by the trees is kept; isotonic regression fitted
     on a single month imposed that month's purchase rate on other seasons and
     calibrated worse in the backtest (README, "Experiments");
  3. refit on all eligible `train` snapshots with the selected number of
     trees, so the most recent behaviour is not wasted on validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cvf.features import CATEGORICAL, TARGET_CLASS, TARGET_REG
from cvf.models.preprocess import Preprocessor

Task = Literal["propensity", "revenue"]

TARGETS: dict[Task, str] = {"propensity": TARGET_CLASS, "revenue": TARGET_REG}
EVAL_METRIC: dict[Task, str] = {"propensity": "binary_logloss", "revenue": "tweedie"}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6)

    def fit(self, p: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        self.model.fit(_logit(p)[:, None], y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_logit(p)[:, None])[:, 1]


@dataclass
class GBMModel:
    task: Task
    params: dict[str, Any]
    preprocessor: Preprocessor
    booster: lgb.Booster
    best_iteration: int
    valid_score: float
    calibrator: PlattCalibrator | None = None

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self.preprocessor.transform(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.predict_raw(df)
        if self.calibrator is not None:
            pred = self.calibrator.predict(pred)
        return pred

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """TreeSHAP contributions in raw score space (log-odds / log-revenue)."""
        contrib = self.booster.predict(self.preprocessor.transform(df), pred_contrib=True)
        return pd.DataFrame(contrib, columns=[*self.preprocessor.features, "bias"], index=df.index)

    def feature_importance(self) -> pd.Series:
        gain = self.booster.feature_importance(importance_type="gain")
        return pd.Series(gain, index=self.preprocessor.features).sort_values(ascending=False)


def _split_params(params: dict[str, Any]) -> tuple[dict[str, Any], int]:
    params = dict(params)
    n_estimators = int(params.pop("n_estimators", 2000))
    return params, n_estimators


def _dataset(prep: Preprocessor, df: pd.DataFrame, target: str,
             reference: lgb.Dataset | None = None) -> lgb.Dataset:
    return lgb.Dataset(
        prep.transform(df),
        label=df[target].to_numpy(),
        categorical_feature=CATEGORICAL,
        reference=reference,
        free_raw_data=False,
    )


def train_gbm(task: Task, fit_df: pd.DataFrame, valid_df: pd.DataFrame,
              params: dict[str, Any], seed: int = 42, early_stopping_rounds: int = 100,
              refit_df: pd.DataFrame | None = None) -> GBMModel:
    target = TARGETS[task]
    lgb_params, n_estimators = _split_params(params)
    lgb_params |= {"metric": EVAL_METRIC[task], "seed": seed, "verbosity": -1,
                   "deterministic": True, "num_threads": 0}

    prep = Preprocessor().fit(fit_df if refit_df is None else refit_df)
    dfit = _dataset(prep, fit_df, target)
    dvalid = _dataset(prep, valid_df, target, reference=dfit)
    booster = lgb.train(
        lgb_params, dfit, num_boost_round=n_estimators, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    best_iter = booster.best_iteration or n_estimators
    valid_score = booster.best_score["valid_0"][EVAL_METRIC[task]]

    calibrator = None
    if task == "propensity":
        valid_pred = booster.predict(prep.transform(valid_df), num_iteration=best_iter)
        calibrator = PlattCalibrator().fit(valid_pred, valid_df[target].to_numpy())

    if refit_df is not None:
        booster = lgb.train(lgb_params, _dataset(prep, refit_df, target),
                            num_boost_round=best_iter)

    return GBMModel(task=task, params=dict(params), preprocessor=prep, booster=booster,
                    best_iteration=best_iter, valid_score=valid_score, calibrator=calibrator)
