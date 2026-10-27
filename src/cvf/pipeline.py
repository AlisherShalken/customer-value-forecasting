"""End-to-end steps wired together for the CLI and scheduled jobs."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from cvf import features
from cvf.backtest import run_backtest
from cvf.config import Settings
from cvf.db import connect
from cvf.monitoring import reference_profile
from cvf.policy import high_value_threshold
from cvf.registry import ModelBundle, load_bundle, save_bundle
from cvf.report import write_report
from cvf.scoring import score_batch
from cvf.training import TASKS, fit_models, production_cutoff, split_for

log = logging.getLogger(__name__)

DRIFT_REFERENCE_SNAPSHOTS = 3


def build(settings: Settings) -> pd.DataFrame:
    with connect(settings.data.warehouse) as con:
        features.build_staging(con)
        return features.build_feature_store(con, settings)


def load_dataset(settings: Settings) -> pd.DataFrame:
    with connect(settings.data.warehouse, read_only=True) as con:
        return features.load_dataset(con)


def train(settings: Settings) -> ModelBundle:
    """Backtest the procedure, then fit and register the production models."""
    df = load_dataset(settings)

    metrics, predictions, holdout_models = run_backtest(df, settings)
    holdout_cutoff = settings.evaluation.holdout_cutoff
    holdout_features = df[df["snapshot_date"] == holdout_cutoff].reset_index(drop=True)

    cutoff = production_cutoff(df, settings)
    split = split_for(df, settings, cutoff)
    models = fit_models(df, split, settings)
    train_df = split.frame(df, "train")

    holdout_metrics = (metrics.query("cutoff == @holdout_cutoff")
                       .set_index(["task", "model", "metric"])["value"].sort_index())
    # Drift reference: the most recent training snapshots. Early snapshots have
    # a shorter observable history (data starts 2009-12) and would raise false
    # alerts on tenure-like features.
    reference_dates = split.train[-DRIFT_REFERENCE_SNAPSHOTS:]
    reference = train_df[train_df["snapshot_date"].isin(reference_dates)]
    metadata = {
        "train_cutoff": cutoff,
        "train_snapshots": [str(d) for d in split.train],
        "valid_snapshots": [str(d) for d in split.valid],
        "train_rows": len(train_df),
        "horizon_days": settings.snapshots.horizon_days,
        "features": models["propensity"].preprocessor.features,
        "categorical_levels": models["propensity"].preprocessor.levels,
        "params": {t: models[t].params for t in TASKS},
        "best_iteration": {t: models[t].best_iteration for t in TASKS},
        "holdout_cutoff": holdout_cutoff,
        "holdout_metrics": {
            task: holdout_metrics.loc[(task, "lightgbm")].round(4).to_dict() for task in TASKS
        },
        "policy": settings.policy.model_dump(),
        "high_value_revenue_365d": high_value_threshold(reference["revenue_365d"].to_numpy()),
        "reference_snapshots": [str(d) for d in reference_dates],
        "reference_profile": reference_profile(reference,
                                               models["propensity"].preprocessor.features),
    }
    bundle = save_bundle(models, metadata, settings)

    write_report(metrics, predictions, holdout_models, holdout_features, settings,
                 extra={"model_version": bundle.version, "train_cutoff": str(cutoff),
                        "train_rows": len(train_df),
                        "best_iteration": metadata["best_iteration"]})
    log.info("Registered model %s (trained on %d rows up to %s)",
             bundle.version, len(train_df), cutoff)
    return bundle


def default_as_of(settings: Settings) -> date:
    """Day after the last loaded transaction: the 'today' of the warehouse."""
    with connect(settings.data.warehouse, read_only=True) as con:
        (last,) = con.execute("SELECT max(invoice_ts) FROM stg.orders").fetchone()
    return last.date() + timedelta(days=1)


def score(settings: Settings, as_of: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of = as_of or default_as_of(settings)
    bundle = load_bundle(settings)
    with connect(settings.data.warehouse) as con:
        scores, drift = score_batch(con, bundle, settings, as_of)
    out = settings.reports_dir / "scoring"
    out.mkdir(parents=True, exist_ok=True)
    drift.to_csv(out / f"drift_{as_of}.csv", index=False)
    return scores, drift
