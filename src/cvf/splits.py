"""Time-based splits with purging of overlapping label windows.

A snapshot taken at T has labels observed over [T, T + horizon). It may only
be used to train a model evaluated at cutoff C if T + horizon <= C, otherwise
the training labels would contain information from after C.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd


def eligible_snapshots(dates: list[date], cutoff: date, horizon_days: int) -> list[date]:
    return sorted(d for d in dates if d + timedelta(days=horizon_days) <= cutoff)


@dataclass(frozen=True)
class TimeSplit:
    cutoff: date
    fit: list[date]         # used to fit trees
    valid: list[date]       # early stopping and probability calibration
    train: list[date]       # fit + valid + purged gap; used for the final refit
    test: list[date]

    def frame(self, df: pd.DataFrame, part: str) -> pd.DataFrame:
        return df[df["snapshot_date"].isin(getattr(self, part))].reset_index(drop=True)


def make_split(dates: list[date], cutoff: date, horizon_days: int,
               n_valid: int = 1) -> TimeSplit:
    train = eligible_snapshots(dates, cutoff, horizon_days)
    if len(train) <= n_valid:
        raise ValueError(f"Not enough history before {cutoff} for a validation split")
    valid = train[-n_valid:]
    fit = eligible_snapshots(train, valid[0], horizon_days)
    if not fit:
        raise ValueError(f"No snapshots left to fit on before validation {valid[0]}")
    test = [d for d in dates if d == cutoff]
    return TimeSplit(cutoff=cutoff, fit=fit, valid=valid, train=train, test=test)
