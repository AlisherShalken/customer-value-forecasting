"""Typed access to configs/config.yaml."""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


class DataConfig(BaseModel):
    source_url: str
    raw_dir: Path
    warehouse: Path


class SnapshotConfig(BaseModel):
    first: date
    last: date
    horizon_days: int = Field(gt=0)
    active_window_days: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> SnapshotConfig:
        if self.first > self.last:
            raise ValueError("snapshots.first must not be after snapshots.last")
        return self


class EvaluationConfig(BaseModel):
    backtest_cutoffs: list[date]
    holdout_cutoff: date
    top_k_shares: list[float]

    @model_validator(mode="after")
    def _holdout_is_last(self) -> EvaluationConfig:
        if self.holdout_cutoff != max(self.backtest_cutoffs):
            raise ValueError("evaluation.holdout_cutoff must be the latest backtest cutoff")
        return self


class TrainingConfig(BaseModel):
    seed: int = 42
    early_stopping_rounds: int = 100
    n_trials: int = 40
    tune_tasks: list[str] = Field(default_factory=lambda: ["propensity", "revenue"])
    tuned_params: Path = Path("configs/tuned_params.json")
    propensity: dict[str, Any]
    revenue: dict[str, Any]


class PolicyConfig(BaseModel):
    contact_cost: float
    gross_margin: float
    assumed_uplift: float


class MonitoringConfig(BaseModel):
    psi_warn: float = 0.10
    psi_alert: float = 0.25


class Settings(BaseModel):
    data: DataConfig
    snapshots: SnapshotConfig
    evaluation: EvaluationConfig
    training: TrainingConfig
    policy: PolicyConfig
    monitoring: MonitoringConfig
    artifacts_dir: Path
    reports_dir: Path

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        def resolve(p: Path) -> Path:
            return p if p.is_absolute() else PROJECT_ROOT / p

        self.data.raw_dir = resolve(self.data.raw_dir)
        self.data.warehouse = resolve(self.data.warehouse)
        self.training.tuned_params = resolve(self.training.tuned_params)
        self.artifacts_dir = resolve(self.artifacts_dir)
        self.reports_dir = resolve(self.reports_dir)
        return self


def load_settings(path: Path | str | None = None) -> Settings:
    path = Path(path or os.environ.get("CVF_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as fh:
        return Settings.model_validate(yaml.safe_load(fh))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
