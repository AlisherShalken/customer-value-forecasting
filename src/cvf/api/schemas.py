from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class Driver(BaseModel):
    feature: str
    value: float | str | None
    impact: float = Field(description="SHAP contribution to purchase log-odds")


class CustomerScore(BaseModel):
    customer_id: int
    as_of: date
    p_purchase_90d: float = Field(ge=0, le=1)
    expected_revenue_90d: float = Field(ge=0)
    segment: str
    contact: bool
    drivers: list[Driver]
    model_version: str


class StoredScore(BaseModel):
    customer_id: int
    as_of: date
    p_purchase_90d: float
    expected_revenue_90d: float
    value_decile: int
    segment: str
    contact: bool
    model_version: str


class ModelInfo(BaseModel):
    version: str
    created_at: str
    train_cutoff: date
    train_rows: int
    horizon_days: int
    holdout_cutoff: date
    holdout_metrics: dict[str, dict[str, float]]
    policy: dict[str, float]
    data_as_of: date


class Health(BaseModel):
    status: str
    model_version: str
