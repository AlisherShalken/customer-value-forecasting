"""Scoring API.

Online scores are computed from the same SQL feature template as the batch
job, so a customer scored here matches the row written by `cvf score`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Annotated

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Query, Request

from cvf.api.schemas import CustomerScore, Driver, Health, ModelInfo, StoredScore
from cvf.config import Settings, get_settings
from cvf.features import features_as_of
from cvf.registry import ModelBundle, load_bundle
from cvf.scoring import score_frame


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    con = duckdb.connect(str(settings.data.warehouse), read_only=True)
    (last_ts,) = con.execute("SELECT max(invoice_ts) FROM stg.orders").fetchone()
    app.state.settings = settings
    app.state.bundle = load_bundle(settings)
    app.state.con = con
    app.state.data_as_of = last_ts.date() + timedelta(days=1)
    try:
        yield
    finally:
        con.close()


app = FastAPI(title="Customer Value API", version="1.0.0", lifespan=lifespan)


def get_cursor(request: Request) -> duckdb.DuckDBPyConnection:
    # One cursor per request: DuckDB cursors are safe to use from worker threads.
    cursor = request.app.state.con.cursor()
    try:
        yield cursor
    finally:
        cursor.close()


Cursor = Annotated[duckdb.DuckDBPyConnection, Depends(get_cursor)]


def _bundle(request: Request) -> ModelBundle:
    return request.app.state.bundle


def _resolve_as_of(request: Request, as_of: date | None) -> date:
    data_as_of = request.app.state.data_as_of
    if as_of is None:
        return data_as_of
    if as_of > data_as_of:
        raise HTTPException(422, f"as_of must not be after {data_as_of} (latest loaded data)")
    return as_of


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    return Health(status="ok", model_version=_bundle(request).version)


@app.get("/model", response_model=ModelInfo)
def model_info(request: Request) -> ModelInfo:
    meta = _bundle(request).metadata
    return ModelInfo(
        version=meta["version"], created_at=meta["created_at"],
        train_cutoff=meta["train_cutoff"], train_rows=meta["train_rows"],
        horizon_days=meta["horizon_days"], holdout_cutoff=meta["holdout_cutoff"],
        holdout_metrics=meta["holdout_metrics"], policy=meta["policy"],
        data_as_of=request.app.state.data_as_of,
    )


@app.get("/customers/{customer_id}/score", response_model=CustomerScore)
def score_customer(customer_id: int, request: Request, cursor: Cursor,
                   as_of: date | None = None) -> CustomerScore:
    as_of = _resolve_as_of(request, as_of)
    settings: Settings = request.app.state.settings
    features = features_as_of(cursor, settings, as_of, customer_id=customer_id)
    if features.empty:
        raise HTTPException(404, f"Customer {customer_id} has no purchases in the "
                                 f"{settings.snapshots.active_window_days} days before {as_of}")
    row = score_frame(features, _bundle(request), settings).iloc[0]
    return CustomerScore(
        customer_id=customer_id, as_of=as_of,
        p_purchase_90d=row["p_purchase_90d"],
        expected_revenue_90d=max(row["expected_revenue_90d"], 0.0),
        segment=row["segment"], contact=bool(row["contact"]),
        drivers=[Driver(**d) for d in json.loads(row["drivers"])],
        model_version=row["model_version"],
    )


@app.get("/scores", response_model=list[StoredScore])
def stored_scores(request: Request, cursor: Cursor,
                  as_of: date | None = None,
                  segment: str | None = None,
                  contact: bool | None = None,
                  limit: Annotated[int, Query(ge=1, le=10_000)] = 100) -> list[StoredScore]:
    """Batch scores from mart.customer_scores, highest expected revenue first."""
    as_of = _resolve_as_of(request, as_of)
    try:
        rows = cursor.execute(
            """
            SELECT customer_id, snapshot_date AS as_of, p_purchase_90d, expected_revenue_90d,
                   value_decile, segment, contact, model_version
            FROM mart.customer_scores
            WHERE snapshot_date = $as_of
              AND ($segment IS NULL OR segment = $segment)
              AND ($contact IS NULL OR contact = $contact)
            ORDER BY expected_revenue_90d DESC
            LIMIT $limit
            """,
            {"as_of": as_of, "segment": segment, "contact": contact, "limit": limit},
        ).df()
    except duckdb.CatalogException as exc:
        raise HTTPException(404, "No batch scores yet; run `cvf score`") from exc
    if rows.empty:
        raise HTTPException(404, f"No batch scores for {as_of}")
    return [StoredScore(**r) for r in rows.to_dict(orient="records")]
