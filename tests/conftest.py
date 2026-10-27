from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
import yaml

from cvf.config import DEFAULT_CONFIG, Settings

RAW_COLUMNS = ["invoice", "stock_code", "description", "quantity", "invoice_ts",
               "unit_price", "customer_id", "country", "source_sheet", "source_row"]
SHEET = "Year 2010-2011"


def make_settings(tmp_path: Path, **overrides) -> Settings:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["data"]["warehouse"] = str(tmp_path / "warehouse.duckdb")
    raw["data"]["raw_dir"] = str(tmp_path / "raw")
    raw["artifacts_dir"] = str(tmp_path / "artifacts")
    raw["reports_dir"] = str(tmp_path / "reports")
    raw["training"]["tuned_params"] = str(tmp_path / "tuned_params.json")
    for dotted, value in overrides.items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return Settings.model_validate(raw)


def line(invoice: str, customer_id: int | None, ts: str, amount: float, *,
         stock_code: str = "22423", quantity: int = 1, country: str = "United Kingdom",
         sheet: str = SHEET) -> dict:
    """One raw line where quantity * unit_price == amount."""
    return {
        "invoice": invoice, "stock_code": stock_code, "description": "ITEM",
        "quantity": quantity, "invoice_ts": pd.Timestamp(ts),
        "unit_price": abs(amount) / abs(quantity), "customer_id": customer_id,
        "country": country, "source_sheet": sheet, "source_row": 0,
    }


def load_raw(con: duckdb.DuckDBPyConnection, rows: list[dict] | pd.DataFrame) -> None:
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df["customer_id"] = df["customer_id"].astype("Int64")
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.register("raw_df", df)
    con.execute("CREATE OR REPLACE TABLE raw.transactions AS SELECT * FROM raw_df")
    con.unregister("raw_df")


@pytest.fixture
def con():
    connection = duckdb.connect(":memory:")
    yield connection
    connection.close()


def synthetic_transactions(n_customers: int = 400, start: date = date(2020, 1, 1),
                           end: date = date(2021, 12, 31), seed: int = 0) -> pd.DataFrame:
    """Customers with heterogeneous purchase rates, basket sizes and churn."""
    rng = np.random.default_rng(seed)
    days = (end - start).days
    rows, invoice_no = [], 100000
    for cid in range(10000, 10000 + n_customers):
        rate = rng.gamma(1.5, 1 / 30)                     # orders per day
        basket = rng.lognormal(5.5, 0.8)
        first = rng.integers(0, days - 60)
        churn = first + rng.exponential(400)
        country = rng.choice(["United Kingdom", "Germany", "France"], p=[0.8, 0.1, 0.1])
        t = float(first)
        while t < min(days, churn):
            ts = datetime.combine(start, datetime.min.time()) + timedelta(
                days=t, hours=int(rng.integers(8, 18)))
            invoice_no += 1
            for _ in range(rng.integers(1, 5)):
                qty = int(rng.integers(1, 24))
                rows.append(line(str(invoice_no), cid, str(ts), basket / 4 * rng.uniform(0.5, 1.5),
                                 stock_code=str(rng.integers(20000, 20300)), quantity=qty,
                                 country=str(country)))
            t += rng.exponential(1 / rate)
    return pd.DataFrame(rows, columns=RAW_COLUMNS)
