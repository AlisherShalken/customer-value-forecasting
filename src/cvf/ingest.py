"""Load the source transaction extract into the warehouse `raw` schema."""

from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from cvf.config import Settings
from cvf.db import connect

log = logging.getLogger(__name__)

XLSX_NAME = "online_retail_II.xlsx"
PARQUET_NAME = "transactions.parquet"

COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_ts",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def download(settings: Settings, force: bool = False) -> Path:
    raw_dir = settings.data.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    xlsx = raw_dir / XLSX_NAME
    if xlsx.exists() and not force:
        return xlsx
    archive = raw_dir / "online_retail_ii.zip"
    log.info("Downloading %s", settings.data.source_url)
    urllib.request.urlretrieve(settings.data.source_url, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extract(XLSX_NAME, raw_dir)
    return xlsx


def xlsx_to_parquet(xlsx: Path, out: Path) -> Path:
    """Convert the workbook (one sheet per fiscal year) to a typed parquet file.

    Parsing the workbook takes about a minute, so the parquet copy is cached.
    """
    if out.exists() and out.stat().st_mtime >= xlsx.stat().st_mtime:
        return out
    log.info("Parsing %s", xlsx.name)
    sheets = pd.read_excel(xlsx, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    frames = []
    for sheet_name, df in sheets.items():
        df = df.rename(columns=COLUMNS)[list(COLUMNS.values())]
        df["source_sheet"] = sheet_name
        df["source_row"] = range(len(df))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["customer_id"] = df["customer_id"].astype("Int64")
    df["description"] = df["description"].astype("string")
    df["country"] = df["country"].astype("string")
    df.to_parquet(out, index=False)
    log.info("Wrote %s rows to %s", f"{len(df):,}", out)
    return out


def load_raw(settings: Settings, force_download: bool = False) -> int:
    xlsx = download(settings, force=force_download)
    parquet = xlsx_to_parquet(xlsx, settings.data.raw_dir / PARQUET_NAME)
    with connect(settings.data.warehouse) as con:
        con.execute("CREATE SCHEMA IF NOT EXISTS raw")
        con.execute(
            "CREATE OR REPLACE TABLE raw.transactions AS SELECT * FROM read_parquet(?)",
            [str(parquet)],
        )
        (n,) = con.execute("SELECT count(*) FROM raw.transactions").fetchone()
    return n
