"""Download Sharadar tables from the direct Sharadar API and save them as local Parquet.

Every later step reads these files, never the API, so results are reproducible
and we only need the subscription while downloading.

Sharadar moved individual customers from Nasdaq Data Link to its own API
(api.sharadar.com) in 2026. The data is the same but some names changed, so
`to_legacy_schema` maps them back to the Nasdaq-era names the rest of the code
(and Sharadar's own docs) still use: SEP, SF1, TICKERS, ACTIONS, `datekey`.

This module handles per-ticker (sample) pulls. Full-history tables are too large
for row queries and will use the bulk-download endpoint later.
"""

from __future__ import annotations

import argparse
import io
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

API_KEY_VAR = "SHARADAR_API_KEY"
BASE_URL = "https://api.sharadar.com/v1.0/data"
RAW_DIR = Path("data/raw")

# The API silently defaults `from` to one year ago, so always ask for everything.
HISTORY_START = "1990-01-01"
# Max rows per request (the API default). A full response means rows were cut off.
ROW_LIMIT = 10_000

# Our names (legacy codes) -> direct-API table names.
API_TABLES = {"SEP": "stocks", "SF1": "fundamentals", "TICKERS": "tickers", "ACTIONS": "actions"}
DATED_TABLES = {"SEP", "SF1", "ACTIONS"}  # tables where `from` applies

# TICKERS.table names each Sharadar table a security appears in; map back to codes.
LEGACY_TABLE_CODES = {
    "stocks": "SEP",
    "funds": "SFP",
    "fundamentals": "SF1",
    "insiders": "SF2",
    "holdings": "SF3",
    "holdings_ticker": "SF3A",
    "holdings_investor": "SF3B",
}

# Identifier/label columns that can look numeric but must stay strings.
TEXT_COLUMNS = {"ticker", "name", "cusips", "figi", "relatedtickers", "contraticker", "contraname"}

# CSV has no types; parse these so Parquet stores real dates.
DATE_COLUMNS = {
    "date",
    "datekey",
    "reportperiod",
    "calendardate",
    "lastupdated",
    "firstadded",
    "firstpricedate",
    "lastpricedate",
}

# Columns later steps depend on. If the vendor renames one, we want to fail
# here, at download time, not silently deep inside a backtest.
REQUIRED_COLUMNS: dict[str, set[str]] = {
    "SEP": {"ticker", "date", "close", "closeadj", "volume"},
    # datekey = date the filing became public -> the point-in-time anchor.
    "SF1": {"ticker", "dimension", "datekey", "reportperiod", "calendardate"},
    "TICKERS": {"ticker", "permaticker", "isdelisted", "table"},
    "ACTIONS": {"ticker", "date", "action"},
}

# One API request: (api table name, query params) -> rows.
Fetcher = Callable[[str, dict[str, str]], pd.DataFrame]


def load_api_key() -> str:
    """Read the API key from the environment (or a local .env file)."""
    load_dotenv()
    key = os.environ.get(API_KEY_VAR)
    if not key:
        raise RuntimeError(f"{API_KEY_VAR} is not set. Copy .env.example to .env and add it.")
    return key


def api_fetcher(api_key: str | None = None) -> Fetcher:
    """The real network fetcher. Tests pass a fake instead, so they never hit the API."""
    key = api_key or load_api_key()
    session = requests.Session()
    # Header, not query string: keeps the key out of URLs, logs and error messages.
    session.headers["x-api-key"] = key

    def fetch(api_table: str, params: dict[str, str]) -> pd.DataFrame:
        resp = session.get(
            f"{BASE_URL}/{api_table}", params={**params, "format": "csv"}, timeout=60
        )
        if resp.status_code in (401, 403):
            # 403 "Exceeds free tier" = ticker/history outside the plan.
            raise PermissionError(
                f"Sharadar refused {api_table} {params}: {resp.status_code} {resp.text[:200]}"
            )
        resp.raise_for_status()
        return parse_csv(resp.text)

    return fetch


def parse_csv(text: str) -> pd.DataFrame:
    """Parse one API response. Identifiers stay text, whatever they look like."""
    if not text.strip():
        return pd.DataFrame()
    return pd.read_csv(
        io.StringIO(text),
        # Each ticker is fetched separately, so type guessing would differ per
        # stock (a CUSIP of digits -> int) and the frames would not concatenate.
        dtype={col: str for col in TEXT_COLUMNS},
        # Only blanks are missing: "NA" is a real ticker and "N/A" is the
        # vendor's literal placeholder (as in the Nasdaq-era files).
        keep_default_na=False,
        na_values=[""],
    )


def to_legacy_schema(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Rename direct-API fields back to the Nasdaq-era names and parse dates."""
    df = df.copy()
    if table == "SF1":
        # The direct API calls the filing date `date`; Nasdaq's SF1 called it `datekey`.
        df = df.rename(columns={"date": "datekey"})
    if table == "TICKERS" and "table" in df.columns:
        df["table"] = df["table"].replace(LEGACY_TABLE_CODES)
    for col in DATE_COLUMNS & set(df.columns):
        df[col] = pd.to_datetime(df[col])
    return df


def validate_table(table: str, df: pd.DataFrame) -> None:
    """Raise if a downloaded table is empty or missing required columns."""
    if df.empty:
        raise ValueError(f"Sharadar {table} returned no rows (check tickers / subscription).")
    missing = REQUIRED_COLUMNS[table] - set(df.columns)
    if missing:
        raise ValueError(f"Sharadar {table} is missing columns: {sorted(missing)}")


def download_table(
    table: str,
    tickers: list[str],
    out_dir: Path = RAW_DIR,
    fetch: Fetcher | None = None,
    dimension: str | None = None,
) -> Path:
    """Fetch one table for `tickers`, validate it, and write `<out_dir>/<table>.parquet`.

    One request per ticker keeps each response well under the row limit; if one
    still hits it we raise rather than silently save a truncated history.
    """
    if table not in API_TABLES:
        raise ValueError(f"Unknown table {table!r}; expected one of {sorted(API_TABLES)}")
    if not tickers:
        raise ValueError("Pass at least one ticker (full tables will use bulk download).")
    if dimension and table != "SF1":
        raise ValueError("`dimension` only applies to SF1.")

    params = {"limit": str(ROW_LIMIT)}
    if table in DATED_TABLES:
        params["from"] = HISTORY_START
    if dimension:
        params["dimension"] = dimension

    fetch = fetch or api_fetcher()
    frames = []
    for ticker in tickers:
        rows = fetch(API_TABLES[table], {**params, "ticker": ticker})
        if len(rows) >= ROW_LIMIT:
            raise ValueError(f"{table} {ticker}: hit the {ROW_LIMIT}-row limit, data truncated.")
        frames.append(rows)

    df = to_legacy_schema(table, pd.concat(frames, ignore_index=True))
    validate_table(table, df)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{table}.parquet"
    df.to_parquet(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download Sharadar tables to Parquet.")
    parser.add_argument("--tables", nargs="+", default=sorted(API_TABLES))
    parser.add_argument("--tickers", nargs="+", required=True, help="Tickers to download.")
    parser.add_argument("--dimension", help="SF1 dimensions, e.g. ARQ,ART (default: all).")
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args(argv)

    fetch = api_fetcher()
    for table in args.tables:
        dimension = args.dimension if table == "SF1" else None
        path = download_table(table, args.tickers, args.out_dir, fetch, dimension)
        print(f"{table}: {len(pd.read_parquet(path)):,} rows -> {path}")


if __name__ == "__main__":
    main()
