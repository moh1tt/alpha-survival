"""Download Sharadar tables from Nasdaq Data Link and save them as local Parquet.

Every later step reads these files, never the API, so results are reproducible
and we only need the subscription while downloading.

This module handles filtered (e.g. sample-ticker) pulls. Full-history tables
are too large for `get_table` and will use the bulk-export endpoint later.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

API_KEY_VAR = "NASDAQ_DATA_LINK_API_KEY"
RAW_DIR = Path("data/raw")

# Columns later steps depend on. If the vendor renames one, we want to fail
# here, at download time, not silently deep inside a backtest.
REQUIRED_COLUMNS: dict[str, set[str]] = {
    "SEP": {"ticker", "date", "close", "closeadj", "volume"},
    # datekey = date the filing became public -> the point-in-time anchor.
    "SF1": {"ticker", "dimension", "datekey", "reportperiod", "calendardate"},
    "TICKERS": {"ticker", "permaticker", "isdelisted", "table"},
    "ACTIONS": {"ticker", "date", "action"},
}

Fetcher = Callable[..., pd.DataFrame]


def load_api_key() -> str:
    """Read the API key from the environment (or a local .env file)."""
    load_dotenv()
    key = os.environ.get(API_KEY_VAR)
    if not key:
        raise RuntimeError(f"{API_KEY_VAR} is not set. Copy .env.example to .env and add it.")
    return key


def validate_table(table: str, df: pd.DataFrame) -> None:
    """Raise if a downloaded table is empty or missing required columns."""
    if df.empty:
        raise ValueError(f"SHARADAR/{table} returned no rows (check filters / subscription).")
    missing = REQUIRED_COLUMNS[table] - set(df.columns)
    if missing:
        raise ValueError(f"SHARADAR/{table} is missing columns: {sorted(missing)}")


def default_fetcher() -> Fetcher:
    """The real network fetcher. Imported lazily so tests never need the API."""
    import nasdaqdatalink

    nasdaqdatalink.ApiConfig.api_key = load_api_key()
    return nasdaqdatalink.get_table


def download_table(
    table: str,
    out_dir: Path = RAW_DIR,
    fetch: Fetcher | None = None,
    **filters,
) -> Path:
    """Fetch one SHARADAR table, validate it, and write `<out_dir>/<table>.parquet`."""
    if table not in REQUIRED_COLUMNS:
        raise ValueError(f"Unknown table {table!r}; expected one of {sorted(REQUIRED_COLUMNS)}")
    fetch = fetch or default_fetcher()
    df = fetch(f"SHARADAR/{table}", paginate=True, **filters)
    validate_table(table, df)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{table}.parquet"
    df.to_parquet(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download Sharadar tables to Parquet.")
    parser.add_argument("--tables", nargs="+", default=sorted(REQUIRED_COLUMNS))
    parser.add_argument("--tickers", nargs="+", help="Restrict to these tickers (sample pulls).")
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args(argv)

    filters = {"ticker": ",".join(args.tickers)} if args.tickers else {}
    fetch = default_fetcher()
    for table in args.tables:
        path = download_table(table, args.out_dir, fetch=fetch, **filters)
        print(f"{table}: {len(pd.read_parquet(path)):,} rows -> {path}")


if __name__ == "__main__":
    main()
