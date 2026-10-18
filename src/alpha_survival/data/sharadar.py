"""Download Sharadar tables from the direct Sharadar API and save them as local Parquet.

Every later step reads these files, never the API, so results are reproducible
and we only need the subscription while downloading.

Files are saved exactly as the API delivers them: Sharadar's own table and
column names, no renaming. Our clean names live in `real.py`.

Older papers, forums and Nasdaq Data Link use legacy codes for the same tables:
  stocks = SEP, fundamentals = SF1, tickers = TICKERS, actions = ACTIONS.
In `fundamentals`, `date` is the filing date (legacy `datekey`), not a trading day.

Two modes: per-ticker row queries (samples, tests) and bulk files (full tables,
too large for row queries: the stocks zip alone is ~1 GB).
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from collections.abc import Callable
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv

API_KEY_VAR = "SHARADAR_API_KEY"
BASE_URL = "https://api.sharadar.com/v1.0/data"
RAW_DIR = Path("data/raw")

# The API silently defaults `from` to one year ago, so always ask for everything.
HISTORY_START = "1990-01-01"
# Max rows per request (the API default). A full response means rows were cut off.
ROW_LIMIT = 10_000

DATED_TABLES = {"stocks", "fundamentals", "actions"}  # tables where `from` applies

# Identifier/label columns that can look numeric but must stay strings.
TEXT_COLUMNS = {"ticker", "name", "cusips", "figi", "relatedtickers", "contraticker", "contraname"}

# CSV has no types; parse these so Parquet stores real dates.
DATE_COLUMNS = {
    "date",
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
    "stocks": {"ticker", "date", "close", "closeadj", "volume"},
    # date = day the filing became public -> the point-in-time anchor.
    "fundamentals": {"ticker", "dimension", "date", "reportperiod", "calendardate"},
    "tickers": {"ticker", "permaticker", "isdelisted", "table"},
    "actions": {"ticker", "date", "action"},
}

# One API request: (table name, query params) -> rows.
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

    def fetch(table: str, params: dict[str, str]) -> pd.DataFrame:
        resp = session.get(f"{BASE_URL}/{table}", params={**params, "format": "csv"}, timeout=60)
        if resp.status_code in (401, 403):
            # 403 "Exceeds free tier" = ticker/history outside the plan.
            raise PermissionError(
                f"Sharadar refused {table} {params}: {resp.status_code} {resp.text[:200]}"
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
        # vendor's literal placeholder.
        keep_default_na=False,
        na_values=[""],
    )


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
    if table not in REQUIRED_COLUMNS:
        raise ValueError(f"Unknown table {table!r}; expected one of {sorted(REQUIRED_COLUMNS)}")
    if not tickers:
        raise ValueError("Pass at least one ticker (full tables will use bulk download).")
    if dimension and table != "fundamentals":
        raise ValueError("`dimension` only applies to fundamentals.")

    params = {"limit": str(ROW_LIMIT)}
    if table in DATED_TABLES:
        params["from"] = HISTORY_START
    if dimension:
        params["dimension"] = dimension

    fetch = fetch or api_fetcher()
    frames = []
    for ticker in tickers:
        rows = fetch(table, {**params, "ticker": ticker})
        if len(rows) >= ROW_LIMIT:
            raise ValueError(f"{table} {ticker}: hit the {ROW_LIMIT}-row limit, data truncated.")
        frames.append(rows)

    df = pd.concat(frames, ignore_index=True)
    for col in DATE_COLUMNS & set(df.columns):
        df[col] = pd.to_datetime(df[col])
    validate_table(table, df)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{table}.parquet"
    df.to_parquet(path, index=False)
    return path


REDIRECT_CODES = {301, 302, 303, 307, 308}


def fetch_bulk_zip(table: str, dest: Path, history: str = "full", api_key: str | None = None):
    """Download one table's bulk zip (`history`: 5, 10 or full years) to `dest`."""
    resp = requests.get(
        f"{BASE_URL}/{table}",
        params={"years": history},
        headers={"x-api-key": api_key or load_api_key()},
        allow_redirects=False,
        timeout=60,
    )
    if resp.status_code in (401, 403):
        raise PermissionError(
            f"Sharadar refused bulk {table}: {resp.status_code} {resp.text[:200]}"
        )
    if resp.status_code not in REDIRECT_CODES:
        resp.raise_for_status()
        raise ValueError(f"Expected a redirect to the {table} bulk file, got {resp.status_code}")

    # The redirect is a pre-signed link on a storage host: fetch it without our
    # key, which has no business reaching a third party.
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(resp.headers["Location"], stream=True, timeout=60) as file_resp:
        file_resp.raise_for_status()
        expected = int(file_resp.headers.get("Content-Length", -1))
        with open(dest, "wb") as f:
            for chunk in file_resp.iter_content(chunk_size=8 << 20):
                f.write(chunk)

    if expected >= 0 and dest.stat().st_size != expected:
        dest.unlink()
        raise OSError(f"{table} bulk download cut off: {dest.stat().st_size} of {expected} bytes")
    return dest


def convert_bulk_csv(zip_path: Path, table: str, out_dir: Path = RAW_DIR) -> Path:
    """Convert a bulk zip to `<out_dir>/<table>.parquet`, with the same typing as row queries.

    DuckDB streams the CSV from disk, so a multi-GB file never has to fit in
    pandas. The Parquet is written to a temp name and renamed only once complete.
    """
    if table not in REQUIRED_COLUMNS:
        raise ValueError(f"Unknown table {table!r}; expected one of {sorted(REQUIRED_COLUMNS)}")

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{zip_path.name}: expected one CSV, found {members}")
        with zf.open(members[0]) as f:
            header = f.readline().decode("utf-8-sig").strip().replace('"', "").split(",")
        missing = REQUIRED_COLUMNS[table] - set(header)
        if missing:
            raise ValueError(f"Sharadar {table} is missing columns: {sorted(missing)}")
        csv_path = Path(zf.extract(members[0], zip_path.parent))

    # Same rules as parse_csv: identifiers stay text, only blanks are missing,
    # and the whole file is scanned before guessing a numeric column's type.
    types = {c: "VARCHAR" for c in TEXT_COLUMNS & set(header)}
    types |= {c: "TIMESTAMP" for c in DATE_COLUMNS & set(header)}
    path = out_dir / f"{table}.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        con = duckdb.connect()
        con.execute(
            f"COPY (SELECT * FROM read_csv('{csv_path.as_posix()}', header = true, "
            f"sample_size = -1, nullstr = '', types = {types})) "
            f"TO '{tmp.as_posix()}' (FORMAT parquet)"
        )
        con.close()
    finally:
        csv_path.unlink()

    if pq.ParquetFile(tmp).metadata.num_rows == 0:
        tmp.unlink()
        raise ValueError(f"Sharadar {table} bulk file has no rows.")
    tmp.replace(path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download Sharadar tables to Parquet.")
    parser.add_argument("--tables", nargs="+", default=sorted(REQUIRED_COLUMNS))
    parser.add_argument("--tickers", nargs="+", help="Tickers to download (row queries).")
    parser.add_argument("--bulk", action="store_true", help="Download full tables as bulk files.")
    parser.add_argument("--history", default="full", choices=["5", "10", "full"])
    parser.add_argument("--dimension", help="fundamentals dimensions, e.g. ARQ,ART (default: all).")
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args(argv)
    if args.bulk == bool(args.tickers):
        parser.error("pass either --tickers or --bulk")
    if args.bulk and args.dimension:
        parser.error("--dimension applies to row queries only; bulk files hold every dimension")

    if args.bulk:
        for table in args.tables:
            zip_path = fetch_bulk_zip(table, args.out_dir / f"{table}.csv.zip", args.history)
            path = convert_bulk_csv(zip_path, table, args.out_dir)
            zip_path.unlink()
            print(f"{table}: {pq.ParquetFile(path).metadata.num_rows:,} rows -> {path}")
        return

    fetch = api_fetcher()
    for table in args.tables:
        dimension = args.dimension if table == "fundamentals" else None
        path = download_table(table, args.tickers, args.out_dir, fetch, dimension)
        print(f"{table}: {len(pd.read_parquet(path)):,} rows -> {path}")


if __name__ == "__main__":
    main()
