"""Download the free external series some signals need and save them as local Parquet.

Sources (all free, no key):
- Ken French data library: Fama-French 3 factors (Mkt-RF, SMB, HML) and RF, daily
  and monthly (Fama & French 1993).
- FRED: VIXCLS (daily VIX close) and GNPDEF (quarterly GNP implicit price deflator).
- Pastor & Stambaugh (2003) aggregate liquidity, monthly, from Lubos Pastor's site.

Like `sharadar.py`, every later step reads these files, never the web.

Point in time: every saved table has `date` (the period a value is about) and
`available` (the first month end a value may be used at). Read them through
`as_of`, never by `date`. Known caveats, all as in Open Source Asset Pricing:
- We download today's files, not the versions published back then. Ken French
  factors and GNPDEF are revised a little over time; VIX never is.
- GNPDEF comes out about a month after its quarter ends: a quarter's value is
  available only from the end of the next quarter, which leaves room to spare.
- Pastor-Stambaugh re-estimate the whole liquidity series every year, so an old
  month's value depends on later data. Signals built on it are flagged.
- Pastor-Stambaugh is updated once a year, so the latest months are missing.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import requests

EXTERNAL_DIR = Path("data/raw/external")

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
# The file name carries the last year covered and changes with every annual update.
PASTOR_URL = (
    "https://faculty.chicagobooth.edu/-/media/faculty/lubos-pastor/data/liq_data_1962_2025.txt"
)

FRENCH_FILES = {
    "ff3_daily": "F-F_Research_Data_Factors_daily",
    "ff3_monthly": "F-F_Research_Data_Factors",
}
FRENCH_COLUMNS = {"Mkt-RF": "mktrf", "SMB": "smb", "HML": "hml", "RF": "rf"}
FRED_SERIES = {"vix": "VIXCLS", "gnpdef": "GNPDEF"}
PASTOR_COLUMNS = ["agg_liq", "innov_liq", "traded_liq"]
PASTOR_MISSING = -99

# One download: URL -> raw bytes. Tests pass a fake instead, so they never hit the web.
Fetcher = Callable[[str], bytes]


def http_fetcher(url: str) -> bytes:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def parse_french(content: bytes, daily: bool) -> pd.DataFrame:
    """Parse a Ken French factor zip: the first table only (the monthly file also has
    an annual one below it), percent -> decimal, YYYYMMDD / YYYYMM -> dates."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        (member,) = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        lines = zf.read(member).decode("latin-1").splitlines()

    start = next(i for i, line in enumerate(lines) if line.startswith(",Mkt-RF"))
    header = ["date", *(c.strip() for c in lines[start].split(",")[1:])]
    rows = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break  # a blank line ends the table
        rows.append([c.strip() for c in line.split(",")])

    df = pd.DataFrame(rows, columns=header).rename(columns=FRENCH_COLUMNS)
    missing = set(FRENCH_COLUMNS.values()) - set(df.columns)
    if missing:
        raise ValueError(f"Ken French file {member} is missing columns: {sorted(missing)}")
    values = list(FRENCH_COLUMNS.values())
    df[values] = df[values].astype(float) / 100
    if daily:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        # Known at that day's close, so usable at the month end it falls in.
        df["available"] = df["date"] + pd.offsets.MonthEnd(0)
    else:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m") + pd.offsets.MonthEnd(0)
        df["available"] = df["date"]
    return df[["date", "available", *values]]


def parse_fred(content: bytes, series: str) -> pd.DataFrame:
    """Parse a FRED graph CSV. FRED dates each value by the START of its period."""
    df = pd.read_csv(io.BytesIO(content), na_values=["", "."])
    if series not in df.columns:
        raise ValueError(f"FRED file has no {series} column: {list(df.columns)}")
    df = df.rename(columns={"observation_date": "date", series: "value"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["value"])
    if series == "GNPDEF":
        quarter_end = df["date"] + pd.offsets.QuarterEnd(0)
        df["available"] = quarter_end + pd.offsets.QuarterEnd(1)
    else:
        df["available"] = df["date"] + pd.offsets.MonthEnd(0)
    return df[["date", "available", "value"]].reset_index(drop=True)


def parse_pastor(content: bytes) -> pd.DataFrame:
    """Parse the Pastor-Stambaugh liquidity text file: `%` comments, then tab-separated
    month (YYYYMM), level, innovation (the main series) and traded factor; -99 = missing."""
    df = pd.read_csv(
        io.BytesIO(content),
        sep=r"\s+",
        comment="%",
        header=None,
        names=["date", *PASTOR_COLUMNS],
        dtype={"date": str},
    )
    df[PASTOR_COLUMNS] = df[PASTOR_COLUMNS].mask(df[PASTOR_COLUMNS] == PASTOR_MISSING)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df["available"] = df["date"]
    return df[["date", "available", *PASTOR_COLUMNS]]


def as_of(df: pd.DataFrame, month_end: pd.Timestamp) -> pd.DataFrame:
    """The rows a signal computed at `month_end` may use."""
    return df[df["available"] <= month_end]


def download_all(out_dir: Path = EXTERNAL_DIR, fetch: Fetcher = http_fetcher) -> list[Path]:
    """Download and parse every external series to `<out_dir>/<name>.parquet`."""
    tables = {
        name: parse_french(fetch(FRENCH_URL.format(name=file)), daily=name.endswith("daily"))
        for name, file in FRENCH_FILES.items()
    }
    tables |= {
        name: parse_fred(fetch(FRED_URL.format(series=series)), series)
        for name, series in FRED_SERIES.items()
    }
    tables["ps_liquidity"] = parse_pastor(fetch(PASTOR_URL))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables.items():
        if df.empty:
            raise ValueError(f"{name}: no rows parsed")
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        paths.append(path)
    return paths


def load(name: str, raw_dir: Path = EXTERNAL_DIR) -> pd.DataFrame:
    """Read one saved external table, e.g. `load("ff3_daily")`."""
    return pd.read_parquet(raw_dir / f"{name}.parquet")


def monthly_factors(raw_dir: Path = EXTERNAL_DIR) -> pd.DataFrame:
    """Monthly rf, mktrf, smb, hml and ps_innov, indexed by the month end each is
    known at. A monthly value is about the month it is available at the end of, so
    a factor return and a stock return of the same month line up."""
    ff = load("ff3_monthly", raw_dir)
    ps = load("ps_liquidity", raw_dir)
    for name, df in [("ff3_monthly", ff), ("ps_liquidity", ps)]:
        if not (df["available"] == df["date"]).all():
            raise ValueError(f"{name}: monthly values must be available at their own month end")
    ff = ff.set_index("available")[["rf", "mktrf", "smb", "hml"]]
    ps = ps.set_index("available")["innov_liq"].rename("ps_innov")
    return ff.join(ps, how="left").rename_axis("month_end")


def daily_factors(raw_dir: Path = EXTERNAL_DIR) -> pd.DataFrame:
    """Daily rf, mktrf, smb, hml and dvix, indexed by trading day (each known at that
    day's close, so usable within its month).

    `dvix`: change in the VIX close since the previous observation, NaN on days FRED
    has no VIX (OSAP's d_vix). OSAP uses the older VXO until its 2021 end; we use VIX
    throughout, the same idea on S&P 500 instead of S&P 100 options.
    """
    ff = load("ff3_daily", raw_dir).set_index("date")[["rf", "mktrf", "smb", "hml"]]
    vix = load("vix", raw_dir).set_index("date")["value"]
    return ff.join(vix.diff().rename("dvix"), how="left").rename_axis("date")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download external series to Parquet.")
    parser.add_argument("--out-dir", type=Path, default=EXTERNAL_DIR)
    args = parser.parse_args(argv)
    for path in download_all(args.out_dir):
        df = pd.read_parquet(path)
        print(
            f"{path.stem}: {len(df):,} rows, {df['date'].min():%Y-%m-%d} to "
            f"{df['date'].max():%Y-%m-%d} -> {path}"
        )


if __name__ == "__main__":
    main()
