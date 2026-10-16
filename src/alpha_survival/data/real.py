"""Load the raw Sharadar Parquet files into our own clean, point-in-time schema.

Later steps (features, models) read only what this module returns, never raw
vendor columns, so swapping the data source means changing just this file.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from alpha_survival.data.sharadar import RAW_DIR

# Sharadar column -> our column name.
SECURITY_MASTER_COLUMNS = {
    "permaticker": "permaticker",
    "ticker": "ticker",
    "name": "name",
    "exchange": "exchange",
    "category": "category",
    "isdelisted": "is_delisted",
    "firstpricedate": "first_price_date",
    "lastpricedate": "last_price_date",
}


def build_security_master(tickers: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw `tickers` table into one row per stock, keyed on `permaticker`.

    Tickers get renamed and reused, so `permaticker` (permanent per company) is
    the key everything else joins on. Delisted stocks are kept on purpose:
    dropping them is survivorship bias.
    """
    # `tickers` repeats each stock once per Sharadar table (stocks, fundamentals,
    # insiders...). The `stocks` row describes the price history we trade on.
    priced = tickers.loc[tickers["table"] == "stocks", list(SECURITY_MASTER_COLUMNS)]
    master = priced.rename(columns=SECURITY_MASTER_COLUMNS)

    dupes = master["permaticker"].duplicated(keep=False)
    if dupes.any():
        ids = sorted(master.loc[dupes, "permaticker"].unique())
        raise ValueError(f"Security master has duplicate permatickers: {ids[:10]}")

    master["is_delisted"] = master["is_delisted"].map({"Y": True, "N": False})
    if master["is_delisted"].isna().any():
        raise ValueError("isdelisted contains values other than 'Y'/'N'")

    for col in ("first_price_date", "last_price_date"):
        master[col] = pd.to_datetime(master[col])

    return master.sort_values("permaticker").reset_index(drop=True)


def load_security_master(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read `tickers.parquet` from disk and build the security master."""
    return build_security_master(pd.read_parquet(raw_dir / "tickers.parquet"))


def build_monthly_returns(stocks: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Turn daily `stocks` prices into one total return per stock per calendar month.

    The return for month m runs from the last close in month m-1 to the last
    close in month m. Returns are computed from `closeadj` (split + dividend
    adjusted). Use `closeadj` ONLY for returns, never as a feature: Sharadar
    rewrites its past levels every time a dividend is paid, so a historical
    level is not what anyone could have seen on that date (look-ahead).

    Columns: permaticker, month_end, ret, last_trade_date.
    `last_trade_date` lets later steps spot stale prices (no trade near month end).
    """
    if stocks.duplicated(["ticker", "date"]).any():
        raise ValueError("stocks has duplicate (ticker, date) rows")

    # `stocks` is keyed by ticker; map to the permanent ID. Sharadar keeps tickers
    # unique by suffixing reused ones (e.g. "ABC1"), so this mapping is 1:1.
    ids = master.set_index("ticker")["permaticker"]
    unmapped = set(stocks["ticker"]) - set(ids.index)
    if unmapped:
        raise ValueError(f"stocks tickers missing from security master: {sorted(unmapped)[:10]}")

    daily = stocks[["ticker", "date", "closeadj"]].assign(
        permaticker=lambda d: d["ticker"].map(ids),
        month=lambda d: d["date"].dt.to_period("M"),
    )
    # The vendor delivers newest-first; sort so "last" really means latest.
    daily = daily.sort_values(["permaticker", "date"])

    monthly = (
        daily.groupby(["permaticker", "month"], as_index=False)
        .last()
        .rename(columns={"date": "last_trade_date"})
    )

    by_stock = monthly.groupby("permaticker")
    prev_close = by_stock["closeadj"].shift(1)
    prev_month = by_stock["month"].shift(1)
    monthly["ret"] = monthly["closeadj"] / prev_close - 1

    # If a stock skipped a whole month (halt, data gap), the change since the
    # previous row spans several months; don't pretend it is one month's return.
    gap = prev_month.notna() & (monthly["month"] != prev_month + 1)
    monthly.loc[gap, "ret"] = float("nan")

    monthly["month_end"] = monthly["month"].dt.to_timestamp(how="end").dt.normalize()
    return monthly[["permaticker", "month_end", "ret", "last_trade_date"]].reset_index(drop=True)


# Sharadar `actions` types where shareholders likely lost money on exit. Names verified
# against the Nasdaq-era action-type list (re-check on the direct API once we have
# delisted names); for all four, `date` is the last trade date. Acquisitions/mergers
# are not here: the last price already reflects the deal, so their delisting
# return is ~0 and we leave them alone.
DISTRESS_EXITS = {"bankruptcyliquidation", "regulatorydelisting", "voluntarydelisting", "delisted"}

# Average return after a performance delisting, which the last price misses.
# NYSE/AMEX: Shumway (1997); Nasdaq: Shumway & Warther (1999).
DELISTING_PENALTY = {"NYSE": -0.30, "NYSEMKT": -0.30, "NASDAQ": -0.55}
DEFAULT_DELISTING_PENALTY = -0.30  # the few BATS/OTC/NYSEARCA names


def add_delisting_returns(
    monthly: pd.DataFrame, actions: pd.DataFrame, master: pd.DataFrame
) -> pd.DataFrame:
    """Compound a delisting penalty into the exit month of stocks that died in distress.

    Without this, a stock that goes bankrupt mid-month shows only the fall up to
    its last trade; the further loss to shareholders vanishes and distressed
    stocks look better than they were (Shumway 1997).

    Adds `delist_action` (the exit type, or NA) so later reports can count them.
    """
    exits = actions.loc[actions["action"].isin(DISTRESS_EXITS), ["ticker", "date", "action"]]
    exits = exits.merge(master[["ticker", "permaticker", "exchange"]], on="ticker")
    # A stock can get several exit rows (e.g. regulatory delisting, then bankruptcy):
    # penalise it once, using the latest.
    exits = exits.sort_values("date").drop_duplicates("permaticker", keep="last")
    exits["month_end"] = exits["date"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
    exits["penalty"] = exits["exchange"].map(DELISTING_PENALTY).fillna(DEFAULT_DELISTING_PENALTY)

    out = monthly.merge(
        exits[["permaticker", "month_end", "action", "penalty"]],
        on=["permaticker", "month_end"],
        how="left",
        validate="one_to_one",
    )
    hit = out["penalty"].notna()
    out.loc[hit, "ret"] = (1 + out.loc[hit, "ret"]) * (1 + out.loc[hit, "penalty"]) - 1

    unmatched = len(exits[exits["permaticker"].isin(monthly["permaticker"])]) - hit.sum()
    if unmatched:
        warnings.warn(
            f"{unmatched} distress exits have no price row in their exit month; not penalised",
            stacklevel=2,
        )
    return out.rename(columns={"action": "delist_action"}).drop(columns="penalty")


def load_monthly_returns(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read `stocks` + `actions` and build delisting-adjusted monthly returns."""
    master = load_security_master(raw_dir)
    monthly = build_monthly_returns(pd.read_parquet(raw_dir / "stocks.parquet"), master)
    return add_delisting_returns(monthly, pd.read_parquet(raw_dir / "actions.parquet"), master)


# `fundamentals` "dimensions": AR* = as reported (one row per original filing, dated by when it
# became public). MR* = most recent, i.e. silently restated later; its numbers were
# never visible on the date they sit at, so MR* is unusable for backtests.
POINT_IN_TIME_DIMENSIONS = {"ARQ", "ART", "ARY"}

# In `fundamentals`, `date` is the day the filing became public (not a trading day).
# Filings often land after the close, so a filing is usable from the day after
# that date, never on the date itself.
AVAILABILITY_LAG = pd.Timedelta(days=1)

# `fundamentals` identifier/date columns; every other numeric column is a value.
# `lastupdated` is when Sharadar last touched the row: a future date, never a feature.
FUNDAMENTALS_ID_COLUMNS = {
    "ticker",
    "dimension",
    "calendardate",
    "date",
    "reportperiod",
    "fiscalperiod",
    "lastupdated",
}


def build_fundamentals(
    fundamentals: pd.DataFrame, master: pd.DataFrame, keys: pd.DataFrame, dimension: str = "ART"
) -> pd.DataFrame:
    """Attach to each (permaticker, month_end) in `keys` the latest filing public before it.

    Aligns on `date` (the day the filing became public), not on the fiscal
    period: Q4 numbers for December are not known until February. Default ART
    (trailing twelve months, as reported) smooths seasonal quarters.

    Columns: permaticker, month_end, filing_date, period_end, then the fundamental values.
    `filing_date` lets later steps drop stale fundamentals.
    """
    if dimension not in POINT_IN_TIME_DIMENSIONS:
        raise ValueError(
            f"Dimension {dimension!r} is not point-in-time; use one of "
            f"{sorted(POINT_IN_TIME_DIMENSIONS)}"
        )
    rows = fundamentals[fundamentals["dimension"] == dimension]

    if rows.duplicated(["ticker", "date"]).any():
        raise ValueError(f"fundamentals {dimension} has duplicate (ticker, date) rows")
    if (rows["date"] < rows["reportperiod"]).any():
        raise ValueError("fundamentals has filings dated before their period ended")

    ids = master.set_index("ticker")["permaticker"]
    unmapped = set(rows["ticker"]) - set(ids.index)
    if unmapped:
        raise ValueError(
            f"fundamentals tickers missing from security master: {sorted(unmapped)[:10]}"
        )

    values = [c for c in rows.select_dtypes("number").columns if c not in FUNDAMENTALS_ID_COLUMNS]
    # Rename `date` at once: here it is a filing date, and must not be mistaken
    # for the trading `date` used everywhere else.
    filings = rows[["ticker", "date", "reportperiod", *values]].rename(
        columns={"date": "filing_date", "reportperiod": "period_end"}
    )
    filings = filings.assign(
        permaticker=filings["ticker"].map(ids),
        # merge_asof needs both date columns at the same resolution.
        available_date=(filings["filing_date"] + AVAILABILITY_LAG).astype("datetime64[ns]"),
    ).drop(columns="ticker")

    left = keys[["permaticker", "month_end"]].astype({"month_end": "datetime64[ns]"})
    # "Latest filing with available_date <= month_end", per stock.
    out = pd.merge_asof(
        left.sort_values("month_end"),
        filings.sort_values("available_date"),
        left_on="month_end",
        right_on="available_date",
        by="permaticker",
        direction="backward",
    ).drop(columns="available_date")

    cols = ["permaticker", "month_end", "filing_date", "period_end", *values]
    return out[cols].sort_values(["permaticker", "month_end"]).reset_index(drop=True)


def load_fundamentals(
    keys: pd.DataFrame, raw_dir: Path = RAW_DIR, dimension: str = "ART"
) -> pd.DataFrame:
    """Read `fundamentals` + `tickers` from disk and build point-in-time fundamentals."""
    master = load_security_master(raw_dir)
    raw = pd.read_parquet(raw_dir / "fundamentals.parquet")
    return build_fundamentals(raw, master, keys, dimension)
