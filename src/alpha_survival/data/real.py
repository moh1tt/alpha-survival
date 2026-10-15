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
    """Turn the raw TICKERS table into one row per stock, keyed on `permaticker`.

    Tickers get renamed and reused, so `permaticker` (permanent per company) is
    the key everything else joins on. Delisted stocks are kept on purpose:
    dropping them is survivorship bias.
    """
    # TICKERS repeats each stock once per Sharadar table (SEP, SF1, SF2...).
    # The SEP row describes the price history, which is what we trade on.
    sep = tickers.loc[tickers["table"] == "SEP", list(SECURITY_MASTER_COLUMNS)]
    master = sep.rename(columns=SECURITY_MASTER_COLUMNS)

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
    """Read `TICKERS.parquet` from disk and build the security master."""
    return build_security_master(pd.read_parquet(raw_dir / "TICKERS.parquet"))


def build_monthly_returns(sep: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Turn daily SEP prices into one total return per stock per calendar month.

    The return for month m runs from the last close in month m-1 to the last
    close in month m. Returns are computed from `closeadj` (split + dividend
    adjusted). Use `closeadj` ONLY for returns, never as a feature: Sharadar
    rewrites its past levels every time a dividend is paid, so a historical
    level is not what anyone could have seen on that date (look-ahead).

    Columns: permaticker, month_end, ret, last_trade_date.
    `last_trade_date` lets later steps spot stale prices (no trade near month end).
    """
    if sep.duplicated(["ticker", "date"]).any():
        raise ValueError("SEP has duplicate (ticker, date) rows")

    # SEP is keyed by ticker; map to the permanent ID. Sharadar keeps tickers
    # unique by suffixing reused ones (e.g. "ABC1"), so this mapping is 1:1.
    ids = master.set_index("ticker")["permaticker"]
    unmapped = set(sep["ticker"]) - set(ids.index)
    if unmapped:
        raise ValueError(f"SEP tickers missing from security master: {sorted(unmapped)[:10]}")

    daily = sep[["ticker", "date", "closeadj"]].assign(
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


# Sharadar ACTIONS types where shareholders likely lost money on exit. Names verified
# against SHARADAR/INDICATORS (table=ACTIONTYPES); for all four, `date` is the last
# trade date. Acquisitions/mergers are not here: the last price already reflects
# the deal, so their delisting return is ~0 and we leave them alone.
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
    """Read SEP + ACTIONS and build delisting-adjusted monthly returns."""
    master = load_security_master(raw_dir)
    monthly = build_monthly_returns(pd.read_parquet(raw_dir / "SEP.parquet"), master)
    return add_delisting_returns(monthly, pd.read_parquet(raw_dir / "ACTIONS.parquet"), master)
