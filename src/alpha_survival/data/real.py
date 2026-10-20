"""Load the raw Sharadar Parquet files into our own clean, point-in-time schema.

Later steps (features, models) read only what this module returns, never raw
vendor columns, so swapping the data source means changing just this file.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
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

# How `tickers.table` names the stock-price table: row queries say "stocks", the
# bulk file still uses the legacy code "SEP".
PRICE_TABLE_CODES = {"stocks", "SEP"}


def build_security_master(tickers: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw `tickers` table into one row per stock, keyed on `permaticker`.

    Tickers get renamed and reused, so `permaticker` (permanent per company) is
    the key everything else joins on. Delisted stocks are kept on purpose:
    dropping them is survivorship bias.
    """
    # `tickers` repeats each stock once per Sharadar table (stocks, fundamentals,
    # insiders...). The `stocks` row describes the price history we trade on.
    priced = tickers.loc[tickers["table"].isin(PRICE_TABLE_CODES), list(SECURITY_MASTER_COLUMNS)]
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


# The standard academic universe: US common stock on NYSE, AMEX (NYSEMKT) and Nasdaq,
# i.e. CRSP share codes 10/11 and exchange codes 1-3 (e.g. Gu, Kelly & Xiu 2020).
# Leaves out ADRs (currency effects), preferreds (bond-like) and Canadian listings.
UNIVERSE_CATEGORIES = {
    "Domestic Common Stock",
    "Domestic Common Stock Primary Class",
    "Domestic Common Stock Secondary Class",
}
UNIVERSE_EXCHANGES = {"NYSE", "NYSEMKT", "NASDAQ"}


def select_universe(master: pd.DataFrame) -> pd.DataFrame:
    """Keep the stocks in our research universe; delisted ones stay in.

    Caveat: `category` and `exchange` are Sharadar's latest values, not their
    history, so a stock that moved to OTC before dying is dropped. On the full
    history only 6 stocks are on OTC, so the survivorship effect is negligible.
    """
    keep = master["category"].isin(UNIVERSE_CATEGORIES) & master["exchange"].isin(
        UNIVERSE_EXCHANGES
    )
    return master[keep].reset_index(drop=True)


def abdi_ranaldo_product(daily: pd.DataFrame) -> pd.Series:
    """One day-pair's term of the Abdi & Ranaldo (2017) bid-ask spread estimator.

    The close usually trades at the bid or the ask, while the mid-range
    eta = (log high + log low) / 2 sits near the true price. So how far the close
    sits from today's and tomorrow's mid-range reveals the spread:

        S^2 = 4 E[(c_t - eta_t)(c_t - eta_t+1)]       (c = log close)

    This returns (c_t - eta_t)(c_t - eta_t+1) for each pair of adjacent trading
    days, filed under the later day t+1, so it uses no price after that day.
    Average it over many pairs, then S = sqrt(max(4 x mean, 0)); one pair alone is
    mostly noise and often negative.

    Pairs touching a no-trade day (zero volume, or high == low: the vendor just
    repeats the close) are NaN: they would read as "zero spread" and make thinly
    traded stocks look cheap. Corwin & Schultz (2012) was tried first and rejected
    on this data for exactly that reason, and for an upward bias in liquid stocks.

    `daily`: permaticker, high, low, close, volume, sorted by permaticker then date.
    Split-adjusted prices are fine: only log differences within two adjacent days
    are used, which later restatements cannot change.
    """
    traded = (daily["volume"] > 0) & (daily["high"] > daily["low"]) & (daily["low"] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        close = np.log(daily["close"])
        mid = (np.log(daily["high"]) + np.log(daily["low"])) / 2
    by_stock = daily["permaticker"]
    prev_close, prev_mid = close.groupby(by_stock).shift(1), mid.groupby(by_stock).shift(1)
    prev_traded = traded.groupby(by_stock).shift(1, fill_value=False)
    product = (prev_close - prev_mid) * (prev_close - mid)
    return product.where(traded & prev_traded).rename("ar_product")


def build_monthly_returns(stocks: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Turn daily `stocks` prices into one total return per stock per calendar month.

    The return for month m runs from the last close in month m-1 to the last
    close in month m. Returns are computed from `closeadj` (split + dividend
    adjusted). Use `closeadj` ONLY for returns, never as a feature: Sharadar
    rewrites its past levels every time a dividend is paid, so a historical
    level is not what anyone could have seen on that date (look-ahead).

    Columns: permaticker, month_end, ret, last_trade_date, close_unadj, close_split_adj,
    n_days, dret_sum, dret_sumsq, ar_n, ar_sum.
    `last_trade_date` lets later steps spot stale prices (no trade near month end).
    `close_unadj` is the last close as actually traded (no split adjustment), the
    price an investor saw at month end; `close` is split-adjusted backwards, so a
    later reverse split would inflate it (look-ahead). `close_split_adj` is that
    `close`, kept only to multiply with SF1 share counts, which Sharadar restates
    for the same splits (see `add_market_cap`); never use it alone.

    `n_days`, `dret_sum`, `dret_sumsq`: count, sum and sum of squares of the
    month's daily `closeadj` returns (a stock's first day has none). Enough to
    rebuild the std of daily returns over any run of months (see
    `features.add_volatility`) without reading the daily data again.

    `ar_n`, `ar_sum`: count and sum of the month's Abdi-Ranaldo day-pair terms
    (`abdi_ranaldo_product`), so the spread can be estimated over any group of
    stocks and months (see `portfolio.estimated_spread_bps`).
    """
    if stocks.duplicated(["ticker", "date"]).any():
        raise ValueError("stocks has duplicate (ticker, date) rows")

    # `stocks` is keyed by ticker; map to the permanent ID. Sharadar keeps tickers
    # unique by suffixing reused ones (e.g. "ABC1"), so this mapping is 1:1.
    ids = master.set_index("ticker")["permaticker"]
    unmapped = set(stocks["ticker"]) - set(ids.index)
    if unmapped:
        raise ValueError(f"stocks tickers missing from security master: {sorted(unmapped)[:10]}")

    price_cols = ["ticker", "date", "closeadj", "close", "closeunadj", "high", "low", "volume"]
    daily = stocks[price_cols].assign(
        permaticker=lambda d: d["ticker"].map(ids),
        month=lambda d: d["date"].dt.to_period("M"),
    )
    # The vendor delivers newest-first; sort so "last" really means latest.
    daily = daily.sort_values(["permaticker", "date"])
    prev_daily = daily.groupby("permaticker")["closeadj"].shift(1)
    daily["dret"] = daily["closeadj"] / prev_daily - 1
    daily["ar_product"] = abdi_ranaldo_product(daily)
    moments = (
        daily.assign(dret_sq=daily["dret"] ** 2)
        .groupby(["permaticker", "month"])
        .agg(
            n_days=("dret", "count"),
            dret_sum=("dret", "sum"),
            dret_sumsq=("dret_sq", "sum"),
            ar_n=("ar_product", "count"),
            ar_sum=("ar_product", "sum"),
        )
    )

    monthly = (
        daily.drop(columns=["dret", "ar_product", "high", "low", "volume"])
        .groupby(["permaticker", "month"], as_index=False)
        .last()
        .rename(
            columns={
                "date": "last_trade_date",
                "closeunadj": "close_unadj",
                "close": "close_split_adj",
            }
        )
    )

    by_stock = monthly.groupby("permaticker")
    prev_close = by_stock["closeadj"].shift(1)
    prev_month = by_stock["month"].shift(1)
    monthly["ret"] = monthly["closeadj"] / prev_close - 1

    # If a stock skipped a whole month (halt, data gap), the change since the
    # previous row spans several months; don't pretend it is one month's return.
    gap = prev_month.notna() & (monthly["month"] != prev_month + 1)
    monthly.loc[gap, "ret"] = float("nan")

    monthly = monthly.merge(moments, on=["permaticker", "month"], how="left", validate="one_to_one")
    monthly["month_end"] = monthly["month"].dt.to_timestamp(how="end").dt.normalize()
    cols = ["permaticker", "month_end", "ret", "last_trade_date", "close_unadj", "close_split_adj"]
    cols += ["n_days", "dret_sum", "dret_sumsq", "ar_n", "ar_sum"]
    return monthly[cols].reset_index(drop=True)


# Sharadar `actions` types that name a distress exit: always penalised. For these
# and `delisted`, `date` is the last trade date (checked on the full history).
DISTRESS_EXITS = {"bankruptcyliquidation", "regulatorydelisting", "voluntarydelisting"}
# Fires for EVERY delisting, takeovers included (7,379 of 18,303 stocks with it were
# acquired), so on its own it says nothing about why the stock left.
GENERIC_EXIT = "delisted"
# Target side of a takeover: the last price already reflects the deal, so no penalty.
# On the full history these delist within a week 100% of the time, and every other
# acquisition* action co-occurs with one of them. `acquisitionof` / `mergerfrom` are
# the acquirer's side (it keeps trading) and must not exempt a stock.
TAKEOVER_ACTIONS = {"acquisitionby", "mergerto"}

# Average return after a performance delisting, which the last price misses.
# NYSE/AMEX: Shumway (1997); Nasdaq: Shumway & Warther (1999).
DELISTING_PENALTY = {"NYSE": -0.30, "NYSEMKT": -0.30, "NASDAQ": -0.55}
DEFAULT_DELISTING_PENALTY = -0.30  # the few BATS/OTC/NYSEARCA names


def add_delisting_returns(
    monthly: pd.DataFrame,
    actions: pd.DataFrame,
    master: pd.DataFrame,
    penalise_unexplained: bool = True,
) -> pd.DataFrame:
    """Compound a delisting penalty into the exit month of stocks that died in distress.

    Without this, a stock that goes bankrupt mid-month shows only the fall up to
    its last trade; the further loss to shareholders vanishes and distressed
    stocks look better than they were (Shumway 1997).

    Explicit distress exits are always penalised, takeover targets never. A bare
    `delisted` with no takeover is ambiguous (failure, or an uncoded buyout):
    penalised by default, which is closest to Shumway's performance-delisting
    rule; `penalise_unexplained=False` gives the other bound for robustness checks.

    Adds `delist_action` (the exit type, or NA) so later reports can count them.
    """
    penalised = actions["action"].isin(DISTRESS_EXITS)
    if penalise_unexplained:
        taken_over = actions.loc[actions["action"].isin(TAKEOVER_ACTIONS), "ticker"]
        penalised |= (actions["action"] == GENERIC_EXIT) & ~actions["ticker"].isin(taken_over)
    exits = actions.loc[penalised, ["ticker", "date", "action"]]
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


def drop_unfinished_months(monthly: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Drop months that had not ended by `as_of`, the last date the data covers.

    Data pulled mid-October gives an October row holding only half a month of
    return; a backtest would treat it as a full month. A month counts as ended
    once `as_of` reaches its last weekday. Holidays are ignored, so a month whose
    last weekday is a holiday waits for the next pull: conservative, never leaky.
    """
    month_ends = monthly["month_end"].drop_duplicates()
    last_weekday = month_ends.map(pd.offsets.BMonthEnd().rollback)
    ended = month_ends[last_weekday <= pd.Timestamp(as_of).normalize()]
    return monthly[monthly["month_end"].isin(ended)].reset_index(drop=True)


def load_monthly_returns(
    raw_dir: Path = RAW_DIR, as_of: pd.Timestamp | None = None, penalise_unexplained: bool = True
) -> pd.DataFrame:
    """Read `stocks` + `actions` and build delisting-adjusted monthly returns.

    Covers the research universe only (`select_universe`). `as_of` defaults to
    the latest price date in the file (when it was pulled).
    """
    master = load_security_master(raw_dir)
    universe = select_universe(master)
    # Read only what returns need: the full table is ~45M rows.
    stocks = pd.read_parquet(
        raw_dir / "stocks.parquet",
        columns=["ticker", "date", "closeadj", "close", "closeunadj", "high", "low", "volume"],
        filters=[("ticker", "in", list(universe["ticker"]))],
    )
    monthly = build_monthly_returns(stocks, master)
    actions = pd.read_parquet(raw_dir / "actions.parquet")
    monthly = add_delisting_returns(monthly, actions, master, penalise_unexplained)
    # Last, so a distress exit in the unfinished month doesn't warn as "unmatched".
    return drop_unfinished_months(monthly, as_of if as_of is not None else stocks["date"].max())


# `fundamentals` "dimensions": AR* = as reported (one row per original filing, dated by when it
# became public). MR* = most recent, i.e. silently restated later; its numbers were
# never visible on the date they sit at, so MR* is unusable for backtests.
POINT_IN_TIME_DIMENSIONS = {"ARQ", "ART", "ARY"}

# In `fundamentals`, `date` is the day the filing became public (not a trading day).
# Filings often land after the close, so a filing is usable from the day after
# that date, never on the date itself.
AVAILABILITY_LAG = pd.Timedelta(days=1)

# A filing older than this at month end is no longer used: the company stopped
# reporting (or the data has a gap), and years-old earnings would pose as current.
# Normally a new 10-Q/10-K lands every quarter, so the latest one is at most ~135
# days old; a year leaves ample room for late filers.
MAX_FILING_AGE = pd.Timedelta(days=365)

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
    A month whose latest filing is older than `MAX_FILING_AGE` gets all-NaN
    fundamentals (filing_date included), as if the stock had never filed.
    """
    if dimension not in POINT_IN_TIME_DIMENSIONS:
        raise ValueError(
            f"Dimension {dimension!r} is not point-in-time; use one of "
            f"{sorted(POINT_IN_TIME_DIMENSIONS)}"
        )
    rows = fundamentals[fundamentals["dimension"] == dimension]

    if rows.duplicated(["ticker", "date"]).any():
        raise ValueError(f"fundamentals {dimension} has duplicate (ticker, date) rows")
    # Impossible dates (10 of ~690k ART rows, mostly 1996-97): the real filing came
    # later, so trusting the date could leak. Drop them rather than fail the run.
    early = rows["date"] < rows["reportperiod"]
    if early.any():
        warnings.warn(
            f"dropped {early.sum()} {dimension} filings dated before their period ended",
            stacklevel=2,
        )
        rows = rows[~early]

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
    # "Latest filing with available_date <= month_end", per stock, and no match at
    # all if that one is too old. Tolerance counts from available_date, one lag
    # after filing_date, so subtract the lag to cap the age of filing_date itself.
    out = pd.merge_asof(
        left.sort_values("month_end"),
        filings.sort_values("available_date"),
        left_on="month_end",
        right_on="available_date",
        by="permaticker",
        direction="backward",
        tolerance=MAX_FILING_AGE - AVAILABILITY_LAG,
    ).drop(columns="available_date")

    cols = ["permaticker", "month_end", "filing_date", "period_end", *values]
    return out[cols].sort_values(["permaticker", "month_end"]).reset_index(drop=True)


def load_fundamentals(
    keys: pd.DataFrame, raw_dir: Path = RAW_DIR, dimension: str = "ART"
) -> pd.DataFrame:
    """Read `fundamentals` + `tickers` from disk and build point-in-time fundamentals.

    Reads only `dimension` rows for the stocks in `keys`: filings of companies
    with no price history (23 tickers on the full file) can never join the panel.
    """
    master = load_security_master(raw_dir)
    tickers = master.loc[master["permaticker"].isin(keys["permaticker"]), "ticker"]
    raw = pd.read_parquet(
        raw_dir / "fundamentals.parquet",
        filters=[("dimension", "==", dimension), ("ticker", "in", list(tickers))],
    )
    return build_fundamentals(raw, master, keys, dimension)


def build_panel(monthly: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Join returns and fundamentals into one row per (permaticker, month_end), plus the target.

    Every column on a row is known at its `month_end`, except `fwd_ret`: the return
    over the following month, i.e. what we forecast. Shifting it here, in one tested
    place, stops each model from re-deriving the target (and getting it backwards).

    `fwd_ret` is NaN when the next month is missing (gap, end of data), never a
    multi-month return. A distress exit next month is already in that month's
    return, so a stock that dies still counts against whoever held it.
    """
    keys = ["permaticker", "month_end"]
    if monthly.duplicated(keys).any():
        raise ValueError("monthly has duplicate (permaticker, month_end) rows")

    panel = monthly.sort_values(keys).reset_index(drop=True)
    by_stock = panel.groupby("permaticker")
    next_month = by_stock["month_end"].shift(-1).dt.to_period("M")
    panel["fwd_ret"] = by_stock["ret"].shift(-1)
    consecutive = next_month == panel["month_end"].dt.to_period("M") + 1
    panel.loc[~consecutive, "fwd_ret"] = float("nan")

    return panel.merge(fundamentals, on=keys, how="left", validate="one_to_one")


# Stocks below this price at month end are not bought that month: their quoted
# returns are dominated by tick size and bid-ask bounce, and costs eat any edge.
# $5 is the usual cut in the literature (e.g. Jegadeesh & Titman 2001).
MIN_PRICE = 5.0


def flag_investable(panel: pd.DataFrame, min_price: float = MIN_PRICE) -> pd.DataFrame:
    """Add `investable`: True if the stock traded at >= `min_price` at month end.

    A flag, not a filter. A row with investable=False may not be bought at its
    month_end, but stays in the panel: its return is history that later features
    (e.g. momentum) need, and a held stock that drops below the cut still costs
    its holder. Uses only the price at month_end itself, so it adds no look-ahead.
    """
    return panel.assign(investable=panel["close_unadj"] >= min_price)


def add_market_cap(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `market_cap` (USD): what the stock's shares were worth at month end.

    = close_split_adj x sharesbas x sharefactor, from the latest filing's
    cover-page share count (`sharefactor` converts e.g. depositary shares; SF1's
    own `marketcap` is built the same way). Sharadar restates SF1 share counts for
    every split up to today (checked: across 8,565 splits, filed counts jump at
    the split in 29 cases) and `close_split_adj` is scaled by the same splits, so
    those cancel in the product. What remains is the traded price times the filed
    count times the splits between filing and month_end: no look-ahead, and a
    split after the filing is not missed. Never multiply the restated count by
    `close_unadj`: that counts every later split twice.

    NaN when there are no fundamentals, or the filing reports zero shares.
    """
    shares = (panel["sharesbas"] * panel["sharefactor"]).where(panel["sharesbas"] > 0)
    return panel.assign(market_cap=panel["close_split_adj"] * shares)


def load_panel(
    raw_dir: Path = RAW_DIR,
    as_of: pd.Timestamp | None = None,
    dimension: str = "ART",
    penalise_unexplained: bool = True,
    min_price: float = MIN_PRICE,
) -> pd.DataFrame:
    """Read everything from disk and build the monthly modelling panel."""
    monthly = load_monthly_returns(raw_dir, as_of, penalise_unexplained)
    panel = build_panel(monthly, load_fundamentals(monthly, raw_dir, dimension))
    panel = add_market_cap(panel)
    return flag_investable(panel, min_price)
