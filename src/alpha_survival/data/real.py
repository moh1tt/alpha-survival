"""Load the raw Sharadar Parquet files into our own clean, point-in-time schema.

Later steps (features, models) read only what this module returns, never raw
vendor columns, so swapping the data source means changing just this file.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_survival.data.external import daily_factors
from alpha_survival.data.sharadar import RAW_DIR
from alpha_survival.ols import group_ols

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
    "siccode": "sic_code",
    "famaindustry": "fama_industry",
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


# Corwin & Schultz: a high/low ratio above this is a bad print, not a trading range.
CS_MAX_RANGE = 8.0


def corwin_schultz_spread(daily: pd.DataFrame) -> pd.Series:
    """Each day's Corwin & Schultz (2012) two-day bid-ask spread estimate, floored at 0.

    A day's high-low range holds both volatility and the bid-ask spread. Volatility
    grows with the time window, while the spread doesn't, so comparing two one-day
    ranges (beta) with the two-day range (gamma) separates them:

        alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2) - sqrt(gamma / (3 - 2 sqrt 2))
        S = 2 (e^alpha - 1) / (1 + e^alpha)

    Follows Corwin's code as used by OSAP for its BidAskSpread signal:
    - A day with high == low, no volume or a non-positive price has no range of its
      own; it reuses the stock's last valid range, shifted to contain its close.
    - Ranges with high / low > `CS_MAX_RANGE` are dropped.
    - If the previous close is outside today's range, today's range is shifted to
      it, so an overnight move is not read as spread.
    The estimate is filed under the later day of each pair, so it uses no price
    after that day. NaN when either day lacks a range. Not a trading-cost estimate
    here (rejected for that in favour of `abdi_ranaldo_product`): a signal only.

    `daily`: permaticker, high, low, close, volume, sorted by permaticker then date.
    Split-adjusted prices, so a split between the two days can't fake a range.
    """
    by_stock = daily["permaticker"]
    low, high, close = daily["low"], daily["high"], daily["close"].abs()
    valid = (low < high) & (low > 0) & (high > 0) & (close > 0) & (daily["volume"] > 0)
    # Last valid range up to today (today's own if valid).
    kept_low = low.where(valid).groupby(by_stock).ffill()
    kept_high = high.where(valid).groupby(by_stock).ffill()
    below, above = close < kept_low, close > kept_high
    low = low.where(valid, kept_low.mask(below, close).mask(above, kept_low + close - kept_high))
    high = high.where(
        valid, kept_high.mask(below, kept_high - (kept_low - close)).mask(above, close)
    )
    bad = (low <= 0) | (high / low > CS_MAX_RANGE)
    low, high = low.mask(bad), high.mask(bad)

    prev_low, prev_high = low.groupby(by_stock).shift(1), high.groupby(by_stock).shift(1)
    prev_close = close.groupby(by_stock).shift(1)
    # Overnight adjustment: shift today's range to the previous close if outside it.
    up = (prev_close < low) & (prev_close > 0)  # gapped up: previous close below today's low
    down = (prev_close > high) & (prev_close > 0)
    t_low = low.mask(up, prev_close).mask(down, low + (prev_close - high))
    t_high = high.mask(up, high - (low - prev_close)).mask(down, prev_close)

    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.log(t_high / t_low) ** 2 + np.log(prev_high / prev_low) ** 2
        gamma = np.log(np.fmax(t_high, prev_high) / np.fmin(t_low, prev_low)) ** 2
        const = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / const - np.sqrt(gamma / const)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    both = t_low.notna() & prev_low.notna() & (t_low > 0) & (prev_low > 0)
    return spread.clip(lower=0).where(both).rename("cs_spread")


def build_monthly_returns(
    stocks: pd.DataFrame, master: pd.DataFrame, factors: pd.DataFrame | None = None
) -> pd.DataFrame:
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

    More daily summaries for price and trading signals, all from days within the
    month (so known at month end):
    - `dret_max`: largest daily return; `dret_sumcube`: sum of cubed daily returns
      (with the count, sum and sum of squares, enough for skewness).
    - `close_max`: highest split-adjusted close. Only for ratios with other `close`
      values (e.g. 52-week high), where the backward split adjustment cancels.
    - `share_volume`, `dollar_volume`: sums of daily volume and close x volume.
      Sharadar's `volume` is split-adjusted like `close` (checked on AAPL's 2020
      4:1 split), so their product is the true dollars traded; `closeunadj` x
      `volume` would overstate pre-split days.
    - `illiq_sum`, `illiq_n`: sum and count of daily |return| / dollar volume over
      days that traded (Amihud 2002).
    - `n_trade_days`: days with a price row; `n_zero_volume`: those with no volume.
    - `cs_sum`, `cs_n`: sum and count of the month's Corwin-Schultz day-pair
      estimates (`corwin_schultz_spread`), for the BidAskSpread signal.
    - `market_tail`: the month's tail index of all stocks' daily returns pooled
      (`market_tail_index`), the same on every row of the month.

    With daily `factors` (`data.external.daily_factors`), also the factor-regression
    summaries of `daily_factor_summaries` (`FACTOR_SUMMARY_COLUMNS`).
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
    daily["cs_spread"] = corwin_schultz_spread(daily)
    dollars = daily["close"] * daily["volume"]
    moments = (
        daily.assign(
            dret_sq=daily["dret"] ** 2,
            dret_cube=daily["dret"] ** 3,
            dollars=dollars,
            illiq=(daily["dret"].abs() / dollars).where(dollars > 0),
            zero_volume=daily["volume"] == 0,
        )
        .groupby(["permaticker", "month"])
        .agg(
            n_days=("dret", "count"),
            dret_sum=("dret", "sum"),
            dret_sumsq=("dret_sq", "sum"),
            ar_n=("ar_product", "count"),
            ar_sum=("ar_product", "sum"),
            dret_max=("dret", "max"),
            dret_sumcube=("dret_cube", "sum"),
            close_max=("close", "max"),
            share_volume=("volume", "sum"),
            dollar_volume=("dollars", "sum"),
            illiq_sum=("illiq", "sum"),
            illiq_n=("illiq", "count"),
            n_trade_days=("date", "count"),
            n_zero_volume=("zero_volume", "sum"),
            cs_sum=("cs_spread", "sum"),
            cs_n=("cs_spread", "count"),
        )
    )

    moments["market_tail"] = (
        moments.index.get_level_values("month").map(market_tail_index(daily)).to_numpy()
    )
    if factors is not None:
        moments = moments.join(daily_factor_summaries(daily, factors), validate="one_to_one")

    monthly = (
        daily.drop(columns=["dret", "ar_product", "cs_spread", "high", "low", "volume"])
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
    cols += ["n_days", "dret_sum", "dret_sumsq", "ar_n", "ar_sum", *DAILY_SUMMARY_COLUMNS]
    if factors is not None:
        cols += FACTOR_SUMMARY_COLUMNS
    return monthly[cols].reset_index(drop=True)


PRICE_DELAY_LAGS = 4  # Hou & Moskowitz (2005): market returns of the 4 trading days before
# Price-delay regressors, in order: intercept, mktrf, mktrf lagged 1 .. 4 trading days.
PRICE_DELAY_K = 2 + PRICE_DELAY_LAGS
PRICE_DELAY_COLUMNS = [
    "pd_yy",
    *(f"pd_xy{i}" for i in range(PRICE_DELAY_K)),
    *(f"pd_xx{i}{j}" for i in range(PRICE_DELAY_K) for j in range(i, PRICE_DELAY_K)),
]

# Per-month summaries of daily factor regressions (see daily_factor_summaries).
FACTOR_SUMMARY_COLUMNS = [
    "xret_n",
    "xret_sum",
    "xret_sumsq",
    "mkt_sum",
    "mkt_sumsq",
    "xret_mkt_sum",
    "ff3_resid_sumsq",
    "ff3_resid_sumcube",
    "vix_n",
    "vix_beta",
    "lx_sum",
    "lx_sumsq",
    "lm_sum",
    "lm_sumsq",
    "lx_lm_sum",
    "lx_lm2_sum",
    "l3_n",
    "l3x_sum",
    "l3x_sumsq",
    "l3m_sum",
    "l3m_sumsq",
    "l3x_l3m_sum",
    *PRICE_DELAY_COLUMNS,
]


def daily_factor_summaries(daily: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Per (permaticker, month): summaries of daily excess returns y = dret - rf and
    their factor regressions, over the month's days that have y and the FF3 factors.

    `daily`: permaticker, month, date, dret. `factors`: daily rf, mktrf, smb, hml,
    dvix indexed by date. Days without factors (e.g. after the Ken French file ends)
    are left out, so a month after it has no count and no regression.
    - `xret_n`, `xret_sum`, `xret_sumsq`: count, sum and sum of squares of y.
    - `mkt_sum`, `mkt_sumsq`, `xret_mkt_sum`: sums of mktrf, mktrf^2 and y x mktrf,
      enough for a CAPM regression over any run of months.
    - `ff3_resid_sumsq`, `ff3_resid_sumcube`: sums of squared and cubed residuals of
      the month's own regression of y on mktrf, smb, hml with an intercept.
    - `vix_n`, `vix_beta`: days that also have a VIX change, and the month's slope of
      y on dvix with mktrf as a control.
    - `lx_*`, `lm_*`: sums of daily LOG excess returns x = log(1 + dret) - log(1 + rf)
      and m = log(1 + mktrf + rf) - log(1 + rf), their squares, x m and x m^2 (Ang,
      Chen & Xing 2006, as OSAP's CoskewACX). Counted by `xret_n`.
    - `l3_n`, `l3x_*`, `l3m_*`: count and sums of overlapping 3-day log excess
      returns (this day and the stock's two previous days with factors, which may
      fall in the month before), their squares and cross-product (Frazzini &
      Pedersen 2014). OSAP's BetaFP takes log(1 + dret - rf) and log(1 + mktrf):
      the same up to terms of order rf x return, about 1e-6 a day.
    - `pd_yy`, `pd_xy<i>`, `pd_xx<i><j>`: sums of y^2, x_i y and x_i x_j (i <= j) for
      the price-delay regressors x = (1, mktrf, mktrf of the 1 .. 4 previous trading
      days in the factor file; `PRICE_DELAY_COLUMNS`), over days with y and all of x.
      Enough for the regression of y on x, or on its first two columns, over any run
      of months (Hou & Moskowitz 2005). The lags reach back only, never forward.
    All from days up to the month's end, so known at its end.
    """
    d = daily[["permaticker", "month", "date", "dret"]].sort_values(["permaticker", "date"])
    market = factors["mktrf"].sort_index()
    lags = {f"pd_mkt_l{n}": market.shift(n) for n in range(1, PRICE_DELAY_LAGS + 1)}
    d = d.join(factors, on="date").join(pd.DataFrame(lags), on="date")
    groups = d.groupby(["permaticker", "month"], sort=True)
    codes = groups.ngroup().to_numpy()
    index = groups.size().index
    n = len(index)

    y = (d["dret"] - d["rf"]).to_numpy()
    mkt, smb, hml = (d[c].to_numpy() for c in ["mktrf", "smb", "hml"])
    dvix = d["dvix"].to_numpy()
    ones = np.ones(len(d))
    ff = ~np.isnan(y) & ~np.isnan(mkt) & ~np.isnan(smb) & ~np.isnan(hml)
    vx = ff & ~np.isnan(dvix)

    def total(values: np.ndarray, use: np.ndarray = ff) -> np.ndarray:
        return np.bincount(codes, weights=np.where(use, values, 0.0), minlength=n)

    b = group_ols(codes, n, y, [ones, mkt, smb, hml], ff)
    resid = y - b[codes, 0] - b[codes, 1] * mkt - b[codes, 2] * smb - b[codes, 3] * hml
    fitted = ~np.isnan(b[:, 0])
    vix_b = group_ols(codes, n, y, [ones, mkt, dvix], vx)

    rf = d["rf"].to_numpy()
    with np.errstate(invalid="ignore"):
        lx = np.log1p(d["dret"].to_numpy()) - np.log1p(rf)
        lm = np.log1p(mkt + rf) - np.log1p(rf)
    # 3-day sums over each stock's consecutive days with factors (rows are sorted by
    # stock and date): the day itself and the two before it, never a later day.
    rows = pd.Series(np.flatnonzero(ff))
    stock = d["permaticker"].to_numpy()[rows]
    l3x, l3m = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for target, values in [(l3x, lx), (l3m, lm)]:
        v = pd.Series(values[rows])
        by_stock = v.groupby(stock)
        target[rows] = (v + by_stock.shift(1) + by_stock.shift(2)).to_numpy()
    l3 = ~np.isnan(l3x)

    px = [ones, mkt, *(d[c].to_numpy() for c in lags)]
    pdm = ~np.isnan(y) & np.all([~np.isnan(x) for x in px], axis=0)
    delay = {"pd_yy": total(y * y, pdm)}
    delay |= {f"pd_xy{i}": total(px[i] * y, pdm) for i in range(PRICE_DELAY_K)}
    for i in range(PRICE_DELAY_K):
        for j in range(i, PRICE_DELAY_K):
            delay[f"pd_xx{i}{j}"] = total(px[i] * px[j], pdm)

    out = pd.DataFrame(
        {
            "xret_n": total(ones),
            "xret_sum": total(y),
            "xret_sumsq": total(y * y),
            "mkt_sum": total(mkt),
            "mkt_sumsq": total(mkt * mkt),
            "xret_mkt_sum": total(y * mkt),
            "ff3_resid_sumsq": np.where(fitted, total(resid**2), np.nan),
            "ff3_resid_sumcube": np.where(fitted, total(resid**3), np.nan),
            "vix_n": total(ones, vx),
            "vix_beta": vix_b[:, 2],
            "lx_sum": total(lx),
            "lx_sumsq": total(lx * lx),
            "lm_sum": total(lm),
            "lm_sumsq": total(lm * lm),
            "lx_lm_sum": total(lx * lm),
            "lx_lm2_sum": total(lx * lm * lm),
            "l3_n": total(ones, l3),
            "l3x_sum": total(l3x, l3),
            "l3x_sumsq": total(l3x * l3x, l3),
            "l3m_sum": total(l3m, l3),
            "l3m_sumsq": total(l3m * l3m, l3),
            "l3x_l3m_sum": total(l3x * l3m, l3),
            **delay,
        },
        index=index,
    )
    return out


TAIL_QUANTILE = 0.05  # Kelly & Jiang (2014): the worst 5% of the month's daily returns


def market_tail_index(daily: pd.DataFrame) -> pd.Series:
    """Per month: the Hill tail index of every stock's daily return pooled, the mean of
    log(r / p5) over returns r <= p5, with p5 the 5th percentile (Kelly & Jiang 2014;
    OSAP's code). Fatter left tails give a larger value. `daily`: month, dret.
    Uses only the month's own days, so it is known at the month's end.
    """
    rets = daily[["month", "dret"]].dropna()
    p5 = rets.groupby("month")["dret"].quantile(TAIL_QUANTILE)
    cut = rets["month"].map(p5)
    tail = rets[(rets["dret"] <= cut) & (cut < 0)]
    return np.log(tail["dret"] / tail["month"].map(p5)).groupby(tail["month"]).mean()


# Extra per-month daily summaries for price/trading signals (see build_monthly_returns).
DAILY_SUMMARY_COLUMNS = [
    "dret_max",
    "dret_sumcube",
    "close_max",
    "share_volume",
    "dollar_volume",
    "illiq_sum",
    "illiq_n",
    "n_trade_days",
    "n_zero_volume",
    "cs_sum",
    "cs_n",
    "market_tail",
]


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
    # astype(float): with no exits, pandas >= 3.1 keeps the empty map result as str.
    exits["penalty"] = (
        exits["exchange"].map(DELISTING_PENALTY).fillna(DEFAULT_DELISTING_PENALTY).astype(float)
    )

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
    monthly = build_monthly_returns(stocks, master, daily_factors(raw_dir / "external"))
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
    keys: pd.DataFrame,
    raw_dir: Path = RAW_DIR,
    dimension: str = "ART",
    fields: list[str] | None = None,
) -> pd.DataFrame:
    """Read `fundamentals` + `tickers` from disk and build point-in-time fundamentals.

    Reads only `dimension` rows for the stocks in `keys`: filings of companies
    with no price history (23 tickers on the full file) can never join the panel.
    `fields`: the value columns to read (default: all).
    """
    master = load_security_master(raw_dir)
    tickers = master.loc[master["permaticker"].isin(keys["permaticker"]), "ticker"]
    columns = None if fields is None else ["ticker", "dimension", "date", "reportperiod", *fields]
    raw = pd.read_parquet(
        raw_dir / "fundamentals.parquet",
        columns=columns,
        filters=[("dimension", "==", dimension), ("ticker", "in", list(tickers))],
    )
    return build_fundamentals(raw, master, keys, dimension)


# Quarterly (ARQ) fields carried next to the trailing-twelve-month ones, with suffix
# `_q`, for signals built on single quarters (e.g. roaq, NumEarnIncrease). A subset:
# all ~105 would add ~1.5 GB to the panel. Add a field here when a signal needs it.
# ARQ flows (revenue, income, cash flows) are single quarters, not year to date:
# AAPL's four 2022 quarters of revenue sum to its ART revenue for 2022.
QUARTERLY_FIELDS = [
    "revenue",
    "cor",
    "gp",
    "sgna",
    "rnd",
    "opinc",
    "intexp",
    "taxexp",
    "netinc",
    "eps",
    "depamor",
    "ncfo",
    "capex",
    "ncfdiv",
    "assets",
    "assetsc",
    "cashneq",
    "receivables",
    "inventory",
    "ppnenet",
    "liabilities",
    "liabilitiesc",
    "debt",
    "debtc",
    "debtnc",
    "equity",
    "sharesbas",
]
QUARTERLY_SUFFIX = "_q"


def add_quarterly_fundamentals(panel: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    """Join quarterly fundamentals (`build_fundamentals(..., "ARQ")`) onto the panel.

    Every column but the keys gets `QUARTERLY_SUFFIX`: `netinc_q` is the latest
    quarter's net income, `filing_date_q` / `period_end_q` its filing and quarter
    end. The same point-in-time rule as for ART: only filings public before
    month_end, none older than `MAX_FILING_AGE`.
    """
    keys = ["permaticker", "month_end"]
    values = [c for c in quarterly.columns if c not in keys]
    renamed = quarterly.rename(columns={c: c + QUARTERLY_SUFFIX for c in values})
    return panel.merge(renamed, on=keys, how="left", validate="one_to_one")


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


def add_industry_codes(panel: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Add `sic_code` (Standard Industrial Classification) and `fama_industry`
    (Sharadar's Fama-French 48-industry name, e.g. "Banking").

    CAUTION: Sharadar keeps only the CURRENT codes, so a firm that changed industry
    carries its latest code back through its history: a mild look-ahead. Use them
    only where a replication needs an industry (e.g. dropping financials, SIC
    6000-6999, or industry averages) and flag those signals as approximate.
    Industry changes are rare.
    """
    codes = master[["permaticker", "sic_code", "fama_industry"]]
    return panel.merge(codes, on="permaticker", how="left", validate="many_to_one")


# Dividend frequency codes as CRSP's cd3, inferred from the months since the stock's
# previous dividend month (Sharadar has no frequency field): (max gap, code).
DIVIDEND_FREQUENCIES = [(1, 2), (4, 3), (8, 4), (14, 5)]  # monthly, quarterly, semi, annual
UNKNOWN_FREQUENCY = 0  # first dividend, or a gap over 14 months


# ExchSwitch: moves to a more prestigious exchange (Dharan & Ikenberry 1995).
EXCHANGE_UPGRADES = {("NYSEMKT", "NYSE"), ("NASDAQ", "NYSE"), ("NASDAQ", "NYSEMKT")}
ACTION_MONTH_COLUMNS = ["div_amount", "div_freq", "exch_change", "exch_upgrade", "spun_off"]


def build_action_months(actions: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Per (permaticker, month_end) with a corporate action we use (Sharadar ACTIONS):

    - `div_amount`: the sum of the month's cash dividends per share (Sharadar's are
      split-adjusted like `close`, so divide by `close_split_adj`); `div_freq`: a
      CRSP-style frequency code from the months since the previous dividend month
      (`DIVIDEND_FREQUENCIES`; the first dividend, or one after 14 months, is
      `UNKNOWN_FREQUENCY`). Dated by the ex-date. Sharadar does not mark special
      dividends: all count, unlike OSAP's ordinary ones.
    - `exch_change`: 1 if the stock changed exchange this month (any pair, OTC
      included); `exch_upgrade`: 1 if the move was up (`EXCHANGE_UPGRADES`).
    - `spun_off`: 1 if the stock was spun off from another company this month.
    Other columns NaN. Each uses that month's and earlier actions only.
    """
    ids = master.set_index("ticker")["permaticker"]
    acts = actions.assign(
        permaticker=actions["ticker"].map(ids), month=actions["date"].dt.to_period("M")
    ).dropna(subset=["permaticker"])
    keys = ["permaticker", "month"]

    div = acts[(acts["action"] == "dividend") & acts["value"].notna()]
    months = div.groupby(keys, as_index=False)["value"].sum()
    months = months.rename(columns={"value": "div_amount"})
    index = months["month"].dt.year * 12 + months["month"].dt.month
    gap = index - index.groupby(months["permaticker"]).shift(1)
    freq = pd.Series(UNKNOWN_FREQUENCY, index=months.index)
    for limit, code in reversed(DIVIDEND_FREQUENCIES):
        freq[gap <= limit] = code
    months["div_freq"] = freq

    def exchanges(action: str, name: str) -> pd.DataFrame:
        rows = acts.loc[acts["action"] == action].reindex(columns=[*keys, "date", "contraname"])
        return rows.rename(columns={"contraname": name})

    moves = exchanges("exchangefrom", "old").merge(
        exchanges("exchangeto", "new"), on=[*keys, "date"], how="outer"
    )
    pairs = zip(moves["old"], moves["new"], strict=True)
    moves["exch_upgrade"] = [float(pair in EXCHANGE_UPGRADES) for pair in pairs]
    moves = moves.groupby(keys, as_index=False).agg(exch_upgrade=("exch_upgrade", "max"))
    moves["exch_change"] = 1.0

    spun = acts.loc[acts["action"] == "spunofffrom", keys].drop_duplicates()
    out = months.merge(moves, on=keys, how="outer").merge(
        spun.assign(spun_off=1.0), on=keys, how="outer"
    )
    out["month_end"] = out.pop("month").dt.to_timestamp(how="end").dt.normalize()
    out["permaticker"] = out["permaticker"].astype("int64")
    return out[["permaticker", "month_end", *ACTION_MONTH_COLUMNS]].sort_values(
        ["permaticker", "month_end"], ignore_index=True
    )


def add_action_months(panel: pd.DataFrame, action_months: pd.DataFrame) -> pd.DataFrame:
    """Join `build_action_months` onto the panel; months without one are NaN."""
    keys = ["permaticker", "month_end"]
    months = action_months.astype({"month_end": panel["month_end"].dtype})
    return panel.merge(months, on=keys, how="left", validate="one_to_one")


def load_panel(
    raw_dir: Path = RAW_DIR,
    as_of: pd.Timestamp | None = None,
    dimension: str = "ART",
    penalise_unexplained: bool = True,
    min_price: float = MIN_PRICE,
    quarterly: bool = True,
) -> pd.DataFrame:
    """Read everything from disk and build the monthly modelling panel.

    `quarterly`: also join `QUARTERLY_FIELDS` from ARQ filings (suffix `_q`).
    """
    monthly = load_monthly_returns(raw_dir, as_of, penalise_unexplained)
    panel = build_panel(monthly, load_fundamentals(monthly, raw_dir, dimension))
    if quarterly:
        arq = load_fundamentals(monthly, raw_dir, "ARQ", QUARTERLY_FIELDS)
        panel = add_quarterly_fundamentals(panel, arq)
    master = load_security_master(raw_dir)
    panel = add_industry_codes(add_market_cap(panel), master)
    actions = pd.read_parquet(raw_dir / "actions.parquet")
    panel = add_action_months(panel, build_action_months(actions, master))
    return flag_investable(panel, min_price)
