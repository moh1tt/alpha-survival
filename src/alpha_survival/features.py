"""Turn the monthly panel into predictive features (signals).

Every feature on a row must be computable from information known at that row's
`month_end`: earlier rows of the same stock, or the row's own known columns.
Never `fwd_ret` (the target) and never later rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 12-1 momentum compounds the monthly returns of t-11 .. t-1: from the close at
# the end of t-12 to the close at the end of t-1 (Jegadeesh & Titman 1993).
# Month t itself is skipped because one-month returns tend to reverse
# (Jegadeesh 1990), which would pollute the signal.
MOMENTUM_MONTHS = 11


def _sorted_by_stock(panel: pd.DataFrame) -> pd.DataFrame:
    keys = ["permaticker", "month_end"]
    if panel.duplicated(keys).any():
        raise ValueError("panel has duplicate (permaticker, month_end) rows")
    return panel.sort_values(keys).reset_index(drop=True)


def _past_return(panel: pd.DataFrame, first: int, last: int) -> pd.Series:
    """Compounded return of months t-first .. t-last (lags), per stock.

    `panel` must be sorted by stock and month. NaN unless every monthly return in
    the window exists and the rows are consecutive months, so a gap in the price
    history never stretches the window. Uses earlier rows only.
    """
    by_stock = panel.groupby("permaticker")
    growth = pd.Series(1.0, index=panel.index)
    for lag in range(first, last + 1):
        growth = growth * (1 + by_stock["ret"].shift(lag))  # NaN anywhere -> NaN
    # The oldest row must be exactly `last` months back; otherwise rows are
    # missing and the window spans more months than it should.
    oldest = by_stock["month_end"].shift(last).dt.to_period("M")
    consecutive = oldest == panel["month_end"].dt.to_period("M") - last
    return (growth - 1).where(consecutive)


def add_momentum(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `mom_12_1`: the compounded return over months t-11 .. t-1.

    NaN unless all 11 monthly returns exist and come from consecutive months, so a
    gap in the price history never turns into a longer window passed off as a year.
    """
    panel = _sorted_by_stock(panel)
    return panel.assign(mom_12_1=_past_return(panel, 1, MOMENTUM_MONTHS))


# Other windows of past returns, as (first lag, last lag) in months, matching the
# Open Source Asset Pricing code (Chen & Zimmermann 2022), whose Mom12m uses lags
# 1-11 like our mom_12_1. Each signal's sign is in run.PUBLISHED_SIGN.
PAST_RETURN_WINDOWS = {
    "mom_6m": (1, 5),  # Mom6m: Jegadeesh & Titman 1993
    "int_mom": (7, 12),  # IntMom: Novy-Marx 2012, intermediate momentum
    "mr_reversal": (13, 18),  # MRreversal: De Bondt & Thaler 1985, medium-run reversal
    "lr_reversal": (13, 36),  # LRreversal: De Bondt & Thaler 1985, long-run reversal
}
# MomRev (Chan & Ko 2006): the 36-month window for "long-run loser/winner".
MOM_REV_LONG = (13, 36)


def add_past_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the momentum and reversal family: `PAST_RETURN_WINDOWS` plus `mom_rev`.

    `mom_rev` = 1 for stocks in the top fifth on 6-month momentum and the bottom
    fifth on 36-month return (recent winners that were long-run losers), 0 for the
    opposite, NaN for everyone else. Fifths are formed each month across the rows
    passed in, from returns known at month_end: no look-ahead.
    """
    panel = _sorted_by_stock(panel)
    added = {name: _past_return(panel, *lags) for name, lags in PAST_RETURN_WINDOWS.items()}
    long_run = _past_return(panel, *MOM_REV_LONG)

    def fifth(values: pd.Series) -> pd.Series:
        return values.groupby(panel["month_end"]).transform(
            lambda v: pd.qcut(v, 5, labels=False, duplicates="drop") + 1
        )

    short_q, long_q = fifth(added["mom_6m"]), fifth(long_run)
    mom_rev = pd.Series(np.nan, index=panel.index)
    mom_rev[(short_q == 5) & (long_q == 1)] = 1.0
    mom_rev[(short_q == 1) & (long_q == 5)] = 0.0
    return panel.assign(**added, mom_rev=mom_rev)


def _season_lags(first_year: int, last_year: int) -> list[int]:
    """Lags hitting the forecast month's calendar month in those past years.

    The row at t forecasts month t+1, so year y back is lag 12y - 1 (11, 23, ...).
    """
    return [12 * y - 1 for y in range(first_year, last_year + 1)]


def _off_season_lags(first_year: int, last_year: int) -> list[int]:
    """All lags in those past years except the forecast month's calendar month."""
    season = set(_season_lags(first_year, last_year))
    return [k for k in range(12 * (first_year - 1), 12 * last_year) if k not in season]


# Seasonality (OSAP's code; Heston & Sadka 2008 and follow-ups): average monthly
# return over sets of past months, as lags. Same-month averages predict positively,
# other-month averages negatively (see run.PUBLISHED_SIGN).
SEASONAL_LAGS = {
    "mom_season_short": _season_lags(1, 1),  # MomSeasonShort: lag 11
    "mom_season": _season_lags(2, 5),  # MomSeason
    "mom_season_6y": _season_lags(6, 10),  # MomSeason06YrPlus
    "mom_season_11y": _season_lags(11, 15),  # MomSeason11YrPlus
    "mom_season_16y": _season_lags(16, 20),  # MomSeason16YrPlus
    "mom_offseason_12m": list(range(1, 11)),  # Mom12mOffSeason: lags 1-10
    "mom_offseason": _off_season_lags(2, 5),  # MomOffSeason
    "mom_offseason_6y": _off_season_lags(6, 10),  # MomOffSeason06YrPlus
    "mom_offseason_11y": _off_season_lags(11, 15),  # MomOffSeason11YrPlus
    "mom_offseason_16y": _off_season_lags(16, 20),  # MomOffSeason16YrPlus
}


def add_seasonality(panel: pd.DataFrame) -> pd.DataFrame:
    """Add each `SEASONAL_LAGS` feature: the mean `ret` over its lags.

    Lags are calendar months, not rows: returns are laid out on a full month x
    stock grid, so a stock's gap can never shift which calendar month a lag hits.
    Missing months are skipped, as in OSAP (who count them as 0% instead); NaN
    only if none of the lags has a return. Earlier months only: no look-ahead.
    """
    panel = _sorted_by_stock(panel)
    returns = _calendar_grid(panel, "ret")
    grid = returns.to_numpy()

    added = {}
    for name, lags in SEASONAL_LAGS.items():
        total, count = np.zeros_like(grid), np.zeros_like(grid)
        for lag in lags:
            if lag >= len(grid):
                continue
            past = np.full_like(grid, np.nan)
            past[lag:] = grid[:-lag]
            seen = ~np.isnan(past)
            total += np.where(seen, past, 0.0)
            count += seen
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, total / count, np.nan)
        added[name] = _at_rows(pd.DataFrame(mean), panel, returns)
    return panel.assign(**added)


def _calendar_grid(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    """`column` as a month x stock table over every calendar month in the panel.

    Months a stock has no row are NaN, so shifting or rolling this table moves by
    calendar months, never by rows: a gap cannot stretch a window.
    """
    wide = panel.pivot(index="month_end", columns="permaticker", values=column)
    calendar = pd.date_range(wide.index.min(), wide.index.max(), freq="ME")
    if not wide.index.isin(calendar).all():
        raise ValueError("month_end values must be calendar month ends")
    return wide.reindex(calendar)


def _at_rows(values: pd.DataFrame, panel: pd.DataFrame, like: pd.DataFrame) -> np.ndarray:
    """Read a month x stock result (shaped like the grid `like`) back onto panel rows."""
    rows = like.index.get_indexer(panel["month_end"])
    cols = like.columns.get_indexer(panel["permaticker"])
    return values.to_numpy()[rows, cols]


# Price and trading signals from the monthly daily-return summaries (OSAP's code).
# Each sign is in run.PUBLISHED_SIGN.
MIN_SKEW_DAYS = 15  # ReturnSkew: OSAP requires 15 trading days in the month
HIGH52_MONTHS = 12  # High52: this month and the 11 before
ILLIQ_MONTHS = 12  # Illiquidity: 12 monthly Amihud means, all required
DOLVOL_LAG = 2  # DolVol: dollar volume two months back
VOLSD_MONTHS, VOLSD_MIN = 36, 24  # VolSD: std of monthly volume, >= 24 of 36 months


def add_daily_price_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `max_ret`, `return_skew`, `high_52`, `illiquidity`, `dol_vol`, `vol_sd`.

    Built from `data.real.DAILY_SUMMARY_COLUMNS`, so only days up to each
    month_end. Windows run over calendar months (`_calendar_grid`).
    - `max_ret`: the month's largest daily return (MaxRet; Bali, Cakici & Whitelaw 2011).
    - `return_skew`: skewness of the month's daily returns, NaN with fewer than
      `MIN_SKEW_DAYS` trading days (ReturnSkew).
    - `high_52`: month-end close over the highest close of the last 12 months
      (High52; George & Hwang 2004). Both split-adjusted the same way, so the
      backward adjustment cancels in the ratio.
    - `illiquidity`: mean over 12 months of each month's mean daily |return| per
      dollar traded (Illiquidity; Amihud 2002); NaN unless all 12 exist.
    - `dol_vol`: log of dollars traded two months back (DolVol). OSAP uses monthly
      volume x month-end price; we sum the actual daily dollars.
    - `vol_sd`: std of monthly share volume over 36 months, >= 24 required (VolSD).
    """
    panel = _sorted_by_stock(panel)

    n, s1 = panel["n_days"], panel["dret_sum"]
    mean = s1 / n
    m2 = panel["dret_sumsq"] / n - mean**2
    m3 = panel["dret_sumcube"] / n - 3 * mean * panel["dret_sumsq"] / n + 2 * mean**3
    skew = (m3 / m2**1.5).where((panel["n_trade_days"] >= MIN_SKEW_DAYS) & (m2 > 0))

    close_max = _calendar_grid(panel, "close_max")
    high = close_max.rolling(HIGH52_MONTHS, min_periods=1).max()

    monthly_illiq = panel["illiq_sum"] / panel["illiq_n"].where(panel["illiq_n"] > 0)
    illiq = _calendar_grid(panel.assign(_m=monthly_illiq), "_m")
    illiq_12m = illiq.rolling(ILLIQ_MONTHS, min_periods=ILLIQ_MONTHS).mean()

    dollars = _calendar_grid(panel, "dollar_volume")
    lagged = dollars.shift(DOLVOL_LAG)
    volume = _calendar_grid(panel, "share_volume")
    vol_sd = volume.rolling(VOLSD_MONTHS, min_periods=VOLSD_MIN).std()

    with np.errstate(divide="ignore"):
        dol_vol = np.log(lagged.where(lagged > 0))
    return panel.assign(
        max_ret=panel["dret_max"],
        return_skew=skew,
        high_52=panel["close_split_adj"] / _at_rows(high, panel, close_max),
        illiquidity=_at_rows(illiq_12m, panel, illiq),
        dol_vol=_at_rows(dol_vol, panel, dollars),
        vol_sd=_at_rows(vol_sd, panel, volume),
    )


VOLTREND_MONTHS, VOLTREND_MIN = 60, 30  # VolumeTrend: >= 30 of the last 60 months


def _rolling_slope(grid: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    """OLS slope of each column on time (months), over a rolling calendar window.

    Closed form from rolling sums over the months that have a value:
    slope = (n Sxy - Sx Sy) / (n Sxx - Sx^2).
    """
    seen = grid.notna()
    t = pd.Series(np.arange(len(grid), dtype=float), index=grid.index)
    x = seen.mul(t, axis=0).where(seen)

    def total(values: pd.DataFrame) -> pd.DataFrame:
        return values.rolling(window, min_periods=min_periods).sum()

    n = seen.astype(float).rolling(window, min_periods=1).sum()
    sx, sy, sxy, sxx = total(x), total(grid), total(x * grid), total(x * x)
    denom = n * sxx - sx**2
    return ((n * sxy - sx * sy) / denom.where(denom > 0)).where(n >= min_periods)


def add_trend_price_age(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `volume_trend`, `price` and `firm_age` (OSAP's code, published sign -1 each).

    - `volume_trend`: slope of monthly share volume on time over the last 60
      calendar months (>= 30 with volume), divided by mean volume over the same
      months (VolumeTrend; Haugen & Baker 1996). Volume is split-adjusted, so a
      split is not a trend; OSAP's raw CRSP volume would jump at one.
    - `price`: log of the unadjusted month-end close (Price; Blume & Husic 1973).
    - `firm_age`: calendar months since the stock's first month in our data, counting
      that month as 1 (FirmAge; Barry & Brown 1984). Our data start in Dec 1997, so
      a stock already listed then gets a lower bound: still older than every later
      listing, tied with the others listed then. (OSAP sets such stocks to NaN, which
      in CRSP since 1926 is rare; for us it would be thousands of the oldest firms.)
    Each uses this month and earlier only.
    """
    panel = _sorted_by_stock(panel)

    volume = _calendar_grid(panel, "share_volume")
    slope = _rolling_slope(volume, VOLTREND_MONTHS, VOLTREND_MIN)
    mean = volume.rolling(VOLTREND_MONTHS, min_periods=VOLTREND_MIN).mean()
    trend = slope / mean.where(mean > 0)

    month = panel["month_end"].dt.year * 12 + panel["month_end"].dt.month
    age = month - month.groupby(panel["permaticker"]).transform("min") + 1

    return panel.assign(
        volume_trend=_at_rows(trend, panel, volume),
        price=np.log(panel["close_unadj"].where(panel["close_unadj"] > 0)),
        firm_age=age.astype(float),
    )


def add_profitability(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `gp_a`: gross profit (trailing twelve months) over total assets.

    Firms that turn assets into more gross profit earn higher returns, especially
    alongside value (Novy-Marx 2013). Both inputs come from the same filing, so
    the currency cancels. NaN where cost of revenue (`cor`) is 0: there gross
    profit is just revenue (banks, insurers), and revenue over a bank's balance
    sheet would rank every bank least profitable. Decided from the filing itself,
    so point-in-time, unlike today's sector label. NaN if assets <= 0 or inputs are
    missing; negative gross profit is kept. Row's own columns only: no look-ahead.
    """
    gross_profit = panel["gp"].where(panel["cor"] != 0)
    assets = panel["assets"].where(panel["assets"] > 0)
    return panel.assign(gp_a=gross_profit / assets)


ASSET_GROWTH_MONTHS = 12


def add_asset_growth(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `asset_growth`: total assets now over total assets 12 months ago, minus 1.

    Fast-growing firms tend to earn less afterwards (Cooper, Gulen & Schill 2008),
    so expect a NEGATIVE relation with `fwd_ret`. "Now" is the row's own `assets`
    (latest filing public at month_end); "12 months ago" is `assets` on the same
    stock's row dated exactly 12 calendar months earlier, i.e. what was public then.
    Both were known by month_end: no look-ahead. Matched on the calendar month, not
    12 rows back, so a gap in the history gives NaN rather than a longer window.
    NaN if either value is missing or <= 0.
    """
    keys = ["permaticker", "month_end"]
    if panel.duplicated(keys).any():
        raise ValueError("panel has duplicate (permaticker, month_end) rows")

    # Each row's assets, relabelled as the "year-ago" value of the row 12 months later.
    shifted_month = panel["month_end"].dt.to_period("M") + ASSET_GROWTH_MONTHS
    year_ago = pd.DataFrame(
        {
            "permaticker": panel["permaticker"],
            "month_end": shifted_month.dt.to_timestamp(how="end").dt.normalize(),
            "assets_year_ago": panel["assets"],
        }
    ).astype({"month_end": panel["month_end"].dtype})
    matched = panel[keys].merge(year_ago, on=keys, how="left", validate="one_to_one")

    now = panel["assets"].where(panel["assets"] > 0).to_numpy()
    before = matched["assets_year_ago"].where(matched["assets_year_ago"] > 0).to_numpy()
    return panel.assign(asset_growth=now / before - 1)


# Volatility over months t-2 .. t, from daily returns; fewer days than this -> NaN.
VOLATILITY_MONTHS = 3
VOLATILITY_MIN_DAYS = 40


def add_volatility(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `vol_3m`: std of daily returns over months t-2 .. t (not annualised).

    Volatile stocks have earned less than calm ones (Ang, Hodrick, Xing & Zhang
    2006), so expect a NEGATIVE relation with `fwd_ret`. Total, not idiosyncratic,
    volatility. Built from each month's daily-return count, sum and sum of squares
    (`n_days`, `dret_sum`, `dret_sumsq`), pooled over the same stock's rows for the
    3 calendar months ending at t; a missing month just adds no days. All daily
    returns are on or before month_end: no look-ahead.
    """
    keys = ["permaticker", "month_end"]
    if panel.duplicated(keys).any():
        raise ValueError("panel has duplicate (permaticker, month_end) rows")

    moments = ["n_days", "dret_sum", "dret_sumsq"]
    own = panel[keys + moments].fillna({m: 0 for m in moments})
    pooled = own[moments].copy()
    for lag in range(1, VOLATILITY_MONTHS):
        # Row from `lag` calendar months back, relabelled to this month.
        shifted = own.assign(
            month_end=(own["month_end"].dt.to_period("M") + lag)
            .dt.to_timestamp(how="end")
            .dt.normalize()
            .astype(own["month_end"].dtype)
        )
        back = own[keys].merge(shifted, on=keys, how="left", validate="one_to_one")
        pooled += back[moments].fillna(0).to_numpy()

    n, total, total_sq = pooled["n_days"], pooled["dret_sum"], pooled["dret_sumsq"]
    enough = n >= VOLATILITY_MIN_DAYS
    variance = (total_sq - total**2 / n.where(enough)) / (n.where(enough) - 1)
    return panel.assign(vol_3m=np.sqrt(variance.clip(lower=0)).to_numpy())


def add_reversal(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `rev_1m`: the stock's return over month t itself.

    Last month's losers tend to bounce back and winners to give some back
    (Jegadeesh 1990), so expect a NEGATIVE relation with `fwd_ret`; this is why
    momentum skips month t. Part of the effect is bid-ask bounce in month-end
    closes, which the $5 investable cut dampens. The row's own `ret`, known at
    month_end: no look-ahead. NaN where `ret` is (e.g. after a gap).
    """
    return panel.assign(rev_1m=panel["ret"])


def add_size(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `log_mcap`: log of the month-end market cap (USD).

    Small firms have historically earned higher returns (Banz 1981). The log puts
    a $10M -> $100M step on par with $10B -> $100B, so megacaps don't dominate.
    Uses only the row's own `market_cap`, known at month_end. NaN if it is missing.
    """
    market_cap = panel["market_cap"].where(panel["market_cap"] > 0)
    return panel.assign(log_mcap=np.log(market_cap))


def add_value(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `bm`: book-to-market, latest filed book equity over month-end market cap.

    Cheap stocks (high B/M) have historically earned more (Fama & French 1992).
    "Timely" version (Asness & Frazzini 2013): the latest filing public at month_end
    against today's price, rather than Fama-French's December book value used for
    a year. `equityusd`, not `equity`: some filers report in other currencies and
    `market_cap` is in USD. NaN if book equity <= 0 (ratio is meaningless) or
    either input is missing. Uses only the row's own columns: no look-ahead.
    """
    book = panel["equityusd"].where(panel["equityusd"] > 0)
    market = panel["market_cap"].where(panel["market_cap"] > 0)
    return panel.assign(bm=book / market)


def rank_normalise(panel: pd.DataFrame, columns: list[str], suffix: str = "_rank") -> pd.DataFrame:
    """Add `<col><suffix>`: each month, the stock's rank on `col` mapped to [-1, 1].

    Smallest -> -1, largest -> +1, evenly spaced, ties averaged; missing -> 0 (the
    cross-sectional middle), as do months with a single value. Outliers cannot
    dominate and the scale is the same in every month, so a coefficient learned
    on 2005 means the same in 2020 (Gu, Kelly & Xiu 2020). Uses only the rows of
    the same month: no look-ahead. Apply it to the rows that will be modelled
    (e.g. investable only), since the ranks depend on who else is in the month.
    """
    by_month = panel.groupby("month_end")
    added = {}
    for col in columns:
        rank = by_month[col].rank(method="average")
        count = by_month[col].transform("count")
        spread = (2 * (rank - 1) / (count - 1) - 1).where(count > 1)
        added[f"{col}{suffix}"] = spread.fillna(0.0)
    return panel.assign(**added)
