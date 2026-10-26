"""Turn the monthly panel into predictive features (signals).

Every feature on a row must be computable from information known at that row's
`month_end`: earlier rows of the same stock, or the row's own known columns.
Never `fwd_ret` (the target) and never later rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_survival.data.real import PRICE_DELAY_COLUMNS, PRICE_DELAY_K, PRICE_DELAY_LAGS
from alpha_survival.ols import group_ols, solve_normal_equations

# 12-1 momentum compounds the monthly returns of t-11 .. t-1: from the close at
# the end of t-12 to the close at the end of t-1 (Jegadeesh & Titman 1993).
# Month t itself is skipped because one-month returns tend to reverse
# (Jegadeesh 1990), which would pollute the signal.
MOMENTUM_MONTHS = 11


def _with(panel: pd.DataFrame, **columns) -> pd.DataFrame:
    """`panel.assign(**columns)` in one concat: assigning many columns one by one
    fragments a wide frame (slow, and pandas warns)."""
    added = pd.DataFrame(columns, index=panel.index)
    return pd.concat([panel.drop(columns=[c for c in columns if c in panel]), added], axis=1)


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
    return _with(panel, mom_12_1=_past_return(panel, 1, MOMENTUM_MONTHS))


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
    return _with(panel, **added, mom_rev=mom_rev)


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
    return _with(panel, **added)


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
MIN_CS_DAYS = 12  # BidAskSpread: OSAP requires 12 daily estimates in the month


def add_daily_price_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `max_ret`, `return_skew`, `high_52`, `illiquidity`, `dol_vol`, `vol_sd`,
    `bid_ask_spread`.

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
    - `bid_ask_spread`: mean of the month's daily Corwin-Schultz estimates, NaN with
      fewer than `MIN_CS_DAYS` (BidAskSpread; Corwin & Schultz 2012).
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
    return _with(
        panel,
        max_ret=panel["dret_max"],
        return_skew=skew,
        high_52=panel["close_split_adj"] / _at_rows(high, panel, close_max),
        illiquidity=_at_rows(illiq_12m, panel, illiq),
        dol_vol=_at_rows(dol_vol, panel, dollars),
        vol_sd=_at_rows(vol_sd, panel, volume),
        bid_ask_spread=(panel["cs_sum"] / panel["cs_n"]).where(panel["cs_n"] >= MIN_CS_DAYS),
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


def _months_listed(panel: pd.DataFrame) -> pd.Series:
    """Calendar months since the stock's first month in the panel, that month = 1."""
    month = panel["month_end"].dt.year * 12 + panel["month_end"].dt.month
    return month - month.groupby(panel["permaticker"]).transform("min") + 1


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

    age = _months_listed(panel)

    return _with(
        panel,
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
    return _with(panel, gp_a=gross_profit / assets)


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
    return _with(panel, asset_growth=now / before - 1)


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
    return _with(panel, vol_3m=np.sqrt(variance.clip(lower=0)).to_numpy())


def add_reversal(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `rev_1m`: the stock's return over month t itself.

    Last month's losers tend to bounce back and winners to give some back
    (Jegadeesh 1990), so expect a NEGATIVE relation with `fwd_ret`; this is why
    momentum skips month t. Part of the effect is bid-ask bounce in month-end
    closes, which the $5 investable cut dampens. The row's own `ret`, known at
    month_end: no look-ahead. NaN where `ret` is (e.g. after a gap).
    """
    return _with(panel, rev_1m=panel["ret"])


def add_size(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `log_mcap`: log of the month-end market cap (USD).

    Small firms have historically earned higher returns (Banz 1981). The log puts
    a $10M -> $100M step on par with $10B -> $100B, so megacaps don't dominate.
    Uses only the row's own `market_cap`, known at month_end. NaN if it is missing.
    """
    market_cap = panel["market_cap"].where(panel["market_cap"] > 0)
    return _with(panel, log_mcap=np.log(market_cap))


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
    return _with(panel, bm=book / market)


def _usd(panel: pd.DataFrame, column: str) -> pd.Series:
    """A fundamental in US dollars. Sharadar's `fxusd` is reporting-currency units
    per dollar (1 for USD filers), so dividing converts; market cap is in USD."""
    return panel[column] / panel["fxusd"].where(panel["fxusd"] > 0)


def add_valuation_ratios(panel: pd.DataFrame) -> pd.DataFrame:
    """Add nine accounting-to-price ratios (OSAP's code; signs in run.PUBLISHED_SIGN).

    "Timely", like `bm`: the latest filing public at month_end (trailing twelve
    months for flows, latest balance sheet for stocks) over that month-end's market
    cap, not OSAP's Compustat annual values (EP: over market cap 6 months earlier).
    Fundamentals are converted to USD (`_usd`). NaN if market cap <= 0.
    - `am`: total assets / market cap (AM; Fama & French 1992).
    - `sp`: revenue / market cap (SP; Barbee, Mukherji & Raines 1996).
    - `leverage`: total liabilities / market cap (Leverage; Bhandari 1988).
    - `cf`: (net income + depreciation and amortisation) / market cap (CF; Lakonishok,
      Shleifer & Vishny 1994). OSAP: income before extraordinary items.
    - `cfp`: operating cash flow / market cap (cfp; Desai, Rajgopal & Venkatachalam 2004).
    - `rd`: R&D expense / market cap (RD; Chan, Lakonishok & Sougiannis 2001). Sharadar
      reports no R&D as 0, where Compustat is often missing: non-R&D firms rank low
      here instead of dropping out.
    - `ep`: net income / market cap, NaN if negative (EP; Basu 1977).
    - `cash_prod`: (market cap - total assets) / cash, NaN if cash <= 0 (CashProd;
      Chandrashekar & Rao 2009).
    - `ent_mult`: (market cap + debt - cash) / EBITDA, NaN if EBITDA or book equity is
      negative (EntMult; Loughran & Wellman 2011).
    Row's own columns only, all public at month_end: no look-ahead.
    """
    mcap = panel["market_cap"].where(panel["market_cap"] > 0)
    assets, cash = _usd(panel, "assets"), _usd(panel, "cashneq")
    earnings = _usd(panel, "netinc")
    ebitda = _usd(panel, "ebitda")
    ent_mult = (mcap + _usd(panel, "debt") - cash) / ebitda.where(ebitda != 0)
    return _with(
        panel,
        am=assets / mcap,
        sp=_usd(panel, "revenue") / mcap,
        leverage=_usd(panel, "liabilities") / mcap,
        cf=(earnings + _usd(panel, "depamor")) / mcap,
        cfp=_usd(panel, "ncfo") / mcap,
        rd=_usd(panel, "rnd") / mcap,
        ep=(earnings / mcap).where(earnings >= 0),
        cash_prod=(mcap - assets) / cash.where(cash > 0),
        ent_mult=ent_mult.where((ebitda >= 0) & (panel["equity"] >= 0)),
    )


# OSAP rules for the book, payout and debt ratios.
PAYOUT_LAG = 6  # payout yields: over market cap 6 months earlier
MIN_HISTORY = 24  # payout yields: from a stock's 24th month in the data
BMDEC_SWITCH = 6  # BMdec: from June, last December's market cap; before, the one before
NET_DEBT_MIN_BM = 0.4  # NetDebtPrice: only above the 40th percentile of log B/M (top 3 fifths)


def _financial(panel: pd.DataFrame) -> pd.Series:
    """Banks, insurers, etc. (SIC 6000-6999), by today's SIC code
    (see data.real.add_industry_codes)."""
    return panel["sic_code"].between(6000, 6999)


def add_book_payout_debt_ratios(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `bm_dec`, `payout_yield`, `net_payout_yield`, `net_debt_price`, `ebm`, `bpebm`,
    `book_leverage`.

    OSAP's code, on the latest public filing (USD, `_usd`); signs in run.PUBLISHED_SIGN.
    - `bm_dec`: book equity over the market cap of last December, or of the December
      before from January to May (BMdec; Fama & French 1992 timing).
    - `payout_yield`: (dividends paid + net share buybacks) over market cap 6 months
      earlier; NaN if <= 0 (PayoutYield; Boudoukh et al. 2007).
    - `net_payout_yield`: (dividends paid - net equity issued) over market cap 6
      months earlier; NaN if 0 (NetPayoutYield; Boudoukh et al. 2007).
      Sharadar reports equity issuance net of buybacks (`ncfcommon`), so gross
      buybacks are approximated by net ones; preferred stock terms are left out.
      Both: financials excluded, book equity > 0 (payout_yield: or missing), and
      only from the stock's `MIN_HISTORY`th month in the data.
    - `net_debt_price`: (debt - cash) over market cap, financials excluded, only
      for stocks above the month's 40th percentile of log book-to-market
      (NetDebtPrice; Penman, Richardson & Tuna 2007).
    - `ebm`: (book + cash - debt) over (market cap + cash - debt): book-to-market
      of the operating business (EBM; Penman et al. 2007).
    - `bpebm`: book-to-price minus `ebm`, the leverage part of B/M (BPEBM).
    - `book_leverage`: assets / (book equity + deferred taxes), NaN if the
      denominator is 0, negative kept as in OSAP (BookLeverage; Fama & French 1992).
      Book equity falls back to assets - liabilities; Sharadar's `taxliabilities`
      (all tax liabilities, not only deferred) stands in for txditc, 0 if missing.
      A ratio within one filing: currency cancels.
    OSAP's preferred-stock and treasury-stock adjustments are left out.
    Financials use today's SIC code (approximate). Past market caps are this
    stock's own earlier rows, all public by month_end: no look-ahead.
    """
    panel = _sorted_by_stock(panel)
    mcap = panel["market_cap"].where(panel["market_cap"] > 0)
    book, cash, debt = _usd(panel, "equity"), _usd(panel, "cashneq"), _usd(panel, "debt")
    dividends = -_usd(panel, "ncfdiv")  # cash-flow outflows are negative
    net_issued = _usd(panel, "ncfcommon")
    financial = _financial(panel)
    seasoned = _months_listed(panel) >= MIN_HISTORY

    caps = _calendar_grid(panel, "market_cap")
    lagged = _at_rows(caps.shift(PAYOUT_LAG), panel, caps)
    lagged = pd.Series(lagged, index=panel.index).where(lambda s: s > 0)
    payout = (dividends + (-net_issued).clip(lower=0)) / lagged
    payout = payout.where((payout > 0) & ~financial & seasoned & ~(book <= 0))
    net_payout = (dividends - net_issued) / lagged
    net_payout = net_payout.where((net_payout != 0) & ~financial & seasoned & (book > 0))

    month = panel["month_end"].dt.month
    dec_year = panel["month_end"].dt.year - np.where(month >= BMDEC_SWITCH, 1, 2)
    december = panel.loc[month == 12, ["permaticker", "month_end", "market_cap"]]
    december = pd.DataFrame(
        {
            "permaticker": december["permaticker"],
            "dec_year": december["month_end"].dt.year,
            "dec_cap": december["market_cap"],
        }
    )
    keyed = pd.DataFrame({"permaticker": panel["permaticker"], "dec_year": dec_year})
    dec_cap = keyed.merge(december, on=["permaticker", "dec_year"], how="left")["dec_cap"]
    dec_cap = pd.Series(dec_cap.to_numpy(), index=panel.index).where(lambda s: s > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        log_bm = np.log(book / mcap).replace([np.inf, -np.inf], np.nan)
    cut = log_bm.groupby(panel["month_end"]).transform("quantile", NET_DEBT_MIN_BM)
    complete = panel[["assets", "netinc", "equity"]].notna().all(axis=1)
    net_debt = ((debt - cash) / mcap).where(~financial & complete & (log_bm > cut))

    ebm = (book + cash - debt) / (mcap + cash - debt)
    book_equity = panel["equity"].fillna(panel["assets"] - panel["liabilities"])
    book_equity = book_equity + panel["taxliabilities"].fillna(0)
    return _with(
        panel,
        bm_dec=book / dec_cap,
        payout_yield=payout,
        net_payout_yield=net_payout,
        net_debt_price=net_debt,
        ebm=ebm,
        bpebm=book / mcap - ebm,
        book_leverage=panel["assets"] / book_equity.where(book_equity != 0),
    )


def _months_ago(panel: pd.DataFrame, values: str | pd.Series, months: int) -> pd.Series:
    """`values` (a column name, or a Series aligned with `panel`) on the same stock's
    row `months` calendar months earlier (NaN if none): what was public then."""
    series = panel[values] if isinstance(values, str) else values
    keys = pd.DataFrame(
        {"permaticker": panel["permaticker"], "month_end": panel["month_end"], "v": series}
    )
    grid = _calendar_grid(keys, "v")
    return pd.Series(_at_rows(grid.shift(months), keys, grid), index=panel.index)


def _own_rolling(
    panel: pd.DataFrame, values: pd.Series, window: int, min_periods: int, stat: str = "mean"
) -> pd.Series:
    """`stat` ("mean", "std", "sum") of `values` over each stock's last `window`
    calendar months (this one included), NaN with fewer than `min_periods` values."""
    keys = pd.DataFrame(
        {"permaticker": panel["permaticker"], "month_end": panel["month_end"], "v": values}
    )
    grid = _calendar_grid(keys, "v")
    result = getattr(grid.rolling(window, min_periods=min_periods), stat)()
    return pd.Series(_at_rows(result, keys, grid), index=panel.index)


TAX_RATE = 0.35  # Tax: OSAP's top statutory rate from 1993 (kept after 2018's cut to 21%)
OPERPROF_SMALL = 1 / 3  # OperProf: drop the month's smallest third by market cap
TANG_SIC = (2000, 3999)  # tang: manufacturers only
# tang: expected liquidation value per dollar of each asset (Berger, Ofek & Swary 1996).
TANG_WEIGHTS = {"cashneq": 1.0, "receivables": 0.715, "inventory": 0.547, "ppnenet": 0.535}


def add_profitability_ratios(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `roe`, `oper_prof`, `oper_prof_rd`, `cb_oper_prof`, `op_leverage`, `tax`, `tang`.

    OSAP's code on the latest public filing (flows trailing twelve months). Ratios
    within one filing, so the currency cancels. Compustat's SG&A includes R&D;
    Sharadar's `sgna` does not (checked: opex = sgna + rnd), so OSAP's `xsga` is
    `sgna + rnd` here. Missing R&D counts as 0, as in OSAP.
    - `roe`: net income / book equity (RoE; Haugen & Baker 1996).
    - `oper_prof`: (revenue - cost of revenue - SG&A incl. R&D - interest) / book
      equity; the month's smallest third by market cap excluded (OperProf; Fama &
      French 2006).
    - `oper_prof_rd`: (revenue - cost of revenue - SG&A excl. R&D) / assets, i.e.
      R&D added back; financials excluded (OperProfRD; Ball et al. 2016).
    - `cb_oper_prof`: the same profit minus the 12-month increases in receivables
      and inventory, plus those in deferred revenue and payables, / assets;
      financials excluded (CBOperProf; Ball et al. 2016). Sharadar has no prepaid
      or accrued-expense fields: those terms are left out.
    - `op_leverage`: (cost of revenue + SG&A incl. R&D) / assets (OPLeverage; Novy-Marx 2011).
    - `tax`: tax expense / `TAX_RATE` / net income, 1 for loss-makers that pay tax
      (Tax; Lev & Nissim 2004). Sharadar gives total tax only, not current vs deferred.
    - `tang`: liquidation-weighted assets (`TANG_WEIGHTS`) / assets, manufacturers
      only (tang; Hahn & Lee 2009). Net PP&E, where OSAP uses gross.
    Financials and manufacturers by today's SIC code (see data.real.add_industry_codes).
    12-month changes use the stock's own row 12 calendar months earlier: no look-ahead.
    """
    panel = _sorted_by_stock(panel)
    assets = panel["assets"].where(panel["assets"] > 0)
    equity, income = panel["equity"], panel["netinc"]
    rnd = panel["rnd"].fillna(0)
    gross = panel["revenue"] - panel["cor"]
    financial = _financial(panel)

    cap_rank = panel.groupby("month_end")["market_cap"].rank(pct=True)
    oper_prof = (gross - panel["sgna"] - rnd - panel["intexp"]) / equity.where(equity != 0)

    def increase(column: str) -> pd.Series:
        return panel[column] - _months_ago(panel, column, 12)

    cash_profit = gross - panel["sgna"]
    cash_profit = (
        cash_profit
        - increase("receivables")
        - increase("inventory")
        + increase("deferredrev")
        + increase("payables")
    )

    tax_paid = panel["taxexp"]
    tax = tax_paid / TAX_RATE / income.where(income != 0)
    loss = income <= 0
    tax = tax.mask(loss & ((tax_paid > 0) | tax_paid.isna()), 1.0)
    tax = tax.mask((income == 0) & (tax_paid != 0), 1.0)

    weighted = sum(w * panel[c] for c, w in TANG_WEIGHTS.items())
    manufacturer = panel["sic_code"].between(*TANG_SIC)
    return _with(
        panel,
        roe=income / equity.where(equity != 0),
        oper_prof=oper_prof.where(cap_rank > OPERPROF_SMALL),
        oper_prof_rd=((gross - panel["sgna"]) / assets).where(~financial),
        cb_oper_prof=(cash_profit / assets).where(~financial),
        op_leverage=(panel["cor"] + panel["sgna"].fillna(0) + rnd) / assets,
        tax=tax,
        tang=(weighted / assets).where(manufacturer),
    )


# PctAcc: when net income is exactly 0, divide by this instead (OSAP's 0.01 in
# Compustat's millions; Sharadar reports units, so 10,000).
ZERO_INCOME_SCALE = 1e4


def add_accrual_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `accruals`, `pct_acc`, `pct_tot_acc`, `total_accruals`, `noa`, `d_noa`.

    Earnings made of accruals rather than cash, and bloated operating balance
    sheets, predict LOWER returns (signs -1). OSAP's code mapped to Sharadar; each
    ratio is within one company's own filings, so the currency cancels.
    - `accruals`: (change in current assets - change in cash - (change in current
      liabilities - change in short-term debt) - D&A) / average of assets now and
      12 months ago (Accruals; Sloan 1996). No taxes-payable field: that term is out.
    - `pct_acc`: (net income - operating cash flow) / |net income| (PctAcc; Hafzalla,
      Lundholm & Van Winkle 2011). OSAP's balance-sheet fallback for missing cash
      flow is not used: NaN.
    - `pct_tot_acc`: (net income - operating, investing and financing cash flows +
      net equity issued - dividends paid) / |net income| (PctTotAcc; same paper).
      Sharadar's net issuance stands in for Compustat's gross issuance and buybacks.
    - `total_accruals`: net income - (operating + investing + financing cash flows)
      + net equity issued - dividends paid, over assets 12 months ago
      (TotalAccruals; Richardson et al. 2005; OSAP's post-1989 cash-flow version).
    - `noa`: (operating assets - operating liabilities) / assets 12 months ago, with
      operating assets = assets - cash and operating liabilities = liabilities -
      long-term debt (NOA; Hirshleifer et al. 2004). By assets = liabilities +
      equity + minority interest, that is OSAP's assets - LT debt - minority
      interest - equity exactly (deferred charges aside).
    - `d_noa`: change over 12 months of (assets - cash) - (liabilities - total debt),
      over assets 12 months ago (dNoa; same paper).
    "12 months ago" is the stock's own row 12 calendar months earlier: no look-ahead.
    """
    panel = _sorted_by_stock(panel)

    def change(column: str) -> pd.Series:
        return panel[column] - _months_ago(panel, column, 12)

    assets = panel["assets"].where(panel["assets"] > 0)
    assets_before = _months_ago(panel, "assets", 12).where(lambda s: s > 0)
    accruals = (
        change("assetsc")
        - change("cashneq")
        - (change("liabilitiesc") - change("debtc"))
        - panel["depamor"]
    ) / ((assets + assets_before) / 2)

    income = panel["netinc"]
    scale = income.abs().where(income != 0, ZERO_INCOME_SCALE)
    cash_flows = panel["ncfo"] + panel["ncfi"] + panel["ncff"]
    # Sharadar: ncfcommon = issued - bought back; ncfdiv = -(dividends paid).
    equity_flows = panel["ncfcommon"] + panel["ncfdiv"]
    pct_tot_acc = (income - cash_flows + equity_flows) / income.abs().where(income != 0)

    operating = (panel["assets"] - panel["cashneq"]) - (panel["liabilities"] - panel["debtnc"])
    # dNoa as OSAP: missing debt or cash counts as 0, and all debt is financing.
    operating_all = (panel["assets"] - panel["cashneq"].fillna(0)) - (
        panel["liabilities"] - panel["debt"].fillna(0)
    )
    d_noa = (operating_all - _months_ago(panel, operating_all, 12)) / assets_before

    return _with(
        panel,
        accruals=accruals,
        pct_acc=(income - panel["ncfo"]) / scale,
        pct_tot_acc=pct_tot_acc,
        total_accruals=(income - cash_flows + equity_flows) / assets_before,
        noa=operating / assets_before,
        d_noa=d_noa,
    )


DELDRC_MIN_REVENUE = 5e6  # DelDRC: OSAP drops firms with sales under $5m (Compustat: 5)


def add_balance_sheet_changes(panel: pd.DataFrame) -> pd.DataFrame:
    """Add 13 signals built on 12-month changes in balance-sheet items (OSAP's code).

    Growth in operating assets, working capital, investment, book equity and debt
    predicts LOWER returns; growth in deferred revenue, net financial assets and
    asset turnover, higher (signs in run.PUBLISHED_SIGN, from the catalogue).
    "Avg assets" = mean of assets now and 12 months ago.
    Compustat -> Sharadar: act/lct -> assetsc/liabilitiesc, dlc/dltt -> debtc/debtnc,
    ivst/ivao -> investmentsc/investmentsnc, drc -> deferredrev, invt -> inventory,
    ceq -> equity; no preferred stock field (OSAP fills it with 0 when missing).
    - `ch_nncoa`: change in (non-current operating assets - non-current operating
      liabilities) / assets (ChNNCOA; Soliman 2008).
    - `ch_nwc`: change in (current assets - cash - (current liabilities - short-term
      debt)) / assets (ChNWC; Soliman 2008).
    - `del_coa`, `del_col`: change in current operating assets / liabilities over
      avg assets (DelCOA, DelCOL; Richardson et al. 2005).
    - `del_drc`: change in deferred revenue over avg assets; NaN for financials,
      book equity <= 0, revenue < $5m, or no deferred revenue at all (DelDRC;
      Prakash & Sinha 2013). Sharadar combines current and long-term deferred revenue.
    - `del_equ`: change in book equity over avg assets (DelEqu; Richardson et al. 2005).
    - `del_lti`: change in long-term investments over avg assets (DelLTI; same).
    - `del_finl`: change in total debt over avg assets (DelFINL; same). Sharadar's
      total `debt`, which is short- plus long-term where both are reported.
    - `del_net_fin`: change in (short- and long-term investments - debt) over avg
      assets (DelNetFin; same).
    - `ch_inv`: change in inventory over avg assets (ChInv; Thomas & Zhang 2002).
    - `ch_eq`: book equity over book equity 12 months ago, both positive (ChEQ;
      Lockwood & Prombutr 2010).
    - `ch_asset_turnover`: change over 12 months of revenue / avg net operating
      assets, NaN if negative (ChAssetTurnover; Soliman 2008). Sharadar lacks the
      'other' current and long-term items, so net operating assets here are
      (assets - cash - investments) - (liabilities - debt).
    - `invest_ppe_inv`: (change in PP&E + change in inventory) / assets 12 months
      ago (InvestPPEInv; Lyandres, Sun & Zhang 2008). Net, not gross, PP&E.
    Financials by today's SIC code. Values 12 months earlier are the stock's own
    row then (`_months_ago`): no look-ahead.
    """
    panel = _sorted_by_stock(panel)

    def ago(values: str | pd.Series) -> pd.Series:
        return _months_ago(panel, values, 12)

    def change(values: str | pd.Series) -> pd.Series:
        series = panel[values] if isinstance(values, str) else values
        return series - ago(series)

    assets = panel["assets"].where(panel["assets"] > 0)
    assets_before = ago(assets)
    avg_assets = (assets + assets_before) / 2

    nncoa = (panel["assets"] - panel["assetsc"] - panel["investmentsnc"]) - (
        panel["liabilities"] - panel["debtc"] - panel["debtnc"]
    )
    current_assets = panel["assetsc"] - panel["cashneq"]
    current_liabilities = panel["liabilitiesc"] - panel["debtc"]
    nwc = current_assets - current_liabilities
    net_fin = panel["investmentsc"] + panel["investmentsnc"] - panel["debt"]

    drc = panel["deferredrev"]
    del_drc = change(drc) / avg_assets
    no_drc = (drc == 0) & (del_drc == 0)
    drc_out = (
        (panel["equity"] <= 0)
        | no_drc
        | (panel["revenue"] < DELDRC_MIN_REVENUE)
        | _financial(panel)
    )

    equity = panel["equity"].where(panel["equity"] > 0)

    investments = panel["investmentsc"].fillna(0) + panel["investmentsnc"].fillna(0)
    noa = (panel["assets"] - panel["cashneq"] - investments) - (
        panel["liabilities"] - panel["debt"]
    )
    turnover = panel["revenue"] / ((noa + ago(noa)) / 2)
    turnover = turnover.where(turnover >= 0)

    return _with(
        panel,
        ch_nncoa=change(nncoa / assets),
        ch_nwc=change(nwc / assets),
        del_coa=change(current_assets) / avg_assets,
        del_col=change(current_liabilities) / avg_assets,
        del_drc=del_drc.mask(drc_out),
        del_equ=change("equity") / avg_assets,
        del_lti=change("investmentsnc") / avg_assets,
        del_finl=change("debt") / avg_assets,
        del_net_fin=change(net_fin) / avg_assets,
        ch_inv=change("inventory") / avg_assets,
        ch_eq=equity / ago(equity),
        ch_asset_turnover=change(turnover),
        invest_ppe_inv=(change("ppnenet") + change("inventory")) / assets_before,
    )


INVESTMENT_MONTHS, INVESTMENT_MIN = 36, 24  # Investment: own mean over >= 24 of 36 months
INVESTMENT_MIN_REVENUE = 1e7  # Investment: OSAP drops sales under $10m (Compustat: 10)
RD_SURPRISE = 1.05  # SurpriseRD: R&D up more than 5%, in dollars and per dollar of assets


def add_investment_growth(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `investment`, `grcapx`, `grcapx3y`, `gr_sale_to_gr_inv`,
    `gr_sale_to_gr_overhead`, `surprise_rd` (OSAP's code; signs in run.PUBLISHED_SIGN).

    Sharadar reports capital expenditure as a cash outflow (negative), so capex =
    -`capex`; the few positive values (asset sales netted in) become negative capex.
    SG&A is `sgna + rnd`, as Compustat's xsga includes R&D (see
    `add_profitability_ratios`). Ratios within one company's filings: currency cancels.
    - `investment`: capex / revenue over its own mean of the last 36 months (>= 24,
      this month included), NaN if revenue < $10m (Investment; Titman, Wei & Xie 2004).
    - `grcapx`: capex now vs. 24 months ago, (now - then) / then (grcapx; Anderson &
      Garcia-Feijoo 2006). Missing capex is replaced by the 12-month change in net
      PP&E from the stock's 24th month in the data on, as OSAP does.
    - `grcapx3y`: capex over the mean of capex 12, 24 and 36 months ago (grcapx3y).
    - `gr_sale_to_gr_inv`: % growth in sales minus % growth in inventory, each vs.
      the mean of 12 and 24 months ago, or vs. 12 months ago where that is missing
      (GrSaleToGrInv; Abarbanell & Bushee 1998).
    - `gr_sale_to_gr_overhead`: the same with SG&A (GrSaleToGrOverhead; same paper).
    - `surprise_rd`: 1 if R&D is positive and grew more than 5% over 12 months both
      in dollars and as a share of assets, else 0; NaN without R&D now and a year ago
      (SurpriseRD; Eberhart, Maxwell & Siddique 2004). Sharadar reports no R&D as 0,
      so non-R&D firms get 0 where Compustat would often have them missing.
    Earlier values are the stock's own rows then (`_months_ago`): no look-ahead.
    """
    panel = _sorted_by_stock(panel)

    def ago(values: str | pd.Series, months: int) -> pd.Series:
        return _months_ago(panel, values, months)

    def growth(now: pd.Series, before: pd.Series) -> pd.Series:
        return (now - before) / before.where(before != 0)

    revenue = panel["revenue"]
    capex = -panel["capex"]
    ratio = capex / revenue.where(revenue != 0)
    ratio_mean = _own_rolling(panel, ratio, INVESTMENT_MONTHS, INVESTMENT_MIN)
    investment = (ratio / ratio_mean.where(ratio_mean != 0)).where(
        revenue >= INVESTMENT_MIN_REVENUE
    )

    ppe_change = panel["ppnenet"] - ago("ppnenet", 12)
    capex = capex.fillna(ppe_change.where(_months_listed(panel) >= 24))
    capex_before = {m: ago(capex, m) for m in (12, 24, 36)}
    three_year = (capex_before[12] + capex_before[24] + capex_before[36]) / 3

    def growth_gap(values: pd.Series, other: pd.Series) -> pd.Series:
        """% growth of `values` minus that of `other`, each vs. the mean of 12 and 24
        months ago; where that is missing, both vs. 12 months ago (OSAP)."""

        def vs_two_years(series: pd.Series) -> pd.Series:
            return growth(series, (ago(series, 12) + ago(series, 24)) / 2)

        def vs_one_year(series: pd.Series) -> pd.Series:
            return growth(series, ago(series, 12))

        two_years = vs_two_years(values) - vs_two_years(other)
        return two_years.fillna(vs_one_year(values) - vs_one_year(other))

    sga = panel["sgna"] + panel["rnd"].fillna(0)

    rnd = panel["rnd"]
    rnd_before = ago("rnd", 12)
    assets = panel["assets"].where(panel["assets"] > 0)
    rnd_intensity_before = rnd_before / ago(assets, 12)
    surprise = (
        (rnd / revenue > 0)
        & (rnd / assets > 0)
        & (rnd / rnd_before > RD_SURPRISE)
        & ((rnd / assets) / rnd_intensity_before > RD_SURPRISE)
    ).astype(float)

    return _with(
        panel,
        investment=investment,
        grcapx=growth(capex, capex_before[24]),
        grcapx3y=capex / three_year.where(three_year != 0),
        gr_sale_to_gr_inv=growth_gap(revenue, panel["inventory"]),
        gr_sale_to_gr_overhead=growth_gap(revenue, sga),
        surprise_rd=surprise.where(rnd.notna() & rnd_before.notna()),
    )


MAX_FINANCE_TO_ASSETS = 1.0  # Net*Finance: OSAP drops |flow / avg assets| above 1
# ShareIss: share counts `first` and `last` months back (OSAP's lags, after Pontiff &
# Woodgate 2008 and Daniel & Titman 2006).
SHARE_ISSUE_LAGS = {"share_iss_1y": (18, 6), "share_iss_5y": (65, 5)}


def add_financing_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `composite_debt_issuance`, `debt_issuance`, `net_debt_finance`,
    `net_equity_finance`, `xfin`, `share_iss_1y`, `share_iss_5y`, `share_repurchase`.

    Firms raising outside money (debt or equity) tend to earn LESS afterwards;
    buyers-back more (signs in run.PUBLISHED_SIGN). OSAP's code mapped to Sharadar,
    whose cash-flow statement gives only NET flows: `ncfdebt` (debt issued -
    repaid), `ncfcommon` (equity issued - bought back) and `ncfdiv` (-dividends).
    - `composite_debt_issuance`: log(total debt / total debt 60 months ago)
      (CompositeDebtIssuance; Lyandres, Sun & Zhang 2008).
    - `debt_issuance`: 1 if net debt issued > 0, else 0 (DebtIssuance; Spiess &
      Affleck-Graves 1999; OSAP: gross issuance); NaN where log book-to-market is
      undefined (book <= 0 or missing), as in OSAP.
    - `net_debt_finance`: net debt issued / avg assets (NetDebtFinance; Bradshaw,
      Richardson & Sloan 2006).
    - `net_equity_finance`: (net equity issued - dividends) / avg assets
      (NetEquityFinance; same). Both NaN beyond +-`MAX_FINANCE_TO_ASSETS`.
    - `xfin`: (net equity + net debt issued - dividends) / assets (XFIN; same).
    - `share_iss_1y`, `share_iss_5y`: % change in shares outstanding between the
      `SHARE_ISSUE_LAGS` months (ShareIss1Y, ShareIss5Y). Shares from the latest
      filing (`sharesbas` x `sharefactor`), which Sharadar restates for every
      split, so both ends are on one split basis; they change only when a new
      filing lands, where CRSP's count is monthly.
    - `share_repurchase`: 1 if net equity flow < 0 (a net buyback), else 0; NaN
      without a cash-flow statement (ShareRepurchase; Ikenberry, Lakonishok &
      Vermaelen 1995; OSAP: gross buybacks > 0).
    Ratios within one company's filings: currency cancels. Earlier values are the
    stock's own rows then (`_months_ago`): no look-ahead.
    """
    panel = _sorted_by_stock(panel)

    def ago(values: str | pd.Series, months: int) -> pd.Series:
        return _months_ago(panel, values, months)

    assets = panel["assets"].where(panel["assets"] > 0)
    avg_assets = (assets + ago(assets, 12)) / 2
    debt = panel["debt"].where(panel["debt"] > 0)
    net_debt, net_equity = panel["ncfdebt"], panel["ncfcommon"]
    equity_flow = net_equity + panel["ncfdiv"]  # issued - bought back - dividends

    def capped(ratio: pd.Series) -> pd.Series:
        return ratio.where(ratio.abs() <= MAX_FINANCE_TO_ASSETS)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_bm = np.log(panel["equity"] / panel["market_cap"])
    debt_issuance = (net_debt > 0).astype(float).where(np.isfinite(log_bm))

    shares = (panel["sharesbas"] * panel["sharefactor"]).where(panel["sharesbas"] > 0)
    share_issues = {}
    for name, (first, last) in SHARE_ISSUE_LAGS.items():
        before = ago(shares, first)
        share_issues[name] = (ago(shares, last) - before) / before

    return _with(
        panel,
        composite_debt_issuance=np.log(debt / ago(debt, 60)),
        debt_issuance=debt_issuance,
        net_debt_finance=capped(net_debt / avg_assets),
        net_equity_finance=capped(equity_flow / avg_assets),
        xfin=(equity_flow + net_debt) / assets,
        **share_issues,
        share_repurchase=(net_equity < 0).astype(float).where(net_equity.notna()),
    )


SHAREVOL_MONTHS = 3  # ShareVol: average turnover over this month and the 2 before
SHAREVOL_LOW, SHAREVOL_HIGH = 5.0, 10.0  # ShareVol: % monthly turnover cut-offs
VOLMKT_MONTHS, VOLMKT_MIN = 12, 10  # VolMkt: mean dollar volume, >= 10 of 12 months
STDTURN_MONTHS, STDTURN_MIN = 36, 24  # std_turn: std of turnover, >= 24 of 36 months
STDTURN_LARGE = 0.6  # std_turn: drop the month's two largest fifths by market cap
# zerotrade: Liu (2006)'s deflators (fn 4), large enough that the turnover term only
# breaks ties between stocks with the same count of zero-volume days.
ZEROTRADE = {
    "zerotrade_1m": (1, 480_000),
    "zerotrade_6m": (6, 11_000),
    "zerotrade_12m": (12, 11_000),
}
DAYS_PER_MONTH = 21  # zerotrade: counts scaled to 21 trading days a month
MOMVOL_MONTHS, MOMVOL_MIN = 6, 5  # MomVol: mean turnover, >= 5 of 6 months
MOMVOL_MIN_ROWS = 24  # MomVol: from a stock's 24th month in the data


def add_turnover_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `share_vol`, `vol_mkt`, `std_turn`, `zerotrade_1m/6m/12m`, `mom_vol`.

    Turnover = shares traded in the month / shares outstanding. Sharadar's daily
    volume and its filings' shares (`sharesbas` x `sharefactor`) are both restated
    to the latest split basis (checked on AAPL's 2020 4:1 split), so the ratio is
    unaffected by splits. Shares come from the latest filing, so they are fixed
    between filings. OSAP's code, with these differences:
    - `share_vol`: 0 if average turnover over 3 months is under 5% a month, 1 if over
      10%, NaN in between (ShareVol; Datar, Naik & Radcliffe 1998). OSAP also drops
      months whose share count changed in the last 3; with quarterly filing shares
      that would drop most months, so it is not done here.
    - `vol_mkt`: mean monthly dollar volume over 12 months (>= 10) / market cap
      (VolMkt; Haugen & Baker 1996).
    - `std_turn`: std of monthly turnover over 36 months (>= 24), NaN for the month's
      two largest fifths by market cap (std_turn; Chordia, Subrahmanyam & Anshuman 2001).
    - `zerotrade_1m/6m/12m`: Liu's LM: zero-volume days + (1 / turnover) / deflator,
      over the last 1/6/12 months (all required), scaled to 21 trading days a month
      and taken one month back, as OSAP does (zerotrade1M/6M/12M; Liu 2006).
      NaN with no volume at all (1 / 0).
    - `mom_vol`: decile (1-10) of past 6-month return (months t-5..t-1) among
      high-turnover stocks (top third of 6-month mean turnover); NaN otherwise and
      before a stock's 24th month (MomVol; Lee & Swaminathan 2000). OSAP ranks raw
      share volume; turnover is used here as in the paper, since split-restated
      share volume is not comparable across stocks.
    Deciles and size fifths are within the month across the panel's stocks. All
    windows end at this month: no look-ahead.
    """
    panel = _sorted_by_stock(panel)
    shares = panel["sharesbas"] * panel["sharefactor"]
    turnover = panel["share_volume"] / shares.where(shares > 0)

    share_vol_pct = 100 * _own_rolling(panel, turnover, SHAREVOL_MONTHS, SHAREVOL_MONTHS)
    share_vol = pd.Series(np.nan, index=panel.index)
    share_vol[share_vol_pct < SHAREVOL_LOW] = 0.0
    share_vol[share_vol_pct > SHAREVOL_HIGH] = 1.0

    mean_dollar_volume = _own_rolling(panel, panel["dollar_volume"], VOLMKT_MONTHS, VOLMKT_MIN)
    market_cap = panel["market_cap"].where(panel["market_cap"] > 0)

    std_turn = _own_rolling(panel, turnover, STDTURN_MONTHS, STDTURN_MIN, "std")
    cap_rank = panel.groupby("month_end")["market_cap"].rank(pct=True)

    def total(values: pd.Series, months: int) -> pd.Series:
        return _own_rolling(panel, values, months, months, "sum")

    zerotrade = {}
    days, zero_days = panel["n_trade_days"], panel["n_zero_volume"]
    for name, (months, deflator) in ZEROTRADE.items():
        turn = total(turnover, months)
        lm = (total(zero_days, months) + (1 / turn.where(turn > 0)) / deflator) * (
            DAYS_PER_MONTH * months / total(days, months)
        )
        zerotrade[name] = _months_ago(panel, lm, 1)

    ret = panel["ret"].fillna(0)
    past = pd.Series(1.0, index=panel.index)
    for lag in range(1, 6):
        past = past * (1 + _months_ago(panel, ret, lag))
    keys = panel[["month_end"]].assign(
        mom=past - 1, turn=_own_rolling(panel, turnover, MOMVOL_MONTHS, MOMVOL_MIN)
    )
    by_month = keys.groupby("month_end")
    mom_decile = np.ceil(10 * by_month["mom"].rank(pct=True))
    high_turnover = by_month["turn"].rank(pct=True) > 2 / 3
    old_enough = panel.groupby("permaticker").cumcount() >= MOMVOL_MIN_ROWS - 1
    mom_vol = mom_decile.where(high_turnover & old_enough)

    return _with(
        panel,
        share_vol=share_vol,
        vol_mkt=mean_dollar_volume / market_cap,
        std_turn=std_turn.where(cap_rank <= STDTURN_LARGE),
        **zerotrade,
        mom_vol=mom_vol,
    )


EARN_INCREASE_QUARTERS = 8  # NumEarnIncrease: counts up to 8 quarters back


def add_quarterly_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `roaq`, `cash`, `ch_tax`, `num_earn_increase` from quarterly filings
    (the `_q` columns of data.real.add_quarterly_fundamentals; OSAP's code).

    Compustat -> Sharadar: ibq -> netinc_q (net income: Sharadar has no income
    before extraordinary items), atq -> assets_q, cheq -> cashneq_q, txtq -> taxexp_q.
    - `roaq`: quarterly net income / quarterly assets 3 months earlier (roaq;
      Balakrishnan, Bartov & Faurel 2010).
    - `cash`: cash / assets in the latest quarter, assets > 0 (Cash; Palazzo 2012).
    - `ch_tax`: quarterly tax expense minus that 12 months earlier, over (annual)
      assets 12 months earlier (ChTax; Thomas & Zhang 2011).
    - `num_earn_increase`: how many quarters in a row, up to 8, quarterly earnings
      beat the same quarter a year earlier (NumEarnIncrease; Loh & Warachka 2012).
      As OSAP: the count runs back in 3-month steps to the first change <= 0, and a
      missing change counts as an increase; with no change <= 0 within 8 quarters
      it is 0, not 8. NaN without a quarterly filing.
    Earlier values are the stock's own rows then (`_months_ago`): no look-ahead.
    """
    panel = _sorted_by_stock(panel)

    def ago(values: str | pd.Series, months: int) -> pd.Series:
        return _months_ago(panel, values, months)

    assets_q = panel["assets_q"].where(panel["assets_q"] > 0)
    earnings = panel["netinc_q"]
    change = earnings - ago(earnings, 12)

    # A change at quarter lag k counts as an increase if it is > 0 or missing.
    count = pd.Series(0.0, index=panel.index)
    run = pd.Series(True, index=panel.index)  # quarters 0..k-1 all increases
    for k in range(1, EARN_INCREASE_QUARTERS + 1):
        before = change if k == 1 else ago(change, 3 * (k - 1))
        run &= (before > 0) | before.isna()
        at_k = ago(change, 3 * k)
        count[run & (at_k <= 0)] = float(k)

    return _with(
        panel,
        roaq=earnings / ago(assets_q, 3),
        cash=panel["cashneq_q"] / assets_q,
        ch_tax=(panel["taxexp_q"] - ago("taxexp_q", 12))
        / ago(panel["assets"].where(panel["assets"] > 0), 12),
        num_earn_increase=count.where(panel["filing_date_q"].notna()),
    )


def _rolling_ols(
    y: pd.DataFrame, factors: pd.DataFrame, window: int, min_periods: int
) -> dict[str, pd.DataFrame]:
    """OLS with an intercept of every column of `y` (a month x stock grid) on the
    `factors` columns (one value per month, shared by all stocks), over the last
    `window` calendar months: coefficient grids keyed "const" and by factor name.

    b = (X'X)^-1 X'y, where every entry of X'X and X'y is a rolling sum of products
    over the months a stock has a return and every factor has a value, so each
    stock-month only uses that month and earlier. NaN with fewer than
    `min_periods` such months, or when X'X is singular.
    """
    x = factors.reindex(y.index)
    valid = y.notna().to_numpy() & x.notna().all(axis=1).to_numpy()[:, None]
    m = valid.astype(float)
    yv = np.where(valid, y.to_numpy(), 0.0)
    columns = [np.ones((len(y), 1)), *(x[[c]].fillna(0.0).to_numpy() for c in x.columns)]
    names = ["const", *x.columns]
    k = len(columns)

    def roll(values: np.ndarray) -> np.ndarray:
        total = np.cumsum(values, axis=0)
        total[window:] = total[window:] - total[:-window].copy()
        return total

    n = roll(m)
    cells = np.nonzero(n >= max(min_periods, k))
    xtx = np.empty((len(cells[0]), k, k))
    xty = np.empty((len(cells[0]), k))
    for i in range(k):
        xty[:, i] = roll(columns[i] * yv)[cells]
        for j in range(i, k):
            xtx[:, i, j] = xtx[:, j, i] = roll(columns[i] * columns[j] * m)[cells]

    beta = solve_normal_equations(xtx, xty)

    out = {}
    for i, name in enumerate(names):
        grid = np.full(y.shape, np.nan)
        grid[cells] = beta[:, i]
        out[name] = pd.DataFrame(grid, index=y.index, columns=y.columns)
    return out


def _excess_grid(panel: pd.DataFrame, factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly excess returns (ret - rf) as a month x stock grid, and `factors` on its months."""
    factors = factors.reindex(_calendar_grid(panel, "ret").index)
    excess = panel["ret"] - panel["month_end"].map(factors["rf"])
    keys = panel[["permaticker", "month_end"]]
    return _calendar_grid(keys.assign(_x=excess), "_x"), factors


# Monthly factor regressions (OSAP's code). Each sign is in run.PUBLISHED_SIGN.
BETA_MONTHS, BETA_MIN = 60, 20  # Beta: >= 20 of the last 60 months
PS_MONTHS, PS_MIN = 60, 36  # BetaLiquidityPS: >= 36 of the last 60 months


def add_factor_betas(panel: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Add `beta` and `beta_liquidity_ps`: rolling slopes of a stock's monthly excess
    return (ret - rf) on factor returns, over calendar months up to month_end.

    `factors`: indexed by month_end, columns rf, mktrf, smb, hml, ps_innov, each a
    value known by that month end (`data.external.monthly_factors`).
    - `beta`: slope on the equal-weighted market excess return, 60 months, >= 20
      (Beta; Fama & MacBeth 1973). OSAP uses CRSP's equal-weighted index; ours is
      the mean `ret` over every stock in the panel that month, known at its end.
    - `beta_liquidity_ps`: slope on the Pastor-Stambaugh liquidity innovation, with
      mktrf, smb and hml as controls, 60 months, >= 36 (BetaLiquidityPS; Pastor &
      Stambaugh 2003). Caveat: they re-estimate the whole liquidity series every
      year, so its values depend on later data (as in OSAP); months after the
      file's last update simply drop out of the window.
    Windows run over calendar months, not rows: a gap cannot stretch them.
    """
    panel = _sorted_by_stock(panel)
    excess, factors = _excess_grid(panel, factors)
    ew_market = panel.groupby("month_end")["ret"].mean().reindex(excess.index) - factors["rf"]

    beta = _rolling_ols(excess, ew_market.to_frame("ewmktrf"), BETA_MONTHS, BETA_MIN)
    ps = _rolling_ols(excess, factors[["ps_innov", "mktrf", "smb", "hml"]], PS_MONTHS, PS_MIN)
    return _with(
        panel,
        beta=_at_rows(beta["ewmktrf"], panel, excess),
        beta_liquidity_ps=_at_rows(ps["ps_innov"], panel, excess),
    )


RESMOM_FIT, RESMOM_MONTHS = 36, 11  # ResidualMomentum: 36-month FF3 fits, 11 residuals
COSKEW_MONTHS, COSKEW_MIN = 60, 12  # Coskewness: >= 12 of the last 60 months


def add_residual_momentum_coskewness(panel: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Add `residual_momentum` and `coskewness` (OSAP's code; `factors` as in
    `add_factor_betas`). Calendar-month windows, this month and earlier only.

    - `residual_momentum`: each month s, the residual of s in an FF3 regression of
      excess returns over months s-35 .. s (all 36 required). The signal is the mean
      over the std (ddof 1) of those residuals for months t-11 .. t-1, all 11
      required; month t is skipped as in momentum (ResidualMomentum; Blitz, Huij &
      Martens 2011).
    - `coskewness`: E[r m^2] / (sqrt(E[r^2]) E[m^2]) over the last 60 months
      (>= 12), with r the stock's and m the market's (mktrf) excess return, both
      demeaned over the months the stock has a return (Coskewness; Harvey &
      Siddique 2000, eq. B-9).
    """
    panel = _sorted_by_stock(panel)
    excess, factors = _excess_grid(panel, factors)

    ff3 = factors[["mktrf", "smb", "hml"]]
    fit = _rolling_ols(excess, ff3, RESMOM_FIT, RESMOM_FIT)
    fitted = fit["const"] + sum(fit[c].mul(ff3[c], axis=0) for c in ff3.columns)
    lagged = (excess - fitted).shift(1)
    resid_mean = lagged.rolling(RESMOM_MONTHS, min_periods=RESMOM_MONTHS).mean()
    resid_sd = lagged.rolling(RESMOM_MONTHS, min_periods=RESMOM_MONTHS).std()
    residual_momentum = resid_mean / resid_sd.where(resid_sd > 0)

    valid = excess.notna() & factors["mktrf"].notna().to_numpy()[:, None]
    r = excess.where(valid)
    m = valid.mul(factors["mktrf"], axis=0).where(valid)

    def mean(values: pd.DataFrame) -> pd.DataFrame:
        return values.rolling(COSKEW_MONTHS, min_periods=COSKEW_MIN).mean()

    a, b = mean(r), mean(m)
    # Expanding E[(r-a)(m-b)^2], E[(r-a)^2] and E[(m-b)^2] into means of raw powers.
    co = mean(r * m * m) - 2 * b * mean(r * m) - a * mean(m * m) + 2 * a * b * b
    var_r, var_m = mean(r * r) - a * a, mean(m * m) - b * b
    coskewness = co / (np.sqrt(var_r.where(var_r > 0)) * var_m.where(var_m > 0))

    return _with(
        panel,
        residual_momentum=_at_rows(residual_momentum, panel, excess),
        coskewness=_at_rows(coskewness, panel, excess),
    )


MIN_FACTOR_DAYS = 15  # IdioVol3F, RealizedVol, ReturnSkew3F, betaVIX: >= 15 days
AHT_MONTHS, AHT_MIN_DAYS = 12, 100  # IdioVolAHT: 12 months of days, >= 100


def add_daily_factor_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `idio_vol_3f`, `return_skew_3f`, `realized_vol`, `idio_vol_aht`, `beta_vix`
    from the daily factor-regression summaries (`data.real.FACTOR_SUMMARY_COLUMNS`;
    OSAP's code). Days up to month_end only; NaN in months without factor data.

    - `idio_vol_3f`: std (ddof 1) of the residuals of the month's daily FF3
      regression, >= 15 days (IdioVol3F; Ang, Hodrick, Xing & Zhang 2006).
    - `return_skew_3f`: skewness of those residuals (ReturnSkew3F; Bali, Engle &
      Murray 2016). Residuals have mean 0 (intercept), so m3 / m2^1.5.
    - `realized_vol`: std (ddof 1) of the month's daily excess returns, >= 15 days
      (RealizedVol; Ang et al. 2006): total volatility, not residual.
    - `idio_vol_aht`: root mean squared error (n - 2 dof) of a CAPM regression on the
      daily excess returns of the last 12 calendar months, >= 100 days (IdioVolAHT;
      Ali, Hwang & Trombley 2003). OSAP uses the last 252 trading days; 12 whole
      months is about the same and an exact regression from the monthly sums.
    - `beta_vix`: the month's slope on the daily VIX change with mktrf as a control,
      >= 15 days (betaVIX; Ang et al. 2006). OSAP uses the last 20 trading days;
      the month's ~21 days is about the same window.
    """
    panel = _sorted_by_stock(panel)
    n = panel["xret_n"]
    enough = n >= MIN_FACTOR_DAYS

    m2 = panel["ff3_resid_sumsq"] / n
    skew = (panel["ff3_resid_sumcube"] / n) / m2.where(m2 > 0) ** 1.5
    xvar = (panel["xret_sumsq"] - panel["xret_sum"] ** 2 / n) / (n - 1)

    sums = ["xret_n", "xret_sum", "xret_sumsq", "mkt_sum", "mkt_sumsq", "xret_mkt_sum"]
    grids = {c: _calendar_grid(panel, c) for c in sums}
    roll = {c: g.rolling(AHT_MONTHS, min_periods=1).sum() for c, g in grids.items()}
    days = roll["xret_n"]
    sxx = roll["mkt_sumsq"] - roll["mkt_sum"] ** 2 / days
    sxy = roll["xret_mkt_sum"] - roll["mkt_sum"] * roll["xret_sum"] / days
    syy = roll["xret_sumsq"] - roll["xret_sum"] ** 2 / days
    ssr = (syy - sxy**2 / sxx.where(sxx > 0)).clip(lower=0)
    rmse = np.sqrt(ssr / (days - 2)).where(days >= AHT_MIN_DAYS)

    return _with(
        panel,
        idio_vol_3f=np.sqrt(panel["ff3_resid_sumsq"].clip(lower=0) / (n - 1)).where(enough),
        return_skew_3f=skew.where(enough),
        realized_vol=np.sqrt(xvar.clip(lower=0)).where(enough),
        idio_vol_aht=_at_rows(rmse, panel, grids["xret_n"]),
        beta_vix=panel["vix_beta"].where(panel["vix_n"] >= MIN_FACTOR_DAYS),
    )


ACX_MONTHS, ACX_SLACK = 12, 5  # CoskewACX: 12 months, within 5 days of the fullest stock
FP_VOL_MONTHS, FP_VOL_MIN = 12, 120  # BetaFP volatilities: 12 months, >= 120 days
FP_CORR_MONTHS, FP_CORR_MIN = 60, 750  # BetaFP correlation: 60 months, >= 750 3-day returns


def add_coskew_acx_beta_fp(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `coskew_acx` and `beta_fp` from daily log excess returns (the `lx_*`,
    `lm_*` and `l3*` sums of data.real.daily_factor_summaries; OSAP's code).
    Windows are calendar months up to month_end.

    - `coskew_acx`: E[x m^2] / (sqrt(E[x^2]) E[m^2]), x and m the stock's and the
      market's daily log excess returns demeaned over the last 12 months (CoskewACX;
      Ang, Chen & Xing 2006). As OSAP, only stocks with at most 5 days fewer than
      the month's fullest stock: in effect a full year.
    - `beta_fp`: |corr| x sd(x) / sd(m) (BetaFP; Frazzini & Pedersen 2014). The
      correlation is of overlapping 3-day log returns over 60 months (>= 750), so
      stale prices of thinly traded stocks wash out; the volatilities are of daily
      log returns over 12 months (>= 120 days, ddof 1). OSAP's windows are 1,260
      and 252 trading days; whole months are about the same.
    """
    panel = _sorted_by_stock(panel)
    cols = ["xret_n", "lx_sum", "lx_sumsq", "lm_sum", "lm_sumsq", "lx_lm_sum", "lx_lm2_sum"]
    cols += ["l3_n", "l3x_sum", "l3x_sumsq", "l3m_sum", "l3m_sumsq", "l3x_l3m_sum"]
    grids = {c: _calendar_grid(panel, c) for c in cols}

    def summed(names: list[str], months: int) -> dict[str, pd.DataFrame]:
        return {c: grids[c].rolling(months, min_periods=1).sum() for c in names}

    w = summed(cols[:7], ACX_MONTHS)
    n = w["xret_n"].where(w["xret_n"] > 0)
    a, b = w["lx_sum"] / n, w["lm_sum"] / n
    co = w["lx_lm2_sum"] / n - 2 * b * w["lx_lm_sum"] / n - a * w["lm_sumsq"] / n + 2 * a * b * b
    var_x, var_m = w["lx_sumsq"] / n - a * a, w["lm_sumsq"] / n - b * b
    acx = co / (np.sqrt(var_x.where(var_x > 0)) * var_m.where(var_m > 0))
    full = n >= n.max(axis=1).to_numpy()[:, None] - ACX_SLACK
    acx = acx.where(full)

    v = summed(cols[:5], FP_VOL_MONTHS)
    days = v["xret_n"]
    sd_x = np.sqrt((v["lx_sumsq"] - v["lx_sum"] ** 2 / days) / (days - 1))
    sd_m = np.sqrt((v["lm_sumsq"] - v["lm_sum"] ** 2 / days) / (days - 1))
    ratio = (sd_x / sd_m.where(sd_m > 0)).where(days >= FP_VOL_MIN)

    c = summed(cols[7:], FP_CORR_MONTHS)
    k = c["l3_n"]
    sxy = c["l3x_l3m_sum"] - c["l3x_sum"] * c["l3m_sum"] / k
    sxx = c["l3x_sumsq"] - c["l3x_sum"] ** 2 / k
    syy = c["l3m_sumsq"] - c["l3m_sum"] ** 2 / k
    corr = (sxy / np.sqrt((sxx * syy).where((sxx > 0) & (syy > 0)))).where(k >= FP_CORR_MIN)

    like = grids["xret_n"]
    return _with(
        panel,
        coskew_acx=_at_rows(acx, panel, like),
        beta_fp=_at_rows(corr.abs() * ratio, panel, like),
    )


TAIL_MONTHS, TAIL_MIN = 120, 72  # BetaTailRisk: >= 72 of the last 120 months


def add_tail_risk_beta(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `beta_tail_risk`: the slope of the stock's monthly return (raw, as OSAP) on
    the market tail index (`market_tail`, data.real.market_tail_index) over the last
    120 calendar months, >= 72 (BetaTailRisk; Kelly & Jiang 2014). Both are known at
    each month's end. Our data start in Dec 1997, so the first values are in 2003.
    """
    panel = _sorted_by_stock(panel)
    rets = _calendar_grid(panel, "ret")
    tail = panel.groupby("month_end")["market_tail"].first().reindex(rets.index)
    fit = _rolling_ols(rets, tail.to_frame("tail"), TAIL_MONTHS, TAIL_MIN)
    return _with(panel, beta_tail_risk=_at_rows(fit["tail"], panel, rets))


PD_MONTHS, PD_MIN_DAYS = 12, 26  # price delay: July-June daily regressions, >= 26 days
PD_FIT_MONTH = 6  # fitted at each June's end, used from July (one month skipped)


def add_price_delay(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `price_delay_rsq`, `price_delay_slope`, `price_delay_tstat`: how slowly a
    stock's price takes in market news (Hou & Moskowitz 2005; OSAP's code).

    Each June, regress the stock's daily excess returns from July to June (>= 26
    days, at least one in June) on mktrf and its 4 previous trading days' values,
    from the `pd_*` sums of data.real.daily_factor_summaries. With b0 the slope on
    today's market and b1 .. b4 those on the lags, t their t-statistics:
    - `price_delay_rsq` (D1): 1 - R^2 without the lags / R^2 with them. The short
      regression uses the same days.
    - `price_delay_slope` (D2): sum(n b_n) / (b0 + sum(b_n)).
    - `price_delay_tstat` (D3): D2 with t-statistics, sum(n t_n) / (t0 + sum(t_n)),
      as the paper and OSAP's Python code (its Stata code divides by b0). We skip
      OSAP's Stata 10/90 trim: the rank normalisation already tames outliers.
    The values sit on July's row (one month skipped, as the paper) and the next 11,
    so each row only uses days up to the June before it.
    """
    panel = _sorted_by_stock(panel)
    grids = {c: _calendar_grid(panel, c) for c in PRICE_DELAY_COLUMNS}
    like = grids["pd_yy"]
    june = np.flatnonzero(like.index.month == PD_FIT_MONTH)
    year = {c: g.rolling(PD_MONTHS, min_periods=1).sum().to_numpy()[june] for c, g in grids.items()}
    traded_in_june = grids["pd_xx00"].to_numpy()[june] > 0
    rows, cols = np.nonzero(traded_in_june & (year["pd_xx00"] >= PD_MIN_DAYS))

    k = PRICE_DELAY_K
    xtx = np.empty((len(rows), k, k))
    xty = np.empty((len(rows), k))
    for i in range(k):
        xty[:, i] = year[f"pd_xy{i}"][rows, cols]
        for j in range(i, k):
            xtx[:, i, j] = xtx[:, j, i] = year[f"pd_xx{i}{j}"][rows, cols]
    yy, days = year["pd_yy"][rows, cols], xtx[:, 0, 0]

    b = solve_normal_equations(xtx, xty)
    short = solve_normal_equations(xtx[:, :2, :2], xty[:, :2])
    ssr = yy - np.einsum("ij,ij->i", b, xty)
    ssr_short = yy - np.einsum("ij,ij->i", short, xty[:, :2])
    sst = yy - xty[:, 0] ** 2 / days
    inv_diag = np.full(b.shape, np.nan)
    fitted = ~np.isnan(b[:, 0])
    inv_diag[fitted] = np.diagonal(np.linalg.inv(xtx[fitted]), axis1=1, axis2=2)
    weights = np.arange(1, PRICE_DELAY_LAGS + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = b / np.sqrt((np.clip(ssr, 0, None) / (days - k))[:, None] * inv_diag)
        r2, r2_short = 1 - ssr / sst, 1 - ssr_short / sst
        values = {
            "price_delay_rsq": np.where(r2 > 0, 1 - r2_short / r2, np.nan),
            "price_delay_slope": b[:, 2:] @ weights / (b[:, 1] + b[:, 2:].sum(axis=1)),
            "price_delay_tstat": t[:, 2:] @ weights / (t[:, 1] + t[:, 2:].sum(axis=1)),
        }

    added = {}
    for name, value in values.items():
        grid = np.full(like.shape, np.nan)
        grid[june[rows], cols] = value
        fit = pd.DataFrame(grid, index=like.index, columns=like.columns)
        added[name] = _at_rows(fit.shift(1).ffill(limit=PD_MONTHS - 1), panel, like)
    return _with(panel, **added)


INDMOM_LAGS = (1, 5)  # IndMom: each stock's return over months t-5 .. t-1
SIC2 = 100  # IndMom: 2-digit SIC industries
BIG_FIRMS = 0.7  # IndRetBig: the industry's largest 30% by market cap


def add_industry_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `ind_mom` and `ind_ret_big` (OSAP's code). Industries are today's codes
    (data.real.add_industry_codes), so both are approximate.

    - `ind_mom`: mean over the stocks in the same 2-digit SIC industry this month,
      weighted by this month's market cap, of each one's compounded return over
      months t-5 .. t-1 (IndMom; Grinblatt & Moskowitz 1999). As OSAP, a missing
      return on a row the stock has counts as 0; a month without a row gives NaN.
    - `ind_ret_big`: the equal-weighted mean return this month of the industry's
      largest 30% of firms by market cap (percentile rank above 0.7), Fama-French 48
      industries, given to the other firms; NaN for the big firms themselves
      (IndRetBig; Hou 2007). News reaches big firms first.
    Same-month cross-sections over the rows passed in (all stocks, before the
    investable filter) and the stocks' earlier months: known at month_end.
    """
    panel = _sorted_by_stock(panel)
    rets = _calendar_grid(panel.assign(_r=panel["ret"].fillna(0)), "_r")
    growth = pd.DataFrame(1.0, index=rets.index, columns=rets.columns)
    for lag in range(INDMOM_LAGS[0], INDMOM_LAGS[1] + 1):
        growth = growth * (1 + rets.shift(lag))
    mom = pd.Series(_at_rows(growth - 1, panel, rets), index=panel.index)
    mcap = panel["market_cap"].where(panel["market_cap"] > 0)
    sic2 = [panel["month_end"], panel["sic_code"] // SIC2]
    use = mom.notna() & mcap.notna()
    weighted = (mom * mcap).where(use).groupby(sic2).transform("sum")
    weights = mcap.where(use).groupby(sic2).transform("sum")

    ff48 = [panel["month_end"], panel["fama_industry"]]
    rank = mcap.groupby(ff48).rank(pct=True)
    big_ret = panel["ret"].where(rank > BIG_FIRMS).groupby(ff48).transform("mean")
    return _with(
        panel,
        ind_mom=weighted / weights.where(weights > 0),
        ind_ret_big=big_ret.where(~(rank >= BIG_FIRMS)),
    )


HERF_MONTHS, HERF_MIN = 36, 12  # Herf: own mean of the industry index, >= 12 of 36 months
UTILITIES = (4900, 4999)  # Herf: regulated utilities (SIC 49) excluded


def add_industry_accounting(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `herf`, `herf_asset`, `herf_be` and `ch_inv_ia` (OSAP's code). Industries
    are today's SIC codes (data.real.add_industry_codes): approximate.

    - `herf`: each month, the Herfindahl index of sales in the stock's 4-digit SIC
      industry, sum of (firm sales / industry sales)^2 over the rows passed in;
      the signal is the stock's own mean of it over the last 36 months (>= 12).
      NaN for utilities, SIC 49 (Herf; Hou & Robinson 2006). OSAP's other
      regulated-industry exclusions end by 1982, before our data.
    - `herf_asset`, `herf_be`: the same with total assets and with book equity
      (HerfAsset, HerfBE). Book equity: equity, or assets - liabilities, plus
      `taxliabilities` for deferred taxes (as `book_leverage`); OSAP's preferred
      stock terms are left out. All three in USD (`_usd`), as firms are summed.
    - `ch_inv_ia`: % change in capex vs. the mean of 12 and 24 months ago (or vs.
      12 months ago where that is missing), minus the month's mean of it over the
      stock's 2-digit SIC industry (ChInvIA; Abarbanell & Bushee 1998). Capex is
      -`capex` (see `add_investment_growth`), missing filled by the 12-month change
      in net PP&E.
    Same-month cross-sections and the stocks' own earlier rows: known at month_end.
    """
    panel = _sorted_by_stock(panel)
    sic4 = [panel["month_end"], panel["sic_code"]]
    utility = panel["sic_code"].between(*UTILITIES)

    def herfindahl(values: pd.Series) -> pd.Series:
        total = values.groupby(sic4).transform("sum")
        index = ((values / total.where(total != 0)) ** 2).groupby(sic4).transform("sum")
        index = index.where(panel["sic_code"].notna())
        return _own_rolling(panel, index, HERF_MONTHS, HERF_MIN).where(~utility)

    book = panel["equity"].fillna(panel["assets"] - panel["liabilities"])
    book = (book + panel["taxliabilities"].fillna(0)) / panel["fxusd"].where(panel["fxusd"] > 0)

    def ago(values: str | pd.Series, months: int) -> pd.Series:
        return _months_ago(panel, values, months)

    capex = (-panel["capex"]).fillna(panel["ppnenet"] - ago("ppnenet", 12))
    before, year_ago = (ago(capex, 12) + ago(capex, 24)) / 2, ago(capex, 12)
    change = (capex - before) / before.where(before != 0)
    change = change.fillna((capex - year_ago) / year_ago.where(year_ago != 0))
    sic2 = [panel["month_end"], panel["sic_code"] // SIC2]
    return _with(
        panel,
        herf=herfindahl(_usd(panel, "revenue")),
        herf_asset=herfindahl(_usd(panel, "assets")),
        herf_be=herfindahl(book),
        ch_inv_ia=change - change.groupby(sic2).transform("mean"),
    )


def _group_residuals(
    keys: pd.DataFrame, y: pd.Series, xs: list[pd.Series], min_obs: int
) -> pd.Series:
    """Residuals of an OLS of `y` on `xs` with an intercept within each group of the
    `keys` columns, over rows with every value (NaN elsewhere, and in groups with
    fewer than `min_obs` such rows)."""
    use = y.notna() & pd.concat(xs, axis=1).notna().all(axis=1) & keys.notna().all(axis=1)
    codes = keys[use].groupby(list(keys.columns)).ngroup()
    codes = codes[codes.map(codes.value_counts()) >= min_obs]
    rows, c = codes.index, codes.to_numpy()
    cols = [np.ones(len(rows)), *(x[rows].to_numpy() for x in xs)]
    yv = y[rows].to_numpy()
    b = group_ols(c, int(c.max(initial=-1)) + 1, yv, cols, np.ones(len(rows), bool))
    resid = pd.Series(np.nan, index=y.index)
    resid[rows] = yv - sum(b[c, i] * cols[i] for i in range(len(cols)))
    return resid


ABACC_MIN_FIRMS = 6  # AbnormalAccruals: >= 6 firms in the industry-month regression
ABACC_TRIM = (0.001, 0.999)  # AbnormalAccruals: inputs outside the month's 0.1-99.9% dropped


def add_abnormal_accruals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `abnormal_accruals`: the residual of a cross-sectional modified-Jones
    regression (AbnormalAccruals; Xie 2001; OSAP's code).

    Each month and 2-digit SIC industry (today's code: approximate), regress
    accruals / lagged assets on 1 / lagged assets, the 12-month change in revenue /
    lagged assets and net PP&E / lagged assets, with an intercept, over firms with
    all four (>= 6 firms). Accruals = net income - operating cash flow; lagged
    assets are the stock's own row 12 months earlier. OSAP runs it once per fiscal
    year on Compustat (income before extraordinary items, gross PP&E); ours runs
    on the latest public filings each month. Inputs in USD (`_usd`), since 1 /
    assets depends on the currency; each is dropped outside the month's 0.1-99.9%
    percentiles, as OSAP's yearly trim. Known at month_end.
    """
    panel = _sorted_by_stock(panel)
    lagged = _months_ago(panel, _usd(panel, "assets"), 12)
    lagged = lagged.where(lagged > 0)
    revenue = _usd(panel, "revenue")
    inputs = pd.DataFrame(
        {
            "accruals": (_usd(panel, "netinc") - _usd(panel, "ncfo")) / lagged,
            "inv_assets": 1 / lagged,
            "d_revenue": (revenue - _months_ago(panel, revenue, 12)) / lagged,
            "ppe": _usd(panel, "ppnenet") / lagged,
        }
    )
    by_month = inputs.groupby(panel["month_end"])
    low = by_month.transform("quantile", ABACC_TRIM[0])
    high = by_month.transform("quantile", ABACC_TRIM[1])
    inputs = inputs.where((inputs >= low) & (inputs <= high))

    keys = pd.DataFrame({"m": panel["month_end"], "s": panel["sic_code"] // SIC2})
    xs = [inputs[c] for c in ["inv_assets", "d_revenue", "ppe"]]
    resid = _group_residuals(keys, inputs["accruals"], xs, ABACC_MIN_FIRMS)
    return _with(panel, abnormal_accruals=resid)


SURPRISE_LAGS = range(3, 25, 3)  # EarningsSurprise: the last 8 quarters' surprises
MIN_SURPRISE_SD = 1e-8  # EarningsSurprise: a flat history has no scale
CONSISTENCY_YEARS = 4  # EarningsConsistency: this year's growth and the 4 before
CONSISTENCY_MAX_RATIO = 6  # EarningsConsistency: |EPS / EPS a year ago| above 6 excluded


def _lag_moments(grid: pd.DataFrame, lags: range) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and std (ddof 1) of `grid` shifted by each of `lags` months, over the
    lags with a value (NaN with none, std NaN with one)."""
    total = count = squares = 0
    for lag in lags:
        shifted = grid.shift(lag)
        total = total + shifted.fillna(0)
        squares = squares + (shifted**2).fillna(0)
        count = count + shifted.notna()
    n = count.where(count > 0)
    var = (squares - total**2 / n) / (n - 1).where(n > 1)
    return total / n, np.sqrt(var.clip(lower=0))


def _standardised_surprise(panel: pd.DataFrame, values: pd.Series) -> pd.Series:
    """OSAP's seasonal random walk with drift: growth g = x - x 12 months earlier;
    surprise = g - mean(g 3, 6, .. 24 months ago), over the std of the surprises
    3, 6, .. 24 months ago (each over the lags available)."""
    keys = panel[["permaticker", "month_end"]].assign(v=values)
    grid = _calendar_grid(keys, "v")
    growth = grid - grid.shift(12)
    drift, _ = _lag_moments(growth, SURPRISE_LAGS)
    surprise = growth - drift
    _, sd = _lag_moments(surprise, SURPRISE_LAGS)
    standard = surprise / sd.where(sd > MIN_SURPRISE_SD)
    return pd.Series(_at_rows(standard, keys, grid), index=panel.index)


def add_earnings_surprises(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `earnings_surprise`, `revenue_surprise`, `earn_sup_big`, `earnings_consistency`
    (OSAP's code). The latest public quarter (`_q`) or trailing year on each month's
    row; lags are the stock's own rows that many calendar months earlier.

    - `earnings_surprise`: quarterly EPS vs. the same quarter a year before, minus
      the mean of that change over the last 8 quarters, over the std of the last 8
      such surprises (EarningsSurprise; Foster, Olsen & Shevlin 1984).
    - `revenue_surprise`: the same with quarterly revenue per share, revenue over
      basic shares (RevenueSurprise; Jegadeesh & Livnat 2006).
    - `earn_sup_big`: the mean `earnings_surprise` of the largest 30% of firms by
      market cap (percentile rank >= 0.7) in the Fama-French 48 industry this month,
      for the other firms; NaN for the big ones (EarnSupBig; Hou 2007). Today's
      industry codes: approximate.
    - `earnings_consistency`: mean over this year and the 4 before (each 12 months
      apart, those available) of EPS growth, (EPS - EPS a year ago) / mean of |EPS| one
      and two years ago, on trailing-year EPS (EarningsConsistency; Alwathainani
      2009). NaN if EPS now or a year ago is missing, |EPS / EPS a year ago| > 6, or
      growth changed sign since a year ago (or is negative after a missing year).
    Sharadar's EPS is basic EPS, net income to common (OSAP: excluding extraordinary
    items). Known at month_end.
    """
    panel = _sorted_by_stock(panel)
    earnings = _standardised_surprise(panel, panel["eps_q"])
    shares = panel["sharesbas_q"].where(panel["sharesbas_q"] > 0)
    revenue = _standardised_surprise(panel, panel["revenue_q"] / shares)

    mcap = panel["market_cap"].where(panel["market_cap"] > 0)
    ff48 = [panel["month_end"], panel["fama_industry"]]
    big = mcap.groupby(ff48).rank(pct=True) >= BIG_FIRMS
    big_mean = earnings.where(big).groupby(ff48).transform("mean")

    def ago(values: str | pd.Series, months: int) -> pd.Series:
        return _months_ago(panel, values, months)

    eps, eps_1y, eps_2y = panel["eps"], ago("eps", 12), ago("eps", 24)
    scale = 0.5 * (eps_1y.abs() + eps_2y.abs())
    growth = ((eps - eps_1y) / scale.where(scale != 0)).replace([np.inf, -np.inf], np.nan)
    past = [ago(growth, 12 * y) for y in range(1, CONSISTENCY_YEARS + 1)]
    consistency = pd.concat([growth, *past], axis=1).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        jump = (eps / eps_1y).abs() > CONSISTENCY_MAX_RATIO
    flipped = ((growth > 0) & (past[0] < 0)) | ((growth < 0) & ~(past[0] <= 0))
    excluded = eps.isna() | eps_1y.isna() | jump | flipped
    return _with(
        panel,
        earnings_surprise=earnings,
        revenue_surprise=revenue,
        earn_sup_big=big_mean.where(~big),
        earnings_consistency=consistency.where(~excluded),
    )


DIVINIT_QUIET, DIVINIT_MONTHS = 24, 6  # DivInit: first dividend in 24 months, last 6 months
# DivOmit: (k, L) a regular payer of one dividend every k months over the last L months.
DIVOMIT_RULES = [(3, 18), (6, 18), (12, 24)]
DIV_BURN_IN = {"div_init": 24, "div_omit": 36, "div_season": 12, "div_yield_st": 12}
QUARTERLY_LIKE = (0, 1, 3)  # DivSeason, DivYieldST: unknown or quarterly frequency
# DivSeason: months back that put a dividend in month t+1, by frequency code.
SEASON_LAGS = {3: (2, 5, 8, 11), 4: (5, 11), 5: (11,)}
EXPECTED_LAG = {3: 2, 4: 5, 5: 11}  # DivYieldST: last dividend expected to repeat at t+1


def add_dividend_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `div_init`, `div_omit`, `div_season`, `div_yield_st` (OSAP's code) from the
    monthly cash dividends (`div_amount`, `div_freq`: data.real.build_action_months).

    Windows are the stock's own calendar months; a month without a dividend is 0.
    Dividend data start with the panel, so each signal is NaN until it has a full
    window after the panel's first month (`DIV_BURN_IN`): before that, "no dividend
    in the last 24 months" would just mean no data.
    - `div_init`: 1 if in one of the last 6 months the stock paid its first dividend
      in 24 months, else 0 (DivInit; Michaely, Thaler & Womack 1995).
    - `div_omit`: 1 if the stock just stopped a regular dividend: for k = 3, 6, 12, it
      paid in the k months to t-1 but none in the k to t, after a run where every
      k-month window held exactly one dividend (18, 18, 24 months, ending k months
      ago); 1 only if that happened at exactly one of t-1 and t (DivOmit; same paper).
    - `div_season`: 1 if a dividend is due next month by the stock's frequency
      (quarterly or unknown: paid 2, 5, 8 or 11 months ago; semi-annual: 5 or 11;
      annual: 11), 0 if it paid in the last 12 months but none is due, NaN if it
      paid none; monthly payers NaN (DivSeason; Hartzmark & Salomon 2013).
    - `div_yield_st`: the dividend expected next month (the one 2, 5 or 11 months
      ago by frequency) over the month-end split-adjusted price, in terciles 1-3
      each month over the positive ones, 0 if none expected; only for stocks that
      paid in the last 12 months, monthly payers NaN (DivYieldST; Litzenberger &
      Ramaswamy 1979).
    Frequency codes are inferred (see data.real.DIVIDEND_FREQUENCIES), and Sharadar
    does not mark special dividends: approximate. Known at month_end.
    """
    panel = _sorted_by_stock(panel)
    keys = panel[["permaticker", "month_end"]]
    amount = _calendar_grid(keys.assign(v=panel["div_amount"].fillna(0)), "v")
    exists = amount.notna()
    paid = (amount > 0).astype(float).where(exists)
    freq = _calendar_grid(keys.assign(v=panel["div_freq"]), "v").where(paid > 0)
    freq = freq.ffill().where(exists)

    def window_sum(grid: pd.DataFrame, months: int) -> pd.DataFrame:
        return grid.rolling(months, min_periods=1).sum().where(exists)

    quiet = window_sum(paid, DIVINIT_QUIET).shift(1)
    first = ((paid > 0) & (quiet == 0)).astype(float)
    div_init = first.rolling(DIVINIT_MONTHS, min_periods=1).max().where(exists)

    omit_now = pd.DataFrame(False, index=paid.index, columns=paid.columns)
    for k, run in DIVOMIT_RULES:
        count = window_sum(paid, k)
        regular = (count == 1).astype(float).where(exists)
        payer = regular.rolling(run, min_periods=1).min() == 1
        omit_now |= (count == 0) & (count.shift(1) > 0) & payer.shift(k, fill_value=False)
    div_omit = (omit_now.astype(float).rolling(2, min_periods=1).sum() == 1).astype(float)
    div_omit = div_omit.where(exists)

    recent = window_sum(paid, 12) > 0
    due = pd.DataFrame(False, index=paid.index, columns=paid.columns)
    for code, lags in SEASON_LAGS.items():
        like = freq.isin(QUARTERLY_LIKE) if code == 3 else freq == code
        for lag in lags:
            due |= like & (paid.shift(lag) > 0)
    seasonal = freq.notna() & (freq != 2)
    div_season = due.astype(float).where(recent | due).where(seasonal & exists)

    expected = pd.DataFrame(np.nan, index=paid.index, columns=paid.columns)
    for code, lag in EXPECTED_LAG.items():
        like = freq.isin(QUARTERLY_LIKE) | freq.isna() if code == 3 else freq == code
        expected = expected.mask(like, amount.shift(lag))
    price = _calendar_grid(panel, "close_split_adj")
    yield_ = expected / price.where(price > 0)
    yield_ = yield_.where(recent & (freq != 2) & exists)
    positive = yield_.where(yield_ > 0)
    tercile = positive.apply(
        lambda row: pd.qcut(row, 3, labels=False, duplicates="drop") + 1, axis=1
    )
    div_yield_st = tercile.mask(yield_ == 0, 0.0)

    out = {}
    start = amount.index[0]
    for name, grid in [
        ("div_init", div_init),
        ("div_omit", div_omit),
        ("div_season", div_season),
        ("div_yield_st", div_yield_st),
    ]:
        ready = amount.index >= start + pd.offsets.MonthEnd(DIV_BURN_IN[name])
        out[name] = _at_rows(grid.where(pd.Series(ready, index=grid.index), axis=0), keys, amount)
    return _with(panel, **out)


EXCH_SWITCH_MONTHS = 12  # ExchSwitch: moved up an exchange within the last 12 months
SPINOFF_MONTHS = 24  # Spinoff: spun off within the last 24 months


def add_event_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `exch_switch` and `spinoff` from Sharadar's corporate actions
    (`exch_change`, `exch_upgrade`, `spun_off`: data.real.build_action_months).

    - `exch_switch`: 1 if the stock's latest exchange change, in months t-11 .. t,
      was a move up (Nasdaq or NYSE MKT to NYSE, Nasdaq to NYSE MKT), else 0
      (ExchSwitch; Dharan & Ikenberry 1995). OSAP compares CRSP's monthly exchange
      codes with the last 12 months' instead; the same up to moves within a month.
    - `spinoff`: 1 if the stock was spun off from another company in months
      t-23 .. t, else 0 (Spinoff; Cusatis, Miles & Woolridge 1993). OSAP flags
      CRSP spin-off companies in their first 24 months, which is about the same.
    Events up to month_end only.
    """
    panel = _sorted_by_stock(panel)
    keys = panel[["permaticker", "month_end"]]

    def grid(column: str) -> pd.DataFrame:
        return _calendar_grid(keys.assign(v=panel[column].fillna(0)), "v")

    change, upgrade, spun = grid("exch_change"), grid("exch_upgrade"), grid("spun_off")
    exists = change.notna()
    month = np.arange(len(change))[:, None]

    def last_event(events: pd.DataFrame) -> pd.DataFrame:
        """The grid row of each stock's latest event so far (NaN before the first)."""
        at = np.where(events.to_numpy() > 0, month, np.nan)
        return pd.DataFrame(at, index=events.index, columns=events.columns).ffill()

    last_upgrade = last_event(upgrade)
    recent = (month - last_upgrade < EXCH_SWITCH_MONTHS) & (last_upgrade == last_event(change))
    spinoff = spun.rolling(SPINOFF_MONTHS, min_periods=1).max() > 0
    return _with(
        panel,
        exch_switch=_at_rows(recent.astype(float).where(exists), keys, change),
        spinoff=_at_rows(spinoff.astype(float).where(exists), keys, change),
    )


IPO_WINDOW = (3, 36)  # IndIPO: months 3 to 36 after the IPO
RDIPO_WINDOW = (7, 36)  # RDIPO: months 7 to 36 after the IPO
FIRMAGEMOM_MIN_AGE = 12  # FirmAgeMom: at least 12 months in the data
FIRMAGEMOM_LAGS = (1, 5)  # FirmAgeMom: return over months t-5 .. t-1
MIN_PRICE_FIRMAGEMOM = 5.0  # FirmAgeMom: OSAP's $5 price screen


def add_ipo_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `ind_ipo`, `rd_ipo` and `firm_age_mom` (OSAP's code), with the stock's first
    month in our data standing in for its IPO date (OSAP: Ritter's IPO dates).

    A stock already in the panel's first month listed before our data start, so it
    counts as no IPO (0), as OSAP does for stocks without an IPO date. A stock that
    first appears later may be a new listing rather than an IPO (an uplisting from
    OTC, a spin-off, a relisting): approximate.
    - `ind_ipo`: 1 in months 3 to 36 after the IPO month, else 0 (IndIPO; Ritter 1991).
    - `rd_ipo`: 1 in months 7 to 36 after the IPO for firms reporting zero R&D, else
      0 (RDIPO; Gou, Lev & Shi 2006). Sharadar reports no R&D as 0.
    - `firm_age_mom`: the return over months t-5 .. t-1 (a missing return on a row
      the stock has counts as 0, as OSAP), for the youngest fifth of firms by months
      in the data each month, among those with 12 or more months and a price of at
      least $5 (or none); NaN otherwise (FirmAgeMom; Zhang 2006). Our data start in
      Dec 1997, so ages are truncated for older firms, which only puts them in the
      oldest fifths.
    Uses the stock's own months up to month_end.
    """
    panel = _sorted_by_stock(panel)
    listed = _months_listed(panel)
    since = listed - 1
    first_month = panel.groupby("permaticker")["month_end"].transform("min")
    ipo = first_month > panel["month_end"].min()
    ind_ipo = (ipo & since.between(*IPO_WINDOW)).astype(float)
    rd_ipo = (ipo & since.between(*RDIPO_WINDOW) & (panel["rnd"] == 0)).astype(float)

    rets = _calendar_grid(panel.assign(_r=panel["ret"].fillna(0)), "_r")
    growth = pd.DataFrame(1.0, index=rets.index, columns=rets.columns)
    for lag in range(FIRMAGEMOM_LAGS[0], FIRMAGEMOM_LAGS[1] + 1):
        growth = growth * (1 + rets.shift(lag))
    mom = pd.Series(_at_rows(growth - 1, panel, rets), index=panel.index)
    price = panel["close_unadj"]
    eligible = (listed >= FIRMAGEMOM_MIN_AGE) & ((price >= MIN_PRICE_FIRMAGEMOM) | price.isna())
    age = listed.where(eligible)
    fifth = age.groupby(panel["month_end"]).transform(
        lambda v: pd.qcut(v, 5, labels=False, duplicates="drop") + 1
    )
    return _with(
        panel,
        ind_ipo=ind_ipo,
        rd_ipo=rd_ipo,
        firm_age_mom=mom.where(fifth == 1),
    )


INTAN_MONTHS = 60  # Intan*, CompEquIss: five years
INTAN_TRIM = (0.01, 0.99)  # Intan*: five-year returns outside the month's 1-99% dropped
INTAN_MIN_FIRMS = 3  # Intan*: enough firms to fit an intercept and two slopes


def add_intangible_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `intan_bm`, `intan_sp`, `intan_cfp`, `intan_ep` and `comp_equ_iss`
    (Daniel & Titman 2006; OSAP's code).

    r = the stock's compounded return over the last 60 months (a missing return on a
    row the stock has counts as 0, as OSAP), with a row 60 months back required.
    - `intan_*`: for x = log(fundamental / market cap), positive ratios only, the
      residual of each month's cross-sectional regression of r on x 60 months ago
      and the "tangible return" x - x 60 months ago + r (IntanBM, IntanSP, IntanCFP,
      IntanEP: book equity, revenue, net income + D&A, net income). The part of a
      five-year return that accounting growth does not explain. Fundamentals in USD
      (`_usd`). r is trimmed at each month's 1st/99th percentiles; OSAP trims over
      the whole sample at once, which mixes in later months.
    - `comp_equ_iss`: log growth of market cap over 60 months minus r, the growth
      that came from issuing shares rather than returns (CompEquIss).
    Rows up to month_end only.
    """
    panel = _sorted_by_stock(panel)
    keys = panel[["permaticker", "month_end"]]
    logs = _calendar_grid(keys.assign(v=np.log1p(panel["ret"].fillna(0))), "v")
    exists = logs.notna()
    total = logs.fillna(0).cumsum()
    five_year = np.exp(total - total.shift(INTAN_MONTHS)) - 1
    five_year = five_year.where(exists & exists.shift(INTAN_MONTHS, fill_value=False))
    r = pd.Series(_at_rows(five_year, keys, logs), index=panel.index)
    by_month = r.groupby(panel["month_end"])
    low = by_month.transform("quantile", INTAN_TRIM[0])
    high = by_month.transform("quantile", INTAN_TRIM[1])
    trimmed = r.where((r >= low) & (r <= high))

    mcap = panel["market_cap"].where(panel["market_cap"] > 0)
    numerators = {
        "intan_bm": _usd(panel, "equity"),
        "intan_sp": _usd(panel, "revenue"),
        "intan_cfp": _usd(panel, "netinc") + _usd(panel, "depamor"),
        "intan_ep": _usd(panel, "netinc"),
    }
    month = panel[["month_end"]]
    added = {}
    for name, numerator in numerators.items():
        ratio = numerator / mcap
        x = np.log(ratio.where(ratio > 0))
        x_then = _months_ago(panel, x, INTAN_MONTHS)
        tangible = x - x_then + r
        added[name] = _group_residuals(month, trimmed, [x_then, tangible], INTAN_MIN_FIRMS)
    cap_then = _months_ago(panel, mcap, INTAN_MONTHS)
    added["comp_equ_iss"] = np.log(mcap / cap_then) - r
    return _with(panel, **added)


def rank_normalise(panel: pd.DataFrame, columns: list[str], suffix: str = "_rank") -> pd.DataFrame:
    """Add `<col><suffix>`: each month, the stock's rank on `col` mapped to [-1, 1].

    Smallest -> -1, largest -> +1, evenly spaced, ties averaged; missing -> 0 (the
    cross-sectional middle), as do months with a single value. Outliers cannot
    dominate and the scale is the same in every month, so a coefficient learned
    on 2005 means the same in 2020 (Gu, Kelly & Xiu 2020). Uses only the rows of
    the same month: no look-ahead. Apply it to the rows that will be modelled
    (e.g. investable only), since the ranks depend on who else is in the month.
    """
    # Only the columns needed: grouping the whole (wide, fragmented) panel is slow.
    by_month = panel[["month_end", *columns]].groupby("month_end")
    added = {}
    for col in columns:
        rank = by_month[col].rank(method="average")
        count = by_month[col].transform("count")
        spread = (2 * (rank - 1) / (count - 1) - 1).where(count > 1)
        added[f"{col}{suffix}"] = spread.fillna(0.0)
    return _with(panel, **added)
