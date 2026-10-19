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


def add_momentum(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `mom_12_1`: the compounded return over months t-11 .. t-1.

    NaN unless all 11 monthly returns exist and come from consecutive months, so a
    gap in the price history never turns into a longer window passed off as a year.
    """
    keys = ["permaticker", "month_end"]
    if panel.duplicated(keys).any():
        raise ValueError("panel has duplicate (permaticker, month_end) rows")

    panel = panel.sort_values(keys).reset_index(drop=True)
    by_stock = panel.groupby("permaticker")

    growth = pd.Series(1.0, index=panel.index)
    for lag in range(1, MOMENTUM_MONTHS + 1):
        growth = growth * (1 + by_stock["ret"].shift(lag))  # NaN anywhere -> NaN

    # The oldest row in the window must be exactly 11 months back; otherwise rows
    # are missing and the 11 rows span more than 11 months.
    oldest = by_stock["month_end"].shift(MOMENTUM_MONTHS).dt.to_period("M")
    consecutive = oldest == panel["month_end"].dt.to_period("M") - MOMENTUM_MONTHS

    return panel.assign(mom_12_1=(growth - 1).where(consecutive))


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
