"""Turn the monthly panel into predictive features (signals).

Every feature on a row must be computable from information known at that row's
`month_end`: earlier rows of the same stock, or the row's own known columns.
Never `fwd_ret` (the target) and never later rows.
"""

from __future__ import annotations

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
