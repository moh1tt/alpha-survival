"""Turn monthly forecasts into decile long-short portfolios and measure their returns.

Each month t: rank stocks by forecast, buy the top decile and short the bottom
decile, hold over month t+1 (so each stock earns its `fwd_ret`). The short pays
for the long, so the strategy's return is long-leg minus short-leg return.
GROSS of trading and borrowing costs; those come later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_survival.metrics import newey_west_t_stat
from alpha_survival.models import TARGET

N_GROUPS = 10
WEIGHTINGS = ("equal", "value")


def assign_legs(oos: pd.DataFrame, n_groups: int = N_GROUPS) -> pd.Series:
    """+1 (long), -1 (short) or 0 (neither) per row; NaN where there is no forecast.

    Each month the k = count // n_groups highest forecasts go long and the k lowest
    go short, so both legs always hold the same number of stocks. Legs depend only
    on forecasts, never on outcomes: a stock whose next return turns out missing
    still takes its slot. Ties are broken by row order.
    """
    by_month = oos.groupby("month_end")["pred"]
    rank = by_month.rank(method="first")
    count = by_month.transform("count")
    k = (count // n_groups).clip(lower=1)
    leg = pd.Series(0.0, index=oos.index).where(rank.notna())
    leg[rank <= k] = -1.0
    leg[rank > count - k] = 1.0
    return leg.rename("leg")


def long_short_returns(
    oos: pd.DataFrame, n_groups: int = N_GROUPS, weighting: str = "equal"
) -> pd.DataFrame:
    """Monthly returns of the top-minus-bottom portfolio (see `assign_legs`).

    `oos` holds month_end, pred, fwd_ret and, for value weighting, `market_cap`
    at formation (known at month_end). Stocks with a missing `fwd_ret` (data gap,
    end of sample) drop out of their leg after legs are set: a small bias,
    since a real holder would have earned something.

    Columns: long, short, long_short (= long - short), n_long, n_short; indexed by
    formation month. Months missing either leg are left out.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    if weighting == "value" and "market_cap" not in oos.columns:
        raise KeyError("value weighting needs a market_cap column")

    rows = oos.assign(leg=assign_legs(oos, n_groups))
    rows = rows[rows["leg"].isin([1, -1])].dropna(subset=[TARGET])
    if weighting == "value":
        rows = rows[rows["market_cap"] > 0]
        weight = rows["market_cap"]
    else:
        weight = pd.Series(1.0, index=rows.index)
    rows = rows.assign(w=weight, w_ret=weight * rows[TARGET])

    legs = rows.groupby(["month_end", "leg"]).agg(
        w=("w", "sum"), w_ret=("w_ret", "sum"), n=("w", "size")
    )
    leg_ret = (legs["w_ret"] / legs["w"]).unstack("leg")
    leg_n = legs["n"].unstack("leg")
    out = pd.DataFrame(
        {
            "long": leg_ret.get(1.0),
            "short": leg_ret.get(-1.0),
            "n_long": leg_n.get(1.0),
            "n_short": leg_n.get(-1.0),
        }
    ).dropna(subset=["long", "short"])
    out["long_short"] = out["long"] - out["short"]
    return out[["long", "short", "long_short", "n_long", "n_short"]]


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough fall of the compounded value of $1 (e.g. -0.35 = -35%)."""
    wealth = np.concatenate([[1.0], (1 + returns.dropna()).cumprod().to_numpy()])
    return float((wealth / np.maximum.accumulate(wealth) - 1).min())


def summarise(returns: pd.Series) -> dict[str, float]:
    """Annualised mean, volatility and Sharpe ratio, Newey-West t, drawdown, hit rate.

    Sharpe = mean / std * sqrt(12) of monthly returns. A long-short return needs
    no risk-free rate subtracted: the short funds the long, so it is already an
    excess return.
    """
    r = returns.dropna()
    if len(r) < 2:
        raise ValueError("need at least 2 months of returns")
    mean, std = float(r.mean()), float(r.std())
    return {
        "ann_return": 12 * mean,
        "ann_vol": np.sqrt(12) * std,
        "sharpe": np.sqrt(12) * mean / std if std > 0 else float("nan"),
        "nw_t_stat": newey_west_t_stat(r),
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()),
        "n_months": len(r),
    }
