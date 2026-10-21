"""Turn monthly forecasts into decile long-short portfolios and measure their returns.

Each month t: rank stocks by forecast, buy the top decile and short the bottom
decile, hold over month t+1 (so each stock earns its `fwd_ret`). The short pays
for the long, so the strategy's return is long-leg minus short-leg return.
`long_short_returns` is GROSS of costs; `turnover` measures how much is traded
and `net_returns` charges a flat cost per dollar traded plus a short-borrow fee.
With `exit_groups`, held stocks keep their leg until they leave a wider band,
which cuts turnover (`assign_legs_banded`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_survival.metrics import newey_west_t_stat
from alpha_survival.models import TARGET

N_GROUPS = 10
WEIGHTINGS = ("equal", "value")
# Holding band: enter in the top (bottom) decile, hold until out of the top
# (bottom) 2 deciles (a buy/hold spread, Novy-Marx & Velikov 2016). Fixed in
# advance, not tuned on our results.
EXIT_GROUPS = 2


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


def assign_legs_banded(
    oos: pd.DataFrame, n_groups: int = N_GROUPS, exit_groups: int = EXIT_GROUPS
) -> pd.Series:
    """Like `assign_legs`, but a stock already held keeps its leg while it stays in
    the top (bottom) `exit_groups` of `n_groups`, e.g. enters in the top 10% and
    is only sold once it falls out of the top 20%.

    Stocks near the cut-off no longer flip in and out every month, so turnover
    falls; legs hold somewhat more names. Path-dependent, so months are processed
    in order; each month's legs use only that month's forecasts and the previous
    month's legs. A stock with no forecast leaves its leg. `exit_groups=1` gives
    `assign_legs` exactly.
    """
    if not 1 <= exit_groups <= n_groups // 2:
        raise ValueError(f"exit_groups must be in [1, {n_groups // 2}]")
    leg = pd.Series(np.nan, index=oos.index, name="leg")
    held_long: set = set()
    held_short: set = set()
    for _, month in oos.groupby("month_end", sort=True):
        rank = month["pred"].rank(method="first")
        count = rank.count()
        k = max(count // n_groups, 1)
        k_exit = max(count * exit_groups // n_groups, k)
        stock = month["permaticker"]
        is_long = (rank > count - k) | ((rank > count - k_exit) & stock.isin(held_long))
        is_short = (rank <= k) | ((rank <= k_exit) & stock.isin(held_short))
        month_leg = pd.Series(0.0, index=month.index).where(rank.notna())
        month_leg[is_long] = 1.0
        month_leg[is_short] = -1.0
        leg[month.index] = month_leg
        held_long = set(stock[is_long])
        held_short = set(stock[is_short])
    return leg


def leg_weights(
    oos: pd.DataFrame,
    n_groups: int = N_GROUPS,
    weighting: str = "equal",
    exit_groups: int | None = None,
) -> pd.DataFrame:
    """One row per held stock-month: month_end, permaticker, leg, weight, fwd_ret.

    `oos` holds month_end, permaticker, pred, fwd_ret and, for value weighting,
    `market_cap` at formation (known at month_end). Weights sum to 1 within each
    leg and month, so the portfolio is $1 long and $1 short. Stocks with a missing
    `fwd_ret` (data gap, end of sample) drop out of their leg after legs are set:
    a small bias, since a real holder would have earned something. With
    `exit_groups`, legs come from `assign_legs_banded` instead of `assign_legs`.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    if weighting == "value" and "market_cap" not in oos.columns:
        raise KeyError("value weighting needs a market_cap column")

    if exit_groups is None:
        legs = assign_legs(oos, n_groups)
    else:
        legs = assign_legs_banded(oos, n_groups, exit_groups)
    rows = oos.assign(leg=legs)
    rows = rows[rows["leg"].isin([1, -1])].dropna(subset=[TARGET])
    if weighting == "value":
        rows = rows[rows["market_cap"] > 0]
        raw = rows["market_cap"]
    else:
        raw = pd.Series(1.0, index=rows.index)
    weight = raw / raw.groupby([rows["month_end"], rows["leg"]]).transform("sum")
    cols = [c for c in ("month_end", "permaticker") if c in rows.columns]
    return rows[cols].assign(leg=rows["leg"], weight=weight, **{TARGET: rows[TARGET]})


def long_short_returns(
    oos: pd.DataFrame,
    n_groups: int = N_GROUPS,
    weighting: str = "equal",
    exit_groups: int | None = None,
) -> pd.DataFrame:
    """Monthly returns of the top-minus-bottom portfolio (see `assign_legs`).

    Inputs as for `leg_weights`. Columns: long, short, long_short (= long - short),
    n_long, n_short; indexed by formation month. Months missing either leg are
    left out.
    """
    rows = leg_weights(oos, n_groups, weighting, exit_groups)
    rows = rows.assign(w_ret=rows["weight"] * rows[TARGET])
    legs = rows.groupby(["month_end", "leg"]).agg(w_ret=("w_ret", "sum"), n=("weight", "size"))
    leg_ret = legs["w_ret"].unstack("leg")
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


def trades(weights: pd.DataFrame) -> pd.DataFrame:
    """Dollars traded per stock at each formation date: month_end, permaticker, traded.

    `weights` is the output of `leg_weights`. Over a month each holding drifts
    with its own return: weight w_i becomes w_i (1 + r_i) / (1 + R_leg). At the
    next formation date we trade from those drifted weights to the new targets,
    so traded_i,t = |target_i,t - drifted_i,t| (short weights negative). A stock
    leaving the portfolio appears with its sale; one entering, with its purchase.

    The first month buys both legs from cash. Each leg is reset to $1 each month:
    the cash from that reset is not counted as trading.
    """
    w = weights.assign(signed=weights["leg"] * weights["weight"])
    leg_ret = (w["weight"] * w[TARGET]).groupby([w["month_end"], w["leg"]]).transform("sum")
    w = w.assign(drifted=w["signed"] * (1 + w[TARGET]) / (1 + leg_ret))

    months = np.sort(w["month_end"].unique())
    next_month = dict(zip(months[:-1], months[1:], strict=True))
    held = w.assign(month_end=w["month_end"].map(next_month)).dropna(subset=["month_end"])
    out = w[["month_end", "permaticker", "signed"]].merge(
        held[["month_end", "permaticker", "drifted"]],
        on=["month_end", "permaticker"],
        how="outer",
    )
    traded = (out["signed"].fillna(0) - out["drifted"].fillna(0)).abs()
    return out[["month_end", "permaticker"]].assign(traded=traded)


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Dollars traded each month per $1 long (and $1 short), both legs together.

    Sum of `trades` per month: 2 when both legs are bought from cash or fully
    replaced; 0 for identical names and weights through flat returns.
    """
    t = trades(weights)
    return t.groupby("month_end")["traded"].sum().rename("turnover")


def estimated_spread_bps(panel: pd.DataFrame, n_groups: int = N_GROUPS) -> pd.DataFrame:
    """Cost per dollar traded (half the bid-ask spread, in bps) for each stock-month.

    Abdi & Ranaldo (2017) estimates from a single stock are mostly noise, so each
    month stocks are sorted into `n_groups` size groups by `market_cap` and every
    stock is charged its group's spread, pooled over all the group's day pairs:
    S = sqrt(max(4 x sum(ar_sum) / sum(ar_n), 0)). Half of S is paid per trade
    (buy at the ask or sell at the bid, vs. the mid). Uses only days up to each
    month_end, the date the trade is placed.

    `panel`: permaticker, month_end, market_cap, ar_n, ar_sum. Returns permaticker,
    month_end, size_group (1 = smallest) and cost_bps; rows without a positive
    market_cap are left out.
    """
    rows = panel[["permaticker", "month_end", "market_cap", "ar_n", "ar_sum"]]
    rows = rows[rows["market_cap"] > 0]
    pct = rows.groupby("month_end")["market_cap"].rank(pct=True)
    group = np.ceil(pct * n_groups).clip(1, n_groups).astype(int)
    pooled = rows.groupby([rows["month_end"], group])[["ar_n", "ar_sum"]].transform("sum")
    spread = np.sqrt((4 * pooled["ar_sum"] / pooled["ar_n"]).clip(lower=0))
    return rows[["permaticker", "month_end"]].assign(size_group=group, cost_bps=spread / 2 * 1e4)


def trading_costs(trade_rows: pd.DataFrame, cost_bps: float | pd.Series) -> pd.Series:
    """Monthly trading cost as a fraction of $1 per leg: sum_i traded_i x cost_i.

    `trade_rows` is the output of `trades`. `cost_bps` (basis points, 1 bp = 0.01%,
    per dollar traded) is one flat number, or a Series indexed by (permaticker,
    month_end) giving each stock's cost at that date. A trade with no cost on
    record (e.g. selling a stock that has left the universe) pays the highest
    cost seen that month: such exits are usually illiquid, so this errs high.
    """
    if isinstance(cost_bps, pd.Series):
        key = pd.MultiIndex.from_frame(trade_rows[["permaticker", "month_end"]])
        cost = pd.Series(cost_bps.reindex(key).to_numpy(), index=trade_rows.index)
        worst = cost.groupby(trade_rows["month_end"]).transform("max")
        cost = cost.fillna(worst)
    else:
        cost = cost_bps
    paid = trade_rows["traded"] * cost / 1e4
    return paid.groupby(trade_rows["month_end"]).sum().rename("trading_cost")


def net_returns(long_short: pd.Series, trading_cost: pd.Series, borrow_bps_pa: float) -> pd.Series:
    """Long-short returns after trading and short-borrow costs.

    Each month pays its `trading_cost` (from `trading_costs`) at formation plus
    1/12 of the annual fee for borrowing the $1 of shorted stock. Both series are
    indexed by formation month: the trade at month_end t buys the return earned
    over t+1.
    """
    return long_short - trading_cost.reindex(long_short.index) - borrow_bps_pa / 1e4 / 12


def breakeven_cost_bps(long_short: pd.Series, turnover: pd.Series, borrow_bps_pa: float) -> float:
    """Trading cost (bps per dollar traded) at which the average net return is zero.

    Above this, the strategy loses money on average. NaN if it never trades;
    negative if the borrow fee alone already wipes out the gross return.
    """
    avg_turnover = float(turnover.reindex(long_short.index).mean())
    if avg_turnover == 0:
        return float("nan")
    return (float(long_short.mean()) - borrow_bps_pa / 1e4 / 12) / avg_turnover * 1e4


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
