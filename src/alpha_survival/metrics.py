"""Score out-of-sample forecasts: are they close (R^2), and do they order stocks (rank IC)?

All functions take the table from `walk_forward_predict`: one row per stock-month
with `month_end`, `fwd_ret` (realised) and `pred` (forecast). Rows where either is
missing are skipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_survival.models import TARGET


def _complete(oos: pd.DataFrame) -> pd.DataFrame:
    rows = oos.dropna(subset=[TARGET, "pred"])
    if rows.empty:
        raise ValueError("no rows with both a forecast and a realised return")
    return rows


def oos_r2(oos: pd.DataFrame) -> float:
    """Out-of-sample R^2 against a zero forecast, pooled over all stock-months.

    1 - sum((r - pred)^2) / sum(r^2) (Gu, Kelly & Xiu 2020). The benchmark is 0, not
    the historical mean: they argue historical mean returns of individual stocks are
    too noisy to be a fair bar. Zero is also fixed, so every model faces the same
    bar in every fold. > 0 beats 'predict 0'.
    """
    rows = _complete(oos)
    r, pred = rows[TARGET].to_numpy(), rows["pred"].to_numpy()
    return float(1 - np.sum((r - pred) ** 2) / np.sum(r**2))


def rank_ic(oos: pd.DataFrame, min_stocks: int = 10) -> pd.Series:
    """Monthly rank IC: Spearman correlation of `pred` with `fwd_ret` across stocks.

    Spearman = Pearson correlation of the ranks, so only the ordering counts and
    outliers cannot dominate. Months with fewer than `min_stocks` scored stocks,
    or where every forecast is the same (nothing to order), are dropped.
    """
    rows = _complete(oos)
    ranks = rows[["month_end"]].assign(
        pred=rows.groupby("month_end")["pred"].rank(),
        actual=rows.groupby("month_end")[TARGET].rank(),
    )
    by_month = ranks.groupby("month_end")
    varies = (by_month["pred"].nunique() > 1) & (by_month["actual"].nunique() > 1)
    enough = by_month.size() >= min_stocks
    months = (varies & enough)[lambda ok: ok].index
    ic = {m: by_month.get_group(m)["pred"].corr(by_month.get_group(m)["actual"]) for m in months}
    return pd.Series(ic, dtype=float, name="rank_ic").rename_axis("month_end")


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Mean, std, t-stat (mean / (std / sqrt(n))) and hit rate of a monthly IC series.

    The t-stat treats months as independent; overlapping or autocorrelated ICs would
    need a Newey-West correction.
    """
    ic = ic.dropna()
    if len(ic) < 2:
        raise ValueError("need at least 2 months of IC")
    mean, std, n = float(ic.mean()), float(ic.std()), len(ic)
    return {
        "mean_ic": mean,
        "std_ic": std,
        "t_stat": mean / (std / np.sqrt(n)) if std > 0 else float("nan"),
        "hit_rate": float((ic > 0).mean()),
        "n_months": n,
    }
