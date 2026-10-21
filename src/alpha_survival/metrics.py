"""Score out-of-sample forecasts: are they close (R^2), and do they order stocks (rank IC)?

All functions take the table from `walk_forward_predict`: one row per stock-month
with `month_end`, `fwd_ret` (realised) and `pred` (forecast). Rows where either is
missing are skipped.
"""

from __future__ import annotations

from statistics import NormalDist

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


def newey_west_lags(n: int) -> int:
    """Rule-of-thumb lag count floor(4 * (n / 100) ** (2/9)) (Newey & West 1994)."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def newey_west_t_stat(x: pd.Series, lags: int | None = None) -> float:
    """t-stat of the mean of `x`, robust to autocorrelation (Newey & West 1987).

    Same mean as the plain t-stat, but the variance adds the autocovariances up to
    `lags` months, weighted 1 - j/(lags+1) (Bartlett) so it cannot go negative.
    A persistent signal's ICs come in streaks: positively autocorrelated months
    carry less independent evidence, and this t-stat shrinks accordingly.
    """
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 observations")
    lags = newey_west_lags(n) if lags is None else lags
    if not 0 <= lags < n:
        raise ValueError(f"lags must be in [0, {n - 1}]")

    e = x - x.mean()
    variance = e @ e / n
    for j in range(1, lags + 1):
        variance += 2 * (1 - j / (lags + 1)) * (e[j:] @ e[:-j]) / n
    return float(x.mean() / np.sqrt(variance / n)) if variance > 0 else float("nan")


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Mean, std, t-stats and hit rate of a monthly IC series.

    `t_stat` = mean / (std / sqrt(n)) treats months as independent; `nw_t_stat`
    allows for autocorrelation (Newey-West, `nw_lags` lags) and is the one to trust.
    """
    ic = ic.dropna()
    if len(ic) < 2:
        raise ValueError("need at least 2 months of IC")
    mean, std, n = float(ic.mean()), float(ic.std()), len(ic)
    return {
        "mean_ic": mean,
        "std_ic": std,
        "t_stat": mean / (std / np.sqrt(n)) if std > 0 else float("nan"),
        "nw_t_stat": newey_west_t_stat(ic),
        "nw_lags": newey_west_lags(n),
        "ic_autocorr_1": float(ic.autocorr(1)) if n > 2 else float("nan"),
        "hit_rate": float((ic > 0).mean()),
        "n_months": n,
    }


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """Probability that the true Sharpe ratio beats `sr_benchmark` (Bailey & Lopez de Prado 2012).

    Phi((SR - SR*) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2)).
    SR and SR* are per period (monthly here), NOT annualised. The estimated SR is
    noisier when returns have negative skew or fat tails (kurt > 3), so the same
    SR earns a lower probability. Normal returns reduce the denominator to
    sqrt(1 + SR^2 / 2). Moments are population (ddof=0), as in the paper.
    """
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    t = len(x)
    if t < 3:
        raise ValueError("need at least 3 returns")
    e = x - x.mean()
    std = np.sqrt(e @ e / t)
    if std == 0:
        raise ValueError("returns have zero volatility")
    sr = x.mean() / std
    skew = np.mean(e**3) / std**3
    kurt = np.mean(e**4) / std**4
    # kurt >= skew^2 + 1 for any sample, so this is >= (1 - skew * sr / 2)^2 >= 0.
    variance = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if variance <= 0:
        return 1.0 if sr > sr_benchmark else 0.0
    return NormalDist().cdf((sr - sr_benchmark) * np.sqrt(t - 1) / np.sqrt(variance))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected best Sharpe among `n_trials` strategies with zero true skill.

    sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e))), g = Euler-Mascheroni
    (Bailey & Lopez de Prado 2014). V is the variance of the estimated Sharpes across
    trials, per period like the Sharpes. Luck alone gets this far, so it is the bar a
    selected strategy must clear. One trial involves no selection: the bar is 0.
    """
    if n_trials < 1:
        raise ValueError("need at least 1 trial")
    if sharpe_variance < 0:
        raise ValueError("sharpe_variance cannot be negative")
    if n_trials == 1:
        return 0.0
    inv = NormalDist().inv_cdf
    return float(
        np.sqrt(sharpe_variance)
        * (
            (1 - np.euler_gamma) * inv(1 - 1 / n_trials)
            + np.euler_gamma * inv(1 - 1 / (n_trials * np.e))
        )
    )


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, sharpe_variance: float) -> float:
    """Probability the true Sharpe beats the best of `n_trials` skill-less tries.

    The probabilistic Sharpe ratio with the benchmark raised to `expected_max_sharpe`
    (Bailey & Lopez de Prado 2014).
    > 0.95 means the result survives the selection. `n_trials` should count every
    variant tried, not just the ones reported; correlated variants may be counted as
    fewer independent trials.
    """
    return probabilistic_sharpe_ratio(returns, expected_max_sharpe(n_trials, sharpe_variance))
