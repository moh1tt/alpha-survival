from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from alpha_survival.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    ic_summary,
    newey_west_lags,
    newey_west_t_stat,
    oos_r2,
    probabilistic_sharpe_ratio,
    rank_ic,
)
from alpha_survival.models import TARGET


def make_oos(actual, pred, month="2020-01-31") -> pd.DataFrame:
    return pd.DataFrame({"month_end": pd.Timestamp(month), TARGET: actual, "pred": pred})


def test_r2_reference_points():
    actual = [0.02, -0.01, 0.03]
    assert oos_r2(make_oos(actual, [0.0, 0.0, 0.0])) == 0.0  # the zero benchmark itself
    assert oos_r2(make_oos(actual, actual)) == 1.0  # perfect
    assert oos_r2(make_oos(actual, [-0.02, 0.01, -0.03])) < 0  # worse than zero


def test_r2_by_hand_and_skips_missing():
    oos = make_oos([0.1, -0.1, np.nan, 0.2], [0.05, 0.0, 0.3, np.nan])
    # Only the first two rows count: errors 0.05, -0.1; benchmark 0.1, -0.1.
    expected = 1 - (0.05**2 + 0.1**2) / (0.1**2 + 0.1**2)
    assert oos_r2(oos) == pytest.approx(expected)


def test_rank_ic_only_cares_about_order():
    actual = list(np.linspace(-0.1, 0.1, 12))
    perfect = make_oos(actual, [a**3 * 1000 for a in actual])  # same order, different scale
    reversed_ = make_oos(actual, actual[::-1])
    assert rank_ic(perfect).iloc[0] == pytest.approx(1.0)
    assert rank_ic(reversed_).iloc[0] == pytest.approx(-1.0)


def test_rank_ic_is_computed_within_each_month():
    # Month A: perfect order. Month B: reversed. Pooling the months would mix them.
    a = make_oos(list(range(10)), list(range(10)), "2020-01-31")
    b = make_oos(list(range(10)), list(range(10))[::-1], "2020-02-29")
    ic = rank_ic(pd.concat([a, b]))
    assert ic.loc["2020-01-31"] == pytest.approx(1.0)
    assert ic.loc["2020-02-29"] == pytest.approx(-1.0)


def test_rank_ic_drops_thin_and_constant_months():
    thin = make_oos([0.1, 0.2, 0.3], [1, 2, 3], "2020-01-31")
    flat = make_oos(list(range(10)), [0.5] * 10, "2020-02-29")  # e.g. HistoricalMean
    ok = make_oos(list(range(10)), list(range(10)), "2020-03-31")
    ic = rank_ic(pd.concat([thin, flat, ok]), min_stocks=10)
    assert list(ic.index) == [pd.Timestamp("2020-03-31")]
    # Every month dropped -> an empty Series, not an error or a DataFrame.
    none_left = rank_ic(pd.concat([thin, flat]), min_stocks=10)
    assert isinstance(none_left, pd.Series) and none_left.empty


def test_rank_ic_matches_pandas_spearman():
    rng = np.random.default_rng(0)
    actual, pred = rng.normal(size=50), rng.normal(size=50)
    expected = pd.Series(pred).rank().corr(pd.Series(actual).rank())
    assert rank_ic(make_oos(actual, pred)).iloc[0] == pytest.approx(expected)


def test_ic_summary():
    ic = pd.Series([0.05, 0.03, -0.01, 0.05])
    s = ic_summary(ic)
    assert s["mean_ic"] == pytest.approx(0.03)
    assert s["t_stat"] == pytest.approx(0.03 / (ic.std() / 2))
    assert s["hit_rate"] == 0.75 and s["n_months"] == 4
    assert s["nw_lags"] == 1 and np.isfinite(s["nw_t_stat"])
    with pytest.raises(ValueError):
        ic_summary(pd.Series([0.1]))


def test_empty_input_raises():
    with pytest.raises(ValueError, match="no rows"):
        oos_r2(make_oos([np.nan], [0.1]))


def test_newey_west_lag_rule():
    assert newey_west_lags(285) == 5
    assert newey_west_lags(100) == 4


def test_newey_west_with_zero_lags_is_the_plain_t_stat():
    x = pd.Series([0.05, 0.03, -0.01, 0.05, 0.02])
    # Lag 0 uses the population variance (divide by n), not the sample one.
    assert newey_west_t_stat(x, lags=0) == pytest.approx(x.mean() / (x.std(ddof=0) / np.sqrt(5)))


def test_newey_west_by_hand_with_one_lag():
    x = np.array([1.0, 2.0, 3.0, 2.0])  # mean 2, deviations -1, 0, 1, 0
    gamma0 = (1 + 0 + 1 + 0) / 4
    gamma1 = (0 * -1 + 1 * 0 + 0 * 1) / 4
    variance = gamma0 + 2 * 0.5 * gamma1
    assert newey_west_t_stat(pd.Series(x), lags=1) == pytest.approx(2 / np.sqrt(variance / 4))


def test_newey_west_shrinks_t_for_persistent_series():
    # AR(1) with strong persistence: streaky, like the IC of a slow-moving signal.
    rng = np.random.default_rng(0)
    x = np.empty(300)
    x[0] = 0.0
    for t in range(1, 300):
        x[t] = 0.8 * x[t - 1] + rng.normal(0, 0.02)
    x = pd.Series(x + 0.01)
    plain = x.mean() / (x.std() / np.sqrt(len(x)))
    assert newey_west_t_stat(x) < 0.6 * plain


def test_newey_west_bad_lags_raise():
    with pytest.raises(ValueError):
        newey_west_t_stat(pd.Series([0.1, 0.2, 0.3]), lags=3)


def test_newey_west_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    x = rng.normal(0.02, 0.1, 120)
    fit = sm.OLS(x, np.ones_like(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": False}
    )
    assert newey_west_t_stat(pd.Series(x), lags=4) == pytest.approx(fit.tvalues[0])


def standardised(x, mean=0.02, std=0.1) -> pd.Series:
    """Rescale `x` to an exact (population) mean and std, so only its shape differs."""
    x = np.asarray(x, dtype=float)
    return pd.Series(mean + std * (x - x.mean()) / x.std())


def symmetric(x) -> np.ndarray:
    """`x` plus its mirror image about its mean: skew is exactly 0."""
    x = np.asarray(x, dtype=float)
    return np.concatenate([x, 2 * x.mean() - x])


def test_psr_by_hand():
    # mean 0.02, std 0.01 -> SR 2; symmetric two-point: skew 0, kurt 1 -> denominator 1.
    returns = pd.Series([0.03, 0.01, 0.03, 0.01])
    assert probabilistic_sharpe_ratio(returns) == pytest.approx(NormalDist().cdf(2 * np.sqrt(3)))


def test_psr_is_half_at_the_benchmark():
    returns = standardised(np.random.default_rng(0).normal(size=200))
    assert probabilistic_sharpe_ratio(returns, sr_benchmark=0.2) == pytest.approx(0.5)


def test_psr_normal_returns_match_simple_formula():
    # Skew 0 and kurt ~3, so the denominator is ~sqrt(1 + SR^2 / 2).
    returns = standardised(symmetric(np.random.default_rng(1).normal(size=10_000)))
    sr, sr_star, t = 0.2, 0.19, len(returns)
    simple = NormalDist().cdf((sr - sr_star) * np.sqrt(t - 1) / np.sqrt(1 + sr**2 / 2))
    assert probabilistic_sharpe_ratio(returns, sr_star) == pytest.approx(simple, abs=1e-3)


def test_psr_penalises_negative_skew():
    right_skewed = np.random.default_rng(2).lognormal(0, 0.8, 120)
    pos = standardised(right_skewed)
    neg = standardised(-right_skewed)  # mirror: same SR and kurtosis, skew flipped
    assert probabilistic_sharpe_ratio(neg) < probabilistic_sharpe_ratio(pos)


def test_psr_penalises_fat_tails():
    rng = np.random.default_rng(3)
    thin = standardised(symmetric(rng.normal(size=60)))
    fat = standardised(symmetric(rng.standard_t(3, size=60)))  # same SR, skew 0, kurt > 3
    assert probabilistic_sharpe_ratio(fat) < probabilistic_sharpe_ratio(thin)


def test_psr_bad_input_raises():
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(pd.Series([0.01, 0.02]))
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(pd.Series([0.01, 0.01, 0.01, np.nan]))


def test_expected_max_sharpe_shape():
    assert expected_max_sharpe(1, 0.01) == 0.0  # no selection
    assert expected_max_sharpe(16, 0.0) == 0.0  # no dispersion, no luck
    assert expected_max_sharpe(10, 0.01) < expected_max_sharpe(100, 0.01)  # more tries
    assert expected_max_sharpe(16, 0.04) == pytest.approx(2 * expected_max_sharpe(16, 0.01))


def test_expected_max_sharpe_matches_simulation():
    # 100 skill-less strategies, 120 months each: the best Sharpe should average out
    # close to the formula, with V the actual spread of the Sharpe estimates.
    rng = np.random.default_rng(4)
    returns = rng.normal(0, 0.05, size=(1_000, 100, 120))
    sharpes = returns.mean(axis=2) / returns.std(axis=2)
    expected = expected_max_sharpe(100, sharpes.var())
    assert sharpes.max(axis=1).mean() == pytest.approx(expected, rel=0.05)


def test_expected_max_sharpe_bad_input_raises():
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 0.01)
    with pytest.raises(ValueError):
        expected_max_sharpe(10, -0.01)


def test_dsr_is_psr_against_the_luck_bar():
    returns = standardised(np.random.default_rng(5).normal(size=240))  # SR 0.2
    assert deflated_sharpe_ratio(returns, 1, 0.01) == probabilistic_sharpe_ratio(returns)
    bar = expected_max_sharpe(16, 0.01)
    assert deflated_sharpe_ratio(returns, 16, 0.01) == probabilistic_sharpe_ratio(returns, bar)
    assert deflated_sharpe_ratio(returns, 16, 0.01) < probabilistic_sharpe_ratio(returns)
