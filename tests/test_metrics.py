import numpy as np
import pandas as pd
import pytest

from alpha_survival.metrics import ic_summary, oos_r2, rank_ic
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
    with pytest.raises(ValueError):
        ic_summary(pd.Series([0.1]))


def test_empty_input_raises():
    with pytest.raises(ValueError, match="no rows"):
        oos_r2(make_oos([np.nan], [0.1]))
