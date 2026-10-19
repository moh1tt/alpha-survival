import numpy as np
import pandas as pd
import pytest

from alpha_survival.models import TARGET
from alpha_survival.portfolio import (
    assign_legs,
    long_short_returns,
    max_drawdown,
    summarise,
)


def month_oos(month, pred, fwd_ret, market_cap=None):
    rows = pd.DataFrame({"month_end": pd.Timestamp(month), "pred": pred, TARGET: fwd_ret})
    if market_cap is not None:
        rows["market_cap"] = market_cap
    return rows


def test_legs_hold_the_top_and_bottom_tenth_by_forecast():
    oos = month_oos("2020-01-31", np.arange(20.0)[::-1], np.zeros(20))
    legs = assign_legs(oos)
    # Highest forecasts (rows 0-1) long, lowest (rows 18-19) short, 2 stocks each.
    assert legs.iloc[:2].tolist() == [1.0, 1.0] and legs.iloc[18:].tolist() == [-1.0, -1.0]
    assert (legs.iloc[2:18] == 0).all()


def test_legs_are_symmetric_when_count_is_not_a_multiple_of_ten():
    legs = assign_legs(month_oos("2020-01-31", np.arange(15.0), np.zeros(15)))
    assert (legs == 1).sum() == (legs == -1).sum() == 1


def test_legs_skip_missing_forecasts_and_are_per_month():
    a = month_oos("2020-01-31", [1.0, np.nan, 3.0], [0.0] * 3)
    b = month_oos("2020-02-29", [100.0, 200.0], [0.0] * 2)
    legs = assign_legs(pd.concat([a, b], ignore_index=True), n_groups=2)
    assert np.isnan(legs.iloc[1])  # no forecast, no leg
    assert legs.iloc[0] == -1 and legs.iloc[2] == 1
    assert legs.iloc[3:].tolist() == [-1.0, 1.0]  # February ranked on its own


def test_equal_weighted_long_short_by_hand():
    # 4 stocks, 2 groups: long the top two forecasts, short the bottom two.
    oos = month_oos("2020-01-31", [4.0, 3.0, 2.0, 1.0], [0.10, 0.02, -0.04, 0.00])
    out = long_short_returns(oos, n_groups=2)
    row = out.loc[pd.Timestamp("2020-01-31")]
    assert row["long"] == pytest.approx(0.06)
    assert row["short"] == pytest.approx(-0.02)
    assert row["long_short"] == pytest.approx(0.08)
    assert row["n_long"] == 2 and row["n_short"] == 2


def test_value_weighted_long_short_by_hand():
    oos = month_oos(
        "2020-01-31", [4.0, 3.0, 2.0, 1.0], [0.10, 0.02, -0.04, 0.00], [1.0, 3.0, 1.0, 1.0]
    )
    row = long_short_returns(oos, n_groups=2, weighting="value").iloc[0]
    assert row["long"] == pytest.approx((0.10 * 1 + 0.02 * 3) / 4)
    assert row["short"] == pytest.approx(-0.02)


def test_missing_outcome_keeps_its_slot_but_leaves_the_leg():
    # Stock with the top forecast has no next return: groups still form on forecasts.
    oos = month_oos("2020-01-31", [4.0, 3.0, 2.0, 1.0], [np.nan, 0.02, -0.04, 0.00])
    row = long_short_returns(oos, n_groups=2).iloc[0]
    assert row["n_long"] == 1 and row["long"] == pytest.approx(0.02)
    assert row["n_short"] == 2  # stock 2 did not move down into the long leg


def test_months_are_independent():
    # Leakage check: another month's forecasts and outcomes cannot move this month.
    jan = month_oos("2020-01-31", [4.0, 3.0, 2.0, 1.0], [0.10, 0.02, -0.04, 0.00])
    feb = month_oos("2020-02-29", [1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5])
    feb_wild = feb.assign(pred=[9.0, -9.0, 0.0, 1.0], **{TARGET: [-0.9, 3.0, 0.0, 0.0]})
    a = long_short_returns(pd.concat([jan, feb]), n_groups=2)
    b = long_short_returns(pd.concat([jan, feb_wild]), n_groups=2)
    assert a.loc["2020-01-31"].equals(b.loc["2020-01-31"])


def test_bad_inputs_raise():
    oos = month_oos("2020-01-31", [1.0, 2.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        long_short_returns(oos, weighting="cap")
    with pytest.raises(KeyError, match="market_cap"):
        long_short_returns(oos, weighting="value")


def test_max_drawdown_by_hand():
    # $1 -> 1.10 -> 0.88 (-20% from the 1.10 peak) -> 0.968 -> 1.0648
    assert max_drawdown(pd.Series([0.10, -0.20, 0.10, 0.10])) == pytest.approx(-0.20)
    assert max_drawdown(pd.Series([-0.5, 0.1])) == pytest.approx(-0.5)  # from the start
    assert max_drawdown(pd.Series([0.01, 0.02])) == 0.0


def test_summarise_by_hand():
    r = pd.Series([0.02, -0.01, 0.03, 0.00])
    s = summarise(r)
    assert s["ann_return"] == pytest.approx(12 * 0.01)
    assert s["sharpe"] == pytest.approx(np.sqrt(12) * 0.01 / r.std())
    assert s["hit_rate"] == 0.5 and s["n_months"] == 4
    with pytest.raises(ValueError):
        summarise(pd.Series([0.1]))
