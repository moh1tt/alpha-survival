import numpy as np
import pandas as pd
import pytest

from alpha_survival.models import TARGET
from alpha_survival.portfolio import (
    assign_legs,
    breakeven_cost_bps,
    estimated_spread_bps,
    leg_weights,
    long_short_returns,
    max_drawdown,
    net_returns,
    summarise,
    trades,
    trading_costs,
    turnover,
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


def stocks_oos(month, names, pred, fwd_ret):
    return month_oos(month, pred, fwd_ret).assign(permaticker=names)


def test_leg_weights_sum_to_one_per_leg():
    oos = month_oos("2020-01-31", [4.0, 3.0, 2.0, 1.0], [0.1, 0.0, 0.0, 0.0], [1.0, 3.0, 1.0, 1.0])
    w = leg_weights(oos, n_groups=2, weighting="value")
    assert w.groupby("leg")["weight"].sum().tolist() == pytest.approx([1.0, 1.0])
    assert w["weight"].tolist() == pytest.approx([0.25, 0.75, 0.5, 0.5])


def test_turnover_first_month_buys_both_legs_then_nothing_if_unchanged():
    names, pred, flat = ["A", "B", "C", "D"], [4.0, 3.0, 2.0, 1.0], [0.0] * 4
    oos = pd.concat([stocks_oos(m, names, pred, flat) for m in ("2020-01-31", "2020-02-29")])
    t = turnover(leg_weights(oos, n_groups=2))
    assert t.tolist() == pytest.approx([2.0, 0.0])


def test_turnover_rebalances_drifted_weights_by_hand():
    # Long A, B at 0.5 each; A doubles, so the long leg grows to 1.5 and drifts to
    # A = 1/1.5 = 2/3, B = 1/3. Back to 0.5/0.5 trades 1/6 + 1/6 = 1/3. Short is flat.
    names, pred = ["A", "B", "C", "D"], [4.0, 3.0, 2.0, 1.0]
    jan = stocks_oos("2020-01-31", names, pred, [1.0, 0.0, 0.0, 0.0])
    feb = stocks_oos("2020-02-29", names, pred, [0.0] * 4)
    t = turnover(leg_weights(pd.concat([jan, feb]), n_groups=2))
    assert t.loc["2020-02-29"] == pytest.approx(1 / 3)


def test_turnover_counts_both_sell_and_buy_when_names_change():
    jan = stocks_oos(
        "2020-01-31", ["A", "B", "X", "Y", "C", "D"], [4.0, 3.0, 2.5, 2.5, 2.0, 1.0], [0.0] * 6
    )
    # New long names E, F; old longs A, B now in the middle and not held.
    feb = stocks_oos(
        "2020-02-29", ["E", "F", "A", "B", "C", "D"], [9.0, 8.0, 3.0, 3.0, 2.0, 1.0], [0.0] * 6
    )
    t = turnover(leg_weights(pd.concat([jan, feb]), n_groups=3))
    assert t.loc["2020-02-29"] == pytest.approx(2.0)  # sell $1 of A, B; buy $1 of E, F


def test_turnover_ignores_later_months():
    # Leakage check: what happens in March cannot change January or February.
    names, pred = ["A", "B", "C", "D"], [4.0, 3.0, 2.0, 1.0]
    jan = stocks_oos("2020-01-31", names, pred, [0.2, -0.1, 0.05, 0.0])
    feb = stocks_oos("2020-02-29", names, [1.0, 2.0, 3.0, 4.0], [0.0, 0.3, -0.2, 0.1])
    mar = stocks_oos("2020-03-31", names, pred, [0.0] * 4)
    mar_wild = stocks_oos("2020-03-31", ["X", "Y", "Z", "W"], pred, [5.0, -0.9, 0.0, 1.0])
    a = turnover(leg_weights(pd.concat([jan, feb, mar]), n_groups=2))
    b = turnover(leg_weights(pd.concat([jan, feb, mar_wild]), n_groups=2))
    assert a.iloc[:2].tolist() == pytest.approx(b.iloc[:2].tolist())


def two_months_of_trades():
    # Jan: buy A (long) and B (short) from cash. Feb: sell A, buy C; B unchanged.
    jan = stocks_oos("2020-01-31", ["A", "B"], [2.0, 1.0], [0.0, 0.0])
    feb = stocks_oos("2020-02-29", ["C", "B", "A"], [2.0, 1.0, 1.5], [0.0] * 3)
    return trades(leg_weights(pd.concat([jan, feb]), n_groups=2))


def test_trades_are_per_stock_and_sum_to_turnover():
    t = two_months_of_trades().set_index(["month_end", "permaticker"])["traded"]
    assert t.loc[("2020-01-31", "A")] == 1.0 and t.loc[("2020-01-31", "B")] == 1.0
    assert t.loc[("2020-02-29", "A")] == 1.0 and t.loc[("2020-02-29", "C")] == 1.0
    assert t.loc[("2020-02-29", "B")] == 0.0


def test_flat_trading_cost_is_cost_times_turnover():
    costs = trading_costs(two_months_of_trades(), 20)
    assert costs.tolist() == pytest.approx([2 * 0.0020, 2 * 0.0020])


def test_per_stock_trading_cost_by_hand():
    idx = pd.MultiIndex.from_tuples(
        [
            ("A", pd.Timestamp("2020-01-31")),
            ("B", pd.Timestamp("2020-01-31")),
            ("B", pd.Timestamp("2020-02-29")),
            ("C", pd.Timestamp("2020-02-29")),
        ],
        names=["permaticker", "month_end"],
    )
    cost_bps = pd.Series([10.0, 40.0, 40.0, 30.0], index=idx)
    costs = trading_costs(two_months_of_trades(), cost_bps)
    # Jan: $1 of A at 10 bp + $1 of B at 40 bp. Feb: buy C at 30 bp; selling A has
    # no Feb cost on record, so it pays that month's worst (40 bp).
    assert costs.tolist() == pytest.approx([0.0050, 0.0070])


def spread_panel(month="2020-01-31", ar_sum=(0.0005, 0.0015, 2.5e-6, 2.5e-6), caps=(1, 2, 3, 4)):
    return pd.DataFrame(
        {
            "permaticker": ["a", "b", "c", "d"],
            "month_end": pd.Timestamp(month),
            "market_cap": list(caps),
            "ar_n": 10,
            "ar_sum": list(ar_sum),
        }
    )


def test_estimated_spread_pools_each_size_group_by_hand():
    out = estimated_spread_bps(spread_panel(), n_groups=2).set_index("permaticker")
    assert out["size_group"].tolist() == [1, 1, 2, 2]
    # Small group: (0.0005 + 0.0015) / 20 pairs = 1e-4 -> S = sqrt(4e-4) = 2%, half = 100 bp.
    # Pooled, not an average of each stock's own spread (which would give ~96.5 bp).
    assert out.loc[["a", "b"], "cost_bps"].tolist() == pytest.approx([100.0, 100.0])
    # Large group: 5e-6 / 20 = 2.5e-7 -> S = 0.1%, half = 5 bp.
    assert out.loc[["c", "d"], "cost_bps"].tolist() == pytest.approx([5.0, 5.0])


def test_estimated_spread_floors_negative_and_skips_missing_caps():
    panel = spread_panel(ar_sum=(-0.001, -0.001, 1e-6, 1e-6), caps=(1, 2, 3, np.nan))
    out = estimated_spread_bps(panel, n_groups=2).set_index("permaticker")
    assert "d" not in out.index
    assert out.loc["a", "cost_bps"] == 0.0


def test_estimated_spread_uses_only_its_own_month():
    # Leakage check: February's prices cannot change January's cost.
    jan = spread_panel()
    feb = spread_panel("2020-02-29")
    feb_wild = spread_panel("2020-02-29", ar_sum=(9.0, 9.0, 9.0, 9.0), caps=(4, 3, 2, 1))
    a = estimated_spread_bps(pd.concat([jan, feb]), n_groups=2)
    b = estimated_spread_bps(pd.concat([jan, feb_wild]), n_groups=2)
    first = a["month_end"] == pd.Timestamp("2020-01-31")
    assert a[first]["cost_bps"].tolist() == b[first]["cost_bps"].tolist()


def test_net_returns_by_hand():
    months = pd.to_datetime(["2020-01-31", "2020-02-29"])
    gross = pd.Series([0.02, 0.01], index=months)
    cost = pd.Series([0.0040, 0.0010], index=months)
    # Trading cost, plus 60 bps / 12 = 5 bps borrow each month.
    net = net_returns(gross, cost, borrow_bps_pa=60)
    assert net.tolist() == pytest.approx([0.02 - 0.0040 - 0.0005, 0.01 - 0.0010 - 0.0005])
    assert net_returns(gross, cost * 0, 0).equals(gross)


def test_net_returns_charge_each_month_its_own_cost():
    # Cost for a month the long-short series lacks is ignored, not shifted.
    gross = pd.Series([0.01], index=pd.to_datetime(["2020-02-29"]))
    cost = pd.Series([0.01, 0.0], index=pd.to_datetime(["2020-01-31", "2020-02-29"]))
    assert net_returns(gross, cost, 0).iloc[0] == pytest.approx(0.01)


def test_breakeven_cost_by_hand():
    months = pd.to_datetime(["2020-01-31", "2020-02-29"])
    gross = pd.Series([0.02, 0.01], index=months)  # mean 1.5%
    to = pd.Series([2.0, 1.0], index=months)  # mean 1.5
    # (0.015 - 0.0005) / 1.5 = 0.00967 -> 96.7 bps; net mean there is exactly zero.
    b = breakeven_cost_bps(gross, to, borrow_bps_pa=60)
    assert b == pytest.approx(96.6667, abs=1e-3)
    assert net_returns(gross, b / 1e4 * to, 60).mean() == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(breakeven_cost_bps(gross, to * 0, 60))


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
