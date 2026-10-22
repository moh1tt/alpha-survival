import numpy as np
import pandas as pd
import pytest

from alpha_survival.models import TARGET
from alpha_survival.portfolio import (
    assign_legs,
    assign_legs_banded,
    breakeven_cost_bps,
    estimated_spread_bps,
    leg_weights,
    long_short_returns,
    max_drawdown,
    net_returns,
    no_view_months,
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


TEN = [f"S{i}" for i in range(10)]


def ten_stocks(month, order):
    """10 stocks; `order` lists names from highest forecast to lowest."""
    pred = {name: 10.0 - i for i, name in enumerate(order)}
    return stocks_oos(month, TEN, [pred[n] for n in TEN], [0.0] * 10)


def banded_legs(months):
    oos = pd.concat(months, ignore_index=True)
    legs = oos.assign(leg=assign_legs_banded(oos, n_groups=10, exit_groups=2))
    return {
        m: (set(g.loc[g["leg"] == 1, "permaticker"]), set(g.loc[g["leg"] == -1, "permaticker"]))
        for m, g in legs.groupby("month_end")
    }


def test_banded_legs_hold_until_the_stock_leaves_the_band():
    # 10 stocks: enter in the top/bottom 1, hold while in the top/bottom 2.
    jan = ten_stocks("2020-01-31", TEN)  # S0 best, S9 worst
    feb = ten_stocks("2020-02-29", ["S1", "S0", *TEN[2:8], "S9", "S8"])  # S0, S9 slip a place
    mar = ten_stocks("2020-03-31", ["S1", "S2", "S0", *TEN[3:8], "S9", "S8"])  # S0 drops to 3rd
    legs = banded_legs([jan, feb, mar])
    assert legs[pd.Timestamp("2020-01-31")] == ({"S0"}, {"S9"})
    assert legs[pd.Timestamp("2020-02-29")] == ({"S1", "S0"}, {"S8", "S9"})  # held in the band
    assert legs[pd.Timestamp("2020-03-31")] == ({"S1"}, {"S8", "S9"})  # S0 out of the band


def test_banded_legs_new_stocks_enter_only_at_the_top():
    jan = ten_stocks("2020-01-31", TEN)
    # S0 falls to the middle; S2 is now 2nd: inside the band, but never held, so not bought.
    feb = ten_stocks("2020-02-29", ["S1", "S2", "S3", "S4", "S0", *TEN[5:]])
    assert banded_legs([jan, feb])[pd.Timestamp("2020-02-29")][0] == {"S1"}


def test_banded_legs_drop_a_stock_with_no_forecast():
    jan = ten_stocks("2020-01-31", TEN)
    feb = ten_stocks("2020-02-29", ["S1", "S0", *TEN[2:]])
    feb.loc[feb["permaticker"] == "S0", "pred"] = np.nan
    oos = pd.concat([jan, feb], ignore_index=True)
    legs = assign_legs_banded(oos, n_groups=10, exit_groups=2)
    assert np.isnan(legs[(oos["permaticker"] == "S0") & (oos["month_end"] == "2020-02-29")]).all()


def random_panel(n_stocks=50, n_months=24, persistence=0.8, seed=0):
    """Forecasts that drift month to month, like a slow-moving signal plus noise."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    signal = rng.normal(size=n_stocks)
    frames = []
    for m in months:
        signal = persistence * signal + np.sqrt(1 - persistence**2) * rng.normal(size=n_stocks)
        frames.append(stocks_oos(m, list(range(n_stocks)), signal, rng.normal(0, 0.05, n_stocks)))
    return pd.concat(frames, ignore_index=True)


def test_banded_legs_with_one_exit_group_are_plain_deciles():
    oos = random_panel()
    assert assign_legs_banded(oos, exit_groups=1).equals(assign_legs(oos))


def test_banding_cuts_turnover():
    # Enough stocks for the band to matter (with 50, each leg is only 5 names).
    oos = random_panel(n_stocks=500)
    plain = turnover(leg_weights(oos)).iloc[1:].mean()
    banded = turnover(leg_weights(oos, exit_groups=2)).iloc[1:].mean()
    assert banded < 0.8 * plain


def test_banded_legs_ignore_future_months():
    # Leakage check: what happens after month t cannot change month t's legs.
    oos = random_panel()
    first = oos["month_end"] <= "2020-12-31"
    wild = oos.copy()
    wild.loc[~first, "pred"] = -wild.loc[~first, "pred"]
    wild.loc[~first, TARGET] = 5.0
    a = assign_legs_banded(oos)[first]
    b = assign_legs_banded(wild)[first]
    assert a.equals(b)


def test_banded_legs_bad_band_raises():
    with pytest.raises(ValueError, match="exit_groups"):
        assign_legs_banded(random_panel(), n_groups=10, exit_groups=6)


def three_months_with_no_view_in_feb():
    # Jan and Mar rank A > B > C > D; in Feb the model has no view (all forecasts equal).
    names = ["A", "B", "C", "D"]
    jan = stocks_oos("2020-01-31", names, [4.0, 3.0, 2.0, 1.0], [0.1, 0.0, 0.0, -0.1])
    feb = stocks_oos("2020-02-29", names, [0.5] * 4, [0.2, 0.0, 0.0, -0.2])
    mar = stocks_oos("2020-03-31", names, [4.0, 3.0, 2.0, 1.0], [0.0] * 4)
    return pd.concat([jan, feb, mar], ignore_index=True)


def test_no_view_month_holds_nothing():
    oos = three_months_with_no_view_in_feb()
    assert no_view_months(oos).tolist() == [pd.Timestamp("2020-02-29")]
    for legs in (assign_legs(oos, n_groups=2), assign_legs_banded(oos, n_groups=4, exit_groups=2)):
        feb = oos["month_end"] == "2020-02-29"
        assert (legs[feb] == 0).all()  # not long A, B by row order
        assert (legs[~feb] != 0).any()


def test_no_view_month_earns_zero_and_is_not_dropped():
    # Feb's long-by-row-order would have earned +0.2 - (-0.2) = 0.4: that must not appear.
    ls = long_short_returns(three_months_with_no_view_in_feb(), n_groups=2)
    assert ls.loc["2020-02-29", "long_short"] == 0.0
    assert ls.loc["2020-02-29", ["n_long", "n_short"]].tolist() == [0, 0]
    assert ls.loc["2020-01-31", "long_short"] == pytest.approx(0.1)


def test_no_view_month_without_outcomes_is_left_out():
    oos = three_months_with_no_view_in_feb()
    oos.loc[oos["month_end"] == "2020-02-29", TARGET] = np.nan
    assert pd.Timestamp("2020-02-29") not in long_short_returns(oos, n_groups=2).index


def test_no_view_month_sells_the_book_and_rebuys_after():
    oos = three_months_with_no_view_in_feb()
    months = pd.DatetimeIndex(oos["month_end"].unique())
    weights = leg_weights(oos, n_groups=2)
    t = turnover(weights, months)
    assert t.tolist() == pytest.approx([2.0, 2.0, 2.0])  # buy, sell everything, buy again
    # Without the calendar, Feb vanishes and Jan's book looks held straight into Mar.
    assert turnover(weights).tolist() == pytest.approx([2.0, 0.0], abs=0.2)


def test_banded_legs_restart_after_a_no_view_month():
    # S1 was long in Jan; after the no-view Feb it is 2nd in Mar: inside the band, but
    # nothing is held any more, so only the top stock enters.
    jan = ten_stocks("2020-01-31", ["S1", *[n for n in TEN if n != "S1"]])
    feb = stocks_oos("2020-02-29", TEN, [1.0] * 10, [0.0] * 10)
    mar = ten_stocks("2020-03-31", TEN)
    legs = banded_legs([jan, feb, mar])
    assert legs[pd.Timestamp("2020-02-29")] == (set(), set())
    assert legs[pd.Timestamp("2020-03-31")] == ({"S0"}, {"S9"})
