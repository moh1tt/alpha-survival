"""Tests for the feature builders. Small hand-made panels, no files or network."""

import numpy as np
import pandas as pd
import pytest

from alpha_survival import features


def panel_for(permaticker, rets, start="2023-01-31"):
    """One stock's monthly panel: consecutive month ends, the given `ret`s, fwd_ret = next ret."""
    months = pd.date_range(start, periods=len(rets), freq="ME")
    rets = pd.Series(rets, dtype=float)
    return pd.DataFrame(
        {
            "permaticker": permaticker,
            "month_end": months,
            "ret": rets,
            "fwd_ret": rets.shift(-1),
        }
    )


# 13 months: the first ret is NaN (no previous close), as in the real panel.
RETS = [np.nan, 0.10, -0.05, 0.02, 0.03, -0.01, 0.04, 0.00, 0.05, -0.02, 0.01, 0.06, 0.50]


def test_momentum_compounds_months_t_minus_11_to_t_minus_1():
    out = features.add_momentum(panel_for(1, RETS))
    expected = np.prod([1 + r for r in RETS[1:12]]) - 1  # months t-11 .. t-1
    assert out["mom_12_1"].iloc[12] == pytest.approx(expected)


def test_momentum_is_nan_until_eleven_past_returns_exist():
    out = features.add_momentum(panel_for(1, RETS))
    assert out["mom_12_1"].iloc[:12].isna().all()


def test_momentum_ignores_current_month_target_and_future():
    # Leakage test: changing month t's ret, the target, or any later row must not move it.
    base = features.add_momentum(panel_for(1, RETS + [0.2, 0.3]))
    shocked = panel_for(1, RETS + [0.2, 0.3])
    shocked.loc[12:, "ret"] = [-0.9, 5.0, -0.5]
    shocked["fwd_ret"] = 99.0
    out = features.add_momentum(shocked)
    assert out["mom_12_1"].iloc[12] == pytest.approx(base["mom_12_1"].iloc[12])


def test_momentum_is_nan_when_a_month_is_missing():
    # Drop a row inside the window: 11 rows now span 12 months, not a valid window.
    panel = panel_for(1, RETS + [0.2]).drop(index=5)
    out = features.add_momentum(panel).set_index("month_end")
    assert out["mom_12_1"].isna().all()


def test_momentum_does_not_bleed_across_stocks():
    # Stock 2 starts right after stock 1 ends; stock 2's first rows must stay NaN.
    panel = pd.concat([panel_for(1, RETS), panel_for(2, RETS, start="2024-02-29")])
    out = features.add_momentum(panel)
    second = out[out["permaticker"] == 2]
    assert second["mom_12_1"].iloc[:12].isna().all()
    assert second["mom_12_1"].iloc[12] == pytest.approx(out["mom_12_1"].iloc[12])


def test_momentum_duplicate_rows_raise():
    panel = pd.concat([panel_for(1, RETS), panel_for(1, RETS)])
    with pytest.raises(ValueError, match="duplicate"):
        features.add_momentum(panel)


def test_size_is_log_market_cap():
    panel = pd.DataFrame({"market_cap": [1e6, 1e9, np.e]})
    out = features.add_size(panel)
    assert list(out["log_mcap"]) == pytest.approx([np.log(1e6), np.log(1e9), 1.0])


def test_size_is_nan_without_a_positive_market_cap():
    out = features.add_size(pd.DataFrame({"market_cap": [np.nan, 0.0, -5.0]}))
    assert out["log_mcap"].isna().all()


def test_size_uses_only_its_own_month():
    # Leakage test: changing other months' market caps leaves this month untouched.
    base = features.add_size(pd.DataFrame({"market_cap": [1e8, 2e8, 3e8]}))
    shocked = features.add_size(pd.DataFrame({"market_cap": [1e8, 2e8, 9e12]}))
    assert list(shocked["log_mcap"].iloc[:2]) == list(base["log_mcap"].iloc[:2])


def month_rows(month, values):
    """One month's cross-section: stock i has feature value values[i]."""
    return pd.DataFrame(
        {"permaticker": range(len(values)), "month_end": pd.Timestamp(month), "x": values}
    )


def test_rank_normalise_spreads_each_month_over_minus_one_to_one():
    out = features.rank_normalise(month_rows("2020-01-31", [10.0, -3.0, 500.0, 7.0, 0.0]), ["x"])
    # Order -3 < 0 < 7 < 10 < 500 -> -1, -0.5, 0, 0.5, 1; the 500 outlier is just +1.
    assert out["x_rank"].tolist() == [0.5, -1.0, 1.0, 0.0, -0.5]
    assert out["x"].tolist() == [10.0, -3.0, 500.0, 7.0, 0.0]  # raw column kept


def test_rank_normalise_ties_missing_and_single_values():
    tied = features.rank_normalise(month_rows("2020-01-31", [1.0, 1.0, 2.0]), ["x"])
    assert tied["x_rank"].tolist() == [-0.5, -0.5, 1.0]

    gaps = features.rank_normalise(month_rows("2020-01-31", [1.0, np.nan, 3.0]), ["x"])
    assert gaps["x_rank"].tolist() == [-1.0, 0.0, 1.0]  # missing -> the middle

    single = features.rank_normalise(month_rows("2020-01-31", [np.nan, 4.0]), ["x"])
    assert single["x_rank"].tolist() == [0.0, 0.0]  # one value has no order


def test_rank_normalise_is_per_month_so_other_months_cannot_move_it():
    jan = month_rows("2020-01-31", [1.0, 2.0, 3.0])
    feb = month_rows("2020-02-29", [5.0, 4.0, 6.0])
    feb_wild = month_rows("2020-02-29", [-100.0, 1e9, 0.0])
    a = features.rank_normalise(pd.concat([jan, feb]), ["x"])
    b = features.rank_normalise(pd.concat([jan, feb_wild]), ["x"])
    jan_a = a[a["month_end"] == "2020-01-31"]["x_rank"]
    jan_b = b[b["month_end"] == "2020-01-31"]["x_rank"]
    assert jan_a.tolist() == jan_b.tolist() == [-1.0, 0.0, 1.0]


def test_rank_normalise_several_columns_and_suffix():
    rows = month_rows("2020-01-31", [1.0, 2.0, 3.0]).assign(y=[3.0, 2.0, 1.0])
    out = features.rank_normalise(rows, ["x", "y"], suffix="_r")
    assert out["x_r"].tolist() == [-1.0, 0.0, 1.0]
    assert out["y_r"].tolist() == [1.0, 0.0, -1.0]


def test_add_value_is_book_over_market_in_usd():
    rows = pd.DataFrame(
        {
            "equity": [50.0, 900.0, -10.0, 5.0, np.nan],  # 900 is in a foreign currency
            "equityusd": [50.0, 100.0, -10.0, 5.0, np.nan],
            "market_cap": [100.0, 400.0, 100.0, 0.0, 100.0],
        }
    )
    out = features.add_value(rows)
    # USD book / USD market; negative book, zero cap and missing inputs -> NaN.
    assert out["bm"].tolist()[:2] == [0.5, 0.25]
    assert out["bm"].iloc[2:].isna().all()


def test_add_value_uses_only_the_row_itself():
    rows = pd.DataFrame({"equityusd": [10.0, 20.0], "market_cap": [100.0, 100.0]})
    changed = rows.assign(equityusd=[10.0, 1e9])  # e.g. a later filing on another row
    assert features.add_value(rows)["bm"].iloc[0] == features.add_value(changed)["bm"].iloc[0]


def test_add_reversal_is_this_months_return_never_the_target():
    panel = panel_for(1, [0.10, -0.20, np.nan, 0.05])
    out = features.add_reversal(panel)
    assert out["rev_1m"].tolist()[:2] == [0.10, -0.20]
    assert np.isnan(out["rev_1m"].iloc[2])
    # Leakage check: changing next month's return (fwd_ret) cannot move it.
    shocked = features.add_reversal(panel.assign(fwd_ret=panel["fwd_ret"] * 100))
    assert shocked["rev_1m"].equals(out["rev_1m"])


def test_add_profitability_is_gross_profit_over_assets():
    rows = pd.DataFrame(
        {
            "gp": [30.0, -5.0, 80.0, 10.0, np.nan],
            "cor": [70.0, 20.0, 0.0, 5.0, 1.0],  # cor 0: a bank, gp is just revenue
            "assets": [100.0, 50.0, 1000.0, 0.0, 10.0],
        }
    )
    out = features.add_profitability(rows)
    assert out["gp_a"].tolist()[:2] == [0.3, -0.1]  # negative gross profit is kept
    assert out["gp_a"].iloc[2:].isna().all()  # no COGS, zero assets, missing gp


def test_add_profitability_uses_only_the_row_itself():
    rows = pd.DataFrame({"gp": [10.0, 20.0], "cor": [1.0, 1.0], "assets": [100.0, 100.0]})
    changed = rows.assign(gp=[10.0, 1e9])
    a = features.add_profitability(rows)["gp_a"].iloc[0]
    assert a == features.add_profitability(changed)["gp_a"].iloc[0]


def assets_panel(assets, start="2020-01-31", permaticker=1):
    months = pd.date_range(start, periods=len(assets), freq="ME")
    return pd.DataFrame(
        {"permaticker": permaticker, "month_end": months, "assets": pd.Series(assets, dtype=float)}
    )


def test_asset_growth_compares_with_the_row_twelve_months_earlier():
    assets = [100.0] * 3 + [110.0] * 9 + [150.0, 120.0, 0.0, 130.0]
    out = features.add_asset_growth(assets_panel(assets))
    assert out["asset_growth"].iloc[:12].isna().all()  # no year-ago row yet
    # Month 12 vs month 0 (100), month 13 vs month 1 (100), month 15 vs month 3 (110).
    assert out["asset_growth"].iloc[12] == pytest.approx(0.5)
    assert out["asset_growth"].iloc[13] == pytest.approx(0.2)
    assert np.isnan(out["asset_growth"].iloc[14])  # zero assets now
    assert out["asset_growth"].iloc[15] == pytest.approx(130 / 110 - 1)


def test_asset_growth_gap_gives_nan_not_a_longer_window():
    panel = assets_panel([100.0 + i for i in range(20)]).drop(index=2)  # month 2 missing
    out = features.add_asset_growth(panel).set_index("month_end")["asset_growth"]
    months = pd.date_range("2020-01-31", periods=20, freq="ME")
    assert np.isnan(out[months[14]])  # its year-ago row (month 2) is gone
    assert out[months[15]] == pytest.approx(115 / 103 - 1)  # unaffected neighbours


def test_asset_growth_ignores_the_future_and_other_stocks():
    a = assets_panel([100.0 + i for i in range(24)])
    b = assets_panel([5.0] * 24, permaticker=2)
    base = features.add_asset_growth(pd.concat([a, b], ignore_index=True))
    # Leakage check: rewrite every row after month 15 and the other stock entirely.
    a_future = a.assign(assets=np.where(a.index > 15, 1e9, a["assets"]))
    shocked = features.add_asset_growth(
        pd.concat([a_future, b.assign(assets=1.0)], ignore_index=True)
    )
    first = base["permaticker"] == 1
    assert (
        base.loc[first, "asset_growth"]
        .iloc[:16]
        .equals(shocked.loc[first, "asset_growth"].iloc[:16])
    )


def test_asset_growth_rejects_duplicate_rows():
    panel = assets_panel([100.0, 110.0])
    with pytest.raises(ValueError, match="duplicate"):
        features.add_asset_growth(pd.concat([panel, panel]))


def moments_panel(daily_by_month, start="2020-01-31", permaticker=1):
    """Panel rows carrying each month's daily-return count, sum and sum of squares."""
    months = pd.date_range(start, periods=len(daily_by_month), freq="ME")
    return pd.DataFrame(
        {
            "permaticker": permaticker,
            "month_end": months,
            "n_days": [len(d) for d in daily_by_month],
            "dret_sum": [float(np.sum(d)) for d in daily_by_month],
            "dret_sumsq": [float(np.sum(np.square(d))) for d in daily_by_month],
        }
    )


def test_volatility_matches_std_of_the_pooled_daily_returns():
    rng = np.random.default_rng(0)
    daily = [rng.normal(0, 0.02, 21) for _ in range(5)]
    out = features.add_volatility(moments_panel(daily))
    assert np.isnan(out["vol_3m"].iloc[0])  # 21 days < 40
    assert out["vol_3m"].iloc[1] == pytest.approx(np.std(np.concatenate(daily[:2]), ddof=1))
    for t in range(2, 5):
        expected = np.std(np.concatenate(daily[t - 2 : t + 1]), ddof=1)
        assert out["vol_3m"].iloc[t] == pytest.approx(expected)


def test_volatility_missing_month_adds_no_days_and_min_days_applies():
    rng = np.random.default_rng(1)
    daily = [rng.normal(0, 0.02, 21) for _ in range(4)]
    panel = moments_panel(daily).drop(index=1)  # month 1 has no row at all
    out = features.add_volatility(panel).set_index("month_end")["vol_3m"]
    months = pd.date_range("2020-01-31", periods=4, freq="ME")
    # Month 2 pools months 0 and 2; month 3 pools months 2 and 3 (42 days each).
    assert out[months[2]] == pytest.approx(np.std(np.concatenate([daily[0], daily[2]]), ddof=1))
    assert out[months[3]] == pytest.approx(np.std(np.concatenate(daily[2:4]), ddof=1))

    short = features.add_volatility(moments_panel([rng.normal(0, 0.02, 13) for _ in range(3)]))
    assert np.isnan(short["vol_3m"].iloc[2])  # 39 days < 40


def test_volatility_ignores_later_months_and_other_stocks():
    rng = np.random.default_rng(2)
    a = moments_panel([rng.normal(0, 0.02, 21) for _ in range(6)])
    b = moments_panel([rng.normal(0, 0.05, 21) for _ in range(6)], permaticker=2)
    base = features.add_volatility(pd.concat([a, b], ignore_index=True))
    # Leakage check: wild daily returns after month 3, and a different other stock.
    a_future = a.copy()
    a_future.loc[4:, ["dret_sum", "dret_sumsq"]] = 1e6
    shocked = features.add_volatility(
        pd.concat([a_future, b.assign(dret_sumsq=9.0)], ignore_index=True)
    )
    assert base["vol_3m"].iloc[:4].equals(shocked["vol_3m"].iloc[:4])


# 40 months of distinct returns, so every window's product is unique.
LONG_RETS = [np.nan] + [0.01 * ((i % 7) - 3) + 0.001 * i for i in range(1, 40)]


@pytest.mark.parametrize("name", list(features.PAST_RETURN_WINDOWS))
def test_past_return_windows_compound_exactly_their_lags(name):
    first, last = features.PAST_RETURN_WINDOWS[name]
    out = features.add_past_returns(panel_for(1, LONG_RETS))
    t = 39
    expected = np.prod([1 + LONG_RETS[t - lag] for lag in range(first, last + 1)]) - 1
    assert out[name].iloc[t] == pytest.approx(expected)
    # NaN until the oldest lag exists: row `last` is the first with t-last at month 1's ret.
    assert out[name].iloc[: last + 1].isna().all() and out[name].iloc[last + 1 :].notna().all()


def test_past_return_window_matches_mom_12_1_convention():
    # Lags 1-11 through the shared helper reproduce mom_12_1 exactly.
    panel = features._sorted_by_stock(panel_for(1, LONG_RETS))
    mom = features.add_momentum(panel)["mom_12_1"]
    assert features._past_return(panel, 1, 11).equals(mom)


def test_past_returns_ignore_current_month_target_and_future():
    # Leakage test: shocking month t's ret, every target and all later rows moves nothing at t.
    base = features.add_past_returns(panel_for(1, LONG_RETS))
    shocked = panel_for(1, LONG_RETS)
    shocked.loc[37:, "ret"] = [5.0, -0.9, 3.0]
    shocked["fwd_ret"] = 99.0
    out = features.add_past_returns(shocked)
    cols = [*features.PAST_RETURN_WINDOWS, "mom_rev"]
    pd.testing.assert_frame_equal(out.loc[:36, cols], base.loc[:36, cols])


def test_past_returns_are_nan_across_a_gap():
    panel = panel_for(1, LONG_RETS).drop(index=20)  # month 20 missing
    out = features.add_past_returns(panel).set_index("month_end")
    months = panel_for(1, LONG_RETS)["month_end"]
    # mr_reversal at month 33 spans months 15-20 (lags 13-18): the gap makes it NaN;
    # at month 39 it spans 21-26, after the gap: defined again.
    assert np.isnan(out.loc[months[33], "mr_reversal"])
    assert np.isfinite(out.loc[months[39], "mr_reversal"])


def test_mom_rev_flags_recent_winners_that_were_long_run_losers():
    # 10 stocks, 40 months. In the last month, stocks 0-1 rose recently after
    # falling long-run (MomRev = 1); stocks 8-9 the reverse (0); the rest NaN.
    stocks = []
    for s in range(10):
        # Last row is month 39: lags 1-5 are months 34-38, lags 13-36 months 3-26.
        tilt = 0.05 * (s - 4.5)  # stock 0 most negative, stock 9 most positive
        rets = [np.nan] + [0.0] * 39
        rets[34:39] = [-tilt] * 5  # recent 6 months: low s = winners
        rets[4:27] = [tilt] * 23  # long run: low s = losers
        stocks.append(panel_for(s, rets))
    out = features.add_past_returns(pd.concat(stocks, ignore_index=True))
    last = out[out["month_end"] == out["month_end"].max()].set_index("permaticker")["mom_rev"]
    assert last.loc[[0, 1]].tolist() == [1.0, 1.0]
    assert last.loc[[8, 9]].tolist() == [0.0, 0.0]
    assert last.loc[2:7].isna().all()


def test_season_lag_sets_match_osap():
    lags = features.SEASONAL_LAGS
    assert lags["mom_season_short"] == [11]
    assert lags["mom_season"] == [23, 35, 47, 59]
    assert lags["mom_season_16y"] == [191, 203, 215, 227, 239]
    assert lags["mom_offseason_12m"] == list(range(1, 11))
    assert lags["mom_offseason"] == [k for k in range(12, 60) if k not in (23, 35, 47, 59)]
    assert len(lags["mom_offseason_6y"]) == 55 and 71 not in lags["mom_offseason_6y"]


def seasonal_panel(n_months=70, permaticker=1):
    """Returns encode their own month index, so any average can be checked by hand."""
    rets = [np.nan] + [0.001 * i for i in range(1, n_months)]
    return panel_for(permaticker, rets, start="2000-01-31")


def test_seasonality_averages_exactly_its_lags():
    out = features.add_seasonality(seasonal_panel())
    t = 65  # forecasting month 66 (a July if month 0 is Jan 2000)
    assert out["mom_season_short"].iloc[t] == pytest.approx(0.001 * (t - 11))
    assert out["mom_season"].iloc[t] == pytest.approx(
        np.mean([0.001 * (t - k) for k in (23, 35, 47, 59)])
    )
    assert out["mom_offseason_12m"].iloc[t] == pytest.approx(
        np.mean([0.001 * (t - k) for k in range(1, 11)])
    )
    # Calendar check: lag 11 from row t is the same calendar month as t + 1, a year earlier.
    months = out["month_end"]
    assert (months.iloc[t] - pd.DateOffset(months=11)).month == (
        months.iloc[t] + pd.offsets.MonthEnd(1)
    ).month


def test_seasonality_uses_available_lags_and_is_nan_with_none():
    out = features.add_seasonality(seasonal_panel(n_months=40))
    t = 39  # only lags 23 and 35 of mom_season exist (47, 59 predate the data)
    assert out["mom_season"].iloc[t] == pytest.approx(np.mean([0.001 * 16, 0.001 * 4]))
    assert out["mom_season"].iloc[:23].isna().all()  # no lag 23 yet
    assert out["mom_season_6y"].isna().all()  # 6+ years of history never exists


def test_seasonality_lags_by_calendar_month_across_a_gap():
    # Drop a row: with row shifts, lag 11 would land a month too far back.
    full = features.add_seasonality(seasonal_panel()).set_index("month_end")
    gappy = features.add_seasonality(seasonal_panel().drop(index=30)).set_index("month_end")
    t = full.index[60]  # lag 11 = month 49, after the gap: unaffected
    assert gappy.loc[t, "mom_season_short"] == full.loc[t, "mom_season_short"]
    t41 = full.index[41]  # lag 11 = month 30, the missing month -> NaN
    assert np.isnan(gappy.loc[t41, "mom_season_short"])


def test_seasonality_ignores_current_month_target_and_future_and_other_stocks():
    base = features.add_seasonality(seasonal_panel())
    shocked = seasonal_panel()
    shocked.loc[50:, "ret"] = 9.0
    shocked["fwd_ret"] = 99.0
    other = seasonal_panel(permaticker=2).assign(ret=-0.5)
    out = features.add_seasonality(pd.concat([shocked, other], ignore_index=True))
    out = out[out["permaticker"] == 1].reset_index(drop=True)
    cols = list(features.SEASONAL_LAGS)
    pd.testing.assert_frame_equal(out.loc[:49, cols], base.loc[:49, cols])


def daily_summary_panel(n_months=40, permaticker=1, start="2020-01-31"):
    """Monthly rows with hand-set daily summaries; month i's values encode i."""
    i = np.arange(n_months, dtype=float)
    return pd.DataFrame(
        {
            "permaticker": permaticker,
            "month_end": pd.date_range(start, periods=n_months, freq="ME"),
            "ret": 0.01,
            "fwd_ret": 0.0,
            "n_days": 20.0,
            "n_trade_days": 20.0,
            "dret_sum": 0.0,
            "dret_sumsq": 0.02,
            "dret_sumcube": 0.0,
            "dret_max": 0.01 * (i + 1),
            "close_split_adj": 100.0 + i,
            "close_max": 110.0 + i,
            "illiq_sum": 20 * (i + 1),
            "illiq_n": 20.0,
            "dollar_volume": 1e6 * (i + 1),
            "share_volume": 1e4 * (i + 1) ** 2,
        }
    )


def test_daily_price_signals_by_hand():
    out = features.add_daily_price_signals(daily_summary_panel())
    t = 30
    assert out["max_ret"].iloc[t] == pytest.approx(0.31)
    assert out["return_skew"].iloc[t] == 0.0  # zero third moment
    # High52: close at t over the max close_max of months t-11..t (rising, so month t's).
    assert out["high_52"].iloc[t] == pytest.approx((100 + t) / (110 + t))
    # Illiquidity: mean of monthly means (i + 1) over months t-11..t.
    assert out["illiquidity"].iloc[t] == pytest.approx(np.mean(np.arange(t - 10, t + 2)))
    assert out["dol_vol"].iloc[t] == pytest.approx(np.log(1e6 * (t - 1)))  # month t-2
    months = np.arange(max(0, t - 35), t + 1)  # 36-month window, clipped at the first month
    assert out["vol_sd"].iloc[t] == pytest.approx(np.std(1e4 * (months + 1.0) ** 2, ddof=1))


def test_return_skew_by_hand_and_needs_15_days():
    rets = np.array([0.05, -0.01, -0.01, -0.01, -0.02])
    row = daily_summary_panel(n_months=1).assign(
        n_days=5.0,
        dret_sum=rets.sum(),
        dret_sumsq=(rets**2).sum(),
        dret_sumcube=(rets**3).sum(),
    )
    from scipy.stats import skew

    few = features.add_daily_price_signals(row.assign(n_trade_days=14.0))
    assert np.isnan(few["return_skew"].iloc[0])
    enough = features.add_daily_price_signals(row.assign(n_trade_days=15.0))
    assert enough["return_skew"].iloc[0] == pytest.approx(skew(rets))  # population skew


def test_daily_price_signal_windows_need_their_history():
    out = features.add_daily_price_signals(daily_summary_panel())
    assert out["illiquidity"].iloc[:11].isna().all() and out["illiquidity"].iloc[11:].notna().all()
    assert out["dol_vol"].iloc[:2].isna().all() and np.isfinite(out["dol_vol"].iloc[2])
    assert out["vol_sd"].iloc[:23].isna().all() and out["vol_sd"].iloc[23:].notna().all()
    assert out["high_52"].notna().all()  # uses whatever of the 12 months exist


def test_daily_price_windows_run_over_calendar_months():
    # Drop month 25: illiquidity at month 30 needs months 19-30, so it is NaN;
    # at month 37 (window 26-37) it is back. dol_vol at month 27 reads month 25: NaN.
    out = features.add_daily_price_signals(daily_summary_panel().drop(index=25))
    out = out.set_index(out["month_end"].dt.to_period("M"))
    first = pd.Period("2020-01", "M")
    assert np.isnan(out.loc[first + 30, "illiquidity"])
    assert np.isfinite(out.loc[first + 37, "illiquidity"])
    assert np.isnan(out.loc[first + 27, "dol_vol"])


def test_daily_price_signals_ignore_current_target_future_and_other_stocks():
    base = features.add_daily_price_signals(daily_summary_panel())
    shocked = daily_summary_panel()
    later = shocked.index >= 31
    shocked.loc[later, ["dret_max", "close_max", "illiq_sum", "dollar_volume", "share_volume"]] = (
        9e9
    )
    shocked["fwd_ret"] = 99.0
    other = daily_summary_panel(permaticker=2).assign(close_max=1e9, share_volume=1.0)
    out = features.add_daily_price_signals(pd.concat([shocked, other], ignore_index=True))
    out = out[out["permaticker"] == 1].reset_index(drop=True)
    cols = ["max_ret", "return_skew", "high_52", "illiquidity", "dol_vol", "vol_sd"]
    pd.testing.assert_frame_equal(out.loc[:30, cols], base.loc[:30, cols])


def test_trend_price_age_by_hand():
    panel = daily_summary_panel(n_months=70).assign(close_unadj=lambda d: 20.0 + d.index)
    panel["share_volume"] = 1000.0 + 10.0 * np.arange(70)  # exact linear trend
    out = features.add_trend_price_age(panel)
    t = 65  # window is months 6..65, mean volume 1000 + 10 * 35.5
    assert out["volume_trend"].iloc[t] == pytest.approx(10.0 / (1000 + 10 * 35.5))
    assert out["price"].iloc[t] == pytest.approx(np.log(20.0 + t))
    assert list(out["firm_age"].iloc[:3]) == [1.0, 2.0, 3.0]


def test_volume_trend_matches_least_squares_and_needs_30_months():
    rng = np.random.default_rng(0)
    panel = daily_summary_panel(n_months=70).assign(close_unadj=10.0)
    panel["share_volume"] = rng.uniform(1e5, 1e6, 70)
    out = features.add_trend_price_age(panel)
    assert (
        out["volume_trend"].iloc[:29].isna().all() and out["volume_trend"].iloc[29:].notna().all()
    )
    t = 50
    window = panel["share_volume"].iloc[t - 50 : t + 1].to_numpy()  # 51 months, all < 60
    slope = np.polyfit(np.arange(len(window)), window, 1)[0]
    assert out["volume_trend"].iloc[t] == pytest.approx(slope / window.mean())


def test_volume_trend_and_age_run_over_calendar_months():
    # Drop months 10-14: the trend's time axis must still be calendar months, and
    # age must count the missing months too.
    panel = daily_summary_panel(n_months=70).assign(close_unadj=10.0)
    panel["share_volume"] = 1000.0 + 10.0 * np.arange(70)
    out = features.add_trend_price_age(panel.drop(index=range(10, 15)))
    out = out.set_index(out["month_end"].dt.to_period("M"))
    first = pd.Period("2020-01", "M")
    # Volume is linear in calendar time, so the slope is exactly 10 despite the gap
    # (counting rows instead would squeeze the axis and steepen it).
    present = np.array([m for m in range(41) if not 10 <= m <= 14])
    expected = 10.0 / (1000.0 + 10.0 * present).mean()
    assert out.loc[first + 40, "volume_trend"] == pytest.approx(expected)
    assert out.loc[first + 40, "firm_age"] == 41.0


def test_trend_price_age_ignore_future_target_and_other_stocks():
    base = features.add_trend_price_age(daily_summary_panel(n_months=70).assign(close_unadj=10.0))
    shocked = daily_summary_panel(n_months=70).assign(close_unadj=10.0)
    shocked.loc[shocked.index >= 41, ["share_volume", "close_unadj"]] = 9e9
    shocked["fwd_ret"] = 99.0
    other = daily_summary_panel(n_months=70, permaticker=2, start="2015-01-31").assign(
        close_unadj=1.0, share_volume=1.0
    )
    out = features.add_trend_price_age(pd.concat([shocked, other], ignore_index=True))
    out = out[out["permaticker"] == 1].reset_index(drop=True)
    cols = ["volume_trend", "price", "firm_age"]
    pd.testing.assert_frame_equal(out.loc[:40, cols], base.loc[:40, cols])
