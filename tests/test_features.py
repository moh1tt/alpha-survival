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
            "cs_sum": 0.2 * (i + 1),
            "cs_n": 20.0,
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
    assert out["bid_ask_spread"].iloc[t] == pytest.approx(0.01 * (t + 1))


def test_bid_ask_spread_needs_12_daily_estimates():
    panel = daily_summary_panel(n_months=2).assign(cs_n=[11.0, 12.0], cs_sum=[1.1, 1.2])
    out = features.add_daily_price_signals(panel)
    assert np.isnan(out["bid_ask_spread"].iloc[0])
    assert out["bid_ask_spread"].iloc[1] == pytest.approx(0.1)


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
    summaries = ["dret_max", "close_max", "illiq_sum", "dollar_volume", "share_volume", "cs_sum"]
    shocked.loc[later, summaries] = 9e9
    shocked["fwd_ret"] = 99.0
    other = daily_summary_panel(permaticker=2).assign(close_max=1e9, share_volume=1.0)
    out = features.add_daily_price_signals(pd.concat([shocked, other], ignore_index=True))
    out = out[out["permaticker"] == 1].reset_index(drop=True)
    cols = [
        "max_ret",
        "return_skew",
        "high_52",
        "illiquidity",
        "dol_vol",
        "vol_sd",
        "bid_ask_spread",
    ]
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


def accounting_row(**overrides):
    """One stock-month with round accounting numbers (USD filer unless fxusd given)."""
    row = {
        "permaticker": 1,
        "month_end": pd.Timestamp("2020-01-31"),
        "market_cap": 1000.0,
        "fxusd": 1.0,
        "assets": 2000.0,
        "liabilities": 1200.0,
        "equity": 800.0,
        "revenue": 500.0,
        "netinc": 50.0,
        "depamor": 30.0,
        "ncfo": 90.0,
        "rnd": 20.0,
        "cashneq": 100.0,
        "debt": 400.0,
        "ebitda": 130.0,
    }
    return pd.DataFrame([row | overrides])


def test_valuation_ratios_by_hand():
    out = features.add_valuation_ratios(accounting_row()).iloc[0]
    assert out["am"] == 2.0 and out["sp"] == 0.5 and out["leverage"] == 1.2
    assert out["cf"] == pytest.approx(0.08) and out["cfp"] == pytest.approx(0.09)
    assert out["rd"] == pytest.approx(0.02) and out["ep"] == pytest.approx(0.05)
    assert out["cash_prod"] == pytest.approx((1000 - 2000) / 100)
    assert out["ent_mult"] == pytest.approx((1000 + 400 - 100) / 130)


def test_valuation_ratios_convert_foreign_filers_to_usd():
    # Same company reporting in a currency worth half a dollar: every number doubles
    # in local units, fxusd = 2, and every USD ratio is unchanged.
    usd = features.add_valuation_ratios(accounting_row())
    values = ["assets", "liabilities", "equity", "revenue", "netinc", "depamor", "ncfo"]
    values += ["rnd", "cashneq", "debt", "ebitda"]
    local = accounting_row(fxusd=2.0, **{c: 2 * accounting_row()[c].iloc[0] for c in values})
    cols = ["am", "sp", "leverage", "cf", "cfp", "rd", "ep", "cash_prod", "ent_mult"]
    pd.testing.assert_frame_equal(features.add_valuation_ratios(local)[cols], usd[cols])


def test_valuation_ratio_exclusions():
    loss = features.add_valuation_ratios(accounting_row(netinc=-10.0)).iloc[0]
    assert np.isnan(loss["ep"]) and loss["cf"] == pytest.approx(0.02)  # CF keeps losses
    assert np.isnan(features.add_valuation_ratios(accounting_row(ebitda=-5.0))["ent_mult"].iloc[0])
    assert np.isnan(features.add_valuation_ratios(accounting_row(equity=-5.0))["ent_mult"].iloc[0])
    assert np.isnan(features.add_valuation_ratios(accounting_row(cashneq=0.0))["cash_prod"].iloc[0])
    no_cap = features.add_valuation_ratios(accounting_row(market_cap=0.0)).iloc[0]
    assert no_cap[["am", "sp", "ep", "cfp"]].isna().all()
    assert np.isnan(features.add_valuation_ratios(accounting_row(fxusd=np.nan))["am"].iloc[0])


def test_valuation_ratios_use_only_their_own_row():
    # Leakage test: fundamentals are already point-in-time per row (build_fundamentals),
    # so each ratio must read nothing from other rows, e.g. a later filing.
    rows = pd.concat([accounting_row(), accounting_row(month_end=pd.Timestamp("2020-02-29"))])
    later = rows.assign(assets=[2000.0, 9e9], market_cap=[1000.0, 1.0], netinc=[50.0, -9e9])
    cols = ["am", "sp", "leverage", "cf", "cfp", "rd", "ep", "cash_prod", "ent_mult"]
    a = features.add_valuation_ratios(rows.reset_index(drop=True))
    b = features.add_valuation_ratios(later.reset_index(drop=True))
    pd.testing.assert_frame_equal(a.loc[:0, cols], b.loc[:0, cols])


def payout_panel(n_months=30, permaticker=1, start="2020-01-31", **overrides):
    """One stock's monthly rows; market cap = 1000 + 10 x month number."""
    i = np.arange(n_months, dtype=float)
    frame = pd.concat([accounting_row(permaticker=permaticker)] * n_months, ignore_index=True)
    frame["month_end"] = pd.date_range(start, periods=n_months, freq="ME")
    frame["market_cap"] = 1000.0 + 10 * i
    frame["ncfdiv"] = -20.0  # dividends paid (an outflow)
    frame["ncfcommon"] = -30.0  # net buyback
    frame["sic_code"] = 3571.0
    frame["taxliabilities"] = 50.0
    for col, value in overrides.items():
        frame[col] = value
    return frame


def test_payout_yields_by_hand_over_cap_six_months_earlier():
    out = features.add_book_payout_debt_ratios(payout_panel())
    t = 26  # cap at month 20 = 1200
    assert out["payout_yield"].iloc[t] == pytest.approx((20 + 30) / 1200)
    assert out["net_payout_yield"].iloc[t] == pytest.approx((20 + 30) / 1200)
    assert out["payout_yield"].iloc[:23].isna().all()  # before the 24th month
    issuer = features.add_book_payout_debt_ratios(payout_panel(ncfcommon=50.0))
    assert issuer["payout_yield"].iloc[t] == pytest.approx(20 / 1200)  # issuance ignored
    assert issuer["net_payout_yield"].iloc[t] == pytest.approx((20 - 50) / 1200)


def test_payout_and_net_debt_exclusions():
    bank = features.add_book_payout_debt_ratios(payout_panel(sic_code=6022.0))
    assert bank[["payout_yield", "net_payout_yield", "net_debt_price"]].isna().all().all()
    assert bank["ebm"].notna().all()  # EBM has no industry rule
    none = features.add_book_payout_debt_ratios(payout_panel(ncfdiv=0.0, ncfcommon=0.0))
    assert none[["payout_yield", "net_payout_yield"]].isna().all().all()
    negative_book = features.add_book_payout_debt_ratios(payout_panel(equity=-1.0))
    assert negative_book["net_payout_yield"].isna().all()


def test_bm_dec_uses_last_december_from_june_and_the_one_before_until_may():
    out = features.add_book_payout_debt_ratios(payout_panel(n_months=36, start="2020-01-31"))
    out = out.set_index(out["month_end"].dt.strftime("%Y-%m"))
    dec_2020, dec_2021 = 1000 + 10 * 11, 1000 + 10 * 23
    assert out.loc["2022-05", "bm_dec"] == pytest.approx(800 / dec_2020)
    assert out.loc["2022-06", "bm_dec"] == pytest.approx(800 / dec_2021)
    assert np.isnan(out.loc["2021-05", "bm_dec"])  # needs Dec 2019: before the data


def test_ebm_bpebm_and_net_debt_by_hand():
    out = features.add_book_payout_debt_ratios(payout_panel()).iloc[0]
    ebm = (800 + 100 - 400) / (1000 + 100 - 400)
    assert out["ebm"] == pytest.approx(ebm)
    assert out["bpebm"] == pytest.approx(800 / 1000 - ebm)


def test_book_leverage_by_hand():
    out = features.add_book_payout_debt_ratios(payout_panel(n_months=1))
    assert out["book_leverage"].iloc[0] == pytest.approx(2000 / (800 + 50))
    no_equity = features.add_book_payout_debt_ratios(payout_panel(n_months=1, equity=np.nan))
    assert no_equity["book_leverage"].iloc[0] == pytest.approx(2000 / (2000 - 1200 + 50))
    no_tax = features.add_book_payout_debt_ratios(payout_panel(n_months=1, taxliabilities=np.nan))
    assert no_tax["book_leverage"].iloc[0] == pytest.approx(2000 / 800)
    zero = features.add_book_payout_debt_ratios(payout_panel(n_months=1, equity=-50.0))
    assert zero["book_leverage"].isna().all()
    negative = features.add_book_payout_debt_ratios(payout_panel(n_months=1, equity=-150.0))
    assert negative["book_leverage"].iloc[0] == pytest.approx(-20.0)  # kept, as OSAP


def test_net_debt_price_only_for_the_top_three_fifths_of_book_to_market():
    stocks = [payout_panel(n_months=1, permaticker=k, equity=100.0 * (k + 1)) for k in range(5)]
    out = features.add_book_payout_debt_ratios(pd.concat(stocks, ignore_index=True))
    out = out.set_index("permaticker")
    assert out.loc[[0, 1], "net_debt_price"].isna().all()  # lowest two B/M fifths
    assert out.loc[[2, 3, 4], "net_debt_price"].to_numpy() == pytest.approx([0.3] * 3)


def test_book_payout_debt_ratios_ignore_later_months_and_other_stocks():
    cols = ["bm_dec", "payout_yield", "net_payout_yield", "ebm", "bpebm", "book_leverage"]
    base = features.add_book_payout_debt_ratios(payout_panel(n_months=36))
    shocked = payout_panel(n_months=36)
    shocked.loc[shocked.index >= 30, ["market_cap", "equity", "ncfdiv"]] = [1.0, -9e9, 9e9]
    other = payout_panel(n_months=36, permaticker=2, market_cap=1e12, equity=1.0)
    out = features.add_book_payout_debt_ratios(pd.concat([shocked, other], ignore_index=True))
    out = out[out["permaticker"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(out.loc[:29, cols], base.loc[:29, cols])


def profit_panel(n_months=14, permaticker=1, **overrides):
    """One stock's monthly rows with round income-statement and balance-sheet numbers."""
    base = {
        "revenue": 1000.0,
        "cor": 600.0,
        "sgna": 150.0,  # excludes R&D, as in Sharadar
        "intexp": 10.0,
        "taxexp": 35.0,
        "receivables": 100.0,
        "inventory": 80.0,
        "deferredrev": 20.0,
        "payables": 60.0,
        "ppnenet": 400.0,
        "sic_code": 3571.0,
    }
    frame = payout_panel(n_months=n_months, permaticker=permaticker, **base)
    for col, value in overrides.items():
        frame[col] = value
    return frame


def test_profitability_ratios_by_hand():
    out = features.add_profitability_ratios(profit_panel()).iloc[-1]
    # assets 2000, equity 800, netinc 50, rnd 20 (accounting_row)
    assert out["roe"] == pytest.approx(50 / 800)
    assert out["oper_prof"] == pytest.approx((1000 - 600 - 150 - 20 - 10) / 800)
    assert out["oper_prof_rd"] == pytest.approx((1000 - 600 - 150) / 2000)
    assert out["cb_oper_prof"] == pytest.approx((1000 - 600 - 150) / 2000)  # no changes
    assert out["op_leverage"] == pytest.approx((600 + 150 + 20) / 2000)
    assert out["tax"] == pytest.approx(35 / 0.35 / 50)
    assert out["tang"] == pytest.approx((100 + 0.715 * 100 + 0.547 * 80 + 0.535 * 400) / 2000)


def test_cash_based_profit_subtracts_twelve_month_working_capital_increases():
    panel = profit_panel()
    panel.loc[13, ["receivables", "inventory", "deferredrev", "payables"]] = [130, 90, 25, 70]
    out = features.add_profitability_ratios(panel)
    expected = (1000 - 600 - 150) - 30 - 10 + 5 + 10
    assert out["cb_oper_prof"].iloc[13] == pytest.approx(expected / 2000)
    assert out["cb_oper_prof"].iloc[:12].isna().all()  # no row 12 months earlier


def test_profitability_exclusions():
    bank = features.add_profitability_ratios(profit_panel(sic_code=6022.0))
    assert bank[["oper_prof_rd", "cb_oper_prof", "tang"]].isna().all().all()
    assert bank["roe"].notna().all()
    loss = features.add_profitability_ratios(profit_panel(netinc=-10.0)).iloc[-1]
    assert loss["tax"] == 1.0  # pays tax on a loss
    refund = features.add_profitability_ratios(profit_panel(netinc=-10.0, taxexp=-5.0)).iloc[-1]
    assert refund["tax"] == pytest.approx(-5 / 0.35 / -10)
    no_rnd = features.add_profitability_ratios(profit_panel(rnd=np.nan)).iloc[-1]
    assert no_rnd["op_leverage"] == pytest.approx((600 + 150) / 2000)


def test_oper_prof_drops_the_smallest_third_by_market_cap():
    stocks = [profit_panel(n_months=1, permaticker=k, market_cap=100.0 * (k + 1)) for k in range(6)]
    out = features.add_profitability_ratios(pd.concat(stocks, ignore_index=True))
    out = out.set_index("permaticker")
    assert out.loc[[0, 1], "oper_prof"].isna().all()
    assert out.loc[[2, 3, 4, 5], "oper_prof"].notna().all()


def test_profitability_ratios_ignore_later_months_and_other_stocks():
    cols = ["roe", "oper_prof_rd", "cb_oper_prof", "op_leverage", "tax", "tang"]
    base = features.add_profitability_ratios(profit_panel(n_months=20))
    shocked = profit_panel(n_months=20)
    shocked.loc[shocked.index >= 15, ["receivables", "netinc", "assets"]] = [9e9, -9e9, 1.0]
    other = profit_panel(n_months=20, permaticker=2, receivables=1e12)
    out = features.add_profitability_ratios(pd.concat([shocked, other], ignore_index=True))
    out = out[out["permaticker"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(out.loc[:14, cols], base.loc[:14, cols])


def accrual_panel(n_months=14, permaticker=1, **overrides):
    """Balance sheet constant over time unless a test changes month 13."""
    base = {
        "assetsc": 600.0,
        "liabilitiesc": 300.0,
        "debtc": 50.0,
        "debtnc": 350.0,
        "ncfi": -40.0,
        "ncff": -60.0,
    }
    frame = profit_panel(n_months=n_months, permaticker=permaticker, **base)
    for col, value in overrides.items():
        frame[col] = value
    return frame


def test_accrual_signals_by_hand():
    # accounting_row: assets 2000, liabilities 1200, cash 100, debt 400, netinc 50,
    # ncfo 90, depamor 30; payout_panel: ncfdiv -20, ncfcommon -30.
    panel = accrual_panel()
    panel.loc[13, ["assetsc", "cashneq", "liabilitiesc", "debtc"]] = [700.0, 120.0, 330.0, 60.0]
    panel.loc[13, ["assets", "liabilities"]] = [2200.0, 1300.0]
    out = features.add_accrual_signals(panel).iloc[13]
    assert out["accruals"] == pytest.approx((100 - 20 - (30 - 10) - 30) / 2100)
    assert out["pct_acc"] == pytest.approx((50 - 90) / 50)
    flows = 90 - 40 - 60
    assert out["pct_tot_acc"] == pytest.approx((50 - flows + (-30) + (-20)) / 50)
    assert out["total_accruals"] == pytest.approx((50 - flows - 30 - 20) / 2000)
    assert out["noa"] == pytest.approx(((2200 - 120) - (1300 - 350)) / 2000)
    noa_now, noa_before = (2200 - 120) - (1300 - 400), (2000 - 100) - (1200 - 400)
    assert out["d_noa"] == pytest.approx((noa_now - noa_before) / 2000)


def test_accrual_signals_need_a_row_twelve_months_earlier():
    out = features.add_accrual_signals(accrual_panel())
    lagged = ["accruals", "total_accruals", "noa", "d_noa"]
    assert out[lagged].iloc[:12].isna().all().all()
    assert out[lagged].iloc[12:].notna().all().all()
    assert out[["pct_acc", "pct_tot_acc"]].notna().all().all()  # this filing only


def test_pct_acc_zero_income_and_unclassified_balance_sheets():
    zero = features.add_accrual_signals(accrual_panel(netinc=0.0)).iloc[-1]
    assert zero["pct_acc"] == pytest.approx(-90 / 1e4)
    assert np.isnan(zero["pct_tot_acc"])
    bank = features.add_accrual_signals(accrual_panel(assetsc=np.nan, debtnc=np.nan)).iloc[-1]
    assert np.isnan(bank["accruals"]) and np.isnan(bank["noa"])
    assert np.isfinite(bank["d_noa"])  # total debt only


def test_accrual_signals_ignore_later_months_and_other_stocks():
    cols = ["accruals", "pct_acc", "pct_tot_acc", "total_accruals", "noa", "d_noa"]
    base = features.add_accrual_signals(accrual_panel(n_months=20))
    shocked = accrual_panel(n_months=20)
    shocked.loc[shocked.index >= 15, ["assetsc", "netinc", "assets", "ncfo"]] = [9e9, -1, 1, 9e9]
    other = accrual_panel(n_months=20, permaticker=2, assets=1.0, assetsc=1e12)
    out = features.add_accrual_signals(pd.concat([shocked, other], ignore_index=True))
    out = out[out["permaticker"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(out.loc[:14, cols], base.loc[:14, cols])


BS_CHANGES = [
    "ch_nncoa",
    "ch_nwc",
    "del_coa",
    "del_col",
    "del_drc",
    "del_equ",
    "del_lti",
    "del_finl",
    "del_net_fin",
    "ch_inv",
    "ch_eq",
    "ch_asset_turnover",
    "invest_ppe_inv",
]


def change_panel(n_months=26, permaticker=1, **overrides):
    defaults = {"investmentsc": 50.0, "investmentsnc": 150.0, "revenue": 1e7}
    return accrual_panel(n_months=n_months, permaticker=permaticker, **(defaults | overrides))


def test_balance_sheet_changes_by_hand():
    # accounting_row: assets 2000, liabilities 1200, equity 800, cash 100, debt 400;
    # accrual_panel: assetsc 600, liabilitiesc 300, debtc 50, debtnc 350;
    # profit_panel: inventory 80, deferredrev 20, ppnenet 400.
    panel = change_panel()
    new = {
        "assets": 2200.0,
        "assetsc": 700.0,
        "cashneq": 120.0,
        "liabilities": 1300.0,
        "liabilitiesc": 330.0,
        "debtc": 60.0,
        "debtnc": 390.0,
        "debt": 450.0,
        "equity": 900.0,
        "investmentsc": 40.0,
        "investmentsnc": 180.0,
        "deferredrev": 30.0,
        "inventory": 100.0,
        "ppnenet": 450.0,
    }
    for col, value in new.items():
        panel.loc[25, col] = value
    out = features.add_balance_sheet_changes(panel).iloc[25]
    avg = 2100.0
    nncoa = lambda a, ac, inv, lia, dc, dn: ((a - ac - inv) - (lia - dc - dn)) / a  # noqa: E731
    assert out["ch_nncoa"] == pytest.approx(
        nncoa(2200, 700, 180, 1300, 60, 390) - nncoa(2000, 600, 150, 1200, 50, 350)
    )
    assert out["ch_nwc"] == pytest.approx((580 - 270) / 2200 - (500 - 250) / 2000)
    assert out["del_coa"] == pytest.approx((580 - 500) / avg)
    assert out["del_col"] == pytest.approx((270 - 250) / avg)
    assert out["del_drc"] == pytest.approx(10 / avg)
    assert out["del_equ"] == pytest.approx(100 / avg)
    assert out["del_lti"] == pytest.approx(30 / avg)
    assert out["del_finl"] == pytest.approx(50 / avg)
    assert out["del_net_fin"] == pytest.approx(((220 - 450) - (200 - 400)) / avg)
    assert out["ch_inv"] == pytest.approx(20 / avg)
    assert out["ch_eq"] == pytest.approx(900 / 800)
    assert out["invest_ppe_inv"] == pytest.approx((50 + 20) / 2000)


def test_change_in_asset_turnover_by_hand():
    # Net operating assets (assets - cash - investments) - (liabilities - debt):
    # 2000 - 100 - 200 - (1200 - 400) = 900 throughout, until revenue doubles at 25.
    panel = change_panel()
    panel.loc[25, "revenue"] = 2e7
    out = features.add_balance_sheet_changes(panel)
    assert out["ch_asset_turnover"].iloc[24] == pytest.approx(0.0)
    assert out["ch_asset_turnover"].iloc[25] == pytest.approx(2e7 / 900 - 1e7 / 900)
    assert out["ch_asset_turnover"].iloc[:24].isna().all()  # needs two years


def test_balance_sheet_change_exclusions():
    bank = features.add_balance_sheet_changes(change_panel(sic_code=6022.0))
    assert bank["del_drc"].isna().all() and bank["del_equ"].iloc[12:].notna().all()
    small = features.add_balance_sheet_changes(change_panel(revenue=1e6))
    assert small["del_drc"].isna().all()
    no_drc = features.add_balance_sheet_changes(change_panel(deferredrev=0.0))
    assert no_drc["del_drc"].isna().all()
    negative = features.add_balance_sheet_changes(change_panel(equity=-5.0))
    assert negative["ch_eq"].isna().all() and negative["del_drc"].isna().all()


def test_balance_sheet_changes_need_twelve_months_and_ignore_the_future():
    out = features.add_balance_sheet_changes(change_panel())
    assert out[BS_CHANGES].iloc[:12].isna().all().all()
    base = features.add_balance_sheet_changes(change_panel())
    shocked = change_panel()
    shocked.loc[shocked.index >= 20, ["assets", "equity", "inventory", "debt"]] = 9e9
    other = change_panel(permaticker=2, assets=1.0, inventory=1e12)
    out = features.add_balance_sheet_changes(pd.concat([shocked, other], ignore_index=True))
    out = out[out["permaticker"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(out.loc[:19, BS_CHANGES], base.loc[:19, BS_CHANGES])


GROWTH_SIGNALS = [
    "investment",
    "grcapx",
    "grcapx3y",
    "gr_sale_to_gr_inv",
    "gr_sale_to_gr_overhead",
    "surprise_rd",
]


def growth_panel(n_months=42, permaticker=1, **overrides):
    defaults = {"capex": -100.0}  # Sharadar: capex is a cash outflow
    return change_panel(n_months=n_months, permaticker=permaticker, **(defaults | overrides))


def test_investment_is_capex_intensity_over_its_own_three_year_mean():
    panel = growth_panel()
    panel.loc[41, "capex"] = -200.0
    out = features.add_investment_growth(panel)
    ratio = 100 / 1e7
    mean = (35 * ratio + 2 * ratio) / 36  # months 6..41
    assert out["investment"].iloc[41] == pytest.approx(2 * ratio / mean)
    assert out["investment"].iloc[:23].isna().all() and out["investment"].iloc[23] == 1.0
    small = features.add_investment_growth(growth_panel(revenue=5e6))
    assert small["investment"].isna().all()  # revenue under $10m


def test_capex_growth_by_hand_and_ppe_fallback():
    m = np.arange(42, dtype=float)
    out = features.add_investment_growth(growth_panel(capex=-(100 + m)))
    assert out["grcapx"].iloc[30] == pytest.approx((130 - 106) / 106)
    assert out["grcapx3y"].iloc[40] == pytest.approx(140 / ((128 + 116 + 104) / 3))
    assert out["grcapx"].iloc[:24].isna().all() and out["grcapx3y"].iloc[:36].isna().all()
    # Capex missing in month 30: replaced by the 12-month change in net PP&E.
    panel = growth_panel(capex=-(100 + m), ppnenet=400 + 5 * m)
    panel.loc[30, "capex"] = np.nan
    out = features.add_investment_growth(panel)
    assert out["grcapx"].iloc[30] == pytest.approx((60 - 106) / 106)


def test_sales_growth_gaps_vs_two_years_and_one_year_fallback():
    m = np.arange(42, dtype=float)
    out = features.add_investment_growth(growth_panel(revenue=1000 + 10 * m, inventory=80.0))
    assert out["gr_sale_to_gr_inv"].iloc[30] == pytest.approx((1300 - 1120) / 1120)
    assert out["gr_sale_to_gr_inv"].iloc[15] == pytest.approx((1150 - 1030) / 1030)
    assert out["gr_sale_to_gr_inv"].iloc[:12].isna().all()
    # Overhead is SG&A incl. R&D: 150 + 20 = 170, growing 2 a month.
    out = features.add_investment_growth(growth_panel(revenue=1000.0, sgna=150 + 2 * m))
    overhead_growth = (170 + 60 - (170 + 36 + 170 + 12) / 2) / ((170 + 36 + 170 + 12) / 2)
    assert out["gr_sale_to_gr_overhead"].iloc[30] == pytest.approx(-overhead_growth)


def test_surprise_rd():
    panel = growth_panel(n_months=14)
    panel.loc[13, "rnd"] = 22.0  # +10% with assets unchanged
    out = features.add_investment_growth(panel)
    assert out["surprise_rd"].iloc[13] == 1.0 and out["surprise_rd"].iloc[12] == 0.0
    assert out["surprise_rd"].iloc[:12].isna().all()
    panel.loc[13, "assets"] = 2400.0  # R&D per dollar of assets falls: no surprise
    assert features.add_investment_growth(panel)["surprise_rd"].iloc[13] == 0.0
    no_rnd = features.add_investment_growth(growth_panel(n_months=14, rnd=np.nan))
    assert no_rnd["surprise_rd"].isna().all()


def test_investment_growth_ignores_later_months_and_other_stocks():
    m = np.arange(42, dtype=float)
    base = features.add_investment_growth(growth_panel(capex=-(100 + m), revenue=1e7 + m))
    shocked = growth_panel(capex=-(100 + m), revenue=1e7 + m)
    shocked.loc[shocked.index >= 38, ["capex", "revenue", "rnd", "inventory"]] = [-9e9, 1, 9e9, 0]
    other = growth_panel(permaticker=2, capex=-1e12, revenue=1.0)
    out = features.add_investment_growth(pd.concat([shocked, other], ignore_index=True))
    out = out[out["permaticker"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(out.loc[:37, GROWTH_SIGNALS], base.loc[:37, GROWTH_SIGNALS])


FINANCING_SIGNALS = [
    "composite_debt_issuance",
    "debt_issuance",
    "net_debt_finance",
    "net_equity_finance",
    "xfin",
    "share_iss_1y",
    "share_iss_5y",
    "share_repurchase",
]


def financing_panel(n_months=70, permaticker=1, **overrides):
    defaults = {"ncfdebt": 40.0, "sharesbas": 100.0, "sharefactor": 1.0}
    return growth_panel(n_months=n_months, permaticker=permaticker, **(defaults | overrides))


def test_financing_signals_by_hand():
    # accounting_row: assets 2000, debt 400, equity 800, market cap 1000 (+10/month);
    # payout_panel: ncfdiv -20, ncfcommon -30 (a net buyback).
    m = np.arange(70, dtype=float)
    out = features.add_financing_signals(financing_panel(debt=400 + m, sharesbas=100 + m)).iloc[69]
    assert out["composite_debt_issuance"] == pytest.approx(np.log(469 / 409))
    assert out["debt_issuance"] == 1.0 and out["share_repurchase"] == 1.0
    assert out["net_debt_finance"] == pytest.approx(40 / 2000)
    assert out["net_equity_finance"] == pytest.approx((-30 - 20) / 2000)
    assert out["xfin"] == pytest.approx((-30 - 20 + 40) / 2000)
    assert out["share_iss_1y"] == pytest.approx((163 - 151) / 151)  # months 63 vs 51
    assert out["share_iss_5y"] == pytest.approx((164 - 104) / 104)  # months 64 vs 4


def test_financing_signals_need_their_history():
    out = features.add_financing_signals(financing_panel())
    assert out["composite_debt_issuance"].iloc[:60].isna().all()
    assert out["share_iss_1y"].iloc[:18].isna().all() and out["share_iss_1y"].iloc[18] == 0
    assert out["share_iss_5y"].iloc[:65].isna().all() and out["share_iss_5y"].iloc[65] == 0
    assert out["net_debt_finance"].iloc[:12].isna().all()


def test_financing_exclusions():
    issuer = features.add_financing_signals(financing_panel(ncfcommon=50.0, ncfdebt=-10.0))
    assert (issuer["share_repurchase"] == 0).all() and (issuer["debt_issuance"] == 0).all()
    huge = features.add_financing_signals(financing_panel(ncfdebt=5000.0))
    assert huge["net_debt_finance"].isna().all()  # |flow| > avg assets
    no_book = features.add_financing_signals(financing_panel(equity=-1.0))
    assert no_book["debt_issuance"].isna().all()
    no_cash_flow = features.add_financing_signals(financing_panel(ncfcommon=np.nan))
    assert no_cash_flow["share_repurchase"].isna().all()


def test_financing_signals_ignore_later_months_and_other_stocks():
    m = np.arange(70, dtype=float)
    base = features.add_financing_signals(financing_panel(debt=400 + m, sharesbas=100 + m))
    shocked = financing_panel(debt=400 + m, sharesbas=100 + m)
    shocked.loc[shocked.index >= 66, ["debt", "sharesbas", "ncfdebt", "assets"]] = [1, 9e9, 9e9, 1]
    other = financing_panel(permaticker=2, debt=1e12, sharesbas=1.0)
    out = features.add_financing_signals(pd.concat([shocked, other], ignore_index=True))
    out = out[out["permaticker"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(out.loc[:65, FINANCING_SIGNALS], base.loc[:65, FINANCING_SIGNALS])


TURNOVER_SIGNALS = [
    "share_vol",
    "vol_mkt",
    "std_turn",
    "zerotrade_1m",
    "zerotrade_6m",
    "zerotrade_12m",
    "mom_vol",
]


def turnover_panel(turnover=0.1, n_months=40, permaticker=1, market_cap=1e8, **overrides):
    """1,000 shares outstanding, so share_volume = 1,000 x monthly turnover."""
    panel = daily_summary_panel(n_months=n_months, permaticker=permaticker)
    columns = {
        "share_volume": 1000.0 * np.asarray(turnover, dtype=float),
        "sharesbas": 500.0,
        "sharefactor": 2.0,
        "market_cap": market_cap,
        "n_zero_volume": 0.0,
    }
    return panel.assign(**(columns | overrides))


def test_share_vol_buckets_three_month_turnover():
    turnover = np.r_[[0.03] * 10, [0.12] * 10, [0.07] * 20]
    out = features.add_turnover_signals(turnover_panel(turnover))["share_vol"]
    assert out.iloc[:2].isna().all()  # needs 3 months
    assert out.iloc[2] == 0.0 and out.iloc[19] == 1.0  # 3% and 12% a month
    assert np.isnan(out.iloc[11])  # (3 + 12 + 12) / 3 = 9%: between the cut-offs
    assert np.isnan(out.iloc[30])  # 7%


def test_vol_mkt_is_mean_dollar_volume_over_market_cap():
    out = features.add_turnover_signals(turnover_panel())["vol_mkt"]
    # dollar_volume = 1e6 x (i + 1); months 19..30 hold 20..31, averaging 25.5.
    assert out.iloc[30] == pytest.approx(25.5e6 / 1e8)
    assert out.iloc[:9].isna().all() and out.iloc[9] == pytest.approx(5.5e6 / 1e8)


def test_std_turn_by_hand_and_drops_large_caps():
    turnover = np.tile([0.1, 0.3], 20)
    stocks = [turnover_panel(turnover, permaticker=k, market_cap=1e8 * k) for k in range(1, 6)]
    out = features.add_turnover_signals(pd.concat(stocks, ignore_index=True))
    first = out[out["permaticker"] == 1]["std_turn"]
    assert first.iloc[:23].isna().all()  # needs 24 months
    assert first.iloc[35] == pytest.approx(np.std([0.1, 0.3] * 18, ddof=1))
    large = out[out["permaticker"] >= 4]["std_turn"]
    assert large.isna().all()  # the two largest fifths by market cap


def test_zerotrade_counts_zero_volume_days_one_month_back():
    out = features.add_turnover_signals(turnover_panel(n_zero_volume=2.0))
    # 20 trading days a month, 2 without volume, turnover 10% a month.
    assert out["zerotrade_1m"].iloc[0:1].isna().all()  # one month back
    assert out["zerotrade_1m"].iloc[5] == pytest.approx((2 + 10 / 480_000) * 21 / 20)
    assert out["zerotrade_6m"].iloc[:6].isna().all()
    six = (12 + (1 / 0.6) / 11_000) * (21 * 6 / 120)
    assert out["zerotrade_6m"].iloc[6] == pytest.approx(six)
    assert out["zerotrade_12m"].iloc[:12].isna().all() and out["zerotrade_12m"].iloc[12] > 0
    silent = turnover_panel(np.r_[[0.1] * 10, [0.0] * 30])
    assert features.add_turnover_signals(silent)["zerotrade_1m"].iloc[11:].isna().all()


def test_mom_vol_is_momentum_decile_among_high_turnover_stocks():
    # 30 stocks: stock k earns 0.001 x k a month and turns over k% a month.
    stocks = [turnover_panel(k / 100, permaticker=k).assign(ret=0.001 * k) for k in range(1, 31)]
    out = features.add_turnover_signals(pd.concat(stocks, ignore_index=True))
    at = out[out["month_end"] == out["month_end"].iloc[30]].set_index("permaticker")["mom_vol"]
    assert at.loc[30] == 10.0 and at.loc[21] == 7.0  # momentum rank 21 of 30
    assert at.loc[:20].isna().all()  # bottom two thirds by turnover
    assert out.groupby("permaticker")["mom_vol"].apply(lambda m: m.iloc[:23].isna().all()).all()


def test_turnover_signals_ignore_later_months_and_other_stocks():
    turnover = np.linspace(0.02, 0.2, 40)
    base = features.add_turnover_signals(turnover_panel(turnover, n_zero_volume=1.0))
    shocked = turnover_panel(turnover, n_zero_volume=1.0)
    cols = ["share_volume", "dollar_volume", "n_zero_volume", "ret"]
    shocked.loc[shocked.index >= 36, cols] = [9e9, 9e9, 20.0, 5.0]
    other = turnover_panel(5.0, permaticker=2, market_cap=1.0)
    out = features.add_turnover_signals(pd.concat([shocked, other], ignore_index=True))
    out = out[out["permaticker"] == 1].reset_index(drop=True)
    # Cross-sectional ranks (std_turn size cut, mom_vol) depend on the other stock.
    own = ["share_vol", "vol_mkt", "zerotrade_1m", "zerotrade_6m", "zerotrade_12m"]
    pd.testing.assert_frame_equal(out.loc[:35, own], base.loc[:35, own])


QUARTERLY_SIGNALS = ["roaq", "cash", "ch_tax", "num_earn_increase"]


def quarterly_panel(earnings, permaticker=1, **overrides):
    """One quarterly filing per 3 months: quarter j's values on months 3j..3j+2."""
    quarterly = np.repeat(np.asarray(earnings, dtype=float), 3)
    n = len(quarterly)
    columns = {
        "permaticker": permaticker,
        "month_end": pd.date_range("2015-01-31", periods=n, freq="ME"),
        "netinc_q": quarterly,
        "assets_q": 1000.0 + 10 * np.arange(n),
        "cashneq_q": 100.0,
        "taxexp_q": 5.0,
        "assets": 1000.0,
        "filing_date_q": pd.Timestamp("2014-12-15"),
    }
    return pd.DataFrame(columns | overrides)


def test_quarterly_ratios_by_hand():
    out = features.add_quarterly_signals(
        quarterly_panel([10.0] * 8, taxexp_q=np.repeat([5.0] * 4 + [8.0] * 4, 3))
    )
    assert out["roaq"].iloc[20] == pytest.approx(10 / 1170)  # assets 3 months earlier
    assert out["cash"].iloc[20] == pytest.approx(100 / 1200)
    assert out["ch_tax"].iloc[12] == pytest.approx((8 - 5) / 1000)
    assert out["roaq"].iloc[:3].isna().all() and out["ch_tax"].iloc[:12].isna().all()
    assert (
        features.add_quarterly_signals(quarterly_panel([1.0] * 4, assets_q=0.0))["cash"]
        .isna()
        .all()
    )


@pytest.mark.parametrize(
    "earnings, expected",
    [
        # vs the same quarter a year earlier (4 quarters back): rises in the last 3.
        ([5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6], 3),
        ([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4], 0),  # latest change < 0
        ([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6], 1),
        # 8 rises then the change 8 quarters back <= 0: 8.
        ([1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9], 8),
        # Rising for more than 8 quarters: OSAP's 0, not 8.
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 0),
    ],
)
def test_num_earn_increase_counts_consecutive_increases(earnings, expected):
    out = features.add_quarterly_signals(quarterly_panel(earnings))
    assert out["num_earn_increase"].iloc[-1] == expected


def test_num_earn_increase_treats_a_missing_change_as_an_increase():
    earnings = [5.0] * 8 + [np.nan, 6.0, 6.0]
    out = features.add_quarterly_signals(quarterly_panel(earnings))
    # Changes: 6-5 > 0, 6-5 > 0, NaN (counts as up), 5-5 = 0 -> 3.
    assert out["num_earn_increase"].iloc[-1] == 3
    none = features.add_quarterly_signals(quarterly_panel([5.0] * 4, filing_date_q=pd.NaT))
    assert none["num_earn_increase"].isna().all()


def test_quarterly_signals_ignore_later_months_and_other_stocks():
    earnings = np.arange(1.0, 13.0)
    base = features.add_quarterly_signals(quarterly_panel(earnings))
    shocked = quarterly_panel(earnings)
    cols = ["netinc_q", "assets_q", "cashneq_q", "taxexp_q", "assets"]
    shocked.loc[shocked.index >= 30, cols] = [-9e9, 1.0, 9e9, 9e9, 1.0]
    other = quarterly_panel(-earnings, permaticker=2, assets_q=1.0)
    out = features.add_quarterly_signals(pd.concat([shocked, other], ignore_index=True))
    out = out[out["permaticker"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(out.loc[:29, QUARTERLY_SIGNALS], base.loc[:29, QUARTERLY_SIGNALS])


def ols_by_hand(y, x):
    """Intercept + slopes by numpy least squares, the reference for _rolling_ols."""
    design = np.column_stack([np.ones(len(y)), x])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def test_rolling_ols_matches_least_squares_over_valid_months():
    rng = np.random.default_rng(0)
    months = pd.date_range("2020-01-31", periods=30, freq="ME")
    factors = pd.DataFrame(rng.normal(0, 0.05, (30, 2)), index=months, columns=["f1", "f2"])
    y = pd.DataFrame({7: rng.normal(0, 0.1, 30)}, index=months)
    y.iloc[25, 0] = np.nan  # a missing stock month
    factors.iloc[27, 1] = np.nan  # a missing factor month
    out = features._rolling_ols(y, factors, window=12, min_periods=8)

    t = 29  # window = months 18..29, minus 25 and 27
    keep = [i for i in range(18, 30) if i not in (25, 27)]
    expected = ols_by_hand(y.iloc[keep, 0].to_numpy(), factors.iloc[keep].to_numpy())
    got = [out[name].iloc[t, 0] for name in ["const", "f1", "f2"]]
    assert got == pytest.approx(expected)


def test_rolling_ols_needs_min_periods_and_a_varying_factor():
    months = pd.date_range("2020-01-31", periods=12, freq="ME")
    y = pd.DataFrame({1: np.linspace(0, 0.1, 12)}, index=months)
    moving = pd.DataFrame({"f": np.linspace(-0.05, 0.05, 12)}, index=months)
    out = features._rolling_ols(y, moving, window=12, min_periods=10)
    assert out["f"].iloc[:9, 0].isna().all()
    assert out["f"].iloc[9:, 0].notna().all()

    flat = pd.DataFrame({"f": 0.02}, index=months)  # no variation: slope undefined
    assert features._rolling_ols(y, flat, window=12, min_periods=10)["f"].isna().all().all()


def factor_panel(n_months=70, n_stocks=4, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    panel = pd.concat(
        [panel_for(s, rng.normal(0.01, 0.08, n_months), start=months[0]) for s in range(n_stocks)],
        ignore_index=True,
    )
    factors = pd.DataFrame(
        rng.normal(0, 0.04, (n_months, 4)),
        index=months,
        columns=["mktrf", "smb", "hml", "ps_innov"],
    ).assign(rf=0.002)
    return panel, factors


def test_beta_is_slope_on_equal_weighted_market_over_60_months():
    panel, factors = factor_panel()
    out = features.add_factor_betas(panel, factors)
    ew = panel.groupby("month_end")["ret"].mean() - 0.002
    stock = panel[panel["permaticker"] == 1].set_index("month_end")["ret"] - 0.002
    window = stock.index[-60:]
    expected = ols_by_hand(stock[window].to_numpy(), ew[window].to_numpy())[1]
    got = out[out["permaticker"] == 1]["beta"].iloc[-1]
    assert got == pytest.approx(expected)
    # 20 months with a return are needed: the 20th month is the first with a beta.
    first = out[out["permaticker"] == 1]["beta"]
    assert first.iloc[:19].isna().all() and first.iloc[19:].notna().all()


def test_beta_liquidity_ps_controls_for_ff3_and_skips_missing_liquidity():
    panel, factors = factor_panel()
    factors.loc[factors.index[-5:], "ps_innov"] = np.nan  # the file stops before the panel
    out = features.add_factor_betas(panel, factors)
    stock = panel[panel["permaticker"] == 2].set_index("month_end")["ret"] - 0.002
    window = stock.index[-60:-5]  # the last 5 months have no liquidity value
    x = factors.loc[window, ["ps_innov", "mktrf", "smb", "hml"]].to_numpy()
    expected = ols_by_hand(stock[window].to_numpy(), x)[1]
    got = out[out["permaticker"] == 2]["beta_liquidity_ps"].iloc[-1]
    assert got == pytest.approx(expected)


def test_factor_regressions_ignore_the_future():
    # Leakage test: later returns, the target and later factor values must not move month t.
    panel, factors = factor_panel()
    t = panel["month_end"].unique()[50]

    def build(p, f):
        return features.add_residual_momentum_coskewness(features.add_factor_betas(p, f), f)

    before = build(panel, factors)
    changed = panel.copy()
    later = changed["month_end"] > t
    changed.loc[later, "ret"] = 0.9
    changed["fwd_ret"] = 0.9
    later_factors = factors.copy()
    later_factors.loc[factors.index > t] = 0.3
    after = build(changed, later_factors)
    at_t = before["month_end"] == t
    for col in ["beta", "beta_liquidity_ps", "residual_momentum", "coskewness"]:
        assert after.loc[at_t, col].to_numpy() == pytest.approx(before.loc[at_t, col].to_numpy())


def test_residual_momentum_by_hand():
    panel, factors = factor_panel(n_months=60)
    out = features.add_residual_momentum_coskewness(panel, factors)
    y = panel[panel["permaticker"] == 3]["ret"].to_numpy() - 0.002
    x = factors[["mktrf", "smb", "hml"]].to_numpy()

    def residual(s):  # residual of month s in the 36-month fit ending at s
        window = slice(s - 35, s + 1)
        b = ols_by_hand(y[window], x[window])
        return y[s] - b[0] - x[s] @ b[1:]

    t = 59
    resid = np.array([residual(s) for s in range(t - 11, t)])  # months t-11 .. t-1
    expected = resid.mean() / resid.std(ddof=1)
    got = out[out["permaticker"] == 3]["residual_momentum"]
    assert got.iloc[t] == pytest.approx(expected)
    # The first residual is at month 35, so the first signal needs months 35..45: t = 46.
    assert got.iloc[:46].isna().all() and got.iloc[46:].notna().all()


def test_coskewness_by_hand_with_a_missing_month():
    panel, factors = factor_panel(n_months=70)
    panel.loc[(panel["permaticker"] == 0) & (panel["month_end"] == factors.index[65]), "ret"] = (
        np.nan
    )
    out = features.add_residual_momentum_coskewness(panel, factors)
    r = panel[panel["permaticker"] == 0]["ret"].to_numpy() - 0.002
    m = factors["mktrf"].to_numpy()
    keep = [i for i in range(10, 70) if i != 65]  # last 60 months, minus the missing one
    rd, md = r[keep] - r[keep].mean(), m[keep] - m[keep].mean()
    expected = (rd * md**2).mean() / (np.sqrt((rd**2).mean()) * (md**2).mean())
    got = out[out["permaticker"] == 0]["coskewness"]
    assert got.iloc[69] == pytest.approx(expected)
    assert got.iloc[:11].isna().all() and got.iloc[11:].notna().all()  # >= 12 months


def daily_factor_panel(months=14, seed=0):
    """One stock's daily excess returns and factors, and its panel of monthly summaries."""
    from alpha_survival.data import real

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=months * 21 + 30)
    dates = dates[dates < pd.Timestamp("2023-01-01") + pd.offsets.MonthEnd(months)]
    factors = pd.DataFrame(
        rng.normal(0, 0.01, (len(dates), 4)), index=dates, columns=["mktrf", "smb", "hml", "dvix"]
    ).assign(rf=0.0001)
    factors.index.name = "date"
    daily = pd.DataFrame({"permaticker": 1, "date": dates, "month": dates.to_period("M")}).assign(
        dret=0.5 * factors["mktrf"].to_numpy() + rng.normal(0, 0.02, len(dates))
    )
    sums = real.daily_factor_summaries(daily, factors).reset_index()
    sums["month_end"] = sums["month"].dt.to_timestamp(how="end").dt.normalize()
    return sums.drop(columns="month"), daily.join(factors, on="date")


def test_daily_factor_signals_by_hand():
    panel, d = daily_factor_panel()
    out = features.add_daily_factor_signals(panel)
    d["y"] = d["dret"] - d["rf"]
    last = d[d["month"] == d["month"].max()]
    design = np.column_stack([np.ones(len(last)), last[["mktrf", "smb", "hml"]]])
    resid = last["y"] - design @ np.linalg.lstsq(design, last["y"], rcond=None)[0]
    row = out.iloc[-1]
    assert row["idio_vol_3f"] == pytest.approx(resid.std(ddof=1))
    assert row["return_skew_3f"] == pytest.approx((resid**3).mean() / (resid**2).mean() ** 1.5)
    assert row["realized_vol"] == pytest.approx(last["y"].std(ddof=1))
    vix = np.column_stack([np.ones(len(last)), last[["mktrf", "dvix"]]])
    assert row["beta_vix"] == pytest.approx(np.linalg.lstsq(vix, last["y"], rcond=None)[0][2])

    year = d[d["month"] > d["month"].max() - 12]  # the last 12 calendar months
    capm = np.column_stack([np.ones(len(year)), year["mktrf"]])
    e = year["y"] - capm @ np.linalg.lstsq(capm, year["y"], rcond=None)[0]
    assert row["idio_vol_aht"] == pytest.approx(np.sqrt((e**2).sum() / (len(year) - 2)))


def test_daily_factor_signals_need_enough_days():
    panel, _ = daily_factor_panel()
    panel.loc[0, ["xret_n", "vix_n"]] = 14  # fewer than 15 days in the first month
    out = features.add_daily_factor_signals(panel)
    first = out.iloc[0]
    cols = ["idio_vol_3f", "return_skew_3f", "realized_vol", "beta_vix"]
    assert first[cols].isna().all() and out.iloc[1][cols].notna().all()
    # IdioVolAHT needs >= 100 days: about 5 months of ~21 trading days.
    aht = out["idio_vol_aht"]
    days = panel["xret_n"].cumsum()
    assert aht[days < 100].isna().all() and aht[days >= 100].notna().all()


def test_daily_factor_signals_ignore_later_months():
    # Leakage test: wild later months cannot move month t (IdioVolAHT rolls over 12).
    panel, _ = daily_factor_panel()
    t = 9
    wild = panel.copy()
    sums = [c for c in wild.columns if c not in ("permaticker", "month_end", "xret_n", "vix_n")]
    wild.loc[t + 1 :, sums] = 7.0
    cols = ["idio_vol_3f", "return_skew_3f", "realized_vol", "idio_vol_aht", "beta_vix"]
    a = features.add_daily_factor_signals(panel).loc[t, cols]
    b = features.add_daily_factor_signals(wild).loc[t, cols]
    pd.testing.assert_series_equal(a, b)


def logs(d):
    x = np.log1p(d["dret"]) - np.log1p(d["rf"])
    m = np.log1p(d["mktrf"] + d["rf"]) - np.log1p(d["rf"])
    return x, m


def test_coskew_acx_and_beta_fp_by_hand():
    panel, d = daily_factor_panel(months=62)
    out = features.add_coskew_acx_beta_fp(panel)
    x, m = logs(d)
    last = d["month"].max()
    row = out.iloc[-1]

    year = (d["month"] > last - 12).to_numpy()
    xd, md = x[year] - x[year].mean(), m[year] - m[year].mean()
    expected = (xd * md**2).mean() / (np.sqrt((xd**2).mean()) * (md**2).mean())
    assert row["coskew_acx"] == pytest.approx(expected)

    five = (d["month"] > last - 60).to_numpy()
    x3, m3 = x.rolling(3).sum(), m.rolling(3).sum()
    corr = np.corrcoef(x3[five], m3[five])[0, 1]
    assert row["beta_fp"] == pytest.approx(abs(corr) * x[year].std(ddof=1) / m[year].std(ddof=1))


def test_beta_fp_needs_750_three_day_returns_and_acx_a_full_year():
    panel, _ = daily_factor_panel(months=62)
    out = features.add_coskew_acx_beta_fp(panel)
    three_day = panel["l3_n"].rolling(60, min_periods=1).sum()
    assert out["beta_fp"][three_day < 750].isna().all()
    assert out["beta_fp"][three_day >= 750].notna().all()

    # A second stock missing 6 days in its latest month is short of a full year.
    short = panel.assign(permaticker=2)
    short.loc[short.index[-1], ["xret_n"]] -= 6
    out = features.add_coskew_acx_beta_fp(pd.concat([panel, short], ignore_index=True))
    latest = out[out["month_end"] == panel["month_end"].max()].set_index("permaticker")
    assert np.isfinite(latest.loc[1, "coskew_acx"]) and np.isnan(latest.loc[2, "coskew_acx"])


def test_coskew_acx_and_beta_fp_ignore_later_months():
    # Leakage test: wild later months cannot move month t.
    panel, _ = daily_factor_panel(months=62)
    t = 55
    wild = panel.copy()
    sums = [c for c in wild.columns if c.startswith(("lx", "lm", "l3x", "l3m"))]
    wild.loc[t + 1 :, sums] = 3.0
    cols = ["coskew_acx", "beta_fp"]
    a = features.add_coskew_acx_beta_fp(panel).loc[t, cols]
    b = features.add_coskew_acx_beta_fp(wild).loc[t, cols]
    assert a.notna().all()
    pd.testing.assert_series_equal(a, b)


def tail_panel(n_months=130, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    tail = pd.Series(rng.uniform(0.3, 0.6, n_months), index=months)
    rets = 0.2 * tail.to_numpy() + rng.normal(0, 0.05, n_months)
    panel = panel_for(1, rets, start=months[0])
    return panel.assign(market_tail=panel["month_end"].map(tail)), tail


def test_beta_tail_risk_by_hand_over_120_months():
    panel, tail = tail_panel()
    out = features.add_tail_risk_beta(panel)
    window = slice(10, 130)  # the last 120 months
    expected = ols_by_hand(panel["ret"].to_numpy()[window], tail.to_numpy()[window])[1]
    assert out["beta_tail_risk"].iloc[-1] == pytest.approx(expected)
    assert out["beta_tail_risk"].iloc[:71].isna().all()  # >= 72 months
    assert out["beta_tail_risk"].iloc[71:].notna().all()


def test_beta_tail_risk_ignores_later_months():
    # Leakage test: later returns and later tail values cannot move month t.
    panel, _ = tail_panel()
    t = 100
    wild = panel.copy()
    wild.loc[t + 1 :, ["ret", "market_tail"]] = 9.0
    wild["fwd_ret"] = 9.0
    a = features.add_tail_risk_beta(panel).loc[t, "beta_tail_risk"]
    b = features.add_tail_risk_beta(wild).loc[t, "beta_tail_risk"]
    assert a == pytest.approx(b)


def delay_panel(end="2024-08-31", wild_after=None, drop=None, seed=0):
    """Monthly panel built from simulated daily data by data.real.daily_factor_summaries:
    one stock that reacts to the market partly a day late. `wild_after`: a date after
    which returns and market go wild; `drop`: a date range the stock does not trade."""
    from alpha_survival.data import real

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-07-01", end)
    mkt = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    ret = 0.6 * mkt + 0.4 * mkt.shift(1).fillna(0) + rng.normal(0, 0.01, len(dates))
    if wild_after is not None:
        mkt[mkt.index > wild_after] = 0.5
        ret[ret.index > wild_after] = 9.0
    factors = pd.DataFrame({"mktrf": mkt, "smb": 0.001, "hml": -0.001, "dvix": 0.0, "rf": 0.0})
    factors["smb"] = rng.normal(0, 0.005, len(dates))  # vary, so the FF3 fit is solvable
    factors["hml"] = rng.normal(0, 0.005, len(dates))
    daily = pd.DataFrame({"permaticker": 1, "date": dates, "dret": ret.to_numpy()})
    if drop is not None:
        daily = daily[~daily["date"].between(*drop)]
    daily["month"] = daily["date"].dt.to_period("M")
    sums = real.daily_factor_summaries(daily, factors).reset_index()
    sums["month_end"] = sums.pop("month").dt.to_timestamp(how="end").dt.normalize()
    return sums, factors, daily


def price_delay_by_hand(factors, daily, first, last):
    """D1, D2, D3 from a least-squares fit on the stock's days in [first, last]."""
    m = factors["mktrf"]
    x = pd.concat([m, *(m.shift(n) for n in range(1, 5))], axis=1).loc[daily["date"]]
    use = daily["date"].between(first, last).to_numpy() & x.notna().all(axis=1).to_numpy()
    design = np.column_stack([np.ones(use.sum()), x.to_numpy()[use]])
    y = daily["dret"].to_numpy()[use]
    b, ssr = np.linalg.lstsq(design, y, rcond=None)[:2]
    se = np.sqrt(ssr[0] / (len(y) - 6) * np.diag(np.linalg.inv(design.T @ design)))
    t = b / se
    sst = ((y - y.mean()) ** 2).sum()
    short = np.linalg.lstsq(design[:, :2], y, rcond=None)[1][0]
    d1 = 1 - (1 - short / sst) / (1 - ssr[0] / sst)
    w = np.arange(1, 5)
    return d1, b[2:] @ w / (b[1] + b[2:].sum()), t[2:] @ w / (t[1] + t[2:].sum())


DELAY_COLS = ["price_delay_rsq", "price_delay_slope", "price_delay_tstat"]


def test_price_delay_by_hand_july_to_june_used_from_july():
    sums, factors, daily = delay_panel()
    out = features.add_price_delay(sums).set_index("month_end")[DELAY_COLS]
    fit23 = price_delay_by_hand(factors, daily, "2022-07-01", "2023-06-30")
    fit24 = price_delay_by_hand(factors, daily, "2023-07-01", "2024-06-30")
    assert out.loc[:"2023-06-30"].isna().all().all()  # nothing before the first July
    for month in ["2023-07-31", "2024-06-30"]:  # July and the 11 months after
        assert out.loc[month].to_numpy() == pytest.approx(fit23)
    assert out.loc["2024-07-31"].to_numpy() == pytest.approx(fit24)
    assert out.loc["2023-07-31", "price_delay_slope"] > 0  # the stock does react late


def test_price_delay_needs_26_days_and_a_june_trade():
    # No trades from 1 May to 30 June 2023: the 2023 fit has days but none in June.
    sums, _, _ = delay_panel(drop=("2023-05-01", "2023-06-30"))
    out = features.add_price_delay(sums).set_index("month_end")[DELAY_COLS]
    assert out.loc["2023-07-31":"2024-06-30"].isna().all().all()
    assert out.loc["2024-07-31"].notna().all()
    # Trading only from 6 June 2023: under 26 days in the 2023 window.
    sums, _, _ = delay_panel(drop=("2022-07-01", "2023-06-05"))
    out = features.add_price_delay(sums).set_index("month_end")[DELAY_COLS]
    assert out.loc["2023-07-31"].isna().all()


def test_price_delay_ignores_days_after_the_june_fit():
    # Leakage test: wild returns and market after June 2023 cannot move the values
    # used from July 2023 to June 2024.
    a, _, _ = delay_panel()
    b, _, _ = delay_panel(wild_after="2023-06-30")
    rows = slice("2023-07-31", "2024-06-30")
    got_a = features.add_price_delay(a).set_index("month_end").loc[rows, DELAY_COLS]
    got_b = features.add_price_delay(b).set_index("month_end").loc[rows, DELAY_COLS]
    assert got_a.notna().all().all()
    pd.testing.assert_frame_equal(got_a, got_b)


def industry_panel(n_months=7, seed=0):
    """Four stocks over `n_months`: 1-3 in SIC 35xx / FF48 "Computers", 4 in banking."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    rows = [(s, m) for s in [1, 2, 3, 4] for m in months]
    panel = pd.DataFrame(rows, columns=["permaticker", "month_end"])
    panel["ret"] = rng.normal(0.01, 0.05, len(panel))
    panel["market_cap"] = rng.uniform(1e8, 1e9, len(panel))
    panel["sic_code"] = panel["permaticker"].map({1: 3571.0, 2: 3572.0, 3: 3579.0, 4: 6022.0})
    panel["fama_industry"] = np.where(panel["permaticker"] == 4, "Banking", "Computers")
    return panel


def test_ind_mom_is_cap_weighted_six_month_momentum_of_the_sic2_industry():
    panel = industry_panel()
    panel.loc[(panel["permaticker"] == 2) & (panel["month_end"] == "2020-03-31"), "ret"] = np.nan
    out = features.add_industry_signals(panel)
    t = pd.Timestamp("2020-07-31")
    moms, caps = [], []
    for s in [1, 2, 3]:  # stock 4 is a bank: another industry
        rets = panel.loc[panel["permaticker"] == s, "ret"].fillna(0).to_numpy()
        moms.append(np.prod(1 + rets[1:6]) - 1)  # months t-5 .. t-1, a missing return as 0
        caps.append(
            panel.loc[(panel["permaticker"] == s) & (panel["month_end"] == t), "market_cap"]
        )
    expected = np.average(moms, weights=np.concatenate(caps))
    got = out.loc[out["month_end"] == t].set_index("permaticker")["ind_mom"]
    assert got.loc[[1, 2, 3]].to_numpy() == pytest.approx([expected] * 3)
    assert out.loc[out["month_end"] < "2020-06-30", "ind_mom"].isna().all()  # 5 lags needed


def test_ind_ret_big_is_the_mean_return_of_the_top_30_percent_for_the_rest():
    month = pd.Timestamp("2020-01-31")
    panel = pd.DataFrame(
        {
            "permaticker": range(1, 12),
            "month_end": month,
            "ret": np.arange(1, 12) / 100,
            "market_cap": [*range(1, 11), 99.0],  # 1-10 in Computers, 11 a bank
            "sic_code": [3571.0] * 10 + [6022.0],
            "fama_industry": ["Computers"] * 10 + ["Banking"],
        }
    )
    out = features.add_industry_signals(panel).set_index("permaticker")["ind_ret_big"]
    big = np.mean([0.08, 0.09, 0.10])  # ranks 0.8, 0.9, 1.0
    assert out.loc[1:6].to_numpy() == pytest.approx([big] * 6)
    assert out.loc[7:10].isna().all()  # rank 0.7 and above: big, or on the edge
    assert np.isnan(out.loc[11])  # the only bank is its industry's biggest


def test_industry_signals_ignore_later_months():
    # Leakage test: later returns and market caps cannot move month t.
    panel = industry_panel(n_months=9)
    t = pd.Timestamp("2020-07-31")
    wild = panel.copy()
    later = wild["month_end"] > t
    wild.loc[later, "ret"] = 9.0
    wild.loc[later, "market_cap"] = 1.0
    cols = ["ind_mom", "ind_ret_big"]
    a = features.add_industry_signals(panel).loc[lambda d: d["month_end"] == t, cols]
    b = features.add_industry_signals(wild).loc[lambda d: d["month_end"] == t, cols]
    assert a["ind_mom"].notna().all()
    pd.testing.assert_frame_equal(a, b)


def concentration_panel(n_months=25, seed=0):
    """Stocks 1-2 in SIC 3571, 3 in 3572 (same 2-digit industry), 4 a utility (4911)."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    panel = pd.DataFrame([(s, m) for s in [1, 2, 3, 4] for m in months])
    panel.columns = ["permaticker", "month_end"]
    for col in ["revenue", "assets", "equity", "ppnenet"]:
        panel[col] = rng.uniform(10, 100, len(panel))
    panel["liabilities"] = panel["assets"] - panel["equity"]
    panel["taxliabilities"] = rng.uniform(0, 5, len(panel))
    panel["capex"] = -rng.uniform(1, 10, len(panel))
    panel["fxusd"] = 1.0
    panel["sic_code"] = panel["permaticker"].map({1: 3571.0, 2: 3571.0, 3: 3572.0, 4: 4911.0})
    return panel


def test_herf_is_own_mean_of_the_industry_sales_herfindahl():
    panel = concentration_panel(n_months=14)
    panel.loc[(panel["permaticker"] == 1) & (panel["month_end"] == "2020-03-31"), "fxusd"] = 2.0
    out = features.add_industry_accounting(panel)
    usd = panel["revenue"] / panel["fxusd"]
    pair = panel["permaticker"].isin([1, 2])
    share = usd[pair] / usd[pair].groupby(panel["month_end"][pair]).transform("sum")
    monthly = (share**2).groupby(panel["month_end"][pair]).sum().to_numpy()
    got = out.set_index(["permaticker", "month_end"])["herf"]
    assert got.loc[2].iloc[11] == pytest.approx(monthly[:12].mean())  # 12 months: enough
    assert got.loc[2].iloc[13] == pytest.approx(monthly[:14].mean())
    assert got.loc[2].iloc[:11].isna().all()
    assert got.loc[3].iloc[11] == pytest.approx(1.0)  # alone in SIC 3572
    assert got.loc[4].isna().all()  # utilities excluded


def test_herf_be_uses_book_equity_plus_tax_liabilities():
    panel = concentration_panel(n_months=12)
    panel.loc[panel["permaticker"] == 2, "equity"] = np.nan  # falls back to assets - liabilities
    out = features.add_industry_accounting(panel).set_index(["permaticker", "month_end"])
    book = panel["assets"] - panel["liabilities"]
    book = book.where(panel["permaticker"] == 2, panel["equity"]) + panel["taxliabilities"]
    pair = panel["permaticker"].isin([1, 2])
    share = book[pair] / book[pair].groupby(panel["month_end"][pair]).transform("sum")
    expected = (share**2).groupby(panel["month_end"][pair]).sum().mean()
    assert out.loc[(1, panel["month_end"].iloc[11]), "herf_be"] == pytest.approx(expected)


def test_ch_inv_ia_is_capex_growth_minus_the_sic2_mean():
    panel = concentration_panel()
    panel.loc[(panel["permaticker"] == 3) & (panel["month_end"] == "2020-01-31"), "capex"] = np.nan
    out = features.add_industry_accounting(panel).set_index(["permaticker", "month_end"])
    t = panel["month_end"].iloc[24]
    capex = -panel.set_index(["permaticker", "month_end"])["capex"].unstack(0)
    growth = {}
    for s in [1, 2, 3]:
        now, l12, l24 = capex[s].iloc[24], capex[s].iloc[12], capex[s].iloc[0]
        if s == 3:
            assert np.isnan(l24)  # no PP&E a year before the first month: 12-month growth
            growth[s] = (now - l12) / l12
        else:
            growth[s] = (now - (l12 + l24) / 2) / ((l12 + l24) / 2)
    mean = np.mean(list(growth.values()))
    assert [out.loc[(s, t), "ch_inv_ia"] for s in [1, 2, 3]] == pytest.approx(
        [growth[s] - mean for s in [1, 2, 3]]
    )


def test_industry_accounting_ignores_later_months():
    # Leakage test: later filings cannot move month t.
    panel = concentration_panel(n_months=30)
    t = panel["month_end"].iloc[24]
    wild = panel.copy()
    later = wild["month_end"] > t
    wild.loc[later, ["revenue", "assets", "equity", "capex", "ppnenet"]] = 1e9
    cols = ["herf", "herf_asset", "herf_be", "ch_inv_ia"]
    a = features.add_industry_accounting(panel).loc[lambda d: d["month_end"] == t, cols]
    b = features.add_industry_accounting(wild).loc[lambda d: d["month_end"] == t, cols]
    assert a.iloc[:3].notna().all().all()
    pd.testing.assert_frame_equal(a, b)


def accruals_panel(n_firms=12, seed=0):
    """Two months a year apart; firms 1..n in SIC 3571, plus 3 banks (too few to fit)."""
    rng = np.random.default_rng(seed)
    months = pd.to_datetime(["2020-01-31", "2021-01-31"])
    firms = list(range(1, n_firms + 4))
    panel = pd.DataFrame([(s, m) for s in firms for m in months])
    panel.columns = ["permaticker", "month_end"]
    for col in ["assets", "revenue", "ppnenet"]:
        panel[col] = rng.uniform(50, 500, len(panel))
    panel["netinc"] = rng.normal(10, 5, len(panel))
    panel["ncfo"] = rng.normal(15, 5, len(panel))
    panel["fxusd"] = 1.0
    panel["sic_code"] = np.where(panel["permaticker"] <= n_firms, 3571.0, 6022.0)
    return panel


def test_abnormal_accruals_is_the_industry_regression_residual():
    panel = accruals_panel()
    out = features.add_abnormal_accruals(panel)
    now = panel[panel["month_end"] == "2021-01-31"].set_index("permaticker")
    then = panel[panel["month_end"] == "2020-01-31"].set_index("permaticker")
    lag = then["assets"]
    inputs = pd.DataFrame(
        {
            "y": (now["netinc"] - now["ncfo"]) / lag,
            "inv": 1 / lag,
            "drev": (now["revenue"] - then["revenue"]) / lag,
            "ppe": now["ppnenet"] / lag,
        }
    )
    # With 15 firms, the 0.1% and 99.9% percentiles sit just inside the min and max
    # of each input over the month (banks included): those firms drop out.
    extreme = (inputs == inputs.min()) | (inputs == inputs.max())
    kept = [s for s in range(1, 13) if not extreme.loc[s].any()]
    design = np.column_stack([np.ones(len(kept)), inputs.loc[kept, ["inv", "drev", "ppe"]]])
    y = inputs.loc[kept, "y"].to_numpy()
    resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]

    got = out[out["month_end"] == "2021-01-31"].set_index("permaticker")["abnormal_accruals"]
    assert got.loc[kept].to_numpy() == pytest.approx(resid)
    assert got.drop(kept).isna().all()  # trimmed firms, and 3 banks: fewer than 6 firms
    assert len(kept) >= 6
    assert out.loc[out["month_end"] == "2020-01-31", "abnormal_accruals"].isna().all()


def test_abnormal_accruals_ignores_later_months():
    # Leakage test: a later month's filings cannot move month t.
    panel = accruals_panel()
    extra = panel[panel["month_end"] == "2021-01-31"].assign(
        month_end=pd.Timestamp("2022-01-31"), netinc=1e9, assets=1e-3
    )
    wild = pd.concat([panel, extra], ignore_index=True)
    t = panel["month_end"] == "2021-01-31"
    a = features.add_abnormal_accruals(panel).loc[t.to_numpy(), "abnormal_accruals"]
    b = features.add_abnormal_accruals(wild)
    b = b.loc[b["month_end"] == "2021-01-31", "abnormal_accruals"]
    assert a.notna().any()
    assert a.to_numpy() == pytest.approx(b.to_numpy(), nan_ok=True)


def surprise_by_hand(x, t):
    """The OSAP seasonal-random-walk surprise at index t of a monthly series x."""
    g = lambda i: x[i] - x[i - 12] if i >= 12 else np.nan  # noqa: E731

    def s(i):
        past = [v for k in range(3, 25, 3) if i - k >= 0 and not np.isnan(v := g(i - k))]
        return g(i) - (np.mean(past) if past else np.nan)

    past = [s(t - k) for k in range(3, 25, 3) if t - k >= 0]
    return s(t) / np.nanstd(past, ddof=1)


def surprise_panel(n_months=48, seed=0, n_firms=1):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    panel = pd.DataFrame([(s, m) for s in range(1, n_firms + 1) for m in months])
    panel.columns = ["permaticker", "month_end"]
    quarter = np.repeat(rng.normal(1, 0.5, (n_firms * n_months + 2) // 3 + 1), 3)
    panel["eps_q"] = quarter[: len(panel)]  # each quarter's value held for 3 months
    panel["revenue_q"] = rng.uniform(50, 100, len(panel))
    panel["sharesbas_q"] = rng.uniform(5, 10, len(panel))
    panel["eps"] = rng.normal(2, 1, len(panel))
    panel["market_cap"] = rng.uniform(1e8, 1e9, len(panel))
    panel["fama_industry"] = "Computers"
    return panel


def test_earnings_surprise_by_hand():
    panel = surprise_panel()
    panel.loc[20, "eps_q"] = np.nan  # a missing quarter drops out of the means
    out = features.add_earnings_surprises(panel)
    x = panel["eps_q"].to_numpy()
    for t in [30, 40, 47]:
        assert out["earnings_surprise"].iloc[t] == pytest.approx(surprise_by_hand(x, t))
    assert out["earnings_surprise"].iloc[:18].isna().all()  # no two past surprises yet


def test_revenue_surprise_is_the_same_on_revenue_per_share():
    panel = surprise_panel()
    out = features.add_earnings_surprises(panel)
    x = (panel["revenue_q"] / panel["sharesbas_q"]).to_numpy()
    assert out["revenue_surprise"].iloc[45] == pytest.approx(surprise_by_hand(x, 45))


def test_earn_sup_big_is_the_big_firms_mean_surprise_for_the_rest():
    panel = surprise_panel(n_months=36, n_firms=10)
    t = pd.Timestamp("2020-12-31")
    caps = dict(zip(range(1, 11), np.arange(1, 11) * 1e8, strict=True))
    panel.loc[panel["month_end"] == t, "market_cap"] = panel["permaticker"].map(caps)
    out = features.add_earnings_surprises(panel)
    now = out[out["month_end"] == t].set_index("permaticker")
    big = now.loc[[7, 8, 9, 10], "earnings_surprise"].mean()  # ranks 0.7 .. 1.0
    assert now.loc[1:6, "earn_sup_big"].to_numpy() == pytest.approx([big] * 6)
    assert now.loc[7:10, "earn_sup_big"].isna().all()


def test_earnings_consistency_by_hand_and_exclusions():
    months = pd.date_range("2015-12-31", periods=6 * 12 + 1, freq="ME")
    eps = np.repeat([1.0, 1.1, 1.3, 1.4, 1.6, 1.8, 2.0], 12)[: len(months)]
    panel = pd.DataFrame({"permaticker": 1, "month_end": months, "eps": eps})
    for col in ["eps_q", "revenue_q", "sharesbas_q", "market_cap"]:
        panel[col] = 1.0
    panel["fama_industry"] = "Computers"
    out = features.add_earnings_surprises(panel)["earnings_consistency"]
    annual = [1.0, 1.1, 1.3, 1.4, 1.6, 1.8, 2.0]
    growth = [
        (annual[y] - annual[y - 1]) / (0.5 * (abs(annual[y - 1]) + abs(annual[y - 2])))
        for y in range(2, 7)
    ]
    assert out.iloc[72] == pytest.approx(np.mean(growth))  # this year and the 4 before
    assert out.iloc[24] == pytest.approx(growth[0])  # only one year of growth so far

    falls = panel.assign(eps=np.where(panel["month_end"] >= "2021-12-31", 1.5, eps))
    out = features.add_earnings_surprises(falls)["earnings_consistency"]
    assert np.isnan(out.iloc[72])  # growth turned negative after positive growth
    jumps = panel.assign(eps=np.where(panel["month_end"] >= "2021-12-31", 20.0, eps))
    out = features.add_earnings_surprises(jumps)["earnings_consistency"]
    assert np.isnan(out.iloc[72])  # EPS more than 6 times last year's


def test_earnings_surprises_ignore_later_months():
    # Leakage test: later filings and caps cannot move month t.
    panel = surprise_panel(n_months=48, n_firms=10)
    t = pd.Timestamp("2020-12-31")
    wild = panel.copy()
    later = wild["month_end"] > t
    wild.loc[later, ["eps_q", "revenue_q", "eps", "market_cap"]] = 1e6
    cols = ["earnings_surprise", "revenue_surprise", "earn_sup_big", "earnings_consistency"]
    a = features.add_earnings_surprises(panel).loc[lambda d: d["month_end"] == t, cols]
    b = features.add_earnings_surprises(wild).loc[lambda d: d["month_end"] == t, cols]
    assert a[["earnings_surprise", "revenue_surprise"]].notna().all().all()
    pd.testing.assert_frame_equal(a, b)


def dividend_panel(schedules, start="2000-01-31", n_months=48, price=20.0):
    """Monthly rows for each stock in `schedules` ({permaticker: {month index: amount}}),
    with dividends run through data.real.build_action_months as on real data."""
    from alpha_survival.data import real

    months = pd.date_range(start, periods=n_months, freq="ME")
    stocks = list(schedules)
    panel = pd.DataFrame([(s, m) for s in stocks for m in months])
    panel.columns = ["permaticker", "month_end"]
    panel["close_split_adj"] = price
    rows = [
        {
            "ticker": f"T{s}",
            "action": "dividend",
            "date": months[i] - pd.Timedelta(days=5),
            "value": v,
        }
        for s, schedule in schedules.items()
        for i, v in schedule.items()
    ]
    master = pd.DataFrame({"ticker": [f"T{s}" for s in stocks], "permaticker": stocks})
    return real.add_action_months(panel, real.build_action_months(pd.DataFrame(rows), master))


def div_series(out, stock, column):
    return out[out["permaticker"] == stock].set_index("month_end")[column]


def test_div_init_flags_six_months_after_a_first_dividend_in_24_months():
    panel = dividend_panel({1: {29: 0.5}, 2: {2: 0.5, 29: 0.5}, 3: {20: 0.5, 23: 0.5, 29: 0.5}})
    out = features.add_dividend_signals(panel)
    for stock in [1, 2]:  # stock 2's earlier dividend is 27 months back: an initiation
        s = div_series(out, stock, "div_init")
        assert s.iloc[:24].isna().all()  # burn-in: no 24 months of dividend data yet
        assert s.iloc[29:35].eq(1).all()
        assert s.iloc[24:29].eq(0).all() and s.iloc[35:].eq(0).all()
    assert div_series(out, 3, "div_init").iloc[29:35].eq(0).all()  # not a first dividend


def test_div_omit_flags_a_stopped_quarterly_dividend():
    quarterly = {i: 0.25 for i in range(0, 40, 3)}  # every 3 months to month 39
    panel = dividend_panel({1: quarterly, 2: {i: 0.25 for i in range(0, 48, 3)}})
    out = features.add_dividend_signals(panel)
    s = div_series(out, 1, "div_omit")
    assert s.iloc[:36].isna().all()  # burn-in
    assert s.iloc[[42, 43]].eq(1).all()  # no dividend at 42: omitted at t, then t-1
    assert s.drop(s.index[[42, 43]]).iloc[36:].eq(0).all()
    assert div_series(out, 2, "div_omit").iloc[36:].eq(0).all()  # still paying


def test_div_season_expects_next_months_dividend_by_frequency():
    quarterly = {i: 0.25 for i in range(2, 48, 3)}  # months 2, 5, 8, ...
    semi = {i: 0.5 for i in range(5, 48, 6)}  # months 5, 11, 17, ...
    monthly = {i: 0.1 for i in range(48)}
    out = features.add_dividend_signals(dividend_panel({1: quarterly, 2: semi, 3: monthly}))
    q = div_series(out, 1, "div_season")
    assert q.iloc[:12].isna().all()  # burn-in
    assert q.iloc[12:].tolist() == [1.0 if (t + 1) % 3 == 2 else 0.0 for t in range(12, 48)]
    s = div_series(out, 2, "div_season")
    assert s.iloc[12:].tolist() == [1.0 if (t + 1) % 6 == 5 else 0.0 for t in range(12, 48)]
    assert div_series(out, 3, "div_season").isna().all()  # monthly payers excluded


def test_div_yield_st_terciles_of_the_expected_dividend_yield():
    t = 30  # quarterly payers whose last dividend was 2 months before t
    schedules = {s: {i: 0.1 * s for i in range(t - 26, t - 1, 3)} for s in range(1, 7)}
    schedules[7] = {i: 0.5 for i in range(t - 25, t, 3)}  # paid 1 month ago: none due
    schedules[8] = {}  # never pays
    out = features.add_dividend_signals(dividend_panel(schedules))
    now = out[out["month_end"] == out["month_end"].unique()[t]].set_index("permaticker")
    assert now.loc[1:6, "div_yield_st"].tolist() == [1, 1, 2, 2, 3, 3]
    assert now.loc[7, "div_yield_st"] == 0
    assert np.isnan(now.loc[8, "div_yield_st"])


def test_dividend_signals_ignore_later_months():
    # Leakage test: later dividends and prices cannot move month t.
    schedules = {s: {i: 0.1 * s for i in range(s % 3, 40, 3)} for s in range(1, 7)}
    panel = dividend_panel(schedules)
    t = panel["month_end"].unique()[38]
    wild = dividend_panel({s: {**v, 39: 5.0, 40: 7.0} for s, v in schedules.items()})
    wild.loc[wild["month_end"] > t, "close_split_adj"] = 0.01
    cols = ["div_init", "div_omit", "div_season", "div_yield_st"]
    a = features.add_dividend_signals(panel).loc[lambda d: d["month_end"] == t, cols]
    b = features.add_dividend_signals(wild).loc[lambda d: d["month_end"] == t, cols]
    assert a.notna().all().all()
    pd.testing.assert_frame_equal(a, b)


def event_panel(n_months=40):
    months = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    panel = pd.DataFrame([(s, m) for s in [1, 2, 3] for m in months])
    panel.columns = ["permaticker", "month_end"]
    for col in ["exch_change", "exch_upgrade", "spun_off"]:
        panel[col] = np.nan
    at = lambda s, i: (panel["permaticker"] == s) & (panel["month_end"] == months[i])  # noqa: E731
    panel.loc[at(1, 5), ["exch_change", "exch_upgrade"]] = 1.0  # stock 1 moves up at 5
    panel.loc[at(2, 5), ["exch_change", "exch_upgrade"]] = 1.0  # stock 2 moves up at 5 ...
    panel.loc[at(2, 9), "exch_change"] = 1.0  # ... and down again at 9
    panel.loc[at(3, 10), "spun_off"] = 1.0
    return panel


def test_exch_switch_flags_twelve_months_after_a_move_up():
    out = features.add_event_signals(event_panel())
    s1, s2 = div_series(out, 1, "exch_switch"), div_series(out, 2, "exch_switch")
    assert s1.tolist() == [0.0] * 5 + [1.0] * 12 + [0.0] * 23
    assert s2.tolist() == [0.0] * 5 + [1.0] * 4 + [0.0] * 31  # the later move undoes it
    assert div_series(out, 3, "exch_switch").eq(0).all()


def test_spinoff_flags_24_months_after_the_spin():
    out = features.add_event_signals(event_panel())
    assert div_series(out, 3, "spinoff").tolist() == [0.0] * 10 + [1.0] * 24 + [0.0] * 6
    assert div_series(out, 1, "spinoff").eq(0).all()


def test_event_signals_ignore_later_months():
    # Leakage test: an event after month t cannot move month t.
    panel = event_panel()
    wild = panel.copy()
    wild.loc[wild["month_end"] > "2020-08-31", ["exch_change", "spun_off"]] = 1.0
    t = panel["month_end"] == "2020-08-31"
    cols = ["exch_switch", "spinoff"]
    a = features.add_event_signals(panel).loc[t.to_numpy(), cols]
    b = features.add_event_signals(wild)
    pd.testing.assert_frame_equal(a, b.loc[b["month_end"] == "2020-08-31", cols])
    assert a["exch_switch"].sum() == 2


def ipo_panel():
    """Stock 1 in the data from its start; 2 lists in month 10 with no R&D; 3 in month 10
    with R&D; 4 in month 12."""
    months = pd.date_range("2000-01-31", periods=60, freq="ME")
    first = {1: 0, 2: 10, 3: 10, 4: 12}
    rows = [(s, m) for s, f in first.items() for m in months[f:]]
    panel = pd.DataFrame(rows, columns=["permaticker", "month_end"])
    rng = np.random.default_rng(0)
    panel["ret"] = rng.normal(0.01, 0.05, len(panel))
    panel["rnd"] = np.where(panel["permaticker"] == 3, 5.0, 0.0)
    panel["close_unadj"] = 20.0
    return panel, months


def test_ind_ipo_and_rd_ipo_flag_months_after_the_first_month_in_the_data():
    panel, months = ipo_panel()
    out = features.add_ipo_signals(panel)
    s2 = div_series(out, 2, "ind_ipo")
    assert s2.loc[months[13] : months[46]].eq(1).all()  # months 3 .. 36 after month 10
    assert s2.loc[: months[12]].eq(0).all() and s2.loc[months[47] :].eq(0).all()
    assert div_series(out, 1, "ind_ipo").eq(0).all()  # listed before our data start
    r2, r3 = div_series(out, 2, "rd_ipo"), div_series(out, 3, "rd_ipo")
    assert r2.loc[months[17] : months[46]].eq(1).all()  # months 7 .. 36, no R&D
    assert r2.loc[: months[16]].eq(0).all()
    assert r3.eq(0).all()  # reports R&D


def test_firm_age_mom_is_six_month_momentum_of_the_youngest_fifth():
    panel, months = ipo_panel()
    panel.loc[(panel["permaticker"] == 4) & (panel["month_end"] == months[40]), "ret"] = np.nan
    out = features.add_ipo_signals(panel)
    t = months[43]
    now = out[out["month_end"] == t].set_index("permaticker")["firm_age_mom"]
    # Ages 44, 34, 34, 32: the youngest fifth of four stocks is stock 4 alone.
    rets = panel[panel["permaticker"] == 4].set_index("month_end")["ret"].fillna(0)
    expected = np.prod(1 + rets.loc[months[38] : months[42]]) - 1
    assert now.loc[4] == pytest.approx(expected)
    assert now.drop(4).isna().all()
    young = out[out["month_end"] == months[20]].set_index("permaticker")["firm_age_mom"]
    assert np.isnan(young.loc[4])  # 9 months in the data: under 12


def test_ipo_signals_ignore_later_months():
    # Leakage test: later returns cannot move month t.
    panel, months = ipo_panel()
    t = months[43]
    wild = panel.copy()
    wild.loc[wild["month_end"] > t, "ret"] = 9.0
    cols = ["ind_ipo", "rd_ipo", "firm_age_mom"]
    a = features.add_ipo_signals(panel).loc[lambda d: d["month_end"] == t, cols]
    b = features.add_ipo_signals(wild).loc[lambda d: d["month_end"] == t, cols]
    pd.testing.assert_frame_equal(a, b)
