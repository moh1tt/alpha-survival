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
