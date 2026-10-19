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
