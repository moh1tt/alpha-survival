"""Tests for the real-data loaders. Small hand-made frames, no files or network."""

import pandas as pd
import pytest

from alpha_survival.data import real


def raw_tickers(**overrides):
    """A tiny TICKERS table: two stocks (one delisted), each listed for SEP and SF1."""
    df = pd.DataFrame(
        {
            "table": ["SEP", "SF1", "SEP", "SF1"],
            "permaticker": [101, 101, 202, 202],
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "name": ["Alpha Inc", "Alpha Inc", "Beta Corp", "Beta Corp"],
            "exchange": ["NYSE", "NYSE", "NASDAQ", "NASDAQ"],
            "category": ["Domestic Common Stock"] * 4,
            "isdelisted": ["N", "N", "Y", "Y"],
            "firstpricedate": pd.to_datetime(["2000-01-03"] * 4),
            "lastpricedate": pd.to_datetime(
                ["2026-10-12", "2026-10-12", "2009-03-31", "2009-03-31"]
            ),
        }
    )
    for col, values in overrides.items():
        df[col] = values
    return df


def test_one_row_per_stock_from_sep_rows_only():
    master = real.build_security_master(raw_tickers())
    assert list(master["permaticker"]) == [101, 202]
    assert master["permaticker"].is_unique


def test_delisted_stocks_are_kept_as_booleans():
    master = real.build_security_master(raw_tickers()).set_index("permaticker")
    assert master.loc[202, "is_delisted"]
    assert not master.loc[101, "is_delisted"]
    assert master["is_delisted"].dtype == bool


def test_output_uses_our_column_names():
    master = real.build_security_master(raw_tickers())
    assert list(master.columns) == list(real.SECURITY_MASTER_COLUMNS.values())


def test_duplicate_permaticker_raises():
    # Two different SEP rows claiming the same permanent ID -> joins would double count.
    raw = raw_tickers(table=["SEP", "SEP", "SEP", "SF1"])
    with pytest.raises(ValueError, match="duplicate permatickers"):
        real.build_security_master(raw)


def test_unexpected_delisted_flag_raises():
    raw = raw_tickers(isdelisted=["N", "N", "maybe", "maybe"])
    with pytest.raises(ValueError, match="isdelisted"):
        real.build_security_master(raw)


# --- monthly returns -------------------------------------------------------


def master_for(*tickers):
    return pd.DataFrame({"ticker": list(tickers), "permaticker": range(1, len(tickers) + 1)})


def sep_rows(ticker, dates, closeadj, close=None):
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(dates),
            "closeadj": closeadj,
            "close": close if close is not None else closeadj,
        }
    )


def test_monthly_return_matches_hand_calculation():
    sep = sep_rows(
        "AAA",
        ["2024-01-30", "2024-01-31", "2024-02-15", "2024-02-29", "2024-03-28"],
        [99.0, 100.0, 104.0, 110.0, 99.0],
    )
    out = real.build_monthly_returns(sep, master_for("AAA"))
    assert list(out["month_end"]) == list(
        pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    )
    assert pd.isna(out["ret"].iloc[0])  # first month: no previous close
    assert out["ret"].iloc[1] == pytest.approx(110 / 100 - 1)  # Jan 31 -> Feb 29
    assert out["ret"].iloc[2] == pytest.approx(99 / 110 - 1)  # Feb 29 -> Mar 28


def test_input_order_does_not_matter():
    # The vendor sends newest-first; shuffling must not change the result.
    sep = sep_rows("AAA", ["2024-01-31", "2024-02-29", "2024-03-28"], [100.0, 110.0, 99.0])
    expected = real.build_monthly_returns(sep, master_for("AAA"))
    shuffled = sep.iloc[[2, 0, 1]]
    pd.testing.assert_frame_equal(real.build_monthly_returns(shuffled, master_for("AAA")), expected)


def test_dividends_count_via_closeadj():
    # Raw close is flat, but a dividend was paid: closeadj (earlier prices scaled down) rises.
    sep = sep_rows("AAA", ["2024-01-31", "2024-02-29"], [98.0, 100.0], close=[100.0, 100.0])
    out = real.build_monthly_returns(sep, master_for("AAA"))
    assert out["ret"].iloc[1] == pytest.approx(100 / 98 - 1)


def test_last_trade_date_flags_mid_month_stop():
    sep = sep_rows("AAA", ["2024-01-31", "2024-02-12"], [100.0, 50.0])
    out = real.build_monthly_returns(sep, master_for("AAA"))
    assert out["last_trade_date"].iloc[1] == pd.Timestamp("2024-02-12")
    assert out["month_end"].iloc[1] == pd.Timestamp("2024-02-29")


def test_skipped_month_gives_nan_not_a_multi_month_return():
    # No trades at all in February: the Jan -> Mar change is two months, not one.
    sep = sep_rows("AAA", ["2024-01-31", "2024-03-28"], [100.0, 120.0])
    out = real.build_monthly_returns(sep, master_for("AAA"))
    assert pd.isna(out["ret"].iloc[1])


def test_stocks_do_not_bleed_into_each_other():
    sep = pd.concat(
        [
            sep_rows("AAA", ["2024-01-31", "2024-02-29"], [100.0, 110.0]),
            sep_rows("BBB", ["2024-01-31", "2024-02-29"], [10.0, 9.0]),
        ]
    )
    out = real.build_monthly_returns(sep, master_for("AAA", "BBB")).set_index("permaticker")
    assert pd.isna(out.loc[2, "ret"].iloc[0])  # BBB's first month doesn't use AAA's price
    assert out.loc[2, "ret"].iloc[1] == pytest.approx(9 / 10 - 1)


def test_no_return_depends_on_future_prices():
    # Leakage test: changing prices after Feb must leave Jan and Feb returns untouched.
    dates = ["2024-01-31", "2024-02-29", "2024-03-28", "2024-04-30"]
    base = sep_rows("AAA", dates, [100.0, 110.0, 99.0, 105.0])
    shocked = sep_rows("AAA", dates, [100.0, 110.0, 500.0, 1.0])
    a = real.build_monthly_returns(base, master_for("AAA"))
    b = real.build_monthly_returns(shocked, master_for("AAA"))
    pd.testing.assert_frame_equal(a.iloc[:2], b.iloc[:2])


def test_unmapped_ticker_raises():
    sep = sep_rows("ZZZ", ["2024-01-31"], [1.0])
    with pytest.raises(ValueError, match="missing from security master"):
        real.build_monthly_returns(sep, master_for("AAA"))


def test_duplicate_price_rows_raise():
    sep = sep_rows("AAA", ["2024-01-31", "2024-01-31"], [100.0, 101.0])
    with pytest.raises(ValueError, match="duplicate"):
        real.build_monthly_returns(sep, master_for("AAA"))


# --- delisting returns -----------------------------------------------------


def monthly_rows(permaticker, month_ends, rets):
    return pd.DataFrame(
        {"permaticker": permaticker, "month_end": pd.to_datetime(month_ends), "ret": rets}
    )


def master_with_exchange(**ticker_to_exchange):
    return pd.DataFrame(
        {
            "ticker": list(ticker_to_exchange),
            "permaticker": range(1, len(ticker_to_exchange) + 1),
            "exchange": list(ticker_to_exchange.values()),
        }
    )


def action_rows(ticker, dates, actions):
    return pd.DataFrame({"ticker": ticker, "date": pd.to_datetime(dates), "action": actions})


def test_bankruptcy_on_nyse_compounds_minus_30pct():
    monthly = monthly_rows(1, ["2024-01-31", "2024-02-29"], [0.05, -0.50])
    actions = action_rows("AAA", ["2024-02-12"], ["bankruptcyliquidation"])
    out = real.add_delisting_returns(monthly, actions, master_with_exchange(AAA="NYSE"))
    assert out["ret"].iloc[0] == pytest.approx(0.05)  # earlier month untouched
    assert out["ret"].iloc[1] == pytest.approx(0.5 * 0.7 - 1)
    assert out["delist_action"].iloc[1] == "bankruptcyliquidation"


def test_nasdaq_penalty_is_minus_55pct():
    monthly = monthly_rows(1, ["2024-02-29"], [0.0])
    actions = action_rows("AAA", ["2024-02-12"], ["regulatorydelisting"])
    out = real.add_delisting_returns(monthly, actions, master_with_exchange(AAA="NASDAQ"))
    assert out["ret"].iloc[0] == pytest.approx(-0.55)


def test_unknown_exchange_uses_default_penalty():
    monthly = monthly_rows(1, ["2024-02-29"], [0.0])
    actions = action_rows("AAA", ["2024-02-12"], ["delisted"])
    out = real.add_delisting_returns(monthly, actions, master_with_exchange(AAA="BATS"))
    assert out["ret"].iloc[0] == pytest.approx(real.DEFAULT_DELISTING_PENALTY)


def test_acquisition_is_not_penalised():
    monthly = monthly_rows(1, ["2024-02-29"], [0.20])
    actions = action_rows("AAA", ["2024-02-12"], ["acquisitionby"])
    out = real.add_delisting_returns(monthly, actions, master_with_exchange(AAA="NYSE"))
    assert out["ret"].iloc[0] == pytest.approx(0.20)
    assert pd.isna(out["delist_action"].iloc[0])


def test_several_exit_rows_penalise_once():
    monthly = monthly_rows(1, ["2024-02-29"], [0.0])
    actions = action_rows(
        "AAA", ["2024-02-09", "2024-02-12"], ["regulatorydelisting", "bankruptcyliquidation"]
    )
    out = real.add_delisting_returns(monthly, actions, master_with_exchange(AAA="NYSE"))
    assert out["ret"].iloc[0] == pytest.approx(-0.30)
    assert len(out) == 1


def test_exit_without_price_row_warns_and_is_not_applied():
    monthly = monthly_rows(1, ["2024-01-31"], [0.10])
    actions = action_rows("AAA", ["2024-03-15"], ["delisted"])
    with pytest.warns(UserWarning, match="1 distress exits"):
        out = real.add_delisting_returns(monthly, actions, master_with_exchange(AAA="NYSE"))
    assert out["ret"].iloc[0] == pytest.approx(0.10)


def test_actions_for_securities_outside_master_are_ignored():
    # ACTIONS also covers funds etc. that aren't in our stock universe.
    monthly = monthly_rows(1, ["2024-02-29"], [0.10])
    actions = action_rows("SOMEETF", ["2024-02-12"], ["delisted"])
    out = real.add_delisting_returns(monthly, actions, master_with_exchange(AAA="NYSE"))
    assert out["ret"].iloc[0] == pytest.approx(0.10)
