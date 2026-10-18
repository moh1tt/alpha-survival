"""Tests for the real-data loaders. Small hand-made frames, no files or network."""

import pandas as pd
import pytest

from alpha_survival.data import real


def raw_tickers(**overrides):
    """A tiny `tickers` table: two stocks (one delisted), each listed in stocks and fundamentals."""
    df = pd.DataFrame(
        {
            "table": ["stocks", "fundamentals", "stocks", "fundamentals"],
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


def test_one_row_per_stock_from_stocks_rows_only():
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
    # Two different `stocks` rows claiming the same permanent ID -> joins would double count.
    raw = raw_tickers(table=["stocks", "stocks", "stocks", "fundamentals"])
    with pytest.raises(ValueError, match="duplicate permatickers"):
        real.build_security_master(raw)


def test_unexpected_delisted_flag_raises():
    raw = raw_tickers(isdelisted=["N", "N", "maybe", "maybe"])
    with pytest.raises(ValueError, match="isdelisted"):
        real.build_security_master(raw)


# --- monthly returns -------------------------------------------------------


def master_for(*tickers):
    return pd.DataFrame({"ticker": list(tickers), "permaticker": range(1, len(tickers) + 1)})


def price_rows(ticker, dates, closeadj, close=None):
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(dates),
            "closeadj": closeadj,
            "close": close if close is not None else closeadj,
        }
    )


def test_monthly_return_matches_hand_calculation():
    stocks = price_rows(
        "AAA",
        ["2024-01-30", "2024-01-31", "2024-02-15", "2024-02-29", "2024-03-28"],
        [99.0, 100.0, 104.0, 110.0, 99.0],
    )
    out = real.build_monthly_returns(stocks, master_for("AAA"))
    assert list(out["month_end"]) == list(
        pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    )
    assert pd.isna(out["ret"].iloc[0])  # first month: no previous close
    assert out["ret"].iloc[1] == pytest.approx(110 / 100 - 1)  # Jan 31 -> Feb 29
    assert out["ret"].iloc[2] == pytest.approx(99 / 110 - 1)  # Feb 29 -> Mar 28


def test_input_order_does_not_matter():
    # The vendor sends newest-first; shuffling must not change the result.
    stocks = price_rows("AAA", ["2024-01-31", "2024-02-29", "2024-03-28"], [100.0, 110.0, 99.0])
    expected = real.build_monthly_returns(stocks, master_for("AAA"))
    shuffled = stocks.iloc[[2, 0, 1]]
    pd.testing.assert_frame_equal(real.build_monthly_returns(shuffled, master_for("AAA")), expected)


def test_dividends_count_via_closeadj():
    # Raw close is flat, but a dividend was paid: closeadj (earlier prices scaled down) rises.
    stocks = price_rows("AAA", ["2024-01-31", "2024-02-29"], [98.0, 100.0], close=[100.0, 100.0])
    out = real.build_monthly_returns(stocks, master_for("AAA"))
    assert out["ret"].iloc[1] == pytest.approx(100 / 98 - 1)


def test_last_trade_date_flags_mid_month_stop():
    stocks = price_rows("AAA", ["2024-01-31", "2024-02-12"], [100.0, 50.0])
    out = real.build_monthly_returns(stocks, master_for("AAA"))
    assert out["last_trade_date"].iloc[1] == pd.Timestamp("2024-02-12")
    assert out["month_end"].iloc[1] == pd.Timestamp("2024-02-29")


def test_skipped_month_gives_nan_not_a_multi_month_return():
    # No trades at all in February: the Jan -> Mar change is two months, not one.
    stocks = price_rows("AAA", ["2024-01-31", "2024-03-28"], [100.0, 120.0])
    out = real.build_monthly_returns(stocks, master_for("AAA"))
    assert pd.isna(out["ret"].iloc[1])


def test_stocks_do_not_bleed_into_each_other():
    stocks = pd.concat(
        [
            price_rows("AAA", ["2024-01-31", "2024-02-29"], [100.0, 110.0]),
            price_rows("BBB", ["2024-01-31", "2024-02-29"], [10.0, 9.0]),
        ]
    )
    out = real.build_monthly_returns(stocks, master_for("AAA", "BBB")).set_index("permaticker")
    assert pd.isna(out.loc[2, "ret"].iloc[0])  # BBB's first month doesn't use AAA's price
    assert out.loc[2, "ret"].iloc[1] == pytest.approx(9 / 10 - 1)


def test_no_return_depends_on_future_prices():
    # Leakage test: changing prices after Feb must leave Jan and Feb returns untouched.
    dates = ["2024-01-31", "2024-02-29", "2024-03-28", "2024-04-30"]
    base = price_rows("AAA", dates, [100.0, 110.0, 99.0, 105.0])
    shocked = price_rows("AAA", dates, [100.0, 110.0, 500.0, 1.0])
    a = real.build_monthly_returns(base, master_for("AAA"))
    b = real.build_monthly_returns(shocked, master_for("AAA"))
    pd.testing.assert_frame_equal(a.iloc[:2], b.iloc[:2])


def test_unfinished_last_month_is_dropped():
    # Pulled on Oct 16: the October row is half a month, not a monthly return.
    monthly = real.build_monthly_returns(
        price_rows("AAA", ["2026-08-31", "2026-09-30", "2026-10-16"], [100.0, 110.0, 99.0]),
        master_for("AAA"),
    )
    out = real.drop_unfinished_months(monthly, pd.Timestamp("2026-10-16"))
    assert list(out["month_end"]) == list(pd.to_datetime(["2026-08-31", "2026-09-30"]))


def test_month_ending_on_weekend_counts_once_last_weekday_is_reached():
    # Aug 31 2024 was a Saturday; data through Friday Aug 30 covers all of August.
    monthly = real.build_monthly_returns(
        price_rows("AAA", ["2024-07-31", "2024-08-30"], [100.0, 105.0]), master_for("AAA")
    )
    out = real.drop_unfinished_months(monthly, pd.Timestamp("2024-08-30"))
    assert out["month_end"].iloc[-1] == pd.Timestamp("2024-08-31")


def test_holiday_on_last_weekday_drops_month_conservatively():
    # Good Friday, Mar 29 2024: market closed, so Mar 28 was really the last day.
    # Without a trading calendar we wait for the next pull rather than guess.
    monthly = real.build_monthly_returns(
        price_rows("AAA", ["2024-02-29", "2024-03-28"], [100.0, 105.0]), master_for("AAA")
    )
    out = real.drop_unfinished_months(monthly, pd.Timestamp("2024-03-28"))
    assert list(out["month_end"]) == [pd.Timestamp("2024-02-29")]


def test_unmapped_ticker_raises():
    stocks = price_rows("ZZZ", ["2024-01-31"], [1.0])
    with pytest.raises(ValueError, match="missing from security master"):
        real.build_monthly_returns(stocks, master_for("AAA"))


def test_duplicate_price_rows_raise():
    stocks = price_rows("AAA", ["2024-01-31", "2024-01-31"], [100.0, 101.0])
    with pytest.raises(ValueError, match="duplicate"):
        real.build_monthly_returns(stocks, master_for("AAA"))


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
    # `actions` also covers funds etc. that aren't in our stock universe.
    monthly = monthly_rows(1, ["2024-02-29"], [0.10])
    actions = action_rows("SOMEETF", ["2024-02-12"], ["delisted"])
    out = real.add_delisting_returns(monthly, actions, master_with_exchange(AAA="NYSE"))
    assert out["ret"].iloc[0] == pytest.approx(0.10)


def test_takeover_target_with_generic_delisted_is_not_penalised():
    # Found on the full history: `delisted` fires for takeovers too (7,379 stocks).
    monthly = monthly_rows(1, ["2024-02-29"], [0.20])
    actions = action_rows(
        "AAA", ["2024-02-12"] * 3, ["delisted", "acquisitionby", "acquisitioncash"]
    )
    out = real.add_delisting_returns(monthly, actions, master_with_exchange(AAA="NYSE"))
    assert out["ret"].iloc[0] == pytest.approx(0.20)
    assert pd.isna(out["delist_action"].iloc[0])


def test_acquirer_side_does_not_exempt_a_later_unexplained_delisting():
    # AAA bought someone in 2020 (`acquisitionof`), then quietly delisted in 2024.
    monthly = monthly_rows(1, ["2024-02-29"], [0.0])
    actions = action_rows("AAA", ["2020-05-01", "2024-02-12"], ["acquisitionof", "delisted"])
    out = real.add_delisting_returns(monthly, actions, master_with_exchange(AAA="NYSE"))
    assert out["ret"].iloc[0] == pytest.approx(-0.30)


def test_unexplained_delisting_penalised_only_when_asked():
    monthly = monthly_rows(1, ["2024-02-29"], [0.0])
    actions = action_rows("AAA", ["2024-02-12"], ["delisted"])
    master = master_with_exchange(AAA="NYSE")
    on = real.add_delisting_returns(monthly, actions, master)
    off = real.add_delisting_returns(monthly, actions, master, penalise_unexplained=False)
    assert on["ret"].iloc[0] == pytest.approx(-0.30)
    assert off["ret"].iloc[0] == pytest.approx(0.0)
    assert pd.isna(off["delist_action"].iloc[0])


def test_explicit_distress_is_penalised_in_both_modes_even_after_a_takeover_bid():
    monthly = monthly_rows(1, ["2024-02-29"], [0.0])
    actions = action_rows("AAA", ["2024-02-12"] * 2, ["bankruptcyliquidation", "acquisitionby"])
    master = master_with_exchange(AAA="NYSE")
    for flag in (True, False):
        out = real.add_delisting_returns(monthly, actions, master, penalise_unexplained=flag)
        assert out["ret"].iloc[0] == pytest.approx(-0.30)


# --- point-in-time fundamentals ----------------------------------------------


def filing_rows(ticker, filing_dates, reportperiods, revenue, dimension="ART"):
    return pd.DataFrame(
        {
            "ticker": ticker,
            "dimension": dimension,
            "date": pd.to_datetime(filing_dates),
            "reportperiod": pd.to_datetime(reportperiods),
            "lastupdated": pd.Timestamp("2026-07-31"),
            "revenue": revenue,
        }
    )


def keys_for(permaticker, month_ends):
    return pd.DataFrame({"permaticker": permaticker, "month_end": pd.to_datetime(month_ends)})


MONTHS = ["2024-01-31", "2024-02-29", "2024-03-31"]


def test_filing_is_used_only_after_it_became_public():
    # FY2023 ended Dec 31 but was filed Feb 14: January must not see it.
    filings = filing_rows("AAA", ["2024-02-14"], ["2023-12-31"], [500.0])
    out = real.build_fundamentals(filings, master_for("AAA"), keys_for(1, MONTHS))
    assert pd.isna(out["revenue"].iloc[0])
    assert list(out["revenue"].iloc[1:]) == [500.0, 500.0]


def test_filing_on_month_end_waits_until_next_month():
    # Filed on Jan 31 (maybe after the close): not usable for the Jan 31 rebalance.
    filings = filing_rows("AAA", ["2024-01-31"], ["2023-12-31"], [500.0])
    out = real.build_fundamentals(filings, master_for("AAA"), keys_for(1, MONTHS))
    assert pd.isna(out["revenue"].iloc[0])
    assert out["revenue"].iloc[1] == 500.0


def test_later_filing_does_not_rewrite_earlier_months():
    # An amended filing for the same period applies from its own filing date onward only.
    filings = filing_rows(
        "AAA", ["2024-01-10", "2024-03-05"], ["2023-12-31", "2023-12-31"], [500.0, 450.0]
    )
    out = real.build_fundamentals(filings, master_for("AAA"), keys_for(1, MONTHS))
    assert list(out["revenue"]) == [500.0, 500.0, 450.0]


def test_no_month_uses_a_filing_from_its_future():
    filings = filing_rows(
        "AAA",
        ["2023-11-02", "2024-01-31", "2024-02-28", "2024-03-30"],
        ["2023-09-30", "2023-12-31", "2023-12-31", "2024-03-29"],
        [1.0, 2.0, 3.0, 4.0],
    )
    out = real.build_fundamentals(filings, master_for("AAA"), keys_for(1, MONTHS)).dropna()
    assert (out["filing_date"] < out["month_end"]).all()


def test_stocks_do_not_bleed_into_each_other_fundamentals():
    filings = pd.concat(
        [
            filing_rows("AAA", ["2024-01-10"], ["2023-12-31"], [500.0]),
            filing_rows("BBB", ["2024-02-10"], ["2023-12-31"], [70.0]),
        ]
    )
    keys = pd.concat([keys_for(1, MONTHS[:2]), keys_for(2, MONTHS[:2])])
    out = real.build_fundamentals(filings, master_for("AAA", "BBB"), keys)
    bbb = out[out["permaticker"] == 2]
    assert pd.isna(bbb["revenue"].iloc[0])  # BBB had no filing yet in January
    assert bbb["revenue"].iloc[1] == 70.0


def test_other_dimensions_are_ignored():
    filings = pd.concat(
        [
            filing_rows("AAA", ["2024-01-10"], ["2023-12-31"], [500.0], dimension="ART"),
            filing_rows("AAA", ["2024-01-10"], ["2023-12-31"], [130.0], dimension="ARQ"),
        ]
    )
    out = real.build_fundamentals(filings, master_for("AAA"), keys_for(1, MONTHS[:1]))
    assert out["revenue"].iloc[0] == 500.0


def test_output_columns_exclude_vendor_bookkeeping():
    filings = filing_rows("AAA", ["2024-01-10"], ["2023-12-31"], [500.0])
    out = real.build_fundamentals(filings, master_for("AAA"), keys_for(1, MONTHS[:1]))
    assert list(out.columns) == ["permaticker", "month_end", "filing_date", "period_end", "revenue"]


@pytest.mark.parametrize("dimension", ["MRQ", "MRT", "MRY"])
def test_restated_dimensions_raise(dimension):
    filings = filing_rows("AAA", ["2024-01-10"], ["2023-12-31"], [500.0], dimension=dimension)
    with pytest.raises(ValueError, match="not point-in-time"):
        real.build_fundamentals(filings, master_for("AAA"), keys_for(1, MONTHS), dimension)


def test_duplicate_filing_rows_raise():
    filings = filing_rows(
        "AAA", ["2024-01-10", "2024-01-10"], ["2023-12-31", "2023-12-31"], [1.0, 2.0]
    )
    with pytest.raises(ValueError, match="duplicate"):
        real.build_fundamentals(filings, master_for("AAA"), keys_for(1, MONTHS))


def test_filing_before_period_end_raises():
    filings = filing_rows("AAA", ["2023-12-01"], ["2023-12-31"], [500.0])
    with pytest.raises(ValueError, match="before their period ended"):
        real.build_fundamentals(filings, master_for("AAA"), keys_for(1, MONTHS))


def test_unmapped_fundamentals_ticker_raises():
    filings = filing_rows("ZZZ", ["2024-01-10"], ["2023-12-31"], [500.0])
    with pytest.raises(ValueError, match="missing from security master"):
        real.build_fundamentals(filings, master_for("AAA"), keys_for(1, MONTHS))


# --- modelling panel ---------------------------------------------------------


def test_fwd_ret_is_next_months_return():
    monthly = monthly_rows(1, MONTHS, [0.01, 0.02, 0.03])
    out = real.build_panel(monthly, keys_for(1, MONTHS))
    assert out["fwd_ret"].iloc[0] == pytest.approx(0.02)
    assert out["fwd_ret"].iloc[1] == pytest.approx(0.03)
    assert pd.isna(out["fwd_ret"].iloc[2])  # last month: future not known yet


def test_skipped_month_gives_nan_target():
    monthly = monthly_rows(1, ["2024-01-31", "2024-03-31"], [0.01, 0.03])
    out = real.build_panel(monthly, keys_for(1, ["2024-01-31", "2024-03-31"]))
    assert out["fwd_ret"].isna().all()


def test_targets_do_not_bleed_across_stocks():
    monthly = pd.concat(
        [monthly_rows(1, MONTHS, [0.01, 0.02, 0.03]), monthly_rows(2, MONTHS, [0.10, 0.20, 0.30])]
    )
    keys = pd.concat([keys_for(1, MONTHS), keys_for(2, MONTHS)])
    out = real.build_panel(monthly.iloc[::-1], keys).set_index(["permaticker", "month_end"])
    assert pd.isna(out.loc[(1, pd.Timestamp("2024-03-31")), "fwd_ret"])
    assert out.loc[(2, pd.Timestamp("2024-01-31")), "fwd_ret"] == pytest.approx(0.20)


def test_panel_keeps_every_return_row_and_joins_fundamentals():
    filings = filing_rows("AAA", ["2024-02-14"], ["2023-12-31"], [500.0])
    fundamentals = real.build_fundamentals(filings, master_for("AAA"), keys_for(1, MONTHS))
    out = real.build_panel(monthly_rows(1, MONTHS, [0.01, 0.02, 0.03]), fundamentals)
    assert len(out) == 3
    assert pd.isna(out["revenue"].iloc[0])  # row kept even before the first filing
    assert list(out["revenue"].iloc[1:]) == [500.0, 500.0]


def test_duplicate_monthly_rows_raise():
    monthly = monthly_rows(1, ["2024-01-31", "2024-01-31"], [0.01, 0.02])
    with pytest.raises(ValueError, match="duplicate"):
        real.build_panel(monthly, keys_for(1, ["2024-01-31"]))


def test_only_the_target_sees_the_next_month():
    # Leakage test, end to end: shocking April prices and adding an April filing may
    # change March's target, but no March feature, and nothing at all before March.
    dates = ["2024-01-31", "2024-02-29", "2024-03-28", "2024-04-30"]
    master = master_for("AAA")
    filings = filing_rows("AAA", ["2024-01-10"], ["2023-12-31"], [500.0])
    late = filing_rows("AAA", ["2024-04-15"], ["2024-03-31"], [999.0])

    def panel(closes, filings):
        monthly = real.build_monthly_returns(price_rows("AAA", dates, closes), master)
        return real.build_panel(monthly, real.build_fundamentals(filings, master, monthly))

    base = panel([100.0, 110.0, 99.0, 105.0], filings)
    shocked = panel([100.0, 110.0, 99.0, 1.0], pd.concat([filings, late]))
    pd.testing.assert_frame_equal(base.iloc[:2], shocked.iloc[:2])
    features = [c for c in base.columns if c != "fwd_ret"]
    pd.testing.assert_series_equal(base.loc[2, features], shocked.loc[2, features])
    assert base.loc[2, "fwd_ret"] != shocked.loc[2, "fwd_ret"]
