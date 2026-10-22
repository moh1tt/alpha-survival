"""Tests for the real-data loaders. Small hand-made frames, no files or network."""

import numpy as np
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


def test_bulk_file_legacy_table_codes_work_too():
    # The bulk `tickers` file labels the price table "SEP", row queries "stocks".
    raw = raw_tickers(table=["SEP", "SF1", "SEP", "SF1"])
    pd.testing.assert_frame_equal(
        real.build_security_master(raw), real.build_security_master(raw_tickers())
    )


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


def test_universe_keeps_us_common_stock_on_major_exchanges_including_delisted():
    master = pd.DataFrame(
        {
            "permaticker": [1, 2, 3, 4, 5, 6],
            "category": [
                "Domestic Common Stock",
                "Domestic Common Stock Secondary Class",
                "ADR Common Stock",
                "Domestic Preferred Stock",
                "Domestic Common Stock",
                "Domestic Common Stock Primary Class",
            ],
            "exchange": ["NYSE", "NASDAQ", "NYSE", "NYSE", "OTC", "NYSEMKT"],
            "is_delisted": [False, True, False, False, False, True],
        }
    )
    assert list(real.select_universe(master)["permaticker"]) == [1, 2, 6]


# --- monthly returns -------------------------------------------------------


def master_for(*tickers):
    return pd.DataFrame({"ticker": list(tickers), "permaticker": range(1, len(tickers) + 1)})


def price_rows(ticker, dates, closeadj, close=None, closeunadj=None, high=None, low=None):
    close = close if close is not None else closeadj
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(dates),
            "closeadj": closeadj,
            "close": close,
            "closeunadj": closeunadj if closeunadj is not None else closeadj,
            "high": high if high is not None else close,
            "low": low if low is not None else close,
            "volume": 1000.0,
        }
    )


def ar_daily(highs, lows, closes, volumes=None, permaticker=1):
    volumes = volumes if volumes is not None else [1000.0] * len(closes)
    return pd.DataFrame(
        {"permaticker": permaticker, "high": highs, "low": lows, "close": closes, "volume": volumes}
    )


def ar_spread(products):
    return np.sqrt(max(4 * np.nanmean(products), 0))


X = np.log(101 / 99)  # log range of a day trading 99-101


def test_abdi_ranaldo_by_hand():
    # Day 1 closes at its high (the ask); both days trade 99-101, so the close sits
    # X/2 above both mid-ranges: product = (X/2)^2 and S = sqrt(4 (X/2)^2) = X.
    p = real.abdi_ranaldo_product(ar_daily([101.0, 101.0], [99.0, 99.0], [101.0, 100.0]))
    assert np.isnan(p.iloc[0])  # first day: no previous day
    assert p.iloc[1] == pytest.approx((X / 2) ** 2)
    assert ar_spread(p) == pytest.approx(X)


def test_abdi_ranaldo_close_at_mid_is_zero_spread():
    mid = np.sqrt(101 * 99)  # exactly the mid-range in logs
    p = real.abdi_ranaldo_product(ar_daily([101.0, 101.0], [99.0, 99.0], [mid, mid]))
    assert p.iloc[1] == pytest.approx(0.0, abs=1e-15)


def test_abdi_ranaldo_skips_no_trade_days_and_other_stocks():
    daily = pd.concat(
        [
            # Day 2 has no volume; day 3 has high == low (vendor repeated the close).
            ar_daily([101.0] * 4, [99.0, 99.0, 101.0, 99.0], [101.0] * 4, [1e3, 0.0, 1e3, 1e3]),
            ar_daily([101.0, 101.0], [99.0, 99.0], [101.0, 101.0], permaticker=2),
        ],
        ignore_index=True,
    )
    p = real.abdi_ranaldo_product(daily)
    assert p.iloc[:4].isna().all()  # every pair touches a no-trade day
    assert np.isnan(p.iloc[4])  # stock 2's first day is not paired with stock 1
    assert p.iloc[5] == pytest.approx((X / 2) ** 2)


def test_monthly_spread_terms_file_each_pair_under_its_later_day():
    # Jan 30 -> Jan 31 is a January pair; Jan 31 -> Feb 1 a February one.
    stocks = price_rows(
        "AAA",
        ["2024-01-30", "2024-01-31", "2024-02-01"],
        [101.0] * 3,
        high=[101.0] * 3,
        low=[99.0] * 3,
    )
    out = real.build_monthly_returns(stocks, master_for("AAA"))
    assert out["ar_n"].tolist() == [1, 1]
    assert out["ar_sum"].tolist() == pytest.approx([(X / 2) ** 2] * 2)


def test_monthly_spread_terms_ignore_later_days():
    # Leakage check: a wild March range cannot move January or February.
    dates = ["2024-01-30", "2024-01-31", "2024-02-28", "2024-02-29", "2024-03-01"]
    base = price_rows("AAA", dates, [101.0] * 5, high=[101.0] * 5, low=[99.0] * 5)
    wild = base.copy()
    wild.loc[4, ["high", "low"]] = [500.0, 1.0]
    a = real.build_monthly_returns(base, master_for("AAA"))
    b = real.build_monthly_returns(wild, master_for("AAA"))
    pd.testing.assert_frame_equal(a.iloc[:2], b.iloc[:2])
    assert b["ar_sum"].iloc[2] != a["ar_sum"].iloc[2]  # March itself does change


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


def test_close_unadj_is_the_traded_price_not_the_adjusted_one():
    # 1:10 reverse split in March: adjusted history shows $5, the market showed $0.50.
    stocks = price_rows(
        "AAA", ["2024-02-29", "2024-03-28"], [5.0, 5.5], close=[5.0, 5.5], closeunadj=[0.5, 5.5]
    )
    out = real.build_monthly_returns(stocks, master_for("AAA"))
    assert list(out["close_unadj"]) == [0.5, 5.5]
    assert out["ret"].iloc[1] == pytest.approx(5.5 / 5.0 - 1)  # returns still from closeadj


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


def test_daily_return_moments_per_month():
    stocks = pd.concat(
        [
            price_rows(
                "AAA",
                ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"],
                [100.0, 110.0, 99.0, 99.0],
            ),
            price_rows("BBB", ["2024-02-01", "2024-02-02"], [10.0, 20.0]),
        ]
    )
    out = real.build_monthly_returns(stocks, master_for("AAA", "BBB")).set_index(
        ["permaticker", "month_end"]
    )
    jan, feb = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")
    # AAA: Jan has one daily return (+10%; Jan 30 is its first day), Feb has
    # -10% (Jan 31 -> Feb 1, counted in the month it ends) and 0%.
    assert out.loc[(1, jan), "n_days"] == 1
    assert out.loc[(1, jan), "dret_sum"] == pytest.approx(0.10)
    assert out.loc[(1, feb), "n_days"] == 2
    assert out.loc[(1, feb), "dret_sum"] == pytest.approx(-0.10)
    assert out.loc[(1, feb), "dret_sumsq"] == pytest.approx(0.01)
    # BBB's first day does not use AAA's last price.
    assert out.loc[(2, feb), "n_days"] == 1
    assert out.loc[(2, feb), "dret_sum"] == pytest.approx(1.0)


def test_daily_summaries_for_price_and_trading_signals_by_hand():
    # Feb: three days. Daily returns (closeadj): Jan 31 -> Feb 1 +10%, then -20%, then 0%.
    stocks = price_rows(
        "AAA",
        ["2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05"],
        [100.0, 110.0, 88.0, 88.0],
        close=[50.0, 55.0, 44.0, 44.0],  # split-adjusted close: half closeadj
    ).assign(volume=[10.0, 20.0, 5.0, 0.0])
    out = real.build_monthly_returns(stocks, master_for("AAA")).set_index("month_end")
    feb = out.loc[pd.Timestamp("2024-02-29")]
    assert feb["dret_max"] == pytest.approx(0.10)
    assert feb["dret_sumcube"] == pytest.approx(0.1**3 + (-0.2) ** 3)
    assert feb["close_max"] == 55.0
    assert feb["share_volume"] == 25.0
    assert feb["dollar_volume"] == pytest.approx(55.0 * 20 + 44.0 * 5)  # close x volume
    # Amihud: |ret| / dollars, only on days that traded (Feb 5 had no volume).
    assert feb["illiq_n"] == 2
    assert feb["illiq_sum"] == pytest.approx(0.10 / 1100 + 0.20 / 220)
    assert feb["n_trade_days"] == 3 and feb["n_zero_volume"] == 1


def test_daily_summaries_ignore_later_days():
    # Leakage test: a wild March cannot change February's summaries.
    dates = ["2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01"]
    base = price_rows("AAA", dates, [100.0, 110.0, 99.0, 100.0])
    wild = price_rows("AAA", dates, [100.0, 110.0, 99.0, 900.0]).assign(volume=[1000.0] * 3 + [0])
    a = real.build_monthly_returns(base, master_for("AAA"))
    b = real.build_monthly_returns(wild, master_for("AAA"))
    cols = real.DAILY_SUMMARY_COLUMNS
    pd.testing.assert_frame_equal(a.loc[:1, cols], b.loc[:1, cols])


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


def test_filing_expires_after_max_age():
    # Filed 2023-01-31: exactly 365 days old on 2024-01-31 (still used), 394 on 2024-02-29.
    filings = filing_rows("AAA", ["2023-01-31"], ["2022-12-31"], [500.0])
    out = real.build_fundamentals(filings, master_for("AAA"), keys_for(1, MONTHS))
    assert out["revenue"].iloc[0] == 500.0
    assert out[["revenue", "filing_date", "period_end"]].iloc[1:].isna().all().all()


def test_new_filing_revives_fundamentals_after_expiry():
    # Stopped reporting for over a year, then filed again: used from the new filing on.
    filings = filing_rows(
        "AAA", ["2022-11-15", "2024-02-20"], ["2022-09-30", "2023-12-31"], [500.0, 450.0]
    )
    out = real.build_fundamentals(filings, master_for("AAA"), keys_for(1, MONTHS))
    assert pd.isna(out["revenue"].iloc[0])
    assert list(out["revenue"].iloc[1:]) == [450.0, 450.0]


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


def test_filing_before_period_end_is_dropped_with_warning():
    # Found on the full history (e.g. a "FY1996" filing dated May 1996): the real
    # filing came later, so the row must not be used at its impossible date.
    filings = filing_rows(
        "AAA", ["2023-12-01", "2024-02-14"], ["2023-12-31", "2023-12-31"], [999.0, 500.0]
    )
    with pytest.warns(UserWarning, match="dropped 1 ART filings"):
        out = real.build_fundamentals(filings, master_for("AAA"), keys_for(1, MONTHS))
    assert pd.isna(out["revenue"].iloc[0])
    assert list(out["revenue"].iloc[1:]) == [500.0, 500.0]


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


def test_investable_uses_this_months_traded_price():
    panel = pd.DataFrame({"close_unadj": [4.99, 5.0, 120.0, float("nan")]})
    out = real.flag_investable(panel)
    assert list(out["investable"]) == [False, True, True, False]  # no price: can't buy


def test_investable_ignores_future_prices():
    # Leakage test: a stock cheap in January is not investable in January even if it
    # later trades above the cut (and vice versa).
    dates = ["2024-01-31", "2024-02-29", "2024-03-28"]
    stocks = price_rows("AAA", dates, [4.0, 50.0, 3.0])
    monthly = real.build_monthly_returns(stocks, master_for("AAA"))
    out = real.flag_investable(monthly)
    assert list(out["investable"]) == [False, True, False]


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


def test_monthly_returns_keep_split_adjusted_close_for_market_cap():
    stocks = price_rows(
        "AAA", ["2024-02-29", "2024-03-28"], [5.0, 5.5], close=[5.0, 5.5], closeunadj=[0.5, 5.5]
    )
    out = real.build_monthly_returns(stocks, master_for("AAA"))
    assert list(out["close_split_adj"]) == [5.0, 5.5]


def cap_panel(close_split_adj, sharesbas, sharefactor=1.0):
    n = len(close_split_adj)
    return pd.DataFrame(
        {
            "close_split_adj": close_split_adj,
            "sharesbas": sharesbas if isinstance(sharesbas, list) else [sharesbas] * n,
            "sharefactor": [sharefactor] * n,
        }
    )


def test_market_cap_is_price_times_filed_shares_times_sharefactor():
    out = real.add_market_cap(cap_panel([10.0, 12.0], 1_000.0, sharefactor=0.5))
    assert list(out["market_cap"]) == [5_000.0, 6_000.0]


def test_split_after_filing_is_handled_by_restated_data():
    # Filed in January with 1,000 shares at $10; 2-for-1 split in February, after
    # which the stock trades at $5. Sharadar today shows the split-adjusted close
    # (Jan $5, Feb $5) and the restated count (2,000): $10,000 in both months.
    out = real.add_market_cap(cap_panel([5.0, 5.0], 2_000.0))
    assert list(out["market_cap"]) == [10_000.0, 10_000.0]


def test_market_cap_does_not_change_when_a_later_split_restates_history():
    # Leakage test: a split after month_end rescales the stored close by 1/k and the
    # stored share count by k. The market cap at month_end must not move.
    before = real.add_market_cap(cap_panel([10.0, 12.0], 1_000.0))
    after = real.add_market_cap(cap_panel([10.0 / 7, 12.0 / 7], 7_000.0))
    assert list(after["market_cap"]) == pytest.approx(list(before["market_cap"]))


def test_market_cap_is_nan_without_shares():
    out = real.add_market_cap(cap_panel([10.0, 10.0, 10.0], [float("nan"), 0.0, 1_000.0]))
    assert out["market_cap"].isna().tolist() == [True, True, False]


CS_C = 3 - 2 * np.sqrt(2)


def cs_by_hand(high_1, low_1, high_2, low_2):
    beta = np.log(high_1 / low_1) ** 2 + np.log(high_2 / low_2) ** 2
    gamma = np.log(max(high_1, high_2) / min(low_1, low_2)) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / CS_C - np.sqrt(gamma / CS_C)
    return max(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)), 0.0)


def test_corwin_schultz_by_hand():
    # Two identical 99-101 days: no extra range over two days, so it is all spread.
    # alpha works out to log(101/99) and S = 2 (101/99 - 1) / (1 + 101/99) = 2%.
    s = real.corwin_schultz_spread(ar_daily([101.0, 101.0], [99.0, 99.0], [100.0, 100.0]))
    assert np.isnan(s.iloc[0])  # first day: no previous day
    assert s.iloc[1] == pytest.approx(0.02)
    # A second day trading higher adds volatility: the estimate falls (floored at 0).
    s = real.corwin_schultz_spread(ar_daily([101.0, 103.0], [99.0, 101.0], [100.0, 102.0]))
    assert s.iloc[1] == pytest.approx(cs_by_hand(101.0, 99.0, 103.0, 101.0))
    assert s.iloc[1] == 0.0


def test_corwin_schultz_shifts_range_after_overnight_gap():
    # Day 1 closes at 98, below day 2's low of 99: day 2's 99-101 range is shifted
    # down to 98-100 before pairing, so the gap is not read as spread.
    daily = ar_daily([100.0, 101.0], [97.0, 99.0], [98.0, 100.0])
    s = real.corwin_schultz_spread(daily)
    assert s.iloc[1] == pytest.approx(cs_by_hand(100.0, 98.0, 100.0, 97.0))


def test_corwin_schultz_no_trade_day_reuses_last_range():
    # Day 2 has no volume and high == low: it takes day 1's 99-101 range (close 100
    # inside it), so the pair looks like two identical days.
    daily = ar_daily([101.0, 100.0], [99.0, 100.0], [100.0, 100.0], [1e3, 0.0])
    assert real.corwin_schultz_spread(daily).iloc[1] == pytest.approx(0.02)
    # Close above the kept range: the range moves up to end at the close.
    daily = ar_daily([101.0, 102.0], [99.0, 102.0], [100.0, 102.0], [1e3, 0.0])
    assert real.corwin_schultz_spread(daily).iloc[1] == pytest.approx(
        cs_by_hand(101.0, 99.0, 102.0, 100.0)
    )


def test_corwin_schultz_drops_bad_ranges_and_other_stocks():
    daily = pd.concat(
        [
            ar_daily([101.0, 900.0, 101.0], [99.0, 100.0, 99.0], [100.0, 100.0, 100.0]),
            ar_daily([101.0, 101.0], [99.0, 99.0], [100.0, 100.0], permaticker=2),
        ],
        ignore_index=True,
    )
    s = real.corwin_schultz_spread(daily)
    assert s.iloc[1:3].isna().all()  # high / low = 9 > 8: both pairs touching it
    assert np.isnan(s.iloc[3])  # stock 2's first day is not paired with stock 1
    assert s.iloc[4] == pytest.approx(0.02)


def test_monthly_corwin_schultz_terms_file_each_pair_under_its_later_day():
    stocks = price_rows(
        "AAA",
        ["2024-01-30", "2024-01-31", "2024-02-01"],
        [100.0, 100.0, 100.0],
        high=[101.0, 101.0, 101.0],
        low=[99.0, 99.0, 99.0],
    )
    out = real.build_monthly_returns(stocks, master_for("AAA")).set_index("month_end")
    for month in ["2024-01-31", "2024-02-29"]:  # one pair each
        row = out.loc[pd.Timestamp(month)]
        assert row["cs_n"] == 1 and row["cs_sum"] == pytest.approx(0.02)
