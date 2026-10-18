"""Tests for the data-quality checks. Small hand-made panels, no files or network."""

import numpy as np
import pandas as pd
import pytest

from alpha_survival.data import quality


def panel_rows(**overrides):
    """Two stocks over two months; every check passes unless overridden."""
    df = pd.DataFrame(
        {
            "permaticker": [1, 1, 2, 2],
            "month_end": pd.to_datetime(["2024-01-31", "2024-02-29"] * 2),
            "ret": [0.01, 0.02, -0.01, 0.03],
            "last_trade_date": pd.to_datetime(["2024-01-31", "2024-02-29"] * 2),
            "fwd_ret": [0.02, None, 0.03, None],
            "filing_date": pd.to_datetime(["2024-01-10"] * 4),
            "revenue": [500.0, 500.0, 70.0, 70.0],
            "investable": [True, True, True, True],
        }
    )
    for col, values in overrides.items():
        df[col] = values
    return df


def test_coverage_counts_stocks_and_shares():
    panel = panel_rows(filing_date=pd.to_datetime(["2024-01-10", "2024-01-10", None, None]))
    cov = quality.coverage_by_month(panel)
    assert list(cov["stocks"]) == [2, 2]
    assert list(cov["has_target"]) == [1.0, 0.0]  # last month: target not known yet
    assert list(cov["has_fundamentals"]) == [0.5, 0.5]


def test_missing_by_column_skips_complete_columns():
    missing = quality.missing_by_column(panel_rows(revenue=[500.0, None, 70.0, 70.0]))
    assert missing["fwd_ret"] == pytest.approx(0.5)
    assert missing["revenue"] == pytest.approx(0.25)
    assert "ret" not in missing


def test_weekend_month_end_is_not_a_stale_price():
    # Mar 31 2024 is a Sunday; last trade on Friday Mar 29 is normal.
    panel = panel_rows(
        month_end=pd.to_datetime(["2024-03-31"] * 4),
        last_trade_date=pd.to_datetime(["2024-03-29", "2024-03-29", "2024-03-29", "2024-03-01"]),
    )
    stale = quality.stale_prices(panel)
    assert list(stale.index) == [3]


def test_old_filing_is_flagged():
    panel = panel_rows(filing_date=pd.to_datetime(["2024-01-10", "2024-01-10", "2023-06-01", None]))
    stale = quality.stale_filings(panel)
    assert list(stale.index) == [2]  # missing filing is a coverage issue, not staleness


def test_extreme_returns_outside_band():
    stale = quality.extreme_returns(panel_rows(ret=[0.01, 1.5, -0.6, -0.5]))
    assert list(stale["ret"]) == [1.5, -0.6]  # -50% exactly is inside the band


def test_report_has_every_section():
    report = quality.render_report(panel_rows())
    for title in [
        "Coverage by year",
        "Flagged rows by year",
        "Missing values",
        "Stale prices",
        "Stale filings",
    ]:
        assert f"## {title}" in report
    assert "Rows: 4; stocks: 2" in report


def test_markdown_table_formats_cells():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-31"]), "x": [0.123], "n": [3], "m": [None]})
    assert quality.markdown_table(df).splitlines() == [
        "| d | x | n | m |",
        "| --- | --- | --- | --- |",
        "| 2024-01-31 | 0.12 | 3 |  |",
    ]
    assert quality.markdown_table(df.iloc[:0]) == "None."


def test_flags_by_year_counts_each_check():
    prices = pd.DataFrame({"month_end": pd.to_datetime(["2023-01-31", "2024-01-31"])})
    extremes = pd.DataFrame({"month_end": pd.to_datetime(["2024-03-31"] * 2)})
    out = quality.flags_by_year({"stale_prices": prices, "extreme_returns": extremes})
    assert out.to_dict("list") == {
        "year": [2023, 2024],
        "stale_prices": [1, 1],
        "extreme_returns": [0, 2],
    }


def test_worst_ranks_falls_with_rises_in_log_terms():
    extremes = quality.extreme_returns(
        panel_rows(permaticker=[1, 2, 3, 4], ret=[1.5, -0.99, -0.6, 0.0])
    )
    top = quality.worst(extremes, np.log1p(extremes["ret"]).abs(), n=2)
    assert list(top["ret"]) == [-0.99, 1.5]  # -99% is a bigger move than +150%


def test_coverage_by_year_condenses_months():
    cov = quality.coverage_by_year(quality.coverage_by_month(panel_rows()))
    assert list(cov["year"]) == [2024]
    assert cov.loc[0, "stocks_max"] == 2
    assert cov.loc[0, "has_target"] == pytest.approx(0.5)


def test_worst_shows_each_stock_once():
    extremes = quality.extreme_returns(panel_rows(ret=[2.0, 3.0, 1.5, 0.0]))
    top = quality.worst(extremes, extremes["ret"], n=5)
    assert list(top["permaticker"]) == [1, 2]
    assert list(top["ret"]) == [3.0, 1.5]  # stock 1's worst month only


def test_report_counts_extremes_only_among_investable_stocks():
    # Stock 2 jumps +500% next month but is a penny stock in January: not counted.
    panel = panel_rows(fwd_ret=[2.0, None, 5.0, None], investable=[True, True, False, True])
    report = quality.render_report(panel)
    assert "Investable rows (price >= $5 at month end): 3 (75%)" in report
    assert "extreme next-month returns: 1" in report
