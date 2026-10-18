"""Tests for the data-quality checks. Small hand-made panels, no files or network."""

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
    for title in ["Coverage by month", "Missing values", "Stale prices", "Stale filings"]:
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
