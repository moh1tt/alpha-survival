"""Data-quality checks on the monthly panel, rendered as a Markdown report.

Bad data rarely raises errors: a broken join or a stale price just produces
plausible-looking numbers. These checks make such problems visible before any
model sees the data. Run `python -m alpha_survival.data.quality` to regenerate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_survival.data.real import MIN_PRICE, load_panel

# A month-end price older than this (calendar days) is stale: halt, illiquidity or gap.
# 7 covers weekends plus a holiday.
STALE_PRICE_DAYS = 7
# Filings arrive quarterly (10-Q/10-K deadlines are 40-90 days after period end), so
# a latest filing older than this means the company stopped reporting or data is missing.
STALE_FILING_DAYS = 180
# Monthly moves outside this band are either real events or data errors: list them.
EXTREME_RETURN = (-0.5, 1.0)
# The full universe flags tens of thousands of rows: the report shows counts per
# year plus this many worst examples per check, never every row.
WORST_N = 20

REPORT_PATH = Path("reports/data_quality.md")


def coverage_by_month(panel: pd.DataFrame) -> pd.DataFrame:
    """Per month: stock count, and the share of stocks with a return, a target, fundamentals."""
    has = panel.assign(
        has_ret=panel["ret"].notna(),
        has_target=panel["fwd_ret"].notna(),
        has_fundamentals=panel["filing_date"].notna(),
    )
    return has.groupby("month_end").agg(
        stocks=("permaticker", "nunique"),
        has_ret=("has_ret", "mean"),
        has_target=("has_target", "mean"),
        has_fundamentals=("has_fundamentals", "mean"),
    )


def missing_by_column(panel: pd.DataFrame) -> pd.Series:
    """Share of missing values per column, worst first; complete columns left out."""
    missing = panel.isna().mean().sort_values(ascending=False)
    return missing[missing > 0]


def stale_prices(panel: pd.DataFrame, max_days: int = STALE_PRICE_DAYS) -> pd.DataFrame:
    """Rows whose last trade was more than `max_days` before month end."""
    age = (panel["month_end"] - panel["last_trade_date"]).dt.days
    out = panel.loc[age > max_days, ["permaticker", "month_end", "last_trade_date"]]
    return out.assign(age_days=age[age > max_days].astype(int))


def stale_filings(panel: pd.DataFrame, max_days: int = STALE_FILING_DAYS) -> pd.DataFrame:
    """Rows whose latest filing is more than `max_days` old at month end."""
    age = (panel["month_end"] - panel["filing_date"]).dt.days
    out = panel.loc[age > max_days, ["permaticker", "month_end", "filing_date"]]
    return out.assign(age_days=age[age > max_days].astype(int))


def extreme_returns(
    panel: pd.DataFrame, band: tuple[float, float] = EXTREME_RETURN
) -> pd.DataFrame:
    """Rows whose monthly return falls outside `band`."""
    low, high = band
    hit = (panel["ret"] < low) | (panel["ret"] > high)
    return panel.loc[hit, ["permaticker", "month_end", "ret"]]


def coverage_by_year(coverage: pd.DataFrame) -> pd.DataFrame:
    """Condense monthly coverage to one row per year: stock-count range and mean shares."""
    by_year = coverage.groupby(coverage.index.year)
    out = by_year["stocks"].agg(stocks_min="min", stocks_median="median", stocks_max="max")
    out = out.join(by_year[["has_ret", "has_target", "has_fundamentals"]].mean())
    return out.rename_axis("year").reset_index()


def flags_by_year(flags: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count each check's flagged rows per year, so a bad vintage stands out."""
    counts = {name: rows.groupby(rows["month_end"].dt.year).size() for name, rows in flags.items()}
    out = pd.DataFrame(counts).fillna(0).astype(int)
    return out.rename_axis("year").reset_index()


def worst(rows: pd.DataFrame, key: pd.Series, n: int = WORST_N) -> pd.DataFrame:
    """The `n` stocks with the largest `key`, one row each (their worst): examples to
    inspect, not the full list. One row per stock stops a single bad history filling it."""
    ranked = rows.loc[key.sort_values(ascending=False).index]
    return ranked.drop_duplicates("permaticker").head(n)


def markdown_table(df: pd.DataFrame) -> str:
    """A plain Markdown table: dates as YYYY-MM-DD, floats to 2 decimals, None if empty."""
    if df.empty:
        return "None."

    def cell(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, pd.Timestamp):
            return f"{value:%Y-%m-%d}"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    rows = [" | ".join(df.columns), " | ".join("---" for _ in df.columns)]
    rows += [" | ".join(cell(v) for v in row) for row in df.itertuples(index=False)]
    return "\n".join(f"| {row} |" for row in rows)


def render_report(panel: pd.DataFrame) -> str:
    """All checks as one Markdown document."""
    coverage = coverage_by_month(panel)
    missing = missing_by_column(panel)
    prices, filings, extremes = stale_prices(panel), stale_filings(panel), extreme_returns(panel)
    low, high = EXTREME_RETURN
    # What a portfolio would earn: next month's return, on stocks it may buy now.
    buyable = panel[panel["investable"]]
    buyable_extremes = (buyable["fwd_ret"] < low) | (buyable["fwd_ret"] > high)

    lines = [
        "# Data-quality report",
        "",
        f"- Rows: {len(panel):,}; stocks: {panel['permaticker'].nunique():,}; "
        f"months: {panel['month_end'].min():%Y-%m} to {panel['month_end'].max():%Y-%m}",
        f"- Stale prices (last trade > {STALE_PRICE_DAYS} days before month end): {len(prices):,}",
        f"- Stale filings (latest filing > {STALE_FILING_DAYS} days old): {len(filings):,}",
        f"- Extreme returns (below {low:.0%} or above {high:.0%}): {len(extremes):,}",
        f"- Investable rows (price >= ${MIN_PRICE:g} at month end): {len(buyable):,} "
        f"({len(buyable) / len(panel):.0%}); of these, extreme next-month returns: "
        f"{buyable_extremes.sum():,}",
        "",
        "## Coverage by year",
        "",
        "Stocks per month (min / median / max) and mean share of rows with each field.",
        "",
        markdown_table(coverage_by_year(coverage)),
        "",
        "## Missing values by column",
        "",
        markdown_table(missing.rename_axis("column").rename("share_missing").reset_index()),
        "",
        "## Flagged rows by year",
        "",
        markdown_table(
            flags_by_year(
                {"stale_prices": prices, "stale_filings": filings, "extreme_returns": extremes}
            )
        ),
    ]
    examples = [
        ("Stale prices", prices, prices["age_days"]),
        ("Stale filings", filings, filings["age_days"]),
        # Size of the move in log terms, so -99% ranks with +9,900%.
        ("Extreme returns", extremes, np.log1p(extremes["ret"]).abs()),
    ]
    for title, rows, key in examples:
        lines += ["", f"## {title}: {WORST_N} worst stocks", "", markdown_table(worst(rows, key))]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the data-quality report.")
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(load_panel()), encoding="utf-8")
    print(f"Report -> {args.out}")


if __name__ == "__main__":
    main()
