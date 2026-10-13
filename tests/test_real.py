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
