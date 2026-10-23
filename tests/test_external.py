"""Tests for the external-series downloader. No network: fake files shaped like the real ones."""

import io
import zipfile

import pandas as pd
import pytest

from alpha_survival.data import external


def french_zip(body: str, name: str = "F-F_Research_Data_Factors.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, body.replace("\n", "\r\n"))  # the real files have CRLF endings
    return buf.getvalue()


FRENCH_MONTHLY = """This file was created using the 202608 CRSP database.
The 1-month TBill rate data until 202405 are from Ibbotson Associates.

,Mkt-RF,SMB,HML,RF
202606,   2.89,  -2.42,  -2.75,   0.22
202607,   2.64,  -1.44,   4.13,   0.25

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
  2025,  13.30, -10.79,   8.71,   4.25

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""

FRENCH_DAILY = """This file was created by using the 202608 CRSP database.

,Mkt-RF,SMB,HML,RF
20260730,    0.09,   -0.23,   -0.28,    0.01
20260803,    0.45,   -0.34,   -0.03,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""

VIX_CSV = b"observation_date,VIXCLS\n2026-07-30,16.68\n2026-07-31,\n2026-08-03,19.03\n"
GNPDEF_CSV = b"observation_date,GNPDEF\n2026-01-01,131.345\n2026-04-01,133.315\n"
PASTOR_TXT = (
    b"% LIQUIDITY FACTORS OF PASTOR AND STAMBAUGH (JPE 2003)\r\n"
    b"% Month\tAgg Liq.\tInnov Liq (eq8)\tTraded Liq (LIQ_V)\r\n"
    b"196208\t -0.01753789\t  0.00425841\t-99\r\n"
    b"202512\t -0.04086720\t -0.00585127\t -0.03618302\t\r\n"
)


def test_french_monthly_first_table_only_in_decimals():
    df = external.parse_french(french_zip(FRENCH_MONTHLY), daily=False)
    # The annual table below the monthly one must not leak in.
    assert df["date"].tolist() == [pd.Timestamp("2026-06-30"), pd.Timestamp("2026-07-31")]
    assert df.loc[0, "mktrf"] == pytest.approx(0.0289)
    assert df.loc[1, "rf"] == pytest.approx(0.0025)
    assert list(df.columns) == ["date", "available", "mktrf", "smb", "hml", "rf"]


def test_french_daily_available_at_its_month_end():
    df = external.parse_french(french_zip(FRENCH_DAILY), daily=True)
    assert df["date"].tolist() == [pd.Timestamp("2026-07-30"), pd.Timestamp("2026-08-03")]
    assert df["available"].tolist() == [pd.Timestamp("2026-07-31"), pd.Timestamp("2026-08-31")]
    assert df.loc[1, "smb"] == pytest.approx(-0.0034)


def test_french_missing_column_raises():
    bad = "header text\n\n,Mkt-RF,SMB,HML\n202606,   2.89,  -2.42,  -2.75\n"
    with pytest.raises(ValueError, match="rf"):
        external.parse_french(french_zip(bad), daily=False)


def test_fred_drops_blank_days():
    df = external.parse_fred(VIX_CSV, "VIXCLS")
    assert df["date"].tolist() == [pd.Timestamp("2026-07-30"), pd.Timestamp("2026-08-03")]
    assert df["value"].tolist() == [16.68, 19.03]


def test_fred_wrong_series_raises():
    with pytest.raises(ValueError, match="GNPDEF"):
        external.parse_fred(VIX_CSV, "GNPDEF")


def test_gnpdef_available_one_quarter_after_its_quarter_ends():
    df = external.parse_fred(GNPDEF_CSV, "GNPDEF")
    # FRED dates Q2 2026 as 2026-04-01; the quarter ends 06-30 and is published in late July.
    assert df["available"].tolist() == [pd.Timestamp("2026-06-30"), pd.Timestamp("2026-09-30")]


def test_pastor_parses_tabs_and_missing_code():
    df = external.parse_pastor(PASTOR_TXT)
    assert df["date"].tolist() == [pd.Timestamp("1962-08-31"), pd.Timestamp("2025-12-31")]
    assert pd.isna(df.loc[0, "traded_liq"])
    assert df.loc[1, "innov_liq"] == pytest.approx(-0.00585127)


def test_as_of_never_returns_values_not_yet_known():
    """Leakage: at each month end, only values available by then are visible."""
    gnpdef = external.parse_fred(GNPDEF_CSV, "GNPDEF")
    # Q2 ends 06-30, but its deflator is not out yet at 06-30 or 08-31.
    for month_end in ["2026-06-30", "2026-08-31"]:
        seen = external.as_of(gnpdef, pd.Timestamp(month_end))
        assert seen["date"].tolist() == [pd.Timestamp("2026-01-01")]
    assert len(external.as_of(gnpdef, pd.Timestamp("2026-09-30"))) == 2

    daily = external.parse_french(french_zip(FRENCH_DAILY), daily=True)
    seen = external.as_of(daily, pd.Timestamp("2026-07-31"))
    assert seen["date"].max() <= pd.Timestamp("2026-07-31")
    assert len(seen) == 1


def test_download_all_writes_every_table(tmp_path):
    files = {
        external.FRENCH_URL.format(name="F-F_Research_Data_Factors_daily"): french_zip(
            FRENCH_DAILY
        ),
        external.FRENCH_URL.format(name="F-F_Research_Data_Factors"): french_zip(FRENCH_MONTHLY),
        external.FRED_URL.format(series="VIXCLS"): VIX_CSV,
        external.FRED_URL.format(series="GNPDEF"): GNPDEF_CSV,
        external.PASTOR_URL: PASTOR_TXT,
    }
    paths = external.download_all(tmp_path, fetch=files.__getitem__)
    assert sorted(p.stem for p in paths) == sorted(
        ["ff3_daily", "ff3_monthly", "vix", "gnpdef", "ps_liquidity"]
    )
    vix = external.load("vix", tmp_path)
    assert pd.api.types.is_datetime64_any_dtype(vix["available"])
    assert len(vix) == 2
