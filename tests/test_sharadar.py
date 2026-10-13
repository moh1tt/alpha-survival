"""Tests for the Sharadar downloader. No network: a fake fetcher stands in for the API."""

import pandas as pd
import pytest

from alpha_survival.data import sharadar


def fake_sep(datatable_code, **options):
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "close": [185.6, 184.3],
            "closeadj": [184.9, 183.6],
            "volume": [82_488_700, 58_414_500],
        }
    )


def test_download_writes_parquet_roundtrip(tmp_path):
    path = sharadar.download_table("SEP", out_dir=tmp_path, fetch=fake_sep)
    assert path == tmp_path / "SEP.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(path), fake_sep("SHARADAR/SEP"))


def test_filters_and_table_code_are_passed_to_fetcher(tmp_path):
    calls = {}

    def spy(datatable_code, **options):
        calls["code"], calls["options"] = datatable_code, options
        return fake_sep(datatable_code)

    sharadar.download_table("SEP", out_dir=tmp_path, fetch=spy, ticker="AAPL,LEHMQ")
    assert calls["code"] == "SHARADAR/SEP"
    assert calls["options"] == {"paginate": True, "ticker": "AAPL,LEHMQ"}


def test_missing_required_column_fails(tmp_path):
    def no_datekey(datatable_code, **options):
        return pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "dimension": ["ART"],
                "reportperiod": ["2023-09-30"],
                "calendardate": ["2023-09-30"],
            }
        )

    with pytest.raises(ValueError, match="datekey"):
        sharadar.download_table("SF1", out_dir=tmp_path, fetch=no_datekey)
    assert not (tmp_path / "SF1.parquet").exists()


def test_empty_result_fails(tmp_path):
    with pytest.raises(ValueError, match="no rows"):
        sharadar.download_table("SEP", out_dir=tmp_path, fetch=lambda *a, **k: pd.DataFrame())


def test_unknown_table_fails(tmp_path):
    with pytest.raises(ValueError, match="Unknown table"):
        sharadar.download_table("NOPE", out_dir=tmp_path, fetch=fake_sep)


def test_missing_api_key_gives_clear_error(monkeypatch):
    # load_dotenv searches upward from the caller's file, so it would find a real
    # repo .env regardless of cwd. Stub it out: we're testing the error, not lookup.
    monkeypatch.setattr(sharadar, "load_dotenv", lambda: None)
    monkeypatch.delenv(sharadar.API_KEY_VAR, raising=False)
    with pytest.raises(RuntimeError, match=sharadar.API_KEY_VAR):
        sharadar.load_api_key()
