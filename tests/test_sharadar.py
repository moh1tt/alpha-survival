"""Tests for the Sharadar downloader. No network: a fake fetcher stands in for the API."""

import zipfile

import pandas as pd
import pytest

from alpha_survival.data import sharadar


def fake_stocks(api_table, params):
    # Shaped like the direct API's CSV: dates arrive as strings.
    return pd.DataFrame(
        {
            "ticker": [params["ticker"]] * 2,
            "date": ["2024-01-03", "2024-01-02"],
            "close": [184.3, 185.6],
            "closeadj": [183.6, 184.9],
            "volume": [58_414_500, 82_488_700],
        }
    )


def fake_fundamentals(api_table, params):
    return pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "dimension": ["ARQ"],
            "calendardate": ["2026-06-30"],
            "date": ["2026-07-31"],  # in fundamentals, `date` is the filing date
            "reportperiod": ["2026-06-27"],
            "revenue": [94_036_000_000],
        }
    )


def test_download_writes_parquet_with_parsed_dates(tmp_path):
    path = sharadar.download_table("stocks", ["AAPL"], out_dir=tmp_path, fetch=fake_stocks)
    assert path == tmp_path / "stocks.parquet"
    df = pd.read_parquet(path)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["close"].tolist() == [184.3, 185.6]


def test_one_request_per_ticker_with_full_history(tmp_path):
    calls = []

    def spy(api_table, params):
        calls.append((api_table, params))
        return fake_stocks(api_table, params)

    path = sharadar.download_table("stocks", ["AAPL", "MSFT"], out_dir=tmp_path, fetch=spy)
    assert [c[0] for c in calls] == ["stocks", "stocks"]
    assert [c[1]["ticker"] for c in calls] == ["AAPL", "MSFT"]
    # Without an explicit `from`, the API would return only the last year.
    assert all(c[1]["from"] == sharadar.HISTORY_START for c in calls)
    assert set(pd.read_parquet(path)["ticker"]) == {"AAPL", "MSFT"}


def test_fundamentals_saved_with_vendor_names_and_parsed_dates(tmp_path):
    path = sharadar.download_table("fundamentals", ["AAPL"], tmp_path, fetch=fake_fundamentals)
    df = pd.read_parquet(path)
    assert path == tmp_path / "fundamentals.parquet"
    assert list(df.columns) == list(fake_fundamentals("fundamentals", {}).columns)
    assert df.loc[0, "date"] == pd.Timestamp("2026-07-31")
    assert df.loc[0, "reportperiod"] == pd.Timestamp("2026-06-27")


def test_dimension_filter_is_sent_for_fundamentals_only(tmp_path):
    seen = {}

    def spy(api_table, params):
        seen.update(params)
        return fake_fundamentals(api_table, params)

    sharadar.download_table("fundamentals", ["AAPL"], tmp_path, fetch=spy, dimension="ARQ,ART")
    assert seen["dimension"] == "ARQ,ART"
    with pytest.raises(ValueError, match="only applies to fundamentals"):
        sharadar.download_table("stocks", ["AAPL"], tmp_path, fetch=fake_stocks, dimension="ARQ")


def test_tickers_saved_verbatim_without_date_filter(tmp_path):
    def fake_tickers(api_table, params):
        # `from` would filter `tickers` on a date it doesn't have; it must not be sent.
        assert "from" not in params
        return pd.DataFrame(
            {
                "table": ["stocks", "fundamentals", "insiders"],
                "permaticker": [199059] * 3,
                "ticker": ["AAPL"] * 3,
                "isdelisted": ["N"] * 3,
            }
        )

    path = sharadar.download_table("tickers", ["AAPL"], tmp_path, fetch=fake_tickers)
    assert pd.read_parquet(path)["table"].tolist() == ["stocks", "fundamentals", "insiders"]


def test_response_at_row_limit_is_treated_as_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(sharadar, "ROW_LIMIT", 2)
    with pytest.raises(ValueError, match="truncated"):
        sharadar.download_table("stocks", ["AAPL"], out_dir=tmp_path, fetch=fake_stocks)
    assert not (tmp_path / "stocks.parquet").exists()


def test_parse_csv_keeps_identifiers_as_text():
    # Found on the first real download: per-ticker type guessing made `cusips`
    # int for one stock and str for another, and Parquet refused the mix.
    a = sharadar.parse_csv("ticker,cusips,contraticker,value\nNA,037833100,N/A,\n")
    b = sharadar.parse_csv("ticker,cusips,contraticker,value\nV,92826C839,MA,1.5\n")
    both = pd.concat([a, b], ignore_index=True)
    assert both["ticker"].tolist() == ["NA", "V"]  # "NA" is a ticker, not missing
    assert both["cusips"].tolist() == ["037833100", "92826C839"]  # leading zero kept
    assert both["contraticker"].tolist() == ["N/A", "MA"]
    assert both["value"].isna().tolist() == [True, False]  # blanks are still missing


def test_parse_csv_empty_body():
    assert sharadar.parse_csv("").empty


def test_missing_required_column_fails(tmp_path):
    def no_filing_date(api_table, params):
        return fake_fundamentals(api_table, params).drop(columns="date")

    with pytest.raises(ValueError, match="'date'"):
        sharadar.download_table("fundamentals", ["AAPL"], tmp_path, fetch=no_filing_date)
    assert not (tmp_path / "fundamentals.parquet").exists()


def test_empty_result_fails(tmp_path):
    with pytest.raises(ValueError, match="no rows"):
        sharadar.download_table("stocks", ["AAPL"], tmp_path, fetch=lambda *a: pd.DataFrame())


def test_unknown_table_fails(tmp_path):
    with pytest.raises(ValueError, match="Unknown table"):
        sharadar.download_table("NOPE", ["AAPL"], out_dir=tmp_path, fetch=fake_stocks)


def test_no_tickers_fails(tmp_path):
    with pytest.raises(ValueError, match="at least one ticker"):
        sharadar.download_table("stocks", [], out_dir=tmp_path, fetch=fake_stocks)


def test_missing_api_key_gives_clear_error(monkeypatch):
    # load_dotenv searches upward from the caller's file, so it would find a real
    # repo .env regardless of cwd. Stub it out: we're testing the error, not lookup.
    monkeypatch.setattr(sharadar, "load_dotenv", lambda: None)
    monkeypatch.delenv(sharadar.API_KEY_VAR, raising=False)
    with pytest.raises(RuntimeError, match=sharadar.API_KEY_VAR):
        sharadar.load_api_key()


# --- bulk files ----------------------------------------------------------------


def bulk_zip(tmp_path, csv_text, name="SHARADAR_SEP.csv"):
    path = tmp_path / "stocks.csv.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name, csv_text)
    return path


BULK_STOCKS = (
    "ticker,date,close,closeadj,volume,cusips\n"
    "NA,2024-01-03,184.3,183.6,58414500,037833100\n"
    "V,2024-01-02,185.6,,82488700,92826C839\n"
)


def test_bulk_matches_row_query_typing(tmp_path):
    path = sharadar.convert_bulk_csv(bulk_zip(tmp_path, BULK_STOCKS), "stocks", tmp_path / "raw")
    df = pd.read_parquet(path)
    assert df["ticker"].tolist() == ["NA", "V"]  # "NA" is a ticker, not missing
    assert df["cusips"].tolist() == ["037833100", "92826C839"]  # leading zero kept
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["closeadj"].isna().tolist() == [False, True]  # blanks are missing
    assert df["volume"].dtype.kind in "iu"


def test_bulk_removes_extracted_csv_and_temp_file(tmp_path):
    out = tmp_path / "raw"
    sharadar.convert_bulk_csv(bulk_zip(tmp_path, BULK_STOCKS), "stocks", out)
    assert sorted(p.name for p in out.iterdir()) == ["stocks.parquet"]
    assert not (tmp_path / "SHARADAR_SEP.csv").exists()


def test_bulk_missing_required_column_fails_before_converting(tmp_path):
    zip_path = bulk_zip(tmp_path, "ticker,date,close\nAAPL,2024-01-03,184.3\n")
    with pytest.raises(ValueError, match="missing columns"):
        sharadar.convert_bulk_csv(zip_path, "stocks", tmp_path / "raw")
    assert not (tmp_path / "raw" / "stocks.parquet").exists()


def test_bulk_empty_file_fails_and_keeps_no_parquet(tmp_path):
    zip_path = bulk_zip(tmp_path, "ticker,date,close,closeadj,volume\n")
    with pytest.raises(ValueError, match="no rows"):
        sharadar.convert_bulk_csv(zip_path, "stocks", tmp_path / "raw")
    assert not any((tmp_path / "raw").iterdir())


def test_cli_needs_exactly_one_mode():
    with pytest.raises(SystemExit):
        sharadar.main([])
    with pytest.raises(SystemExit):
        sharadar.main(["--bulk", "--tickers", "AAPL"])
