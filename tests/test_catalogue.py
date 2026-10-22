from pathlib import Path

import pandas as pd
import pytest

from alpha_survival.catalogue import (
    EXISTING_FEATURES,
    EXTERNAL_SOURCES,
    FEATURE_ACRONYMS,
    SHARADAR_TABLES,
    STATUSES,
    USABLE,
    load_catalogue,
    parse_sources,
)
from alpha_survival.run import PUBLISHED_SIGN, RAW_FEATURES

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="module")
def cat():
    return load_catalogue()


def test_one_row_per_osap_predictor(cat):
    assert len(cat) == 212
    assert cat["acronym"].is_unique
    assert set(cat["status"]) <= set(STATUSES)
    assert set(cat["sign"]) == {-1, 1}
    assert (cat["reference"].str.len() > 0).all()


def test_usable_signals_have_a_batch_and_inputs_others_say_why_not(cat):
    usable = cat[cat["status"].isin(USABLE)]
    assert usable["batch"].isin([1, 2, 3]).all()
    assert (usable["inputs"] != "").all()
    other = cat[~cat["status"].isin(USABLE)]
    assert other["batch"].isna().all()
    assert (cat.loc[cat["status"] == "not_available", "note"] != "").all()
    # An approximation must say what it substitutes.
    assert (cat.loc[cat["status"] == "approximate", "note"] != "").all()


def test_existing_rows_are_our_features_with_our_published_signs(cat):
    existing = cat[cat["status"] == "existing"].set_index("acronym")
    assert set(existing.index) == set(EXISTING_FEATURES)
    for acronym, feature in EXISTING_FEATURES.items():
        assert feature in RAW_FEATURES
        assert existing.loc[acronym, "sign"] == PUBLISHED_SIGN[feature]


def test_every_built_feature_replicates_a_usable_catalogue_signal(cat):
    status = cat.set_index("acronym")["status"]
    assert set(FEATURE_ACRONYMS.values()) <= set(status.index)
    assert status[list(FEATURE_ACRONYMS.values())].isin(["existing", *USABLE]).all()
    assert len(set(FEATURE_ACRONYMS.values())) == len(FEATURE_ACRONYMS)  # one each


def test_inputs_name_only_known_tables_and_sources(cat):
    for spec in cat["inputs"]:
        assert set(parse_sources(spec)) <= set(SHARADAR_TABLES), spec
    for spec in cat["external"]:
        assert set(parse_sources(spec)) <= set(EXTERNAL_SOURCES), spec


def test_parse_sources():
    spec = "sep:closeadj; sf1:assets,revenue (quarterly); tickers:siccode"
    assert parse_sources(spec) == {
        "sep": ["closeadj"],
        "sf1": ["assets", "revenue"],
        "tickers": ["siccode"],
    }
    assert parse_sources("sf1:equityusd; sep") == {"sf1": ["equityusd"], "sep": []}
    assert parse_sources("") == {}


@pytest.mark.skipif(not (RAW / "fundamentals.parquet").exists(), reason="needs local Sharadar data")
def test_every_input_field_exists_in_our_sharadar_data(cat):
    import pyarrow.parquet as pq

    columns = {
        "sep": set(pq.read_schema(RAW / "stocks.parquet").names),
        "sf1": set(pq.read_schema(RAW / "fundamentals.parquet").names),
        "tickers": set(pq.read_schema(RAW / "tickers.parquet").names),
        "actions": set(pd.read_parquet(RAW / "actions.parquet", columns=["action"])["action"]),
    }
    for row in cat.itertuples():
        for table, fields in parse_sources(row.inputs).items():
            missing = set(fields) - columns[table]
            assert not missing, f"{row.acronym}: {table} lacks {missing}"
