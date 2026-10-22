"""Catalogue of candidate signals: which Open Source Asset Pricing predictors we can build.

`config/signal_catalogue.csv` lists all 212 OSAP predictors (Chen & Zimmermann 2022).
`sign` and `reference` are copied from OSAP's SignalDoc.csv; the rest is our
classification, fixed BEFORE any of these signals is computed or tested, so the
list cannot be shaped by results:

- `status`: `existing` (already a pipeline feature), `computable` (every input is
  in Sharadar or a free source in the project brief), `approximate` (a minor input
  is missing; `note` says what replaces it) or `not_available`.
- `batch`: build order by implementation cost, not by expected performance:
  1 = daily prices only, 2 = simple accounting ratios, 3 = regressions,
  industries, events or external data.
- `inputs`: `table:field,field; ...` with tables sep, sf1, tickers, actions
  (for actions, the fields are action types). `external`: `source:series; ...`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CATALOGUE_PATH = Path(__file__).resolve().parents[2] / "config" / "signal_catalogue.csv"
STATUSES = ("existing", "computable", "approximate", "not_available")
USABLE = ("computable", "approximate")
SHARADAR_TABLES = ("sep", "sf1", "tickers", "actions")
EXTERNAL_SOURCES = ("kenfrench", "fred", "pastor")
# OSAP acronym -> our pipeline feature, for signals we already build.
EXISTING_FEATURES = {
    "Mom12m": "mom_12_1",
    "Size": "log_mcap",
    "BM": "bm",
    "STreversal": "rev_1m",
    "GP": "gp_a",
    "AssetGrowth": "asset_growth",
}


def load_catalogue(path: Path = CATALOGUE_PATH) -> pd.DataFrame:
    """The catalogue as a DataFrame; `batch` is a nullable integer (NA if not usable)."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table["sign"] = table["sign"].astype(int)
    table["batch"] = pd.to_numeric(table["batch"].replace("", None)).astype("Int64")
    return table


def parse_sources(spec: str) -> dict[str, list[str]]:
    """'sep:closeadj; sf1:assets,revenue (quarterly)' -> {'sep': [...], 'sf1': [...]}.

    A bare table name ('sep') means "prices, field not named". Remarks in
    parentheses are dropped.
    """
    out: dict[str, list[str]] = {}
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        source, _, fields = part.partition(":")
        fields = fields.split("(")[0]
        out.setdefault(source.strip(), []).extend(f.strip() for f in fields.split(",") if f.strip())
    return out
