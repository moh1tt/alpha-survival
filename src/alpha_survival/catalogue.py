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
# OSAP acronym -> our pipeline feature, for signals the catalogue marks `existing`.
EXISTING_FEATURES = {
    "Mom12m": "mom_12_1",
    "Size": "log_mcap",
    "BM": "bm",
    "STreversal": "rev_1m",
    "GP": "gp_a",
    "AssetGrowth": "asset_growth",
}
# Every built feature that replicates an OSAP signal: feature -> OSAP acronym, in
# the order features enter the models.
FEATURE_ACRONYMS = {
    **{feature: acronym for acronym, feature in EXISTING_FEATURES.items()},
    # Step 47: momentum and reversal
    "mom_6m": "Mom6m",
    "int_mom": "IntMom",
    "mr_reversal": "MRreversal",
    "lr_reversal": "LRreversal",
    "mom_rev": "MomRev",
    # Step 48: seasonality
    "mom_season_short": "MomSeasonShort",
    "mom_season": "MomSeason",
    "mom_season_6y": "MomSeason06YrPlus",
    "mom_season_11y": "MomSeason11YrPlus",
    "mom_season_16y": "MomSeason16YrPlus",
    "mom_offseason_12m": "Mom12mOffSeason",
    "mom_offseason": "MomOffSeason",
    "mom_offseason_6y": "MomOffSeason06YrPlus",
    "mom_offseason_11y": "MomOffSeason11YrPlus",
    "mom_offseason_16y": "MomOffSeason16YrPlus",
    # Step 50: daily-price signals
    "max_ret": "MaxRet",
    "return_skew": "ReturnSkew",
    "high_52": "High52",
    "illiquidity": "Illiquidity",
    "dol_vol": "DolVol",
    "vol_sd": "VolSD",
    "bid_ask_spread": "BidAskSpread",
    # Step 52: volume trend, price, firm age
    "volume_trend": "VolumeTrend",
    "price": "Price",
    "firm_age": "FirmAge",
    # Step 54: valuation ratios
    "am": "AM",
    "sp": "SP",
    "leverage": "Leverage",
    "cf": "CF",
    "cfp": "cfp",
    "rd": "RD",
    "ep": "EP",
    "cash_prod": "CashProd",
    "ent_mult": "EntMult",
    # Step 55: book, payout and net debt ratios
    "bm_dec": "BMdec",
    "payout_yield": "PayoutYield",
    "net_payout_yield": "NetPayoutYield",
    "net_debt_price": "NetDebtPrice",
    "ebm": "EBM",
    "bpebm": "BPEBM",
    # Step 56: profitability
    "roe": "RoE",
    "oper_prof": "OperProf",
    "oper_prof_rd": "OperProfRD",
    "cb_oper_prof": "CBOperProf",
    "op_leverage": "OPLeverage",
    "tax": "Tax",
    "tang": "tang",
    # Step 57a: accruals and net operating assets
    "accruals": "Accruals",
    "pct_acc": "PctAcc",
    "pct_tot_acc": "PctTotAcc",
    "total_accruals": "TotalAccruals",
    "noa": "NOA",
    "d_noa": "dNoa",
    # Step 57b: balance-sheet changes
    "ch_nncoa": "ChNNCOA",
    "ch_nwc": "ChNWC",
    "del_coa": "DelCOA",
    "del_col": "DelCOL",
    "del_drc": "DelDRC",
    "del_equ": "DelEqu",
    "del_lti": "DelLTI",
    "del_finl": "DelFINL",
    "del_net_fin": "DelNetFin",
    "ch_inv": "ChInv",
    "ch_eq": "ChEQ",
    "ch_asset_turnover": "ChAssetTurnover",
    "invest_ppe_inv": "InvestPPEInv",
    # Step 58: investment and growth
    "investment": "Investment",
    "grcapx": "grcapx",
    "grcapx3y": "grcapx3y",
    "gr_sale_to_gr_inv": "GrSaleToGrInv",
    "gr_sale_to_gr_overhead": "GrSaleToGrOverhead",
    "surprise_rd": "SurpriseRD",
    # Step 59: financing
    "composite_debt_issuance": "CompositeDebtIssuance",
    "debt_issuance": "DebtIssuance",
    "net_debt_finance": "NetDebtFinance",
    "net_equity_finance": "NetEquityFinance",
    "xfin": "XFIN",
    "share_iss_1y": "ShareIss1Y",
    "share_iss_5y": "ShareIss5Y",
    "share_repurchase": "ShareRepurchase",
    # Step 60: turnover and volume
    "share_vol": "ShareVol",
    "vol_mkt": "VolMkt",
    "std_turn": "std_turn",
    "zerotrade_1m": "zerotrade1M",
    "zerotrade_6m": "zerotrade6M",
    "zerotrade_12m": "zerotrade12M",
    "mom_vol": "MomVol",
    # Step 61b: quarterly
    "roaq": "roaq",
    "cash": "Cash",
    "ch_tax": "ChTax",
    "num_earn_increase": "NumEarnIncrease",
    # Step 62: book leverage
    "book_leverage": "BookLeverage",
    # Step 65: monthly factor regressions
    "beta": "Beta",
    "beta_liquidity_ps": "BetaLiquidityPS",
    # Step 66: residual momentum and coskewness
    "residual_momentum": "ResidualMomentum",
    "coskewness": "Coskewness",
    # Step 67: daily factor regressions
    "idio_vol_3f": "IdioVol3F",
    "return_skew_3f": "ReturnSkew3F",
    "realized_vol": "RealizedVol",
    "idio_vol_aht": "IdioVolAHT",
    "beta_vix": "betaVIX",
    # Step 68: daily log-return coskewness and Frazzini-Pedersen beta
    "coskew_acx": "CoskewACX",
    "beta_fp": "BetaFP",
    # Step 69: tail risk beta
    "beta_tail_risk": "BetaTailRisk",
    # Step 70: price delay
    "price_delay_rsq": "PriceDelayRsq",
    "price_delay_slope": "PriceDelaySlope",
    "price_delay_tstat": "PriceDelayTstat",
    # Step 71: industry momentum and big-firm industry return
    "ind_mom": "IndMom",
    "ind_ret_big": "IndRetBig",
    # Step 72: industry concentration and industry-adjusted investment change
    "herf": "Herf",
    "herf_asset": "HerfAsset",
    "herf_be": "HerfBE",
    "ch_inv_ia": "ChInvIA",
    # Step 73: abnormal accruals
    "abnormal_accruals": "AbnormalAccruals",
    # Step 74: earnings and revenue surprises
    "earnings_surprise": "EarningsSurprise",
    "revenue_surprise": "RevenueSurprise",
    "earn_sup_big": "EarnSupBig",
    "earnings_consistency": "EarningsConsistency",
    # Step 75: dividends
    "div_init": "DivInit",
    "div_omit": "DivOmit",
    "div_season": "DivSeason",
    "div_yield_st": "DivYieldST",
    # Step 76: exchange switches and spin-offs
    "exch_switch": "ExchSwitch",
    "spinoff": "Spinoff",
}


def published_signs(path: Path = CATALOGUE_PATH) -> dict[str, int]:
    """feature -> the sign its original paper published (from the catalogue)."""
    signs = load_catalogue(path).set_index("acronym")["sign"]
    return {feature: int(signs[acronym]) for feature, acronym in FEATURE_ACRONYMS.items()}


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
