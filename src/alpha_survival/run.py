"""Run forecasters through walk-forward on the real panel and report out-of-sample scores.

    python -m alpha_survival.run      # -> reports/baselines.md

Only summary statistics are written; raw Sharadar data never leaves data/raw.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from alpha_survival.catalogue import FEATURE_ACRONYMS, published_signs
from alpha_survival.combine import rank_average
from alpha_survival.data.quality import markdown_table
from alpha_survival.data.real import load_panel
from alpha_survival.features import (
    add_accrual_signals,
    add_asset_growth,
    add_balance_sheet_changes,
    add_book_payout_debt_ratios,
    add_daily_price_signals,
    add_financing_signals,
    add_investment_growth,
    add_momentum,
    add_past_returns,
    add_profitability,
    add_profitability_ratios,
    add_quarterly_signals,
    add_reversal,
    add_seasonality,
    add_size,
    add_trend_price_age,
    add_turnover_signals,
    add_valuation_ratios,
    add_value,
    add_volatility,
    rank_normalise,
)
from alpha_survival.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    ic_summary,
    oos_r2,
    probabilistic_sharpe_ratio,
    rank_ic,
)
from alpha_survival.models import (
    ElasticNetForecaster,
    Forecaster,
    HistoricalMean,
    LGBMForecaster,
    SignalForecast,
    SklearnForecaster,
    ZeroForecast,
)
from alpha_survival.portfolio import (
    EXIT_GROUPS,
    WEIGHTINGS,
    breakeven_cost_bps,
    estimated_spread_bps,
    leg_weights,
    long_short_returns,
    net_returns,
    summarise,
    trades,
    trading_costs,
    turnover,
)
from alpha_survival.walkforward import walk_forward_predict

REPORT_PATH = Path("reports/baselines.md")
# Direction each signal was PUBLISHED with (+1: high value -> high return), from the
# signal catalogue (OSAP's SignalDoc), never from our test results: reading signs
# off the test period would be data snooping. `vol_3m` (total volatility over 3
# months) is not an OSAP signal: Ang, Hodrick, Xing & Zhang 2006.
PUBLISHED_SIGN = {**published_signs(), "vol_3m": -1}
# The original 7 features first (Steps 10-26), then the OSAP families (Steps 47-62).
ORIGINAL_FEATURES = ["mom_12_1", "log_mcap", "bm", "rev_1m", "gp_a", "asset_growth", "vol_3m"]
RAW_FEATURES = ORIGINAL_FEATURES + [f for f in FEATURE_ACRONYMS if f not in ORIGINAL_FEATURES]
FEATURES = [f"{c}_rank" for c in RAW_FEATURES]
# Rolling window: train only on the latest 5 years, so old regimes are forgotten.
ROLLING_WINDOW = 60
# Flat trading cost per dollar traded, in basis points: roughly large caps traded
# by a big institution (10) up to small caps (50).
COST_BPS = (10, 20, 50)
# Annual fee for borrowing the shorted stock. 50 bps is typical for easy-to-borrow
# names; small, heavily shorted stocks cost far more, so this is optimistic.
BORROW_BPS_PA = 50
# Eras for the spread table: before decimalisation (2001), then roughly 7-year blocks.
SPREAD_ERAS = [1996, 2000, 2008, 2015, 2026]
SPREAD_ERA_LABELS = ["1997-2000", "2001-2008", "2009-2015", "2016-2026"]
# Portfolios looked at (or tried) but not in this report. The deflated Sharpe ratio
# counts every strategy we have seen, not just the ones still shown: dropping rows
# from a report does not un-see them, and ignoring them flatters the winner.
EARLIER_TRIALS = {
    "single-signal portfolios on the original 7 features, built in Steps 27-45": 28,
    "ols, enet, lgbm and combo portfolios on the original 7 features (Steps 27-45)": 16,
    "ols, enet, lgbm and combo portfolios on 28 features (Step 51)": 16,
    "ridge and rolling-window OLS, equal- or value-weighted, on 7 and on 28 features and "
    "on this run's (tried, never built)": 12,
}
UNREPORTED_TRIALS = sum(EARLIER_TRIALS.values())
# Trial counts to show the deflated Sharpe ratio at, besides the honest count.
SENSITIVITY_TRIALS = (20, 100)
# Equal-weight rank combination of the learned models, members fixed before seeing
# how the combination does.
COMBO = "combo"
COMBO_MEMBERS = ["ols", "enet", "lgbm"]


def baseline_models() -> list[Forecaster]:
    """Baselines, every raw signal (scored by IC only), and the learned models."""
    return [
        ZeroForecast(),
        HistoricalMean(),
        *(SignalForecast(c) for c in RAW_FEATURES),
        SklearnForecaster(LinearRegression(), FEATURES, "ols"),
        SklearnForecaster(Ridge(alpha=1.0), FEATURES, "ridge"),
        ElasticNetForecaster(FEATURES),
        LGBMForecaster(FEATURES),
    ]


def rolling_models() -> list[Forecaster]:
    """Models rerun with a rolling window. Only ones that learn: a raw signal has no
    training set to roll, and an untuned alpha=1 Ridge is all but OLS here."""
    return [HistoricalMean(), SklearnForecaster(LinearRegression(), FEATURES, "ols")]


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    """Add features on the full panel, keep the rows that may be bought, rank them.

    Order matters: momentum needs a stock's history from months it was below the
    price cut, so features come first and the investable filter second. Ranks come
    last, so each stock is ranked among the stocks that could actually be bought.
    """
    panel = add_reversal(add_value(add_size(add_momentum(panel))))
    panel = add_volatility(add_asset_growth(add_profitability(panel)))
    panel = add_daily_price_signals(add_seasonality(add_past_returns(panel)))
    panel = add_valuation_ratios(add_trend_price_age(panel))
    panel = add_profitability_ratios(add_book_payout_debt_ratios(panel))
    panel = add_balance_sheet_changes(add_accrual_signals(panel))
    panel = add_financing_signals(add_investment_growth(panel))
    panel = add_quarterly_signals(add_turnover_signals(panel))
    panel = panel[panel["investable"]].reset_index(drop=True)
    return rank_normalise(panel, RAW_FEATURES)


def evaluate(
    panel: pd.DataFrame,
    models: list[Forecaster],
    keep: dict[str, pd.DataFrame] | None = None,
    fitted: dict[str, list[Forecaster]] | None = None,
    **split_kwargs,
) -> pd.DataFrame:
    """One row of out-of-sample scores per model.

    R^2 only for forecasts that are returns: a raw signal's scale is not a return.
    Rank IC only where forecasts differ across stocks: constant forecasts rank nothing.
    If `keep` is given, each model's out-of-sample forecasts are stored in it by name;
    if `fitted` is, each model's per-fold fitted copies.
    """
    rows = []
    for model in models:
        folds: list[Forecaster] = []
        oos = walk_forward_predict(panel, model, fitted=folds, **split_kwargs)
        if keep is not None:
            keep[model.name] = oos
        if fitted is not None:
            fitted[model.name] = folds
        is_return = not isinstance(model, SignalForecast)
        rows.append(score(oos, model.name, split_kwargs.get("window"), is_return))
    return pd.DataFrame(rows)


def score(oos: pd.DataFrame, name: str, window: int | None, is_return: bool) -> dict:
    """The out-of-sample scores of one set of forecasts (see `evaluate`)."""
    row = {
        "model": name,
        "window": "expanding" if window is None else f"rolling {window}m",
        "first_test_month": oos["month_end"].min().date(),
        "last_test_month": oos["month_end"].max().date(),
        "n_forecasts": len(oos.dropna(subset=["fwd_ret", "pred"])),
        "oos_r2_pct": float("nan"),
        "mean_ic": float("nan"),
        "ic_t_stat": float("nan"),
        "ic_nw_t_stat": float("nan"),
        "ic_autocorr_1": float("nan"),
        "ic_hit_rate": float("nan"),
        "ic_months": 0,
    }
    if row["n_forecasts"] == 0:  # e.g. a signal whose history starts after the tests end
        return row
    if is_return:
        row["oos_r2_pct"] = 100 * oos_r2(oos)
    ic = rank_ic(oos)
    if len(ic) >= 2:
        s = ic_summary(ic)
        row |= {
            "mean_ic": s["mean_ic"],
            "ic_t_stat": s["t_stat"],
            "ic_nw_t_stat": s["nw_t_stat"],
            "ic_autocorr_1": s["ic_autocorr_1"],
            "ic_hit_rate": s["hit_rate"],
            "ic_months": s["n_months"],
        }
    return row


SCORE_DIGITS = {
    "oos_r2_pct": 3,
    "mean_ic": 4,
    "ic_t_stat": 2,
    "ic_nw_t_stat": 2,
    "ic_autocorr_1": 2,
    "ic_hit_rate": 3,
}


def _fmt(table: pd.DataFrame, digits: dict[str, int]) -> pd.DataFrame:
    return table.assign(
        **{
            c: table[c].map(lambda v, d=d: "" if pd.isna(v) else f"{v:.{d}f}")
            for c, d in digits.items()
        }
    )


def render_report(table: pd.DataFrame, panel: pd.DataFrame) -> str:
    models = table[~table["model"].str.startswith("signal_")]
    shown = _fmt(models.assign(n_forecasts=models["n_forecasts"].map("{:,}".format)), SCORE_DIGITS)
    return "\n".join(
        [
            "# Forecasts: out-of-sample scores",
            "",
            "**Real data (Sharadar US equities), out-of-sample, walk-forward. "
            "No transaction costs; not a trading result.**",
            "",
            f"- Panel: investable rows (price >= $5 at month end), "
            f"{panel['month_end'].min():%Y-%m} to {panel['month_end'].max():%Y-%m}, "
            f"{len(panel):,} stock-months, {panel['permaticker'].nunique():,} stocks.",
            f"- Features: {len(RAW_FEATURES)} signals, each rank-normalised to [-1, 1] every "
            "month (missing = 0, the middle): the original 7 plus Open Source Asset Pricing "
            "replications (Chen & Zimmermann 2022). List and single-signal scores in the "
            "appendix at the end.",
            "- Models: `zero` and `hist_mean` (pooled training mean) baselines; `ols` and "
            "`ridge` (alpha=1, untuned) on all features; `enet` (ElasticNet, half Lasso half "
            "Ridge penalty, its strength picked from 1e-5 to 0.1 by MSE on the latest 20% of "
            "each training window, then refit on the whole window); `lgbm` (LightGBM "
            "gradient-boosted trees, fixed untuned settings: depth 4, 200 trees, learning "
            "rate 0.05, each leaf >= 0.1% of training rows), which can pick up nonlinear "
            "effects and interactions.",
            "- `combo`: each month, the forecasts of `ols`, `enet` and `lgbm` ranked to "
            "[-1, 1] and averaged with equal weights (fixed in advance, nothing fit). A rank "
            "score, not a return, so no R^2.",
            "- Splits: first 60 months train only, 12-month test blocks, model refit at the "
            "start of each block, horizon 1, embargo 0. `window`: `expanding` trains on all "
            f"months so far; `rolling {ROLLING_WINDOW}m` on the latest {ROLLING_WINDOW} only. "
            "Both start testing in the same month.",
            "- `oos_r2_pct`: pooled R^2 vs. a zero forecast, in percent (Gu, Kelly & Xiu 2020).",
            "- `mean_ic`: average monthly Spearman rank correlation of forecast with next-month "
            "return; months with < 10 stocks skipped. Blank for constant forecasts.",
            "- `ic_t_stat` treats months as independent; `ic_nw_t_stat` corrects for "
            "autocorrelated ICs (Newey & West 1987; lags floor(4 (n/100)^(2/9)), 5 for 285 "
            "months) and is the one to trust. `ic_autocorr_1`: correlation of each month's "
            "IC with the previous month's.",
            "",
            markdown_table(shown),
            "",
        ]
    )


def signal_table(table: pd.DataFrame) -> pd.DataFrame:
    """Single-signal IC rows with the published sign and whether the IC agrees with it."""
    signals = table[table["model"].str.startswith("signal_")].copy()
    signals["feature"] = signals["model"].str.removeprefix("signal_")
    signals["osap"] = signals["feature"].map(FEATURE_ACRONYMS).fillna("")
    signals["published_sign"] = signals["feature"].map(PUBLISHED_SIGN)
    agrees = np.sign(signals["mean_ic"]) == signals["published_sign"]
    signals["sign_agrees"] = agrees.map({True: "yes", False: "no"}).where(
        signals["mean_ic"].notna(), ""
    )
    cols = ["feature", "osap", "published_sign", "mean_ic", "ic_nw_t_stat", "ic_hit_rate"]
    return signals[[*cols, "ic_months", "sign_agrees"]].reset_index(drop=True)


def render_signals(table: pd.DataFrame) -> str:
    shown = _fmt(table, {"mean_ic": 4, "ic_nw_t_stat": 2, "ic_hit_rate": 3}).assign(
        published_sign=table["published_sign"].map("{:+d}".format)
    )
    agree = (table["sign_agrees"] == "yes").sum()
    strong = (table["sign_agrees"] == "yes") & (table["ic_nw_t_stat"].abs() >= 2)
    return "\n".join(
        [
            "## Appendix: single-signal scores",
            "",
            "**Real data, out-of-sample months (same as the models). Each raw signal used "
            "directly as a forecast; IC only, no portfolios.**",
            "",
            "- `osap`: the Open Source Asset Pricing signal it replicates (blank: our own). "
            "`published_sign`: the direction its original paper reported, from the signal "
            "catalogue; a NEGATIVE IC is the expected result for a `-1` signal.",
            "- `sign_agrees`: our out-of-sample IC has the published sign. "
            f"{agree} of {len(table)} agree; {strong.sum()} agree with |NW t| >= 2.",
            "- Signals are not traded on their own any more; the 28 single-signal "
            "portfolios built earlier still count as trials (see the deflated Sharpe ratio).",
            "",
            markdown_table(shown),
            "",
        ]
    )


def run_all(
    panel: pd.DataFrame,
    keep: dict[str, pd.DataFrame] | None = None,
    fitted: dict[str, list[Forecaster]] | None = None,
    **split_kwargs,
) -> pd.DataFrame:
    """Every model with an expanding window, their combination, then the learning
    ones with a rolling window.

    `keep` collects the expanding-window forecasts by model name (for portfolios),
    combination included; `fitted` the expanding-window per-fold fitted models.
    """
    keep = {} if keep is None else keep
    expanding = evaluate(panel, baseline_models(), keep=keep, fitted=fitted, **split_kwargs)
    keep[COMBO] = rank_average(keep, COMBO_MEMBERS)
    combo = score(keep[COMBO], COMBO, None, is_return=False)
    rolling = evaluate(panel, rolling_models(), window=ROLLING_WINDOW, **split_kwargs)
    return pd.concat([expanding, pd.DataFrame([combo]), rolling], ignore_index=True)


def penalty_table(folds: list[ElasticNetForecaster], oos: pd.DataFrame) -> pd.DataFrame:
    """What ElasticNet chose in each fold: test period, penalty, features kept.

    A fold that keeps no feature forecasts the same return for every stock: it has
    no view (and its portfolios hold nothing that year).
    """
    span = oos.groupby("fold")["month_end"].agg(["min", "max"])
    rows = [
        {
            "test_from": span.loc[fold, "min"].date(),
            "test_to": span.loc[fold, "max"].date(),
            "alpha": model.alpha_,
            "n_kept": int((model.estimator_.coef_ != 0).sum()),
        }
        for fold, model in enumerate(folds)
    ]
    return pd.DataFrame(rows)


def coefficient_table(folds: list[ElasticNetForecaster]) -> pd.DataFrame:
    """Per feature: in how many folds ElasticNet kept it, and its average coefficient.

    Coefficients are in bps per month for a move from the median stock (rank 0) to
    the top one (rank +1), others held fixed; averaged over all folds (a dropped
    feature counts as 0). Sorted by how often kept, then by size.
    """
    coefs = pd.DataFrame([m.estimator_.coef_ for m in folds], columns=RAW_FEATURES) * 1e4
    table = pd.DataFrame(
        {
            "feature": RAW_FEATURES,
            "published_sign": [PUBLISHED_SIGN[f] for f in RAW_FEATURES],
            "folds_kept": (coefs != 0).sum().to_numpy(),
            "mean_coef_bps": coefs.mean().to_numpy(),
        }
    )
    table["abs"] = table["mean_coef_bps"].abs()
    table = table.sort_values(["folds_kept", "abs"], ascending=False).drop(columns="abs")
    return table.reset_index(drop=True)


def render_penalties(table: pd.DataFrame, coefs: pd.DataFrame) -> str:
    no_view = table[table["n_kept"] == 0]
    shown = table.assign(alpha=table["alpha"].map("{:.0e}".format))
    shown_coefs = coefs.assign(
        published_sign=coefs["published_sign"].map("{:+d}".format),
        mean_coef_bps=coefs["mean_coef_bps"].map(lambda v: "0" if v == 0 else f"{v:+.1f}"),
    )
    return "\n".join(
        [
            "## ElasticNet: what it chose each year",
            "",
            "**Real data. Chosen in training only (latest 20% of each training window); "
            "the test year is never seen.**",
            "",
            "- One row per refit. `alpha`: penalty strength picked (1e-5 barely shrinks, 0.1 "
            f"zeroes everything). `n_kept`: features (of {len(RAW_FEATURES)}) with a non-zero "
            "coefficient.",
            f"- {len(no_view)} of {len(table)} refits kept no feature: in validation, nothing "
            "linear beat forecasting the average, so the model held no position that year.",
            "",
            markdown_table(shown),
            "",
            "Per feature, over all refits: `folds_kept` = refits with a non-zero coefficient; "
            "`mean_coef_bps` = average coefficient in bps per month from the median stock to "
            "the top one on that feature (dropped = 0), others held fixed.",
            "",
            markdown_table(shown_coefs),
            "",
        ]
    )


# Models traded as portfolios; their forecasts carry the directions learned in training.
LEARNED = ["ols", "enet", "lgbm", COMBO]
# How legs are formed: plain deciles every month, or a holding band (enter in the
# top/bottom decile, sell once out of the top/bottom EXIT_GROUPS deciles).
LEG_RULES = {"decile": None, "band": EXIT_GROUPS}


def portfolio_table(
    oos_by_model: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    keep_net: dict[tuple[str, str, str], pd.Series] | None = None,
    candidates: list[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    """Long-short performance for each (forecast name, sign) in `candidates`: EW and
    VW, each with plain decile legs and with a holding band (`LEG_RULES`).

    By default the learned models (`LEARNED`), sign +1: their forecasts already
    carry the directions learned in training. A signal is traded in its published
    direction (`PUBLISHED_SIGN`), never the one our test period showed.

    Gross stats, plus average turnover, net Sharpe at each flat `COST_BPS` and at
    each stock's estimated spread (`estimated_spread_bps`), all with the
    `BORROW_BPS_PA` short fee, and the break-even trading cost. Value weights use
    market cap at formation (month_end), known when the trade is placed. If
    `keep_net` is given, each portfolio's monthly returns net of estimated costs are
    stored in it by (portfolio, weighting, legs).
    """
    caps = panel[["permaticker", "month_end", "market_cap"]]
    spread = estimated_spread_bps(panel).set_index(["permaticker", "month_end"])["cost_bps"]
    if candidates is None:
        candidates = [(name, 1) for name in LEARNED]
    rows = []
    for name, sign in candidates:
        if name not in oos_by_model:
            continue
        oos = oos_by_model[name].merge(caps, on=["permaticker", "month_end"], how="left")
        oos = oos.assign(pred=sign * oos["pred"])
        for weighting in WEIGHTINGS:
            for legs, exit_groups in LEG_RULES.items():
                ls = long_short_returns(oos, weighting=weighting, exit_groups=exit_groups)
                gross = ls["long_short"]
                weights = leg_weights(oos, weighting=weighting, exit_groups=exit_groups)
                # Full calendar, so months that hold nothing still pay to sell the book.
                months = pd.DatetimeIndex(oos["month_end"].unique())
                trade_rows, to = trades(weights, months), turnover(weights, months)
                net = {}
                for c in COST_BPS:
                    net_ret = net_returns(gross, trading_costs(trade_rows, c), BORROW_BPS_PA)
                    net[f"sharpe_net_{c}bps"] = summarise(net_ret)["sharpe"]
                est_cost = trading_costs(trade_rows, spread).reindex(gross.index)
                net_est = net_returns(gross, est_cost, BORROW_BPS_PA)
                net["sharpe_net_est"] = summarise(net_est)["sharpe"]
                if keep_net is not None:
                    keep_net[(name, weighting, legs)] = net_est
                # Average cost paid per dollar traded: what the estimated spreads amount to.
                net["avg_cost_bps"] = float(est_cost.sum() / to.reindex(gross.index).sum() * 1e4)
                rows.append(
                    {
                        "portfolio": name,
                        "weighting": weighting,
                        "legs": legs,
                        **summarise(gross),
                        "avg_stocks_per_leg": float(ls[["n_long", "n_short"]].mean().mean()),
                        "avg_turnover": float(to.reindex(gross.index).mean()),
                        **net,
                        "breakeven_bps": breakeven_cost_bps(gross, to, BORROW_BPS_PA),
                    }
                )
    return pd.DataFrame(rows)


def render_portfolios(table: pd.DataFrame) -> str:
    keys = ["portfolio", "weighting", "legs"]
    gross_digits = {
        "ann_return": 4,
        "ann_vol": 4,
        "sharpe": 2,
        "nw_t_stat": 2,
        "max_drawdown": 3,
        "hit_rate": 3,
        "n_months": 0,
        "avg_stocks_per_leg": 0,
    }
    net_digits = {
        "avg_turnover": 2,
        "sharpe": 2,
        **{f"sharpe_net_{c}bps": 2 for c in COST_BPS},
        "avg_cost_bps": 0,
        "sharpe_net_est": 2,
        "breakeven_bps": 0,
    }

    def shown(digits: dict[str, int]) -> pd.DataFrame:
        cols = {c: table[c].map(lambda v, d=d: f"{v:.{d}f}") for c, d in digits.items()}
        return table[keys].assign(**cols)

    costs = ", ".join(str(c) for c in COST_BPS)
    return "\n".join(
        [
            "## Decile long-short portfolios (GROSS of costs)",
            "",
            "**Real data, out-of-sample. No trading or borrowing costs yet: an upper bound, "
            "not achievable returns.**",
            "",
            "- Each month: long the top 10% of stocks by forecast, short the bottom 10%, hold "
            "one month. Return = long leg - short leg (the short funds the long).",
            "- Only the models are traded; each uses the directions it learned in its "
            "training window. Single signals are scored in the appendix (IC only).",
            "- `equal`: every stock the same weight; `value`: weighted by market cap at "
            "formation. A big gap means the edge lives in small, costly-to-trade stocks.",
            "- `legs`: `decile` re-forms both legs from scratch each month; `band` buys "
            "into the top (bottom) 10% but only sells once a stock leaves the top (bottom) "
            f"{10 * EXIT_GROUPS}%, so names near the cut-off stop flipping in and out "
            "(buy/hold spread, Novy-Marx & Velikov 2016). Band fixed in advance, not tuned.",
            "- `ann_return` = 12 x mean monthly; `sharpe` = mean / std x sqrt(12); `nw_t_stat` "
            "of the mean monthly return; `max_drawdown` of $1 compounded.",
            "",
            markdown_table(shown(gross_digits)),
            "",
            "## Decile long-short portfolios NET of costs",
            "",
            "**Real data, out-of-sample. Same portfolios as above; bid-ask spread and short "
            "borrow fee only, no market impact (fine for a small account, optimistic for "
            "a large fund).**",
            "",
            "- `avg_turnover`: average dollars traded per month per $1 in each leg, both "
            "legs together (2 = both legs fully replaced). Includes trades back to target "
            "after each holding drifts with its own return.",
            f"- `sharpe_net_Xbps`: Sharpe after paying X bps (0.01%) per dollar traded and "
            f"a {BORROW_BPS_PA} bps/year fee on the $1 short. X = {costs}: roughly big "
            "institution trading large caps up to small caps. A flat cost is generous to "
            "equal-weighted portfolios, whose small stocks cost more to trade.",
            "- `sharpe_net_est`: Sharpe after paying each stock's estimated half-spread "
            "(table below) plus the borrow fee. `avg_cost_bps`: what that averaged per "
            "dollar traded. Stocks sold after leaving the universe pay that month's "
            "highest group cost.",
            "- `breakeven_bps`: the flat trading cost at which the average net return is "
            "zero. Compare it with `avg_cost_bps`.",
            "",
            markdown_table(shown(net_digits)),
            "",
        ]
    )


def honest_trials(n_portfolios: int) -> int:
    """Every portfolio in the table plus the variants tried but not built."""
    return n_portfolios + UNREPORTED_TRIALS


def earlier_signal_candidates() -> list[tuple[str, int]]:
    """The single-signal portfolios of Steps 27-45: the original 7, published sign.

    No longer shown, but rebuilt so their Sharpes count in the spread of trials.
    Without them the spread would shrink to that of the similar model portfolios
    alone, lowering the luck bar and flattering the winner.
    """
    return [(f"signal_{c}", PUBLISHED_SIGN[c]) for c in ORIGINAL_FEATURES]


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std(ddof=0))


def monthly_sharpes(
    net: dict[tuple[str, str, str], pd.Series],
    hidden: dict[tuple[str, str, str], pd.Series] | None = None,
) -> tuple[dict, float]:
    """Per-month Sharpe of each shown portfolio, and the variance of Sharpes across
    every trial we can rebuild: the shown ones plus `hidden` (earlier, not shown)."""
    sharpes = {k: _sharpe(r) for k, r in net.items()}
    every = [*sharpes.values(), *(_sharpe(r) for r in (hidden or {}).values())]
    return sharpes, float(np.var(every, ddof=1))


def selection_table(
    net: dict[tuple[str, str, str], pd.Series],
    hidden: dict[tuple[str, str, str], pd.Series] | None = None,
) -> pd.DataFrame:
    """Does each net-of-cost Sharpe survive having tried this many portfolios?

    Sharpes are per month here, as the formulas need. The variance across trials is
    that of the portfolios' actual Sharpes: how far luck and real differences spread
    them. `psr` asks "true Sharpe > 0?"; `dsr_nN` asks "true Sharpe > the best of N
    skill-less tries?" (Bailey & Lopez de Prado 2014). Best net Sharpe first.
    """
    sharpes, variance = monthly_sharpes(net, hidden)
    trials = sorted({honest_trials(len(net)), *SENSITIVITY_TRIALS})
    rows = []
    for (name, weighting, legs), r in net.items():
        row = {
            "portfolio": name,
            "weighting": weighting,
            "legs": legs,
            "sharpe_net_est": sharpes[(name, weighting, legs)] * np.sqrt(12),
            "psr": probabilistic_sharpe_ratio(r),
        }
        row |= {f"dsr_n{n}": deflated_sharpe_ratio(r, n, variance) for n in trials}
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("sharpe_net_est", ascending=False)
    return table.reset_index(drop=True)


def render_selection(
    table: pd.DataFrame,
    net: dict[tuple[str, str, str], pd.Series],
    hidden: dict[tuple[str, str, str], pd.Series] | None = None,
) -> str:
    n = honest_trials(len(net))
    n_hidden = len(hidden or {})
    _, variance = monthly_sharpes(net, hidden)
    bar = expected_max_sharpe(n, variance) * np.sqrt(12)
    dsr_cols = [c for c in table.columns if c.startswith("dsr_n")]
    shown = table.assign(
        sharpe_net_est=table["sharpe_net_est"].map("{:.2f}".format),
        **{c: table[c].map("{:.3f}".format) for c in ["psr", *dsr_cols]},
    )
    return "\n".join(
        [
            "## Selection bias: deflated Sharpe ratio",
            "",
            "**Real data, out-of-sample, net of estimated costs. Asks whether the best "
            "portfolio beats what luck alone would give the best of many tries.**",
            "",
            f"- Trials: {n} = the {len(net)} portfolios above plus {UNREPORTED_TRIALS} "
            "seen or tried earlier and not shown here: "
            + "; ".join(f"{count} {what}" for what, count in EARLIER_TRIALS.items())
            + ". A strategy we have looked at stays a trial even after it leaves the report. "
            "Every portfolio counts twice, plain and banded: the band is a second try at the "
            "same idea. Every row is judged against the same bar, since any of them could "
            "have been the one we picked.",
            f"- Luck bar: with {n} skill-less trials whose Sharpes spread as ours do "
            f"(std {np.sqrt(variance * 12):.2f} annualised), the best would show a Sharpe "
            f"of about {bar:.2f} by chance (Bailey & Lopez de Prado 2014). The spread is "
            f"measured on the {len(net)} portfolios above plus the {n_hidden} earlier "
            "single-signal ones, rebuilt but not shown: the model portfolios alone are "
            "alike, and their narrower spread would lower the bar. The other earlier trials "
            "count in N only.",
            "- `psr`: probability the true Sharpe is above 0, allowing for skew and fat "
            "tails (Bailey & Lopez de Prado 2012). `dsr_nN`: probability it is above the "
            f"luck bar for N trials. `dsr_n{n}` is the honest count; the others show how "
            "much the verdict depends on N. Above 0.95: survives selection.",
            "",
            markdown_table(shown),
            "",
        ]
    )


def spread_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Average estimated half-spread (bps) by size decile, for a few eras."""
    est = estimated_spread_bps(panel)
    # One value per group and month (every stock in a group shares it).
    by_month = est.groupby(["month_end", "size_group"], as_index=False)["cost_bps"].first()
    by_month["era"] = pd.cut(by_month["month_end"].dt.year, SPREAD_ERAS, labels=SPREAD_ERA_LABELS)
    table = by_month.pivot_table(
        index="era", columns="size_group", values="cost_bps", aggfunc="mean", observed=True
    )
    table.columns = [f"size_{c}" for c in table.columns]
    return table.reset_index()


def render_spreads(table: pd.DataFrame) -> str:
    shown = table.assign(**{c: table[c].map(lambda v: f"{v:.1f}") for c in table.columns[1:]})
    return "\n".join(
        [
            "## Estimated trading cost by company size",
            "",
            "**Real data. Half the bid-ask spread (bps per dollar traded), the cost of one "
            "trade vs. the mid price.**",
            "",
            "- Estimated each month from daily high, low and close prices (Abdi & Ranaldo "
            "2017), pooled over all stocks in the same market-cap decile (`size_1` = "
            "smallest 10%, `size_10` = largest) because one stock's estimate is mostly "
            "noise. Days with no trading are skipped. Values are averages over the months "
            "of each era.",
            "- Corwin & Schultz (2012) was tried first and rejected: on this data it made "
            "the smallest stocks look cheapest (no-trade days read as zero spread).",
            "- Spreads should fall with size, and fall after 2001, when US prices moved "
            "from 1/16 dollar ticks to cents.",
            "- The estimator has a noise floor: for the largest stocks it gives ~15 bp, "
            "well above their real quoted half-spreads (a few bp), and volatile years "
            "(2008) push it up. So costs for large stocks are overstated, i.e. "
            "conservative.",
            "",
            markdown_table(shown),
            "",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score baseline forecasters on the real panel.")
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    panel = prepare(load_panel())
    forecasts: dict[str, pd.DataFrame] = {}
    fitted: dict[str, list[Forecaster]] = {}
    table = run_all(panel, keep=forecasts, fitted=fitted)
    print(table.to_string(index=False))
    penalties = penalty_table(fitted["enet"], forecasts["enet"])
    coefs = coefficient_table(fitted["enet"])
    print(penalties.to_string(index=False))
    signals = signal_table(table)
    net: dict[tuple[str, str, str], pd.Series] = {}
    portfolios = portfolio_table(forecasts, panel, keep_net=net)
    print(portfolios.to_string(index=False))
    hidden: dict[tuple[str, str, str], pd.Series] = {}
    portfolio_table(forecasts, panel, keep_net=hidden, candidates=earlier_signal_candidates())
    selection = selection_table(net, hidden)
    print(selection.to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(
        [
            render_report(table, panel),
            render_penalties(penalties, coefs),
            render_portfolios(portfolios),
            render_selection(selection, net, hidden),
            render_spreads(spread_table(panel)),
            render_signals(signals),
        ]
    )
    args.out.write_text(report, encoding="utf-8")
    print(f"Report -> {args.out}")


if __name__ == "__main__":
    main()
