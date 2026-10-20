"""Run forecasters through walk-forward on the real panel and report out-of-sample scores.

    python -m alpha_survival.run      # -> reports/baselines.md

Only summary statistics are written; raw Sharadar data never leaves data/raw.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from alpha_survival.data.quality import markdown_table
from alpha_survival.data.real import load_panel
from alpha_survival.features import (
    add_asset_growth,
    add_momentum,
    add_profitability,
    add_reversal,
    add_size,
    add_value,
    add_volatility,
    rank_normalise,
)
from alpha_survival.metrics import ic_summary, oos_r2, rank_ic
from alpha_survival.models import (
    Forecaster,
    HistoricalMean,
    SignalForecast,
    SklearnForecaster,
    ZeroForecast,
)
from alpha_survival.portfolio import (
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
RAW_FEATURES = ["mom_12_1", "log_mcap", "bm", "rev_1m", "gp_a", "asset_growth", "vol_3m"]
FEATURES = [f"{c}_rank" for c in RAW_FEATURES]
# Rolling window: train only on the latest 5 years, so old regimes are forgotten.
ROLLING_WINDOW = 60
# Direction each signal was PUBLISHED with (+1: high value -> high return). Single-
# signal portfolios trade this sign, never the one our test period happened to show:
# picking signs from the test results would be data snooping.
PUBLISHED_SIGN = {
    "mom_12_1": 1,  # Jegadeesh & Titman 1993
    "log_mcap": -1,  # Banz 1981: small beats big
    "bm": 1,  # Fama & French 1992
    "rev_1m": -1,  # Jegadeesh 1990
    "gp_a": 1,  # Novy-Marx 2013
    "asset_growth": -1,  # Cooper, Gulen & Schill 2008
    "vol_3m": -1,  # Ang, Hodrick, Xing & Zhang 2006
}
# Flat trading cost per dollar traded, in basis points: roughly large caps traded
# by a big institution (10) up to small caps (50).
COST_BPS = (10, 20, 50)
# Annual fee for borrowing the shorted stock. 50 bps is typical for easy-to-borrow
# names; small, heavily shorted stocks cost far more, so this is optimistic.
BORROW_BPS_PA = 50
# Eras for the spread table: before decimalisation (2001), then roughly 7-year blocks.
SPREAD_ERAS = [1996, 2000, 2008, 2015, 2026]
SPREAD_ERA_LABELS = ["1997-2000", "2001-2008", "2009-2015", "2016-2026"]


def baseline_models() -> list[Forecaster]:
    return [
        ZeroForecast(),
        HistoricalMean(),
        SignalForecast("mom_12_1"),
        SignalForecast("log_mcap"),
        SignalForecast("bm"),
        SignalForecast("rev_1m"),
        SignalForecast("gp_a"),
        SignalForecast("asset_growth"),
        SignalForecast("vol_3m"),
        SklearnForecaster(LinearRegression(), FEATURES, "ols"),
        SklearnForecaster(Ridge(alpha=1.0), FEATURES, "ridge"),
    ]


def rolling_models() -> list[Forecaster]:
    """Models rerun with a rolling window. Only ones that learn: a raw signal has no
    training set to roll, and Ridge equals OLS with this few features."""
    return [HistoricalMean(), SklearnForecaster(LinearRegression(), FEATURES, "ols")]


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    """Add features on the full panel, keep the rows that may be bought, rank them.

    Order matters: momentum needs a stock's history from months it was below the
    price cut, so features come first and the investable filter second. Ranks come
    last, so each stock is ranked among the stocks that could actually be bought.
    """
    panel = add_reversal(add_value(add_size(add_momentum(panel))))
    panel = add_volatility(add_asset_growth(add_profitability(panel)))
    panel = panel[panel["investable"]].reset_index(drop=True)
    return rank_normalise(panel, RAW_FEATURES)


def evaluate(
    panel: pd.DataFrame,
    models: list[Forecaster],
    keep: dict[str, pd.DataFrame] | None = None,
    **split_kwargs,
) -> pd.DataFrame:
    """One row of out-of-sample scores per model.

    R^2 only for forecasts that are returns: a raw signal's scale is not a return.
    Rank IC only where forecasts differ across stocks: constant forecasts rank nothing.
    If `keep` is given, each model's out-of-sample forecasts are stored in it by name.
    """
    rows = []
    for model in models:
        oos = walk_forward_predict(panel, model, **split_kwargs)
        if keep is not None:
            keep[model.name] = oos
        scored = oos.dropna(subset=["fwd_ret", "pred"])
        window = split_kwargs.get("window")
        row = {
            "model": model.name,
            "window": "expanding" if window is None else f"rolling {window}m",
            "first_test_month": oos["month_end"].min().date(),
            "last_test_month": oos["month_end"].max().date(),
            "n_forecasts": len(scored),
            "oos_r2_pct": float("nan"),
            "mean_ic": float("nan"),
            "ic_t_stat": float("nan"),
            "ic_nw_t_stat": float("nan"),
            "ic_autocorr_1": float("nan"),
            "ic_hit_rate": float("nan"),
            "ic_months": 0,
        }
        if not isinstance(model, SignalForecast):
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
        rows.append(row)
    return pd.DataFrame(rows)


def render_report(table: pd.DataFrame, panel: pd.DataFrame) -> str:
    digits = {
        "oos_r2_pct": 3,
        "mean_ic": 4,
        "ic_t_stat": 2,
        "ic_nw_t_stat": 2,
        "ic_autocorr_1": 2,
        "ic_hit_rate": 3,
    }
    shown = table.assign(
        n_forecasts=table["n_forecasts"].map("{:,}".format),
        **{
            c: table[c].map(lambda v, d=d: "" if pd.isna(v) else f"{v:.{d}f}")
            for c, d in digits.items()
        },
    )
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
            "- Models: `zero` and `hist_mean` (pooled training mean) baselines; raw signals "
            "`mom_12_1` (12-1 momentum), `log_mcap` (log size) and `bm` (book-to-market: "
            "latest filed USD book equity / month-end market cap, NaN if book <= 0) and "
            "`rev_1m` (this month's return, short-term reversal; expect a NEGATIVE IC) and "
            "`gp_a` (gross profit / total assets, NaN where cost of revenue is 0) and "
            "`asset_growth` (assets vs. 12 months earlier; expect a NEGATIVE IC) and "
            "`vol_3m` (std of daily returns over the last 3 months; expect a NEGATIVE IC); "
            "`ols` and `ridge` (alpha=1, untuned) on all seven rank-normalised to [-1, 1] "
            "each month.",
            "- Splits: first 60 months train only, 12-month test blocks, model refit at the "
            "start of each block, horizon 1, embargo 0. `window`: `expanding` trains on all "
            f"months so far; `rolling {ROLLING_WINDOW}m` on the latest {ROLLING_WINDOW} only. "
            "Both start testing in the same month.",
            "- `oos_r2_pct`: pooled R^2 vs. a zero forecast, in percent (Gu, Kelly & Xiu 2020). "
            "Blank for raw signals, whose scale is not a return.",
            "- `mean_ic`: average monthly Spearman rank correlation of forecast with next-month "
            "return; months with < 10 stocks skipped. Blank for constant forecasts. "
            "`signal_log_mcap` is log size, so a small-firm premium shows as a NEGATIVE IC.",
            "- `ic_t_stat` treats months as independent; `ic_nw_t_stat` corrects for "
            "autocorrelated ICs (Newey & West 1987; lags floor(4 (n/100)^(2/9)), 5 for 285 "
            "months) and is the one to trust. `ic_autocorr_1`: correlation of each month's "
            "IC with the previous month's.",
            "",
            markdown_table(shown),
            "",
        ]
    )


def run_all(
    panel: pd.DataFrame, keep: dict[str, pd.DataFrame] | None = None, **split_kwargs
) -> pd.DataFrame:
    """Every model with an expanding window, then the learning ones with a rolling one.

    `keep` collects the expanding-window forecasts by model name (for portfolios).
    """
    expanding = evaluate(panel, baseline_models(), keep=keep, **split_kwargs)
    rolling = evaluate(panel, rolling_models(), window=ROLLING_WINDOW, **split_kwargs)
    return pd.concat([expanding, rolling], ignore_index=True)


def portfolio_table(oos_by_model: dict[str, pd.DataFrame], panel: pd.DataFrame) -> pd.DataFrame:
    """Decile long-short performance for OLS and each signal, EW and VW.

    Gross stats, plus average turnover, net Sharpe at each flat `COST_BPS` and at
    each stock's estimated spread (`estimated_spread_bps`), all with the
    `BORROW_BPS_PA` short fee, and the break-even trading cost. Signals trade their
    published direction (`PUBLISHED_SIGN`); OLS's forecast already carries the
    directions it learned in training. Value weights use market cap at formation
    (month_end), known when the trade is placed.
    """
    caps = panel[["permaticker", "month_end", "market_cap"]]
    spread = estimated_spread_bps(panel).set_index(["permaticker", "month_end"])["cost_bps"]
    candidates = [("ols", 1)] + [(f"signal_{c}", sign) for c, sign in PUBLISHED_SIGN.items()]
    rows = []
    for name, sign in candidates:
        if name not in oos_by_model:
            continue
        oos = oos_by_model[name].merge(caps, on=["permaticker", "month_end"], how="left")
        oos = oos.assign(pred=sign * oos["pred"])
        direction = "learned" if name == "ols" else ("high" if sign > 0 else "low")
        for weighting in WEIGHTINGS:
            ls = long_short_returns(oos, weighting=weighting)
            gross = ls["long_short"]
            weights = leg_weights(oos, weighting=weighting)
            trade_rows, to = trades(weights), turnover(weights)
            net = {}
            for c in COST_BPS:
                net_ret = net_returns(gross, trading_costs(trade_rows, c), BORROW_BPS_PA)
                net[f"sharpe_net_{c}bps"] = summarise(net_ret)["sharpe"]
            est_cost = trading_costs(trade_rows, spread).reindex(gross.index)
            net["sharpe_net_est"] = summarise(net_returns(gross, est_cost, BORROW_BPS_PA))["sharpe"]
            # Average cost paid per dollar traded: what the estimated spreads amount to.
            net["avg_cost_bps"] = float(est_cost.sum() / to.reindex(gross.index).sum() * 1e4)
            rows.append(
                {
                    "portfolio": name,
                    "direction": direction,
                    "weighting": weighting,
                    **summarise(gross),
                    "avg_stocks_per_leg": float(ls[["n_long", "n_short"]].mean().mean()),
                    "avg_turnover": float(to.reindex(gross.index).mean()),
                    **net,
                    "breakeven_bps": breakeven_cost_bps(gross, to, BORROW_BPS_PA),
                }
            )
    return pd.DataFrame(rows)


def render_portfolios(table: pd.DataFrame) -> str:
    keys = ["portfolio", "direction", "weighting"]
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
            "- `direction`: signals trade the sign their original paper published (`high` = "
            "buy high values), never the sign seen in our test period; `ols` uses the signs "
            "it learned in each training window.",
            "- `equal`: every stock the same weight; `value`: weighted by market cap at "
            "formation. A big gap means the edge lives in small, costly-to-trade stocks.",
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
    table = run_all(panel, keep=forecasts)
    print(table.to_string(index=False))
    portfolios = portfolio_table(forecasts, panel)
    print(portfolios.to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(
        [
            render_report(table, panel),
            render_portfolios(portfolios),
            render_spreads(spread_table(panel)),
        ]
    )
    args.out.write_text(report, encoding="utf-8")
    print(f"Report -> {args.out}")


if __name__ == "__main__":
    main()
