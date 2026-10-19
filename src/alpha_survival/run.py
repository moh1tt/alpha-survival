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
from alpha_survival.walkforward import walk_forward_predict

REPORT_PATH = Path("reports/baselines.md")
RAW_FEATURES = ["mom_12_1", "log_mcap", "bm", "rev_1m", "gp_a", "asset_growth"]
FEATURES = [f"{c}_rank" for c in RAW_FEATURES]
# Rolling window: train only on the latest 5 years, so old regimes are forgotten.
ROLLING_WINDOW = 60


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
    panel = add_asset_growth(add_profitability(panel))
    panel = panel[panel["investable"]].reset_index(drop=True)
    return rank_normalise(panel, RAW_FEATURES)


def evaluate(panel: pd.DataFrame, models: list[Forecaster], **split_kwargs) -> pd.DataFrame:
    """One row of out-of-sample scores per model.

    R^2 only for forecasts that are returns: a raw signal's scale is not a return.
    Rank IC only where forecasts differ across stocks: constant forecasts rank nothing.
    """
    rows = []
    for model in models:
        oos = walk_forward_predict(panel, model, **split_kwargs)
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
            "`asset_growth` (assets vs. 12 months earlier; expect a NEGATIVE IC); "
            "`ols` and `ridge` (alpha=1, untuned) on all six rank-normalised to [-1, 1] "
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


def run_all(panel: pd.DataFrame, **split_kwargs) -> pd.DataFrame:
    """Every model with an expanding window, then the learning ones with a rolling one."""
    expanding = evaluate(panel, baseline_models(), **split_kwargs)
    rolling = evaluate(panel, rolling_models(), window=ROLLING_WINDOW, **split_kwargs)
    return pd.concat([expanding, rolling], ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score baseline forecasters on the real panel.")
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    panel = prepare(load_panel())
    table = run_all(panel)
    print(table.to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(table, panel), encoding="utf-8")
    print(f"Report -> {args.out}")


if __name__ == "__main__":
    main()
