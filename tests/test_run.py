import numpy as np
import pandas as pd
import pytest

from alpha_survival.models import HistoricalMean
from alpha_survival.run import (
    COST_BPS,
    PUBLISHED_SIGN,
    RAW_FEATURES,
    baseline_models,
    evaluate,
    portfolio_table,
    prepare,
    render_portfolios,
    render_report,
    render_spreads,
    run_all,
    spread_table,
)

N_STOCKS, N_MONTHS = 15, 40
MONTHS = pd.date_range("2000-01-31", periods=N_MONTHS, freq="ME")


def make_panel(seed: int = 0) -> pd.DataFrame:
    """Toy panel shaped like load_panel's output, with the columns the features need."""
    rng = np.random.default_rng(seed)
    panel = pd.DataFrame(
        [(s, m) for s in range(N_STOCKS) for m in MONTHS], columns=["permaticker", "month_end"]
    )
    panel["ret"] = rng.normal(0.01, 0.08, len(panel))
    panel["fwd_ret"] = panel.groupby("permaticker")["ret"].shift(-1)
    panel["market_cap"] = np.exp(rng.normal(20, 2, len(panel)))
    panel["equityusd"] = panel["market_cap"] * rng.uniform(-0.2, 1.5, len(panel))
    panel["assets"] = panel["market_cap"] * rng.uniform(0.5, 3, len(panel))
    panel["gp"] = panel["assets"] * rng.normal(0.2, 0.1, len(panel))
    panel["cor"] = 1.0
    panel["n_days"] = 21
    panel["dret_sum"] = rng.normal(0, 0.05, len(panel))
    panel["dret_sumsq"] = 21 * rng.uniform(0.0001, 0.002, len(panel))
    panel["ar_n"] = 20
    panel["ar_sum"] = 20 * rng.uniform(0, 1e-4, len(panel))
    panel["investable"] = True
    return panel


def test_prepare_computes_features_before_dropping_uninvestable_rows():
    panel = make_panel()
    # Stock 0 is below the price cut in month 5 only; its momentum in month 15
    # needs month 5's return (t-10), so features must come before the filter.
    panel.loc[(panel["permaticker"] == 0) & (panel["month_end"] == MONTHS[5]), "investable"] = False
    out = prepare(panel)
    assert out["investable"].all()
    assert len(out) == len(panel) - 1
    row = out[(out["permaticker"] == 0) & (out["month_end"] == MONTHS[15])]
    assert row["mom_12_1"].notna().all()


def test_evaluate_scores_each_model_the_right_way():
    table = evaluate(prepare(make_panel()), baseline_models(), min_train=12)
    table = table.set_index("model")
    assert list(table.index) == [
        "zero",
        "hist_mean",
        "signal_mom_12_1",
        "signal_log_mcap",
        "signal_bm",
        "signal_rev_1m",
        "signal_gp_a",
        "signal_asset_growth",
        "signal_vol_3m",
        "ols",
        "ridge",
    ]

    assert table.loc["zero", "oos_r2_pct"] == 0.0  # the benchmark scores exactly 0
    assert np.isfinite(table.loc["hist_mean", "oos_r2_pct"])
    # Constant forecasts have no rank IC; raw signals have no R^2.
    assert table.loc[["zero", "hist_mean"], "mean_ic"].isna().all()
    signals = [f"signal_{c}" for c in RAW_FEATURES]
    assert table.loc[signals, "oos_r2_pct"].isna().all()
    assert table.loc[signals, "ic_months"].gt(0).all()
    # Learned models forecast returns AND order stocks: both metrics.
    assert table.loc[["ols", "ridge"], ["oos_r2_pct", "mean_ic"]].notna().all().all()
    assert (table["first_test_month"] == MONTHS[12].date()).all()


def test_report_labels_the_results_honestly():
    panel = prepare(make_panel())
    text = render_report(evaluate(panel, baseline_models(), min_train=12), panel)
    lowered = text.lower()
    assert "real data" in lowered and "out-of-sample" in lowered
    assert "no transaction costs" in lowered
    assert "| signal_mom_12_1 |" in text


def test_prepare_ranks_among_investable_stocks_only():
    panel = make_panel()
    month = panel["month_end"] == MONTHS[20]
    # Make the biggest stock that month uninvestable: the next biggest must rank +1.
    biggest = panel.loc[month, "market_cap"].idxmax()
    panel.loc[biggest, "investable"] = False
    out = prepare(panel)
    in_month = out[out["month_end"] == MONTHS[20]]
    assert in_month["log_mcap_rank"].max() == 1.0
    assert in_month["log_mcap_rank"].min() == -1.0
    assert out[[f"{c}_rank" for c in RAW_FEATURES]].notna().all().all()


def test_run_all_adds_rolling_runs_with_the_same_test_period():
    panel = prepare(make_panel())
    table = run_all(panel, min_train=12)
    rolling = table[table["window"] != "expanding"]
    assert set(rolling["model"]) == {"hist_mean", "ols"}
    assert (rolling["window"] == "rolling 60m").all()
    assert table["first_test_month"].nunique() == 1  # like-for-like comparison


def test_rolling_window_forgets_old_months():
    # The target jumps from +5% to -5% halfway: a rolling mean adapts, an expanding one lags.
    panel = prepare(make_panel())
    late = panel["month_end"] >= MONTHS[20]
    panel["fwd_ret"] = np.where(late, -0.05, 0.05)
    exp = evaluate(panel, [HistoricalMean()], min_train=12, test_size=4, step=4)
    roll = evaluate(panel, [HistoricalMean()], min_train=12, test_size=4, step=4, window=12)
    assert roll.loc[0, "oos_r2_pct"] > exp.loc[0, "oos_r2_pct"]


def test_portfolio_table_covers_ols_and_every_signal_both_weightings():
    panel = prepare(make_panel())
    forecasts = {}
    run_all(panel, keep=forecasts, min_train=12)
    table = portfolio_table(forecasts, panel)
    assert set(PUBLISHED_SIGN) == set(RAW_FEATURES)  # every feature has a published sign
    assert set(table["portfolio"]) == {"ols"} | {f"signal_{c}" for c in RAW_FEATURES}
    assert (table.groupby("portfolio")["weighting"].apply(set) == {"equal", "value"}).all()
    assert table["sharpe"].notna().all()
    report = render_portfolios(table)
    assert "GROSS of costs" in report and "NET of costs" in report
    # Costs only ever subtract: net Sharpe falls as the trading cost rises.
    net = table[[f"sharpe_net_{c}bps" for c in COST_BPS]]
    assert (table["sharpe"] > net.iloc[:, 0]).all()
    assert (net.diff(axis=1).iloc[:, 1:] < 0).all().all()
    assert (table["avg_turnover"] > 0).all()
    assert (table["avg_cost_bps"] > 0).all()
    assert (table["sharpe_net_est"] < table["sharpe"]).all()


def test_spread_table_has_one_column_per_size_decile():
    table = spread_table(prepare(make_panel()))
    assert list(table.columns) == ["era"] + [f"size_{g}" for g in range(1, 11)]
    assert (table.drop(columns="era") >= 0).all().all()
    assert "Abdi & Ranaldo" in render_spreads(table)


def test_signal_portfolios_trade_the_published_direction():
    # A signal published as "low is good" must short its high values: flipping the
    # forecast's sign must flip the long-short return exactly.
    panel = prepare(make_panel())
    forecasts = {}
    run_all(panel, keep=forecasts, min_train=12)
    vol = forecasts["signal_vol_3m"]
    table = portfolio_table({"signal_vol_3m": vol}, panel).set_index("weighting")
    flipped = portfolio_table({"signal_vol_3m": vol.assign(pred=-vol["pred"])}, panel)
    flipped = flipped.set_index("weighting")
    assert table.loc["equal", "direction"] == "low"
    assert table.loc["equal", "ann_return"] == pytest.approx(-flipped.loc["equal", "ann_return"])
