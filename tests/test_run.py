import numpy as np
import pandas as pd
import pytest

from alpha_survival import run
from alpha_survival.data.real import PRICE_DELAY_K
from alpha_survival.models import HistoricalMean
from alpha_survival.run import (
    COST_BPS,
    LEARNED,
    PUBLISHED_SIGN,
    RAW_FEATURES,
    UNREPORTED_TRIALS,
    baseline_models,
    coefficient_table,
    earlier_signal_candidates,
    evaluate,
    monthly_sharpes,
    penalty_table,
    portfolio_table,
    prepare,
    render_penalties,
    render_portfolios,
    render_report,
    render_selection,
    render_signals,
    render_spreads,
    run_all,
    selection_table,
    signal_table,
    spread_table,
)

N_STOCKS, N_MONTHS = 15, 40
MONTHS = pd.date_range("2000-01-31", periods=N_MONTHS, freq="ME")


@pytest.fixture(autouse=True)
def toy_factors(monkeypatch):
    """prepare() loads factor returns from data/raw; tests get toy ones, no files."""
    rng = np.random.default_rng(1)
    factors = pd.DataFrame(
        rng.normal(0, 0.04, (N_MONTHS, 4)),
        index=MONTHS,
        columns=["mktrf", "smb", "hml", "ps_innov"],
    ).assign(rf=0.002)
    monkeypatch.setattr(run, "monthly_factors", lambda: factors)


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
    panel["fxusd"] = 1.0
    panel["equity"] = panel["equityusd"]
    panel["liabilities"] = panel["assets"] - panel["equity"]
    panel["sic_code"] = rng.choice([3571.0, 6022.0], len(panel))
    panel["fama_industry"] = panel["sic_code"].map({3571.0: "Computers", 6022.0: "Banking"})
    cols = ["revenue", "netinc", "depamor", "ncfo", "rnd", "cashneq", "debt", "ebitda"]
    cols += ["sgna", "intexp", "taxexp", "receivables", "inventory", "deferredrev", "payables"]
    cols += ["assetsc", "liabilitiesc", "debtc", "debtnc", "ncfi", "ncff"]
    cols += ["investmentsc", "investmentsnc", "capex", "ncfdebt"]
    for col in [*cols, "ppnenet", "ncfdiv", "ncfcommon"]:
        panel[col] = panel["assets"] * rng.normal(0.1, 0.1, len(panel))
    panel["sharesbas"] = panel["market_cap"] * rng.uniform(0.01, 0.02, len(panel))
    panel["sharefactor"] = 1.0
    panel["n_days"] = 21
    panel["dret_sum"] = rng.normal(0, 0.05, len(panel))
    panel["dret_sumsq"] = 21 * rng.uniform(0.0001, 0.002, len(panel))
    panel["ar_n"] = 20
    panel["ar_sum"] = 20 * rng.uniform(0, 1e-4, len(panel))
    # Daily summaries for the price and trading signals (data.real.DAILY_SUMMARY_COLUMNS).
    panel["n_trade_days"] = 21
    panel["dret_sumcube"] = rng.normal(0, 1e-4, len(panel))
    panel["dret_max"] = rng.uniform(0, 0.1, len(panel))
    panel["close_split_adj"] = rng.uniform(5, 100, len(panel))
    panel["close_unadj"] = panel["close_split_adj"]
    panel["cs_n"] = 20
    panel["cs_sum"] = 20 * rng.uniform(0, 0.02, len(panel))
    panel["close_max"] = panel["close_split_adj"] * rng.uniform(1, 1.2, len(panel))
    panel["illiq_n"] = 21
    panel["illiq_sum"] = 21 * rng.uniform(0, 1e-8, len(panel))
    panel["share_volume"] = rng.uniform(1e5, 1e7, len(panel))
    panel["dollar_volume"] = panel["share_volume"] * panel["close_split_adj"]
    panel["investable"] = True
    panel["n_zero_volume"] = rng.integers(0, 3, len(panel))
    panel["taxliabilities"] = panel["assets"] * rng.uniform(0, 0.05, len(panel))
    panel["eps"] = panel["netinc"] / panel["sharesbas"]
    for col in ["netinc", "assets", "cashneq", "taxexp", "eps", "revenue"]:  # quarterly, _q
        panel[f"{col}_q"] = panel[col] * rng.uniform(0.2, 0.3, len(panel))
    panel["sharesbas_q"] = panel["sharesbas"]
    quarter_end = panel["month_end"].dt.month % 3 == 0  # quarterly dividends
    panel["div_amount"] = np.where(quarter_end, 0.25, np.nan)
    panel["div_freq"] = np.where(quarter_end, 3.0, np.nan)
    event = rng.uniform(size=len(panel)) < 0.02
    panel["exch_change"] = panel["exch_upgrade"] = panel["spun_off"] = np.where(event, 1.0, np.nan)
    panel["filing_date_q"] = panel["month_end"] - pd.Timedelta(days=30)
    panel = panel.copy()  # defragment: pandas warns past ~100 single-column inserts
    # Daily factor-regression summaries (data.real.FACTOR_SUMMARY_COLUMNS).
    panel["xret_n"] = panel["vix_n"] = 21
    panel["xret_sum"] = rng.normal(0, 0.05, len(panel))
    panel["xret_sumsq"] = 21 * rng.uniform(0.0002, 0.002, len(panel))
    panel["mkt_sum"] = rng.normal(0, 0.03, len(panel))
    panel["mkt_sumsq"] = 21 * rng.uniform(0.00005, 0.0002, len(panel))
    panel["xret_mkt_sum"] = rng.normal(0, 0.001, len(panel))
    panel["ff3_resid_sumsq"] = panel["xret_sumsq"] * rng.uniform(0.5, 0.9, len(panel))
    panel["ff3_resid_sumcube"] = rng.normal(0, 1e-5, len(panel))
    panel["vix_beta"] = rng.normal(0, 0.1, len(panel))
    for prefix in ["lx", "lm", "l3x", "l3m"]:  # daily (and 3-day) log-return sums
        panel[f"{prefix}_sum"] = rng.normal(0, 0.05, len(panel))
        panel[f"{prefix}_sumsq"] = 21 * rng.uniform(0.0002, 0.002, len(panel))
    panel["lx_lm_sum"] = panel["l3x_l3m_sum"] = rng.normal(0, 0.001, len(panel))
    panel["lx_lm2_sum"] = rng.normal(0, 1e-5, len(panel))
    panel["l3_n"] = 21
    # Price-delay sums: 21 days of (1, mktrf, 4 lags), a diagonal-heavy X'X.
    delay = {}
    for i in range(PRICE_DELAY_K):
        delay[f"pd_xy{i}"] = rng.normal(0, 1e-3, len(panel))
        delay[f"pd_xx{i}{i}"] = 21 * rng.uniform(5e-5, 2e-4, len(panel))
        for j in range(i + 1, PRICE_DELAY_K):
            delay[f"pd_xx{i}{j}"] = rng.normal(0, 1e-5, len(panel))
    delay |= {"pd_xx00": 21, "pd_yy": panel["xret_sumsq"]}
    panel = pd.concat([panel, pd.DataFrame(delay, index=panel.index)], axis=1)
    panel["market_tail"] = panel["month_end"].map(
        dict(zip(MONTHS, rng.uniform(0.3, 0.6, N_MONTHS), strict=True))
    )
    return panel.copy()  # one block: column-by-column inserts fragment the frame


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
    panel = prepare(make_panel())
    forecasts = {}
    table = evaluate(panel, baseline_models(), keep=forecasts, min_train=12)
    table = table.set_index("model")
    assert list(table.index) == [
        "zero",
        "hist_mean",
        *(f"signal_{c}" for c in RAW_FEATURES),
        "ols",
        "ridge",
        "enet",
        "lgbm",
    ]

    assert table.loc["zero", "oos_r2_pct"] == 0.0  # the benchmark scores exactly 0
    assert np.isfinite(table.loc["hist_mean", "oos_r2_pct"])
    # Constant forecasts have no rank IC; raw signals have no R^2.
    assert table.loc[["zero", "hist_mean"], "mean_ic"].isna().all()
    signals = [f"signal_{c}" for c in RAW_FEATURES]
    assert table.loc[signals, "oos_r2_pct"].isna().all()
    # Signals whose lags reach past the 40-month toy panel are blank, not an error.
    long_lags = ["mom_season_6y", "mom_season_11y", "mom_season_16y", "mom_offseason_6y"]
    long_lags += ["mom_offseason_11y", "mom_offseason_16y"]
    long_lags += ["composite_debt_issuance", "share_iss_5y"]  # 60 and 65 months
    long_lags += ["residual_momentum"]  # 36-month fits, then 11 residuals: 46 months
    long_lags += ["beta_tail_risk"]  # 72 months
    long_lags += ["firm_age_mom"]  # every fixture stock has the same age: no youngest fifth
    empty = table.loc[signals, "n_forecasts"] == 0
    assert set(empty[empty].index) <= {f"signal_{c}" for c in long_lags}
    assert table.loc[empty[empty].index, "mean_ic"].isna().all()
    # Every signal with 2+ rankable months (>= 10 stocks with a forecast and a return,
    # not all equal; `score` needs two for a t-stat) among those tested gets an IC.
    # Some are too sparse on 15 toy stocks (e.g. net_debt_price) or constant
    # (firm_age: every toy stock starts in the same month).
    test_months = forecasts["zero"]["month_end"].unique()
    tested = panel[panel["month_end"].isin(test_months)].dropna(subset=["fwd_ret"])
    by_month = tested.groupby("month_end")
    rankable = [
        f"signal_{c}"
        for c in RAW_FEATURES
        if ((by_month[c].count() >= 10) & (by_month[c].nunique() > 1)).sum() >= 2
    ]
    assert len(rankable) > 30
    assert table.loc[rankable, "ic_months"].gt(0).all()
    # Learned models forecast returns AND order stocks: both metrics.
    assert (
        table.loc[["ols", "ridge", "enet", "lgbm"], ["oos_r2_pct", "mean_ic"]].notna().all().all()
    )
    assert (table["first_test_month"] == MONTHS[12].date()).all()


def test_report_labels_the_results_honestly():
    panel = prepare(make_panel())
    text = render_report(evaluate(panel, baseline_models(), min_train=12), panel)
    lowered = text.lower()
    assert "real data" in lowered and "out-of-sample" in lowered
    assert "no transaction costs" in lowered
    # Models in the main table; single signals only in the appendix.
    assert "| ols |" in text and "signal_mom_12_1" not in text


def test_signal_appendix_lists_every_feature_with_its_published_sign():
    panel = prepare(make_panel())
    table = signal_table(evaluate(panel, baseline_models(), min_train=12))
    assert list(table["feature"]) == RAW_FEATURES
    assert (table["published_sign"] == table["feature"].map(PUBLISHED_SIGN)).all()
    assert table.set_index("feature").loc["mom_6m", "osap"] == "Mom6m"
    assert table.set_index("feature").loc["vol_3m", "osap"] == ""  # not an OSAP signal
    agrees = np.sign(table["mean_ic"]) == table["published_sign"]
    assert ((table["sign_agrees"] == "yes") == agrees).all()
    text = render_signals(table)
    assert "Appendix" in text and "no portfolios" in text and "| Mom6m |" in text


def test_published_signs_come_from_the_catalogue_and_match_the_papers():
    assert set(PUBLISHED_SIGN) == set(RAW_FEATURES)
    # The original 7, as hand-coded before the catalogue (Steps 10-26).
    original = {"mom_12_1": 1, "log_mcap": -1, "bm": 1, "rev_1m": -1, "gp_a": 1}
    original |= {"asset_growth": -1, "vol_3m": -1}
    assert {f: PUBLISHED_SIGN[f] for f in original} == original
    assert len(RAW_FEATURES) == len(set(RAW_FEATURES)) == 134


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


def test_portfolio_table_trades_only_the_models_both_weightings():
    panel = prepare(make_panel())
    forecasts = {}
    run_all(panel, keep=forecasts, min_train=12)
    table = portfolio_table(forecasts, panel)
    assert set(table["portfolio"]) == set(LEARNED) == {"ols", "enet", "lgbm", "combo"}
    assert (table.groupby("portfolio")["weighting"].apply(set) == {"equal", "value"}).all()
    combos = table.groupby(["portfolio", "weighting"])["legs"].apply(set)
    assert (combos == {"decile", "band"}).all()
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


def test_selection_table_deflates_every_portfolio_by_the_honest_trial_count():
    panel = prepare(make_panel())
    forecasts, net = {}, {}
    run_all(panel, keep=forecasts, min_train=12)
    portfolio_table(forecasts, panel, keep_net=net)
    assert len(net) == 16  # 4 models x 2 weightings x 2 leg rules
    assert UNREPORTED_TRIALS == 92  # earlier portfolios stay trials
    table = selection_table(net)
    honest = f"dsr_n{16 + UNREPORTED_TRIALS}"
    assert list(table.columns) == [
        "portfolio",
        "weighting",
        "legs",
        "sharpe_net_est",
        "psr",
        "dsr_n20",
        honest,
        "dsr_n200",
    ]
    assert table["sharpe_net_est"].is_monotonic_decreasing  # best first
    # Same Sharpe as the NET table, and more trials only ever make the test harder.
    best = table.iloc[0]
    assert best["psr"] > best["dsr_n20"] > best[honest] > best["dsr_n200"]
    probs = table[["psr", "dsr_n20", honest, "dsr_n200"]]
    assert ((probs >= 0) & (probs <= 1)).all().all()
    report = render_selection(table, net)
    assert "deflated Sharpe" in report and f"Trials: {16 + UNREPORTED_TRIALS}" in report
    assert "single-signal portfolios" in report


def test_signal_portfolios_trade_the_published_direction():
    # A signal published as "low is good" must short its high values: flipping the
    # sign must flip the long-short return exactly.
    panel = prepare(make_panel())
    forecasts = {}
    run_all(panel, keep=forecasts, min_train=12)
    low = portfolio_table(forecasts, panel, candidates=[("signal_vol_3m", -1)])
    high = portfolio_table(forecasts, panel, candidates=[("signal_vol_3m", 1)])
    low, high = (t[t["legs"] == "decile"].set_index("weighting") for t in (low, high))
    assert low.loc["equal", "ann_return"] == pytest.approx(-high.loc["equal", "ann_return"])
    assert ("signal_vol_3m", -1) in earlier_signal_candidates()
    assert len(earlier_signal_candidates()) == 7


def test_hidden_trials_widen_the_spread_but_not_the_rows():
    panel = prepare(make_panel())
    forecasts, net, hidden = {}, {}, {}
    run_all(panel, keep=forecasts, min_train=12)
    portfolio_table(forecasts, panel, keep_net=net)
    portfolio_table(forecasts, panel, keep_net=hidden, candidates=earlier_signal_candidates())
    assert len(hidden) == 28  # 7 signals x 2 weightings x 2 leg rules
    shown, variance = monthly_sharpes(net, hidden)
    assert set(shown) == set(net)
    every = [r.mean() / r.std(ddof=0) for r in [*net.values(), *hidden.values()]]
    assert variance == pytest.approx(np.var(every, ddof=1))
    table = selection_table(net, hidden)
    assert len(table) == len(net)  # hidden trials are not reported as rows
    assert "28 earlier" in render_selection(table, net, hidden)


def test_selection_table_sharpe_matches_the_net_table():
    panel = prepare(make_panel())
    forecasts, net = {}, {}
    run_all(panel, keep=forecasts, min_train=12)
    portfolios = portfolio_table(forecasts, panel, keep_net=net)
    keys = ["portfolio", "weighting", "legs"]
    merged = portfolios.merge(selection_table(net), on=keys, suffixes=("", "_sel"))
    # Same returns; the selection table uses population std (ddof=0), so close, not equal.
    assert merged["sharpe_net_est_sel"].to_numpy() == pytest.approx(
        merged["sharpe_net_est"].to_numpy(), rel=0.05
    )


def test_penalty_table_reports_each_enet_fold():
    panel = prepare(make_panel())
    forecasts, fitted = {}, {}
    run_all(panel, keep=forecasts, fitted=fitted, min_train=12)
    folds, oos = fitted["enet"], forecasts["enet"]
    table = penalty_table(folds, oos)
    assert len(table) == oos["fold"].nunique() == len(folds)
    assert table["test_from"].iloc[0] == MONTHS[12].date()
    for row, model in zip(table.itertuples(), folds, strict=True):
        assert row.alpha == model.alpha_
        assert row.n_kept == (model.estimator_.coef_ != 0).sum()
    coefs = coefficient_table(folds).set_index("feature")
    assert set(coefs.index) == set(RAW_FEATURES)
    first = RAW_FEATURES[0]
    by_fold = [1e4 * m.estimator_.coef_[0] for m in folds]
    assert coefs.loc[first, "mean_coef_bps"] == pytest.approx(np.mean(by_fold))
    assert coefs.loc[first, "folds_kept"] == sum(c != 0 for c in by_fold)
    assert coefs["folds_kept"].is_monotonic_decreasing
    text = render_penalties(table, coefficient_table(folds))
    assert "real data" in text.lower() and "| alpha |" in text and "| folds_kept |" in text
    assert f"of {len(table)} refits kept no feature" in text


def test_run_all_adds_the_combination_scored_as_a_rank():
    panel = prepare(make_panel())
    forecasts = {}
    table = run_all(panel, keep=forecasts, min_train=12).set_index(["model", "window"])
    combo = table.loc[("combo", "expanding")]
    assert np.isnan(combo["oos_r2_pct"]) and np.isfinite(combo["mean_ic"])
    assert len(forecasts["combo"]) == len(forecasts["lgbm"])
    assert forecasts["combo"]["pred"].between(-1, 1).all()
