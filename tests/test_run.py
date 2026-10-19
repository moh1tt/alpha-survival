import numpy as np
import pandas as pd

from alpha_survival.run import baseline_models, evaluate, prepare, render_report

N_STOCKS, N_MONTHS = 15, 40
MONTHS = pd.date_range("2000-01-31", periods=N_MONTHS, freq="ME")


def make_panel(seed: int = 0) -> pd.DataFrame:
    """Toy panel shaped like load_panel's output: ret, fwd_ret, market_cap, investable."""
    rng = np.random.default_rng(seed)
    panel = pd.DataFrame(
        [(s, m) for s in range(N_STOCKS) for m in MONTHS], columns=["permaticker", "month_end"]
    )
    panel["ret"] = rng.normal(0.01, 0.08, len(panel))
    panel["fwd_ret"] = panel.groupby("permaticker")["ret"].shift(-1)
    panel["market_cap"] = np.exp(rng.normal(20, 2, len(panel)))
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
    assert list(table.index) == ["zero", "hist_mean", "signal_mom_12_1", "signal_log_mcap"]

    assert table.loc["zero", "oos_r2_pct"] == 0.0  # the benchmark scores exactly 0
    assert np.isfinite(table.loc["hist_mean", "oos_r2_pct"])
    # Constant forecasts have no rank IC; raw signals have no R^2.
    assert table.loc[["zero", "hist_mean"], "mean_ic"].isna().all()
    assert table.loc[["signal_mom_12_1", "signal_log_mcap"], "oos_r2_pct"].isna().all()
    assert table.loc[["signal_mom_12_1", "signal_log_mcap"], "ic_months"].gt(0).all()
    assert (table["first_test_month"] == MONTHS[12].date()).all()


def test_report_labels_the_results_honestly():
    panel = prepare(make_panel())
    text = render_report(evaluate(panel, baseline_models(), min_train=12), panel)
    assert "Real data" in text and "out-of-sample" in text and "no transaction costs" in text
    assert "| signal_mom_12_1 |" in text
