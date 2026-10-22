import numpy as np
import pandas as pd
import pytest

from alpha_survival.models import TARGET, Forecaster, HistoricalMean, SignalForecast
from alpha_survival.walkforward import walk_forward_predict

N_STOCKS, N_MONTHS = 3, 30
MONTHS = pd.date_range("2000-01-31", periods=N_MONTHS, freq="ME")
SPLITS = {"min_train": 12, "test_size": 6, "step": 6}


def make_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    panel = pd.DataFrame(
        [(s, m) for m in MONTHS for s in range(N_STOCKS)], columns=["permaticker", "month_end"]
    )
    panel["stock_id_signal"] = panel["permaticker"].astype(float)
    panel[TARGET] = rng.normal(0.01, 0.05, len(panel))
    panel.loc[[4, 10], TARGET] = np.nan
    return panel


class Spy(Forecaster):
    """Records what it is shown. `calls` is a class attribute so deep copies share it."""

    name = "spy"
    calls: list = []

    def fit(self, train):
        Spy.calls.append(("fit", train))
        return self

    def _predict(self, rows):
        Spy.calls.append(("predict", rows))
        return np.zeros(len(rows))


def test_forecasts_cover_exactly_the_test_months_once():
    panel = make_panel()
    out = walk_forward_predict(panel, HistoricalMean(), **SPLITS)
    assert list(out.columns) == ["permaticker", "month_end", "fold", TARGET, "pred"]
    assert set(out["month_end"]) == set(MONTHS[12:])
    assert len(out) == N_STOCKS * (N_MONTHS - 12)
    assert not out.duplicated(["permaticker", "month_end"]).any()
    assert out["fold"].tolist() == sorted(out["fold"])  # chronological


def test_forecasts_stay_matched_to_their_stock():
    panel = make_panel().sample(frac=1, random_state=1)  # shuffled rows
    out = walk_forward_predict(panel, SignalForecast("stock_id_signal"), **SPLITS)
    assert (out["pred"] == out["permaticker"]).all()


def test_each_fold_refits_on_its_own_training_months():
    panel = make_panel()
    out = walk_forward_predict(panel, HistoricalMean(), **SPLITS)
    for fold, test_start in enumerate(range(12, N_MONTHS, 6)):
        expected = panel.loc[panel["month_end"] < MONTHS[test_start], TARGET].mean()  # skips NaN
        assert np.allclose(out.loc[out["fold"] == fold, "pred"], expected)


def test_fitted_copies_are_kept_one_per_fold_on_request():
    model, fitted = HistoricalMean(), []
    out = walk_forward_predict(make_panel(), model, fitted=fitted, **SPLITS)
    assert len(fitted) == out["fold"].nunique() == 3
    assert all(f is not model for f in fitted)
    for fold, f in enumerate(fitted):  # the kept copy is the one that made that fold's forecasts
        assert np.allclose(out.loc[out["fold"] == fold, "pred"], f.mean_)


def test_original_model_is_never_fit():
    model = HistoricalMean()
    walk_forward_predict(make_panel(), model, **SPLITS)
    assert model.mean_ is None


@pytest.mark.parametrize("horizon", [1, 3])
def test_leakage_model_never_sees_unknown_information(horizon):
    Spy.calls = []
    walk_forward_predict(make_panel(), Spy(), horizon=horizon, **SPLITS)

    last_train_month = None
    for kind, rows in Spy.calls:
        if kind == "fit":
            assert rows[TARGET].notna().all()
            last_train_month = rows["month_end"].max()
            continue
        # One month per predict call, never the target column ...
        assert rows["month_end"].nunique() == 1
        assert TARGET not in rows.columns
        # ... and every training label was known by the time of this forecast.
        forecast_month = rows["month_end"].iloc[0].to_period("M")
        assert last_train_month.to_period("M") + horizon <= forecast_month


def test_split_kwargs_are_passed_through():
    out = walk_forward_predict(make_panel(), HistoricalMean(), min_train=24, test_size=3, step=3)
    assert out["month_end"].min() == MONTHS[24]
    assert out["fold"].nunique() == 2


def test_bad_panels_raise():
    panel = make_panel()
    with pytest.raises(ValueError, match="duplicate"):
        walk_forward_predict(pd.concat([panel, panel.head(1)]), HistoricalMean(), **SPLITS)
    with pytest.raises(KeyError, match="fwd_ret"):
        walk_forward_predict(panel.drop(columns=TARGET), HistoricalMean(), **SPLITS)
