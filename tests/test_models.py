import numpy as np
import pandas as pd
import pytest

from alpha_survival.models import (
    TARGET,
    Forecaster,
    HistoricalMean,
    SignalForecast,
    ZeroForecast,
)


def make_rows(fwd_ret: list[float], index: list[int] | None = None) -> pd.DataFrame:
    n = len(fwd_ret)
    return pd.DataFrame(
        {
            "permaticker": range(n),
            "month_end": pd.Timestamp("2020-01-31"),
            "mom_12_1": np.linspace(-0.2, 0.3, n),
            TARGET: fwd_ret,
        },
        index=index,
    )


def features(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.drop(columns=TARGET)


def test_zero_forecast():
    rows = make_rows([0.1, -0.2, 0.05])
    pred = ZeroForecast().fit(rows).predict(features(rows))
    assert (pred == 0).all() and pred.name == "zero"


def test_historical_mean_uses_training_target_and_skips_nan():
    train = make_rows([0.01, 0.03, np.nan])
    test = make_rows([0.5, -0.5])
    pred = HistoricalMean().fit(train).predict(features(test))
    assert np.allclose(pred, 0.02)


def test_historical_mean_ignores_test_outcomes():
    # Leakage check: changing what happens in the test months cannot move the forecast.
    train = make_rows([0.01, 0.03])
    model = HistoricalMean().fit(train)
    calm, crash = make_rows([0.0, 0.0]), make_rows([-0.9, -0.9])
    assert model.predict(features(calm)).equals(model.predict(features(crash)))


def test_predict_refuses_rows_that_still_hold_the_target():
    rows = make_rows([0.1, 0.2])
    for model in (ZeroForecast(), HistoricalMean().fit(rows), SignalForecast("mom_12_1")):
        with pytest.raises(ValueError, match="must not see the target"):
            model.predict(rows)


def test_historical_mean_errors():
    with pytest.raises(RuntimeError, match="fit before predict"):
        HistoricalMean().predict(features(make_rows([0.1])))
    with pytest.raises(ValueError, match="no non-missing"):
        HistoricalMean().fit(make_rows([np.nan, np.nan]))


def test_signal_forecast_passes_column_through():
    rows = features(make_rows([0.0, 0.0, 0.0]))
    rows.loc[1, "mom_12_1"] = np.nan
    pred = SignalForecast("mom_12_1").predict(rows)
    assert pred.name == "signal_mom_12_1"
    assert pred.equals(rows["mom_12_1"].rename("signal_mom_12_1"))
    with pytest.raises(KeyError):
        SignalForecast("bm").predict(rows)


def test_forecasts_keep_the_input_index():
    # Rows with a non-default, shuffled index: forecasts must stay matched to their rows.
    rows = features(make_rows([0.0, 0.0, 0.0], index=[42, 7, 19]))
    for model in (
        ZeroForecast(),
        HistoricalMean().fit(make_rows([0.1])),
        SignalForecast("mom_12_1"),
    ):
        pred = model.predict(rows)
        assert pred.index.equals(rows.index)
    assert pred.loc[7] == rows.loc[7, "mom_12_1"]


def test_wrong_length_output_is_rejected():
    class Broken(Forecaster):
        name = "broken"

        def _predict(self, rows):
            return np.zeros(len(rows) + 1)

    with pytest.raises(ValueError, match="returned 3 forecasts for 2 rows"):
        Broken().predict(features(make_rows([0.0, 0.0])))
