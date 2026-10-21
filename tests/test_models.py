import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from alpha_survival.models import (
    TARGET,
    Forecaster,
    HistoricalMean,
    LGBMForecaster,
    SignalForecast,
    SklearnForecaster,
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


def linear_rows(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """fwd_ret = 0.01 + 0.02 * a - 0.03 * b exactly, plus an unused column."""
    rng = np.random.default_rng(seed)
    rows = pd.DataFrame({"a": rng.uniform(-1, 1, n), "b": rng.uniform(-1, 1, n)})
    rows["unused"] = rng.normal(size=n)
    rows[TARGET] = 0.01 + 0.02 * rows["a"] - 0.03 * rows["b"]
    return rows


def test_sklearn_forecaster_learns_and_predicts_from_listed_features():
    train = linear_rows()
    model = SklearnForecaster(LinearRegression(), ["a", "b"], "ols").fit(train)
    assert model.estimator_.coef_ == pytest.approx([0.02, -0.03])
    assert model.estimator_.intercept_ == pytest.approx(0.01)

    test = features(linear_rows(seed=1))
    pred = model.predict(test)
    assert pred.name == "ols"
    assert np.allclose(pred, 0.01 + 0.02 * test["a"] - 0.03 * test["b"])
    # Columns not listed cannot change the forecast.
    assert pred.equals(model.predict(test.assign(unused=1e9)))


def test_sklearn_forecaster_clones_the_estimator():
    estimator = Ridge(alpha=1.0)
    model = SklearnForecaster(estimator, ["a", "b"], "ridge").fit(linear_rows())
    assert model.estimator_ is not estimator
    assert not hasattr(estimator, "coef_")  # the template stays unfitted


def test_sklearn_forecaster_rejects_missing_or_nan_features():
    model = SklearnForecaster(LinearRegression(), ["a", "b"], "ols")
    with pytest.raises(RuntimeError, match="fit before predict"):
        model.predict(features(linear_rows()))
    gappy = linear_rows()
    gappy.loc[3, "a"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model.fit(gappy)
    with pytest.raises(KeyError, match="'b'"):
        model.fit(linear_rows().drop(columns="b"))


def u_shaped_rows(n: int = 5_000, seed: int = 0) -> pd.DataFrame:
    """fwd_ret depends on a^2: extremes beat the middle, which no straight line captures."""
    rng = np.random.default_rng(seed)
    rows = pd.DataFrame({"a": rng.uniform(-1, 1, n), "b": rng.uniform(-1, 1, n)})
    rows[TARGET] = 0.05 * (rows["a"] ** 2 - 1 / 3) + rng.normal(0, 0.01, n)
    return rows


def test_lgbm_learns_a_nonlinear_effect_ols_cannot():
    train, test = u_shaped_rows(), u_shaped_rows(seed=1)
    truth = 0.05 * (test["a"] ** 2 - 1 / 3)
    lgbm = LGBMForecaster(["a", "b"]).fit(train).predict(features(test))
    ols = SklearnForecaster(LinearRegression(), ["a", "b"], "ols").fit(train)
    assert lgbm.name == "lgbm"
    assert np.corrcoef(lgbm, truth)[0, 1] > 0.95
    assert abs(np.corrcoef(ols.predict(features(test)), truth)[0, 1]) < 0.1


def test_lgbm_min_leaf_scales_with_training_rows():
    model = LGBMForecaster(["a", "b"])
    assert model.fit(u_shaped_rows(5_000)).estimator_.get_params()["min_child_samples"] == 20
    assert model.fit(u_shaped_rows(50_000)).estimator_.get_params()["min_child_samples"] == 50
    assert model.estimator.get_params()["min_child_samples"] != 50  # template untouched


def test_lgbm_refits_are_identical():
    train, test = u_shaped_rows(), features(u_shaped_rows(seed=1))
    first = LGBMForecaster(["a", "b"]).fit(train).predict(test)
    second = LGBMForecaster(["a", "b"]).fit(train).predict(test)
    assert first.equals(second)


def test_lgbm_rejects_nan_features_and_target_in_predict():
    model = LGBMForecaster(["a", "b"])
    gappy = u_shaped_rows()
    gappy.loc[3, "a"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model.fit(gappy)
    with pytest.raises(ValueError, match="must not see the target"):
        model.fit(u_shaped_rows()).predict(u_shaped_rows())
