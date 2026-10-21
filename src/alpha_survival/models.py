"""Forecasters of next-month stock returns, from naive baselines upward.

Every model works on panel rows (DataFrames), not a bare feature matrix, so very
different models share one interface and one walk-forward loop:

- `fit(train)` sees training rows, target `fwd_ret` included.
- `predict(rows)` sees rows WITHOUT `fwd_ret` and returns one forecast per row,
  aligned to `rows.index`. Anything on a row is known at its `month_end` (keys,
  features, the month's own `ret`), so a cross-sectional model can read the
  feature columns and a time-series model can rebuild each stock's return history.

The baselines here are the bar every real model must clear (Gu, Kelly & Xiu 2020
score out-of-sample R^2 against a zero forecast).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin, clone

TARGET = "fwd_ret"


class Forecaster:
    """Base class: subclasses override `fit` if they learn, and `_predict`."""

    name = "forecaster"

    def fit(self, train: pd.DataFrame) -> Forecaster:
        return self

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        if TARGET in rows.columns:
            raise ValueError(f"predict() must not see the target column '{TARGET}'")
        values = np.asarray(self._predict(rows), dtype=float)
        if len(values) != len(rows):
            raise ValueError(f"{self.name} returned {len(values)} forecasts for {len(rows)} rows")
        return pd.Series(values, index=rows.index, name=self.name)

    def _predict(self, rows: pd.DataFrame) -> np.ndarray | pd.Series:
        raise NotImplementedError


class ZeroForecast(Forecaster):
    """Predict 0 for every stock: 'no idea, assume nothing happens'."""

    name = "zero"

    def _predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(rows))


class HistoricalMean(Forecaster):
    """Predict the average training `fwd_ret` (pooled over all stock-months) for every row."""

    name = "hist_mean"

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, train: pd.DataFrame) -> HistoricalMean:
        target = train[TARGET].dropna()
        if target.empty:
            raise ValueError("no non-missing training targets to average")
        self.mean_ = float(target.mean())
        return self

    def _predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("HistoricalMean must be fit before predict")
        return np.full(len(rows), self.mean_)


class SignalForecast(Forecaster):
    """Use one raw signal column as the forecast, e.g. `mom_12_1`.

    Its scale is not a return, so it is meaningful for ranking stocks (rank IC,
    long-short portfolios), not for R^2. NaN signal -> NaN forecast.
    """

    def __init__(self, column: str) -> None:
        self.column = column
        self.name = f"signal_{column}"

    def _predict(self, rows: pd.DataFrame) -> pd.Series:
        if self.column not in rows.columns:
            raise KeyError(f"signal column '{self.column}' not in rows")
        return rows[self.column]


class SklearnForecaster(Forecaster):
    """Any scikit-learn regressor, trained on a fixed list of feature columns.

    Each `fit` trains a fresh clone, so the estimator passed in is never changed.
    Features must have no missing values: rank-normalised features fill gaps with 0,
    and silently dropping rows would make models forecast different stock sets.
    """

    def __init__(self, estimator: RegressorMixin, features: list[str], name: str) -> None:
        self.estimator = estimator
        self.features = list(features)
        self.name = name
        self.estimator_: RegressorMixin | None = None

    def _matrix(self, rows: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in rows.columns]
        if missing:
            raise KeyError(f"{self.name}: feature columns {missing} not in rows")
        x = rows[self.features].to_numpy(dtype=float)
        if np.isnan(x).any():
            raise ValueError(f"{self.name}: features contain NaN; rank-normalise them first")
        return x

    def fit(self, train: pd.DataFrame) -> SklearnForecaster:
        self.estimator_ = clone(self.estimator).fit(self._matrix(train), train[TARGET])
        return self

    def _predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError(f"{self.name} must be fit before predict")
        return self.estimator_.predict(self._matrix(rows))


# Fixed, conservative settings (not tuned): shallow trees, slow learning, row and
# column subsampling, all seeded. Deterministic mode makes refits bit-identical.
LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "num_leaves": 15,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 0,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


class LGBMForecaster(SklearnForecaster):
    """Gradient-boosted trees (LightGBM): can learn nonlinear effects and interactions.

    Every leaf must hold at least `min_leaf_frac` of the training rows (and >= 20),
    so the guard against fitting noise scales from small early folds to large late
    ones. Set from the training rows only: no peeking at test months.
    """

    def __init__(
        self, features: list[str], name: str = "lgbm", min_leaf_frac: float = 0.001, **params
    ) -> None:
        super().__init__(LGBMRegressor(**(LGBM_PARAMS | params)), features, name)
        self.min_leaf_frac = min_leaf_frac

    def fit(self, train: pd.DataFrame) -> LGBMForecaster:
        min_leaf = max(20, int(self.min_leaf_frac * len(train)))
        estimator = clone(self.estimator).set_params(min_child_samples=min_leaf)
        self.estimator_ = estimator.fit(self._matrix(train), train[TARGET])
        return self
