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
