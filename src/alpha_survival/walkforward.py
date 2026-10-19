"""Run a forecaster through purged walk-forward splits and collect out-of-sample forecasts.

Per fold: fit a fresh copy of the model on the training months, then predict the
test months ONE MONTH AT A TIME. Within a test block, month m+1's `ret` is month
m's `fwd_ret`; handing a model the whole block at once would let a time-series
model read the answer off the next row. One month per call rules that out.
"""

from __future__ import annotations

import copy

import pandas as pd

from alpha_survival.cv import walk_forward_splits
from alpha_survival.models import TARGET, Forecaster

KEYS = ["permaticker", "month_end"]


def walk_forward_predict(panel: pd.DataFrame, model: Forecaster, **split_kwargs) -> pd.DataFrame:
    """Return one out-of-sample forecast per panel row in a test month.

    Columns: permaticker, month_end, fold, fwd_ret, pred. `split_kwargs` go to
    `walk_forward_splits` (min_train, test_size, horizon, embargo, window, ...).
    `model` is never fit itself; each fold fits a deep copy, so no state leaks
    from one fold into the next. Training rows with a missing target are dropped;
    test rows are all kept (a NaN `fwd_ret` is left for scoring to skip).
    """
    missing = [c for c in [*KEYS, TARGET] if c not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing columns {missing}")
    if panel.duplicated(KEYS).any():
        raise ValueError("panel has duplicate (permaticker, month_end) rows")

    by_month = panel.groupby("month_end")
    out = []
    for fold, (train_months, test_months) in enumerate(
        walk_forward_splits(panel["month_end"], **split_kwargs)
    ):
        train = panel[panel["month_end"].isin(train_months) & panel[TARGET].notna()]
        fitted = copy.deepcopy(model).fit(train)
        for month in test_months:
            rows = by_month.get_group(month)
            pred = fitted.predict(rows.drop(columns=TARGET))
            out.append(rows[[*KEYS, TARGET]].assign(fold=fold, pred=pred))

    result = pd.concat(out, ignore_index=True)
    return result[["permaticker", "month_end", "fold", TARGET, "pred"]]
