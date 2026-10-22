"""Combine several models' out-of-sample forecasts into one.

Equal weights, fixed in advance: weights estimated from past accuracy are noisy
and rarely beat a simple average out of sample (the "forecast combination puzzle",
Timmermann 2006). Nothing is fit, so a combination uses only the forecasts made
in the same month and adds no look-ahead of its own.
"""

from __future__ import annotations

import pandas as pd

from alpha_survival.features import rank_normalise
from alpha_survival.models import TARGET

KEYS = ["permaticker", "month_end"]


def rank_average(oos_by_model: dict[str, pd.DataFrame], members: list[str]) -> pd.DataFrame:
    """Each month, rank each member's forecasts to [-1, 1] and average them.

    Ranks put members on one scale: a model with wide forecasts would otherwise
    outvote one with compressed forecasts. A member with no view that month (all
    forecasts equal) ranks every stock at 0 and so drops out of the ordering.
    `oos_by_model` holds `walk_forward_predict` outputs, which must cover the same
    stock-months. Returns the same columns (permaticker, month_end, fold, fwd_ret,
    pred); `pred` is a rank score, not a return. NaN where no member has a forecast.
    """
    if len(members) < 2:
        raise ValueError("a combination needs at least two members")
    base = oos_by_model[members[0]][[*KEYS, "fold", TARGET]].reset_index(drop=True)
    preds = base[KEYS].copy()
    for name in members:
        oos = oos_by_model[name]
        if len(oos) != len(base):
            raise ValueError(f"{name} forecasts {len(oos)} rows, {members[0]} {len(base)}")
        member = oos[[*KEYS, "pred"]].rename(columns={"pred": name})
        preds = preds.merge(member, on=KEYS, how="left", validate="one_to_one")
    ranked = rank_normalise(preds, members)
    score = ranked[[f"{name}_rank" for name in members]].mean(axis=1)
    return base.assign(pred=score.where(preds[members].notna().any(axis=1)))
