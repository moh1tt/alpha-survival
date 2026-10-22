import numpy as np
import pandas as pd
import pytest

from alpha_survival.combine import rank_average
from alpha_survival.models import TARGET

JAN, FEB = pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")


def oos(preds_jan, preds_feb, fwd=None):
    """Walk-forward-shaped output: 3 stocks over two months."""
    n = len(preds_jan)
    rows = pd.DataFrame(
        {
            "permaticker": list(range(n)) * 2,
            "month_end": [JAN] * n + [FEB] * n,
            "fold": 0,
            TARGET: fwd if fwd is not None else [0.0] * (2 * n),
        }
    )
    return rows.assign(pred=list(preds_jan) + list(preds_feb))


def test_scale_does_not_matter_only_order():
    # b has 1000x wider forecasts but must not outvote a: ranks only.
    a = oos([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    b = oos([3000.0, 2000.0, 1000.0], [3000.0, 2000.0, 1000.0])
    combo = rank_average({"a": a, "b": b}, ["a", "b"])
    assert np.allclose(combo["pred"], 0.0)  # opposite orderings cancel exactly


def test_by_hand_and_per_month():
    a = oos([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    b = oos([1.0, 3.0, 2.0], [3.0, 2.0, 1.0])
    combo = rank_average({"a": a, "b": b}, ["a", "b"]).set_index(["month_end", "permaticker"])
    # Jan ranks: a = (-1, 0, 1), b = (-1, 1, 0) -> mean (-1, 0.5, 0.5). Feb: both (1, 0, -1).
    assert combo.loc[JAN, "pred"].tolist() == pytest.approx([-1.0, 0.5, 0.5])
    assert combo.loc[FEB, "pred"].tolist() == pytest.approx([1.0, 0.0, -1.0])


def test_member_with_no_view_drops_out_of_the_ordering():
    a = oos([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    flat = oos([0.01] * 3, [0.01] * 3)
    combo = rank_average({"a": a, "flat": flat}, ["a", "flat"])
    assert combo["pred"].tolist() == pytest.approx([-0.5, 0.0, 0.5] * 2)  # a's order, halved


def test_keeps_keys_folds_and_outcomes_and_sees_only_its_own_month():
    a = oos([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], fwd=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    b = oos([2.0, 1.0, 3.0], [2.0, 1.0, 3.0], fwd=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    combo = rank_average({"a": a, "b": b}, ["a", "b"])
    assert combo[["permaticker", "month_end", "fold", TARGET]].equals(
        a[["permaticker", "month_end", "fold", TARGET]]
    )
    # Leakage check: changing February's forecasts cannot move January's combination.
    b2 = oos([2.0, 1.0, 3.0], [9.0, -9.0, 0.0])
    jan = combo["month_end"] == JAN
    assert combo.loc[jan, "pred"].equals(
        rank_average({"a": a, "b": b2}, ["a", "b"]).loc[jan, "pred"]
    )


def test_missing_everywhere_stays_missing():
    a = oos([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
    b = oos([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
    combo = rank_average({"a": a, "b": b}, ["a", "b"])
    assert np.isnan(combo.loc[1, "pred"]) and combo["pred"].notna().sum() == 5


def test_bad_inputs_raise():
    a = oos([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="at least two"):
        rank_average({"a": a}, ["a"])
    with pytest.raises(ValueError, match="forecasts 4 rows"):
        rank_average({"a": a, "b": a.iloc[:4]}, ["a", "b"])
