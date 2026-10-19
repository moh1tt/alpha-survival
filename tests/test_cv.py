import pandas as pd
import pytest

from alpha_survival.cv import walk_forward_splits


def month_ends(n: int, start: str = "2000-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def months_between(earlier: pd.Timestamp, later: pd.Timestamp) -> int:
    return (later.to_period("M") - earlier.to_period("M")).n


@pytest.mark.parametrize("horizon", [1, 3, 12])
@pytest.mark.parametrize("embargo", [0, 2])
def test_every_training_label_is_known_before_the_first_forecast(horizon, embargo):
    months = month_ends(100)
    splits = walk_forward_splits(months, min_train=24, horizon=horizon, embargo=embargo)
    for train, test in splits:
        assert train.intersection(test).empty
        # The last training row's label is known `horizon` months after its date;
        # the first test forecast is made at test[0]. Leak-free: label <= forecast.
        assert months_between(train[-1], test[0]) - horizon >= 0
        # And no more data is thrown away than the embargo asks for.
        assert months_between(train[-1], test[0]) == horizon + embargo


def test_one_month_horizon_needs_no_gap():
    # Train rows Jan..Dec (Dec's label = Jan return, known Jan 31); first test row
    # is Jan, forecast at Jan 31. Adjacent months, no leak.
    months = month_ends(24, start="2020-01-31")
    train, test = next(walk_forward_splits(months, min_train=12, test_size=4, step=4))
    assert train[-1] == pd.Timestamp("2020-12-31")
    assert test[0] == pd.Timestamp("2021-01-31")


def test_test_blocks_ordered_and_disjoint_and_cover_the_tail():
    months = month_ends(100)
    tests = [test for _, test in walk_forward_splits(months, min_train=24, test_size=12)]
    for earlier, later in zip(tests, tests[1:], strict=False):
        assert earlier[-1] < later[0]
    tested = tests[0].append(tests[1:])
    assert tested.is_unique
    # Every month after the first training stretch gets tested exactly once.
    assert tested.equals(months[24:])


def test_first_split_layout_with_defaults():
    months = month_ends(100)
    train, test = next(walk_forward_splits(months))
    assert train.equals(months[:60])
    assert test.equals(months[60:72])


def test_expanding_vs_rolling_window():
    months = month_ends(120)
    expanding = list(walk_forward_splits(months, min_train=36))
    assert all(train[0] == months[0] for train, _ in expanding)
    assert len(expanding[-1][0]) > 36

    rolling = list(walk_forward_splits(months, min_train=36, window=36))
    assert all(len(train) == 36 for train, _ in rolling)


def test_accepts_panel_column_with_repeats_and_any_order():
    months = month_ends(40)
    column = pd.Series(list(months[::-1]) * 3)  # 3 stocks, reversed order
    a = list(walk_forward_splits(column, min_train=12, test_size=6, step=6))
    b = list(walk_forward_splits(months, min_train=12, test_size=6, step=6))
    assert len(a) == len(b)
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b, strict=True):
        assert tr_a.equals(tr_b) and te_a.equals(te_b)


def test_short_final_block_is_kept():
    months = month_ends(30)
    *_, (_, last_test) = walk_forward_splits(months, min_train=12, test_size=12)
    assert last_test[-1] == months[-1]
    assert len(last_test) == 30 - 12 - 12  # 6 leftover months


def test_too_little_history_raises():
    with pytest.raises(ValueError, match="too few"):
        list(walk_forward_splits(month_ends(61), min_train=60, embargo=1))


def test_gap_in_months_raises():
    months = month_ends(50).delete(20)
    with pytest.raises(ValueError, match="not consecutive"):
        list(walk_forward_splits(months, min_train=12))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_train": 0},
        {"test_size": 0},
        {"horizon": 0},
        {"embargo": -1},
        {"test_size": 12, "step": 6},
        {"min_train": 36, "window": 24},
    ],
)
def test_bad_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        list(walk_forward_splits(month_ends(100), **kwargs))
