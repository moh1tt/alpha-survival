"""Purged walk-forward cross-validation over months.

Standing at a past date, a model may train only on what was known then and is
scored on the months that follow. Splits are made on whole months, never on rows,
so a month's cross-section is entirely in train or entirely in test.

Timing. A row dated month t holds features known at the end of t and a label
covering the next `horizon` months, so the label is only known at the end of
t + horizon. A test row dated s is forecast at the end of s. Training on row t is
therefore leak-free only if t + horizon <= s: the last `horizon - 1` months before
each test block must be purged (Lopez de Prado 2018, Advances in Financial Machine
Learning, ch. 7). With our one-month `fwd_ret`, nothing needs purging.

`embargo` drops extra months on top, as a safety margin (late data, effects that
spill into neighbouring months). It costs training data, so it defaults to 0.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pandas as pd


def walk_forward_splits(
    months: Iterable[pd.Timestamp],
    *,
    min_train: int = 60,
    test_size: int = 12,
    step: int = 12,
    horizon: int = 1,
    embargo: int = 0,
    window: int | None = None,
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield (train_months, test_months) pairs that walk forward through time.

    `months` may hold repeats (e.g. a panel's `month_end` column); they are
    deduplicated and sorted, and must be consecutive calendar months.

    Between the last training month and the first test month, `horizon - 1 +
    embargo` months are dropped. The first test block starts after `min_train`
    training months plus that gap; each later block starts `step` months on. The
    training set grows from the first month (expanding) unless `window` caps it at
    that many most recent months (rolling). The final test block may be shorter
    than `test_size` so the latest months are still tested.
    """
    if min_train < 1 or test_size < 1 or horizon < 1 or embargo < 0:
        raise ValueError("need min_train, test_size, horizon >= 1 and embargo >= 0")
    # step < test_size would score some months twice; step > test_size skips some.
    if step < test_size:
        raise ValueError("step must be >= test_size so test blocks never overlap")
    if window is not None and window < min_train:
        raise ValueError("window must be >= min_train")

    months = pd.DatetimeIndex(pd.unique(pd.Series(months))).sort_values()
    periods = months.to_period("M")
    if len(periods) > 1 and ((periods[1:] - periods[:-1]) != periods.freq).any():
        raise ValueError("months are not consecutive; a gap would make the purge too short")

    gap = horizon - 1 + embargo
    first_test = min_train + gap
    if len(months) <= first_test:
        raise ValueError(
            f"{len(months)} months is too few: need more than min_train + gap = {first_test}"
        )

    for test_start in range(first_test, len(months), step):
        train_end = test_start - gap  # exclusive
        train_start = 0 if window is None else max(0, train_end - window)
        yield months[train_start:train_end], months[test_start : test_start + test_size]
