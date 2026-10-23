"""Batched least squares from normal equations, shared by the daily summaries
(`data.real`) and the rolling regressions (`features`)."""

from __future__ import annotations

import numpy as np


def solve_normal_equations(xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
    """Solve X'X b = X'y for a stack of regressions: (m, k, k), (m, k) -> (m, k).

    NaN where X'X is singular (a regressor that does not vary). |det| is at most the
    product of the diagonal (Hadamard), so the test is scale-free.
    """
    det = np.linalg.det(xtx)
    solvable = np.abs(det) > 1e-10 * np.prod(np.diagonal(xtx, axis1=1, axis2=2), axis=1)
    beta = np.full(xty.shape, np.nan)
    if solvable.any():
        beta[solvable] = np.linalg.solve(xtx[solvable], xty[solvable][..., None])[..., 0]
    return beta
