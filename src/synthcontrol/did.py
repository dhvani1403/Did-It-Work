"""Difference-in-differences as a special case of a weighted comparison.

With one treated unit and J controls, the two-way fixed-effects DiD estimate is

    tau_hat_t = (Y_1t - mean_pre Y_1) - (Ybar_t - mean_pre Ybar),

where Ybar_t is the equally weighted average of the controls.  This is a
synthetic control with weights fixed at 1/J plus an intercept (the level
difference in the pre-period), so it needs no optimisation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DiDResult:
    T0: int
    observed: np.ndarray
    counterfactual: np.ndarray

    @property
    def gap(self):
        return self.observed - self.counterfactual

    @property
    def att(self):
        return float(np.mean(self.gap[self.T0:]))

    @property
    def pre_rmspe(self):
        return float(np.sqrt(np.mean(self.gap[: self.T0] ** 2)))


def did(Y, treated, T0, donors=None):
    Y = np.asarray(Y, dtype=float)
    N = Y.shape[1]
    if donors is None:
        donors = [j for j in range(N) if j != treated]
    y = Y[:, treated]
    ybar = Y[:, donors].mean(axis=1)
    counterfactual = ybar + (y[:T0].mean() - ybar[:T0].mean())
    return DiDResult(T0, y.copy(), counterfactual)
