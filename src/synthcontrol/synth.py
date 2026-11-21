"""Synthetic control on a T x N panel.

Rows are time periods, columns are units.  The first ``T0`` rows are the
pre-treatment period.  One column is the treated unit; the donors are the
other columns (or a chosen subset).
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .solvers import solve


@dataclass
class SCResult:
    treated: int
    donors: np.ndarray        # column indices of the donors, shape (J,)
    weights: np.ndarray       # shape (J,), non-negative, sums to 1
    T0: int
    observed: np.ndarray      # treated unit's outcome, shape (T,)
    synthetic: np.ndarray     # synthetic control, shape (T,)
    intercept: float          # 0 unless demean=True
    objective: float
    fw_gap: float
    iterations: int
    converged: bool

    @property
    def gap(self):
        """tau_hat_t = Y_1t - synthetic_t for every t (pre-period gaps are fit errors)."""
        return self.observed - self.synthetic

    @property
    def pre_rmspe(self):
        g = self.gap[: self.T0]
        return float(np.sqrt(np.mean(g ** 2)))

    @property
    def post_rmspe(self):
        g = self.gap[self.T0:]
        return float(np.sqrt(np.mean(g ** 2)))

    @property
    def rmspe_ratio(self):
        return self.post_rmspe / max(self.pre_rmspe, 1e-300)

    @property
    def att(self):
        """Average post-treatment gap (the average effect on the treated)."""
        return float(np.mean(self.gap[self.T0:]))

    def weight_table(self, names=None, threshold=1e-4):
        """List of (name, weight) for donors with weight above ``threshold``, largest first."""
        order = np.argsort(-self.weights)
        out = []
        for k in order:
            if self.weights[k] > threshold:
                d = self.donors[k]
                out.append((names[d] if names is not None else int(d), float(self.weights[k])))
        return out


def fit(Y, treated, T0, donors=None, demean=False, solver="pgd", **solver_kwargs):
    """Fit synthetic control weights and build the synthetic series.

    Parameters
    ----------
    Y : array (T, N)
    treated : int, column of the treated unit
    T0 : int, number of pre-treatment periods (rows 0 .. T0-1)
    donors : sequence of column indices, default every column except ``treated``
    demean : bool
        If True, subtract each unit's pre-period mean before fitting, which adds
        a free intercept to the synthetic control (Doudchenko and Imbens 2016;
        Ferman and Pinto 2021).  The weights then match the *shape* of the
        treated series, not its level.
    solver : "pgd", "fista" or "scipy"
    """
    Y = np.asarray(Y, dtype=float)
    T, N = Y.shape
    if not 1 <= T0 < T:
        raise ValueError("need 1 <= T0 < T")
    if donors is None:
        donors = [j for j in range(N) if j != treated]
    donors = np.asarray(donors, dtype=int)
    if treated in set(donors.tolist()):
        raise ValueError("the treated unit cannot be its own donor")

    y = Y[:, treated]
    X = Y[:, donors]
    if demean:
        mu_y = y[:T0].mean()
        mu_X = X[:T0].mean(axis=0)
        y_fit, X_fit = y - mu_y, X - mu_X
    else:
        mu_y, mu_X = 0.0, np.zeros(len(donors))
        y_fit, X_fit = y, X

    res = solve(X_fit[:T0], y_fit[:T0], solver=solver, **solver_kwargs)
    w = res.weights
    intercept = float(mu_y - mu_X @ w)
    synthetic = X @ w + intercept
    return SCResult(treated, donors, w, T0, y.copy(), synthetic, intercept,
                    res.objective, res.fw_gap, res.iterations, res.converged)
