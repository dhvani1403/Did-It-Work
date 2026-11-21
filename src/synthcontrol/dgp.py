"""Simulated panels from a linear factor model with a planted treatment effect.

    Y_it(0) = alpha_i + sum_f lambda_tf * mu_if + eps_it,      eps_it ~ N(0, sigma^2)
    Y_1t    = Y_1t(0) + tau   for t > T0   (unit 0 is treated)

Factors:
  * f = 0 is a deterministic linear trend, lambda_t0 = trend_scale * t / T.
    Units load on it differently, so their trends are not parallel and
    difference-in-differences is biased.
  * f = 1 .. F-1 are stationary AR(1) processes with unit variance.

Donor loadings (alpha_i, mu_i) are drawn independently.  The treated unit's
loadings are either

  * "inside":  a Dirichlet-weighted average of the ``k`` donors with the largest
    trend loadings, so it lies inside the donors' convex hull but far from their
    average (the setting where synthetic control should beat DiD), or
  * "outside": the same, then its trend loading is pushed ``outside_shift``
    beyond the largest donor's, so no convex combination can match it.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Panel:
    Y: np.ndarray           # (T, J+1), column 0 treated
    Y0: np.ndarray          # untreated potential outcomes, same shape
    T0: int
    tau: float
    true_weights: np.ndarray  # weights that reproduce the treated loadings ("inside" only)


def ar1(T, rho, rng):
    x = np.empty(T)
    x[0] = rng.normal()
    s = np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + s * rng.normal()
    return x


def simulate_panel(rng, J=20, T0=40, T1=10, F=3, tau=0.0, sigma=1.0,
                   trend_scale=6.0, rho=0.8, k=4, design="inside", outside_shift=0.6,
                   trend_T=None):
    T = T0 + T1
    t = np.arange(1, T + 1)
    lam = np.empty((T, F))
    # by default the trend rises by trend_scale over the whole sample; pass
    # trend_T to fix the slope per period instead (used when T0 is varied)
    lam[:, 0] = trend_scale * t / (T if trend_T is None else trend_T)
    for f in range(1, F):
        lam[:, f] = ar1(T, rho, rng)

    alpha_d = rng.normal(0.0, 1.0, J)
    mu_d = rng.uniform(0.0, 1.0, (J, F))

    top = np.argsort(-mu_d[:, 0])[:k]
    v = rng.dirichlet(np.ones(k))
    true_w = np.zeros(J)
    true_w[top] = v
    alpha_1 = true_w @ alpha_d
    mu_1 = true_w @ mu_d
    if design == "outside":
        mu_1 = mu_1.copy()
        mu_1[0] = mu_d[:, 0].max() + outside_shift
    elif design != "inside":
        raise ValueError("design must be 'inside' or 'outside'")

    alpha = np.concatenate([[alpha_1], alpha_d])
    mu = np.vstack([mu_1, mu_d])
    Y0 = alpha[None, :] + lam @ mu.T + sigma * rng.normal(size=(T, J + 1))
    Y = Y0.copy()
    Y[T0:, 0] += tau
    return Panel(Y, Y0, T0, tau, true_w if design == "inside" else np.full(J, np.nan))
