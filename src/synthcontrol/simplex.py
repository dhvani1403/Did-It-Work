"""Euclidean projection onto the probability simplex.

The simplex is  Δ = {w : w_j >= 0, sum_j w_j = z}.  The projection of v is the
solution of  min_w 0.5 ||w - v||^2  subject to  w in Δ.

The KKT conditions give w_j = max(v_j - θ, 0) for a single scalar θ chosen so
that the weights sum to z.  The sum is a decreasing, piecewise-linear function
of θ, so sorting v finds the right piece in O(n log n).  This is the algorithm
of Duchi, Shalev-Shwartz, Singer and Chandra (2008) and Wang and
Carreira-Perpiñán (2013).
"""

import numpy as np


def project_simplex(v, z=1.0):
    """Project the vector ``v`` onto {w >= 0, sum(w) = z}.

    Parameters
    ----------
    v : array_like, shape (n,)
    z : float, the simplex radius (1 for weights that sum to one).

    Returns
    -------
    numpy.ndarray, shape (n,)
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("v must be one-dimensional")
    if z <= 0:
        raise ValueError("z must be positive")
    n = v.size
    u = np.sort(v)[::-1]                      # u_1 >= u_2 >= ... >= u_n
    css = np.cumsum(u) - z                    # partial sums minus the radius
    j = np.arange(1, n + 1)
    # rho = largest j with u_j - (sum_{r<=j} u_r - z) / j > 0.  The condition is
    # true for j = 1 and stays true up to rho, then false, so take the last True.
    cond = u - css / j > 0
    rho = j[cond][-1]
    theta = css[rho - 1] / rho
    return np.maximum(v - theta, 0.0)
