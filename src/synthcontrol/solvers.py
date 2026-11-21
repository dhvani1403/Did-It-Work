"""Two solvers for the synthetic control weights.

Both solve the same constrained least-squares problem

    minimise  f(w) = || y - X w ||^2   subject to  w >= 0,  sum(w) = 1,

where y (length T0) is the treated unit's pre-treatment outcome and the columns
of X (T0 x J) are the donors' pre-treatment outcomes.

* ``solve_scipy`` hands the problem to SciPy's SLSQP.
* ``solve_pgd`` is my own projected gradient descent with step 1/L, where
  L = largest eigenvalue of 2 X^T X is the Lipschitz constant of the gradient.

``fw_gap`` computes the Frank-Wolfe duality gap, an upper bound on
f(w) - f(w*) that is zero exactly when the KKT conditions hold.  I use it as the
stopping rule for PGD and as an optimality certificate for any solution.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from .simplex import project_simplex


@dataclass
class SolverResult:
    weights: np.ndarray
    objective: float
    fw_gap: float
    iterations: int
    converged: bool
    history: List[float] = field(default_factory=list)


def objective(X, y, w):
    r = y - X @ w
    return float(r @ r)


def gradient(X, y, w):
    return -2.0 * X.T @ (y - X @ w)


def lipschitz(X):
    """Largest eigenvalue of the Hessian 2 X^T X (the gradient's Lipschitz constant)."""
    s = np.linalg.norm(X, 2)   # largest singular value of X
    return 2.0 * s * s


def fw_gap(X, y, w):
    """Frank-Wolfe gap  g(w) = grad^T w - min_j grad_j  >= f(w) - f(w*).

    By convexity f(w*) >= f(w) + grad^T (w* - w) >= f(w) + min_j grad_j - grad^T w,
    because a linear function on the simplex is minimised at a vertex.  The gap is
    zero exactly when every donor with positive weight has the smallest partial
    derivative, which is the KKT condition for this problem.
    """
    g = gradient(X, y, w)
    return float(g @ w - g.min())


def _check(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be (T0, J) and y must be (T0,)")
    return X, y


def solve_scipy(X, y, w0=None, tol=1e-12, maxiter=2000):
    """Solve with SciPy's SLSQP (sequential quadratic programming)."""
    X, y = _check(X, y)
    J = X.shape[1]
    w0 = np.full(J, 1.0 / J) if w0 is None else np.asarray(w0, dtype=float)
    # Scale the objective so SLSQP's absolute tolerance means the same thing
    # whatever the units of y.
    scale = max(float(y @ y), 1e-300)
    res = minimize(
        lambda w: objective(X, y, w) / scale,
        w0,
        jac=lambda w: gradient(X, y, w) / scale,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * J,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones_like(w)}],
        options={"ftol": tol, "maxiter": maxiter},
    )
    # SLSQP can return tiny negative weights or a sum slightly off 1; tidy up.
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    return SolverResult(w, objective(X, y, w), fw_gap(X, y, w), int(res.nit), bool(res.success))


def solve_pgd(X, y, w0=None, tol=1e-10, max_iter=200_000, accelerated=False,
              record_history=False, step=None):
    """Projected gradient descent  w <- P_simplex(w - grad f(w) / L).

    Stops when the Frank-Wolfe gap, relative to f at the uniform weights, falls
    below ``tol``.  With ``accelerated=True`` it uses FISTA momentum with an
    adaptive restart; the default is the plain method.
    """
    X, y = _check(X, y)
    J = X.shape[1]
    L = lipschitz(X) if step is None else 1.0 / step
    if L == 0:
        w = np.full(J, 1.0 / J)
        return SolverResult(w, objective(X, y, w), 0.0, 0, True)
    w = np.full(J, 1.0 / J) if w0 is None else project_simplex(np.asarray(w0, float))
    ref = max(objective(X, y, np.full(J, 1.0 / J)), 1e-300)

    XtX = X.T @ X
    Xty = X.T @ y
    history = []
    z, t = w.copy(), 1.0
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        point = z if accelerated else w
        g = 2.0 * (XtX @ point - Xty)
        w_new = project_simplex(point - g / L)
        if accelerated:
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = w_new + ((t - 1.0) / t_new) * (w_new - w)
            # restart the momentum if it points uphill
            if (point - w_new) @ (w_new - w) > 0:
                z, t_new = w_new.copy(), 1.0
            t = t_new
        w = w_new
        if record_history:
            history.append(objective(X, y, w))
        if it % 10 == 0 or it == 1:
            gw = 2.0 * (XtX @ w - Xty)
            if (gw @ w - gw.min()) / ref < tol:
                converged = True
                break
    return SolverResult(w, objective(X, y, w), fw_gap(X, y, w), it, converged, history)


def solve(X, y, solver="pgd", **kwargs):
    if solver == "pgd":
        return solve_pgd(X, y, **kwargs)
    if solver == "fista":
        return solve_pgd(X, y, accelerated=True, **kwargs)
    if solver == "scipy":
        return solve_scipy(X, y, **kwargs)
    raise ValueError("unknown solver: %s" % solver)
