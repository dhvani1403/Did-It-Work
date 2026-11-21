"""Simplex projection, the two weight solvers, and their agreement."""

import numpy as np
import pytest

from synthcontrol.simplex import project_simplex
from synthcontrol import solvers


# Simplex projection -----------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_projection_lands_on_simplex(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 3, rng.integers(1, 40))
    w = project_simplex(v)
    assert np.all(w >= 0)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_projection_is_the_closest_point(seed):
    """Compare with a brute-force check: no feasible perturbation gets closer to v."""
    rng = np.random.default_rng(100 + seed)
    v = rng.normal(0, 2, 8)
    w = project_simplex(v)
    best = np.sum((w - v) ** 2)
    for _ in range(2000):
        u = rng.dirichlet(np.ones(8) * rng.uniform(0.1, 2))
        assert np.sum((u - v) ** 2) >= best - 1e-12


@pytest.mark.parametrize("seed", range(10))
def test_projection_satisfies_kkt(seed):
    """w_j = max(v_j - theta, 0) with one shared theta."""
    rng = np.random.default_rng(200 + seed)
    v = rng.normal(0, 1, 15)
    w = project_simplex(v)
    pos = w > 0
    theta = v[pos] - w[pos]
    assert np.allclose(theta, theta[0], atol=1e-12)
    assert np.all(v[~pos] <= theta[0] + 1e-12)


def test_projection_fixed_points_and_idempotence():
    w = np.array([0.2, 0.5, 0.3])
    assert np.allclose(project_simplex(w), w)
    rng = np.random.default_rng(1)
    v = rng.normal(size=10)
    p = project_simplex(v)
    assert np.allclose(project_simplex(p), p)
    # shifting every entry by a constant does not change the projection
    assert np.allclose(project_simplex(v + 7.3), p)
    # a vector with one huge entry projects to a vertex
    assert np.allclose(project_simplex(np.array([10.0, 0.0, -1.0])), [1, 0, 0])


def test_projection_is_non_expansive():
    rng = np.random.default_rng(2)
    for _ in range(200):
        a, b = rng.normal(size=6), rng.normal(size=6)
        assert np.linalg.norm(project_simplex(a) - project_simplex(b)) <= np.linalg.norm(a - b) + 1e-12


# Solvers ----------------------------------------------------------------------

def random_problem(seed, T0=30, J=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T0, J)).cumsum(axis=0)
    y = X @ rng.dirichlet(np.ones(J)) + rng.normal(0, 0.5, T0)
    return X, y


def test_lipschitz_constant_is_largest_hessian_eigenvalue():
    X, _ = random_problem(0)
    assert solvers.lipschitz(X) == pytest.approx(np.linalg.eigvalsh(2 * X.T @ X).max(), rel=1e-10)


@pytest.mark.parametrize("seed", range(15))
def test_pgd_and_scipy_agree(seed):
    X, y = random_problem(seed)
    a = solvers.solve_scipy(X, y)
    b = solvers.solve_pgd(X, y, tol=1e-12)
    assert b.converged
    assert b.objective <= a.objective * (1 + 1e-8)
    assert np.max(np.abs(a.weights - b.weights)) < 1e-4
    assert b.weights.sum() == pytest.approx(1.0, abs=1e-12) and np.all(b.weights >= 0)


@pytest.mark.parametrize("seed", range(5))
def test_fista_matches_plain_pgd(seed):
    X, y = random_problem(seed)
    a = solvers.solve_pgd(X, y, tol=1e-12)
    b = solvers.solve_pgd(X, y, tol=1e-12, accelerated=True)
    assert np.max(np.abs(a.weights - b.weights)) < 1e-5
    assert b.iterations < a.iterations


def test_pgd_objective_never_increases():
    """With step 1/L each projected gradient step decreases f (descent lemma)."""
    X, y = random_problem(3)
    r = solvers.solve_pgd(X, y, tol=1e-14, max_iter=3000, record_history=True)
    h = np.array(r.history)
    assert np.all(np.diff(h) <= 1e-9 * h[0])


def test_pgd_sublinear_rate_bound():
    """f(w_k) - f* <= L ||w_0 - w*||^2 / (2k) for projected gradient with step 1/L."""
    X, y = random_problem(4)
    ref = solvers.solve_pgd(X, y, tol=1e-14, max_iter=500_000)
    r = solvers.solve_pgd(X, y, tol=0, max_iter=2000, record_history=True)
    L = solvers.lipschitz(X)
    w0 = np.full(X.shape[1], 1 / X.shape[1])
    bound = L * np.sum((w0 - ref.weights) ** 2) / (2 * np.arange(1, 2001))
    assert np.all(np.array(r.history) - ref.objective <= bound + 1e-8)


def test_fw_gap_bounds_suboptimality():
    X, y = random_problem(5)
    best = solvers.solve_pgd(X, y, tol=1e-14).objective
    rng = np.random.default_rng(0)
    for _ in range(100):
        w = rng.dirichlet(np.ones(X.shape[1]))
        assert solvers.objective(X, y, w) - best <= solvers.fw_gap(X, y, w) + 1e-9


def test_exact_convex_combination_is_recovered():
    """Noiseless: y is exactly X w_true with X of full column rank, so w_true is the unique optimum."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(40, 8))
    w_true = np.array([0.5, 0.3, 0.2, 0, 0, 0, 0, 0])
    y = X @ w_true
    for solve in (solvers.solve_scipy, lambda X, y: solvers.solve_pgd(X, y, tol=1e-14)):
        r = solve(X, y)
        assert np.allclose(r.weights, w_true, atol=1e-5)
