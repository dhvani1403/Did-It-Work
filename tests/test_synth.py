"""Synthetic control fits, placebo inference and the DiD comparison."""

import numpy as np
import pytest

from synthcontrol import dgp, did, inference, synth


def noiseless_panel(tau, T0=30, T1=8, J=10, seed=0):
    rng = np.random.default_rng(seed)
    donors = rng.normal(size=(T0 + T1, J)).cumsum(axis=0) + 5
    w = np.zeros(J)
    w[[1, 4, 7]] = [0.6, 0.3, 0.1]
    treated = donors @ w
    treated[T0:] += tau
    return np.column_stack([treated, donors]), w


@pytest.mark.parametrize("tau", [0.0, 1.5, -3.0])
@pytest.mark.parametrize("solver", ["scipy", "pgd"])
def test_known_effect_recovered_without_noise(tau, solver):
    Y, w = noiseless_panel(tau)
    r = synth.fit(Y, 0, 30, solver=solver)
    assert np.allclose(r.weights, w, atol=1e-4)
    assert r.pre_rmspe < 1e-4
    assert np.allclose(r.gap[30:], tau, atol=1e-3)
    assert r.att == pytest.approx(tau, abs=1e-3)


def test_time_varying_effect_recovered():
    Y, _ = noiseless_panel(0.0)
    effect = np.linspace(0.5, 4.0, 8)
    Y[30:, 0] += effect
    r = synth.fit(Y, 0, 30)
    assert np.allclose(r.gap[30:], effect, atol=1e-3)


def test_demeaned_fit_absorbs_level_shift():
    Y, w = noiseless_panel(2.0)
    Y[:, 0] += 100.0                      # treated far above every donor
    plain = synth.fit(Y, 0, 30)
    dm = synth.fit(Y, 0, 30, demean=True)
    assert plain.pre_rmspe > 10           # outside the convex hull in levels
    assert dm.pre_rmspe < 1e-4
    assert dm.intercept == pytest.approx(100.0, abs=1e-3)
    assert dm.att == pytest.approx(2.0, abs=1e-3)


def test_weights_are_on_simplex_and_donor_subset_respected():
    Y, _ = noiseless_panel(1.0)
    r = synth.fit(Y, 0, 30, donors=[1, 2, 3, 4])
    assert list(r.donors) == [1, 2, 3, 4]
    assert np.all(r.weights >= 0) and r.weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        synth.fit(Y, 0, 30, donors=[0, 1])


def test_permutation_p_value_range():
    J = 19
    for rank in range(1, J + 2):
        placebos = np.r_[np.full(rank - 1, 10.0), np.full(J - rank + 1, 0.0)]
        assert inference.permutation_p_value(5.0, placebos) == pytest.approx(rank / (J + 1))
    # ties are counted against the treated unit
    assert inference.permutation_p_value(1.0, np.ones(4)) == 1.0


def test_placebo_p_values_are_on_the_grid():
    rng = np.random.default_rng(3)
    p = dgp.simulate_panel(rng, J=12, tau=0.0)
    pl = inference.in_space_placebos(p.Y, 0, p.T0)
    J = 12
    assert 1 / (J + 1) <= pl.p_value <= 1
    assert (pl.p_value * (J + 1)) == pytest.approx(round(pl.p_value * (J + 1)))
    assert pl.gaps.shape == (p.Y.shape[0], J + 1)
    assert list(pl.units) == list(range(J + 1))


def test_large_effect_gets_smallest_p_value():
    rng = np.random.default_rng(4)
    p = dgp.simulate_panel(rng, J=15, tau=15.0)
    pl = inference.in_space_placebos(p.Y, 0, p.T0)
    assert pl.p_value == pytest.approx(1 / 16)


def test_in_time_placebo_uses_only_pre_period():
    Y, _ = noiseless_panel(5.0)
    r = inference.in_time_placebo(Y, 0, T0_fake=20, T_end=30)
    assert r.observed.shape[0] == 30
    assert abs(r.att) < 1e-3              # no effect before the real date


def test_did_equals_equal_weights_plus_intercept():
    Y, _ = noiseless_panel(1.0)
    d = did.did(Y, 0, 30)
    ybar = Y[:, 1:].mean(axis=1)
    manual = (Y[30:, 0].mean() - Y[:30, 0].mean()) - (ybar[30:].mean() - ybar[:30].mean())
    assert d.att == pytest.approx(manual)


def test_did_unbiased_with_parallel_trends_biased_without():
    rng = np.random.default_rng(5)
    T, J, T0 = 40, 10, 30
    common = rng.normal(size=T).cumsum()
    Y = common[:, None] + rng.normal(size=J + 1)[None, :]
    Y[T0:, 0] += 2.0
    assert did.did(Y, 0, T0).att == pytest.approx(2.0, abs=1e-10)
    # give the treated unit its own trend: DiD now picks it up as "effect"
    Y[:, 0] += 0.1 * np.arange(T)
    assert abs(did.did(Y, 0, T0).att - 2.0) > 1.0


def test_dgp_inside_hull_true_weights_reproduce_loadings():
    rng = np.random.default_rng(7)
    p = dgp.simulate_panel(rng, sigma=0.0, tau=0.0)
    X = p.Y0[:, 1:]
    assert np.allclose(X @ p.true_weights, p.Y0[:, 0])
    # X has rank 4 here, so the weights are not unique; the fitted path is.
    r = synth.fit(p.Y, 0, p.T0, solver="fista")
    assert r.pre_rmspe < 1e-4 and abs(r.att) < 1e-3


def test_dgp_outside_hull_cannot_be_fitted():
    rng = np.random.default_rng(8)
    p = dgp.simulate_panel(rng, sigma=0.0, design="outside")
    r = synth.fit(p.Y, 0, p.T0)
    assert r.pre_rmspe > 0.05
