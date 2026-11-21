"""Placebo (permutation) inference for synthetic control.

In-space placebos: pretend each donor was the treated unit, fit a synthetic
control for it from the remaining donors (the real treated unit is left out,
because its post-period is contaminated by the treatment), and compare the
treated unit's post/pre RMSPE ratio with the placebo distribution.

In-time placebo: move the treatment date back into the pre-period and use only
data before the real treatment.  A large "effect" there is a warning sign.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .synth import fit


@dataclass
class PlaceboResult:
    units: np.ndarray        # treated first, then each donor
    ratios: np.ndarray       # post/pre RMSPE ratio per unit
    pre_rmspe: np.ndarray
    post_rmspe: np.ndarray
    atts: np.ndarray         # mean post-period gap per unit
    gaps: np.ndarray         # (T, J+1) gap series, column 0 is the treated unit
    p_value: float
    p_value_att: float       # two-sided, based on |ATT| / pre-RMSPE


def permutation_p_value(stat_treated, stat_placebos):
    """p = rank / (J + 1), where rank 1 is the largest statistic.

    Ties count against the treated unit, so the p-value is conservative.  The
    smallest possible value is 1 / (J + 1).
    """
    stat_placebos = np.asarray(stat_placebos, dtype=float)
    rank = 1 + int(np.sum(stat_placebos >= stat_treated))
    return rank / (len(stat_placebos) + 1)


def in_space_placebos(Y, treated, T0, donors=None, demean=False, solver="pgd", **kw):
    Y = np.asarray(Y, dtype=float)
    N = Y.shape[1]
    if donors is None:
        donors = [j for j in range(N) if j != treated]
    donors = list(donors)
    units = [treated] + donors
    results = [fit(Y, treated, T0, donors, demean=demean, solver=solver, **kw)]
    for d in donors:
        others = [j for j in donors if j != d]
        results.append(fit(Y, d, T0, others, demean=demean, solver=solver, **kw))

    ratios = np.array([r.rmspe_ratio for r in results])
    pre = np.array([r.pre_rmspe for r in results])
    post = np.array([r.post_rmspe for r in results])
    atts = np.array([r.att for r in results])
    gaps = np.column_stack([r.gap for r in results])
    p = permutation_p_value(ratios[0], ratios[1:])
    z = np.abs(atts) / np.maximum(pre, 1e-300)
    p_att = permutation_p_value(z[0], z[1:])
    return PlaceboResult(np.array(units), ratios, pre, post, atts, gaps, p, p_att)


def in_time_placebo(Y, treated, T0_fake, T_end, donors=None, demean=False, solver="pgd", **kw):
    """Fit with a fake treatment date ``T0_fake`` using rows 0 .. T_end-1 only.

    ``T_end`` should be the real T0 so that no truly treated period is used.
    """
    if not 1 <= T0_fake < T_end:
        raise ValueError("need 1 <= T0_fake < T_end")
    Y = np.asarray(Y, dtype=float)[:T_end]
    return fit(Y, treated, T0_fake, donors, demean=demean, solver=solver, **kw)
