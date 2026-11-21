"""Check my implementation against itself and against an established library.

1. Solver agreement: SciPy SLSQP vs my projected gradient descent (plain and
   accelerated) on 300 simulated problems, plus a convergence trace.
2. pysyncon (Fordham, MIT licence) on exactly the same objective: pre-period
   outcomes only, with its per-row scaling cancelled through custom_V.
3. The California Proposition 99 study (Abadie, Diamond and Hainmueller 2010):
   my outcome-only synthetic control and placebo test, and pysyncon's full
   covariate-based fit, compared with the published weights.

Writes results/solver_agreement.{csv,json}, results/pgd_convergence.csv,
results/library_agreement.json, results/prop99.{json,csv} and
results/prop99_placebo_gaps.csv.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from synthcontrol import dgp, inference, solvers, synth  # noqa: E402

RESULTS = ROOT / "results"
PROP99 = ROOT / "data" / "external" / "prop99_smoking.csv"

# Abadie, Diamond and Hainmueller (2010), Table 2
ADH_WEIGHTS = {"Utah": 0.334, "Nevada": 0.234, "Montana": 0.199, "Colorado": 0.164, "Connecticut": 0.069}


# 1. Solver agreement ------------------------------------------------------------

def solver_agreement(n=300):
    rows = []
    for i in range(n):
        rng = np.random.default_rng([99, i])
        design = "inside" if i % 3 else "outside"
        J = int(rng.choice([5, 10, 20, 40]))
        T0 = int(rng.choice([10, 20, 40, 80]))
        p = dgp.simulate_panel(rng, J=J, T0=T0, design=design)
        X, y = p.Y[:T0, 1:], p.Y[:T0, 0]
        t = time.perf_counter(); a = solvers.solve_scipy(X, y); ta = time.perf_counter() - t
        t = time.perf_counter(); b = solvers.solve_pgd(X, y, tol=1e-10); tb = time.perf_counter() - t
        t = time.perf_counter(); c = solvers.solve_pgd(X, y, tol=1e-10, accelerated=True); tc = time.perf_counter() - t
        rank = np.linalg.matrix_rank(X)
        rows.append(dict(
            problem=i, design=design, J=J, T0=T0, full_rank=bool(rank == J),
            max_w_diff=float(np.max(np.abs(a.weights - b.weights))),
            max_w_diff_fista=float(np.max(np.abs(c.weights - b.weights))),
            max_fit_diff=float(np.max(np.abs(X @ (a.weights - b.weights)))),
            rel_obj_diff=float((a.objective - b.objective) / max(b.objective, 1e-300)),
            scipy_fw_gap=a.fw_gap, pgd_fw_gap=b.fw_gap,
            pgd_converged=b.converged, pgd_iters=b.iterations, fista_iters=c.iterations,
            scipy_ms=1e3 * ta, pgd_ms=1e3 * tb, fista_ms=1e3 * tc,
            n_positive=int(np.sum(b.weights > 1e-6))))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "solver_agreement.csv", index=False)
    fr = df[df.full_rank]
    summary = dict(
        problems=len(df), full_rank_problems=len(fr),
        pgd_converged_share=float(df.pgd_converged.mean()),
        max_w_diff_full_rank=float(fr.max_w_diff.max()),
        median_w_diff_full_rank=float(fr.max_w_diff.median()),
        max_w_diff_all=float(df.max_w_diff.max()),
        max_fit_diff_all=float(df.max_fit_diff.max()),
        max_rel_obj_diff=float(df.rel_obj_diff.abs().max()),
        share_pgd_lower_objective=float(np.mean(df.rel_obj_diff >= 0)),
        median_pgd_iters=float(df.pgd_iters.median()), median_fista_iters=float(df.fista_iters.median()),
        median_ms=dict(scipy=float(df.scipy_ms.median()), pgd=float(df.pgd_ms.median()),
                       fista=float(df.fista_ms.median())))
    return summary


def convergence_trace():
    """Objective gap per iteration for plain PGD and FISTA, with the O(1/k) bound."""
    rng = np.random.default_rng(5)
    p = dgp.simulate_panel(rng, J=20, T0=40)
    X, y = p.Y[:40, 1:], p.Y[:40, 0]
    ref = solvers.solve_pgd(X, y, tol=1e-15, accelerated=True, max_iter=10 ** 6)
    K = 5000
    a = solvers.solve_pgd(X, y, tol=0, max_iter=K, record_history=True)
    b = solvers.solve_pgd(X, y, tol=0, max_iter=K, record_history=True, accelerated=True)
    L = solvers.lipschitz(X)
    w0 = np.full(20, 1 / 20)
    k = np.arange(1, K + 1)
    pd.DataFrame(dict(k=k, pgd=np.array(a.history) - ref.objective,
                      fista=np.array(b.history) - ref.objective,
                      bound=L * np.sum((w0 - ref.weights) ** 2) / (2 * k))
                 ).to_csv(RESULTS / "pgd_convergence.csv", index=False)
    eig = np.linalg.eigvalsh(2 * X.T @ X)
    return dict(L=float(L), lambda_min=float(eig.min()), condition_number=float(eig.max() / eig.min()),
                f_star=ref.objective)


# 2. pysyncon on the same objective --------------------------------------------

def pysyncon_same_objective(Z0, Z1):
    """Run pysyncon with X = pre-period outcomes and the row scaling undone.

    pysyncon divides each row of X by its standard deviation across units and
    then minimises sum_t V_t (X1_t - X0_t w)^2.  Setting V_t to that row's
    variance makes its objective identical to mine.
    """
    from pysyncon import Synth
    X = pd.concat([Z0, Z1], axis=1)
    V = (X.std(axis=1) ** 2).to_numpy()
    s = Synth()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.fit(X0=Z0, X1=Z1, Z0=Z0, Z1=Z1, custom_V=V)
    return s.W


def library_checks():
    out = {}
    import pysyncon
    out["pysyncon_version"] = getattr(pysyncon, "__version__", "1.7.0 (pip)")

    # (a) 50 simulated panels
    diffs, effdiffs = [], []
    for i in range(50):
        rng = np.random.default_rng([7, i])
        p = dgp.simulate_panel(rng, J=20, T0=40, tau=1.0)
        T0 = p.T0
        cols = ["u%d" % j for j in range(21)]
        Y = pd.DataFrame(p.Y, columns=cols)
        W = pysyncon_same_objective(Y.iloc[:T0, 1:], Y.iloc[:T0, 0])
        r = synth.fit(p.Y, 0, T0, solver="pgd", tol=1e-12)
        diffs.append(np.max(np.abs(W - r.weights)))
        eff_lib = np.mean(p.Y[T0:, 0] - p.Y[T0:, 1:] @ W)
        effdiffs.append(abs(eff_lib - r.att))
    out["simulated"] = dict(panels=50, max_weight_diff=float(np.max(diffs)),
                            median_weight_diff=float(np.median(diffs)),
                            max_effect_diff=float(np.max(effdiffs)))

    # (b) the Italian ChatGPT panel used in the real-data analysis (demeaned log views)
    import italy_analysis as ia
    df = ia.load("chatgpt")
    donors, _ = ia.screen(df, [l for l in ia.cfg.DONOR_LANGS if l in df.columns])
    Ymat, dates, T0 = ia.make_panel(df, donors)
    Yd = Ymat - Ymat[:T0].mean(axis=0)
    cols = [ia.TREATED] + donors
    Ydf = pd.DataFrame(Yd, columns=cols)
    W = pysyncon_same_objective(Ydf.iloc[:T0, 1:], Ydf.iloc[:T0, 0])
    r = synth.fit(Ymat, 0, T0, demean=True, solver="pgd")
    lib_gap = Yd[T0:, 0] - Yd[T0:, 1:] @ W
    out["italy_chatgpt"] = dict(max_weight_diff=float(np.max(np.abs(W - r.weights))),
                                mean_log_gap_mine=r.att, mean_log_gap_pysyncon=float(lib_gap.mean()),
                                weights_mine={c: round(float(w), 4) for c, w in zip(donors, r.weights) if w > 1e-3},
                                weights_pysyncon={c: round(float(w), 4) for c, w in zip(donors, W) if w > 1e-3})
    return out


# 3. Proposition 99 --------------------------------------------------------------

def prop99():
    df = pd.read_csv(PROP99)
    df["year"] = df.year.astype(int)
    wide = df.pivot(index="year", columns="state", values="cigsale")
    states = ["California"] + [s for s in wide.columns if s != "California"]
    wide = wide[states]
    years = wide.index.to_numpy()
    T0 = int(np.sum(years < 1989))
    Y = wide.to_numpy()
    out = dict(T0=T0, J=len(states) - 1, years=[int(years[0]), int(years[-1])])

    # (a) my outcome-only synthetic control, both solvers
    # X has rank 19 < 38 donors here, so f is not strongly convex and plain PGD
    # only converges at the O(1/k) rate.  I record where it gets to after 200,000
    # steps, and use the accelerated version (same projection and step) as the answer.
    r_plain = synth.fit(Y, 0, T0, solver="pgd", tol=1e-12, max_iter=200_000)
    r = synth.fit(Y, 0, T0, solver="fista", tol=1e-13, max_iter=2_000_000)
    r_sp = synth.fit(Y, 0, T0, solver="scipy")
    out["mine"] = dict(
        weights={n: round(w, 3) for n, w in r.weight_table(names=states, threshold=1e-3)},
        pre_rmspe=r.pre_rmspe, gap_2000=float(r.gap[-1]),
        mean_gap_1989_2000=r.att, rank_X=int(np.linalg.matrix_rank(Y[:T0, 1:])),
        scipy_max_weight_diff=float(np.max(np.abs(r.weights - r_sp.weights))),
        scipy_max_synthetic_diff=float(np.max(np.abs(r.synthetic - r_sp.synthetic))),
        scipy_objective=r_sp.objective, fista_objective=r.objective, fista_iterations=r.iterations,
        fista_converged=r.converged, fista_fw_gap=r.fw_gap, scipy_fw_gap=r_sp.fw_gap,
        plain_pgd_objective_200k=r_plain.objective, plain_pgd_converged=r_plain.converged,
        plain_pgd_fw_gap=r_plain.fw_gap,
        plain_pgd_max_weight_diff=float(np.max(np.abs(r_plain.weights - r.weights))))
    pl = inference.in_space_placebos(Y, 0, T0, solver="fista", tol=1e-12)
    out["mine"]["p_value"] = pl.p_value
    out["mine"]["rmspe_ratio"] = float(pl.ratios[0])
    # ADH also report the test after dropping placebos whose pre-period fit is
    # more than 2x (or 5x) worse than California's
    for k in (2, 5):
        ok = pl.pre_rmspe[1:] <= k * pl.pre_rmspe[0]
        out["mine"]["p_value_drop_%dx" % k] = inference.permutation_p_value(pl.ratios[0], pl.ratios[1:][ok])
        out["mine"]["placebos_kept_%dx" % k] = int(ok.sum())
    gaps = pd.DataFrame(pl.gaps, columns=[states[u] for u in pl.units])
    gaps.insert(0, "year", years)
    gaps.to_csv(RESULTS / "prop99_placebo_gaps.csv", index=False)

    # (b) pysyncon on the same objective
    Z = wide.loc[years < 1989]
    W = pysyncon_same_objective(Z.iloc[:, 1:], Z.iloc[:, 0])
    syn_lib = Y[:, 1:] @ W
    out["pysyncon_same_objective"] = dict(
        weights={s: round(float(w), 3) for s, w in zip(states[1:], W) if w > 1e-3},
        max_weight_diff_vs_mine=float(np.max(np.abs(W - r.weights))),
        max_synthetic_diff_vs_mine=float(np.max(np.abs(syn_lib - r.synthetic))),
        gap_2000=float(Y[-1, 0] - syn_lib[-1]))

    # (c) pysyncon's full ADH specification (covariates + nested V optimisation)
    from pysyncon import Dataprep, Synth
    controls = states[1:]
    dp = Dataprep(
        foo=df, predictors=["lnincome", "age15to24", "retprice"], predictors_op="mean",
        time_predictors_prior=range(1980, 1989),
        special_predictors=[("beer", range(1984, 1989), "mean"), ("cigsale", [1975], "mean"),
                            ("cigsale", [1980], "mean"), ("cigsale", [1988], "mean")],
        dependent="cigsale", unit_variable="state", time_variable="year",
        treatment_identifier="California", controls_identifier=controls,
        time_optimize_ssr=range(1970, 1989))
    s = Synth()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.fit(dataprep=dp, optim_method="Nelder-Mead", optim_initial="equal")
    Wc = s.W
    if isinstance(Wc, pd.Series):
        wmap = Wc.to_dict()
    else:
        wmap = dict(zip(controls, Wc))
    wvec = np.array([wmap[c] for c in controls])
    syn_c = Y[:, 1:] @ wvec
    out["pysyncon_adh_covariates"] = dict(
        weights={k: round(float(v), 3) for k, v in sorted(wmap.items(), key=lambda kv: -kv[1]) if v > 1e-3},
        pre_rmspe=float(np.sqrt(np.mean((Y[:T0, 0] - syn_c[:T0]) ** 2))),
        gap_2000=float(Y[-1, 0] - syn_c[-1]),
        mean_gap_1989_2000=float(np.mean(Y[T0:, 0] - syn_c[T0:])))
    out["published_adh_weights"] = ADH_WEIGHTS
    pd.DataFrame(dict(year=years, california=Y[:, 0], synthetic_mine=r.synthetic,
                      synthetic_pysyncon_same=syn_lib, synthetic_pysyncon_adh=syn_c,
                      donor_mean=Y[:, 1:].mean(axis=1))).to_csv(RESULTS / "prop99.csv", index=False)
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    sa = solver_agreement()
    sa["convergence_example"] = convergence_trace()
    with open(RESULTS / "solver_agreement.json", "w") as f:
        json.dump(sa, f, indent=2)
    print(json.dumps(sa, indent=1))
    lib = library_checks()
    with open(RESULTS / "library_agreement.json", "w") as f:
        json.dump(lib, f, indent=2)
    print(json.dumps(lib, indent=1))
    p99 = prop99()
    with open(RESULTS / "prop99.json", "w") as f:
        json.dump(p99, f, indent=2)
    print(json.dumps(p99, indent=1))


if __name__ == "__main__":
    main()
