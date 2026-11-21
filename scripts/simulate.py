"""Simulation study: does synthetic control recover planted effects, and is the
placebo test calibrated?

For each design and true effect tau, simulate panels from the factor model in
``synthcontrol.dgp`` and record, per replication:
  * the synthetic control estimate of the average post-period effect,
  * the same with a free intercept (demeaned synthetic control),
  * the difference-in-differences estimate,
  * the pre-period RMSPE, and
  * the in-space placebo p-value (post/pre RMSPE ratio).

Writes results/simulation_reps.csv, results/simulation_summary.{csv,md,json}
and results/simulation_examples.csv (one panel per design, for the figures).
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from synthcontrol import dgp, did, inference, synth  # noqa: E402

RESULTS = ROOT / "results"
SOLVER = "fista"   # my accelerated projected gradient; same answer as plain PGD, ~10x fewer steps

CONFIG = dict(J=20, T0=40, T1=10, F=3, sigma=1.0, trend_scale=6.0, rho=0.8, k=4, outside_shift=0.6)
DESIGNS = [
    # (design, tau, replications)
    ("inside", 0.0, 1000),
    ("inside", 0.5, 500),
    ("inside", 1.0, 500),
    ("inside", 2.0, 500),
    ("inside", 4.0, 500),
    ("outside", 0.0, 500),
    ("outside", 2.0, 500),
]
BASE_SEED = 20260101


def one_rep(args):
    design, tau, rep = args
    # one independent stream per (design, tau, rep), so results do not depend on scheduling
    seed = [BASE_SEED, ["inside", "outside"].index(design), int(round(tau * 10)), rep]
    rng = np.random.default_rng(seed)
    p = dgp.simulate_panel(rng, tau=tau, design=design, **CONFIG)
    pl = inference.in_space_placebos(p.Y, 0, p.T0, solver=SOLVER)
    sc_dm = synth.fit(p.Y, 0, p.T0, demean=True, solver=SOLVER)
    d = did.did(p.Y, 0, p.T0)
    # pre-RMSPE of the treated unit relative to the median placebo pre-RMSPE
    return dict(design=design, tau=tau, rep=rep,
                sc_att=pl.atts[0], sc_dm_att=sc_dm.att, did_att=d.att,
                pre_rmspe=pl.pre_rmspe[0], did_pre_rmspe=d.pre_rmspe,
                pre_rmspe_rank=int(1 + np.sum(pl.pre_rmspe[1:] >= pl.pre_rmspe[0])),
                p_value=pl.p_value, p_value_att=pl.p_value_att)


SWEEP_REPS = 400
SWEEP_T0 = [10, 20, 40, 80, 160, 320]
SWEEP_SIGMA = [0.0, 0.25, 0.5, 1.0, 2.0]


def sweep_rep(args):
    """No placebos here: only the point estimates, for the bias-vs-T0 and bias-vs-noise curves."""
    kind, value, rep = args
    seed = [BASE_SEED, 7, ["T0", "sigma"].index(kind), int(round(value * 100)), rep]
    rng = np.random.default_rng(seed)
    cfg = dict(CONFIG)
    if kind == "T0":
        # fix the trend's slope per period at its default, so a longer pre-period
        # means more history, not a flatter trend
        cfg.update(T0=int(value), trend_T=CONFIG["T0"] + CONFIG["T1"])
    else:
        cfg.update(sigma=value)
    p = dgp.simulate_panel(rng, tau=0.0, design="inside", **cfg)
    r = synth.fit(p.Y, 0, p.T0, solver=SOLVER)
    return dict(kind=kind, value=value, rep=rep, sc_att=r.att, did_att=did.did(p.Y, 0, p.T0).att,
                pre_rmspe=r.pre_rmspe)


def summarise(df):
    rows = []
    for (design, tau), g in df.groupby(["design", "tau"], sort=False):
        row = dict(design=design, tau=tau, reps=len(g))
        for name in ["sc", "sc_dm", "did"]:
            err = g[name + "_att"] - tau
            row[name + "_bias"] = err.mean()
            row[name + "_bias_se"] = err.std(ddof=1) / np.sqrt(len(g))
            row[name + "_rmse"] = np.sqrt(np.mean(err ** 2))
        row["pre_rmspe_median"] = g.pre_rmspe.median()
        row["reject_05"] = np.mean(g.p_value <= 0.05)
        row["reject_10"] = np.mean(g.p_value <= 0.10)
        row["reject_05_att"] = np.mean(g.p_value_att <= 0.05)
        row["mean_p"] = g.p_value.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def examples():
    """One noisy panel per design (tau = 2) with the SC and DiD counterfactuals, for figures."""
    out = []
    for design, seed in [("inside", 11), ("outside", 11)]:
        rng = np.random.default_rng(seed)
        p = dgp.simulate_panel(rng, tau=2.0, design=design, **CONFIG)
        r = synth.fit(p.Y, 0, p.T0, solver="pgd")
        d = did.did(p.Y, 0, p.T0)
        T = p.Y.shape[0]
        for t in range(T):
            out.append(dict(design=design, t=t + 1, T0=p.T0, observed=p.Y[t, 0], untreated=p.Y0[t, 0],
                            synthetic=r.synthetic[t], did=d.counterfactual[t],
                            donor_min=p.Y[t, 1:].min(), donor_max=p.Y[t, 1:].max(),
                            pre_rmspe=r.pre_rmspe))
    return pd.DataFrame(out)


def main():
    RESULTS.mkdir(exist_ok=True)
    tasks = [(d, tau, r) for d, tau, R in DESIGNS for r in range(R)]
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(one_rep, tasks, chunksize=20))
    df = pd.DataFrame(rows)
    print("simulated %d panels in %.0f s" % (len(df), time.time() - t0))
    df.to_csv(RESULTS / "simulation_reps.csv", index=False)

    s = summarise(df)
    s.to_csv(RESULTS / "simulation_summary.csv", index=False)

    # false-positive rate under the null, with a binomial standard error
    null = df[(df.design == "inside") & (df.tau == 0)]
    J = CONFIG["J"]
    hist = np.bincount(np.rint(null.p_value * (J + 1)).astype(int), minlength=J + 2)[1:]
    fpr = float(np.mean(null.p_value <= 0.05))
    meta = dict(config=CONFIG, designs=DESIGNS, solver=SOLVER,
                null_reps=len(null), null_reject_05=fpr,
                null_reject_05_se=float(np.sqrt(fpr * (1 - fpr) / len(null))),
                null_reject_10=float(np.mean(null.p_value <= 0.10)),
                null_mean_p=float(null.p_value.mean()),
                nominal_reject_05=1 / (J + 1), nominal_reject_10=2 / (J + 1),
                null_rank_counts=hist.tolist())
    # how well the pre-period fit separates the inside and outside designs
    ins = df[df.design == "inside"].pre_rmspe
    outs = df[df.design == "outside"].pre_rmspe
    thr = float(np.quantile(ins, 0.95))
    meta["pre_rmspe_inside_median"] = float(ins.median())
    meta["pre_rmspe_outside_median"] = float(outs.median())
    meta["pre_rmspe_flag_threshold_q95_inside"] = thr
    meta["outside_flagged_share"] = float(np.mean(outs > thr))
    meta["outside_treated_worst_pre_fit_share"] = float(np.mean(df[df.design == "outside"].pre_rmspe_rank == 1))
    meta["inside_treated_worst_pre_fit_share"] = float(np.mean(df[df.design == "inside"].pre_rmspe_rank == 1))
    with open(RESULTS / "simulation_summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    cols = ["design", "tau", "reps", "sc_bias", "sc_rmse", "sc_dm_bias", "sc_dm_rmse",
            "did_bias", "did_rmse", "pre_rmspe_median", "reject_05", "reject_10"]
    with open(RESULTS / "simulation_summary.md", "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in s[cols].iterrows():
            f.write("| " + " | ".join(
                "%.3f" % r[c] if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n")
    examples().to_csv(RESULTS / "simulation_examples.csv", index=False)

    tasks = ([("T0", v, r) for v in SWEEP_T0 for r in range(SWEEP_REPS)]
             + [("sigma", v, r) for v in SWEEP_SIGMA for r in range(SWEEP_REPS)])
    with ProcessPoolExecutor() as ex:
        sw = pd.DataFrame(list(ex.map(sweep_rep, tasks, chunksize=20)))
    agg = sw.groupby(["kind", "value"]).agg(
        reps=("sc_att", "size"),
        sc_bias=("sc_att", "mean"), sc_sd=("sc_att", "std"),
        sc_rmse=("sc_att", lambda x: float(np.sqrt(np.mean(x ** 2)))),
        did_bias=("did_att", "mean"), did_sd=("did_att", "std"),
        pre_rmspe=("pre_rmspe", "median")).reset_index()
    agg["sc_bias_se"] = agg.sc_sd / np.sqrt(agg.reps)
    agg.to_csv(RESULTS / "simulation_sweeps.csv", index=False)
    print(agg.round(3).to_string(index=False))
    print(s[cols].round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("designs", "config")}, indent=1))


if __name__ == "__main__":
    main()
