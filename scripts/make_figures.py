"""Draw every figure from the files in results/ (run the analysis scripts first)."""

import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style as ps  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
BAN = pd.Timestamp("2023-03-31")
LIFT = pd.Timestamp("2023-04-28")
GREY = "#b9b8b3"
BAN_SHADE = "#f3d9cf"


def load_json(name):
    with open(R / name) as f:
        return json.load(f)


def ban_shade(ax, label=True):
    ax.axvspan(BAN, LIFT, color=BAN_SHADE, alpha=0.6, lw=0, zorder=0)
    if label:
        ax.text(BAN + (LIFT - BAN) / 2, 1.0, "ChatGPT blocked in Italy", transform=ax.get_xaxis_transform(),
                va="top", ha="center", fontsize=9, color=ps.INK_2)


def month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


# Real data: Italy's ChatGPT ban --------------------------------------------------

def fig_italy_hero():
    s = pd.read_csv(R / "italy_chatgpt_series.csv", parse_dates=["date"])
    m = load_json("italy_chatgpt_main.json")
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
    a.plot(s.date, s.observed_views, color=ps.BLUE, lw=1.6, label="Italian Wikipedia (observed)")
    a.plot(s.date, s.synthetic_views, color=ps.ORANGE, lw=1.6, label="Synthetic Italy (other language editions)")
    a.set_yscale("log")
    a.set_yticks([2000, 3000, 5000, 10000, 20000])
    a.set_yticklabels(["2k", "3k", "5k", "10k", "20k"])
    a.yaxis.set_minor_formatter(mticker.NullFormatter())
    a.set_ylabel("Daily views of 'ChatGPT' (log scale)")
    a.legend(loc="upper left", fontsize=9)
    a.set_title("Italian Wikipedia views of the ChatGPT article vs its synthetic control")
    ban_shade(a)
    b.plot(s.date, s.gap, color=ps.BLUE, lw=1.4)
    b.axhline(0, color=ps.INK_2, lw=0.8)
    ban_shade(b, label=False)
    b.set_ylabel("Gap in log views")
    for key, x, y, txt in [("ban_week1", pd.Timestamp("2023-02-05"), 1.0, "ban week 1: %+.0f%%"),
                           ("ban_weeks2_4", pd.Timestamp("2023-02-05"), -0.75, "ban weeks 2-4: %+.0f%%"),
                           ("lift", pd.Timestamp("2023-05-01"), -0.75, "after the lift: %+.0f%%")]:
        b.text(x, y, txt % m[key + "_pct"], fontsize=9.5, color=ps.INK)
    b.set_ylim(-1.0, 1.4)
    month_axis(b)
    ps.save(fig, "italy_chatgpt_synthetic_control.png")


def fig_italy_placebos():
    g = pd.read_csv(R / "italy_chatgpt_placebo_gaps.csv", parse_dates=["date"])
    st = pd.read_csv(R / "italy_chatgpt_placebo_stats.csv")
    m = load_json("italy_chatgpt_main.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.7, 1]))
    for c in g.columns[2:]:
        a.plot(g.date, g[c], color=GREY, lw=0.8, alpha=0.8)
    a.plot([], [], color=GREY, lw=1, label="each of the %d donor languages, treated as if it were Italian" % (len(g.columns) - 2))
    a.plot(g.date, g["it"], color=ps.BLUE, lw=1.6, label="Italian")
    a.axhline(0, color=ps.INK_2, lw=0.8)
    ban_shade(a, label=False)
    a.set_ylabel("Gap in log views (actual - synthetic)")
    a.set_title("In-space placebos")
    a.legend(loc="upper left", fontsize=8.5)
    month_axis(a)
    a.tick_params(axis="x", labelsize=8.5)

    # standardised mean gap in ban weeks 2-4, per unit
    gg = g.set_index("date")
    pre = gg.loc[:BAN - pd.Timedelta(days=1)]
    w24 = gg.loc[BAN + pd.Timedelta(days=7):LIFT - pd.Timedelta(days=1)]
    z = (w24.mean() / np.sqrt((pre ** 2).mean())).sort_values(ascending=False)
    colors = [ps.BLUE if u == "it" else GREY for u in z.index]
    y = np.arange(len(z))
    b.hlines(y, 0, z.values, color=colors, lw=2)
    b.scatter(z.values, y, color=colors, s=18, zorder=3)
    b.axvline(0, color=ps.INK_2, lw=0.8)
    b.set_yticks(y)
    b.set_yticklabels(z.index, fontsize=8)
    b.set_xlabel("Mean gap, ban weeks 2-4 (pre-RMSPE units)")
    b.set_title("Italian ranks last of %d: p = %.3f" % (len(z), m["ban_weeks2_4_p_mean_gap_two_sided"]), fontsize=11)
    b.grid(axis="y", visible=False)
    ps.save(fig, "italy_chatgpt_placebos.png")


def fig_italy_vpn():
    s = pd.read_csv(R / "italy_vpn_series.csv", parse_dates=["date"])
    g = pd.read_csv(R / "italy_vpn_placebo_gaps.csv", parse_dates=["date"])
    m = load_json("italy_vpn_main.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9))
    roll = lambda x: x.rolling(7, min_periods=1).mean()   # trailing, so nothing leaks backwards
    a.plot(s.date, roll(s.observed_views), color=ps.BLUE, lw=1.6, label="Italian Wikipedia (observed)")
    a.plot(s.date, roll(s.synthetic_views), color=ps.ORANGE, lw=1.6, label="synthetic")
    ban_shade(a)
    a.set_ylabel("Daily views, trailing 7-day mean")
    a.set_title("'Rete privata virtuale' (VPN)", fontsize=11)
    a.legend(loc="upper left", fontsize=9)
    month_axis(a)
    a.tick_params(axis="x", labelsize=8.5)
    for c in g.columns[2:]:
        b.plot(g.date, roll(g[c]), color=GREY, lw=0.9)
    b.plot([], [], color=GREY, label="%d donor languages" % (len(g.columns) - 2))
    b.plot(g.date, roll(g["it"]), color=ps.BLUE, lw=1.6, label="Italian")
    b.axhline(0, color=ps.INK_2, lw=0.8)
    ban_shade(b, label=False)
    b.set_ylabel("Gap in log views, trailing 7-day mean")
    b.set_title("Ban %+.0f%%, after the lift %+.0f%%; p = %.3f (lowest possible)" % (
        m["ban_pct"], m["lift_pct"], m["ban_p_rmspe_ratio"]), fontsize=10.5)
    b.legend(loc="upper left", fontsize=9)
    month_axis(b)
    b.tick_params(axis="x", labelsize=8.5)
    ps.save(fig, "italy_vpn.png")


def fig_italy_in_time():
    t = pd.read_csv(R / "italy_chatgpt_in_time_placebo.csv", parse_dates=["date"])
    m = load_json("italy_chatgpt_main.json")["in_time_placebo"]
    fake = pd.Timestamp(m["fake_date"])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(t.date, np.exp(t.observed_log), color=ps.BLUE, lw=1.5, label="Italian Wikipedia (observed)")
    ax.plot(t.date, np.exp(t.synthetic_log), color=ps.ORANGE, lw=1.5, label="synthetic, fitted only up to the fake date")
    ax.axvline(fake, color=ps.INK_2, ls="--", lw=1)
    ax.text(fake, 1.0, " fake date: 3 Mar 2023", transform=ax.get_xaxis_transform(), va="top",
            fontsize=9, color=ps.INK_2)
    ax.axvline(pd.Timestamp("2023-03-14"), color=GREY, lw=1)
    ax.text(pd.Timestamp("2023-03-14"), 0.08, " GPT-4 released", transform=ax.get_xaxis_transform(),
            fontsize=8.5, color=ps.INK_2)
    ax.set_yscale("log")
    ax.set_yticks([2000, 3000, 5000, 10000])
    ax.set_yticklabels(["2k", "3k", "5k", "10k"])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("Daily views (log scale)")
    ax.set_title("In-time placebo: fake effect %+.1f%%, p = %.2f" % (m["pct"], m["p_rmspe_ratio"]))
    ax.legend(loc="upper left", fontsize=9)
    month_axis(ax)
    ps.save(fig, "italy_in_time_placebo.png")


def fig_italy_sensitivity():
    s = pd.read_csv(R / "italy_chatgpt_sensitivity.csv").iloc[::-1].reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.4), sharey=True)
    y = np.arange(len(s))
    for ax, col, title in [(axes[0], "ban_week1_pct", "Ban, week 1"),
                           (axes[1], "ban_weeks2_4_pct", "Ban, weeks 2-4"),
                           (axes[2], "lift_pct", "28 days after the lift")]:
        main = s[col][s.label == "main"].iloc[0]
        colors = [ps.BLUE if l == "main" else ps.INK_2 for l in s.label]
        ax.axvline(main, color=ps.BLUE, lw=0.8, alpha=0.5)
        ax.axvline(0, color=ps.INK_2, lw=0.8)
        ax.scatter(s[col], y, color=colors, s=34, zorder=3)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Effect on total views (%)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(s.label, fontsize=8.5)
    ps.save(fig, "italy_sensitivity.png")


def fig_sim_bias():
    s = pd.read_csv(R / "simulation_summary.csv")
    sw = pd.read_csv(R / "simulation_sweeps.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ins = s[s.design == "inside"]
    x = np.arange(len(ins))
    ax = axes[0]
    ax.bar(x - 0.2, ins.sc_bias, 0.38, color=ps.BLUE, label="synthetic control")
    ax.bar(x + 0.2, ins.did_bias, 0.38, color=ps.ORANGE, label="difference-in-differences")
    ax.errorbar(x - 0.2, ins.sc_bias, 1.96 * ins.sc_bias_se, fmt="none", ecolor=ps.INK, lw=1)
    ax.errorbar(x + 0.2, ins.did_bias, 1.96 * ins.did_bias_se, fmt="none", ecolor=ps.INK, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["%.1f" % t for t in ins.tau])
    ax.set_xlabel("True effect τ (in units of noise s.d.)")
    ax.set_ylabel("Bias of the average effect")
    ax.set_title("Bias does not depend on τ")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.5)
    for ax, kind, xlabel, title in [(axes[1], "sigma", "Noise s.d. σ", "SC bias vanishes as noise → 0"),
                                     (axes[2], "T0", "Pre-period length T0", "SC bias shrinks with longer T0")]:
        d = sw[sw.kind == kind]
        ax.errorbar(d.value, d.sc_bias, 1.96 * d.sc_bias_se, color=ps.BLUE, marker="o", lw=2, capsize=2,
                    label="synthetic control")
        ax.plot(d.value, d.did_bias, color=ps.ORANGE, marker="s", lw=2, label="DiD")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        if kind == "sigma":
            ax.axhline(0, color=ps.INK_2, lw=0.8)
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks([10, 20, 40, 80, 160, 320])
    axes[2].set_xticklabels(["10", "20", "40", "80", "160", "320"])
    axes[2].set_yscale("log")
    axes[2].set_yticks([0.1, 0.2, 0.5, 1, 2, 5])
    axes[2].set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
    axes[2].set_ylim(0.08, 10)
    axes[1].set_ylim(-0.05, 1.45)
    axes[1].legend(loc="lower right", fontsize=9)
    ps.save(fig, "sim_bias.png")


def fig_sim_calibration():
    reps = pd.read_csv(R / "simulation_reps.csv")
    s = pd.read_csv(R / "simulation_summary.csv")
    meta = load_json("simulation_summary.json")
    J = meta["config"]["J"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    counts = np.array(meta["null_rank_counts"])
    ax.bar(np.arange(1, J + 2), counts / counts.sum(), color=ps.BLUE, width=0.8)
    ax.axhline(1 / (J + 1), color=ps.ORANGE, lw=1.5, ls="--", label="uniform (exact calibration)")
    ax.set_xlabel("Rank of the treated unit's RMSPE ratio (1 = largest)")
    ax.set_ylabel("Share of 1,000 null panels")
    ax.set_title("Under τ = 0: rejection rate at 5%% is %.1f%% (nominal %.1f%%)" % (
        100 * meta["null_reject_05"], 100 * meta["nominal_reject_05"]), fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax = axes[1]
    ins = s[s.design == "inside"]
    ax.plot(ins.tau, ins.reject_05, color=ps.BLUE, marker="o", label="inside convex hull")
    out = s[s.design == "outside"]
    ax.plot(out.tau, out.reject_05, color=ps.ORANGE, marker="s", ls="none", ms=8,
            label="outside convex hull")
    ax.axhline(1 / (J + 1), color=ps.INK_2, lw=0.8, ls="--")
    ax.set_xlabel("True effect τ")
    ax.set_ylabel("Share with p ≤ 0.05")
    ax.set_ylim(0, 1.05)
    ax.set_title("Power, and false positives when the hull fails", fontsize=11)
    ax.annotate("%.0f%% false positives" % (100 * out.reject_05.iloc[0]), xy=(0, out.reject_05.iloc[0]),
                xytext=(0.4, 0.45), fontsize=9, arrowprops=dict(arrowstyle="->", color=ps.INK_2, lw=0.8))
    ax.legend(loc="lower right", fontsize=9)
    ps.save(fig, "sim_calibration.png")


def fig_convex_hull():
    e = pd.read_csv(R / "simulation_examples.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, design, title in [(axes[0], "inside", "Treated unit inside the donors' convex hull"),
                              (axes[1], "outside", "Treated unit outside the hull (trend too steep)")]:
        d = e[e.design == design]
        T0 = d.T0.iloc[0]
        ax.fill_between(d.t, d.donor_min, d.donor_max, color=GREY, alpha=0.35, lw=0, label="range of 20 donors")
        ax.plot(d.t, d.untreated, color=ps.INK_2, lw=1.2, ls=":", label="true Y(0)")
        ax.plot(d.t, d.observed, color=ps.BLUE, lw=1.8, label="treated (τ = 2 after T0)")
        ax.plot(d.t, d.synthetic, color=ps.ORANGE, lw=1.8, label="synthetic control")
        ax.plot(d.t, d.did, color=ps.VIOLET, lw=1.4, ls="--", label="DiD counterfactual")
        ax.axvline(T0 + 0.5, color=ps.INK_2, lw=1, ls="--")
        ax.set_title("%s\npre-period RMSPE = %.2f" % (title, d.pre_rmspe.iloc[0]), fontsize=11)
        ax.set_xlabel("Period")
    axes[0].set_ylabel("Outcome")
    axes[0].legend(loc="upper left", fontsize=8.5)
    ps.save(fig, "sim_convex_hull.png")


# Solvers and Prop 99 ------------------------------------------------------------

def fig_solvers():
    c = pd.read_csv(R / "pgd_convergence.csv")
    a = pd.read_csv(R / "solver_agreement.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    ax.loglog(c.k, c.bound, color=ps.INK_2, ls="--", lw=1.2, label="bound L‖w₀ − w*‖² / 2k")
    ax.loglog(c.k, np.maximum(c.pgd, 1e-16), color=ps.BLUE, label="projected gradient, step 1/L")
    ax.loglog(c.k, np.maximum(c.fista, 1e-16), color=ps.ORANGE, label="same, with momentum (FISTA)")
    ax.set_ylim(1e-13, None)
    ax.set_xlabel("Iteration k")
    ax.set_ylabel("f(w_k) − f(w*)")
    ax.set_title("Convergence on one simulated problem (J = 20, T0 = 40)", fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
    ax = axes[1]
    v = np.log10(np.maximum(a.max_w_diff, 1e-12))
    ax.hist(v, bins=30, color=ps.BLUE)
    ax.set_xlabel("log10 max |w_SciPy − w_PGD| over donors (−12 = identical)")
    ax.set_ylabel("Problems")
    ax.set_title("SciPy SLSQP vs my PGD on 300 problems", fontsize=11)
    ps.save(fig, "solver_agreement.png")


def fig_prop99():
    p = pd.read_csv(R / "prop99.csv")
    g = pd.read_csv(R / "prop99_placebo_gaps.csv")
    m = load_json("prop99.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    ax.plot(p.year, p.california, color=ps.BLUE, lw=2, label="California")
    ax.plot(p.year, p.synthetic_mine, color=ps.ORANGE, lw=2, label="synthetic (mine, outcomes only)")
    ax.plot(p.year, p.synthetic_pysyncon_adh, color=ps.VIOLET, lw=1.4, ls="--",
            label="synthetic (pysyncon, ADH covariates)")
    ax.plot(p.year, p.donor_mean, color=GREY, lw=1.4, label="average of 38 donor states")
    ax.axvline(1988.5, color=ps.INK_2, ls="--", lw=1)
    ax.set_ylabel("Cigarette packs per capita")
    ax.set_title("Proposition 99: gap in 2000 = %.1f packs (ADH: about -26)" % m["mine"]["gap_2000"], fontsize=11)
    ax.legend(loc="lower left", fontsize=8.5)
    ax = axes[1]
    for c in g.columns[2:]:
        ax.plot(g.year, g[c], color=GREY, lw=0.8)
    ax.plot(g.year, g["California"], color=ps.BLUE, lw=2, label="California")
    ax.plot([], [], color=GREY, label="38 placebo states")
    ax.axhline(0, color=ps.INK_2, lw=0.8)
    ax.axvline(1988.5, color=ps.INK_2, ls="--", lw=1)
    ax.set_ylim(-40, 40)
    ax.set_ylabel("Gap (packs per capita)")
    ax.set_title("Placebos: California ranks 3 of 39 (p = %.3f)" % m["mine"]["p_value"], fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
    ps.save(fig, "prop99.png")


def main():
    ps.apply()
    fig_italy_hero()
    fig_italy_placebos()
    fig_italy_vpn()
    fig_italy_in_time()
    fig_italy_sensitivity()
    fig_sim_bias()
    fig_sim_calibration()
    fig_convex_hull()
    fig_solvers()
    fig_prop99()


if __name__ == "__main__":
    main()
