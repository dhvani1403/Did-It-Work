"""Did Italy's ChatGPT ban change how often Italians looked ChatGPT up on Wikipedia?

Treated unit: daily views of the ChatGPT article on Italian Wikipedia (summed
over all titles that redirect to it). Donors: the same article in other
language editions. Outcome: log daily views. Synthetic control with a free
intercept (every series demeaned over the pre-period), weights from my
projected gradient solver.

Two post-treatment windows, both 28 days long:
  ban   31 Mar - 27 Apr 2023  (ChatGPT blocked in Italy)
  lift  28 Apr - 25 May 2023  (access restored)
The weights are fitted on 1 Jan - 30 Mar 2023 only and then held fixed, so the
lift window is a second, out-of-sample test of the same synthetic control.

Reads data/processed/italy_*_daily.csv only (runs offline). Writes results/italy_*.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import italy_config as cfg  # noqa: E402
from synthcontrol import inference, synth  # noqa: E402
from synthcontrol.did import did  # noqa: E402

RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"
TREATED = cfg.TREATED_LANG
SOLVER = "pgd"          # main fit: plain projected gradient, step 1/L
FAST_SOLVER = "fista"   # placebos and sensitivity: same solver with momentum
WINDOW_DAYS = 28
# Languages whose own data-protection regulators publicly reacted to the
# Italian decision within the ban window (Spain's AEPD opened an inquiry and
# asked the EDPB to act, Germany's and France's regulators said they were
# looking at ChatGPT). Their readers saw related domestic news, so they are
# the donors most at risk of spillover.
REACTING_LANGS = ["es", "de", "fr"]


# Data -------------------------------------------------------------------------

def load(outcome):
    return pd.read_csv(PROCESSED / ("italy_%s_daily.csv" % outcome), index_col=0, parse_dates=True)


def screen(df, donors, start=cfg.PRE_START, end=cfg.POST_END, min_views=100):
    """Data-quality screen on donors only (Italy's series is never looked at here).

    Drop a donor if, between ``start`` and ``end``,
      * it has any day with zero views (article missing or not yet created), or
      * its mean daily views are below ``min_views`` (too noisy in logs), or
      * any single day is more than 5 times the median of the 15 days around it
        (one-day bursts of automated traffic labelled as human).
    """
    w = df.loc[start:end]
    keep, rows = [], []
    for lang in donors:
        if lang not in w:
            rows.append(dict(lang=lang, dropped=True, reason="no article"))
            continue
        s = w[lang]
        med15 = s.rolling(15, center=True, min_periods=8).median()
        burst = float((s / med15).max())
        reason = ""
        if (s <= 0).any():
            reason = "days with zero views"
        elif s.mean() < min_views:
            reason = "fewer than %d views a day" % min_views
        elif burst > 5:
            reason = "one day reaches %.1fx its 15-day median" % burst
        rows.append(dict(lang=lang, mean_daily=float(s.mean()), max_day_over_15d_median=burst,
                         dropped=bool(reason), reason=reason))
        if not reason:
            keep.append(lang)
    return keep, pd.DataFrame(rows)


def make_panel(df, donors, pre_start=cfg.PRE_START, end=cfg.POST_END, freq="D", ban_start=cfg.BAN_START):
    """(Y, dates, T0) with Italy in column 0; log views."""
    w = df.loc[pre_start:end, [TREATED] + list(donors)]
    if freq == "W":
        # weeks run Friday to Thursday, so the ban (a Friday) starts a week
        start = pre_start + pd.Timedelta(days=(ban_start - pre_start).days % 7)
        w = w.loc[start:].resample("W-THU").sum()
    T0 = int((w.index < ban_start).sum())
    return np.log(w.values), w.index, T0


def windows(dates, T0, freq="D"):
    """Row slices for the post-treatment windows.

    ban and lift are the two 28-day windows I fixed before fitting anything.
    After the first run showed a news spike followed by a fall, I also split
    the ban window into its first week (news) and the remaining three weeks.
    """
    lift_row = int((dates < cfg.LIFT).sum())
    week = 1 if freq == "W" else 7
    news_end = int((dates < cfg.BAN_START + pd.Timedelta(days=7)).sum())
    return {"ban": slice(T0, lift_row), "ban_week1": slice(T0, news_end),
            "ban_weeks2_4": slice(news_end, lift_row), "lift": slice(lift_row, len(dates))}


# Estimation -------------------------------------------------------------------

def window_stats(r, sl):
    o, s = np.exp(r.observed[sl]), np.exp(r.synthetic[sl])
    return dict(pct=100 * (o.sum() / s.sum() - 1), extra_views=float((o - s).sum()),
                observed=float(o.sum()), synthetic=float(s.sum()),
                mean_log_gap=float(np.mean(r.gap[sl])))


def window_p_values(pl, T0, sl):
    """RMSPE-ratio and mean-gap permutation p-values for one post window."""
    pre = np.sqrt(np.mean(pl.gaps[:T0] ** 2, axis=0))
    post = np.sqrt(np.mean(pl.gaps[sl] ** 2, axis=0))
    ratio = post / pre
    z = np.mean(pl.gaps[sl], axis=0) / pre
    return dict(rmspe_ratio=float(ratio[0]),
                p_rmspe_ratio=inference.permutation_p_value(ratio[0], ratio[1:]),
                p_mean_gap_two_sided=inference.permutation_p_value(abs(z[0]), np.abs(z[1:])),
                p_mean_gap_lower=inference.permutation_p_value(-z[0], -z[1:]))


def estimate(df, donors, label, pre_start=cfg.PRE_START, freq="D", demean=True,
             placebos=True, solver=FAST_SOLVER, ban_start=cfg.BAN_START):
    Y, dates, T0 = make_panel(df, donors, pre_start, cfg.POST_END, freq, ban_start)
    r = synth.fit(Y, 0, T0, demean=demean, solver=solver)
    win = windows(dates, T0, freq)
    out = dict(label=label, freq=freq, demean=demean, J=len(donors), T0=T0,
               pre_start=str(dates[0].date()), pre_rmspe=r.pre_rmspe,
               last28_pre_mean_gap=float(np.mean(r.gap[max(0, T0 - (4 if freq == "W" else 28)):T0])),
               weights={n: round(w, 4) for n, w in r.weight_table(names=[TREATED] + list(donors))})
    for name, sl in win.items():
        for k, v in window_stats(r, sl).items():
            out["%s_%s" % (name, k)] = v
    pl = None
    if placebos:
        pl = inference.in_space_placebos(Y, 0, T0, demean=demean, solver=FAST_SOLVER)
        for name, sl in win.items():
            for k, v in window_p_values(pl, T0, sl).items():
                out["%s_%s" % (name, k)] = v
    return out, r, pl, dates


def fmt_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append("%.0f" % v if abs(v) >= 1000 else "%.3f" % v)
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run_outcome(outcome, full=True):
    df = load(outcome)
    candidates = [l for l in cfg.DONOR_LANGS if l in df.columns]
    donors, scr = screen(df, candidates)
    scr.to_csv(RESULTS / ("italy_%s_screening.csv" % outcome), index=False)
    print("%s: %d of %d donor candidates kept" % (outcome, len(donors), len(cfg.DONOR_LANGS)))

    main, r, pl, dates = estimate(df, donors, "main", solver=SOLVER)
    T0 = r.T0
    Y, _, _ = make_panel(df, donors)
    r_sp = synth.fit(Y, 0, T0, demean=True, solver="scipy")
    main["solver_check"] = dict(max_abs_weight_diff=float(np.max(np.abs(r_sp.weights - r.weights))),
                                pgd_iterations=r.iterations, pgd_fw_gap=r.fw_gap,
                                scipy_fw_gap=r_sp.fw_gap, converged=r.converged)
    main["pre_mean_daily_views"] = float(np.exp(r.observed[:T0]).mean())
    win = windows(dates, T0)
    d = did(Y, 0, T0)
    main["did"] = dict(pre_rmspe=d.pre_rmspe)
    for name, sl in win.items():
        o, c = np.exp(Y[sl, 0]), np.exp(d.counterfactual[sl])
        main["did"]["%s_pct" % name] = 100 * (o.sum() / c.sum() - 1)
    # weekly breakdown across both windows
    wk = []
    for k in range((len(dates) - T0) // 7):
        sl = slice(T0 + 7 * k, T0 + 7 * (k + 1))
        st = window_stats(r, sl)
        wk.append(dict(week=k + 1, start=str(dates[sl.start].date()), **st))
    main["by_week"] = wk
    # weeks 2-4 of the ban without Italy's two public holidays in that window
    # (Easter Monday 10 Apr, Liberation Day 25 Apr), which could lower views on their own
    hol = [cfg.LIFT - pd.Timedelta(days=3), pd.Timestamp("2023-04-10")]
    sl = win["ban_weeks2_4"]
    mask = np.array([d not in hol for d in dates[sl]])
    o, s_ = np.exp(r.observed[sl][mask]), np.exp(r.synthetic[sl][mask])
    main["ban_weeks2_4_pct_without_holidays"] = 100 * (o.sum() / s_.sum() - 1)
    main["holidays_excluded"] = [str(d.date()) for d in hol]
    # first three days of the ban, and of the lift
    for name, sl in [("ban", win["ban"]), ("lift", win["lift"])]:
        s3 = slice(sl.start, sl.start + 3)
        main["%s_first3_pct" % name] = window_stats(r, s3)["pct"]
    names = [TREATED] + donors
    pd.DataFrame(dict(date=dates, observed_log=r.observed, synthetic_log=r.synthetic, gap=r.gap,
                      observed_views=np.exp(r.observed), synthetic_views=np.exp(r.synthetic),
                      did_log=d.counterfactual)).to_csv(RESULTS / ("italy_%s_series.csv" % outcome), index=False)
    g = pd.DataFrame(pl.gaps, columns=[names[u] for u in pl.units])
    g.insert(0, "date", dates)
    g.to_csv(RESULTS / ("italy_%s_placebo_gaps.csv" % outcome), index=False)
    stats = []
    for j, u in enumerate(pl.units):
        pre = np.sqrt(np.mean(pl.gaps[:T0, j] ** 2))
        row = dict(unit=names[u], pre_rmspe=pre)
        for name, sl in win.items():
            row["%s_ratio" % name] = np.sqrt(np.mean(pl.gaps[sl, j] ** 2)) / pre
            row["%s_mean_gap" % name] = float(np.mean(pl.gaps[sl, j]))
        stats.append(row)
    pd.DataFrame(stats).to_csv(RESULTS / ("italy_%s_placebo_stats.csv" % outcome), index=False)

    # In-time placebo: pretend the ban started 28 days earlier (3 Mar 2023).
    fake = cfg.BAN_START - pd.Timedelta(days=WINDOW_DAYS)
    T0f = int((dates < fake).sum())
    rt = inference.in_time_placebo(Y, 0, T0f, T0, demean=True, solver=SOLVER)
    plt_ = inference.in_space_placebos(Y[:T0], 0, T0f, demean=True, solver=FAST_SOLVER)
    st = window_stats(rt, slice(T0f, T0))
    main["in_time_placebo"] = dict(fake_date=str(fake.date()), T0=T0f, pre_rmspe=rt.pre_rmspe,
                                   pct=st["pct"], mean_log_gap=st["mean_log_gap"],
                                   **window_p_values(plt_, T0f, slice(T0f, T0)))
    pd.DataFrame(dict(date=dates[:T0], observed_log=rt.observed, synthetic_log=rt.synthetic,
                      gap=rt.gap)).to_csv(RESULTS / ("italy_%s_in_time_placebo.csv" % outcome), index=False)

    if full:
        sens = [dict(main, group="main")]
        for lang in list(main["weights"]):
            o_, *_ = estimate(df, [l for l in donors if l != lang], "drop %s" % lang)
            sens.append(dict(o_, group="leave-one-out"))
        for start in ["2022-12-15", "2023-02-01"]:
            ds, _ = screen(df, candidates, start=pd.Timestamp(start))
            o_, *_ = estimate(df, ds, "pre from %s" % start, pre_start=pd.Timestamp(start))
            sens.append(dict(o_, group="pre-period window"))
        o_, *_ = estimate(df, donors, "weekly (Fri-Thu weeks)", freq="W")
        sens.append(dict(o_, group="frequency"))
        o_, *_ = estimate(df, donors, "no intercept (plain SC)", demean=False)
        sens.append(dict(o_, group="method"))
        o_, *_ = estimate(df, donors, "ban dated 30 Mar (order date)", ban_start=pd.Timestamp("2023-03-30"))
        sens.append(dict(o_, group="treatment date"))
        o_, *_ = estimate(df, [l for l in donors if l not in REACTING_LANGS], "drop es, de, fr")
        sens.append(dict(o_, group="donor pool"))
        with_data = [l for l in candidates if (df.loc[cfg.PRE_START:cfg.POST_END, l] > 0).all()]
        o_, *_ = estimate(df, with_data, "every donor with complete data (no screen)")
        sens.append(dict(o_, group="donor pool"))
        sdf = pd.DataFrame(sens)
        sdf["weights"] = sdf.weights.apply(lambda w: "; ".join("%s %.2f" % kv for kv in list(w.items())[:4]))
        keep = ["group", "label", "J", "T0", "pre_rmspe", "ban_pct", "ban_p_rmspe_ratio",
                "ban_week1_pct", "ban_week1_p_rmspe_ratio", "ban_weeks2_4_pct",
                "ban_weeks2_4_p_mean_gap_two_sided", "lift_pct", "lift_p_mean_gap_two_sided",
                "last28_pre_mean_gap", "weights"]
        sdf[keep].to_csv(RESULTS / ("italy_%s_sensitivity.csv" % outcome), index=False)
        with open(RESULTS / ("italy_%s_sensitivity.md" % outcome), "w") as f:
            f.write(fmt_table(sdf[keep].to_dict("records"), keep))
        main["sensitivity_range"] = {k: [float(sdf[k].min()), float(sdf[k].max())]
                                     for k in ["ban_pct", "ban_week1_pct", "ban_weeks2_4_pct", "lift_pct",
                                               "ban_p_rmspe_ratio", "ban_week1_p_rmspe_ratio",
                                               "ban_weeks2_4_p_mean_gap_two_sided",
                                               "lift_p_mean_gap_two_sided", "pre_rmspe"]}
        print(sdf[keep[:-1]].round(3).to_string(index=False))

    with open(RESULTS / ("italy_%s_main.json" % outcome), "w") as f:
        json.dump(main, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in main.items() if k != "by_week"}, indent=1, default=float))
    print(pd.DataFrame(wk).round(3).to_string(index=False))
    return main


def main():
    RESULTS.mkdir(exist_ok=True)
    run_outcome("chatgpt")
    run_outcome("vpn")


if __name__ == "__main__":
    main()
