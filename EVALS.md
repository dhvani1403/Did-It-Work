# Evaluation notes

How I tested whether this synthetic control implementation is correct, and whether its answer on real data can be trusted. Every number here comes from a run saved in [`results/`](results/).

## 1. What I measure and why

A synthetic control project can go wrong in three separate places, so I measure each one on its own.

1. **The optimiser.** Are the weights the true minimiser of the constrained least-squares problem? I measure the largest weight difference between two independent solvers (SciPy SLSQP and my projected gradient descent), the relative difference in objective, and the Frank–Wolfe gap, which bounds how far a solution is from optimal and is zero exactly when the KKT conditions hold.
2. **The estimator.** When the true effect is known, how far off is the estimate? On simulated panels I measure the **bias** and **RMSE** of the average post-period effect, against true effects of 0, 0.5, 1, 2 and 4 noise standard deviations, and compare with difference-in-differences on the same panels.
3. **The inference.** Does the placebo test reject as often as it says it will? I measure the **false-positive rate** (share of panels with p ≤ 0.05 when the true effect is zero) against the nominal 1/(J+1), the full distribution of the treated unit's rank, and **power** for non-zero effects.

On the real data (Italy's ChatGPT ban) there is no ground truth, so I measure what can be measured: pre-period RMSPE (fit), placebo p-values for each post-treatment window, an in-time placebo (an effect at a fake date, which should be zero), whether the effect stops when the ban is lifted, and how far the estimate moves across leave-one-out donors, pre-period windows, frequency, treatment date and donor pools.

## 2. Test set (how built, limits)

**Simulated panels** ([`dgp.py`](src/synthcontrol/dgp.py), [`simulate.py`](scripts/simulate.py)). A linear factor model with 20 donors, 40 pre-periods and 10 post-periods: one linear-trend factor, two AR(1) factors, unit fixed effects, Gaussian noise with s.d. 1. The treated unit is a convex combination of the 4 donors with the steepest trends ("inside" the hull), or has its trend loading pushed beyond every donor's ("outside"). 4,000 panels with full placebo runs, plus 4,400 for sweeps over the noise level and the pre-period length. Every panel has its own seed, so the study is reproducible.

Limits: it is one family of data-generating processes. The noise is Gaussian and independent across units, the effect is constant over time, and the treated unit is placed near the edge of the donors on purpose (the case where synthetic control should beat DiD). Real page-view series have heavier tails, weekly cycles, holidays and news spikes that this does not reproduce.

**Optimisation problems.** 300 simulated problems with J from 5 to 40 and T0 from 10 to 80, 58 of them rank-deficient, plus the Italian and Proposition 99 panels.

**Proposition 99** (39 states, 1970 to 2000). The one case with a published answer to compare against: donor weights and a gap of about −26 packs per capita in 2000 (Abadie, Diamond and Hainmueller 2010).

**Real data: Italy's ChatGPT ban.** Daily views of the ChatGPT article and of the VPN article in Italian and 26 other language editions, 1 December 2022 to 31 July 2023, with views summed over every redirect to each article ([`fetch_italy.py`](scripts/fetch_italy.py)). Saved in [`data/processed/`](data/processed/). Limits: no ground truth, one event, a pre-period of only 89 days because ChatGPT launched at the end of November 2022, and Wikipedia views measure attention, not usage.

## 3. Results log

One row per change I actually made, in the order I made them.

| Version | What changed | Metric | Result | Takeaway |
|---|---|---|---|---|
| v1 | Baseline: simplex projection, SLSQP and PGD solvers, placebos, DiD, simulator, 94 tests | Tests passing | 94 of 94 | Core maths checks out on small known cases. |
| v2 | First simulation study (inside and outside designs, τ = 0 to 4) | Bias / RMSE, inside, τ = 0 | SC 0.541 / 0.724; DiD 1.145 / 1.231 | SC beats DiD, but is clearly biased. I did not expect a bias of half a noise s.d. |
| v2 | (same run) | False positives at 5%, inside | 6.0% (s.e. 0.75%; nominal 4.8%) | Close to calibrated, slightly liberal. |
| v2 | (same run) | False positives at 5%, outside hull | 30.6% | The test is not safe when the hull assumption fails. |
| v3 | Added sweeps over noise σ and pre-period length T0 | SC bias at σ = 0 / 0.25 / 1 / 2 | 7 × 10⁻⁹ / 0.127 / 0.568 / 0.924 | The bias is the noise term in the ADH argument, not a coding error. |
| v3 | (same run) | SC bias at T0 = 20 / 80 / 320 | 0.543 / 0.408 / 0.144 | Longer pre-periods shrink it, as theory says. DiD bias grows instead (0.669 / 2.054 / 7.458). |
| v4 | Plain PGD on all problems vs SciPy | Median iterations; largest weight difference | 1,740 iterations; 3.1 × 10⁻⁶ | Correct, but about 16 times slower than SLSQP in median time. Too slow for placebo-heavy runs. |
| v5 | Added momentum (FISTA) for placebos and simulations | Median iterations | 220 | About 8 times fewer iterations, same weights. |
| v6 | Proposition 99 with plain PGD, 200,000 steps | Frank–Wolfe gap; largest weight error | 2.9; 0.021 | Did not converge (rank 19 with 38 donors). |
| v6 | Proposition 99 with FISTA | Iterations; difference from SciPy | 9,060; 1.9 × 10⁻⁵ | Converged. Gap in 2000 = −26.6 packs (ADH: about −26). Placebo p = 0.077, not ADH's 0.026. |
| v7 | pysyncon on the same objective | Largest weight difference: simulated / Prop 99 | 8.7 × 10⁻⁵ / 8.6 × 10⁻⁶ | An independent implementation gives the same weights. |
| v8 | Italy data: downloaded 27 language editions with all redirects; donor screen | Donors kept: ChatGPT / VPN | 20 of 26 / 7 of 26 | Ukrainian dropped for an 11-fold one-day burst; five languages had no views in early January. Most VPN articles are too small. |
| v9 | First Italy fit, with the two 28-day windows fixed in advance (ban, lift) | Ban effect; p (RMSPE ratio); p (average gap) | +12.5%; 0.048; 0.524 | Italy ranks first on the ratio, but the average is not unusual. |
| v9 | (same run) | Ban effect across leave-one-out | +0.8% to +18.7% | The 28-day total is unstable: dropping Hebrew alone takes it to +0.8%. |
| v9 | (same run) | Lift effect; p (average gap) | +25.7%; 0.619 | No deficit after access returns. |
| v9 | (same run) | In-time placebo at 3 Mar | −3.1%, p = 0.67 | No drift, even across the GPT-4 release. Unlike the ban window, the fit itself looks trustworthy. |
| v10 | Split the ban window into week 1 and weeks 2–4 (after seeing v9's weekly path) | Week 1; p (RMSPE ratio) | +130%; 0.048 | The news spike. Italy ranks first in all 14 specifications. |
| v10 | (same run) | Weeks 2–4; p (average gap) | −32.5%; 0.048 | The deficit while access was blocked. The v9 total was adding these two together. |
| v10 | (same run) | Weeks 2–4 across 14 specifications | −24.6% to −36.8% | Stable sign and size. Italy ranks first in 8 and in the top four in all 14. |
| v10 | Added checks: ban dated 30 Mar; weeks 2–4 without Italian holidays | Weeks 2–4 effect | −29.6%; −31.1% | Neither the date nor Easter Monday and Liberation Day explain the deficit. |
| v10 | Dropped the donors whose regulators reacted (es, de, fr) | Weeks 2–4; p | −24.6%; 0.222 | Smaller and no longer significant: some of the deficit may be spillover. |
| v10 | Second outcome: Italian VPN article | Ban; weeks 2–4; lift; p | +66%; +41%; +5%; 0.125 | Same timing as the ChatGPT deficit, but 0.125 is the floor with 7 donors. |
| v11 | pysyncon on the Italian ChatGPT panel | Largest weight difference | 2.3 × 10⁻⁸ | Same weights on the real data too. |

## 4. What did not work

* **The 28-day ban window as the headline.** I fixed it before looking at the data. It added a +130% news week to three weeks of −24% to −41%, and gave +12.5%, which describes neither. Its leave-one-out range was +0.8% to +18.7%. I kept it in the README as the pre-specified result, and report the split (chosen afterwards, and labelled as such) next to it.
* **Weekly aggregation.** With 12 pre-period weeks and 20 donors, many placebo units fit almost perfectly before the ban. The weeks 2–4 effect was unchanged (−33%) but its p-value rose from 0.048 to 0.14. I kept daily data.
* **A longer pre-period.** Starting on 15 December 2022 instead of 1 January loses 6 donors whose articles did not exist yet (14 left). The estimate barely moved (−34%), so I kept 1 January.
* **Synthetic control without an intercept for the VPN article.** The level of the Italian VPN series is hard to match with a convex combination: the pre-period RMSPE was 0.267 against 0.166 with the intercept, and it produced a −32% "effect" after the lift. The demeaned version is the main specification for both outcomes.
* **The 100-views rule for the VPN article.** It leaves only 7 donors, so the placebo test cannot go below 0.125. Including every language with complete data (24 donors) gives p = 0.04 but uses very noisy series. I report both and do not claim significance.
* **Plain projected gradient descent on rank-deficient problems.** With 19 pre-period years and 38 donor states, the objective is not strongly convex, and plain PGD only has the $O(1/k)$ guarantee. It had not converged after 200,000 steps. I kept the plain method as the reference implementation (it is what the convergence proof covers) and used the accelerated version where $T_0 < J$ and for large placebo runs.
* **The first download crashed.** A read timeout from the Pageviews API stopped it at the 15th language. I made timeouts and connection errors retry with back-off, like rate limits, and resumed from the cache.

## 5. Error examples

**1. Treated unit outside the convex hull (simulation).** When the treated unit's trend loading is 0.6 above the steepest donor's, no weighted average can follow it. With no true effect, the average estimated effect is 2.89 noise s.d. and the placebo test rejects at 5% in 30.6% of panels. In the example panel ([figure](figures/sim_convex_hull.png)) the synthetic control lags the treated unit through the pre-period (pre-RMSPE 1.86, against 1.20 for the inside example) and the gap keeps growing after $T_0$. The pre-period fit gives the warning: the treated unit had the worst pre-period fit of all 21 units in 59.6% of outside panels and 0.3% of inside panels.

**2. Opposite effects cancelling inside one window (real data).** The pre-specified ban window put a news spike and an access deficit into one number. The result, +12.5%, had a placebo p-value of 0.048 on the RMSPE ratio (which reacts to any large deviation) but 0.524 on the average gap (which lets the two phases cancel). It moved from +0.8% to +18.7% when single donors were dropped. Nothing was wrong with the fit: the in-time placebo was clean. The estimand was wrong for an event whose effect changes sign.

**3. A donor that may have been treated too (real data).** In weeks 2–4 of the ban, Spanish Wikipedia has the largest *positive* standardised gap of all 21 units (+1.74 pre-period RMSPEs), and Spain's data-protection agency opened its own ChatGPT investigation on 13 April, inside that window. Spanish has weight 0.262 in the synthetic Italy, so if Spanish interest rose because of the ban, the synthetic Italy rose with it and the Italian deficit is overstated. Dropping Spanish alone gives −26.7% (p = 0.15); dropping Spanish, German and French gives −24.6% (p = 0.22), against −32.5% with them. I cannot tell this apart from noise with 17 donors, so I report −25% as the conservative size.

## 6. Next steps

* Replace single p-values with conformal confidence intervals for each week (Chernozhukov, Wüthrich and Zhu 2021), so the news week, the deficit and the recovery each get an interval.
* Use Wikimedia's per-country page-view data to separate readers in Italy from readers in Switzerland, where ChatGPT stayed available.
* Fit a BSTS / CausalImpact model on the same panel and compare the weekly estimates.
* Extend the simulator with heavy-tailed noise, weekly cycles and effects that change sign, and check calibration of the split-window tests.
* Apply the same pipeline to other temporary national blocks of online services, to see whether a drop of a quarter to a third is typical.
