# WP7 review: what the hardware data support (2026-10-28)

Independent check of `hw_results.md` §1–§8 and the WP7 summary against
`hw_summary.json`, `hw_summary_2026-10-28.json`, `sim_per_instance.csv`,
`sim_models.csv`, `sim_pathology.csv`, `sim_sweep_fake/`, and the raw records in
`hw_2026-10-25/` and `hw_2026-10-28/`. Every number below was recomputed from
those files. Nothing here was run on the QPU.

---

## 1. Numbers that do not match

| # | Where | Written | Data | Fix |
|---|---|---|---|---|
| 1 | §2 Mitiq agreement | \|Δ\| = 0.009 (job 1, n = 2); 0.003 (job 2, n = 2) | 0.0096 → **0.010**; 0.0037 → **0.004** | rounding |
| 2 | §2 M = 0, n = 3 | "36× and 42×" | 0.748 / 0.0215 = **35×**; 0.875 / 0.0215 = **41×** | ratios were taken from the rounded 0.021 |
| 3 | §3 survival at α̂ | "within 0.06 of the measured signal" | 0.099 vs 0.160 = **0.061** | "within about 0.06" or "within 0.07" |
| 4 | §4 | "13.4σ below day 1" | mixes one day-1 copy with both day-2 copies. One copy each: **12.8σ** | write "more than 12σ" |
| 5 | §7 ¶1 | median recovery 0.90 at n = 3 (planned) | **0.886** | 0.89 |
| 6 | §7 ¶6 | n = 4 noisy sweep 0.149 at r = 3 | **0.148** | trivial |
| 7 | WP7 summary | job 1 "2026-10-25" | ran 2026-10-26 06:05–06:14 UTC; `_2026-10-25` is the plan date | paper: give the UTC run date; say folder labels are plan dates |
| 8 | WP7 summary | "a 235-2Q circuit resolved on the gentle grid but not on the odd grid" | the odd grid ran a **212**-2Q transpilation; the 235-2Q circuit never ran on the odd grid | "the same instance (212 2Q on the odd grid, 235 2Q on the gentle grid)" |
| 9 | WP7 summary, §1 | "prediction committed before the job" | commit `09b9c46` is 2026-10-28 01:32 UTC, after the job started (01:18). Only the file's internal stamp (01:04:36) is earlier | "timestamped 14 min before the job started"; push the next prediction before submitting (§7) |
| 10 | §1 proposed text | "13 unfolded circuits", "14 circuits" | 13 = 11 distinct circuits + 2 drift copies; 14 = 14 PUBs from **4** records | count circuits, not PUBs |

Everything else in §1–§8 that I checked matches (all of §1's table, §2 table and
recoveries, §3 table and σ values, §4 diagnostics, §6 rows, §7 ¶2–¶5, all of §8).

## 2. Claims stronger than the data

A. **§2 "Pointwise works only where the scalar fit is resolved."** False as
written. On the 11 hardware records with signal where the scalar fit is not
resolved, clipping *lowered* D_TV to the ideal on 5 and raised it on 6:
P-n4-M1-10_r2 0.812 → 0.137, P-n3-M2-17 (job 1) 0.647 → 0.065, P-n3-M1-17_r1
0.465 → 0.217, P-n4-M2-02 0.803 → 0.521, P-n4-M1-10 0.884 → 0.864; up: P-n3-M1-17
0.689 → 0.990, S-n4 0.814 → 0.986, P-n4-M1-10_r1 0.359 → 0.926, r3 0.184 → 0.668,
r4 0.519 → 0.939, P-n3-M2-17 (job 2) 0.754 → 0.998. §2 lists only the bad ones.
Write: "Outside the resolved fits, pointwise extrapolation is a coin toss: it
lowered D_TV on 5 of 11 records and raised it on 6, up to 0.99. Nothing in the
data says which will happen." That is a stronger argument for the gate, not a
weaker one.

B. **§7 ¶3 and §8: "the gate keeps unphysical values out" / "says when the
number can be trusted."** In simulation the strict rule itself reports 19 of 340
planned-grid values outside [0, 1] (mostly 1.00–1.06, but also 1.33, 1.52, 1.76
and **4.17**). The relaxed rule adds 7 more (one is 298). So the honest trade is:
strict costs 0.10 in median bias (−0.44 vs −0.34) and removes the worst
tail, not all unphysical values. Either cap reported values at 1 (and say so),
or write "the gate removes the catastrophic failures (298 in simulation, 376.6
and 2275 on hardware) but a resolved fit can still exceed 1 (19 of 186 in
simulation, up to 4.17)."

C. **§3: survival "correctly predicted that the gentler grid resolves n = 3."**
The uncorrected survival estimate (α = 1) predicts the same thing: 3 of 4 right,
the same miss (`identifiable_agreement` in `hw_summary_2026-10-28.json`). Job 2
does not separate α = 1 from α̂ for the survival rule. Drop this clause, or add
the record that would separate them (§7).

D. **§3: survival at α̂ "places the odd-grid limit exactly."** In-sample, and
weak: with these 11 records any 2Q threshold between 113 and 194 gives 10 of 11.
So the result is "consistent", not "exact". The pre-registered test (WP2 step 5)
is α = 1, and it failed (7 of 11; limit placed at 233–350 instead of 71–194).
That failure is the reportable C1→C3 result.

E. **§3: "S is a coherence-free upper bound."** Not a bound. (1−ε)^n is the
probability that no listed error fires; it ignores idling, crosstalk and leakage,
which is why it is optimistic. Call it an estimate.

F. **"Cube rule."** Same fact as α̂, not extra evidence: scaling every ε by α
turns S into ≈ S^α, so kept ≈ S^3 and α̂ ≈ 3.2 are one statement. Also the
kept > 0.13 cut is arbitrary: the next two records (0.125, 0.101) give exponents
2.64 and 1.93. Footnote at most.

F2. **§1: the deep records are "too close to zero to pin an exponent."** Not
all of them. P-n4-M1-10_r2 keeps 0.040 ± 0.005 (7σ above random) and gives an
exponent of **2.01 ± 0.08**; P-n4-M1-10 at r_opt gives **1.63 ± 0.11**
(P-n4-M2-02: 2.30). Up to ~230 2Q the exponent is 2.5–3.2; beyond that it falls
to ~1.6–2.3. So deep circuits keep *more* signal than one α predicts. The P(sol)
residuals at α̂ show the same thing (P-n4-M1-10_r1, _r2: +0.022, about 3σ each).
The α fit hides it because deep circuits sit near P_uniform and carry little
weight in P-space. Say it: "one α describes circuits up to ≈ 230 two-qubit
gates; deeper circuits retain more signal than it predicts." This matters for
the QPU job (§7).

G. **§1: "brings the whole distributions to within ~0.008 of the shot floor."**
Job 1 only (fit day). On job 2, even without P-n3-M2-17, D_TV(hw, model at α̂)
is 0.027 against a floor of 0.011 (2.4×). Say which job.

H. **Out-of-sample mean −0.012 (p = 0.42).** The mean hides errors of both signs
that cancel: per record at λ = 1, +0.011, +0.015, **+0.044** (P-n3-M1-17_r1, about
7σ), **−0.110** (P-n3-M2-17). Report per-record deviations or RMS (0.060 over
the 4 records; 0.028 without the failure), not the mean and a sign test.

I. **§6 null #9 (day 1 vs day 2 "no change").** Compares different transpiled
circuits (212 vs 235, 112 vs 126 2Q) on different layouts. Not a clean null;
drop it.

J. **"The fold grid, not circuit size, decides."** Job 2 also changed shots
(3×), day, layout and transpilation. The claim survives, but only with the
argument made explicit: on the odd grid every n ≥ 3 λ = 3 point is already at
random (P-n3-M1-17: 0.128 vs 0.125), so more shots cannot help; and in
simulation at equal shots the odd grid resolves 28 % of n = 3 instances, the
planned grid 98 %. Write "for these instances the grid decided".

K. **The gate has no goodness-of-fit check.** P-n4-M2-02 is "identified" on one
λ = 5 point at z = 2.8 with χ²_red = 8.4; in simulation the n = 3, r = 1 sweep
point is "resolved" with χ²_red = 11.4. With a one-sided 2σ rule on ~20 folded
n ≥ 3 points, about 0.5 false "informative" points are expected, so
P-n4-M2-02 is most likely that false positive. Say so in one sentence.

## 3. What carries the hardware story, in order

Main text:

1. **What the algorithm does on the device** (Table 4, Fig 5): n = 2 keeps 0.47
   of the ideal signal, n = 3 keeps 0.10–0.29, n = 4 at most 0.021 (detectable,
   useless for search). Sweep (Fig 9): fewer iterations win.
2. **Why: the device is its calibration model with every error × 3.18**, one
   factor for folded and unfolded circuits, tested on a second day: within 0.05
   on 3 of 4 records, one record at random. Includes the one-sentence P-n3-M2-17
   account; it is the out-of-sample failure, so it cannot go only to the supplement.
   One sentence: the a priori survival estimate (α = 1) placed the limit ~2× too far out.
3. **ZNE depends on the fold grid** (Fig 6/7 and a promoted Table S2 → Table 5):
   odd grid resolves only n = 2; gentle grid resolves all three M = 1 records,
   intervals contain the ideal. This is the main positive C2 result; it should
   not sit in the supplement.
4. **What the gate prevents** (Fig 7): Mitiq 3.547, raw fit 376.6 / 2275.
5. **Pointwise on distributions** (Fig 10): coin toss where unresolved; always
   harmful at M = 0 (simplex gives D_TV = 1 − 2⁻ⁿ, all mass on one outcome).

Supplement: drift tests (all four in the null list), survival bands at α = 1 and
α̂, α fit table with λ-subsets and leave-one-out, per-PUB model deviations,
Mitiq per record, rule v1 vs v2 (state that 0 hardware statuses changed),
P-n3-M2-17 diagnostics, the null-results table (minus #9). Drop the cube rule
or footnote it. Ten main figures is a lot for 8–10 pages; Fig 2 (worked
example) and Fig 6 (folding curves) are the candidates to move.

## 4. Statistics: what a referee at EPJ QT will attack, in order

1. **Scale and novelty.** n ≤ 4, one device, two days, and the hardware
   instances are the shallowest of each cell: [2,1]→3, [1,1,1]→3, [2,2,1,2]→1.
   "What do we learn beyond 'NISQ devices are noisy'?" Answer with α̂ +
   out-of-sample test, the grid result, and the gate. Say plainly the hardware
   subset is the easiest instance per cell, so these are best cases.
   (Useful defence: the n = 2 and n = 3 solutions are `11` and `111`, high
   Hamming weight, so relaxation bias works *against* them; the signal is not a
   readout artefact. For P-n4-M1-10, solution `0100`, the r = 4 copy on the same
   qubits sits at 0.063, which is the bias control.)
2. **Pseudo-replication.** Sign tests over PUBs treat λ = 1, 3, 5 of one
   circuit, and drift copies, as independent. Per distinct circuit: job 1 at
   α = 1, 11 of 11 below the model (p = 0.001); job 2, 4 of 4 (p = 0.125 alone);
   both, 15 of 15 (p = 6e−5). The effect is −0.18 against a shot error of
   ~0.007, so the test is not needed; report counts and magnitudes.
3. **What α means.** One factor on every error including readout (which is
   measured well) is phenomenological. Expect: "is it 2Q errors, crosstalk,
   leakage?" Two cheap offline checks would pre-empt this: (a) refit scaling only
   2Q errors; (b) compare 3.2 × median CZ error with the layered 2Q error the
   device reports for that day, if the API exposes it. Also report the per-circuit
   residual scatter (job 1: RMS 0.017 at α̂, χ²_red 5.1; structured by sweep
   variant: n = 3 off-optimal r below the model, n = 4 r = 1, 2 above; see §2
   F2), and the
   day-to-day change (P-n2-M1-11 0.600 → 0.646, z = 5.3, new layout).
4. **Extrapolation-model dependence on the gentle grid.** Extrapolating to λ = 0
   from λ ∈ [1, 1.5] is a lever arm of 2. The bootstrap CI does not include model
   choice: for P-n2-M1-11 job 2, a linear fit gives ≈ 0.86 vs the exponential
   0.98. Defence, which you already have: one α fits λ = 1 and λ = 3 alike (3.18,
   3.21; χ²_red 0.5 at λ = 3), i.e. the device's decay in λ is the S^λ
   exponential. State it.
5. **Fixed asymptote P_uniform.** Under relaxation the fully-noisy output is not
   uniform (job 2 P-n3-M2-17: D_TV to uniform 0.043, skewed to `000`, `001`).
   Defence: every n ≥ 3 λ = 5 point sits at P_uniform within |z| ≤ 2.3.
6. **Gate design.** Thresholds (2σ, two scales), no χ² check (§2 K), rule v2
   adopted after job 1 (0 of 75 hardware fit blocks changed: say it).
7. **Survival "prediction" with α̂** is circular (§2 C, D).
8. **"Mitiq baseline"** is the same model and asymptote; it checks the
   implementation and shows the value of the gate. Call it that, not a competing
   method.
9. **Drift.** The shot-noise null is the right test. State the detectable size
   (null 95th percentile D_TV 0.024 at n = 2, 4096 shots) and that it covers
   one ~9 min job only; between days there *is* change.

## 5. Draft: Hardware results subsection

> **Hardware results**
>
> We ran two jobs on ibm_fez (IBM Heron r2). The first, the job of record
> (2026-10-26 UTC; 47 circuits × 4096 shots; fold grid λ ∈ {1, 3, 5}), covers
> ten instances, the shallowest of each (n, M) cell plus the edge instance, and
> two iteration sweeps. The second, 43 h later, repeats three instances with
> n ≤ 3 on a gentler grid, λ ∈ {1, 1.25, 1.5}, with 12 288 shots (14 circuits).
> Each job was simulated beforehand on the calibration of the day and submitted
> once; together they used 133 s of QPU time, including a 7 s probe.
>
> *Unmitigated performance.* Figure 5 shows the signal kept,
> (P − P_u)/(P_ideal − P_u), against the number of two-qubit gates. At n = 2
> (71 gates) the device returns the solution with probability 0.600 [0.586,
> 0.614], against 0.25 for random guessing: it keeps 0.47 of the ideal signal.
> At n = 3 (112–350 gates) it keeps 0.10–0.29. At n = 4, at the optimal number
> of iterations (466–749 gates), it keeps at most 0.021: P-n4-M1-10 gives 0.077
> [0.069, 0.085] against 0.0625, and P-n4-M2-02 gives 0.142 [0.131, 0.153]
> against 0.125 (3.6σ and 3.1σ above random), while the signed instance is
> consistent with random (1.2σ). At n = 4 the signal is detectable but useless
> for search. The sweeps (Fig. 9) show that on the device fewer iterations win.
> At n = 3, r = 1, 2, 3 give 0.317, 0.257, 0.146 (ideal 0.781, 0.945, 0.330); at
> n = 4, r = 1–4 give 0.114, 0.096, 0.077, 0.063 (ideal 0.473, 0.908, 0.961,
> 0.582), reaching random at r = 4. Copies of two circuits placed at the start
> and end of each job agree within shot noise (all four tests p ≥ 0.36).
>
> *The device and its calibration model.* A noise model built from the
> calibration in force during each job overstates P(solution) on every unfolded
> circuit: all 11 distinct circuits of the first job (mean deviation −0.18) and
> all 4 of the second (−0.28). A single factor on every calibrated error removes
> this bias. Fitted to the 35 circuits with signal in the first job, folded ones
> included, it is α̂ = 3.18 [3.07, 3.35]. Fits to the unfolded or to the λ = 3
> circuits alone give 3.18 and 3.21, and leaving out any one record gives
> 3.13–3.21. One factor describes unfolded and folded circuits alike, as expected
> if the extra noise acts like a uniform rescaling of the calibrated errors. A
> gate-count survival estimate built from the unscaled calibration accordingly
> placed the hardware limit about twice too far out (233–350 two-qubit gates,
> against a measured 71–194). The fit is not exact within shot noise (χ²_red =
> 5.1 for the unfolded circuits; residuals up to 0.033), and circuits beyond
> about 230 two-qubit gates keep more signal than the single factor predicts.
>
> We used α̂ to predict the second job before it ran (prediction timestamped
> 14 min before the job started). On three of the four records the prediction
> held to within 0.05 on every circuit, and the device did slightly better than
> predicted (mean deviation +0.016 over 10 circuits). The fourth record,
> P-n3-M2-17, was at random on all of its circuits: 0.246 [0.238, 0.254] at
> λ = 1, against 0.25 for random guessing, a prediction of 0.355, and 0.353
> measured on the first day on a different layout. The failure lasted the whole
> job (copies at the start and end agree, p = 0.36), no qubit was stuck
> (single-qubit marginals 0.46–0.51), and a record sharing four of its qubits
> was unaffected. No calibration available to us flags it. Because the layout
> and the day changed together, we cannot tell a layout defect from a transient.
>
> *Zero-noise extrapolation.* We report the scalar extrapolation only when the
> fit is resolved, and otherwise the λ = 1 value, flagged (Sec. X). On the odd
> grid only n = 2 resolves: 0.953 [0.890, 1.026] (ideal 1). For every n ≥ 3
> circuit, folding to λ = 3 already brings P(solution) to random (P-n3-M1-17:
> 0.128 against 0.125), so one informative point remains and no fit is
> identifiable; more shots would not change this. On the gentle grid all three
> M = 1 records resolve, and every interval contains the ideal (Table 5): 0.981
> [0.928, 1.049] at n = 2, and 0.730 [0.633, 0.831] and 0.821 [0.660, 1.089] for
> the two n = 3 records (ideal 0.781 and 0.945). These close 0.95, 0.89 and 0.82
> of the gap between the unmitigated and the ideal value. For these instances
> the fold grid decided whether ZNE works: P-n3-M1-17 is not identifiable on
> the odd grid (212 gates) and resolves on the gentle grid (235 gates), as the
> simulation anticipated (at equal shots, 28 % of n = 3 instances resolve on the
> odd grid and 98 % on the gentler planned grid).
>
> Passing the same counts to Mitiq's exponential extrapolation, with the same
> asymptote and bootstrap, gives the same answer wherever our fit resolves
> (largest difference 0.027). Where ours does not resolve, Mitiq still returns a
> number: 3.547 for P-n3-M2-17, a probability above 1, and 0.831 for
> P-n4-M1-10_r2 from data only 0.034 above random, close to the ideal 0.908 by
> chance. Our own fit without the gate gives 376.6 and 2275 on P-n4-M2-02 and
> P-n3-M2-17. The gate, not the extrapolation model, keeps these values out.
>
> *Extrapolating whole distributions.* Extrapolating each bitstring probability
> separately helps where the scalar fit resolves (P-n2-M1-11: D_TV to the ideal
> 0.400 → 0.047). Elsewhere it is unpredictable: on the 11 records with signal
> where the scalar fit did not resolve, it lowered D_TV on 5 and raised it on 6,
> up to 0.99. On the infeasible instances, whose exact answer is the uniform
> distribution, it always harms: D_TV to uniform rises from 0.021 to 0.75
> (clipping) and 0.88 (simplex projection) at n = 3, and from 0.035 to 0.81 and
> 0.94 at n = 4. The simplex projection puts all the mass on one outcome (D_TV =
> 1 − 2⁻ⁿ), as in simulation (Fig. 10).

## 6. Abstract statement (two sentences)

> On an IBM Heron processor the search keeps 47 % of the ideal signal at two
> items, 10–29 % at three and at most 2 % at four, and the device behaves like
> its own calibration-based noise model with every error multiplied by 3.18
> [3.07, 3.35], a factor that predicted a second run 43 h later to within 0.05
> on three of its four instances. Zero-noise extrapolation recovered the
> three-item instances only with a gentle fold grid (λ ≤ 1.5), and a resolution
> gate kept unsupported extrapolations, such as 3.55 for a probability, out of
> the reported results.

## 7. The ~60 s job

Run it, after v1, but change it so it tests something the paper claims.

- As specified it answers two questions, and neither changes a main-text
  sentence much. Layout-or-day for P-n3-M2-17 is decisive in only one of its
  three outcomes ("both fine" cannot tell a transient from a recalibration fix),
  and TLS-type defects move over days anyway. P-n4-M1-10_r1 on the gentle grid
  is predicted identifiable by **both** α = 1 and α̂, so like job 2 it cannot
  separate them.
- What it can add: (1) a **third day** for α̂, which is the weakest link in the
  paper's main claim (one out-of-sample day, 4 records, one failure); (2) the
  first test that separates α̂ from α = 1. Add job 1's **P-n4-M1-10_r2**
  circuit (475 2Q, r = 2 < r_opt, ideal 0.908) on the gentle grid. Survival
  at α̂ predicts kept 0.006 / 0.002 / 0.001 at λ = 1 / 1.25 / 1.5, i.e. not
  identifiable at 12 288 shots; α = 1 predicts identifiable (S = 0.20). Job 1's
  exponent of 2.0 (§2 F2) predicts 0.041 / 0.018 / 0.008, i.e. z ≈ 16, 7, 3:
  identifiable. So the record tests both "one α" and the depth effect. Avoid
  r3: it is over-rotated (r > r_opt), where "signal kept" and the survival
  logic do not apply cleanly (the α̂ model gives kept 0.23 against S^α̂ = 0.023).
  +3 PUBs, 17 total, well under the 232 s cap.
- Pre-register properly: commit **and push** the `--predict` output before
  submitting, so the timestamp is GitHub's, not the file's.
- Outcome → text: α̂ holds on day 3 → "predicted two later runs"; it fails →
  the α claim becomes "describes one day", which a referee will ask about
  anyway. Either way it goes into v2 / the journal version, not v1.
