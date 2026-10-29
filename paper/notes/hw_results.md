# Hardware results: analysis and text (WP7)

What the paper says about the two ibm_fez jobs, with every number traced to a file.

- **Job of record:** `dbfeqr7gdj9s73bbgh6g`, ran 2026-10-26 06:05–06:14 UTC (folder label `2026-10-25` is the plan date), odd grid λ ∈ {1, 3, 5}, 47 PUBs × 4096 shots.
- **Second job:** `dbgkqq8te5oc738icnfg`, ran 2026-10-28 01:18–01:19 UTC (folder label `2026-10-28` is the plan date), gentle grid λ ∈ {1, 1.25, 1.5}, 14 PUBs × 12 288 shots.

Folder and file labels (`hw_2026-10-25/`, the summaries' `date`) are plan dates; the paper gives the UTC run date.

Sources:

- `paper/data/hw_summary.json` (job 1) and `hw_summary_2026-10-28.json` (job 2): blocks `model_vs_hardware`, `survival`, `drift`, `records`. Both are built by `paper/scripts/hw_aggregate.py`.
- `paper/data/hw_alpha_exec_2026-10-25/fit.json`.
- `paper/data/sim_per_instance.csv`, `sim_models.csv`, `sim_summary.json`, `sim_sweep_fake/` (for §7–8).
- `paper/data/claims_check.json`, written by `paper/scripts/check_claims.py`: every number the WP7 review (`hw_review_wp7.md` §1–§2) questioned, recomputed from the files above and compared with this note. Claim ids in the corrections log at the end are that file's.

Every random step is seeded (bootstrap seed 0, B = 2000), so regenerating the summaries reproduces these numbers exactly. Deviations are always **hardware − model**.

---

## 1. Does the noise model predict the device?

**No, not at the calibrated error rates. Yes, once every calibrated error is multiplied by α̂ = 3.18 [3.07, 3.35].** The sign of the miss is systematic: the device is worse than its model on every distinct unfolded circuit of both days.

| | PUBs | P(sol): mean dev. | mean \|dev.\| | hw < model | sign test p | D_TV to ideal: mean dev. | D_TV(hw, model) | shot floor |
|---|---|---|---|---|---|---|---|---|
| Job 1, α = 1 (exec. calibration) | 35 / 47 | −0.116 | 0.117 | 32 of 35 | 4e−7 | +0.082 | 0.109 | 0.026 |
| — λ = 1 only | 13 / 17 | −0.198 | 0.198 | 13 of 13 | 2e−4 | +0.151 | | |
| — λ = 3 | 11 / 15 | −0.089 | 0.089 | 11 of 11 | 1e−3 | +0.061 | | |
| — λ = 5 | 11 / 15 | −0.044 | 0.049 | 8 of 11 | 0.23 | +0.026 | | |
| Job 1, α̂ = 3.176 (**in-sample**) | 35 / 47 | −0.0003 | 0.008 | 18 of 35 | 1.0 | −0.002 | 0.034 | 0.026 |
| Job 2, α = 1 (exec. calibration = plan's: 0 entries changed) | 14 / 14 | −0.275 | 0.275 | 14 of 14 | 1e−4 | +0.275 | 0.275 | 0.010 |
| Job 2, α = 3.176 (**out-of-sample**, made before the job) | 14 / 14 | per record, below | | | | +0.012 | 0.045 | 0.012 |

How to read the table:

- **PUBs** is two counts. The first is the PUBs with signal, which are the ones used for P(sol). `M = 0` and `M = N/2` are left out of that count: there P(sol) is fixed whatever the noise. The second is every PUB, which is used for D_TV.
- **PUBs are not circuits.** Job 1's 13 λ = 1 PUBs with signal are 11 distinct circuits plus 2 drift copies; job 2's 14 PUBs come from 4 records (4 distinct circuits at λ = 1, each folded twice, plus 2 drift copies). The means and sign tests in the table treat folded scales and drift copies of one circuit as independent, so the text counts distinct circuits, and job 2 at α̂ is reported per record.
- **D_TV(hw, model)** is the mean distance between the two histograms. The **shot floor** is the mean distance between two samples drawn from the same model distribution.
- The P(sol) deviations at α̂ use exactly the fit's interpolation. The distributions at α̂ are interpolated per outcome in the same way. α̂ is not a grid point, and no new simulation was run.

Reading:

- **At α = 1 the model is optimistic everywhere.** By λ = 5 the miss shrinks only because model and device both sink toward P_uniform.
- **One α rescales the folded circuits too** (§ Open 3 of `status_after_wp6.md`).
- **Whole distributions: on job 1 only.** At α̂, D_TV(hardware, model) on job 1 (the fit day) within 0.008 of the shot floor (0.034 against 0.026), including the 12 PUBs without signal, which were not in the fit. Out of sample it is not: job 2 without P-n3-M2-17: 0.026 against a floor of 0.011 (2.4×).
- **Job 1's agreement at α̂ is in-sample.** α̂ was fitted to those 35 values, so its near-zero mean deviation is a goodness-of-fit, not a test.
- **Job 2 is the test.** The prediction at α = 3.176 was timestamped 14 min before the job started (prediction file 01:04:36 UTC, job start 01:18:35 UTC). Its commit is later: `09b9c46` is 01:32 UTC, after the job started, so the time quoted is the file's stamp, not git's.

  Hardware − model at α̂, λ = 1, one distinct circuit per record:

  | Record | 2Q | hw − model at α̂ |
  |---|---|---|
  | P-n2-M1-11 | 71 | +0.011 |
  | P-n3-M1-17 | 235 | +0.015 |
  | P-n3-M1-17_r1 | 126 | +0.044 |
  | P-n3-M2-17 | 184 | −0.109 |

  - RMS 0.060 over the 4 records; 0.028 without P-n3-M2-17.
  - The other three ran within 0.045 on every PUB (all scales and the drift copy). P-n3-M1-17_r1 ran above the prediction at every scale.
  - P-n3-M2-17 was at random (§4).
  - A mean over PUBs, or a sign test, would let errors of opposite sign cancel, so neither is quoted for job 2.

**Fig S3 (hardware vs model scatter): not made.**

- At α̂, job 1's agreement is fitted by construction.
- The out-of-sample information is 14 points from 4 records.
- Fig 5 already shows hardware, the device model and the model at α̂ per record.

The table above says the same in one line. Proposed text:

> "Simulated on its own calibration, the device model overstates P(solution) on every unfolded circuit of both runs (mean deviation −0.18 on the 11 distinct unfolded circuits with signal of the first run, −0.28 on the 4 of the second). Multiplying every calibrated error by α̂ = 3.18 [3.07, 3.35], fitted to the first run, removes the bias on that run. Applied to the second run before it ran, it predicted three of its four records within 0.045 on every PUB; the fourth was at random that day."

## 2. Does ZNE help, and which estimator?

**The scalar estimator** (exponential, asymptote fixed at P_uniform, identifiability and resolution gate) helps when two conditions hold: the folded points stay above random, and they are close enough to λ = 1 to carry signal. For these instances the grid decided whether that happens (argument under **Odd grid**).

| Record | 2Q (job 1 / 2) | Ideal | Odd grid, job 1 | Gentle grid, job 2 | Recovery on gentle grid |
|---|---|---|---|---|---|
| P-n2-M1-11 | 71 / 71 | 1.000 | 0.953 [0.890, 1.000], resolved | 0.981 [0.928, 1.000], resolved | 0.95 |
| P-n3-M1-17_r1 | 112 / 126 | 0.781 | identified, not resolved | 0.730 [0.633, 0.831], resolved | 0.89 |
| P-n3-M1-17 | 212 / 235 | 0.945 | not identifiable | 0.821 [0.660, 1.000], resolved | 0.82 |
| P-n3-M2-17 | 194 / 184 | 1.000 | not identifiable | not identifiable (at random) | — |

Values and intervals are reported clipped to [P_uniform, 1] (`REPORT_BOUNDS = "clip"`, Step 5; `hw_summary*.json`, `records[].zne_reported`, `_lo`, `_hi`). Clipping moved only the three upper ends, which were above 1; their values as fitted are in `paper/data/report_bounds_diff.json`, `json.<file>.changes`.

**Recovery** = `(ZNE − P(λ=1)) / (ideal − P(λ=1))`.

**Odd grid.** For n ≥ 3, λ = 3 is already at random: P-n3-M1-17 gives 0.128 against 0.125 (z = 0.7). That leaves one informative point, so no fit can be identified, and more shots cannot help on this grid. This matches the simulation's partial-folding result (§7, paragraph 2), and is sharper on hardware.

**Why "the grid decided" needs that argument.** Job 2 changed more than the grid: 3× the shots, the day, the layout and the transpilation. The hardware alone does not isolate the grid. Two facts do: on the odd grid every n ≥ 3 λ = 3 point is already at random (above), and in simulation at equal shots the odd grid resolves 28 % of n = 3 instances with signal, the planned grid 98 %. So: for these instances the grid decided.

**Gentle grid.** It resolved every M = 1 record, and all three intervals contain the ideal. P-n3-M1-17 at 235 2Q is *larger* than P-n3-M2-17 at 194, which marked the upper edge of the odd-grid band (§3).

**Mitiq, on the same counts.**

- *Where we resolve,* it agrees with us: |Δ| = 0.010 (job 1, n = 2), and 0.004 (n = 2), 0.027 (P-n3-M1-17) and 0.005 (P-n3-M1-17_r1), job 2.
- *Where we decline,* it still returns a point estimate (job 1):
  - **3.547** on P-n3-M2-17, a probability above 1 (its own 95 % interval is [0.457, 4.100]).
  - 0.831 on P-n4-M1-10_r2, from a λ = 1 value of 0.096 against a random level of 0.0625. That is close to the ideal 0.908 only by accident (interval [0.156, 1.102]).
  - 0.534 on P-n3-M1-17_r3, 62 % above its ideal of 0.330.
- On job 2's flat P-n3-M2-17 it returns 0.239, a plausible-looking number, with an interval of [−1.1e7, 1.4e5].
- Mitiq's intervals show the problem; its point estimates do not.

**Our raw fit is no better without the gate.** It returns 376.6 on P-n4-M2-02 (identified, not resolved) and 2275 on P-n3-M2-17. So the gate, not the model, removes these catastrophic values. It does not remove every unphysical one: in simulation the strict rule's fit still lands outside [0, 1] on 19 of 340 planned-grid values, 19 of them above 1 (max 4.17), all from fits that passed the gate. Since Step 5 they are reported clipped to [P_uniform, 1]: 0 reported values outside [0, 1] remain (`sim_per_instance.csv`: `zne_scalar_reported`; the fit as fitted is `zne_scalar_value`, flagged by `out_of_range`). This is also the hardware argument for keeping the strict reporting rule (§7, paragraph 3).

**Pointwise ZNE** (one exponential per bitstring):

- **Helps where the scalar fit is resolved.** D_TV to the ideal, raw → clip:
  - job 1, P-n2-M1-11: 0.400 → 0.047;
  - job 2: 0.354 → 0.052, 0.452 → 0.090 (r = 1) and 0.698 → 0.157 (P-n3-M1-17).
- **A coin toss outside the resolved fits.** On the 11 records with signal where the scalar fit is not resolved, clipping lowered D_TV on 5 of 11 records and raised it on 6, up to 0.998. Nothing in the data says which will happen. D_TV to the ideal, raw → clip:
  - lower: P-n4-M1-10_r2 0.812 → 0.137, P-n3-M2-17 (job 1) 0.647 → 0.065, P-n3-M1-17_r1 0.465 → 0.217, P-n4-M2-02 0.803 → 0.521, P-n4-M1-10 0.884 → 0.864;
  - higher: P-n3-M1-17 0.689 → 0.990 (it puts 0.997 of the mass on `101`, which is not a solution), S-n4-g1-M2-00 0.814 → 0.986, P-n4-M1-10_r1 0.359 → 0.926, P-n3-M1-17_r3 0.184 → 0.668, P-n4-M1-10_r4 0.519 → 0.939, P-n3-M2-17 (job 2) 0.754 → 0.998.
  - This is an argument for the gate: where it declines, the pointwise result is not merely noisy but unpredictable in sign.
- **M = 0 (Fig 10):** the exact answer is uniform, so any distance it adds is pathology.
  - n = 3: raw 0.021 → clip 0.748, simplex 0.875 (35× and 41×).
  - n = 4: 0.035 → 0.811 and 0.937.
  - n = 2 is benign: 0.020 → 0.029.
  - This matches the simulation (§7, paragraph 4). The simplex projection does not rescue it.
- **Rule for the paper:** pointwise ZNE only behind the scalar gate.

## 3. Where amplification dies

**Signal kept at λ = 1, job 1:**

| Circuits | Kept |
|---|---|
| n = 2 (71 2Q) | 0.47 |
| n = 3 (112–350 2Q) | 0.10–0.29 |
| n = 4 at r_opt (466–749 2Q) | 0.008–0.021 |

**Correction to `status_after_wp6.md`** (now made there too). At n = 4 the signal is detectable on two of three instances:

| Record | P(sol) [95 % CI] | P_uniform | Above random | Kept |
|---|---|---|---|---|
| P-n4-M1-10 | 0.077 [0.069, 0.085] | 0.0625 | 3.6σ | 1.7 % |
| P-n4-M2-02 | 0.142 [0.131, 0.153] | 0.125 | 3.1σ | 2.1 % |
| S-n4-g1-M2-00 | 0.131 [0.121, 0.141] | 0.125 | 1.2σ (null) | 0.8 % |
| P-n4-M1-10_r4 | 0.063 [0.055, 0.070] | 0.0625 | 0.1σ (null) | — |

Write instead: "at n = 4 the device keeps at most 2 % of the ideal signal: detectable at 3σ on two of three instances, useless for search".

**The band.** Read off the odd-grid statuses, it runs from 71 2Q (largest resolved) to 194 2Q (smallest larger one not identifiable). The one exception is P-n4-M2-02 (495 2Q), identified but not resolved. Its series is not monotone: λ = 3 sits at random (z = −0.8), and its "identified" status rests on one λ = 5 point at z = 2.8 with χ²_red = 8.4. The gate has no goodness-of-fit check, and with a one-sided 2σ rule on the 20 folded n ≥ 3 points about 0.5 false informative points are expected, so P-n4-M2-02 is most likely that false positive. (In simulation the n = 3, r = 1 sweep point is resolved with χ²_red = 11.4.) Keep listing it as the exception; this is the sentence to explain it. Job 2 resolved the same instance (212 2Q on the odd grid, 235 2Q on the gentle grid) that the odd grid could not identify. So the band is a property of the circuit *and* the fold grid and shots, not of the circuit alone. Fig 4 should say "odd grid".

**Does WP2's survival estimate place the band? (b)**

`S = (1 − ε₂)^n₂ (1 − ε₁)^n₁` uses median errors from the execution calibration (ε₂ = 2.61e−3, ε₁ = 3.61e−4) and physical gate counts. A folded circuit keeps S^λ. The WP4 pre-registration rule (`plan_scales`: same shots, k = 2) then says on which scales the signal stays resolvable, and two or more scales are needed for an identifiable fit.

| | Predicted band (by `limit_band`) | Identifiable as measured |
|---|---|---|
| Measured (job 1) | 71 – 194 2Q, exception P-n4-M2-02 | — |
| Survival, α = 1 | 233 – 350 2Q | 7 of 11 records |
| Survival, α̂ = 3.176 | 112 – 194 2Q | 10 of 11 records |

- **At α = 1, no.** S is an estimate, optimistic because it ignores idling, crosstalk and leakage, and here it is too optimistic by about 2× in 2Q count. It calls n = 3, M = 1 at r_opt (212 2Q) and P-n4-M1-10_r1 (233 2Q) identifiable on the odd grid; neither was. This is the reportable result: the pre-registered test, α = 1, failed: 7 of 11, limit placed at 233–350 2Q instead of 71–194 (WP2 step 5).
- **At α̂, consistent, in-sample.** Its upper edge is the measured one, and its only miss is the band's own exception (P-n4-M2-02: predicted not identifiable, measured identified). That is weak evidence: with these 11 records any 2Q threshold in (112, 194] gives 10 of 11 (identifiable iff 2Q < threshold).
  - Its lower edge is 112 rather than 71 because it predicts identifiability, which P-n3-M1-17_r1 had; the measured edge counts resolution.
  - S^α̂ also lands near the measured signal (largest gap 0.061): 0.465 vs 0.466 (n = 2), 0.125 vs 0.138, 0.099 vs 0.160, 0.297 vs 0.292 (n = 3).
- **Caveat:** α̂ comes from these PUBs, so this shows consistency, not prediction.
- **Out of sample (job 2, gentle grid, α = 3.176 from job 1):** it predicted all four records identifiable. Three were resolved; P-n3-M2-17 was at random (§4). The uncorrected estimate predicts the same, with the same miss (`identifiable_agreement` in `hw_summary_2026-10-28.json`: 3 of 4 at α = 1 and 3 of 4 at α̂). Job 2 does not separate α̂ from α = 1.

**Depth: one α holds up to about 230 two-qubit gates.** Per record, the exponent `ln(kept) / ln(S)` (S at α = 1; scaling every error by α turns S into about S^α, so the exponent is the record's own α[^cube]), with delta-method standard errors from the shot noise of `kept`, on the job-1 records whose kept signal is at least 3σ above zero: up to 233 2Q the exponent is 2.52–3.23; from 350 2Q on it is 1.63–2.30.

- P-n4-M1-10_r2 keeps 0.040 ± 0.005 (7σ above random) and gives an exponent of 2.01 ± 0.09.
- P-n4-M1-10 at r_opt gives 1.63 ± 0.11, P-n4-M2-02 2.30 ± 0.19.
- The P(sol) residuals at α̂ show the same thing: P-n4-M1-10_r1 and _r2: +0.022 and +0.022 at λ = 1.
- S-n4-g1-M2-00 and P-n4-M1-10_r4 are left out: their kept signal is under 3σ, so their exponents are not determined.
- The α fit hides this because deep circuits sit near P_uniform and carry little weight in P-space.

Write: "one α describes circuits up to about 230 two-qubit gates; deeper circuits retain more signal than it predicts." This matters for any further QPU job (`hw_review_wp7.md` §7).

[^cube]: The "cube rule" (kept ≈ S³) is the same fact as α̂ ≈ 3.2, not extra evidence. On job 1 the exponents are 3.18, 3.04, 2.52 and 3.23 for the four records that kept more than 0.13 of their signal. The 0.13 cut is arbitrary: the next two records (kept 0.125 and 0.101) give 2.64 and 1.93.

The statement for C3:

> "A gate-count survival estimate built from the device's calibration placed the limit about twice too far out; this was the pre-registered test. With the calibrated errors scaled by the factor that one hardware run measures, the same estimate is consistent with the odd-grid limit on that run (in-sample)."

## 4. P-n3-M2-17 at random on 2026-10-28

**What can be said:**

- **On day 1 it behaved as the α̂ model says.** It kept 0.138 of its signal at λ = 1; the model at α̂ gives 0.146.
- **On day 2 it was at random on every one of its four PUBs.**
  - λ = 1: 0.246 [0.238, 0.254] against 0.25 (z = −1.0).
  - λ = 1.25 and 1.5: also random.
  - Drift repeat: 0.248.
  - This is 12.8σ below day 1 (one copy each day).
- **It is not drift inside the job.** The two copies agree in P(sol) (z = +0.31) and in the whole histogram (D_TV 0.0146 [0.0106, 0.0288]; shot-noise null mean 0.0136, null p = 0.36).
- **It is not a stuck qubit.** The marginals are 0.46–0.51.
- **It is not the whole job.** The other three records ran slightly *better* than the α̂ prediction (§1). P-n2-M1-11 used qubits 90, 91, 92 and 98, which overlap this layout, and was fine.
- **The distribution is not exactly uniform.** D_TV to uniform is 0.043, against a shot floor of about 0.01. The surplus sits on `001` (0.151) and `000` (0.134), not on the solutions `011` and `110`.
- **The calibration in force did not flag it.** That calibration (stamp 2026-10-27 19:08 local, fetched afterwards with `--fetch-calibration`, no QPU time) has 0 entries changed since the plan's calibration (18:12). The method check passes, and the ISA check of all 14 circuits passes. On it the model predicts 0.649 [0.641, 0.657] at λ = 1 (signal kept 0.532), while the device gave 0.246 (kept −0.005). Whatever went wrong is not visible in any calibration we have.

**What cannot be said:**

- **Whether this was a bad layout or a bad day.** The day-2 qubits {78, 88, 89, 90, 91, 92, 98} share only qubit 98 with day 1's {98, 109, 110, 111, 112, 113, 118}, so layout and day changed together. §5's job separates them.

Text (as drafted in the status note, still accurate):

> "…the third n = 3 record was at random on that day although the calibration rated its layout as ordinary."

Add: "Its distribution is near uniform but not exactly, and a repeat at the end of the job reproduced it, so the failure lasted the whole job."

## 5. The remaining ~465 s

**Recommendation: use about 60 s of it, in one job, after arXiv v1 is submitted and before the window ends (≈ 2026-11-21).** v1 does not depend on it: §4's wording stands either way. The budget is lost at the window's end, so the cost is a few hours of plan / dry run / checklist, not the QPU time.

The job answers two questions the paper currently leaves open:

1. **Layout or day (§4).** Run P-n3-M2-17 pinned to the day-1 layout *and* to the day-2 layout, in the same job.

   | Day-2 layout | Day-1 layout | Reading |
   |---|---|---|
   | random | fine | a persistent defect the calibration does not show |
   | fine | fine | a transient on 2026-10-28 |
   | random | random | not a layout effect; reopen |

2. **Does the gentle grid reach n = 4?** P-n4-M1-10_r1 (233 2Q, r = 1, ideal 0.473) is the same size as the 235-2Q record that resolved on day 2. The survival plan at α̂ predicts it identifiable on λ ∈ {1, 1.25, 1.5} at 12 288 shots. On job 1 it kept 0.125 (z = 10 above random).
   - If it resolves, C2's sentence becomes "the gentle grid resolves up to n = 4 at r = 1".
   - If not, the limit depends on more than the 2Q count (a 233-2Q circuit fails where a 235-2Q one resolved), which is equally quotable.

Manifest spec for `--plan` (not run; no submission):

| | |
|---|---|
| Device | ibm_fez, job mode, twirling on, DD off (as job 2) |
| Records | P-n2-M1-11 (control, fresh layout); `P-n3-M2-17_L1` and `P-n3-M2-17_L2` = the scale-1 ISA circuits jobs 1 and 2 sent (194 and 184 2Q); `P-n4-M1-10_r1_L1` = job 1's circuit (233 2Q). Pinned records reuse the saved circuit, so routing is the same as well as layout; the gentle-grid folds are made from it. |
| Layouts (`initial_layout` order) | `L1` P-n3-M2-17 [113, 109, 118, 110, 112, 98, 111], seed 11; `L2` [88, 92, 78, 89, 91, 98, 90], seed 23; P-n4-M1-10_r1 [79, 94, 111, 90, 92, 98, 93, 91], seed 3 |
| Scales | 1, 1.25, 1.5 |
| Shots | 12 288 per PUB |
| Drift repeats | P-n2-M1-11, `P-n3-M2-17_L2` |
| PUBs | 4 records × 3 scales + 2 = 14 (no r_opt record of P-n4-M1-10: it is not in `--only`) |
| Budget | estimate from job 2's measured per-shot time and overhead (`--per-shot-time 0.000622611 --overhead 0.3824`), × 2 safety; cap = half of what remains (≈ 232 s); expected charge ≈ 50–70 s |
| Pre-registration | before submitting: `noise_model_check.py --predict --alpha 3.176 --calibration paper/hardware/calibration_<date>.json --date <date>`, committed **and pushed** before `--submit`, so the time is GitHub's (job 2's prediction carried only its file timestamp, §1) |
| Gate | `CHECKLIST_template.md`, unchanged |

**Plan command** (`--pin-layout` and `--drift` added to `hardware_batch.py` for this; not run against the service):

```
python paper/scripts/hardware_batch.py --plan --backend ibm_fez --only P-n2-M1-11 --no-sweep --pin-layout P-n3-M2-17:L1=2026-10-25 P-n3-M2-17:L2=2026-10-28 P-n4-M1-10_r1:L1=2026-10-25 --drift P-n2-M1-11 P-n3-M2-17_L2 --scales 1 1.25 1.5 --shots 12288 --per-shot-time 0.000622611 --overhead 0.3824
```

Built offline against the 2026-10-28 calibration snapshot, this gives the 14 PUBs above, with the pinned circuits' 2Q counts equal to their source jobs' (194, 184, 233).

Two corrections to the first version of this spec:
- **The layouts it listed were `physical_qubits`, not `initial_layout`.** They contain the same qubits in a different order, so pinning them would have put the wrong logical qubits on those physical qubits. The flag now reads layouts from the manifests instead of taking typed lists. §4's "only qubit 98 is shared" is about sets of qubits, so it still holds.
- **Re-transpiling on a recorded layout with its recorded seed does not reproduce that job's circuit.** In the seed search, SABRE chose the layout and the routing together. With a fixed layout, only the routing is searched, so SWAPs and the 2Q count can change. So `KEY:TAG=<date>` reuses the circuit that job sent. The typed form `KEY:TAG=q0,...[@seed]` is still available, but it transpiles again.

If you would rather not spend the time: **no use** is defensible. Nothing in C1–C3 depends on it.

## 6. Results within CI of "no effect" (null results, not effects)

| # | Quantity | Value | Why it is a null |
|---|---|---|---|
| 1 | Drift, job 1, P-n2-M1-11 (first vs last PUB) | Δ = +0.010, z = +0.90; D_TV 0.0127 [0.0049, 0.0334], null mean 0.0132, p = 0.50 | within shot noise |
| 2 | Drift, job 1, P-n3-M2-17 | Δ = 0, z = 0; D_TV 0.0178 [0.0154, 0.0454], null mean 0.0231, p = 0.78 | within shot noise |
| 3 | Drift, job 2, P-n2-M1-11 | z = −0.61; D_TV 0.0066 [0.0033, 0.0173], null mean 0.0073, p = 0.54 | within shot noise |
| 4 | Drift, job 2, P-n3-M2-17 | z = +0.31; D_TV 0.0146 [0.0106, 0.0288], null mean 0.0136, p = 0.36 | within shot noise |
| 5 | S-n4-g1-M2-00 at λ = 1 | 0.131 [0.121, 0.141] vs P_uniform 0.125 | CI contains random |
| 6 | P-n4-M1-10, r = 4 | 0.063 [0.055, 0.070] vs 0.0625 | CI contains random |
| 7 | P-n3-M2-17, job 2, all scales | 0.246 [0.238, 0.254] vs 0.25 (z −1.0, −0.8, 0.0) | at random |
| 8 | Every n ≥ 3 record at λ = 3 and 5 on the odd grid (except P-n3-M1-17_r1 at λ = 3, z = 5.2, and P-n4-M2-02 at λ = 5, z = 2.8, see §3) | \|z\| ≤ 2.3 vs P_uniform | folded signal gone |
| 11 | Model at α̂ vs job 1, D_TV to ideal | mean dev. −0.002, p = 0.24 | no residual bias (in-sample) |
| 12 | Our ZNE vs Mitiq where we resolve | \|Δ\| ≤ 0.027, inside both intervals | the estimators agree |
| 13 | Odd vs gentle grid on P-n2-M1-11 | 0.953 [0.890, 1.000] vs 0.981 [0.928, 1.000] | grid does not matter at n = 2 |
| 14 | Every resolved hardware ZNE interval vs the ideal | 4 of 4 contain it | no detectable ZNE bias (also no power to see a small one) |
| 15 | M = N/2 control P-n2-M2-03 | 0.494 vs 0.5 | as it must |

For rows 1–4: the bootstrap interval of a D_TV never contains 0, because two finite samples are always apart. The null comparison is the test; the interval alone is not.

Rows 9 and 10 are removed (corrections log, I and H.gone); the other rows keep their numbers. Row 9 compared different transpiled circuits on different layouts, so it was not a clean null. Row 10 was a mean over PUBs that hides one record at random and one above the prediction at every scale (§1).

## 7. Carried over from WP5: the six TEXT TO WRITE paragraphs

Numbers are from `sim_per_instance.csv` / `sim_models.csv` (rule v2, 463 instances, fake_marrakesh).

- The paragraphs live here rather than in `sim_results.template.md`, because that template only takes numbers from `sim_summary.json` and several below are per-n breakdowns that file does not carry.
- Each `TEXT TO WRITE` marker there now reads "see `hw_results.md` §7" (2026-10-28), and `sim_results.md` was regenerated from it.

**1. Recovery across n.**

> "Signal recovery, the share of the gap between the unmitigated and the noiseless P(solution) that the reported value closes, has a median of 0.81, 0.90 and 0.58 at n = 2, 3 and 4 on the planned grid, and 0 at n = 5 and 6. There no fit passes the gate and the unmitigated value is reported. On the odd grid the median is 0 from n = 2 on: even at n = 2 it resolves only 11 of the 27 feasible instances. The pooled planned-grid median, 0.25 over 347 feasible instances, mixes the two regimes and should not be quoted alone."

The noiseless value in the recovery is the ideal (sampled) run, as `aggregate.py` defines it. The n = 3 median is 0.886 with the exact P_ideal as the noiseless value instead. The review's 0.886 (row 5) is this variant, not a correction of 0.90.

**2. Odd vs planned grid.**

> "On instances with signal, the odd grid identifies the exponential on 100 %, 78 %, 39 %, 14 % and 17 % of instances at n = 2…6, and resolves it on 55 %, 28 %, 6 %, 1 % and 3 %. The planned grid identifies 100 %, 100 %, 86 %, 44 % and 35 %, and resolves 100 %, 98 %, 72 %, 12 % and 20 %, at the same shot count. The odd grid stops identifying at n = 4 and the planned grid holds through n = 4."

The hardware repeats this one step earlier: at n = 3 the odd grid fails and the gentle grid resolves (§2).

**3. The reporting rule.**

Keep the pre-registered rule: report the extrapolation only when it is resolved *and* identified. The relaxed rule ("identified") is more accurate in the median but admits unphysical values.

On instances with signal, planned grid:

| Reported value | Median bias vs ideal | Values outside [0, 1] |
|---|---|---|
| Strict rule | −0.44 | 19 of 340 as fitted; 0 as reported (clipped, §2) |
| Relaxed rule | −0.34 | 26 of 340 |
| Raw extrapolation | −0.17 | 71 of 340 (mean bias +16.7) |

On the odd grid the median biases are −0.77 (strict) and −0.76 (relaxed).

On the identified-but-not-resolved fits, the extrapolation is closer to the ideal than the λ = 1 value:

| Grid | Better than λ = 1 | Values outside [0, 1] |
|---|---|---|
| planned | 50 of 52 | 7 |
| odd | 97 of 99 | 15 |
| odd7 | 62 of 62 | 9 |

On hardware, the one identified-not-resolved record would have reported **376.6** (P-n4-M2-02).

The strict rule does not keep every unphysical fit out: its 19 fits outside [0, 1] are all above 1, up to 4.17 (§2), and are reported clipped to 1. The relaxed rule adds 7 more (max 298). So the strict rule pays its higher median bias (table above) to remove the worst tail, not every unphysical value.

> "We report the extrapolation only when resolved; the relaxed rule lowers the median bias from −0.44 to −0.34 on the planned grid but admits 26 fits outside [0, 1] in simulation, against 19 for the strict rule, and a value of 376.6 on hardware."

Open (Step 5): with clipping, "admits values outside [0, 1]" no longer separates the two rules, since a relaxed-rule fit would be clipped too (376.6 would be reported as 1). The argument for the strict rule has to rest on what an unresolved fit's value is (set by the rate cap, `zne_theory.md` §4), not on its range.

**4. M = 0 pathology.**

> "On the 116 infeasible instances, pointwise ZNE raises the median D_TV to the exact (uniform) answer from 0.023 at λ = 1 to 0.86 after clipping (37×), and to 0.94 after simplex projection (41×), on both grids, and makes every instance worse. The projection does not help because the damage is done before it: with P_uniform as the asymptote, each bitstring's extrapolation divides shot noise by the decay of a signal that is not there. Weights that grow as the folded points approach the asymptote turn a ±1σ fluctuation into a peak. Hardware reproduces it: 0.021 → 0.75 (clip) / 0.88 (simplex) at n = 3, and 0.035 → 0.81 / 0.94 at n = 4 (§2)."

**5. Best extrapolation model** (odd7 grid, identified fits only, 137 candidates).

> "Richardson has the lowest median absolute error, 0.057, but on only 13 fits; it excludes 124. The exponential reaches 0.113 on 75, quadratic 0.32 on 30, linear 0.55 on 75. Richardson is not more accurate; it is more selective. We use the exponential, which is identified six times as often, at twice Richardson's median error."

**6. Iteration sweep** (simulation; hardware in §3 of the status note).

> "On fake_marrakesh the noisy P(solution) of the n = 3 sweep instance peaks at r = 1 (0.560), not at the analytic r = 2 (0.945, noisy 0.524). The noisy curve stops tracking the analytic one at the first iteration beyond r = 1. ZNE, resolved at every r for n = 3, moves the peak back: 0.736, 0.922 and 0.412 for r = 1, 2 and 3 (analytic 0.781, 0.945, 0.330). At n = 4 the noisy curve falls monotonically (0.230, 0.215, 0.148, 0.133 for r = 1–4, analytic peak 0.961 at r = 3). ZNE resolves only r = 1 (0.437 vs 0.473), so the reported peak stays at r = 1. On hardware the same ordering holds and is sharper: fewer iterations win at both n (status note, Fig 9)."

## 8. Carried over from WP5: the Mitiq section

Proposed text:

> **Comparison with Mitiq.** To separate what our pipeline adds from what any ZNE implementation would give, we passed the same folded counts to Mitiq's exponential extrapolation, with the same fixed asymptote and the same bootstrap (`mitigation.mitiq_baseline`; folding stays ours, at the ISA level, so only the extrapolation differs).
>
> **Where our gate resolves the fit, the two agree.** In simulation, the median difference is 0.016 on the odd grid (41 fits) and 0.061 on the planned grid (186). The median error to the ideal is 0.063 vs 0.053 (odd) and 0.107 vs 0.128 (planned), ours vs Mitiq. On hardware the four resolved estimates differ by at most 0.027.
>
> **Where our gate declines, Mitiq still returns a point estimate.** On the planned grid, its median error on those 154 instances is 0.86, 47 values lie outside [0, 1], and 119 are off by more than 0.5. On the odd grid (299 declined) the figures are 0.69, 40 and 176.
>
> These estimates are closer to the ideal than the unmitigated value we fall back to in most cases (260 of 299 on the odd grid), because any upward extrapolation is. But they cannot be told apart from the wrong ones without the information the gate uses. On hardware this produced 3.547 for a probability, and a value of 0.239 whose interval spans [−1.1·10⁷, 1.4·10⁵].
>
> Our fallback is not a better estimate of the ideal. It is the unmitigated value, flagged as such, and a lower bound. The gate's contribution is to remove the catastrophic failures; a fit that passes it can still exceed 1 (§2), and is then reported clipped to 1.

---

Fig S3: decided against (§1). Tables S2 and 3–4 unchanged; job 2's Table (`hw_table_results_2026-10-28.tex`) now has its model column, from the execution calibration (fetched 2026-10-27; identical to the plan's, so the values did not change).

## Corrections log (2026-10-28)

From `hw_review_wp7.md` §1–§2. Ids are `check_claims.py`'s; each value is that claim's `recomputed` in `paper/data/claims_check.json`.

- R1a, R1b (§2): Mitiq |Δ| 0.009 → 0.010 (job 1); job 2 0.003 → 0.004, and each job-2 value now names its record.
- R2 (§2): M = 0, n = 3 ratios 36× / 42× → 35× / 41× (taken from the unrounded raw D_TV).
- R3 (§3): "within 0.06" → "largest gap 0.061".
- R4 (§4): 13.4σ → 12.8σ, one copy each day (13.4σ pooled day 2's two copies against one day-1 copy).
- R5, R5x (§7): 0.90 kept; it is `aggregate.py`'s definition. The review's 0.886 uses the exact P_ideal and is added as that variant.
- R6 (§7): n = 4 noisy sweep at r = 3, 0.149 → 0.148.
- R7a, R7b (top): run times in UTC; folder labels are plan dates.
- R8 (§3): "a 235-2Q circuit" → the same instance at 212 2Q (odd grid) and 235 2Q (gentle grid).
- R9a–R9d (§1, §5): the job-2 prediction is "timestamped" 14 min before the job, not committed before it (`09b9c46` is after the job start); the next prediction is to be committed and pushed before submitting.
- R10a, R10b (§1): counts of distinct circuits, not PUBs (11 + 2 drift copies on job 1; 4 records on job 2); mean deviations at α = 1 recomputed over distinct circuits.
- A (§2): pointwise paragraph rewritten as a coin toss outside the resolved fits (5 lower, 6 higher of 11), every record listed (A.*).
- B, B2 (§2, §7, §8): the strict rule still reports 19 values above 1 in simulation; "keeps unphysical values out" / "says when it can be trusted" narrowed to "removes the catastrophic failures".
- C, C2 (§3): "correctly predicted the gentle grid" dropped; α = 1 and α̂ both give 3 of 4 on job 2.
- D, D2, D3 (§3): "places the limit exactly" → consistent, in-sample, any threshold in (112, 194] gives 10 of 11; the α = 1 pre-registered failure stated as the result.
- E1–E5 (§3, docstrings of `survival_estimate` and `plan_scales`, `instance_family.md` §2, `resource_model.md`): S is an estimate, optimistic because it ignores idling, crosstalk and leakage; "upper bound" dropped.
- F, F.next (§1 → §3 footnote): the cube rule moved to a footnote, with the arbitrary 0.13 cut and the next two exponents.
- F2.r2, F2.r, F2.range, F2.res (§1 → §3): "too close to zero to pin an exponent" replaced by the depth effect with delta-method errors; records with kept signal under 3σ are left out of the ranges.
- G1, G2 (§1): "within 0.008 of the shot floor" holds for job 1 only; job 2 without P-n3-M2-17 is 2.4× its floor.
- H.* (§1, §6): job 2 at α̂ reported per record with RMS, not as a mean and a sign test; null #10 removed.
- I (§6): null #9 (day 1 vs day 2 on different circuits and layouts) removed.
- J, J2 (§2): "set by the fold grid" made explicit: odd-grid λ = 3 already at random, and the simulation at equal shots.
- K, K2, K3 (§3): P-n4-M2-02's "identified" is most likely the expected false positive; the gate has no goodness-of-fit check.
- S1, S2 (`status_after_wp6.md`): n = 4 is "detectable but useless for search", with its σ above random.

Where `check_claims.py` disagrees with the review at the last digit, the note uses the script's value: G2 0.026 (review 0.027), H.P-n3-M2-17 −0.109 (review −0.110), F2.r2 standard error 0.09 (review 0.08), A maximum 0.998 (review "0.99").

## Step 5 changes (2026-10-29)

Reported scalar ZNE values outside the physical range are now clipped to [P_uniform, 1] at aggregation time (`REPORT_BOUNDS = "clip"` in `mitigation/zne.py`; comparison of the options in `paper/data/report_bounds.json`; every derived number that moved in `paper/data/report_bounds_diff.json`). Records are unchanged. Ids are `check_claims.py`'s.

- B (§2): "still reports 19 of 340 … outside [0, 1]" → the strict rule's *fit* lands outside on 19 of 340; B3: 0 reported values outside after clipping.
- B4 (§7): strict-rule median bias −0.44 unchanged by clipping.
- B5 (§2), B6 (§6, row 13): the three hardware interval upper ends above 1 now end at 1.000.
- B2 (§7), §8: "values" → "fits" where the unclipped fit is meant; the Methods quote and the open point on the strict-rule argument (§7).
