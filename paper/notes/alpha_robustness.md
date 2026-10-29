# Step 4: robustness of the α result (2026-10-29)

Offline. No QPU, no service call. Main file: `paper/data/alpha_robustness.json`
(written by `paper/scripts/alpha_robustness.py`; keys are below as `A:<key>`).
The per-class fits are in `paper/data/hw_alpha_exec_<class>_2026-10-25/fit.json`
(written by `noise_model_check.py --fit-alpha --scale-only <class>`). Figure:
`paper/figures/fig_alpha_depth.{pdf,png}`. Seeds: simulator 20261020; job-level α
fits B = 500 (the paper's fit, kept for comparability); per-record fits B = 2000,
seed 0.

**What this changes in the review.** Item 2 does not support `hw_review_wp7.md`
§2 F2 ("deeper circuits keep more signal than one α predicts") as a statement
about the α-scaled noise model. It holds only for the survival shortcut
S^α. See §2 below. The F2 sentence must not go into the paper as written.

---

## 0. Acceptance

`noise_model_check.py --fit-alpha --date 2026-10-25 --scale-only all` refits from
the existing cache (no simulation, 9 s). It reproduces
`hw_alpha_exec_2026-10-25/fit.json` exactly: all 4 subsets, every one of `alpha, ci,
ci_scaled, bootstrap_se, chi2_min, dof, n_pubs, loo, profile`. The file is not
rewritten (`--fit-alpha` now only compares an existing `fit.json`; `--force` to
replace). α̂ = 3.176, scaled CI [3.074, 3.348] (`A:variants.all.all.alpha`,
`.ci_scaled`).

## 1. Which errors carry α

| class scaled | α̂ [95 % CI, scaled] | χ²_min / dof | Δχ² vs all | LOO range | λ=1 / λ=3 alone | job 2, λ=1 RMS (without P-n3-M2-17) |
|---|---|---|---|---|---|---|
| all | 3.176 [3.074, 3.348] | 83.6 / 34 (2.46) | 0 | 3.128–3.208 | 3.179 / 3.209 | 0.060 (0.028) |
| 2q only | 4.055 [3.511, 4.462] | 92.6 / 34 (2.72) | +9.1 | 3.836–4.125 | 4.042 / 4.115 | 0.061 (0.012) |
| readout only | ≥ 32, at the grid edge | 1011.2 / 34 (29.7) | +927.6 | — | edge / edge | 0.181 (0.209) |

Keys: `A:variants.<class>.all.{alpha,ci_scaled,chi2_min,dof,chi2_red,loo_range}`,
`A:variants.<class>.delta_chi2_vs_all`, `A:variants.<class>.lambda1.alpha`,
`.lambda3.alpha`, `A:variants.<class>.out_of_sample_job2.{lambda1_rms,lambda1_rms_without_failure}`.

Scaling only the CZ errors fits job 1 nearly as well as scaling everything. It
needs a larger factor (4.06 against 3.18). It costs Δχ² = 9.1 on the same 35
PUBs and 34 dof. Like the full fit, one factor serves unfolded and folded
circuits (4.04 at λ = 1, 4.12 at λ = 3). Out of sample it predicts job 2 at
least as well. Per record at λ = 1 the deviations are −0.015, +0.010, +0.008 for
P-n2-M1-11, P-n3-M1-17, P-n3-M1-17_r1, against +0.011, +0.015, +0.044 for the
full model (`A:variants.<class>.out_of_sample_job2.per_record_lambda1[*].deviation`).
Both models miss P-n3-M2-17 (−0.121 and −0.109).

Readout alone cannot explain the device. The fit runs to the end of the
extended grid (α = 32, where 120 of the 312 readout entries of job 1's
calibration already sit at the 0.5 cap,
`hw_alpha_exec_readout_2026-10-25/counts_alpha_32.json:scale_info`). χ²/dof there is
29.7. As α grows further, more readouts reach the cap (a coin flip) and the
model can only move toward "everything random". The best it can do
leaves P-n2-M1-11 on job 2 off by +0.299. Grid note: the readout fit used the
paper's grid plus 12, 16, 24, 32 (`hw_alpha_exec_readout_2026-10-25/fit.json:alphas`).
The 2q fit used the paper's grid unchanged.

**Paper sentence.** "The excess is carried by the two-qubit gates. Scaling only
the CZ errors fits the first job almost as well (α̂ = 4.1 [3.5, 4.5], Δχ² = 9 on
35 circuits) and predicts the second as well. Scaling only the readout errors
cannot reproduce the data at any factor."

## 2. Depth dependence

Per record, α_r is fitted to its λ = 1 PUB, with its drift copy pooled (same
circuit). Job 1 uses `hw_alpha_exec_2026-10-25`. Job 2 uses
`hw_alpha_exec_insample_2026-10-28`, a new cache (see "Corrections" below). The
exponent is ln(kept)/ln(S) at α = 1, with a delta-method SE. 12 of the 15
records enter the trend (`A:depth.trend_ids`). Three are excluded
(`A:depth.rows[*].excluded_because`):

- P-n3-M1-17_r3 and P-n4-M1-10_r4 are over-rotated (r > r_opt). There the ideal
  state has rotated past the solution, and noise that damps the rotation can raise
  P(solution). So P(α) and "kept" are not a survival probability: the α̂ model
  gives kept 0.23 against S^α̂ = 0.023 (review §7).
- P-n3-M2-17 on day 2 is the out-of-sample failure, at random on every circuit.
  Its α_r sits at the grid edge (8).

| | slope per 100 2Q [95 % CI] | ≤ 230 2Q (n) | > 230 2Q (n) | deep − shallow [95 % CI] |
|---|---|---|---|---|
| α_r (noise model) | −0.091 [−0.301, +0.118] | 2.98 (6) | 2.82 (6) | −0.16 [−0.50, +0.17], z = −1.0 |
| exponent (S^α shortcut) | −0.196 [−0.280, −0.113] | 2.82 (6) | 2.32 (6) | −0.51 [−0.81, −0.20], z = −3.3 |

Keys: `A:depth.alpha_r_vs_2q.{slope_per_100_2q,slope_per_100_2q_ci}`,
`A:depth.exponent_vs_2q.*`, `A:depth.alpha_r_split.{shallow.mean,deep.mean,difference,difference_ci,z}`,
`A:depth.exponent_split.*`. Job 1 alone: α_r slope −0.157 [−0.417, +0.102]
(`A:depth.alpha_r_job1_only_vs_2q`). The 230 split comes from the review, after
the data were seen. It is descriptive, not a test.

The exponent falls with depth, as the review found. The per-record α of the full
noise model does not show a trend we can resolve: its slope interval contains
zero, and so does the shallow/deep difference. What dominates instead is the
scatter between records at any depth. The weighted trend has χ²/dof = 8.15
(`A:depth.alpha_r_vs_2q.chi2_red`), with α_r from 2.18 to 3.76
(`A:depth.rows[*].alpha_r`, rows in the trend). Example: P-n3-M1-17_r1 at 112 2Q has α_r = 3.76; its
day-2 transpilation (126 2Q) has 2.65. The full model already "knows" that a
deep circuit does not decay like S^α (idling, relaxation, readout, and saturation
near P_u). So the depth effect in the exponent is a failure of the
global-depolarising shortcut, not evidence against one α.

**Paper sentence.** "Fitted circuit by circuit, α varies more between circuits
(2.2–3.8, χ²/dof = 8) than with depth: the trend over 71–749 two-qubit gates is
−0.09 [−0.30, 0.12] per 100 gates. The gate-count shortcut S^α does degrade with
depth (exponent −0.20 [−0.28, −0.11] per 100 gates), because the device's decay
is not global depolarising." Supplement: Fig. `fig_alpha_depth`.

## 3. Residual structure at α̂, job 1, λ = 1

13 PUBs at α̂ = 3.176. RMS = 0.0181, largest |residual| 0.033, χ² = 61.7 on 12 dof,
χ²/dof = 5.14 (`A:residuals.{rms,max_abs,chi2,dof,chi2_red}`; dof = n − 1, the
convention of the λ = 1 fit). Group means (`A:residuals.groups.*`):

| group | records | mean residual | mean z |
|---|---|---|---|
| r = r_opt | 6 base records | +0.004 | +0.6 |
| n = 3, r < r_opt | P-n3-M1-17_r1 | −0.033 | −3.1 |
| n = 3, r > r_opt | P-n3-M1-17_r3 | −0.027 | −3.3 |
| n = 4, r < r_opt | P-n4-M1-10_r1, _r2 | +0.022 | +3.4 |
| n = 4, r > r_opt | P-n4-M1-10_r4 | −0.004 | −0.7 |
| drift copies | 2 | −0.003 | −0.3 |

The six r = r_opt records fit within shot noise on average. All of the excess
χ² sits in the iteration sweeps. The n = 3 off-optimal variants are below the
model and the n = 4 r = 1, 2 variants above it, as the review said. The review's
"RMS 0.017" is 0.018 here: residuals at α̂ = 3.176 from the cache's interpolation,
`A:residuals.rms`.

**Paper sentence.** "At α̂ the residuals of the unfolded circuits have RMS 0.018
(χ²/dof = 5.1). The excess comes from the iteration sweeps, which deviate by up
to 0.033 in opposite directions at n = 3 and n = 4, while the six circuits at r_opt agree with
the model within shot noise."

## 4. Day to day

P-n2-M1-11 (71 2Q both days, no qubit in common, `A:day_to_day.shared_records[0]`):
α_r = 3.025 [2.86, 3.18] on 2026-10-26 and 3.068 [2.95, 3.18] on 2026-10-28.
The difference is +0.043 [−0.147, +0.237] (bootstrap, B = 2000), z = 0.45, a
relative change of 1.4 % (`A:day_to_day.{alpha_r,alpha_r_ci_scaled,difference,difference_ci_bootstrap,z,relative_change}`).
Each day's α is relative to that day's execution calibration. So the rise in
P(solution) from 0.600 to 0.646 (`hw_2026-10-25/P-n2-M1-11.json` and
`hw_2026-10-28/P-n2-M1-11.json`, `metrics.solution_probability`) is accounted for
by the calibration change, not by α. The re-transpiled records move more: P-n3-M1-17 −0.12 (z −0.9),
P-n3-M1-17_r1 −1.11 (z −4.9) (`A:day_to_day.shared_records`). That is the
between-circuit scatter of §2, not a change of the device.

**One number.** For the same circuit on a new layout two days later, α moved by
+0.04 [−0.15, +0.24].

**Paper sentence.** "Refitted on the second day, relative to that day's
calibration, the same 71-gate circuit gives α = 3.07 [2.95, 3.18] against 3.03
[2.86, 3.18]: α moved by 0.04 [−0.15, 0.24] between days and layouts."

## 5. Layered error (run-day values, no service call)

Both execution calibrations frozen for the jobs (`calibration_exec_<date>.json`,
`raw_properties.general`) already contain IBM's layer fidelity `lf_4 … lf_100`,
with the chains in `general_qlists`. So the run-day value exists, and today's
value is not needed for the comparison. EPLG = 1 − LF_100^(1/99)
(`dgssp.backends.layer_fidelity`). Keys: `A:layered.days.<date>.*`.

| | job 1 (run 2026-10-26) | job 2 (run 2026-10-28) |
|---|---|---|
| LF_100 / EPLG | 0.5440 / 0.00613 | 0.5766 / 0.00555 |
| LF measured, age at run | 2026-10-25 05:47 −06:00, 18.3 h | 2026-10-27 10:18 −06:00, 9.0 h |
| median CZ error (device) | 0.00261 | 0.00253 |
| EPLG / median CZ | 2.35 | 2.19 |
| chain's own CZ errors as process infidelity (×5/4, product over 99 pairs) | 0.00428 | 0.00372 |
| EPLG / that | 1.43 | 1.49 |
| α̂_all × median CZ | 0.00828 | 0.00802 |
| α̂_2q × median CZ | 0.01057 | 0.01025 |

The device's own layered benchmark puts the error of a CZ in a layer at 2.2–2.4×
its isolated value. In consistent units (process infidelity, the same 99 pairs)
that is 1.4–1.5×. The α fits ask for 3.2× (all errors) or 4.1× (CZ only). So
layering and crosstalk, as IBM measures them on its best 100-qubit chain, explain
part of α, not all of it. EPLG (0.0061) is still below α̂ × median CZ (0.0083). Two
caveats. The LF chain is chosen for quality, and the job's own qubits had a lower
median CZ error (0.00215 on job 1, `A:layered.days.2026-10-25.job_qubits_cz_median`).
And EPLG carries the layer's single-qubit gates, which the comparison leaves out
of the isolated side. Both caveats push the true unexplained factor up, not down.

`describe_layer_fidelity.py` (you run it; read-only) will add today's value to
`paper/hardware/layer_fidelity_<today>.json`. That is **not** the run day's
calibration and should be quoted, if at all, only as a check that the API exposes
the same quantity.

**Paper sentence.** "IBM's layer-fidelity benchmark, measured 9–18 h before each
job, gives an error per layered gate of 0.0061 and 0.0055, 1.4–1.5 times what the
isolated CZ errors of the same chain imply. Layered operation explains part of the
excess the fit finds (×3.2 on all errors, ×4.1 on CZ errors alone), but not most of it."

## 6. What goes where

- Main text, model paragraph: §1 sentence (2q carries it, readout cannot) and §5
  sentence (layered error explains part). Together they answer "what is α?".
- Main text or supplement: §4 one number (α stable between days for the same
  circuit).
- Supplement: §2 figure and table, §3 residual table, §1 full table.
- Remove: "one α describes circuits up to ≈ 230 2Q; deeper circuits keep more
  signal than it predicts" (review §2 F2, outline). Replace it with the §2
  sentence.
- The "depth effect" argument for the ~60 s job (review §7: P-n4-M1-10_r2 tests
  "the depth effect") no longer holds as stated. The record still separates
  α = 1 from α̂, which is its main value.

## Corrections to the spec and what was changed to get here

1. **Job 2 had no α-grid cache**, so item 2 (job-2 rows) and item 4 needed new
   simulation (10 α × 14 PUBs, ~4 min). It lives in
   `hw_alpha_exec_insample_2026-10-28/`, **not** `hw_alpha_exec_2026-10-28/`. A
   `fit.json` under the latter name becomes job 2's in-sample model in
   `hw_aggregate.py` and replaces the out-of-sample prediction in its summary.
2. **`--fit-alpha` overwrote `fit.json`**, so "`--scale-only all` reproduces the
   paper's α̂" would have rewritten the paper's α file. It now refits, compares and
   leaves the file (`--force` to replace).
3. **`--fit-alpha` hard-coded λ ∈ {1, 3, 5}.** On the gentle grid "all" would
   have been λ = 1 only. Subsets now follow the scales the job ran. On the odd
   grid they are the same four as before (acceptance check above).
4. **`hw_aggregate.alpha_prediction_dirs`** matched any `hw_predict_*_x*_<date>`.
   With the 2q/readout predictions present it would have found three folders and
   dropped job 2's α model. It now skips `--scale-only` class folders. For the
   existing data the selection is unchanged.
5. `--alphas` now defaults to the stored fit's grid (the paper used 1–5, 6, 8,
   not `DEFAULT_ALPHAS`).
6. `1q` is selectable but not fitted. Its errors are ~7× smaller than CZ errors
   (median 3.6e−4 vs 2.6e−3, `hw_summary.json:survival.medians.alpha_1`).
