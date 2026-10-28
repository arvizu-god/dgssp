# Project status after WP6

As of 2026-10-28. WP6 (the hardware run) is done: the job of record on
ibm_fez, from one clean commit, with every input frozen by hash, and the data
committed (`3c9cad9`); a second, smaller job on the gentle grid (Open item 2,
data `09b9c46`, Table S2). About 465 s of QPU time are left in the window
(ends ≈ 2026-11-21). Nothing has been pushed yet.

WP1–WP5: see `status_after_wp3.md`, `status_after_wp4.md`, `status_after_wp5.md`.
The tool is `paper/scripts/hardware_batch.py` (`--plan`, `--dry-run`,
`--probe` / `--probe-report`, `--submit`, `--collect`); the gate is
`paper/hardware/CHECKLIST_template.md`.

## The run of record

| | |
|---|---|
| Device | ibm_fez (Heron r2, 156 q), Open plan, instance `experiments`; job mode |
| Code | commit `9c5aa67`, clean; every record stamped with it |
| Job | `dbfeqr7gdj9s73bbgh6g`: created 06:04, ran 06:05–06:14 UTC, 2026-10-26 |
| Circuits | 47 PUBs × 4096 shots = 192 512 shots: 10 instances × λ ∈ {1, 3, 5}; 5 sweep variants (P-n3-M1-17 r = 1, 3; P-n4-M1-10 r = 1, 2, 4) × 3 scales; 2 drift repeats (scale-1 copies of P-n2-M1-11 and P-n3-M2-17, sent as the last PUBs) |
| Transpilation | once per instance on the live target, opt level 3, SABRE seeds 0–31; sweep variants pinned to the r_opt layout; folded with `fold_isa` + `rebase_to_backend`, sent as-is |
| Suppression | gate twirling (32 × 128, `active-accum`); DD **off**; no measurement twirling; no readout mitigation (SamplerV2 has none) |
| Calibration | plan and dry run: 2026-10-25 18:24:25 −06:00 (`calibration_2026-10-25.json`). At submit the device had recalibrated (23:36:48); the live ISA pre-flight passed again before submission. |
| QPU usage | **75 s** main job + 7 s probe. Planning estimate 275.7 s (conservative ×2); IBM's rule of thumb 69 s. Left in the window: 516 s. |
| Records | 17 in `paper/data/hw_2026-10-25/` (executor `hardware`) + the raw result; dry run (executor `noise_model_prediction`) in `paper/data/hw_dryrun_2026-10-25/` |

Acceptance (the WP6 brief): `--plan` and `--dry-run` completed on the live
device; the estimate was within budget (275.7 s ≤ 296 s cap); `--collect` on the
dry run's fake job reproduced all 17 records identically (every field except
provenance).

Not reused: no thesis numbers; no fake_marrakesh seeds (the layout was searched
again on ibm_fez).

## Results

### Signal kept: simulation vs model vs hardware

"Signal kept" = (P_sol − P_uniform) / (P_ideal − P_uniform) at λ = 1; 1 is
perfect, 0 is random guessing. Same 10-instance subset throughout (the
shallowest instance of each n × M cell, plus the edge instance).

| Instance | 2q gates (fez) | WP5 fake_marrakesh | ibm_fez model, plan calibration | ibm_fez model, execution calibration | Hardware |
|---|---|---|---|---|---|
| P-n2-M1-11 | 71 | 0.85 | 0.78 | 0.79 | **0.47** |
| P-n3-M2-17 | 194 | 0.64 | 0.56 | 0.58 | **0.14** |
| P-n3-M1-17 | 212 | 0.61 | 0.53 | 0.54 | **0.16** |
| P-n4-M2-02 | 495 | 0.37 | 0.22 | 0.24 | **0.02** |
| S-n4-g1-M2-00 | 466 | 0.24 | 0.19 | 0.20 | **0.01** |
| P-n4-M1-10 | 749 | 0.14 | 0.10 | 0.10 | **0.02** |

- Hardware signal ≈ (noise-model signal)³ on the three instances with signal
  left (0.78³ = 0.47, 0.56³ = 0.18, 0.53³ = 0.15): ibm_fez behaves as if each
  gate were about 3× noisier than its calibration says. Median calibrated 2q
  error is *lower* than fake_marrakesh's (2.6e-3 vs 3.3e-3), so the gap is
  noise the calibration does not capture. It is not calibration age: the
  probe, run 6 min after the 18:24 calibration, showed the same gap on
  P-n2-M1-11 (0.590 vs 0.837 predicted), and the calibration in force while
  the job ran predicts slightly *more* signal, not less. Measured: α = 3.18
  (95 % CI 3.07–3.35), see Open item 3.
- The algorithm works at n = 2 (0.60 vs 0.25 random), is marginal at n = 3
  (0.26–0.35, 1.4–2.1× random) and is indistinguishable from random at n = 4
  (0.077 / 0.142 / 0.131 vs 0.0625 / 0.125 / 0.125), for the *easiest* n = 4
  instances of the family.

### ZNE (odd grid, exponential with the P_uniform asymptote)

| Record | P_sol λ = 1 / 3 / 5 | ZNE | Mitiq (offline, same counts) | Ideal |
|---|---|---|---|---|
| P-n2-M1-11 | 0.600 / 0.338 / 0.269 | **0.953 [0.890, 1.026]**, resolved | 0.962 | 1.000 |
| P-n3-M1-17_r1 | 0.317 / 0.155 / 0.124 | identified, not resolved (reports 0.317) | 0.613 | 0.781 |
| P-n3-M2-17 | 0.353 / 0.243 / 0.243 | not identifiable (reports 0.353) | **3.547** | 1.000 |
| P-n3-M1-17 | 0.257 / 0.128 / 0.126 | not identifiable (reports 0.257) | 0.673 | 0.945 |
| P-n4-M1-10_r2 | 0.096 / 0.059 / 0.054 | not identifiable (reports 0.096) | 0.831 | 0.908 |

- ZNE works where signal survives the folding (n = 2: 0.60 → 0.95, χ²_red 0.13;
  Mitiq agrees).
- At n ≥ 3 folding to λ = 3 already drives P_sol to P_uniform (P-n3-M1-17:
  0.257 → 0.128, random 0.125), leaving one informative point: the odd grid
  cannot resolve anything. That is WP5's partial-folding result, sharper on
  hardware.
- Where our estimator declines, Mitiq's extrapolation (offline on the same
  dgssp-folded counts; label `dgssp_isa_folds`) returns 3.547 (a probability
  above 1) and 0.831 from counts at the random level. The identifiability
  guard is what prevents confident wrong answers.

### Other

- Drift: none. P-n2-M1-11 first 0.5996 vs last 0.6094 (z = +0.90, p = 0.37);
  P-n3-M2-17 0.3533 vs 0.3533 (z = 0).
- M = 0 pathology reproduces on hardware: D_TV raw 0.021 / 0.035 → pointwise
  clip 0.748 / 0.811, simplex 0.875 / 0.937 at n = 3 / 4 (simulation:
  clip 0.624 / 0.931, simplex 0.875 / 0.937). Benign at n = 2.
- Iteration sweep: fewer iterations win on hardware. n = 3: r = 1, 2, 3 give
  0.317, 0.257, 0.146 (ideal 0.781, 0.945, 0.330); n = 4: 0.114, 0.096, 0.077,
  0.063 for r = 1–4, reaching random by r = 4.
- M = N/2 (P-n2-M2-03) sits at 0.49 ≈ P_uniform, as it must; uninformative.

### The finding, in one paragraph

On ibm_fez the Draper-oracle Grover search for Subset Sum returns a real answer
only for the smallest instances (n = 2, marginally n = 3: circuits up to about
200 two-qubit gates). ZNE recovers the answer while signal survives, but the
device is about 3× noisier per gate than a noise model built from its own
calibration (every calibrated error × 3.18, 95 % CI 3.07–3.35, fitted to the
hardware P(solution)), so simulation overstates how far both the algorithm and
the mitigation reach; an identifiability check is what keeps extrapolation from
reporting confident, wrong numbers.

## Fixed on the way

- Calibration snapshots now keep bare device-specific operations as IBM's
  converter builds them (`measure_2`, `reset_2`, `xslow`; name, arity,
  parameter names), and still refuse anything with meaning beyond its name.
- qiskit-ibm-runtime 0.49.0 bug: `TwirledSliceSpanV2.filter_by_pub` raises
  `IndexError`; shot counts are read from `span.mask()` instead (regression
  test included).
- A failed `--plan` leaves no orphan files; `--submit` refuses a dry run of an
  older manifest. A hand edit of the manifest (`status_at_plan.pending_jobs`
  0 → 114) was caught by the hash check and reverted to the exact original
  bytes before submission.
- DD off by default; offline Mitiq block on every hardware record.

## Open

1. **Done (2026-10-26): "resolved" now means on the ideal's side of random
   guessing** (rule v2, `dgssp.mitigation.zne.RESOLVED_RULE`). Rule v1 was
   two-sided, and took the informative side from λ = 1. So a series pushed
   below P_uniform counted as resolved and identified (n = 6 planned: 9 fits
   extrapolated to ≈ 0 with ideal ≈ 1). M = N/2 instances also counted, though
   they have nothing to resolve. `aggregate.py` re-derives the flags from the
   stored observations (records untouched; v1 kept in `*_v1` columns).
   - 39 of 1123 WP5 fits change, all from "extrapolation" to "λ = 1"; none
     the other way.
   - Resolved fractions: planned 0.436 → 0.402, odd 0.110 → 0.089, odd7
     0.096 → 0.066.
   - Median signal recovery and the model ranking are unchanged.
   - Fractions are now also given over the instances **with signal**
     (`M = 0` and `M = N/2` excluded; 340 of 463 on planned / odd, 130 of
     197 on odd7). Resolved: planned 0.547, odd 0.121, odd7 0.100. At n = 2
     the all-instance 0.43 becomes 1.00 (20 of 20); n = 3 planned 0.98, n = 6
     planned 0.20. The with-signal fraction is the one to quote for ZNE.
   - Hardware: 0 of 75 fit blocks change, so no re-collect is needed. Dry
     run: only P-n2-M2-03 (M = N/2).
2. **Done (2026-10-28): second job, gentle grid. n = 3 resolved; P-n3-M2-17
   fell to random.** Job `dbgkqq8te5oc738icnfg` on ibm_fez, submitted
   2026-10-28T01:18Z at 4f43803, 51 s charged (estimate 224.9 s); about
   465 s left. Results in `paper/data/hw_2026-10-28`, tables in
   `hw_*_2026-10-28` (the paper's Tables 3 and 4 stay on job 1).

   | Record | 2Q (job 1 / job 2) | Ideal | Odd grid, job 1 | Gentle grid, job 2 | Predicted at α̂ |
   |---|---|---|---|---|---|
   | P-n2-M1-11 | 71 / 71 | 1.000 | 0.953 [0.890, 1.026] resolved | 0.981 [0.928, 1.049] resolved | 0.930 [0.874, 0.985] |
   | P-n3-M1-17 (r = 2) | 212 / 235 | 0.945 | not identifiable | 0.821 [0.660, 1.089] resolved | 0.670 [0.507, 0.862] |
   | P-n3-M1-17_r1 | 112 / 126 | 0.781 | identified, not resolved | 0.730 [0.633, 0.831] resolved | 0.609 [0.508, 0.719] |
   | P-n3-M2-17 | 194 / 184 | 1.000 | not identifiable | not identifiable (P_sol 0.246, random 0.25) | 0.806 [0.613, 1.107] |

   - Signal kept at λ = 1 (job 2 / job 1 / predicted): P-n2-M1-11 0.53 /
     0.47 / 0.51; P-n3-M1-17 0.15 / 0.16 / 0.13; r = 1 0.31 / 0.29 / 0.24;
     P-n3-M2-17 −0.01 / 0.14 / 0.14. On the three M = 1 records the device
     did as well as or better than the α̂ model; both ZNE intervals for n = 3
     contain the ideal.
   - P-n3-M2-17 is flat: at λ = 1 all 8 outcomes have 1313–1853 of 12 288
     counts, and each measured bit is 0.46–0.51 ones, so no single stuck
     qubit. Both drift repeats agree (0.246, 0.248; z = +0.31). The planning
     calibration (18:12 local) shows nothing unusual on its layout [88, 92,
     78, 89, 91, 98, 90] (CZ errors 0.0016–0.0028, T1 92–156 µs); P-n2-M1-11
     shares qubits 90, 91, 92, 98 and was fine. The device recalibrated at
     19:08, between plan and submit; that calibration was not saved. The
     cause is not identified, so report it as a failure the calibration
     model did not predict.
   - Reading: the folding grid decides whether n = 3 resolves (odd grid no,
     gentle grid yes, same instances); a layout that the calibration rates
     as fine can still be at random on the day.
   - The aggregator's model column is empty for this date (it reads
     `hw_predict_exec_<date>`, not `hw_predict_plan_x3.176_<date>`); the
     predicted column above is from the latter.
   - **In the paper: Table S2** (`tex/hw_table_grids.tex`, from
     `hw_aggregate.py --grids 2026-10-28 --grids-prediction
     paper/data/hw_predict_plan_x3.176_2026-10-28`). Job 1 stays the job of
     record for Tables 3 and 4 and every hardware figure. The footer's
     "before the job ran" is read off the timestamps (prediction 01:04:36
     UTC, job 01:18:35 UTC).
   - Sentence for the C2 discussion (draft): "With the same instances on the
     same device, a gentler fold grid, λ ∈ {1, 1.25, 1.5}, resolved both
     n = 3, M = 1 records that the odd grid λ ∈ {1, 3, 5} could not, with
     intervals that contain the ideal (Table S2); the third n = 3 record was at
     random on that day although the calibration rated its layout as
     ordinary."

   The plan, as written before submission:

   Can ZNE resolve n = 3
   on hardware when the folding stays gentle (λ = 1, 1.25, 1.5)? On the odd
   grid λ = 3 already reached random.
   - Circuits: P-n2-M1-11 (control: resolved on the odd grid), P-n3-M1-17 and
     its r = 1 variant, P-n3-M2-17; × 3 scales, plus the 2 drift repeats = 14
     PUBs × 12 288 shots. n = 4 is left out: at r_opt it is at random on
     hardware, and its r = 1 variant (233 2Q) is above the band.
   - Flags: `--only P-n2-M1-11 P-n3-M1-17 P-n3-M2-17 --sweep-iterations 1`
     (`--sweep-iterations` new; the job of record used neither).
   - Budget: estimate 224.9 s (job 1's measured per-shot time and overhead,
     × 2) against the 258 s cap (half of 516 s); job 1's actual rate gives
     about 67 s charged.
   - Prediction before sending (offline rehearsal on fake_fez, simulated on
     the 2026-10-25 execution calibration with every error × 3.176):
     P-n3-M1-17_r1 resolved, 0.686 [0.576, 0.840] (ideal 0.781); P-n3-M1-17
     0.725 [0.472, 1.263] (ideal 0.945); P-n2-M1-11 0.961 [0.886, 1.046];
     P-n3-M2-17 "resolved" but uninformative, 1.551 [0.569, 7.626]. At 8192
     shots the intervals were about 20 % wider. So if the device is as on
     2026-10-25, the gentle grid resolves n = 3 where the odd grid could not.
     The live plan's own prediction at α̂ is the one to commit before
     submission.
   - `hw_aggregate.py` now aggregates the job of record (2026-10-25) unless
     `--date` says otherwise; another date writes `hw_*_<date>` files beside
     the paper's.
3. **Done (2026-10-26): the gap is not calibration age, and it is α ≈ 3.2.**
   `paper/scripts/noise_model_check.py`, no QPU time; every output stamped
   with the clean commit `687c5a1`.
   - `--fetch-calibration` froze the calibration in force while the job ran
     (`target_history` at 06:05:15 UTC → the 23:36:48 −06:00 calibration;
     `calibration_exec_2026-10-25.json`, report `…_fetch.json`). Method check
     passed: `target_history` at the plan stamp rebuilds the plan snapshot
     with 0 differences. ISA check of the 47 circuits: ok. On the 25 qubits the
     job used, CZ errors are unchanged (median 2.15e-3), sx and readout moved
     a little (readout median 8.3e-3 → 7.2e-3).
   - `--predict` (`hw_predict_exec_2026-10-25/`): the execution calibration
     predicts slightly **more** signal than the plan one (P-n2-M1-11: 0.79 vs
     0.78; hardware 0.47). So the newer calibration widens the gap. Accepted
     beforehand: on the plan calibration it reproduces the dry run's counts on
     all 47 PUBs.
   - `--fit-alpha` fits α, the factor on every calibrated error (gates,
     readout, reset; T1/T2 and durations fixed; capped at 0.5), at which the
     model matches the hardware P(solution). 35 PUBs with signal (M = 0 and
     M = N/2 excluded), grid α ∈ {1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8}, 500
     bootstrap resamples. 95 % CI scaled by √(χ²/dof) where χ²/dof > 1:

     | fitted on | PUBs | α, execution calibration (paper) | α, plan calibration | χ²/dof | leave-one-record-out (exec) |
     |---|---|---|---|---|---|
     | λ = 1 | 13 | **3.18** [2.97, 3.46] | 3.05 [2.73, 3.42] | 5.1 | 3.13–3.21 |
     | λ = 3 | 11 | 3.21 [2.82, 3.75] | 3.10 [2.73, 3.72] | 0.5 | 3.17–3.32 |
     | λ = 5 | 11 | 4.27 [2.13, ≥ 8] | 4.33 [2.00, ≥ 8] | 1.6 | 2.61–5.88 |
     | all | 35 | **3.18** [3.07, 3.35] | 3.05 [2.76, 3.33] | 2.5 | 3.13–3.21 |

     - One α ≈ 3.2 explains the unfolded and the folded circuits alike. So the
       extra noise behaves like a uniform rescaling of the calibrated errors,
       which is the "signal³" rule of the results above, now measured.
     - λ = 5 sets only a lower limit: at λ = 5 most PUBs are near random, and
       the profile does not rise past the CI threshold before the grid ends
       (α = 8). The script's `ci_scaled` prints 8.9 there, a widened width
       beyond the grid; quote it as "≥ 8".
     - χ²/dof = 5.1 at λ = 1: a single α does not fit every instance within
       shot noise (instance-to-instance scatter, e.g. different qubits). The
       scaled CI absorbs it; the leave-one-record-out range shows no single
       record drives α.
4. **Done (2026-10-28): hardware tables and figures.** `hw_aggregate.py` (tables,
   `hw_summary.json`) and `make_figures.py --set hw`, from the committed files
   alone; flags under rule v2.
   - Done: Table 3 (run configuration, `tex/hw_table_config.tex`), Table 4
     (results per record, `tex/hw_table_results.tex`); Fig 2 worked example
     (`fig_hw_example`), Fig 5 signal kept vs two-qubit gates with the
     WP5 cloud, the device model and the model at α̂ (`fig_hw_signal`), Fig 6
     folding, one record per status (`fig_hw_folding`), Fig 7 ZNE vs Mitiq
     (`fig_hw_zne`), Fig 9 sweep with hardware points (`fig_hw_sweep`), Fig 10
     M = 0 with hardware points (`fig_hw_dtv`).
   - Fixed on the way: the sweep figure drew the fit's interval (up to 820)
     beside a λ = 1 fallback, and took the flags as stored (rule v1); it now
     re-derives them and draws an interval only for a reported extrapolation.
     The D_TV figure's legend no longer widens one panel.
   - Done: Fig 4 (`fig_resources`, `resource_scaling.py --figure-only`).
     Panel (a) is now 2Q count vs n, one line per b, on fake_fez (the job's
     device twin), with the **hardware-limit band** and the hardware records
     marked by status. Panel (b) keeps the three-device comparison; panel (c)
     outlines the cells whose one-iteration circuit fits under the band.
     The band is read off the statuses, no threshold of our own
     (`hw_aggregate.limit_band`, in `hw_summary.json`): from the largest
     resolved circuit (P-n2-M1-11, **71** 2Q) to the next larger one that is
     not identifiable (P-n3-M2-17, **194** 2Q). One exception, listed rather
     than moving the edges: P-n4-M2-02 (495 2Q) is identified, not resolved.
     The fake_fez grid matches the hardware cell by cell (82 vs 71 at
     (n, b) = (2, 3), 106 vs 112, 212 vs 194, 251 vs 233), so only
     (2, 3), (2, 4) and (3, 3) fit under the band at one iteration.
   - Done: Table 2 (`tex/resources_table.tex`). Every closed form re-checked
     from the stored records: 144/144 on qubits, CP, P, H, X, MCZ and the
     ancilla-free 2Q reference. Measured factors: default synthesis 0.63–1.00×
     the reference (median 0.78), transpiled 1.25–2.08× (median 1.64),
     negating an item −15 to 0 2Q (median 0), guard bit 12–44 CP and 29–168
     2Q. Then the 2Q counts of nine cells (the hardware widths and the grid's
     corners) on the three devices.
   - Figure PDFs no longer carry a creation date, so re-running a figure
     script changes a PDF only when the figure changed.
   - Done: Fig 1 (`fig_circuit`: one iteration ×r, the item adder in the
     Fourier basis, the oracle, the diffuser; mirrors
     `DGSSPSolver.build_circuit`) and Fig 3 (`fig_pipeline`: instances →
     circuit → transpile once → fold the ISA circuit → three executors
     through SamplerV2 → three estimators on the same counts → tables and
     figures from `data/`). TikZ sources in `tex/diagrams/`, built by
     `build_diagrams.py`.
   - Every figure and table on the agreed list now exists. Main text:
     Figs 1–10, Tables 1–4; supplement: S1 (`fig_sim_models`), S2
     (`hw_table_grids`, odd grid vs gentle grid, added 2026-10-28).
5. Carried over from WP5: the six TEXT TO WRITE paragraphs; the Mitiq section;
   Zenodo deposit, then delete the bucket (on Arturo's OK).
6. Untracked on purpose: `paper/hardware/account_snapshot.json` (raw account
   data, CRN) and `paper/data/hw_dryrun_2026-10-25_collect/` (a regenerable
   duplicate from the acceptance check).
