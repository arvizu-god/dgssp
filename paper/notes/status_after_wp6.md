# Project status after WP6

As of 2026-10-26. WP6 (the hardware run) is done: one job on ibm_fez, from one
clean commit, with every input frozen by hash, and the data committed
(`3c9cad9`). Nothing has been pushed yet.

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

| Instance | 2q gates (fez) | WP5 fake_marrakesh | ibm_fez noise model | Hardware |
|---|---|---|---|---|
| P-n2-M1-11 | 71 | 0.85 | 0.78 | **0.47** |
| P-n3-M2-17 | 194 | 0.64 | 0.56 | **0.14** |
| P-n3-M1-17 | 212 | 0.61 | 0.53 | **0.16** |
| P-n4-M2-02 | 495 | 0.37 | 0.22 | **0.02** |
| S-n4-g1-M2-00 | 466 | 0.24 | 0.19 | **0.01** |
| P-n4-M1-10 | 749 | 0.14 | 0.10 | **0.02** |

- Hardware signal ≈ (noise-model signal)³ on the three instances with signal
  left (0.78³ = 0.47, 0.56³ = 0.18, 0.53³ = 0.15): ibm_fez behaves as if each
  gate were about 3× noisier than its calibration says. Median calibrated 2q
  error is *lower* than fake_marrakesh's (2.6e-3 vs 3.3e-3), so the gap is
  noise the calibration does not capture. It is not calibration age: the
  probe, run 6 min after the 18:24 calibration, showed the same gap on
  P-n2-M1-11 (0.590 vs 0.837 predicted).
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
calibration, so simulation overstates how far both the algorithm and the
mitigation reach; an identifiability check is what keeps extrapolation from
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
   - Hardware: 0 of 75 fit blocks change, so no re-collect is needed. Dry
     run: only P-n2-M2-03 (M = N/2).
2. Optional second job in this window (516 s left, ends ≈ 2026-11-21): the same
   instances on a narrow partial-folding grid (λ = 1, 1.25, 1.5), the one test
   that could change a finding (ZNE at n = 3 on hardware). About 50–75 s.
3. **In progress: is the gap calibration age, and how big is it?**
   `paper/scripts/noise_model_check.py`, no QPU time.
   - `--fetch-calibration` (needs the token) freezes the calibration in force
     while the job ran (`target_history` at 06:05:15 UTC). It first checks that
     `target_history` at the plan stamp rebuilds the plan snapshot exactly.
     *To run.*
   - `--predict` re-simulates the 47 circuits on it. Accepted: on the plan
     calibration it reproduces the dry run's counts on all 47 PUBs.
   - `--fit-alpha` fits α, the factor on every calibrated error (T1/T2 fixed)
     at which the model matches the hardware P(solution). Against the
     **plan** calibration (preview; the paper number uses the execution-time
     one), on the 35 PUBs with signal:

     | fitted on | α | 95 % CI (scaled by √χ²/dof) | χ²/dof | leave-one-record-out |
     |---|---|---|---|---|
     | λ = 1 | 3.05 | [2.73, 3.42] | 5.1 | 2.97–3.10 |
     | λ = 3 | 3.10 | [2.73, 3.72] | 0.6 | 3.07–3.19 |
     | λ = 5 | 4.33 | [2.00, 8.97] | 1.6 | 3.04–5.03 |
     | all | 3.05 | [2.76, 3.33] | 2.5 | 2.97–3.10 |

     One α ≈ 3 explains the unfolded and the folded circuits alike. So the
     extra noise behaves like a uniform rescaling of the calibrated errors,
     which is the "signal³" rule of the results above, now measured.
4. Hardware tables and figures for the paper (`aggregate.py` covers only the
   simulation data).
5. Carried over from WP5: the six TEXT TO WRITE paragraphs; the Mitiq section;
   Zenodo deposit, then delete the bucket (on Arturo's OK).
6. Untracked on purpose: `paper/hardware/account_snapshot.json` (raw account
   data, CRN) and `paper/data/hw_dryrun_2026-10-25_collect/` (a regenerable
   duplicate from the acceptance check).
