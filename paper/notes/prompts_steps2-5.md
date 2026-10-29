# Claude Code prompts: steps 2–5 (post-WP7 checks, no QPU)

Written 2026-10-28. One Claude Code session per prompt. Each prompt below is
self-contained (shared preamble + task in one block): copy the whole block. Open Claude Code at the repo root.

**Order and conflicts**

| Prompt | Step | Edits | Run after |
|---|---|---|---|
| A | 2. Correct the notes | `paper/notes/`, docstrings, new `check_claims.py` | — (first) |
| B | 3. ZNE theory at scale | `mitigation/zne.py` (new functions only), new script, figure | A |
| C | 4. α robustness | `backends.py`, `noise_model_check.py`, new script | A (can run in parallel with B) |
| D | 5. Values above 1 | `mitigation/zne.py`, `aggregate.py`, `hw_aggregate.py` | B (both touch `zne.py`) |

Commit after each session, before starting the next.

---

The shared preamble is repeated at the top of every block. If you change it, change all four.

---

## Prompt A — Step 2: correct the notes

```
PROJECT CONTEXT — Paper A (Draper–Grover Subset Sum on heavy-hex hardware)

I am Arturo Arvizu. I am turning my MSc thesis (Grover search for Subset Sum
with a Draper QFT-adder oracle) into a journal paper (EPJ Quantum Technology or
Quantum Information Processing). The arXiv deadline is dropped: correctness
first. Repo: this directory (package `dgssp`, src layout, Qiskit 2.x,
qiskit-ibm-runtime 0.49, Aer, Mitiq 1.1.0). Windows + PowerShell, conda env
`dgssp-paper`, run through `conda run -n dgssp-paper ...`.

Current state:
- WP0–WP7 done. HEAD 9dc971c (2026-10-27).
- Hardware (ibm_fez, Heron r2):
  - job of record dbfeqr7gdj9s73bbgh6g, ran 2026-10-26 UTC (folder label
    2026-10-25 = plan date), odd grid λ ∈ {1,3,5}, 47 PUBs × 4096 shots;
  - job 2 dbgkqq8te5oc738icnfg, ran 2026-10-28 UTC, gentle grid
    λ ∈ {1,1.25,1.5}, 14 PUBs × 12 288 shots.
  - ~465 s of QPU left until ≈ 2026-11-21. Nothing in this session uses it.
- Simulation: 463 instances, fake_marrakesh, grids odd / odd7 / planned
  (paper/data/sim_*).
- α̂ = 3.176 [3.07, 3.35] (every calibrated error × α), fitted on job 1.

Read first (untracked notes written in chat on 2026-10-28, treat as the spec):
- paper/notes/hw_review_wp7.md   — number fixes and overclaims in hw_results.md
- paper/notes/paper_outline.md   — revised contributions and paper outline
- paper/notes/zne_theory.md      — theory of the scalar ZNE estimator, the
                                   gate, the grid rule, the pointwise failure

Revised contributions:
  C1 Resource model of the Draper-oracle Grover circuit for signed SSP
     (exact counts, heavy-hex cost, map of what fits under the hardware limit).
  C2 Hardware failure map + a predictive noise model (calibration with every
     error × α̂, fitted on one day, tested on another).
  C3 When nonlinear ZNE can be trusted for a sampling algorithm: scalar
     observable with a known asymptote, a resolution gate, a grid rule;
     why pointwise extrapolation of the distribution fails.

Rules:
1. No QPU. Never run --submit, never create a job, never read the IBM token.
   Anything that contacts the IBM service (even read-only) is a command I run.
2. Raw records in paper/data/hw_*/ and paper/data/sim_*/ are never edited.
   New results go to new files, stamped with _common.record_provenance().
3. One place per concept (encoding.py, decoding.py, runtime.sample_counts,
   mitigation/zne.py, metrics.py, backends.py). Extend, don't duplicate.
   Behaviour changes go behind a module constant (the RESOLVED_RULE pattern),
   with the old behaviour kept and selectable.
4. Every number written into a note comes from a file a script wrote. No
   hand-typed numbers. Each claim in a note names its file and key.
5. Seeds fixed (bootstrap seed 0, B = 2000 unless stated). pytest, ruff and
   mypy green before you hand over.
6. Direct, technically dense, no validation. If my request is wrong or
   underspecified, say so and propose the fix before writing code.
7. End with (a) the exact PowerShell commands I run, (b) the acceptance check
   I should see, (c) what to paste back, (d) a proposed commit message.

----------------------------------------------------------------

TASK: Step 2 — bring the hardware notes in line with the data. No new
analysis, no estimator changes.

0. git add the three untracked notes (hw_review_wp7.md, paper_outline.md,
   zne_theory.md) as they are. Do not rewrite them; if you find an error in
   one, list it and stop for that item.

1. New script paper/scripts/check_claims.py. It recomputes, from the committed
   data only, every quantitative claim listed in hw_review_wp7.md §1 (rows 1–10)
   and §2 (items A–K and F2), and prints one row per claim:
   claim id | file:section | value as written | recomputed | OK / MISMATCH.
   It writes paper/data/claims_check.json. It must reproduce, among others:
   - the Mitiq |Δ| values 0.010 and 0.004;
   - the M = 0 ratios 35× / 41×;
   - 12.8σ (one copy each day) for P-n3-M2-17;
   - n = 3 planned-grid median recovery 0.886;
   - pointwise D_TV raw → clip for all 11 unresolved records with signal
     (5 lower, 6 higher);
   - the strict rule's 19 of 340 planned-grid values outside [0, 1] (max 4.17);
   - identifiable_agreement 3/4 at both α = 1 and α̂ on job 2;
   - the exponents ln(kept)/ln(S) with their delta-method standard errors
     (P-n4-M1-10_r2 ≈ 2.01 ± 0.08, P-n4-M1-10 ≈ 1.63 ± 0.11);
   - the per-record λ = 1 deviations at α̂ on job 2 and their RMS (0.060);
   - commit 09b9c46's UTC time vs the job start (01:18:35 UTC).

2. Edit paper/notes/hw_results.md and paper/notes/status_after_wp6.md so every
   item in hw_review_wp7.md §1–§2 is fixed, using the recomputed numbers from
   claims_check.json. In particular:
   - §1: count distinct circuits, not PUBs; report per-record deviations and
     RMS instead of mean + sign test for job 2; say which job the "within
     0.008 of the shot floor" applies to; replace "committed before" with
     "timestamped".
   - §2: rewrite the pointwise paragraph as "coin toss outside resolved fits"
     with the 5/6 count; state that the strict rule still reports 19 values
     above 1 in simulation.
   - §3: drop "upper bound" for S; drop the "correctly predicted the gentle
     grid" clause; say the α̂ band is in-sample and that any threshold in
     (112, 194] gives 10/11; add the depth effect (F2); move the cube rule to a
     footnote.
   - §6: delete null #9 (different circuits).
   - status_after_wp6.md: n = 4 is "detectable, useless for search", not
     "indistinguishable from random".
   - Add a final section "Corrections log (2026-10-28)" to hw_results.md:
     one line per change, with the claim id from check_claims.py.

3. Docstrings only: in mitigation/zne.py (plan_scales) and wherever
   survival_estimate is documented, replace "upper bound" with "estimate,
   optimistic because it ignores idling, crosstalk and leakage". No code change.

4. paper/notes/sim_results.template.md: replace each TEXT TO WRITE marker
   with "see hw_results.md §7", then regenerate sim_results.md with
   aggregate.py so the template and the filled copy agree.

Acceptance:
- check_claims.py ends with 0 MISMATCH against the *edited* notes.
- git diff touches only paper/notes/, paper/scripts/check_claims.py,
  paper/data/claims_check.json and docstrings. No file under paper/data/hw_*
  or paper/data/sim_* changes.
- pytest, ruff, mypy green.
```

---

## Prompt B — Step 3: test the ZNE theory at scale

```
PROJECT CONTEXT — Paper A (Draper–Grover Subset Sum on heavy-hex hardware)

I am Arturo Arvizu. I am turning my MSc thesis (Grover search for Subset Sum
with a Draper QFT-adder oracle) into a journal paper (EPJ Quantum Technology or
Quantum Information Processing). The arXiv deadline is dropped: correctness
first. Repo: this directory (package `dgssp`, src layout, Qiskit 2.x,
qiskit-ibm-runtime 0.49, Aer, Mitiq 1.1.0). Windows + PowerShell, conda env
`dgssp-paper`, run through `conda run -n dgssp-paper ...`.

Current state:
- WP0–WP7 done. HEAD 9dc971c (2026-10-27).
- Hardware (ibm_fez, Heron r2):
  - job of record dbfeqr7gdj9s73bbgh6g, ran 2026-10-26 UTC (folder label
    2026-10-25 = plan date), odd grid λ ∈ {1,3,5}, 47 PUBs × 4096 shots;
  - job 2 dbgkqq8te5oc738icnfg, ran 2026-10-28 UTC, gentle grid
    λ ∈ {1,1.25,1.5}, 14 PUBs × 12 288 shots.
  - ~465 s of QPU left until ≈ 2026-11-21. Nothing in this session uses it.
- Simulation: 463 instances, fake_marrakesh, grids odd / odd7 / planned
  (paper/data/sim_*).
- α̂ = 3.176 [3.07, 3.35] (every calibrated error × α), fitted on job 1.

Read first (untracked notes written in chat on 2026-10-28, treat as the spec):
- paper/notes/hw_review_wp7.md   — number fixes and overclaims in hw_results.md
- paper/notes/paper_outline.md   — revised contributions and paper outline
- paper/notes/zne_theory.md      — theory of the scalar ZNE estimator, the
                                   gate, the grid rule, the pointwise failure

Revised contributions:
  C1 Resource model of the Draper-oracle Grover circuit for signed SSP
     (exact counts, heavy-hex cost, map of what fits under the hardware limit).
  C2 Hardware failure map + a predictive noise model (calibration with every
     error × α̂, fitted on one day, tested on another).
  C3 When nonlinear ZNE can be trusted for a sampling algorithm: scalar
     observable with a known asymptote, a resolution gate, a grid rule;
     why pointwise extrapolation of the distribution fails.

Rules:
1. No QPU. Never run --submit, never create a job, never read the IBM token.
   Anything that contacts the IBM service (even read-only) is a command I run.
2. Raw records in paper/data/hw_*/ and paper/data/sim_*/ are never edited.
   New results go to new files, stamped with _common.record_provenance().
3. One place per concept (encoding.py, decoding.py, runtime.sample_counts,
   mitigation/zne.py, metrics.py, backends.py). Extend, don't duplicate.
   Behaviour changes go behind a module constant (the RESOLVED_RULE pattern),
   with the old behaviour kept and selectable.
4. Every number written into a note comes from a file a script wrote. No
   hand-typed numbers. Each claim in a note names its file and key.
5. Seeds fixed (bootstrap seed 0, B = 2000 unless stated). pytest, ruff and
   mypy green before you hand over.
6. Direct, technically dense, no validation. If my request is wrong or
   underspecified, say so and propose the fix before writing code.
7. End with (a) the exact PowerShell commands I run, (b) the acceptance check
   I should see, (c) what to paste back, (d) a proposed commit message.

----------------------------------------------------------------

TASK: Step 3 — implement and test the theory in paper/notes/zne_theory.md on
all 463 simulated instances and both hardware jobs. Do not change any
existing estimator, flag or default. Add only.

Background (from zne_theory.md): under depolarising noise scaled by λ,
P(λ) = P_u + (P(0) − P_u) e^{−cλ} exactly, with P_u known. Our scalar
estimator learns c from the folded points. In chat a delta-method SE matched
the hardware bootstrap on the 4 resolved fits (0.035/0.035, 0.032/0.031,
0.053/0.050, 0.107/0.109). Every large raw hardware value equalled
P_u + (P(1) − P_u)·e^{EXP_RATE_MAX}. The optimal spacing of a second scale is
Δλ ≈ 1/c. Four hardware fits are not evidence; this task is.

1. Library (mitigation/zne.py, new section "Error propagation of the
   fixed-asymptote exponential"):
   - exp_fixed_se(scales, observed, shots, asymptote) -> dict:
     weighted fit (reuse _exp_fixed_many), linearised covariance of (b, c),
     SE of P(0) = a + b, and the "known-F" SE = σ1·e^{c·λ1} and their ratio.
   - predicted_se(scales, survival, signal, p_uniform, shots) -> float:
     the same SE evaluated on the model's expected observations.
   - optimal_step(c) -> float: argmin over Δ of the 2-point relative error
     sqrt((1+1/Δ)² + e^{2cΔ}/Δ²) (numerical, not the 1/c approximation;
     return both).
   - plan_scales_spacing(survival, signal, shots, candidates, k, p_uniform,
     n_points) -> list[float]: new planner choosing, among grids built from
     `candidates`, the one with minimal predicted_se subject to every point
     being ≥ k σ above P_u. plan_scales stays unchanged (the campaign used it).
   Tests (tests/test_zne_error.py):
   - on synthetic exponentials with known (b, c), exp_fixed_se agrees with a
     Monte Carlo SE (≥ 2000 replicas) within 10 % for c ∈ {0.5, 1, 2}, three
     grids, two shot counts;
   - optimal_step → 1/c as c grows (relative difference < 5 % at c = 4);
   - for two scales, exp_fixed_se equals the closed form in zne_theory.md §2;
   - the estimator's small-sample bias, measured in the same Monte Carlo, is
     reported (not asserted).

2. Script paper/scripts/zne_theory_check.py → paper/data/zne_theory.json:
   a. Hardware, both jobs: for every fit, SE_formula vs SE_bootstrap
      (from the stored interval, (hi − lo)/3.92) for resolved and for
      identified-not-resolved fits; and a list of every record whose raw
      value equals P_u + (P(1) − P_u)·e^{EXP_RATE_MAX} to 1e-6 relative.
   b. Simulation, each grid: for every resolved fit, ratio SE_formula /
      SE_bootstrap. Report median and 5–95 % range overall and per n.
      Coverage of the ideal by ±1.96·SE_formula vs by the bootstrap
      interval. Read the per-scale counts from the raw records in
      paper/data/sim_zne_fake_*/ (counts keyed by realised scale).
   c. Grid rule, simulation: for every instance with signal, c from the
      noiseless-vs-noisy kept signal at λ = 1 (c = −ln kept). Compute
      predicted_se for the odd, planned and plan_scales_spacing grids at the
      campaign's shots. Table: resolved fraction on each grid by bins of
      predicted SE, and by bins of c·Δλ. The claim to test is "a grid
      resolves when its spacing is near 1/c and its predicted SE is small".
      If the data do not support it, say so.
   d. Pointwise at M = 0 (116 instances per grid, plus the 3 hardware M = 0
      records): fraction of bitstrings whose per-string fit sits at the rate
      cap; how often the simplex output is a single string (D_TV = 1 − 2⁻ⁿ);
      rank correlation between the cap fraction and D_TV after clip.
   e. Linearity check for the Liu–Cai positioning: pointwise Richardson
      (fixed scales) output sums to 1 before renormalisation to 1e-12 on
      every instance; pointwise exponential does not (report the
      distribution of Σ_z raw(z)).

3. Figure paper/figures/fig_zne_grid_theory.{pdf,png} (plotting.py style):
   (a) predicted relative SE vs Δλ for c = 0.7, 1.1, 1.8 (the hardware
   records), with the odd (Δ = 2), gentle (Δ = 0.25 / 0.5) and typical
   planned (Δ = 0.5 / 1) grids marked; (b) SE_formula vs SE_bootstrap,
   simulation (by grid) and hardware, identity line.

4. Append a section "Verified at scale" to paper/notes/zne_theory.md with
   the numbers from zne_theory.json, including anything that failed.

Acceptance:
- Tests green. If SE_formula / SE_bootstrap has a median outside [0.85, 1.15]
  on resolved simulated fits, do not tune anything: report where it breaks
  (near the gate threshold? large c? few scales?) and stop.
- All four hardware cap identities and the 4 hardware SE matches reproduce.
- No existing record, flag, table or figure changes (git diff shows only new
  functions, the new script, its JSON, the new figure and the note).
```

---

## Prompt C — Step 4: how solid is α?

```
PROJECT CONTEXT — Paper A (Draper–Grover Subset Sum on heavy-hex hardware)

I am Arturo Arvizu. I am turning my MSc thesis (Grover search for Subset Sum
with a Draper QFT-adder oracle) into a journal paper (EPJ Quantum Technology or
Quantum Information Processing). The arXiv deadline is dropped: correctness
first. Repo: this directory (package `dgssp`, src layout, Qiskit 2.x,
qiskit-ibm-runtime 0.49, Aer, Mitiq 1.1.0). Windows + PowerShell, conda env
`dgssp-paper`, run through `conda run -n dgssp-paper ...`.

Current state:
- WP0–WP7 done. HEAD 9dc971c (2026-10-27).
- Hardware (ibm_fez, Heron r2):
  - job of record dbfeqr7gdj9s73bbgh6g, ran 2026-10-26 UTC (folder label
    2026-10-25 = plan date), odd grid λ ∈ {1,3,5}, 47 PUBs × 4096 shots;
  - job 2 dbgkqq8te5oc738icnfg, ran 2026-10-28 UTC, gentle grid
    λ ∈ {1,1.25,1.5}, 14 PUBs × 12 288 shots.
  - ~465 s of QPU left until ≈ 2026-11-21. Nothing in this session uses it.
- Simulation: 463 instances, fake_marrakesh, grids odd / odd7 / planned
  (paper/data/sim_*).
- α̂ = 3.176 [3.07, 3.35] (every calibrated error × α), fitted on job 1.

Read first (untracked notes written in chat on 2026-10-28, treat as the spec):
- paper/notes/hw_review_wp7.md   — number fixes and overclaims in hw_results.md
- paper/notes/paper_outline.md   — revised contributions and paper outline
- paper/notes/zne_theory.md      — theory of the scalar ZNE estimator, the
                                   gate, the grid rule, the pointwise failure

Revised contributions:
  C1 Resource model of the Draper-oracle Grover circuit for signed SSP
     (exact counts, heavy-hex cost, map of what fits under the hardware limit).
  C2 Hardware failure map + a predictive noise model (calibration with every
     error × α̂, fitted on one day, tested on another).
  C3 When nonlinear ZNE can be trusted for a sampling algorithm: scalar
     observable with a known asymptote, a resolution gate, a grid rule;
     why pointwise extrapolation of the distribution fails.

Rules:
1. No QPU. Never run --submit, never create a job, never read the IBM token.
   Anything that contacts the IBM service (even read-only) is a command I run.
2. Raw records in paper/data/hw_*/ and paper/data/sim_*/ are never edited.
   New results go to new files, stamped with _common.record_provenance().
3. One place per concept (encoding.py, decoding.py, runtime.sample_counts,
   mitigation/zne.py, metrics.py, backends.py). Extend, don't duplicate.
   Behaviour changes go behind a module constant (the RESOLVED_RULE pattern),
   with the old behaviour kept and selectable.
4. Every number written into a note comes from a file a script wrote. No
   hand-typed numbers. Each claim in a note names its file and key.
5. Seeds fixed (bootstrap seed 0, B = 2000 unless stated). pytest, ruff and
   mypy green before you hand over.
6. Direct, technically dense, no validation. If my request is wrong or
   underspecified, say so and propose the fix before writing code.
7. End with (a) the exact PowerShell commands I run, (b) the acceptance check
   I should see, (c) what to paste back, (d) a proposed commit message.

----------------------------------------------------------------

TASK: Step 4 — robustness of the α result. Offline only; one optional
read-only script that I run with my token.

1. Which errors carry α? In backends.scale_calibration_errors add a keyword
   `which: Literal["all", "2q", "1q", "readout"] = "all"` (default = current
   behaviour, unchanged). Expose it in noise_model_check.py as
   --scale-only {all,2q,1q,readout}. Cached counts must be keyed by `which`
   so the existing α cache is reused for "all" and never overwritten.
   Fit α on job 1 for "2q" and "readout" (same α grid, bootstrap and CI
   scaling as the paper's fit). For each variant report α̂, CI, χ²/dof,
   leave-one-record-out range, and the out-of-sample prediction of job 2
   (per-record λ = 1 deviation, RMS, max). The question: does scaling only the
   two-qubit errors explain the device as well as scaling everything? Expect
   this to need new simulations at each α; estimate the runtime first and
   tell me before running anything longer than ~1 h.

2. Depth dependence (no new simulation):
   - per-record α_r: fit α separately to each record's λ = 1 PUB (job 1 and
     job 2), with bootstrap CI, using the existing cached α grid;
   - per-record exponent ln(kept)/ln(S) with delta-method SE;
   - weighted regression of α_r on the 2Q count, slope with CI.
   Claim to test: "one α describes circuits up to ≈ 230 2Q; deeper circuits
   keep more signal than it predicts". Exclude over-rotated variants
   (r > r_opt) from the trend and say why.
   Figure for the supplement: paper/figures/fig_alpha_depth.{pdf,png}.

3. Residual structure at α̂, job 1: the per-circuit λ = 1 residual table with
   z-scores, grouped by sweep variant (n = 3 r ≠ r_opt below the model;
   n = 4 r = 1, 2 above). RMS of residuals and the χ²/dof it implies.

4. Day-to-day: α_r for P-n2-M1-11 on each day (different layouts) and the
   difference with CI. One number for "how much α moves between days".

5. Layered error (optional, read-only, I run it):
   paper/scripts/describe_layer_fidelity.py. Check what qiskit-ibm-runtime
   0.49 actually exposes for ibm_fez (layer fidelity, EPLG, "2Q error
   (layered)") through the backend object or the service API, without
   creating any job. Save what exists to
   paper/hardware/layer_fidelity_<today>.json with provenance. If nothing is
   exposed, the script says so and exits: do not scrape the dashboard. Then
   compute the implied per-gate error and compare it with 3.18 × median CZ
   error. State that today's value is not the run day's calibration.

6. Write paper/notes/alpha_robustness.md: each result in 2–4 sentences, with
   its file and key, plus the sentence each result allows in the paper.

Acceptance:
- `--scale-only all` reproduces the paper's α̂ = 3.176 and its CI exactly
  from the existing cache (no re-simulation).
- The new fits and the figure regenerate from paper/data/ alone.
- No existing α file, record or table changes.
```

---

## Prompt D — Step 5: what to report when a resolved value is above 1

```
PROJECT CONTEXT — Paper A (Draper–Grover Subset Sum on heavy-hex hardware)

I am Arturo Arvizu. I am turning my MSc thesis (Grover search for Subset Sum
with a Draper QFT-adder oracle) into a journal paper (EPJ Quantum Technology or
Quantum Information Processing). The arXiv deadline is dropped: correctness
first. Repo: this directory (package `dgssp`, src layout, Qiskit 2.x,
qiskit-ibm-runtime 0.49, Aer, Mitiq 1.1.0). Windows + PowerShell, conda env
`dgssp-paper`, run through `conda run -n dgssp-paper ...`.

Current state:
- WP0–WP7 done. HEAD 9dc971c (2026-10-27).
- Hardware (ibm_fez, Heron r2):
  - job of record dbfeqr7gdj9s73bbgh6g, ran 2026-10-26 UTC (folder label
    2026-10-25 = plan date), odd grid λ ∈ {1,3,5}, 47 PUBs × 4096 shots;
  - job 2 dbgkqq8te5oc738icnfg, ran 2026-10-28 UTC, gentle grid
    λ ∈ {1,1.25,1.5}, 14 PUBs × 12 288 shots.
  - ~465 s of QPU left until ≈ 2026-11-21. Nothing in this session uses it.
- Simulation: 463 instances, fake_marrakesh, grids odd / odd7 / planned
  (paper/data/sim_*).
- α̂ = 3.176 [3.07, 3.35] (every calibrated error × α), fitted on job 1.

Read first (untracked notes written in chat on 2026-10-28, treat as the spec):
- paper/notes/hw_review_wp7.md   — number fixes and overclaims in hw_results.md
- paper/notes/paper_outline.md   — revised contributions and paper outline
- paper/notes/zne_theory.md      — theory of the scalar ZNE estimator, the
                                   gate, the grid rule, the pointwise failure

Revised contributions:
  C1 Resource model of the Draper-oracle Grover circuit for signed SSP
     (exact counts, heavy-hex cost, map of what fits under the hardware limit).
  C2 Hardware failure map + a predictive noise model (calibration with every
     error × α̂, fitted on one day, tested on another).
  C3 When nonlinear ZNE can be trusted for a sampling algorithm: scalar
     observable with a known asymptote, a resolution gate, a grid rule;
     why pointwise extrapolation of the distribution fails.

Rules:
1. No QPU. Never run --submit, never create a job, never read the IBM token.
   Anything that contacts the IBM service (even read-only) is a command I run.
2. Raw records in paper/data/hw_*/ and paper/data/sim_*/ are never edited.
   New results go to new files, stamped with _common.record_provenance().
3. One place per concept (encoding.py, decoding.py, runtime.sample_counts,
   mitigation/zne.py, metrics.py, backends.py). Extend, don't duplicate.
   Behaviour changes go behind a module constant (the RESOLVED_RULE pattern),
   with the old behaviour kept and selectable.
4. Every number written into a note comes from a file a script wrote. No
   hand-typed numbers. Each claim in a note names its file and key.
5. Seeds fixed (bootstrap seed 0, B = 2000 unless stated). pytest, ruff and
   mypy green before you hand over.
6. Direct, technically dense, no validation. If my request is wrong or
   underspecified, say so and propose the fix before writing code.
7. End with (a) the exact PowerShell commands I run, (b) the acceptance check
   I should see, (c) what to paste back, (d) a proposed commit message.

----------------------------------------------------------------

TASK: Step 5 — decide how reported ZNE values outside [0, 1] are handled.
Two parts. Do Part 1, then STOP and wait for my choice.

Facts (recompute, don't trust): in simulation the strict rule reports 19 of
340 planned-grid values outside [0, 1] (1.001 … 4.17); on hardware the
resolved intervals reach 1.049 and 1.089. P(solution) lies in [P_u, 1] for
every instance with signal.

PART 1 — analysis only, no default changes.
1. Inventory: every reported (resolved and identifiable) value outside
   [0, 1] or below P_u, in simulation (all grids) and on hardware, with its
   interval; how many intervals still contain a physical value.
2. Implement three options as functions, selectable but not wired in:
   (a) "flag": report the fit unchanged, plus out_of_range = True;
   (b) "clip": clip the point and the interval to [P_u, 1] after fitting;
   (c) "bounded": constrain P(0) = a + b to [P_u, 1] inside the fit
       (bounded b in _exp_fixed_many's projection), bootstrap re-run with the
       bound, so the interval comes from the bounded estimator.
3. Compare on every resolved simulated fit, per grid and per n: median bias,
   RMSE and median absolute error vs the ideal; coverage of the ideal by the
   95 % interval; count outside [0, 1]; and whether the resolved / identifiable
   flags change (they must not, since the gate runs on the data, not the fit).
   Same table for the 4 hardware resolved fits.
4. Write paper/data/report_bounds.json and a short table in chat, with your
   recommendation and the one-sentence Methods wording for each option.
STOP here.

PART 2 — after I reply with my choice:
5. Wire it in behind a constant REPORT_BOUNDS = "flag" | "clip" | "bounded"
   in mitigation/zne.py (default = my choice; the other two selectable).
   Store the option in every re-derived record, as RESOLVED_RULE is stored.
6. Re-derive at aggregation time (no re-simulation, no re-collect):
   aggregate.py (all grids) and hw_aggregate.py for both dates (2026-10-25,
   and 2026-10-28 with --grids). Regenerate the affected tables and figures.
7. Diff report: every number in paper/notes/*.md, tables and figures that
   changed, old → new. Update check_claims.py so it still passes.

Acceptance (Part 2):
- Records untouched; only re-derived columns, tables, figures and notes move.
- The resolved / identifiable counts are identical before and after.
- check_claims.py: 0 MISMATCH.
```
