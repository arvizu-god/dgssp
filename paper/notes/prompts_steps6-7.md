# Claude Code prompts: steps 6–7 (third hardware job: build, pre-register, submit, analyse)

Written 2026-10-29. Same rules as `prompts_steps2-5.md`: one Claude Code
session per prompt; each block is self-contained (preamble + task); commit after
each session.

## What changed since the plan of 2026-10-28, and why the design below

Steps 3–4 changed three things in the job design:

1. **No "depth effect" in the full model** (`alpha_robustness.md` §2). The
   exponent ln(kept)/ln(S) falls with depth, but the per-circuit α of the full
   noise model does not (slope −0.09 [−0.30, 0.12] per 100 2Q). What dominates
   is circuit-to-circuit scatter (α_r 2.2–3.8, χ²/dof 8). The day-1 residuals
   concentrate in the sweep variants. The open question is no longer "depth"
   but **"is a circuit's α_r a property of that circuit, reproducible on
   another day, or noise?"** The job therefore re-sends job-1 circuits
   unchanged (same ISA circuit, layout and routing).
2. **A competing model.** Scaling only the CZ errors (α_2q = 4.055) fits job 1
   almost as well (Δχ² = 9) and predicted job 2 slightly better. A third day
   is the first test between the two.
3. **The theory's limits** (`zne_theory.md` §7). The optimal second step is
   ≈ 1.109/c, not 1/c. The error formula holds only where the predicted
   relative SE ≤ 0.2 and runs low above 0.3. So the pre-registration states
   SE predictions only where relative SE ≤ 0.2, uses `optimal_step` and
   `plan_scales_spacing`, and plans from the **full model at α̂** (simulated
   per PUB), not from the S^α shortcut, which fails at depth.

### The design

12 288 shots per PUB (as job 2), ibm_fez, twirling on, DD off. "L1" = the
circuit job 1 sent, "L2" = the circuit job 2 sent (`--pin-layout KEY:TAG=<date>`).

| Record | Circuit | Scales | Tests |
|---|---|---|---|
| P-n2-M1-11_L1 | job 1, 71 2Q | gentle {1, 1.25, 1.5} + rule grid from `plan_scales_spacing` (≈ {1, 2.8, 4.5}), λ = 1 shared → 5 PUBs | error formula / grid rule (predicted SE ratio ≈ 1.8); α_r reproducibility; control |
| P-n3-M1-17_L1 | job 1, 212 2Q | odd {1, 3, 5} + gentle {1.25, 1.5}, λ = 1 shared → 5 PUBs | **the grid alone decides**: same circuit, layout, shots, job |
| P-n3-M1-17_r1_L1 | job 1, 112 2Q | gentle → 3 PUBs | α_r reproducibility (day-1 α_r = 3.76, the extreme); gentle vs day-1 odd on the same circuit |
| P-n4-M1-10_r1_L1 | job 1, 233 2Q | gentle → 3 PUBs | does ZNE reach n = 4 (r = 1)? |
| P-n4-M1-10_r2_L1 | job 1, 475 2Q | gentle → 3 PUBs | does the largest positive residual (+0.022, z 3.4) reproduce? |
| P-n3-M2-17_L1 | job 1, 194 2Q | λ = 1 → 1 PUB | layout or day |
| P-n3-M2-17_L2 | job 2, 184 2Q | λ = 1 → 1 PUB | layout or day |
| drift | P-n2-M1-11_L1, P-n3-M2-17_L2 at λ = 1 | 2 PUBs | drift within the job |

**23 PUBs.** At job 2's measured rate that is ≈ 85 s charged; the ×2 planning
estimate ≈ 170 s, under the cap (half of ≈ 465 s). If the estimate exceeds the
cap, trim in this order: the folds of P-n4-M1-10_r2_L1 (keep λ = 1), then the
folds of P-n3-M1-17_r1_L1, then P-n4-M1-10_r1_L1.

### Pre-declared criteria (frozen in the pre-registration file before submission)

Reported ZNE values follow step 5 (`REPORT_BOUNDS = "clip"`: value and
interval clipped to [P_u, 1]). **Error-size comparisons (Q4) use the
unclipped bootstrap** (the "flag" view stored alongside), because clipping an
interval at 1 shrinks (hi − lo) and would bias SE_boot low.

| ID | Question | Rule |
|---|---|---|
| Q1 | Does α̂ hold on a third day? | For every record except P-n3-M2-17_*, \|P_hw(λ = 1) − P_pred(α̂)\| ≤ 0.05 → "held". Report per-record deviations and RMS. |
| Q2 | All errors × 3.18 or CZ errors × 4.06? | λ = 1 RMS of each model over the same records. "Better" only if lower by ≥ 0.005; otherwise "indistinguishable". |
| Q3 | Does the grid alone decide? | P-n3-M1-17_L1: odd ∈ {not identifiable, identified-not-resolved} **and** gentle = resolved → "confirmed". |
| Q4 | Does the error formula predict hardware? | Each resolved fit with predicted relative SE ≤ 0.2: SE_boot(unclipped) / SE_pred ∈ [0.75, 1.33]. P-n2: observed SE_gentle / SE_rule within ±30 % of the predicted ratio. |
| Q5 | Predicted ZNE status | Predicted vs observed status per record × grid; count of matches. |
| Q6 | Is α_r a property of the circuit? | Pinned job-1 circuits, α_r(day 3) − α_r(day 1) with CIs. "Reproducible per circuit" iff χ²/dof of the differences ≤ 2 **and** the day-3 between-circuit χ²/dof stays > 2. Spearman reported, not tested (n is small). |
| Q7 | P-n3-M2-17: layout or day? | "At random" = λ = 1 less than 2σ above P_u; "fine" = within 0.05 of the α̂ prediction. Table → sentence (below). |
| Q8 | Does gentle-grid ZNE reach n = 4? | P-n4-M1-10_r1_L1 resolved → "reaches n = 4 at r = 1"; otherwise the λ = 1 value is reported. |
| Q9 | Does the P-n4-M1-10_r2 residual reproduce? | Sign and size of the λ = 1 residual at α̂ vs day 1 (+0.022): reported, not tested. |

Q7 table: L2 random + L1 fine → "a persistent layout defect the calibration
does not show"; both fine → "a transient on 2026-10-28"; both random → "not a
layout effect; open"; L1 random + L2 fine → "the defect moved; open".

Also reported, not tested: the day-3 layer fidelity (EPLG and EPLG / median
CZ) from the execution calibration's `raw_properties.general`, next to the
day-1 and day-2 values in `alpha_robustness.md` §5.

### Order

| Prompt | Step | What | Who runs the QPU |
|---|---|---|---|
| E | 6a | Build everything offline: design file, per-record grids, pre-registration and scoring code, multi-grid aggregation, submit gate. Full rehearsal on a simulated job | nobody |
| F | 6b | Submission day: plan → dry run → pre-register → push → checklist → **you** submit | you |
| G | 7 | Collect, execution calibration, score against the frozen criteria, integrate job 3 into notes, tables, figures, claims | nobody (you run collect / fetch) |

Steps 2–5 are done and committed (HEAD `e4e22da`). Target: E done by about
Nov 6, submit by about Nov 12–14, window ends ≈ Nov 21.

---

## Prompt E — Step 6a: build and rehearse the third job offline

```
PROJECT CONTEXT — Paper A (Draper–Grover Subset Sum on heavy-hex hardware)

I am Arturo Arvizu. I am turning my MSc thesis (Grover search for Subset Sum
with a Draper QFT-adder oracle) into a journal paper (EPJ Quantum Technology or
Quantum Information Processing). Correctness first. Repo: this directory
(package `dgssp`, src layout, Qiskit 2.x, qiskit-ibm-runtime 0.49, Aer,
Mitiq 1.1.0). Windows + PowerShell, conda env `dgssp-paper`, run through
`conda run -n dgssp-paper ...`.

Current state:
- WP0–WP7 and steps 2–5 done and committed. HEAD e4e22da (2026-10-29).
  - Step 2: notes corrected against the data; check_claims.py (0 MISMATCH).
  - Step 3: error propagation of the fixed-asymptote exponential in
    mitigation/zne.py §11 (exp_fixed_se, predicted_se, optimal_step,
    plan_scales_spacing); paper/data/zne_theory.json. Optimal second step
    ≈ 1.109/c; the formula holds where predicted relative SE ≤ 0.2 and runs
    low above 0.3.
  - Step 4: paper/data/alpha_robustness.json. α̂_all = 3.176 [3.074, 3.348];
    α̂_2q (CZ errors only) = 4.055 [3.511, 4.462], Δχ² = 9.1; readout alone
    cannot fit. Per-circuit α_r scatters 2.2–3.8 (χ²/dof 8) with no
    resolvable depth trend; residuals concentrate in the sweep variants.
    Day-to-day α for the same instance on a new layout moved +0.04.
    Execution calibrations carry IBM layer fidelity (EPLG / median CZ
    2.2–2.4).
  - Step 5: REPORT_BOUNDS = "clip" (value and interval clipped to [P_u, 1]),
    "flag" and "bounded" selectable; report_bounds.json.
- Hardware (ibm_fez, Heron r2):
  - job 1 dbfeqr7gdj9s73bbgh6g (ran 2026-10-26 UTC, folder label
    2026-10-25), odd grid, 47 PUBs × 4096;
  - job 2 dbgkqq8te5oc738icnfg (ran 2026-10-28 UTC), gentle grid,
    14 PUBs × 12 288.
  - ≈ 465 s of QPU left until ≈ 2026-11-21.
- origin/main is behind (last pushed 8ab7b73); nothing since has been pushed.
- Untracked on purpose: paper/hardware/account_snapshot.json,
  paper/data/hw_dryrun_*_collect/. Untracked, to decide:
  paper/hardware/layer_fidelity_2026-10-29.json (today's value, not run-day;
  commit it only if a note cites it).

Read first: paper/notes/prompts_steps6-7.md (the design, criteria Q1–Q9 and
why), zne_theory.md §7, alpha_robustness.md, hw_results.md.

Rules:
1. No QPU. Never run --submit or --probe, never create a job, never read the
   IBM token. Anything that contacts the IBM service is a command I run.
2. Raw records in paper/data/hw_*/ and sim_*/ are never edited. New results
   go to new files, stamped with _common.record_provenance().
3. One place per concept. Extend, don't duplicate. Behaviour changes go
   behind constants or flags, with the old behaviour kept and selectable;
   jobs 1 and 2 must re-plan, re-collect and re-aggregate byte-identically.
4. Every number in a note comes from a file a script wrote.
5. Seeds fixed. pytest, ruff, mypy green before you hand over.
6. Direct, technically dense, no validation. If the design or a criterion is
   wrong or untestable, say so and propose the fix before writing code.
7. End with (a) exact PowerShell commands, (b) the acceptance check I should
   see, (c) what to paste back, (d) a proposed commit message.

----------------------------------------------------------------

TASK: Step 6a — build, offline, everything the third hardware job needs, so
that on submission day nothing is written by hand and the analysis code is
frozen before the data exist.

0. Review first. Read the design and Q1–Q9 in prompts_steps6-7.md against
   the step 3–5 results. Before writing code, list any criterion that is
   untestable with this design, any record that cannot discriminate what it
   claims to, and any threshold that contradicts zne_theory.json or
   alpha_robustness.json, with your proposed fix. Wait for my answer.

1. Design file paper/hardware/design_job3.json (schema
   "dgssp/paper/hardware-design/1"):
   - one entry per record: key, source (pinned circuit `KEY:TAG=<date>` or
     instance), named grids ({"odd": [1,3,5], "gentle": [1,1.25,1.5],
     "rule": "plan_scales_spacing"}), shots, drift flag;
   - the trim order;
   - the criteria Q1–Q9 as data (thresholds as numbers, which view of the
     interval each uses: clip for reporting, unclipped for Q4);
   - REPORT_BOUNDS and RESOLVED_RULE in force, recorded.
   Fill it from the design table in prompts_steps6-7.md.

2. hardware_batch.py --plan --design <file>:
   - per-record scale set = the union of that record's named grids, λ = 1
     shared; λ = 1-only records allowed;
   - the "rule" grid is resolved at plan time with plan_scales_spacing, from
     the full noise model at α̂ (simulated λ = 1 kept signal of that circuit on
     the plan calibration), not the S^α shortcut; record the inputs and the
     chosen Δ next to optimal_step's Δ*;
   - pinned circuits reuse the saved ISA circuit (layout and routing
     unchanged) and are folded from it; assert their 2Q counts equal the
     source job's;
   - the manifest stores each record's named grids, so analysis fits each
     grid separately from the same counts;
   - estimate and cap check as now; if over the cap, apply the design's trim
     order and say what was trimmed.
   Existing flags keep working unchanged.

3. paper/scripts/preregister.py --date <d>:
   - simulate every PUB of the manifest on the plan calibration at α̂_all and
     at α̂_2q (--scale-only 2q), reusing noise_model_check machinery;
   - per record × named grid: predicted P(λ); predicted status (the gate
     applied to the expected values at the planned shots); predicted reported
     value under REPORT_BOUNDS = clip; predicted SE (unclipped) via
     predicted_se / exp_fixed_se *only* where the predicted relative SE ≤ 0.2,
     else "formula not applicable" with the value; the predicted
     SE_gentle / SE_rule ratio for P-n2;
   - the day-1 α_r with CI for every pinned job-1 circuit
     (alpha_robustness.json), as the Q6 reference;
   - write paper/hardware/preregistration_<d>.json (criteria copied from the
     design, manifest sha256, calibration stamp, code commit, all
     predictions) and a readable preregistration_<d>.md with the Q1–Q9 table
     and the outcome → sentence map. Print the JSON's sha256.

4. paper/scripts/score_preregistration.py --date <d>:
   - reads the pre-registration and the collected records, and evaluates
     Q1–Q9 exactly as declared (thresholds read from the file, never from
     code constants);
   - writes paper/data/scorecard_<d>.{json,md}: per question predicted,
     observed, verdict, the selected sentence, and a "deviations" section
     that stays empty unless I add an entry by hand with a reason;
   - Q6 needs α_r on day 3: build it on step 4's per-record fit (in-sample
     cache for the job-3 date), callable from here;
   - also reports, untested: day-3 EPLG and EPLG / median CZ from the
     execution calibration, next to days 1–2.

5. Multi-grid aggregation: hw_aggregate.py --date <d> fits each named grid of
   each record separately (same counts, λ = 1 shared), and writes a same-job
   grid table (record × grid: status, reported value and interval (clip),
   SE_boot unclipped, SE_pred) as tex/hw_table_grids_job3.tex. Jobs 1 and 2
   aggregate byte-identically to HEAD.

6. Submit gate (hardware_batch.py --submit), new checks for any manifest made
   from a design file:
   - preregistration_<d>.json exists, is committed, and its manifest sha256
     matches;
   - HEAD is on origin/main after `git fetch` (git merge-base --is-ancestor
     HEAD origin/main), so the pre-registration has a GitHub timestamp
     before the job;
   - the checklist line "Pre-registration pushed: commit <hash>, sha256 <…>,
     pushed at <UTC>" is filled;
   - if the live calibration stamp differs from the plan's, refuse unless
     --accept-recalibration is given (predictions are re-run on the
     execution calibration afterwards either way).
   Add the new line to CHECKLIST_template.md.

7. End-to-end rehearsal, entirely offline:
   - --plan --design against the 2026-10-28 calibration snapshot (or
     --offline-backend fake_fez);
   - --dry-run, then preregister.py;
   - treat the dry run as the hardware: collect it into a scratch date, run
     hw_aggregate.py and score_preregistration.py, and show the scorecard;
   - test that --submit refuses at each new gate (no pre-registration, hash
     mismatch, not pushed, calibration changed) with the service mocked.

8. Write paper/hardware/RUNBOOK_job3.md: the exact submission-day PowerShell
   commands in order (plan, dry run, preregister, commit, push, checklist,
   submit, and what to paste back), with the expected output of each.

Acceptance:
- Jobs 1 and 2: --plan rebuild and hw_aggregate outputs byte-identical to HEAD.
- The rehearsal yields a manifest of ≈ 23 PUBs with pinned 2Q counts equal to
  their sources (71, 212, 112, 233, 475, 194, 184), the estimate and cap
  check, the pre-registration (.json + .md), a scorecard and the multi-grid
  table, with no hand-typed number.
- Each submit gate is covered by a test that fails when the gate is removed.
- pytest, ruff, mypy green.
```

---

## Prompt F — Step 6b: submission day

Do this in one sitting, within one calibration window. You run every command
that touches the service; Claude Code only reviews.

```
PROJECT CONTEXT — Paper A (Draper–Grover Subset Sum on heavy-hex hardware)

Same project, rules and current state as Prompt E in
paper/notes/prompts_steps6-7.md, plus: step 6a is done and committed.
HEAD: «fill in». Today: «YYYY-MM-DD». Runbook: paper/hardware/RUNBOOK_job3.md.

Rules that matter today:
- You never run a service command, never read the token, never submit.
- No criterion, threshold or design entry changes after preregister.py has
  run. If one is wrong, we stop and re-plan from the start.

----------------------------------------------------------------

TASK: Step 6b — review only. I will run RUNBOOK_job3.md step by step and
paste each output. After each paste:
1. Check it against the runbook's expected output and the design: record
   list, named grids, shots, PUB count, pinned 2Q counts and physical qubits
   equal to their source jobs, pre-flight "ok", estimate ≤ cap, calibration
   stamp, REPORT_BOUNDS = clip and RESOLVED_RULE = v2 recorded.
2. After preregister.py: read preregistration_<d>.md and say whether every
   Q1–Q9 has a prediction or an explicit "not applicable" with its reason.
   Flag any prediction that contradicts jobs 1–2 for a pinned circuit
   (e.g. a predicted status the day-1 data rule out), and say why.
3. Before I push: confirm the commit holds the design, manifest, circuits,
   dry run, pre-registration and checklist, and the tree is otherwise clean.
4. Before I submit: confirm the checklist is complete, the pushed commit is
   HEAD, and the live calibration stamp equals the plan's (or tell me to
   decide on --accept-recalibration).
5. After submission: tell me what to record and stop.
If anything fails (rejected options, ISA failure, device offline,
recalibration), stop and tell me whether to re-plan today or another day. Do
not suggest workarounds that skip a gate.
```

Submission-day outline (exact commands come from `RUNBOOK_job3.md`, written by Prompt E):

1. Fresh PowerShell, `$env:QISKIT_IBM_TOKEN` set in this window only.
2. `--plan --design paper/hardware/design_job3.json --backend ibm_fez`.
3. `--dry-run`.
4. `preregister.py --date <d>`.
5. Fill `CHECKLIST_<d>.md`; `git add` design, manifest, circuits, dry run,
   pre-registration and checklist; commit; **push**.
6. Fill the "Pre-registration pushed" line with that commit's hash; commit;
   push.
7. `--submit` (it runs the gates).
8. Wait for DONE; then Prompt G.

If the GitHub repo is private, the push timestamp proves the order to you and
to anyone you give access, not to the public. To make it public without
opening the repo, post the pre-registration's sha256 with a timestamp
somewhere public (a public gist works) at step 6.

---

## Prompt G — Step 7: collect, score, and integrate job 3

```
PROJECT CONTEXT — Paper A (Draper–Grover Subset Sum on heavy-hex hardware)

Same project, rules and current state as Prompt E in
paper/notes/prompts_steps6-7.md, plus: steps 6a–6b done; job 3 submitted.
HEAD: «fill in». Job 3: «job id», ran «UTC», date label «d».

----------------------------------------------------------------

TASK: Step 7 — turn job 3 into results, using the code frozen in step 6a.
Analysis code may only change to fix a bug; every such fix is logged in
the scorecard's "deviations" section with the reason and the diff, and the
scorecard is computed with the code as frozen *and* as fixed.

Commands I run (tell me when; paste back their outputs):
- hardware_batch.py --collect --jobs paper/hardware/jobs_<d>.json
- noise_model_check.py --fetch-calibration --date <d> (execution
  calibration; no QPU)
- usage into paper/hardware/usage.{json,md}

Then, offline:
1. Predictions on the execution calibration at α̂ and α_2q (secondary; the
   pre-registered plan-calibration predictions stay primary). Report how
   many calibration entries changed between plan and execution.
2. Per-record α_r on day 3 (in-sample cache for <d>), as in step 4.
3. score_preregistration.py --date <d> → scorecard_<d>.{json,md}. Q1–Q9
   verdicts as declared; nothing re-thresholded.
4. hw_aggregate.py --date <d>: per-record table, multi-grid table
   (hw_table_grids_job3.tex; reported values clipped, SE_boot unclipped),
   model-vs-hardware block, drift (shot-noise null), survival block; day-3
   EPLG next to days 1–2.
5. Cross-job results (new script hw_crossjob.py → paper/data/hw_crossjob.json):
   - α̂ out of sample on two days (jobs 2 and 3): per-record λ = 1
     deviations, RMS with and without P-n3-M2-17, for all-errors and
     CZ-only;
   - α_r day 1 vs day 3 per pinned circuit (Q6), with the step-4 day-2
     values where they exist;
   - P-n3-M2-17 on days 1, 2, 3 and both layouts (Q7);
   - the error formula on hardware: every resolved fit across jobs 1–3,
     SE_pred vs SE_boot, split into post hoc (jobs 1–2) and pre-registered
     (job 3);
   - total QPU seconds over all jobs and probes.
6. Figures: Fig 5 with job-3 points; fig_alpha_depth with day 3; a new
   fig_hw_grid_samejob (P-n3-M1-17_L1, odd vs gentle fits on one panel,
   and P-n2 gentle vs rule); regenerate everything from paper/data alone.
7. Notes:
   - hw_results.md: new section "Job 3 (pre-registered)", scorecard
     verdicts first, then numbers, then what changed in §1–§6 and why;
   - alpha_robustness.md: day 3 and the Q2, Q6 answers;
   - zne_theory.md §7: "Prospective test on hardware" (Q3, Q4, Q5);
   - paper_outline.md §1.2–1.4: update the contribution evidence and
     novelty rows the verdicts affect, and the abstract numbers;
   - check_claims.py: add every job-3 claim; 0 MISMATCH.
8. Decision list for me, from the scorecard: which paper sentences now
   stand, which must be weakened, which are new; and whether anything in
   C2–C3 must be re-framed. Do not edit paper text beyond the notes.

Acceptance:
- Scorecard computed from the pre-registration file unchanged (its sha256
  printed and equal to the pushed one).
- Jobs 1–2 outputs unchanged; job 3 outputs regenerate from paper/data alone.
- check_claims.py 0 MISMATCH; pytest, ruff, mypy green.
```

---

## After step 7

The paper's evidence is complete. Next is the writing phase (Phase 3 in the
plan of 2026-10-28): Methods → Resource model → Simulation → Hardware (jobs
1–3, scorecard first) → Discussion → Introduction / related work → Abstract /
Conclusion. Each section is drafted only from `paper/notes/` and `paper/data/`.
