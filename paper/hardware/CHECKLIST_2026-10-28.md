# Hardware submission checklist — 2026-10-28

Copy of `CHECKLIST_template.md` for the manifest `manifest_2026-10-28.json`
(second job: gentle grid λ = 1, 1.25, 1.5 on P-n2-M1-11, P-n3-M1-17 and its
r = 1 sweep, P-n3-M2-17; 14 PUBs × 12 288 shots).
`python paper/scripts/hardware_batch.py --submit` reads this file and refuses
to submit unless every box below is ticked, no `<...>` or `____` placeholder
is left, the pytest line names the current clean `HEAD`, and the budget line's
estimate is at most 50 % of the remaining allowance.

Filled by Claude on 2026-10-27 (local) from what it verified itself in the
repository ("verified"). The two items only Arturo can attest are left
unticked: the device check (with its UTC time) and the token.

Commit the plan files and this checklist only after `--submit`: the pytest
line names `4f43803`, and a new commit before submitting would make it stale.

## Before `--submit`

- [x] pytest green at commit 4f43803 (also `ruff check src tests paper/scripts` and `mypy src/dgssp paper/scripts` clean) — verified: full suite passed (exit 0); ruff and mypy (42 files) clean; tracked tree clean
- [x] --plan ran, manifest reviewed by me: `paper/hardware/manifest_2026-10-28.json` (circuits, shots, physical qubits, sampler options, pre-flight all "ok") — plan verified: created 2026-10-28T00:48:51Z at 4f43803 (clean), 14 circuits x 12288 shots = 172032 (subset 9, sweep 3, drift 2), pre-flight all ok, snapshot == live target, calibration stable during the plan, max_execution_time 258 s
- [x] --dry-run ran and P_sol on the two smallest instances matches the expectation: P-n2-M1-11 P_sol 0.859 [0.853, 0.865] vs ideal 1.000; P-n3-M2-17 P_sol 0.649 [0.641, 0.657] vs ideal 1.000; drift lines show no drift (z = +0.16, p = 0.87; z = +0.07, p = 0.95) — verified; collect acceptance PASS, 6 records identical
- [x] Probe done: `--probe-report` says DONE, options accepted (twirling ON, applied, TwirledSliceSpanV2; DD off) — job 1's probe dbf9tu7gdj9s73fn4u40, and this manifest uses the same sampler options; --plan ran with its --per-shot-time 0.000622611 and --overhead 0.3824, then --dry-run ran — verified: dry run is of this manifest (sha256 d3b1c37d)
- [x] Estimated QPU time <= 50% of remaining allowance: estimate 224.9 s, remaining 516 s (after job 1 and its probe) — verified against the manifest: cap 258 s, max_execution_time 258 s; job 1's real rate predicts about 67 s
- [x] Calibration snapshot saved: `paper/hardware/calibration_2026-10-28.json` (its calibration stamp is the one in the manifest) — verified: stamp 2026-10-27T18:12:25-06:00, file hash bf1fefaf matches the manifest
- [x] Device online, no maintenance window: checked on https://quantum.cloud.ibm.com/ at 1:17 AM UTC
- [x] Token in env: `$env:QISKIT_IBM_TOKEN` set in this PowerShell window only, not saved in any file
- [x] Job log path writable: `runs/jobs.jsonl` — verified: job 1 and its probe are logged there
- [x] account_facts.md filled (Status: filled 2026-10-25), and its primary device is the manifest's backend — verified: primary ibm_fez; the manifest records filled and backend listed

## Pre-registered expectation (written before submission)

Noise model of this calibration with every error × α̂ = 3.176
(`paper/data/hw_predict_plan_x3.176_2026-10-28`):

| Record | 2Q at λ = 1 | Ideal | ZNE predicted [95% CI] | Status predicted |
|---|---|---|---|---|
| P-n2-M1-11 | 71 | 1.000 | 0.930 [0.874, 0.985] | resolved |
| P-n3-M2-17 | 184 | 1.000 | 0.806 [0.613, 1.107] | resolved |
| P-n3-M1-17 | 235 | 0.945 | 0.670 [0.507, 0.862] | resolved |
| P-n3-M1-17_r1 | 126 | 0.781 | 0.609 [0.508, 0.719] | resolved |

## After `--submit` (not checked by the script)

- Job id copied from the terminal into my notes: dbgkqq8te5oc738icnfg
  (submitted 2026-10-28T01:18Z; the device had recalibrated at
  2026-10-27T19:08:30-06:00, and the pre-flight against it passed)
- Collected 2026-10-28: DONE, 51 s charged against the 224.9 s estimate.
- `--collect --jobs paper/hardware/jobs_2026-10-28.json` run once the job is DONE;
  `paper/hardware/usage.md` compared with the estimate.
