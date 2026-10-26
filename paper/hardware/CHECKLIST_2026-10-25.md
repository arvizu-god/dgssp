# Hardware submission checklist — 2026-10-25

Copy of `CHECKLIST_template.md` for the manifest `manifest_2026-10-25.json`.
`python paper/scripts/hardware_batch.py --submit` reads this file and refuses
to submit unless every box below is ticked, no `<...>` or `____` placeholder
is left, the pytest line names the current clean `HEAD`, and the budget line's
estimate is at most 50 % of the remaining allowance.

Filled by Claude on 2026-10-25 from what it verified itself in the repository
("verified"). The three items only Arturo can attest are left unticked.

## Before `--submit`

- [x] pytest green at commit 9c5aa67 (also `ruff check src tests paper/scripts` and `mypy src/dgssp paper/scripts` clean) — verified: 772 passed, 5 skipped; ruff and mypy clean; tracked tree clean
- [x] --plan ran, manifest reviewed by me: `paper/hardware/manifest_2026-10-25.json` (circuits, shots, physical qubits, sampler options, pre-flight all "ok") — plan verified: created 2026-10-26T00:58Z at 9c5aa67 (clean), 47 circuits x 4096 shots, pre-flight all ok, snapshot == live target, calibration stable during the plan
- [x] --dry-run ran and P_sol on the two smallest instances matches the expectation: P-n2-M1-11 P_sol 0.837 [0.825, 0.847] vs ideal 1.000; P-n3-M2-17 P_sol 0.671 [0.658, 0.685] vs ideal 1.000; drift lines show no drift (z = -0.36, p = 0.72; z = -0.87, p = 0.39) — verified; collect acceptance PASS, 17 records identical
- [x] Probe done: `--probe-report` says DONE, options accepted (twirling ON, applied, TwirledSliceSpanV2; DD off); --plan re-run with its --per-shot-time 0.000622611 and --overhead 0.3824, then --dry-run re-run — verified: probe job dbf9tu7gdj9s73fn4u40, usage 7 s; manifest uses both flags; dry run is of this manifest
- [x] Estimated QPU time <= 50% of remaining allowance: estimate 275.7 s, remaining 593 s (after the probe's usage) — verified against the manifest: cap 296.5 s, max_execution_time 296 s
- [x] Calibration snapshot saved: `paper/hardware/calibration_2026-10-25.json` (its calibration stamp is the one in the manifest) — verified: stamp 2026-10-25T18:24:25-06:00, file hash matches the manifest
- [x] Device online, no maintenance window: checked on https://quantum.cloud.ibm.com/ at 05:56:10 UTC
- [x] Token in env: `$env:QISKIT_IBM_TOKEN` set in this PowerShell window only, not saved in any file (old key revoked 2026-10-25)
- [x] Job log path writable: `runs/jobs.jsonl` — verified: writable; the probe job is logged there
- [x] account_facts.md filled (Status: filled 2026-10-25), and its primary device is the manifest's backend — verified: primary ibm_fez, fallback ibm_kingston

## After `--submit` (not checked by the script)

- Job id copied from the terminal into my notes: ____
- `--collect --jobs paper/hardware/jobs_2026-10-25.json` run once the job is DONE;
  `paper/hardware/usage.md` compared with the estimate.
