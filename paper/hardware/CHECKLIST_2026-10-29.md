# Hardware submission checklist — <date>

Copy this file to `paper/hardware/CHECKLIST_<date>.md`, where `<date>` is the
date in the manifest's file name (`manifest_<date>.json`). Fill every blank,
then tick each box by changing `[ ]` to `[x]`.

`python paper/scripts/hardware_batch.py --submit` reads that file and refuses
to submit unless:

- every box below is still there and ticked;
- no placeholder is left (`<...>` or `____`);
- the commit on the pytest line is the current `HEAD` (7 or more hex
  characters) and the working tree is clean;
- the budget line gives the estimate (not lower than the manifest's) and the
  remaining allowance, in seconds, with the estimate at most 50 % of it.

For a manifest planned from a design file (`--plan --design`), also:

- `paper/hardware/preregistration_<date>.json` exists, was made from this
  manifest, is committed, and `HEAD` is on `origin/main` after `git fetch`;
- the "Pre-registration pushed" line names the commit that holds it and its
  sha256 (both printed by `preregister.py` and `git log`);
- the pytest line may name the code commit instead of `HEAD`, as long as
  every file changed since is one of this job's (`paper/hardware/`, or a
  `paper/data/` folder named after the date);
- the calibration entries of the job's qubits are unchanged since `--plan`
  (else `--accept-recalibration`).

For any other manifest, tick the pre-registration line with "not a design job".

Keep each line's opening words as they are: the script finds the pytest and
budget lines by how they start.

## Before `--submit`

- [x] pytest green at commit 34a1d964dc (also `ruff check src tests paper/scripts` and `mypy src/dgssp paper/scripts` clean)
- [x] --plan ran, manifest reviewed by me: `paper/hardware/manifest_2026-10-29.json` (circuits, shots, physical qubits, sampler options, pre-flight all "ok")
- [x] --dry-run ran and P_sol on the two smallest instances matches the expectation: P-n2-M1-11_L1 P_sol 0.787 vs ideal 1.000; P-n2-M1-11_L2 P_sol 0.849 vs ideal 1.000; drift lines show no drift
- [x] Probe done: not run: the estimate is usage / floor of jobs 2026-10-25 and 2026-10-28 (manifest `estimate.usage_rate`)
- [x] Estimated QPU time <= 50% of remaining allowance: estimate 182.9 s, remaining 456 s (no probe today; cap 228 s, max_execution_time 228 s)
- [x] Calibration snapshot saved: `paper/hardware/calibration_2026-10-29.json` (its calibration stamp is the one in the manifest)
- [x] Device online, no maintenance window: checked on https://quantum.cloud.ibm.com/ at 04:45 UTC
- [x] Token in env: `$env:QISKIT_IBM_TOKEN` set in this PowerShell window only, not saved in any file
- [x] Job log path writable: `runs/jobs.jsonl`
- [x] account_facts.md filled (Status: filled 2026-10-25), and its primary device is the manifest's backend
- [x] Pre-registration pushed: commit 198346d0cb884c8d104f4e85b9b227c6b29853d2, sha256 cee4f23f9999e8143bbe7d04c15ffc89f199cca8e3ac6d3adc10fd2934880816, pushed at 04:50 UTC

## After `--submit` (not checked by the script)

- Job id copied from the terminal into my notes: ____
- `--collect --jobs paper/hardware/jobs_<date>.json` run once the job is DONE;
  `paper/hardware/usage.md` compared with the estimate.
