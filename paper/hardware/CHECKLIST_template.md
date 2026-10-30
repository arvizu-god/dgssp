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

- [ ] pytest green at commit <hash> (also `ruff check src tests paper/scripts` and `mypy src/dgssp paper/scripts` clean)
- [ ] --plan ran, manifest reviewed by me: `paper/hardware/manifest_<date>.json` (circuits, shots, physical qubits, sampler options, pre-flight all "ok")
- [ ] --dry-run ran and P_sol on the two smallest instances matches the expectation: <instance 1> P_sol ____ vs ideal ____; <instance 2> P_sol ____ vs ideal ____; drift lines show no drift
- [ ] Probe done: `--probe-report` says DONE, options accepted (twirling ____, DD ____); --plan re-run with its --per-shot-time ____ and --overhead ____, then --dry-run re-run
- [ ] Estimated QPU time <= 50% of remaining allowance: estimate ____ s, remaining ____ s (after the probe's usage)
- [ ] Calibration snapshot saved: `paper/hardware/calibration_<date>.json` (its calibration stamp is the one in the manifest)
- [ ] Device online, no maintenance window: checked on https://quantum.cloud.ibm.com/ at ____ UTC
- [ ] Token in env: `$env:QISKIT_IBM_TOKEN` set in this PowerShell window only, not saved in any file
- [ ] Job log path writable: `runs/jobs.jsonl`
- [ ] account_facts.md filled (Status: filled ____), and its primary device is the manifest's backend
- [ ] Pre-registration pushed: commit <hash>, sha256 <sha256>, pushed at ____ UTC

## After `--submit` (not checked by the script)

- Job id copied from the terminal into my notes: ____
- `--collect --jobs paper/hardware/jobs_<date>.json` run once the job is DONE;
  `paper/hardware/usage.md` compared with the estimate.
