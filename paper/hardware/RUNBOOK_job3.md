# Runbook — job 3 (pre-registered), submission day

Prompt F of `paper/notes/prompts_steps6-7.md`. One sitting, one calibration
window, one fresh PowerShell window. **You** run every command; Claude Code
only reviews what you paste back. Nothing here is typed into a data file by
hand except the checklist; every number below is printed by a script.

Design: `paper/hardware/design_job3.json` (10 records, 24 PUBs × 12 288 shots,
named grids odd / gentle / rule, drift on P-n2-M1-11_L1 and P-n3-M2-17_L2,
Q1–Q9 as data). Expected outputs quote the offline rehearsal
(`python paper/scripts/rehearse_design.py`, 2026-10-28 calibration snapshot,
`--remaining-seconds 465`); the live plan will differ in the calibration
stamp, the floor, the rule grid's simulated signal and the predictions.

If any step does not print what is expected: **stop**, paste it, and decide
with Claude whether to re-plan today or another day. Do not skip a gate.
No criterion, threshold or design entry changes after step 6.

## 0. Before the day

- Step 6a committed (the code commit, `<code>` below). `pytest`, `ruff`,
  `mypy` green at it.
- `origin/main` is behind; the push in step 8 publishes every commit since
  `8ab7b73`. Decide before the day that this is fine (it is the point: the
  pre-registration needs a GitHub time).
- The dashboard's remaining QPU seconds at hand (`$rem` below).

## 1. Window setup

```powershell
cd "C:\Users\Arturo Arvizu\python\dgssp"
$env:QISKIT_IBM_TOKEN = "<paste API key>"
$env:QISKIT_IBM_INSTANCE = "<paste instance CRN>"
$d = (Get-Date -Format yyyy-MM-dd)
$rem = <remaining seconds from the dashboard>
git status --short
git log -1 --oneline
```

Expected: `git status --short` lists only the untracked-on-purpose files
(`paper/hardware/account_snapshot.json`, `paper/data/hw_dryrun_*_collect/`,
and `layer_fidelity_*.json` if not committed); `git log` shows `<code>`.

## 2. Plan

```powershell
conda run --no-capture-output -n dgssp-paper python paper/scripts/hardware_batch.py --plan --design paper/hardware/design_job3.json --backend ibm_fez --remaining-seconds $rem --date $d
```

Expected (rehearsal values in brackets):

- `24 circuits ({'pinned': 22, 'drift_repeat': 2}), 294912 shots, 12 records`;
- every pre-flight line `ok`, `snapshot target == live target: True`;
- `usage / floor 2026-10-25: 75 s / 65.9 s = 1.138` and
  `usage / floor 2026-10-28: 51 s / 46.0 s = 1.108`;
  `ESTIMATED QPU TIME = ... = <est> s` [183.0 s at a floor of 80.5 s],
  `cap 50% = <rem/2> s -> WITHIN BUDGET`; no `TRIMMED` line (if there is
  one, it names what the trim removes: stop and decide);
- `REPORT_BOUNDS = clip, RESOLVED_RULE = v2`; `alpha all = 3.176`,
  `alpha 2q = 4.055`;
- the record table: P-n2-M1-11_L1 `gentle [1.0, 1.25, 1.5]; rule [...]`,
  P-n3-M1-17_L1 `odd [1.0, 3.0, 5.0]; gentle [...]`, P-n3-M1-17_r1_L1 and
  P-n4-M1-10_r1_L1 `gentle`, six records `lambda = 1 only`;
- `rule grid P-n2-M1-11_L1: kept ... -> [1.0, 2.5, 3.0]` or `[1.0, 3.0, 4.0]`
  (depends on the simulated signal; [kept 0.4271 -> 1.0, 2.5, 3.0]) with its
  step beside `optimal_step`;
- `pinned 2Q counts vs source jobs: P-n2-M1-11_L1 71/71, P-n3-M2-17_L2 184/184,
  P-n3-M1-17_L1 212/212, P-n3-M1-17_r1_L1 112/112, P-n4-M1-10_r1_L1 233/233,
  P-n4-M1-10_r2_L1 475/475, P-n3-M2-17_L1 194/194, P-n2-M1-11_L2 71/71,
  P-n3-M1-17_L2 235/235, P-n3-M1-17_r1_L2 126/126 -> all equal`;
- `Next: review the manifest, then run --dry-run.` and exit code 0.

Paste: the whole output.

## 3. Dry run, and its collect check

```powershell
conda run --no-capture-output -n dgssp-paper python paper/scripts/hardware_batch.py --dry-run --date $d
conda run --no-capture-output -n dgssp-paper python paper/scripts/hardware_batch.py --collect --jobs paper/hardware/jobs_dryrun_$d.json
```

Expected: `12 record(s) written`; two `drift ...` lines with |z| < 2;
then `ACCEPTANCE PASS: all 12 records identical to paper/data/hw_dryrun_<d>`.

## 4. Pre-register

```powershell
conda run --no-capture-output -n dgssp-paper python paper/scripts/preregister.py --date $d
```

Expected (about 1–2 min): two `predict:` runs (alpha 3.176 all errors;
4.055 2q errors), four `simulating 24 PUBs at ...` lines (all, 2q, band_lo,
band_hi), then the Markdown, then

```
wrote paper/hardware/preregistration_<d>.json
      paper/hardware/preregistration_<d>.md
sha256 <64 hex>  paper/hardware/preregistration_<d>.json
```

In the rehearsal: Q2 `out_of_sample` decidable (predicted RMS difference
0.029), `fitted` **not decidable by design** (0.003); Q7 both layouts
decidable; Q4a eligible = P-n2-M1-11_L1 gentle and rule, P-n3-M1-17_L1 gentle,
P-n3-M1-17_r1_L1 gentle; Q4c ratio ≈ 1.9.

Paste: the Markdown and the sha256 line. Claude checks every Q1–Q9 has a
prediction or a stated "not applicable"/"not decidable", and flags any
prediction that contradicts jobs 1–2 for a pinned circuit.

## 5. Tests at the code commit

```powershell
conda run --no-capture-output -n dgssp-paper python -m pytest -q
conda run --no-capture-output -n dgssp-paper ruff check src tests paper/scripts
conda run --no-capture-output -n dgssp-paper mypy src/dgssp paper/scripts
$code = git rev-parse --short=10 HEAD
```

Expected: all green. `$code` goes on the pytest line (for a design job the
pytest line may name the code commit; the gate checks that every file changed
after it is one of this job's: `paper/hardware/` or `paper/data/*_<d>*`).

## 6. Checklist, first pass

```powershell
Copy-Item paper/hardware/CHECKLIST_template.md paper/hardware/CHECKLIST_$d.md
```

Fill every line of `CHECKLIST_<d>.md` except "Pre-registration pushed"
(leave it as in the template for now). For this job:

- Probe line: write "not run: the estimate is usage / floor of jobs
  2026-10-25 and 2026-10-28 (manifest `estimate.usage_rate`)" in place of the
  `____` blanks.
- Budget line: the estimate and `$rem` from step 2.

## 7. Commit A: the pre-registration

```powershell
git add paper/hardware/manifest_$d.json paper/hardware/circuits_$d.qpy paper/hardware/calibration_$d.json paper/hardware/jobs_dryrun_$d.json paper/hardware/preregistration_$d.json paper/hardware/preregistration_$d.md paper/hardware/CHECKLIST_$d.md paper/data/hw_dryrun_$d "paper/data/hw_predict_plan_x*_$d"
git status --short
git commit -m "Job 3 pre-registration ($d): manifest, circuits, calibration, dry run, predictions"
git push origin main
git log -1 --format="%H %cI"
git ls-remote origin refs/heads/main
```

Expected: `git status --short` before the commit shows only those paths as
staged (plus the untracked-on-purpose files); the push succeeds; `ls-remote`
prints the same hash as `git log -1`. That hash is `<A>`.

Optional, public proof without opening the repo: post
`sha256 <hex> preregistration_<d>.json, commit <A>` as a public gist now.

## 8. Commit B: the pushed line

Fill the last box of `CHECKLIST_<d>.md`:

```
- [x] Pre-registration pushed: commit <A>, sha256 <hex from step 4>, pushed at <UTC> UTC
```

```powershell
git rev-parse HEAD
(Get-FileHash paper/hardware/preregistration_$d.json -Algorithm SHA256).Hash.ToLower()
(Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
```

Tick every box, then:

```powershell
git add paper/hardware/CHECKLIST_$d.md
git commit -m "Job 3 checklist ($d): pre-registration pushed at <A>"
git push origin main
git status --short
```

Expected: push succeeds; `git status --short` shows only the
untracked-on-purpose files.

## 9. Submit

Check the device on https://quantum.cloud.ibm.com/ (online, no maintenance),
then:

```powershell
conda run --no-capture-output -n dgssp-paper python paper/scripts/hardware_batch.py --submit --date $d
```

Expected:

```
calibration entries of the job's qubits changed since --plan: 0
SUBMITTED job <id> (24 PUBs x 12288 shots) to ibm_fez in job mode.
usage estimation: ...
wrote paper/hardware/jobs_<d>.json and appended runs/jobs.jsonl
```

Refusals and what they mean (each one stops the submission):

| Message starts with | Meaning | Do |
|---|---|---|
| `REFUSED by the pre-registration gate: ... does not exist` | step 4 not run for this date | run step 4 |
| `... was made from another manifest` | the plan was redone after pre-registering | re-run steps 3–8 |
| `... is not committed` / `differs from HEAD` | step 7 incomplete | commit it |
| `HEAD is not on origin/main` | step 7/8 push missing | push |
| `REFUSED by the checklist: ... pre-registration line ...` | the pushed line is wrong | fix the line (step 8) |
| `REFUSED: the calibration of the job's qubits changed` | IBM recalibrated those qubits since step 2 | decide: re-plan (new date, back to step 2) or re-run with `--accept-recalibration` (predictions are re-run on the execution calibration in step 7 of the paper plan either way) |

## 10. Paste back

- the full output of step 9 (job id, submission time);
- `paper/hardware/jobs_<d>.json`;
- `git log -3 --oneline`.

Then wait for DONE and go to Prompt G (`--collect`, `--fetch-calibration`,
then `noise_model_check.py --fit-alpha --date <d> --label exec_insample
--calibration paper/hardware/calibration_exec_<d>.json --alphas 1 1.5 2 2.5 3
3.5 4 5 6 8`, then `hw_aggregate.py --date <d>` and
`score_preregistration.py --date <d>`).
