# WP5 on Google Cloud

The simulation campaign of record runs once, from zero, on one Spot VM, at one
pinned commit, in one verified environment. Its results are synced to a bucket
after every stage and pulled back here at the end.

| File | Role |
|---|---|
| `requirements-cloud.txt` | Build input: the full `pip freeze` of `dgssp-paper` (71 distributions). Regenerate with `freeze_versions.py`. |
| `bootstrap.sh` | Debian 12 → Python 3.11 venv → the pinned wheels → `dgssp[mitiq,viz,dev,data]` → `freeze_versions.py --verify` (exact freeze, lock, Python minor, Mitiq importable). |
| `startup.sh` | Instance startup script: pinned commit → bootstrap → pytest → restore from the bucket → `run_campaign.py --upload` → `verify_campaign.py` → sync → power off. Idempotent. |
| `shutdown.sh` | Instance shutdown script: a best-effort last sync when the VM is preempted. |
| `sync.sh` | `up` / `down` between `paper/data` and the bucket; campaign outputs only, never tracked files. |

Every command below is PowerShell, run from the repository root in **one
window** (later steps reuse the variables of earlier ones). Nothing needs
editing. If the window is closed, re-run the block in step 3 that sets the
variables, and the `Start-CampaignVm` definition if you need to relaunch.

## 1. Local verification

```powershell
cd "C:\Users\Arturo Arvizu\python\dgssp"
$Py = "C:/Users/Arturo Arvizu/.conda/envs/dgssp-paper/python.exe"
git status --porcelain
```

Expect `git status --porcelain` to print **nothing**. Anything it lists makes
the tree dirty and `run_campaign.py` refuses to start. An editor's own files
belong in `.gitignore`, as `.vscode/` now is.

The laptop's partial WP5 run (three commits, two thermal shutdowns) was
discarded on 2026-10-12. Only the cloud run's results are kept. If any
`paper/data/sim_*` directory exists before the launch, `run_campaign.py`
refuses to resume on top of it unless it came from the same commit and
environment.

```powershell
& $Py -m pytest
& $Py -m ruff check src tests paper/scripts
& $Py -m mypy src/dgssp
& $Py paper/scripts/freeze_versions.py --verify
& $Py paper/scripts/freeze_versions.py --check-linux
& $Py paper/scripts/run_campaign.py --dry-run
```

Expect:

- `720 passed, 6 skipped`, in about 2 minutes. From PowerShell, where `bash` is the WSL stub, it is `716 passed, 10 skipped`: the four `bash -n` checks of the VM scripts skip there (they run under Git Bash and in CI);
- `All checks passed!`;
- `Success: no issues found in 20 source files`;
- `environment matches requirements-lock.txt and requirements-cloud.txt (71 distributions, python 3.11.16, mitiq importable)`;
- `OK: Linux cp311 resolves to exactly the 71 pinned distributions, all wheels.` (this one needs the network and takes about 30 s);
- the dry run's six-stage table (463, 463, 463, 463, 197, 6, total **2055**, all `0` on disk), every pre-flight line `ok`, and `dry run: nothing executed.`

Then the slice, end to end, into a scratch directory (about 5 minutes):

```powershell
$Slice = "$env:TEMP\dgssp-slice"
Remove-Item -Recurse -Force $Slice -ErrorAction SilentlyContinue
& $Py paper/scripts/run_campaign.py --limit 20 --stages ideal noisy zne-planned --data-dir $Slice
& $Py paper/scripts/verify_campaign.py --limit 20 --stages ideal noisy zne-planned --data-dir $Slice
```

Expect:

- three stages `OK (exit 0)`, then `campaign complete (exit 0)`, in about 1 minute (measured: 66 s);
- then `PASS: one complete campaign, one commit, one environment.`;
- `KB/rec` of 3.2 for `ideal`, 3.8 for `noisy` and 18.5 for `zne-planned`. The first 20 instances are all `n = 2`; larger `n` gives larger records.

## 2. Google Cloud, once

Four parts, **in order, one at a time**. 2a and 2b are interactive, so don't
paste them together with anything else.

**2a. Install the Google Cloud CLI** (skip this if `gcloud version` already works):

```powershell
winget install --id Google.CloudSDK --exact
```

The installer may open its own window; accept its defaults. Then **close
PowerShell and open a new window**, so it sees the new `gcloud` command, and
check it:

```powershell
cd "C:\Users\Arturo Arvizu\python\dgssp"
$Py = "C:/Users/Arturo Arvizu/.conda/envs/dgssp-paper/python.exe"
gcloud version
```

Expect a version table starting with `Google Cloud SDK`. If `winget` is not
available, use the installer from <https://cloud.google.com/sdk/docs/install>.

**2b. Sign in and choose the project:**

```powershell
gcloud init
```

It opens the browser to sign in with your Google account, then asks you to
pick a project or create a new one (any name, e.g. `dgssp-paper`). Answer
`n` if it offers to set a default region and zone. Expect it to end with
`Your Google Cloud SDK is configured and ready to use!`.

**2c. Billing.** Compute Engine refuses to start any VM in a project without
billing. Open <https://console.cloud.google.com/billing/linkedaccount> and
make sure the project shows a linked billing account. If the account is a
**free trial**, click **Activate full account** (or **Upgrade**). The trial
allows at most 8 CPU cores running at once, and the campaign VM needs 32
vCPUs. Upgrading keeps the remaining trial credit, which covers this run.

**2d. Bucket, permissions, quota:**

```powershell
$Project = gcloud config get-value project
$Region = "us-central1"
$Bucket = "gs://$Project-dgssp"
gcloud services enable compute.googleapis.com storage.googleapis.com
gcloud storage buckets create $Bucket --location=$Region --uniform-bucket-level-access
$ProjectNumber = gcloud projects describe $Project --format="value(projectNumber)"
gcloud storage buckets add-iam-policy-binding $Bucket "--member=serviceAccount:$ProjectNumber-compute@developer.gserviceaccount.com" --role=roles/storage.objectAdmin
gcloud compute regions describe $Region --flatten="quotas[]" --format="table(quotas.metric,quotas.limit,quotas.usage)" | Select-String -Pattern "C2D_CPUS|PREEMPTIBLE_CPUS|^CPUS "
gcloud compute project-info describe --flatten="quotas[]" --format="table(quotas.metric,quotas.limit,quotas.usage)" | Select-String -Pattern "CPUS_ALL_REGIONS"
```

Expect:

- **Enabling the services** takes up to a minute. If it fails with a billing message, part 2c is not done yet.
- **Bucket name taken:** if bucket creation fails with *already exists*, someone else owns that name. Set `$Bucket = "gs://$Project-dgssp-wp5"` and re-run the last five commands.
- **Bucket access:** the IAM binding gives the VM's default service account read/write on this bucket only. If it says the service account does not exist, Compute Engine is still being set up: wait a minute and run that line again.
- **Quota:** the limit minus usage must be **≥ 32** for `C2D_CPUS`, `CPUS` and `CPUS_ALL_REGIONS`. If it isn't, request an increase (Console → IAM & Admin → Quotas). Alternatively, replace `c2d-highcpu-32` with `c2d-highcpu-16` in step 3: it takes about twice as long and changes no count, because thread count doesn't enter the draws.

## 3. Launch

**Run 1 and run 2.** Run 1 (commit `ceeb4f3`, bucket folder `wp5/`) ran
2026-10-13 from 04:45 to 10:43 UTC. It stopped inside the `zne-planned` stage
on an analysis bug: the model comparison tried a 3-parameter fit on a
2-scale plan. That bug is fixed, and run 2 starts from zero in the bucket
folder `wp5-run2/`. Run 1's data stays in `wp5/` only until run 2's counts have
been compared with it; then it is deleted. Only run 2 is kept.

```powershell
git status --porcelain
git push origin main
$Commit = git rev-parse HEAD
git ls-remote origin main
$Vm = "dgssp-wp5"
$Prefix = "$Bucket/wp5-run2"
function Start-CampaignVm {
    $global:Zone = $null
    foreach ($z in "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f") {
        gcloud compute instances create $Vm --zone=$z `
            --machine-type=c2d-highcpu-32 --provisioning-model=SPOT `
            --instance-termination-action=DELETE --max-run-duration=14h `
            --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=50GB `
            --scopes=storage-rw `
            "--metadata=dgssp-commit=$Commit,dgssp-bucket=$Prefix" `
            "--metadata-from-file=startup-script=paper/cloud/startup.sh,shutdown-script=paper/cloud/shutdown.sh" | Out-Host
        if ($LASTEXITCODE -eq 0) { $global:Zone = $z; break }
    }
    if ($global:Zone) { "launched $Vm in $global:Zone at $Commit" } else { "no zone had Spot capacity; try again later" }
}
Start-CampaignVm
```

Expect:

- `git status --porcelain` prints nothing.
- `git ls-remote` shows the same SHA as `$Commit`; the VM clones from GitHub, so the commit has to be pushed.
- Then a table with `STATUS RUNNING` and `launched dgssp-wp5 in us-central1-…`. A zone without Spot capacity prints an error and the loop moves on to the next.

What the flags buy:

- **Spot:** the campaign resumes per chunk, and it syncs to the bucket every 15 minutes, so a preemption costs at most about 15 minutes of work.
- **`DELETE` on preemption:** nothing is left billing.
- **`--max-run-duration=14h`:** caps a VM that hangs. The full run measures about 9–10 hours, so the cap leaves room for slower Spot hardware.
- **`storage-rw` scope:** the VM can reach the bucket and nothing else.

## 4. Watch

```powershell
gcloud compute instances describe $Vm --zone=$global:Zone --format="value(status)"
gcloud compute instances get-serial-port-output $Vm --zone=$global:Zone 2>$null | Select-String -Pattern "dgssp:|stage |FAIL|PASS" | Select-Object -Last 25
gcloud storage cat "$Prefix/data/campaign_stdout.log" | Select-Object -Last 30
```

The first two commands work at any time. The third works from the first sync on, which comes after the bootstrap and tests, about 6 minutes in.

Times measured on run 1; the `zne-planned` time is an estimate, because run 1 stopped in it:

| Elapsed | Serial log shows |
|---|---|
| ~1 min | `dgssp: boot on dgssp-wp5: commit=…`, `checked out …` |
| ~2 min | `bootstrap: OK -- Python 3.11.2` (it passed the exact-environment check) |
| ~4 min | the test suite has passed; `starting run_campaign.py`; pre-flight all `ok`; `initial sync … ok` |
| ~5 min | `stage 1/6 ideal` … `OK (exit 0)` (0.8 min) |
| ~1 h | `stage 2/6 noisy` … `OK` (59 min) |
| ~5.3 h | `stage 3/6 zne-odd` … `OK` (4.3 h) |
| ~8–8.5 h | `stage 4/6 zne-planned` … `OK` (estimated 3 h) |
| ~9–10 h | `zne-odd7` (26 min) and `sweep` (1 min); `campaign complete (exit 0)`, `PASS: one complete campaign …`, `finished with status 0`, `powering off` |

A few instances cost 30–50 times more than their neighbours, and doubling the
cores barely helps them. During such a chunk the log can show no new line for
20–30 minutes while the CPU graph stays near 100%. That is normal. Between
stages there is `upload: ok`, and during a stage a `periodic sync: ok` every
15 minutes.

Status goes `RUNNING` → `TERMINATED` when it finishes. Any `FAIL` line powers the VM off too: read the line, fix, then delete and relaunch (step 5). Status codes:

- 1: a step failed;
- 2: metadata missing;
- 3: a sync failed;
- 4: the campaign ran but did not verify.

## 5. If it was preempted

When `describe` says the instance is **not found** while the campaign was
running, it was preempted (or hit the 14-hour cap) and deleted. Relaunch with
the same commit; the new VM restores the bucket and continues:

```powershell
Start-CampaignVm
```

To relaunch automatically instead, leave this running (at most 3 relaunches, checked every 5 minutes):

```powershell
$Relaunches = 0
while ($true) {
    $State = gcloud compute instances describe $Vm --zone=$global:Zone --format="value(status)" 2>$null
    if ($State -eq "TERMINATED") { "finished (TERMINATED)"; break }
    if (-not $State) {
        if ($Relaunches -ge 3) { "deleted 3 times; stopping the watcher"; break }
        $Relaunches++
        "VM gone; relaunch $Relaunches"
        Start-CampaignVm
    }
    Start-Sleep -Seconds 300
}
```

Two rules for relaunching:

- **Same commit, same machine type.** A resumed stage re-runs whole the chunks a preemption cut, so its counts are bit-identical to an uninterrupted run. That holds on the same code and the same CPU family.
- **To start over after a `FAIL`,** delete the stopped VM first with `gcloud compute instances delete $Vm --zone=$global:Zone --quiet`, then run `Start-CampaignVm`.

## 6. Sync back and verify

```powershell
gcloud compute instances describe $Vm --zone=$global:Zone --format="value(status)"
gcloud storage cat "$Prefix/data/campaign_stdout.log" | Select-Object -Last 40
gcloud storage rsync --recursive "$Prefix/data" paper/data
& $Py paper/scripts/verify_campaign.py --commit $Commit
git status --porcelain
gcloud compute instances delete $Vm --zone=$global:Zone --quiet
```

Expect:

- `TERMINATED`.
- The log's tail ends in `PASS` and `finished with status 0`.
- `rsync` copies 6 `sim_*` directories (2055 records, about 25–30 MB; run 1 measured 23.6 MB for its 1752) plus `campaign_log.json` and `campaign_stdout.log`, and nothing else.
- `verify_campaign.py` prints the six-stage table with `missing 0`, `dup 0` and `bad 0`, a single `commit` (= `$Commit`), `dirty 0`, `= requirements-lock.txt`, `python 3.11.2`, then `PASS`.
- `git status --porcelain` shows only untracked `paper/data/` entries.
- Deleting the stopped VM ends the disk's billing.

## 7. Commit the data

Decision F: the campaign files go into git as they are, about 33 MB on disk
and about 7 MB inside git, which compresses them. `.gitattributes` keeps them
byte-for-byte as the VM wrote them. Only after `verify_campaign.py` printed
`PASS`:

```powershell
git add paper/data/sim_* paper/data/campaign_log.json paper/data/campaign_stdout.log
git status --porcelain
git commit -m "WP5: simulation campaign data, cloud run at $Commit, verified"
git status --porcelain
```

Expect:

- after `git add`: `A` lines only, under `paper/data/sim_*/` (2055 files) plus the two log files;
- after the commit: nothing.

Keep the bucket until this commit exists. Only after that: `gcloud storage rm --recursive $Bucket`.
At submission, deposit the same files on Zenodo for a DOI to cite in the
data-availability statement.

## Cost

A `c2d-highcpu-32` Spot VM bills only while `RUNNING`. Run 1 ran for 6 hours. A full run is about 9–10 hours, and `--max-run-duration=14h` bounds the worst case. The actual cost per run is in the billing report (step 2c), which updates with a few hours' delay. The bucket holds about 25 MB per run.
