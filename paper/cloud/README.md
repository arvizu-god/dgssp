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

# The laptop's partial WP5 run (three commits, two thermal shutdowns) goes
# outside the repository. It is never used, and run_campaign.py refuses to
# resume on top of it.
New-Item -ItemType Directory -Force ..\dgssp-wp5-laptop-partial | Out-Null
Move-Item paper\data\sim_ideal, paper\data\sim_noisy_fake, paper\data\sim_zne_fake_odd ..\dgssp-wp5-laptop-partial\
git status --porcelain
```

Expect `git status --porcelain` to print **nothing**.

```powershell
& $Py -m pytest
& $Py -m ruff check src tests paper/scripts
& $Py -m mypy src/dgssp
& $Py paper/scripts/freeze_versions.py --verify
& $Py paper/scripts/freeze_versions.py --check-linux
& $Py paper/scripts/run_campaign.py --dry-run
```

Expect:

- `721 passed, 5 skipped`, in about 2 minutes;
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

- three stages `OK (exit 0)`, then `campaign complete (exit 0)`;
- then `PASS: one complete campaign, one commit, one environment.`, with `KB/rec` around 3 for `ideal` and around 40 for `zne-planned`.

## 2. Google Cloud, once

```powershell
gcloud version
gcloud auth login
gcloud config get-value project
```

If `gcloud version` fails, install the CLI from
<https://cloud.google.com/sdk/docs/install> and open a new window. If the last
line prints nothing, run `gcloud init` and pick (or create) the project.

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

- **Bucket name taken:** if bucket creation fails with *already exists*, someone else owns that name. Set `$Bucket = "gs://$Project-dgssp-wp5"` and re-run the last five commands.
- **Bucket access:** the IAM binding gives the VM's default service account read/write on this bucket only.
- **Quota:** the limit minus usage must be **≥ 32** for `C2D_CPUS`, `CPUS` and `CPUS_ALL_REGIONS`. If it isn't, request an increase (Console → IAM & Admin → Quotas). Alternatively, replace `c2d-highcpu-32` with `c2d-highcpu-16` in step 3: it takes about twice as long and changes no count, because thread count doesn't enter the draws. A free-trial account is capped at 8 vCPUs and cannot run this shape.

## 3. Launch

```powershell
git status --porcelain
git push origin main
$Commit = git rev-parse HEAD
git ls-remote origin main
$Vm = "dgssp-wp5"
$Prefix = "$Bucket/wp5"
function Start-CampaignVm {
    $global:Zone = $null
    foreach ($z in "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f") {
        gcloud compute instances create $Vm --zone=$z `
            --machine-type=c2d-highcpu-32 --provisioning-model=SPOT `
            --instance-termination-action=DELETE --max-run-duration=8h `
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

- **Spot:** the campaign resumes per chunk, so a preemption costs at most the chunk in flight.
- **`DELETE` on preemption:** nothing is left billing.
- **`--max-run-duration=8h`:** caps a VM that hangs.
- **`storage-rw` scope:** the VM can reach the bucket and nothing else.

## 4. Watch

```powershell
gcloud compute instances describe $Vm --zone=$global:Zone --format="value(status)"
gcloud compute instances get-serial-port-output $Vm --zone=$global:Zone 2>$null | Select-String -Pattern "dgssp:|stage |FAIL|PASS" | Select-Object -Last 25
gcloud storage cat "$Prefix/data/campaign_stdout.log" | Select-Object -Last 30
```

The first two commands work at any time. The third works from the first sync on, which comes after the bootstrap and tests, about 6 minutes in.

| Elapsed | Serial log shows |
|---|---|
| ~1 min | `dgssp: boot on dgssp-wp5: commit=…`, `checked out …` |
| ~4 min | `bootstrap: OK -- Python 3.11.2` (after `environment matches … (71 distributions, …, mitiq importable)`) |
| ~7 min | `running the test suite` ends; `starting run_campaign.py`; pre-flight all `ok`; `initial sync … ok` |
| then | `stage 1/6 ideal` … `OK (exit 0)`, an `upload: ok` after every stage, `stage 6/6 sweep` |
| ~2 h | `campaign complete (exit 0)`, `PASS: one complete campaign …`, `finished with status 0`, `powering off` |

Status goes `RUNNING` → `TERMINATED` when it finishes. Any `FAIL` line powers the VM off too: read the line, fix, then delete and relaunch (step 5). Status codes:

- 1: a step failed;
- 2: metadata missing;
- 3: a sync failed;
- 4: the campaign ran but did not verify.

## 5. If it was preempted

When `describe` says the instance is **not found** while the campaign was
running, it was preempted (or hit the 8-hour cap) and deleted. Relaunch with
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
- `rsync` copies 6 `sim_*` directories (2055 records, about 39 MB) plus `campaign_log.json` and `campaign_stdout.log`, and nothing else.
- `verify_campaign.py` prints the six-stage table with `missing 0`, `dup 0` and `bad 0`, a single `commit` (= `$Commit`), `dirty 0`, `= requirements-lock.txt`, `python 3.11.2`, then `PASS`.
- `git status --porcelain` shows only untracked `paper/data/` entries.
- Deleting the stopped VM ends the disk's billing.

**Do not commit `paper/data` yet**: how the campaign data is stored in git is decision F.

Keep the bucket until the data is committed. Only after that: `gcloud storage rm --recursive $Bucket`.

## Cost

A `c2d-highcpu-32` Spot VM bills only while `RUNNING`. The console shows its hourly price on the create page; at Spot rates, about 2 hours of campaign plus 10 minutes of setup stays under the $5 the plan allows. `--max-run-duration=8h` bounds the worst case. The bucket holds about 40 MB.
