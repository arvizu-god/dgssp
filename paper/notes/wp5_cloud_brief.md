# WP5-cloud — work-package brief

The prompt that starts this work in a fresh session. Written 2026-10-02 against
`main` at `4354998`. Everything below the line is the prompt; paste it verbatim.

---

PROJECT CONTEXT — Paper A (Draper–Grover Subset Sum on heavy-hex hardware)

I am Arturo Arvizu. I am converting my MSc thesis (a Grover search for the
Subset Sum Problem whose oracle is built from Draper's QFT adder) into an arXiv
preprint and a journal submission. Target: arXiv by Oct 20–27, 2026; journal EPJ
Quantum Technology or Quantum Information Processing. This preprint supports my
PhD applications (Dec 1–15 deadlines), so correctness and reproducibility matter
more than ambition.

Frozen contribution statement:
  C1  Closed-form + measured resource scaling of the Draper-oracle Grover-SSP
      circuit on heavy-hex devices, including the offset + guard-bit signed
      encoding.
  C2  Hardware execution with ISA-level-folded ZNE (scalar and pointwise
      estimators), compared against Mitiq and against a calibration-derived
      noise model, on a frozen random instance family, with bootstrap CIs.
  C3  Where the algorithm fails on current hardware (Grover-iteration sweep,
      survival estimate) and the limits of ZNE applied to sampling
      distributions (the infeasible-instance pathology).

Codebase: github.com/arvizu-god/dgssp, package `dgssp`, src layout, Qiskit 2.x,
qiskit-ibm-runtime, qiskit-aer, Mitiq. Layered modules: instance · encoding ·
decoding · solvers.draper_grover · solvers.classical · runtime · backends ·
transpilation · mitigation.zne · mitigation.mitiq_baseline · execution ·
experiments · records · metrics · api · cli.

Non-negotiable conventions:
  1. Sum encoding: register holds s + offset, offset = -sum(negative items);
     one guard bit so modulus >= 2*range_len. Only encoding.py decides this.
  2. Transpile ONCE (opt level 3, best seed), then fold the ISA circuit, submit
     folded circuits at opt level 0. Never fold logical circuits. Never
     re-search the layout between noise scales.
  3. Every sampler call goes through runtime.sample_counts. A list of circuits
     is one job.
  4. Extend existing modules; do not duplicate register math, bit ordering, or
     metric code. Keep pytest, ruff and mypy green.
  5. QPU budget is the IBM Open plan (~10 min/month). No hardware jobs in this
     work package.
  6. Thesis numbers are NOT reused.
  7. Every results file carries provenance (git commit, dirty flag, package
     versions, timestamp, backend + calibration date) via paper/scripts/_common.py.

Paper workspace: paper/{notes,scripts,data,figures,instances,hardware,tex}

---

ENVIRONMENT — two traps, read before running anything

1. The `python` on PATH is the Anaconda **base** env: Python 3.12, qiskit 2.0.2.
   That is the WRONG environment and will give wrong answers. The paper's env is
   the conda env `dgssp-paper` (Python 3.11.16, qiskit 2.5.2, qiskit-aer 0.17.2,
   qiskit-ibm-runtime 0.49.0, mitiq 1.1.0, numpy 2.2.6, scipy 1.17.1,
   matplotlib 3.11.1). Always invoke it explicitly:
   `C:/Users/Arturo Arvizu/.conda/envs/dgssp-paper/python.exe`

2. Bash heredoc patching of Python files is unreliable in this shell. Use the
   Write and Edit tools for source changes, not `python - <<'EOF'`.

Platform: Windows 11, PowerShell. 16 logical cores. Repo at
`C:\Users\Arturo Arvizu\python\dgssp`, branch `main`, clean at `4354998`.
639 tests pass (5 skipped), ruff and mypy green.

---

CURRENT STATE

WP1–WP4: see `paper/notes/status_after_wp3.md` and `status_after_wp4.md`.

WP5 (the simulation campaign) is **code-complete**:

  - `paper/scripts/sim_campaign.py` — four stages (ideal / noisy / zne / sweep),
    resumable per instance, chunked by circuits, measured ETA, throttling
    (`--max-threads`, `--max-minutes`, `--cooldown`, `--n`, `--only`, `--limit`).
  - `paper/scripts/aggregate.py` — per-instance CSV, per-cell CSV, pathology CSV,
    odd7 model comparison, `sim_summary.json`, LaTeX tables, fills
    `notes/sim_results.template.md` into `notes/sim_results.md`.
  - `paper/scripts/make_figures.py --set sim` — five figures from `paper/data`
    alone, no backend imported.
  - `paper/scripts/qpu_budget.py` — offline QPU-minute estimate for a hardware
    job shape (hardware prep, not WP5).
  - Library: `analyse_zne_item(methods=...)`, `backends.save_noise_model` /
    `load_noise_model` (calibration-target snapshot), `experiments.
    PreparedInstance` / `prepare_from_records` / `build_instance_result`.

The campaign was run **partially** on this laptop and the machine thermally shut
down twice. Local partial data, to be set aside and NOT used:
`paper/data/sim_ideal` (463), `sim_noisy_fake` (463), `sim_zne_fake_odd` (380).

DECISION TAKEN: the whole campaign is re-run **from zero on Google Cloud**, on
one machine, one commit, one environment, and the results are synced back here.
Rationale: counts depend on `seed_simulator` plus the experiment's index within
its job, so chunk composition is part of the result; one uninterrupted pass on
one machine is the only dataset that is simple to describe.

---

TASK: WP5-cloud. Make the campaign reproducible and runnable unattended on a
Google Cloud VM, then produce the setup I execute.

Work in this order. Commit after each part. Do not run the full campaign here.

### A. Fix the provenance bloat (do this first — it halves the data volume)

`paper/scripts/_common.py::git_state()` stores the full `untracked_outputs` list
(every untracked file under paper/{data,figures,instances,hardware,tex}) in the
provenance block of **every record**. Since the campaign writes into those
directories as it runs, each new record embeds a list of all the previous ones.
The output is O(N^2).

Measured on the partial local run:

```
sim_ideal         7.1 MB = content  2.8 + untracked-list  4.3 MB  (60% waste)
sim_noisy_fake   18.8 MB = content  5.2 + untracked-list 13.6 MB  (73% waste)
sim_zne_fake_odd 31.5 MB = content 12.8 + untracked-list 18.7 MB  (59% waste)
```

The first ideal record carries 0 paths and is 3 KB; a late ZNE record carries
1179 paths and is 89 KB. A from-zero full campaign (~2050 records over six stage
directories) projects to ~130 MB, of which ~90 MB is the same list repeated.
With the fix: ~39 MB.

`record_provenance()` also costs 95 ms per record (two `git` subprocesses that
re-scan a directory holding thousands of files) — about 3 minutes over a full
campaign, and it scales the wrong way.

Required:

  - Keep `dirty`, `dirty_files` (normally empty, and load-bearing — it is what
    says a result is not reproducible from its commit) and `commit` /
    `short_commit` / `branch` exactly as they are.
  - Replace the `untracked_outputs` **list** with a count, and whatever small
    fixed-size summary you judge defensible. The docstring already states the
    list is informational and that untracked *inputs* stay traceable through the
    `family_payload_sha256` digests downstream files carry — so nothing depends
    on the list itself. Say in the docstring why it is a count now.
  - Cache the git probe for the lifetime of the process. It is constant during a
    run, and calling it 2050 times is waste.
  - `tests/test_paper_provenance.py` asserts on `untracked_outputs` at lines
    153, 174, 183 and 202. Update them to the new contract rather than deleting
    them, and add one that pins the invariant that matters: a record's
    provenance size must not grow with the number of files already written.
  - Old records (`smoke*.json`, `zne_synthetic*.json`, the partial `sim_*`) must
    still load through `ResultRecord.from_dict`.

If you think a different split (e.g. dropping the list but keeping a sha256 of
it) is better, say so and argue it before writing.

### B. Pin the environment properly

  - `paper/requirements-lock.txt` pins 7 packages and its own header says it was
    generated from a **dirty** tree at commit `0c75c766`. Regenerate it
    (`paper/scripts/freeze_versions.py`) from the clean tree.
  - Add `paper/cloud/requirements-cloud.txt`: the full `pip freeze` of
    `dgssp-paper` (72 packages), which is what actually has to be reproduced on
    a VM. The 7-package lock stays as the provenance record; this is the build
    input.
  - TRAP: `.github/workflows/ci.yml` installs `.[dev,data,viz]` — **not mitiq**.
    If the VM does the same, `mitiq_is_available()` returns False, the campaign
    prints `mitiq=off` and runs to completion without the Mitiq baseline, which
    is a C2 deliverable. Locally mitiq 1.1.0 with cirq-core 1.6.1 works and `ply`
    is NOT needed (the WP4 rewrite avoided Mitiq's Qiskit path).
    Make this impossible to get wrong: the orchestrator (part C) must refuse to
    start a ZNE stage when Mitiq is absent unless `--no-mitiq` was passed
    explicitly.

### C. `paper/scripts/run_campaign.py` — the orchestrator

One entry point that runs the whole campaign unattended:

  - Stages in order: `ideal --family all`, `noisy --family all`,
    `zne --family all --grid odd`, `zne --family all --grid planned`,
    `zne --family P --grid odd7`, `sweep`.
  - Aborts the whole run if the ideal stage fails its acceptance gate — nothing
    downstream is meaningful if it does.
  - Passes `--max-threads`, `--cooldown`, `--chunk`, `--backend`, `--noise`
    through to each stage.
  - Optional `--upload gs://BUCKET/PATH`: sync `paper/data` to the bucket after
    **each stage**, so a preempted Spot VM loses at most one stage's tail.
  - Timestamped progress to stdout, and a machine-readable run log
    (`paper/data/campaign_log.json`) with per-stage wall clock, record counts
    and exit status.
  - Resumable: re-running skips stages already complete (every stage already
    skips instances already on disk).
  - A `--dry-run` that prints the stage plan and the expected record counts
    without executing.

### D. `paper/scripts/verify_campaign.py` — completeness and consistency

Run after the results come back from the cloud. Asserts, and exits non-zero on
any failure:

  - Every expected instance present in every stage directory, against
    `paper/instances/family.json` (463 for ideal/noisy/odd/planned, the Family-P
    n<=4 subset for odd7, both sweep instances × three executors).
  - Every record carries the **same** git commit, `dirty == False`, and the same
    package versions. A dataset spanning two commits or two environments is not
    one dataset.
  - No `_001`-suffixed duplicates (two runs into one directory).
  - `schema_version == 2` everywhere.
  - Shot totals match the stage's configured shots.
  - Prints a one-screen summary table and the total on-disk size.

### E. `paper/cloud/` — the VM setup

  - `bootstrap.sh`: on a clean Debian 12 VM, install Python 3.11, create a venv,
    `pip install -r paper/cloud/requirements-cloud.txt`, then
    `pip install -e ".[mitiq,viz,dev]"`, and assert that
    `mitiq_is_available()` is True and the pinned versions match.
  - `startup.sh`: the VM metadata startup script. Clone the repo at a **pinned
    commit** (passed as VM metadata, not "latest main"), run `bootstrap.sh`, run
    `run_campaign.py --upload gs://...`, then `shutdown -h now`. It must be
    idempotent: re-running on a fresh VM after a preemption continues from what
    is already in the bucket.
  - `README.md` in `paper/cloud/`: the exact PowerShell commands I run, in order,
    with what to expect from each.

Sizing (measured here, one n=6 instance, planned grid):

```
threads   s/circuit   marginal efficiency
   4        15.07        —
   8         8.51        89%
  16         5.98        71%
```

Near-linear to 8, decaying after. Full campaign from zero is ~165 min at 16
threads, so ~110 min on 32 vCPU. Memory is irrelevant — Aer truncates the idle
qubits, every simulation is <=14 qubits — so **highcpu** machine types are the
right shape. Target: one `c2d-highcpu-32` **Spot** VM, Debian 12, 50 GB disk,
`--scopes=storage-rw`, `--instance-termination-action=DELETE`. Spot is right
because the campaign is resumable per instance, so a preemption costs one chunk.
Cost is under $5 for the whole campaign; do not optimise it.

### F. One decision to put to me, not to resolve yourself

`paper/data/*.json` is tracked in git. Post-fix the campaign is ~39 MB of JSON
across ~2050 files. Lay out the options (commit as-is / gzip the records / track
only the aggregated CSVs plus a tarball / git-lfs), say which you would choose
and why, and wait for my answer before committing any campaign data.

---

ACCEPTANCE

  - `pytest`, `ruff check src tests paper/scripts`, `mypy src/dgssp` all green,
    including new tests for A, C and D.
  - A from-zero local dry run on a small slice reproduces end to end:
    `--stage ideal --limit 20`, then `noisy`, then `zne --grid planned`, then
    `verify_campaign.py` passes on that slice.
  - A fresh ideal record is ~3 KB and a fresh ZNE record ~39 KB — i.e. the
    provenance block no longer grows with the number of records written.
  - `run_campaign.py --dry-run` prints the six stages and the expected counts.
  - `paper/cloud/README.md` contains commands I can paste without editing.

END WITH

  (a) the exact PowerShell commands I run, in order — local verification first,
      then the gcloud setup, then the VM launch, then the sync back;
  (b) the acceptance check I should see at each step;
  (c) what to paste back to you.

STYLE

Direct, technically dense, no validation. If something I asked for is wrong or
underspecified, say so and propose the fix before writing code. Flag anything
you could not verify.
