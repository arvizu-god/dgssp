# Paper A workspace — Draper–Grover Subset Sum on heavy-hex hardware

Everything that turns the `dgssp` package into the preprint lives here. The
package (`src/dgssp/`) holds reusable, tested library code; this directory holds
the *experiments*: what was run, on what, with which numbers coming out.

## Layout

| Directory    | Contents                                                                                      | Tracked in git |
|--------------|-----------------------------------------------------------------------------------------------|----------------|
| `notes/`     | Working notes, derivations, decisions, to-do lists. Markdown.                                  | yes            |
| `scripts/`   | Every runnable experiment and figure script. One script = one artefact.                        | yes            |
| `data/`      | Results. JSON only, written by scripts, read by figure scripts.                                | yes            |
| `figures/`   | Generated figures. `.pdf` is the paper asset; `.png` is a throwaway preview.                   | `.pdf` only    |
| `instances/` | Frozen random instance families (the exact items/targets the paper reports).                   | yes            |
| `hardware/`  | Account facts, device notes, job ledgers — anything about the real machines.                    | yes (`*.json`, `*.md`) |
| `tex/`       | The manuscript source.                                                                          | yes            |

## Rules

1. **`data/` is never edited by hand.** Every file in it is produced by a script
   in `scripts/` and carries a provenance block (git commit, dirty flag, pinned
   package versions, UTC timestamp, host, Python version, and — for hardware or
   noise-model runs — the backend name and its calibration timestamp). If a
   number in the paper cannot be traced to a file in `data/`, it does not go in
   the paper.

2. **Figures regenerate from `data/`.** A figure script reads JSON from `data/`
   and writes to `figures/`. It never runs a circuit. This is what makes a
   referee's "please replot with X on a log axis" a thirty-second job rather
   than a re-run of the QPU budget.

3. **Results files are append-only.** `save_record()` refuses to overwrite: a
   collision gets a numeric suffix (`smoke.json` → `smoke_001.json`). Delete
   deliberately; never silently.

4. **Nothing here re-derives library logic.** Register math lives in
   `dgssp.encoding`, bit ordering in `dgssp.decoding`, circuit size metrics in
   `dgssp.transpilation.transpiled_metrics`, sampling in
   `dgssp.runtime.sample_counts`. Scripts call those; they do not reimplement
   them.

5. **One stage at a time.** `sim_campaign.py` resumes by skipping instances
   whose record already exists, so an interrupted stage continues and a
   finished one re-runs as a no-op. Two campaigns writing into the same
   directory at once defeat that check -- both see the file missing -- and the
   second run's records land beside the first as `<instance>_001.json`.
   Aggregation keys by instance name, so nothing is double-counted, but the
   compute is wasted; the script reports any duplicates it finds.

6. **Hardware is metered.** The IBM Open plan gives roughly ten minutes of QPU
   time a month, so hardware runs happen once a month and only after the exact
   same script has been validated against Aer with a calibration-derived noise
   model. `hardware/account_facts.md` records what the account actually allows.

## Provenance and reproducibility

`scripts/_common.py` is the single entry point for both:

```python
from _common import Record, record_provenance, save_record, load_records
```

The pinned dependency set is `_common.PINNED_PACKAGES`; the same list drives
both the provenance block and `paper/requirements-lock.txt`. That lock is the
provenance record (7 packages). `paper/cloud/requirements-cloud.txt` is the
build input: the full `pip freeze` (every distribution) that a cloud VM
installs to reproduce the environment exactly. One command regenerates both,
from one interpreter, and should be run from a clean tree:

```
python paper/scripts/freeze_versions.py                # write both
python paper/scripts/freeze_versions.py --verify       # this interpreter == both files, mitiq importable
python paper/scripts/freeze_versions.py --check-linux  # pins resolve on Linux cp311 from wheels alone
```

Each record's provenance block holds the git commit, the dirty flag and the
list of dirty files, but only a *count* of untracked outputs: a list would
grow with every record a campaign writes. The git probe runs once per process.

## Scripts

| Script | Produces | For |
|---|---|---|
| `freeze_versions.py` | `requirements-lock.txt` | environment of record |
| `describe_account.py` | stdout, feeds `hardware/account_facts.md` | plan limits, devices |
| `smoke_test.py` | `data/smoke.json` (next free suffix) | wiring check; since WP4 the ZNE leg is the one-job path with schema-2 records |
| `resource_scaling.py` | `data/resources.json`, `data/resources_fits.json`, `figures/fig_resources.pdf` | WP2 / C1 |
| `build_family.py` | `instances/family.json` (`--check` re-derives it) | WP3 frozen instance family |
| `select_hardware_subset.py` | `instances/hardware_subset.json` | WP3 hardware subset + edge instance |
| `build_sweep.py` | `instances/sweep.json` | WP3 iteration-sweep instances |
| `family_table.py` | `tex/family_table.tex` | WP3 appendix table |
| `zne_synthetic.py` | `data/zne_synthetic.json` | WP4 / C3: bias, coverage, resolvability of the ZNE estimators; clip vs simplex (no circuits) |
| `iteration_sweep.py` | `data/sweep_<executor>.json` | WP4 / C3: P_sol vs Grover iterations, one job per instance |
| `sim_campaign.py` | `data/sim_<stage>[_<noise>][_<grid>]/<instance>.json` | WP5: the whole simulation campaign, one stage per command, resumable |
| `run_campaign.py` | the six `sim_*` stages, `data/campaign_log.json` | WP5: the campaign unattended, gated, logged, optionally synced to a bucket |
| `aggregate.py` | `data/sim_*.csv`, `data/sim_summary.json`, `tex/sim_table_<grid>.tex`, `notes/sim_results.md` | WP5: every table the text quotes |
| `make_figures.py` | `figures/fig_sim_*.pdf` | WP5: the five simulation figures, from `data/` alone |
| `qpu_budget.py` | stdout, optional `hardware/qpu_budget.json` | QPU minutes a job shape would cost, offline from the device's calibrated durations |

`resource_scaling.py` is the C1 measurement: it checks every closed form in
`notes/resource_model.md` against the circuit the solver builds (144 asserts),
transpiles the grid on the heavy-hex fake twins of the Open-plan Heron devices,
and writes the data, the fits and the figure. Roughly six minutes on a laptop.
`--logical-only` skips transpilation; `--figure-only` replots from the JSON
without touching a transpiler, which is rule 2 above in executable form.

## Quick check

```
python paper/scripts/smoke_test.py
```

Builds the `[1, 2, 3] / target 5` instance, runs it ideally, on a noisy model of
a bundled heavy-hex fake device, and through the single-job ZNE path at scales
`[1, 1.5, 2, 3]` (partial folding, scalar exponential estimator, pointwise
diagnostics, Mitiq folds when installed), then writes the next free
`data/smoke*.json`. It should finish in about a minute on a laptop and is the
fastest way to confirm that a fresh environment is wired up correctly.

## WP5 run order

The campaign of record runs in one pass on one cloud VM, through the
orchestrator (`paper/cloud/README.md` has the exact commands):

```
python paper/scripts/run_campaign.py --dry-run    # the six stages, expected counts, pre-flight
python paper/scripts/run_campaign.py              # all six stages, logged to data/campaign_log.json
python paper/scripts/verify_campaign.py           # one commit, one environment, nothing missing
```

`run_campaign.py` refuses to start from a dirty tree, from an environment that
differs from the pinned files, without Mitiq when a ZNE stage is planned, or on
top of records from another commit or environment; it stops the run if the
ideal gate fails. Underneath, it runs these commands, one stage each, on
`fake_marrakesh` (the frozen twin) or on `--noise live --backend <device>` once
the real device is known. Durations are measured on this laptop; the script
prints its own measured ETA after the first chunk.

```
python paper/scripts/sim_campaign.py --stage ideal --family all     #  ~1.5 min
python paper/scripts/sim_campaign.py --stage noisy --family all     #    ~5 min
python paper/scripts/sim_campaign.py --stage zne   --family all --grid both
                                                                    #  ~140 min
python paper/scripts/sim_campaign.py --stage zne   --family P --grid odd7
                                                                    #   ~20 min
python paper/scripts/sim_campaign.py --stage sweep                  #    ~1 min
python paper/scripts/aggregate.py                                   #   <1 min
python paper/scripts/make_figures.py --set sim                      #   <1 min
```

The `ideal` stage is the gate: it writes nothing until every instance has been
compared with `sin^2((2 r_opt + 1) theta)` and with the `p_ideal` recorded in
`instances/hardware_subset.json`, and the stage as a whole has been judged (see
the script's docstring on why those per-instance thresholds are applied with a
multiplicity correction). If it stops, nothing downstream should be run.

### Splitting a long stage

Aer takes every core and holds it for the whole run. On a laptop that is a
thermal event, not a speed-up. `--max-threads N` caps it, `--max-minutes M`
stops cleanly at a chunk boundary, `--cooldown S` idles between chunks, and
`--n 2 3 4` selects index-register widths (an `n = 6` instance costs about 26 s
against 4 s at `n = 2`, so `n = 6` alone is a third of the family).

```
python paper/scripts/sim_campaign.py --stage zne --family all --grid odd `
    --n 2 3 4 --max-threads 8 --cooldown 10 --max-minutes 45
```

None of this changes a number. With a fixed `seed_simulator` every experiment
draws from its own seeded stream, so one thread and sixteen give bit-identical
counts and a throttled run merges with an unthrottled one. What *does* change
the counts is an instance's *position* in its job: Aer seeds experiment `i`
from `seed_simulator` and `i`, and from nothing else. So the chunk plan is
drawn over the whole selection and held fixed across sittings: a resumed stage
re-runs whole only the chunks that still hold a missing instance, writes only
the missing records, and gives exactly the counts an uninterrupted pass would
have. Both facts are pinned in `tests/test_paper_sim_campaign.py`. Changing
the *selection* between sittings (`--chunk`, `--n`, `--only`, `--limit`) does
move instances, and gives them different (equally valid) draws; `--max-minutes`,
`--max-threads` and `--cooldown` do not. Every record carries its absolute
`chunk_index`, `chunk_circuits` and `max_parallel_threads`.
