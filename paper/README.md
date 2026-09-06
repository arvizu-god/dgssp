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

5. **Hardware is metered.** The IBM Open plan gives roughly ten minutes of QPU
   time a month, so hardware runs happen once a month and only after the exact
   same script has been validated against Aer with a calibration-derived noise
   model. `hardware/account_facts.md` records what the account actually allows.

## Provenance and reproducibility

`scripts/_common.py` is the single entry point for both:

```python
from _common import Record, record_provenance, save_record, load_records
```

The pinned dependency set is `_common.PINNED_PACKAGES`; the same list drives
both the provenance block and `paper/requirements-lock.txt`. Regenerate the
lock file with:

```
python paper/scripts/freeze_versions.py
```

## Quick check

```
python paper/scripts/smoke_test.py
```

Builds the `[1, 2, 3] / target 5` instance, runs it ideally, on a noisy model of
a bundled heavy-hex fake device, and through ZNE at scales `[1, 3]`, then writes
`data/smoke.json`. It should finish in well under two minutes on a laptop and is
the fastest way to confirm that a fresh environment is wired up correctly.
