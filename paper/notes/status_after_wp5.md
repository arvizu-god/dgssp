# Project status after WP5

As of 2026-10-14. WP5 (the simulation campaign) is done: run once, from zero,
on one Google Cloud VM, at one commit, in one verified environment. The data
are committed. Nothing here has been pushed yet. No hardware job has run; the
noise model is still the `fake_marrakesh` snapshot.

WP1–WP4: see `status_after_wp3.md` and `status_after_wp4.md`. WP5-cloud setup
(provenance fix, pinned environment, orchestrator, verifier, VM scripts): see
`wp5_cloud_brief.md` and `paper/cloud/README.md`.

## The campaign of record

| | |
|---|---|
| Code | commit `21d0075a8d8ba6700b626c61574269202230ca43` |
| Machine | `c2d-highcpu-32` Spot VM, Debian 12, Python 3.11.2, `us-central1-a` |
| Wall clock | 2026-10-13 23:15 → 2026-10-14 09:22 UTC (10.1 h) |
| Environment | 71 pinned distributions; the 7 key ones = `requirements-lock.txt` |
| Records | 2055 (ideal 463, noisy 463, zne-odd 463, zne-planned 463, zne-odd7 197, sweep 6), 32.0 MB |
| Verify | `verify_campaign.py --commit 21d0075…`: PASS (one commit, dirty 0, one package set, schema 2, shot totals) |
| Ideal gate | worst deviation 2.71σ / 3.08σ, 2 instances over 3σ (chance predicts about 1.3 of 463); identical to run 1 |

Stage times (minutes): ideal 0.8, noisy 58.8, zne-odd 257.2, zne-planned 257.9,
zne-odd7 26.4, sweep 0.9.

Data: `paper/data/sim_*/`, `campaign_log.json`, `campaign_stdout.log`,
committed byte-for-byte (decision F = A). Zenodo deposit with a DOI at
submission.

## Reproducibility check (a sentence for the paper)

Run 1 (commit `ceeb4f3`, same machine type) stopped on an analysis bug in the
`zne-planned` stage, after writing 1752 records. Run 2 started from zero at the
fixed commit. Comparing the 1752 records present in both runs:

- **every counts histogram is bit-identical**: 1767 histograms, 31 338 176
  shots;
- every other field is identical too (all fits, bootstrap CIs, metrics),
  except `job_id` and the new `applicable` flag the fix added.

Suggested wording: *"Re-running the campaign from scratch on a fresh VM
reproduced all 31.3 million sampled shots of the 1752 records the two runs
share bit-for-bit."*

Run 1's data (`gs://dgssp-paper-2026-dgssp/wp5/`) were deleted on 2026-10-14,
after this comparison, on Arturo's OK. Only run 2 is kept.

## Aggregation and figures

`aggregate.py` and `make_figures.py --set sim` ran on the committed data:
`paper/data/sim_{per_instance,cells,pathology,models}.csv`, `sim_summary.json`,
`paper/tex/sim_table_{odd,odd7,planned}.tex`, `paper/notes/sim_results.md`,
five figures `paper/figures/fig_sim_*.pdf` (PNGs are ignored).

Fix made on the way: `build_model_comparison` now counts a fit that was never
attempted (a quadratic on a 2-scale planned grid) as `excluded_not_applicable`,
not as `not_identifiable`. On the planned grid that is 18 quadratic fits.

Headline numbers (full set in `sim_results.md`):

| grid | resolved | identifiable | median signal recovery |
|---|---|---|---|
| planned | 0.44 | 0.55 | 0.25 |
| odd | 0.11 | 0.32 | 0 |
| odd7 | 0.10 | 0.41 | 0 |

- Partial folding works as intended: at the same shots, the planned grid
  identifies the fit far more often than the odd grid for n = 3–4
  (0.73 / 0.64 vs 0.56 / 0.29). At n ≥ 5 both collapse, and the reported
  P(solution) bias is about −0.9 on both grids.
- M = 0 pathology: pointwise ZNE takes median D_TV from 0.023 at λ = 1 to
  0.86 (clip) and 0.94 (simplex) on both grids. The simplex projection does not
  rescue it.
- Best model (odd7, identified fits only): Richardson has the lowest median
  error (0.05) but on 16 of 137 fits; exponential 0.10 on 81; linear 0.53 on 81.

## Next

1. Write the `TEXT TO WRITE` paragraphs of `sim_results.template.md` from these
   numbers (C2, C3).
2. Decide the WP4 reporting-rule question from `sim_cells.csv`
   (`bias_zne_scalar_value` vs `bias_zne_scalar_reported`).
3. Hardware: fill `paper/hardware/account_facts.md`, pick the device, run the
   hardware subset within the Open-plan budget.
4. Cleanup on Arturo's OK: delete the bucket (it now holds only `wp5-run2/`)
   after the Zenodo deposit.

## Known, not fixed

- Local `SamplerV2` deep-copies the noisy backend per job (~3 s/job);
  `resolve_fake_backend` instantiates every fake device (~5 s per stage).
- A few instances cost 30–50× their neighbours and do not scale with cores;
  running experiments in parallel might help a future campaign (untested).
- `sim_campaign`'s noisy stage writes records before its `family.json` check.
- CI does not install Mitiq; `mypy paper/scripts` has 9 pre-existing errors.
