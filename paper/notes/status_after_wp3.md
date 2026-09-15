# Project status after WP3

As of 2026-09-14. `main` at `d793891`, clean tree. 475 tests pass, 5 skipped;
ruff and mypy green. Env `dgssp-paper` (Python 3.11), pins in
`paper/requirements-lock.txt`: qiskit 2.5.2, qiskit-ibm-runtime 0.49.0,
qiskit-aer 0.17.2, mitiq 1.1.0, numpy 2.2.6, scipy 1.17.1, matplotlib 3.11.1.
No hardware job run yet; `paper/hardware/account_facts.md` still unfilled.

## Work packages done

**WP1** — paper workspace, provenance (`paper/scripts/_common.py`), pinned env,
plotting style maps, smoke test.

**WP2 (C1)** — `paper/notes/resource_model.md`; `paper/data/resources.json`
(432 records: 36 feasible (n,b) cells × 2 item families × {positive, negated} ×
3 heavy-hex fake twins, r = 1, opt 3, SABRE seeds 0..31), `resources_fits.json`,
`paper/figures/fig_resources.pdf`. Results: `#CP = 2b(b-1) + 2·n_plus·b` per
iteration; `#2Q = r[4b(b-1) + 4·n_plus·b + g(b) + g(n)]`; routing overhead
1.25–2.08× (median 1.64); the signed encoding costs 2b single-qubit phases and
zero two-qubit gates; the guard bit is the larger cost. Survival at n = b = 8 on
fake_marrakesh: 0.0053 at r = 1, 7.6e-10 at r_opt = 4 (upper bound).

**WP3** — frozen instances (`paper/notes/instance_family.md`).

- `paper/instances/family.json`: 463 instances, master seed 20261020,
  `payload_sha256 b6a7b026…`. Family P: n ∈ {2..6}, items i.i.d. uniform in
  [1, a_max(n)], a_max = floor(15/n) = 7,5,3,3,2 (guarantees n_sum ∈ {3,4,5}).
  Family S: n ∈ {3,4,5}, same magnitudes, 1 or 2 items negated. M-classes
  {0, 1, 2, 3+}; M=0 targets are gaps strictly inside [sum_neg, sum_pos]; target
  0 excluded. 40/44 cells full; the 4 short cells are proven exhaustive
  (P n=2 M=2: 7; P n=2 M=3+: 0; P n=3 M=3+: 10; P n=6 M=0: 6).
  `r_opt = optimal_iterations(n, M)`, M=0 clamped to M=1.
  `target_is_representable` is an interval test, true for every instance
  including M=0 — it is not a feasibility test.
- `paper/instances/hardware_subset.json`: all 463 instances built at their own
  r_opt, transpiled once on fake_marrakesh (opt 3, seeds 0..31, `best_seed`
  recorded per instance), survival from median gate errors, P_ideal from seeded
  noiseless Aer at 20 000 shots. Selected 10:

  | name | n | b | M | r_opt | 2Q | 2Q depth | S | P_ideal | P_unif | P_est |
  |---|---|---|---|---|---|---|---|---|---|---|
  | P-n2-M0-06 | 2 | 4 | 0 | 1 | 137 | 112 | 0.595 | 0 | 0 | 0 |
  | P-n2-M1-11 | 2 | 3 | 1 | 1 | 76 | 69 | 0.751 | 1.000 | 0.250 | 0.813 |
  | P-n2-M2-03 | 2 | 3 | 2 | 1 | 83 | 75 | 0.731 | 0.501 | 0.500 | 0.501 |
  | P-n3-M0-06 | 3 | 4 | 0 | 2 | 370 | 279 | 0.245 | 0 | 0 | 0 |
  | P-n3-M1-17 | 3 | 3 | 1 | 2 | 241 | 208 | 0.402 | 0.942 | 0.125 | 0.454 |
  | P-n3-M2-17 | 3 | 4 | 2 | 1 | 175 | 142 | 0.515 | 1.000 | 0.250 | 0.636 |
  | P-n4-M0-09 | 4 | 5 | 0 | 3 | 1206 | 794 | 0.0104 | 0 | 0 | 0 |
  | P-n4-M1-10 | 4 | 4 | 1 | 3 | 708 | 531 | 0.0685 | 0.962 | 0.0625 | 0.124 |
  | P-n4-M2-02 | 4 | 4 | 2 | 2 | 465 | 327 | 0.170 | 0.945 | 0.125 | 0.265 |
  | S-n4-g1-M2-00 (edge) | 4 | 4 | 2 | 2 | 488 | 335 | 0.157 | 0.945 | 0.125 | 0.253 |

  Edge criterion: `P_est = S·P_ideal + (1−S)·P_uniform ≥ 2·P_uniform`, M > 0,
  r_opt ≥ 2, largest by (n, n_sum, 2Q). 55/347 feasible instances pass;
  20 with r_opt ≥ 2; no search with r_opt ≥ 2 passes beyond n = 4. The edge
  margin is 1.3% — a boundary, not a threshold.
- `paper/instances/sweep.json`: P-n3-M1-17 (r = 1..3) and P-n4-M1-10
  (r = 1..4), r_max = r_opt + 1.
- `paper/tex/family_table.tex`: appendix table (needs `booktabs`).

Library gained in WP3: `instance.max_item_value`; `random_instance(...,
n_negative, target_strategy="multiplicity", target_multiplicity, rng)`;
`solvers.classical.subset_sum_multiplicities`, `multiplicity_class`;
`execution.execute_ideal(seed_transpiler=)`;
`backends.physical_gate_counts` and `median_gate_errors()["virtual_gates"]`.
Paper scripts: `build_family.py` (`--check`), `select_hardware_subset.py`,
`build_sweep.py`, `family_table.py`.

## Corrections made during WP3 (now conventions)

1. **Survival is charged to physical gates only.** `dgssp.backends.
   physical_gate_counts(circuit, median_gate_errors(backend))`; virtual `rz`
   (~35% of 1Q ops) excluded, as it is from the median. The old count
   understated S by 1.03–1.5× (WP2) and 1.02–1.35× (WP3). Anything computing
   survival must use it.
2. **`git_state()["dirty"]`** = modified tracked files, plus untracked files
   outside `paper/{data,figures,instances,hardware,tex}/`. Untracked outputs are
   listed in `untracked_outputs`, not counted. Workflow: commit code → regenerate
   → commit artefacts.
3. `resource_model.md` exponent table corrected (dense α 2.36–2.42 → 1.42–1.90;
   the n=8 upturn is a fit-window artefact).

## Facts WP4 must build on

- **Instances come only from `paper/instances/`.** No new generation; thesis
  numbers are not reused.
- **Reproducing the ranked ISA circuit.** `hardware_subset.json` records
  `best_seed` per instance. `ZNESamplingConfig` defaults to seeds 0..63, not the
  0..31 WP3 ranked on, so default ZNE may execute a different layout from the
  one selected. `transpile_once` with `seed_min=best_seed,
  seed_max=best_seed+1` reproduces the WP3 circuit exactly.
- **Ideal references are analytic for the pathology cases.** An M=0 instance's
  ideal output is exactly uniform for every r (the oracle marks nothing and the
  diffuser fixes the uniform state). So is P-n2-M2-03's: at M/N = 1/2, one
  iteration gives |amplitude|² = 1/4 on every state. That makes it a feasible
  null control.
- **P_ideal agrees with sin²((2r+1)θ)** on all 347 feasible instances (max
  2.5σ at 20 000 shots).
- **Folded signal vs shot noise.** Signal = S^λ·(P_ideal − P_uniform), survival
  as an upper bound; σ at 4 000 shots/scale:

  | instance | λ=1 | λ=3 | λ=5 | σ |
  |---|---|---|---|---|
  | P-n2-M1-11 | 0.563 | 0.318 | 0.179 | 0.0068 |
  | P-n3-M1-17 | 0.329 | 0.053 | 0.0086 | 0.0052 |
  | P-n3-M2-17 | 0.386 | 0.102 | 0.027 | 0.0068 |
  | P-n4-M1-10 | 0.062 | 0.0003 | 1e-6 | 0.0038 |
  | P-n4-M2-02 | 0.140 | 0.0040 | 1e-4 | 0.0052 |
  | S-n4-g1-M2-00 | 0.128 | 0.0031 | 8e-5 | 0.0052 |

  With odd-integer folding, every n = 4 instance is below one σ from λ = 3.
- **Job shape.** `run_zne_scales` puts all scales of one instance in one job,
  but `experiments._run_optimized` loops over instances (N jobs), and the Mitiq
  executor samples per folded circuit. The monthly hardware run must be one job.
- **Logical folding is not live.** `fold_local_circuit` is a deprecated alias of
  `fold_transpiled`, and the Mitiq baseline receives ISA circuits.
- `decoding.solution_probability` is imported by `select_hardware_subset.py`.
- `paper/data/` already holds `smoke.json` … `smoke_003.json`; `save_record`
  never overwrites, so the next smoke run writes `smoke_004.json`.

## Open items

- Fill `paper/hardware/account_facts.md`; fake_marrakesh is a stand-in for the
  primary device. Once known, re-run `select_hardware_subset.py --backend <twin>`
  and `build_sweep.py`; the edge will likely move.
- QPU budget (WP5): the circuit count × scales × shots against ~10 min/month.
- `paper/README.md` script table lacks the WP3 scripts.

## Decisions taken for WP4 (2026-09-15)

- ZNE at n = 4 is unresolvable with odd-integer folding. WP4 adds partial
  (fractional-scale) folding of the ISA circuit, plus a resolvability flag on
  every fit.
- The hardware run must be one job. WP4 adds a cross-instance batched ZNE path,
  with Mitiq's folds included, so the Aer dry-run has the hardware job's shape.
- `future_work.md` no longer names a truncated-circuit run as a WP4 experiment;
  truncation stays out of scope for Paper A.
