# Project status after WP4

As of 2026-09-20. `main` at `4109f9a`, clean tree. 588 tests pass, 5 skipped;
ruff and mypy green. Env `dgssp-paper` (Python 3.11), pins in
`paper/requirements-lock.txt`: qiskit 2.5.2, qiskit-ibm-runtime 0.49.0,
qiskit-aer 0.17.2, mitiq 1.1.0, numpy 2.2.6, scipy 1.17.1, matplotlib 3.11.1.
No hardware job run yet; `paper/hardware/account_facts.md` still unfilled, so
fake_marrakesh remains the stand-in for the primary device.

WP1–WP3: see `paper/notes/status_after_wp3.md` (workspace and provenance;
the C1 resource model and `paper/data/resources.json`; the frozen instance
family, hardware subset and sweep in `paper/instances/`).

## WP4 (C3's methodological half) — done

Commits: `e6745a9` (library + scripts), `006e212` (artefacts), `f50e2ff`
(identifiability diagnostics), `4109f9a` (schema-2 artefact).

**New library modules**

- `dgssp.metrics` — the single implementation of every distribution number:
  `solution_probability`, `total_variation`, `classical_fidelity`,
  `hellinger`, `uniform_distribution`, `proportion_sigma`,
  `survival_estimate` (moved from `resource_scaling.py`, which re-exports it),
  and `bootstrap_metric` (multinomial percentile CIs).
  `counts_to_probs` / `solution_probability` /
  `distribution_solution_probability` moved out of `decoding`; the old imports
  work and emit `DeprecationWarning`.
- `dgssp.records` — the frozen schema. `ResultRecord` (schema_version 2),
  `scale_key`, `json_safe`. `paper/scripts/_common.Record` is now a thin
  subclass that only stamps provenance. Old `smoke*.json` still load.

**`dgssp.mitigation.zne`**

- **Partial folding.** `fold_isa(tqc, lam, seed=, virtual_gates=)` accepts any
  real `lam >= 1`: fold every gate `k = floor((lam-1)/2)` times, then a seeded
  subset of two-qubit gates once more, plus an independently seeded subset of
  the same *fraction* of physical 1Q gates. The **realised** scale is the
  exact rational `N_2q(folded)/N_2q`, within `1/N_2q` of nominal, and is what
  every fit uses. Odd integers reproduce the old integer folding gate for
  gate. Deterministic in `(circuit, lam, seed)`; subsets nested across scales.
- **Two estimators.** `mode="scalar"` (WLS on `P(solution)` with binomial
  weights; linear / quadratic / richardson / exponential with the asymptote
  fixed at `P_uniform` via `asymptote="p_uniform"`) and `mode="pointwise"`
  (per-bitstring, then `renorm="clip"` or `"simplex"` — Euclidean projection,
  Duchi et al. ICML 2008). Both are always computed; `mode` picks which one a
  batch reports.
- **Every fit carries** covariance, residuals, `dof`, `chi2_red` (`None` at
  `dof == 0`), a bootstrap CI, and two flags:
  `resolved` (two-sided: every scale >= 2 SE from `P_uniform`) and
  `identifiable` (signed: `n_informative_scales >= n_params`).
  `reported_value` is the extrapolation only when resolved **and** identified,
  otherwise the `lambda = 1` observation (`reported_source = "min_scale"`).
- `plan_scales(...)` pre-registers the scale grid from survival and signal;
  `pathology_diagnostic(...)` reports slope fractions, clipped/moved mass and
  `D_TV` for raw / clip / simplex against a reference.
- **One job.** `run_zne_batch(items, backend, cfg, include_mitiq=)` folds every
  instance at every scale, adds Mitiq's folds of the same ISA circuits at the
  same realised scales, and submits everything in ONE `sample_counts` call.
  `analyse_zne_item(...)` runs every estimator offline, so a hardware job
  fetched later is analysed by the same code.
- Synthetic helpers (no circuits): `synthetic_depolarising_counts`,
  `depolarised_probability`, `expected_extrapolation`.

**Other library changes**

- `runtime.sample_counts` returns a `CountsList` carrying `job_id`.
- `execution.exact_index_distribution(instance, r)` — exact statevector
  marginal on the index register; the pathology reference. Verified exactly
  uniform for M=0 and for P-n2-M2-03.
- `solvers.grover_success_probability(n, M, r)` — analytic `sin^2((2r+1)th)`.
- `experiments`: `InstanceResult` gained physical gate counts, `survival`,
  `best_seed`, `job_id`, `calibration_timestamp`, `p_uniform`,
  `reference_distribution`, `metrics_ci`, `zne_fit`, `pointwise`,
  `pathology`, `mitiq`, and `to_record()`. `_run_optimized` now goes through
  `run_zne_batch` (one job per backend; one job when a backend is passed).
  New `run_iteration_sweep(instance, r_values, backend, executor=...)` — every
  r-variant (and, for ZNE, every fold) in one job.

**Mitiq baseline, rebuilt**

Mitiq's Qiskit path needs `ply` (absent from this env) and renumbers qubits,
so it had never actually run here. `mitiq_fold_isa` now wraps each ISA
instruction as an opaque Cirq gate on its physical qubit and calls Mitiq's
public `fold_gates_at_random`; the circuit returns on the same layout and is
ISA-valid. Extrapolation is offline through Mitiq's own factories
(`mitiq_extrapolate`, `mitiq_bootstrap`). Mitiq measures its scale on a
fidelity-weighted count of *all* gates, so its realised two-qubit ratio
differs from the requested one and is recorded.

## Artefacts

| File | What |
|---|---|
| `paper/data/zne_synthetic_001.json` | schema 2: the C3 synthetic table (24 scalar cells x 200 seeds, 8 pointwise cells), robust statistics + Mitiq column |
| `paper/data/zne_synthetic.json` | schema 1, **superseded** (its headline `bias_mc = +581` is a cap artefact; see below) |
| `paper/data/smoke_004.json` | wiring check through the one-job path, schema 2 |
| `paper/data/sweep_ideal.json` | ideal iteration sweep, both sweep instances |

## Results WP5 must build on

- **Polynomial extrapolations are biased by construction** under exponential
  decay, and the Monte-Carlo means match the analytic bias. At S = 0.40 on
  [1,3,5] the weighted linear fit recovers 41% of the signal and Richardson
  67% (the WP3 note's "~45%" was for an *unweighted* fit). With three scales
  Richardson and quadratic are the same estimator.
- **Exponential with the known asymptote is unbiased where the design
  identifies it**: median bias -0.001 (S=0.75), -0.003 (S=0.40), +0.005
  (S=0.17) on the planned grids, coverage 0.93-0.96.
- **Odd-integer folding fails at n = 4 for a structural reason.** At S = 0.17
  the signal is 0.77 sigma at lambda=3 and 0.02 sigma at lambda=5, so only
  lambda=1 is informative and the two exponential parameters are not
  identified: chi2 is flat in the decay rate (11.1823 at rate 5, 10, 20, 50)
  and its infimum sits at rate -> infinity, where the extrapolation diverges.
  `exp_rate_max` (default 10) only bounds it: values track the bound
  (cap 5 -> 20.5, cap 10 -> 3018, cap 20 -> 1.46e7). Identifiable in 15.5% of
  seeds, resolved in 1%. **Never average a raw extrapolation over unresolved
  fits — use the median, or condition on `resolved` / `identifiable`.**
  The pre-registered grid [1, 1.5, 2] identifies 100% and resolves 98%.
- **Mitiq agrees where the fit is identified** (median bias +0.004 vs +0.005
  on the S=0.17 planned grid) and is ~3x more biased where it is not
  (-0.300 vs -0.110): its log-linear fit is bounded but not unbiased.
- **Pointwise ZNE makes an M=0 distribution worse, and the simplex does not
  save it.** With shot noise alone against the uniform reference,
  `D_TV` goes 0.025 raw -> 0.030 (linear, [1,3,5]) -> 0.239 (Richardson,
  [1,1.5,2]); simplex and clip agree to ~0.003, and simplex is closer in only
  7-39% of seeds. The mechanism is noise amplification by the extrapolation
  weights, which grow as the scales get closer together.
- **The smoke instance is past the edge**: `[1,2,3]/5` at r=2 on fake_algiers
  has S = 0.052 and P_sol(lambda=1) = 0.109 < P_uniform = 0.125, so the
  exponential fit diverges to -351 and is flagged unresolved; the reported
  value falls back to 0.109. Mitiq gives 0.077.

## Facts and costs for the campaign

- **Every one of the 463 instances already has a `best_seed`** in
  `hardware_subset.json["candidates"]`, plus `two_q_count`, `two_q_depth`,
  `physical_qubits`, `one_q_physical`, `two_q_physical`, `survival`,
  `p_ideal`, `p_uniform`, `r_opt`. Transpiling with
  `seed_min=best_seed, seed_max=best_seed+1` reproduces the ranked circuit
  exactly (tested for the 10 subset instances). **Those seeds are only valid
  on fake_marrakesh**; on any other backend the sweep must re-run — which
  costs only ~7 min for all 463 x 32 seeds.
- Instance counts: family P 343 (n = 47/70/80/80/66 for n = 2..6),
  family S 120 (n = 3,4,5). Median two-qubit counts by n:
  270 / 368 / 828 / 1658 / 2440.
- Measured per-instance wall clock on this laptop, median instance at each n,
  AerSimulator.from_backend(FakeMarrakesh) (Aer truncates the idle qubits of
  the 156-qubit ISA circuits, so width is not the problem):

  | n | 2Q | noisy, 10k shots | ZNE, 3 scales + Mitiq, 8192/scale |
  |---|---|---|---|
  | 2 | 270 | 3.1 s | 8.3 s |
  | 3 | 370 | 3.7 s | 8.5 s |
  | 4 | 828 | 4.0 s | 7.5 s |
  | 5 | 1672 | 4.9 s | 15.0 s |
  | 6 | 2447 | 9.2 s | 36.7 s |

  Ideal stage (opt 1, batched): ~0.1 s/instance. Single-seed transpile at
  opt 3: 12-54 ms. Offline analysis per instance (all estimators + two
  bootstraps): 0.1-0.4 s.
- `analyse_zne_item` currently fits **one** method (`cfg.method`). Fitting all
  four from the same counts needs either a loop over
  `replace(cfg, method=...)` in the caller or a small library addition; no
  re-sampling is involved, the counts are archived per realised scale.
- A schema-2 record already holds **both modes and both renormalisations**
  for one instance (`zne_fit`, `pointwise.clip`, `pointwise.simplex`,
  `mitiq`, `pathology`), so one record per (instance, executor, noise) is
  enough; splitting by mode/renorm would duplicate the counts.
- `save_record` never overwrites (it appends `_001`, ...), so a resumable
  campaign must check for an existing file by name before running.

## Open items

- **Reporting-rule decision (blocks the C2/C3 tables).** The frozen rule
  reports the extrapolation only when resolved *and* identified. At S = 0.40
  on odd scales the exponential is identified in 100% of seeds with median
  bias -0.006, yet `resolved` is 0.34 (lambda=5 sits at 1.6 sigma), so the
  paper would quote `lambda = 1` at bias -0.330. Switching the rule to
  "identified, with resolvability as a quality flag" needs no re-run:
  `n_informative_scales`, `value` and `value_at_min_scale` are all recorded.
- `paper/data/zne_synthetic.json` (schema 1) is superseded and still tracked:
  delete deliberately or cite `_001` only.
- `paper/hardware/account_facts.md` unfilled. Once the real device is known:
  re-run `select_hardware_subset.py --backend <twin>` and `build_sweep.py`
  (the edge instance will likely move, and every `best_seed` changes).
- The hardware pre-registration script (turning `plan_scales` +
  `hardware_subset.json` into the single monthly job) is not written.
- `sweep_ideal.json` covers the ideal executor only.
