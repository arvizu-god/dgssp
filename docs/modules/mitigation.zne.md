# `dgssp.mitigation.zne`

## What the module does

Zero-noise extrapolation for a *sampling* algorithm, with two estimators and
the choice explicit:

- **scalar** (the default) — extrapolate one proportion, `P(solution)`, by
  weighted least squares with binomial weights: `linear`, `quadratic`,
  `richardson` or `exponential` (`a + b·exp(−c·λ)`, `a` fixed at `P_uniform`
  in the paper). Every fit returns its covariance, residuals, degrees of
  freedom, `chi2_red`, a bootstrap interval and a **resolvability flag**.
- **pointwise** — extrapolate every bitstring's probability independently, then
  map back to a distribution by clip-and-renormalise or by Euclidean
  projection onto the simplex. This is the estimator whose failure on
  infeasible instances C3 characterises (`pathology_diagnostic`).

Two ordering bugs from the thesis stay fixed: **transpile first, then fold**
(folds land on ISA gates; folded circuits are submitted at optimization level
0) and **search the layout once** (every scale folds the same physical
circuit). To reproduce a WP3-ranked circuit use `seed_min=best_seed,
seed_max=best_seed+1`; the config's default range is 0..63, WP3 ranked on
0..31.

**Scales are real numbers ≥ 1.** Odd integers fold every gate uniformly;
other scales fold a seeded subset once more. Fits use the **realised** scale —
the exact rational ratio of two-qubit gate counts — never the nominal one.

**One job.** `run_zne_batch` folds many instances at all their scales (plus
Mitiq's folds of the same circuits) and submits everything through one
`runtime.sample_counts` call. Analysis (`analyse_zne_item`) is offline.

## Contents

| Name | Kind |
|---|---|
| `fold_plan`, `realised_scale` | functions |
| `FoldResult` | dataclass |
| `fold_isa`, `fold_transpiled` | functions |
| `rebase_to_backend` | function |
| `ZNESamplingConfig` | dataclass |
| `resolve_asymptote` | function |
| `richardson_weights`, `zne_fit_single_value` | functions |
| `fit_scalar_series`, `extrapolate_scalar`, `is_resolved` | functions |
| `project_to_simplex`, `clip_renormalise`, `simplex_renormalise` | functions |
| `pointwise_raw`, `extrapolate_distribution` | functions |
| `zne_bootstrap` | function |
| `plan_scales` | function |
| `pathology_diagnostic` | function |
| `transpile_once`, `transpile_once_seeded`, `run_zne_scales` | functions |
| `ZNEItem`, `ZNEItemRun`, `ZNEBatchRun` | dataclasses |
| `run_zne_batch`, `analyse_zne_item` | functions |
| `zne_mitigated_distribution`, `dgssp_zne_mitigated_distribution` | functions |
| `depolarised_probability`, `synthetic_depolarising_counts`, `expected_extrapolation` | functions |
| `fold_local_circuit` | deprecated function |

---

### Partial folding

#### `fold_plan(scale_factor, n_gates) -> (k, m)` and `realised_scale(scale_factor, n_gates) -> Fraction`

Every gate is folded `k = floor((λ−1)/2)` times (`G (G†G)^k`, scale `2k+1`),
then `m = round(n·(λ−2k−1)/2)` gates once more. Realised scale
`2k+1 + 2m/n`, within `1/n` of `λ`.

#### `fold_isa(tqc, scale_factor, *, seed=0, virtual_gates=("rz",)) -> FoldResult`

The scale is measured on the **two-qubit** count (ε₂ dominates survival). A
seeded subset of `m₂` two-qubit gates gets the extra fold; an independently
seeded subset with the same *fraction* of physical single-qubit gates gets one
too, so single-qubit noise scales alongside (`realised_scale_1q`). Virtual
gates are never in the partial layer. Subsets are prefixes of one permutation,
so for a fixed seed they are nested across scales. Odd integers reproduce the
integer-only implementation gate for gate (tested). Deterministic in
`(circuit, scale, seed)`. Measurements, barriers, delays and resets are never
folded; the layout is preserved.

`FoldResult` fields: `circuit`, `nominal_scale`, `realised_scale`
(`Fraction`), `realised_scale_1q`, `uniform_folds`, `n_two_q`,
`n_two_q_extra`, `n_one_q_physical`, `n_one_q_extra`, `seed`.

> **Correction to the WP4 brief.** The brief said the partial layer folds
> two-qubit gates "with their 1Q neighbours as `fold_transpiled` already
> groups them". `fold_transpiled` never grouped gates — it folds every
> instruction independently. The partial layer therefore selects physical
> single-qubit gates independently, at the same fraction.

#### `fold_transpiled(tqc, scale_factor, *, seed=0, virtual_gates=("rz",)) -> QuantumCircuit`

`fold_isa(...).circuit`. Raises `ValueError` for scales below 1 or non-finite.

#### `rebase_to_backend(circuit, backend) -> QuantumCircuit`

Translates inverse gates (`sx → sxdg`) back to the native basis with
`BasisTranslator` only — no layout, no routing — and re-attaches the input's
layout. After it, a folded circuit is ISA-valid for the backend
(`target.instruction_supported` on every instruction, tested).

---

### `ZNESamplingConfig` — dataclass

- `scales: Sequence[float] = [1, 3, 5]` — nominal scales ≥ 1.
- `shots_per_scale: int = 10_000`.
- `method: "linear" | "quadratic" | "richardson" | "exponential" = "linear"`.
- `clip`, `renormalize: bool = True` — legacy flags of `renorm="clip"`.
- `seed_min = 0`, `seed_max = 64`, `optimization_level = 3`,
  `layout_method = "sabre"`, `seed_simulator = None`.
- `mode: "scalar" | "pointwise" = "scalar"` — which estimate a batch reports
  as the mitigated value (both are always computed).
- `renorm: "clip" | "simplex" = "clip"` — pointwise only.
- `asymptote: float | "p_uniform" | None = None` — exponential only. A float
  fixes `a`; `"p_uniform"` fixes it per instance to that instance's `M/2ⁿ`
  (what a multi-instance batch needs); `None` fits it freely (≥ 3 scales). In
  pointwise mode any non-`None` value fixes each bitstring's asymptote at
  `2⁻ⁿ`.
- `fold_seed: int = 0` — partial-folding and Mitiq-folding seed.
- `n_bootstrap: int = 500`, `bootstrap_seed: int = 0`.

Unknown `method`/`mode`/`renorm`/`asymptote` values raise at construction.

---

### Scalar estimator

#### `fit_scalar_series(scales, values, shots, cfg, *, p_uniform=None) -> dict`

Weighted fit (`w = 1/σ²`, `σ = metrics.proportion_sigma`):

- `linear`/`quadratic` — closed-form WLS, `cov = (XᵀWX)⁻¹`.
- `richardson` — Lagrange interpolation of degree `len(scales)−1`, evaluated at
  0 (`richardson_weights`); exact, `dof = 0`, `chi2_red = None`,
  `value_se = sqrt(Σ Lᵢ² σᵢ²)`.
- `exponential` with a fixed asymptote — variable projection: for fixed rate
  `c` the amplitude is linear, so `χ²(c)` is minimised by a grid plus
  golden-section search over `c ∈ [0, EXP_RATE_MAX = 10]`; no initial guess.
  Covariance from the Jacobian. A fit at the bound is flagged
  `rate_at_bound`.
- `exponential` with a free asymptote — `curve_fit`, bounded rate.

Returns `value`, `value_se`, `params`, `param_names`, `cov`, `fitted`,
`residuals`, `n_params`, `dof`, `chi2`, `chi2_red`, `fit_ok`,
`rate_at_bound`, `scales`, `observed`, `sigma`, `shots`, `asymptote`,
`null_value`, `resolved`, `resolvability_k`, `value_at_min_scale`,
`reported_value`, `reported_source`.

#### Resolvability: `is_resolved(values, shots, null_value, *, k=2)`

`resolved = True` only if **every** scale has `|obs(λ) − P_uniform| ≥ 2·SE`.
SE is the closed-form binomial SE, which is the exact multinomial-bootstrap SE
of a proportion, so the flag is deterministic. An unresolved fit still returns
its numbers, but `reported_value` is the lowest-scale observation
(`reported_source = "min_scale"`): the paper reports `λ = 1` for it.

#### `extrapolate_scalar(counts_per_scale, observable_fn, cfg, *, p_uniform=None) -> dict`

Counts-level entry point (keys = realised scales). Raises on a zero-shot scale.

---

### Pointwise estimator

#### `project_to_simplex(v) -> ndarray`

Euclidean projection onto the probability simplex, sorting-based
O(K log K) — Duchi, Shalev-Shwartz, Singer & Chandra, *Efficient projections
onto the ℓ1-ball for learning in high dimensions*, ICML 2008. Row-wise on 2-D
input. Never farther from `v` in L2 than clip-and-renormalise (provable,
tested); **not** guaranteed closer in D_TV to any reference (recorded in
`paper/data/zne_synthetic.json`, not asserted).

#### `clip_renormalise(raw)`, `simplex_renormalise(raw)`, `pointwise_raw(counts_per_scale, cfg)`

The unconstrained per-bitstring extrapolation (bitstrings zero at every scale
stay zero) and its two renormalisations.

#### `extrapolate_distribution(counts_per_scale, zne_cfg) -> dict`

`pointwise_raw` followed by `cfg.renorm`.

---

### `zne_bootstrap(counts_per_scale, cfg, *, observable_fn=None, reference=None, solution_bitstrings=None, p_uniform=None, B=None, seed=None, alpha=0.05) -> dict`

Resamples the counts at every scale multinomially, re-runs the whole estimator
(weights included), and takes percentile intervals.

- scalar mode: `{"point", "lo", "hi", "se", "n_failed", "B", "seed", "alpha",
  "obs_se", "fit"}`.
- pointwise mode: `{"distribution", "intervals" (per bitstring), "tv" (to
  `reference`), "solution_probability", "renorm", "B", "seed", "alpha"}`.

---

### `plan_scales(survival, signal, shots, candidates, k=3, *, p_uniform=None, n_points=None) -> list[float]`

Pre-registers the scale grid: the prefix of `candidates` (from 1) whose
*predicted* signal `S^λ·(P_ideal − P_uniform)` stays ≥ `k·σ`, with σ evaluated
at the predicted `P(λ)`. `[1.0]` alone means nothing is resolvable. `survival`
is an upper bound, so the plan is optimistic; `resolved` checks it after the
fact. Pathology instances (no signal) need their grid planned separately.

### `pathology_diagnostic(counts_per_scale, cfg, reference) -> dict`

Run on **every** instance. Returns `fraction_positive_slope` (least-squares
linear slope per bitstring, model-free), `raw_total_mass`,
`clip_mass_removed`, `clip_mass_moved`, `simplex_mass_moved` (L1),
`tv_raw_min_scale`, `tv_pointwise_clip`, `tv_pointwise_simplex`,
`clip_worse_than_raw`, `simplex_worse_than_raw`, `reference_is_uniform`,
`raw_distribution`. The reference is `execution.exact_index_distribution`:
exactly uniform for M = 0 and P-n2-M2-03, the noiseless distribution otherwise.

---

### Pipeline

- `transpile_once_seeded(circuit, backend, cfg) -> (tqc, seed)`;
  `transpile_once(...)` returns only the circuit.
- `run_zne_scales(tqc, backend, cfg) -> dict[float, counts]` — one instance,
  one job, keys = realised scales.
- `zne_mitigated_distribution`, `dgssp_zne_mitigated_distribution` — the
  pointwise pipeline end to end.

### One job for many instances

#### `ZNEItem(name, circuit, scales=None, solution_bitstrings=())`

#### `run_zne_batch(items, backend, cfg, *, include_mitiq=False, mode="auto") -> ZNEBatchRun`

Folds every item at every scale (`fold_isa`, seed `cfg.fold_seed`), rebases,
optionally adds Mitiq's `fold_gates_at_random` folds of the same ISA circuit
at the item's **realised** scales (Mitiq's scale-1 circuit is the unfolded one,
so the item's own λ = 1 counts are reused), and submits the list in **one**
`sample_counts` call. `ZNEItemRun` holds `nominal_scales`, `realised_scales`
(+ `_exact`, `_1q`), `counts_per_scale`, `mitiq_scales`,
`mitiq_realised_scales` (Mitiq measures its scale on a fidelity-weighted count
of all gates, so its two-qubit ratio differs; recorded), `mitiq_counts_per_scale`,
`pub_indices`. `ZNEBatchRun` adds `job_id`, `backend_name`, `n_circuits`,
`shots`. Raises if two nominal scales realise to the same scale.

#### `analyse_zne_item(run, cfg, *, solution_bitstrings, p_uniform, reference=None) -> dict`

Every estimator offline: `{"scalar", "pointwise", "pathology", "mitiq"}`.
A hardware job fetched later with `runtime.fetch_result` is analysed by exactly
this code.

### Synthetic depolarising model

`depolarised_probability(S, λ, P_ideal, P_uniform)`,
`synthetic_depolarising_counts(S, scales, P_ideal, P_uniform, shots, rng)`,
`expected_extrapolation(cfg, scales, S, P_ideal, P_uniform, shots)` (the
estimator on the exact curve: its analytic bias). Used by
`tests/test_zne_synthetic.py` and `paper/scripts/zne_synthetic.py`.

### `fold_local_circuit(circuit, scale_factor)` — deprecated

Alias for `fold_transpiled`; emits a `DeprecationWarning`.
