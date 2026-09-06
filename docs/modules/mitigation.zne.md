# `dgssp.mitigation.zne`

## What the module does

Zero-noise extrapolation for a *sampling* algorithm.

Standard ZNE mitigates an expectation value. The D-G algorithm produces a
distribution over bitstrings, so this module extrapolates **each bitstring's
probability independently** to zero noise, then clips and renormalizes the
result into a valid distribution. Every fitting model is available and the
choice is explicit, which makes the extrapolation auditable.

Two ordering bugs from the previous implementation are fixed here, and the fix
is the whole point of the module:

- **Transpile first, then fold.** Folding a *logical* circuit and then transpiling at optimization level 3 lets the gate-cancellation passes undo the folds — the noise never actually gets scaled. Here the circuit is transpiled once to ISA level, the *physical* gates are folded, and the folded circuits are submitted at optimization level 0 so nothing is re-optimized or re-routed.
- **Search the layout once.** Re-running a seed sweep per scale changes the physical circuit between scales, which silently invalidates the extrapolation: the fitted points no longer lie on one noise curve.

All scales are submitted as a single batched job.

## Contents

| Name | Kind |
|---|---|
| `fold_transpiled` | function |
| `rebase_to_backend` | function |
| `zne_fit_single_value` | function |
| `ZNESamplingConfig` | dataclass |
| `transpile_once` | function |
| `run_zne_scales` | function |
| `extrapolate_distribution` | function |
| `zne_mitigated_distribution` | function |
| `dgssp_zne_mitigated_distribution` | function |
| `fold_local_circuit` | deprecated function |

---

### `fold_transpiled(tqc, scale_factor) -> QuantumCircuit`

Folds an already-transpiled circuit's gates as `G (G† G)^n`. Because the input
is ISA-level, the folds land on the physical gates that actually carry the
noise, and the fixed layout is preserved.

- **Inputs:** `tqc: QuantumCircuit` (transpiled); `scale_factor: int`, an odd positive integer — `1` returns the circuit unchanged, `3` triples each gate, and so on.
- **Output:** the folded circuit. Measurements, barriers, delays and resets are copied through once and never folded.
- **Raises:** `ValueError` if the scale factor is even or non-positive.

---

### `rebase_to_backend(circuit, backend) -> QuantumCircuit`

Rewrites a folded circuit back into the backend's native gate set.

Folding inserts `op.inverse()` gates, and the inverse of a native gate is not
necessarily native itself: on IBM devices `sx` inverts to `sxdg`, which the
device — and the Aer noise model derived from it — cannot execute. Without this
step a folded circuit fails at submission with an *"unknown instruction"* error.

Only gate **definitions** are translated. Layout and routing are untouched,
which is the entire point: re-routing between noise scales is exactly the bug
the transpile-once design exists to avoid.

- **Inputs:** `circuit: QuantumCircuit` (a folded ISA-level circuit); `backend: BackendV2` (whose `target` defines the native basis).
- **Output:** the circuit expressed in that basis. If the backend has no target, or translation fails, the input is returned unchanged so permissive simulators keep working.

Applied automatically inside `run_zne_scales` and inside the Mitiq executor.

---

### `zne_fit_single_value(method, xdata, ydata) -> tuple[float, ndarray, ndarray, Callable]`

Fits one scalar against noise scale and extrapolates it to zero.

- **Inputs:** `method: str` — `"linear"`, `"quadratic"` or `"exponential"`; `xdata: Sequence[float]` — the noise scales; `ydata: Sequence[float]` — the observed value at each scale (here, one bitstring's probability).
- **Output:** `(zero_value, ydata_array, popt, model_fn)` — the extrapolated value at zero noise plus the inputs, fitted parameters and model function, the latter three for diagnostics and plots.
- **Raises:** `ValueError` for an unknown model, or when there are fewer data points than the model has parameters.
- **Fallback:** if `curve_fit` fails to converge, the least-noisy observation is returned rather than aborting the run, so one pathological bitstring cannot kill a whole batch.

**How each model is solved.** `linear` and `quadratic` are linear in their
parameters, so they are solved in closed form by least squares
(`np.linalg.lstsq` on a Vandermonde design). That is exact, deterministic, free
of any initial-guess sensitivity, and about ten times faster — which matters
because this runs once *per bitstring*, on probabilities spanning several orders
of magnitude. Only `exponential` is genuinely nonlinear and still uses
`scipy.optimize.curve_fit`. `popt` keeps the same ordering in both paths, so
callers see no difference.

**Degrees of freedom.** With as many noise scales as the model has parameters
(two scales and `linear`, three and `quadratic`/`exponential`) the fit is
*exactly determined*. The extrapolation is still a valid Richardson estimator —
two scales and `linear` gives exactly `(3*y1 - y3)/2` — but there is no residual
to test the model against, so the data say nothing about whether the model is
right. Use **at least three scales** for any claim about the extrapolation
itself; two is a smoke-test setting only.

In that exactly determined case `curve_fit` cannot estimate a parameter
covariance and emits `OptimizeWarning`. Since the covariance is discarded
anyway, the warning is suppressed for that case *only* — a genuine fit failure
at three or more scales still surfaces.

---

### `ZNESamplingConfig` — dataclass

**Fields**

- `scales: Sequence[int] = [1, 3, 5]` — odd positive noise-scale factors.
- `shots_per_scale: int = 10_000` — shots collected at each scale.
- `method: str = "linear"` — extrapolation model.
- `clip: bool = True` — clip negative extrapolated probabilities to zero.
- `renormalize: bool = True` — rescale the distribution to sum to one.
- `seed_min: int = 0`, `seed_max: int = 64` — seed range for the *single* layout search; set `seed_max = seed_min + 1` to skip it.
- `optimization_level: int = 3` — level for that one transpilation. Folded circuits are always submitted at level 0.
- `layout_method: str = "sabre"`.
- `seed_simulator: int | None = None` — applied only on simulators.

---

### `transpile_once(circuit, backend, zne_cfg) -> QuantumCircuit`

Produces the single ISA-level circuit that every scale will fold.

- **Inputs:** the logical circuit; the backend; the config.
- **Output:** the transpiled circuit — from `find_best_seed` when the seed range has length > 1, otherwise a single-seed transpilation.

---

### `run_zne_scales(tqc, backend, zne_cfg) -> dict[int, dict[str, int]]`

Folds at every scale and samples all of them in **one** job.

- **Inputs:** `tqc` (ISA level); the backend; the config.
- **Output:** counts keyed by noise scale.
- **Raises:** `ValueError` if `scales` is empty or contains a non-odd / non-positive value.

---

### `extrapolate_distribution(counts_per_scale, zne_cfg) -> dict[str, float]`

- **Inputs:** counts keyed by scale; the config.
- **Output:** the mitigated distribution over every bitstring seen at any scale. Bitstrings that were zero at all scales stay zero without a fit. Clipping and renormalization are applied per the config.
- **Raises:** `ValueError` if any scale collected zero shots.

---

### `zne_mitigated_distribution(circuit, backend, zne_cfg, *, transpiled=None) -> dict[str, float]`

The full pipeline: transpile once, fold, batch-sample, extrapolate.

- **Inputs:** the logical circuit; a noisy backend; the config; `transpiled: QuantumCircuit | None` — an ISA-level circuit to reuse instead of transpiling again. Pass this when the caller has already searched a layout, so the unmitigated and mitigated numbers describe exactly the same physical circuit.
- **Output:** the zero-noise-extrapolated distribution.

---

### `dgssp_zne_mitigated_distribution(instance, dg_config, backend, zne_cfg, *, num_solutions=None) -> dict[str, float]`

Convenience wrapper that builds the D-G circuit and then mitigates it.

- **Inputs:** the instance; a `DGConfig`; a noisy backend; the ZNE config; `num_solutions: int | None`, used when `iterations == "auto"`.
- **Output:** the mitigated distribution over index-register bitstrings.

---

### `fold_local_circuit(circuit, scale_factor)` — deprecated

Alias for `fold_transpiled`, kept so existing notebooks keep running. Emits a
`DeprecationWarning`. Note the semantic difference: folding is only meaningful
on an ISA-level circuit.
