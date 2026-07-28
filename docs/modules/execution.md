# `dgssp.execution`

## What the module does

Three single-circuit execution strategies, from cheapest to most involved:

- **`execute_ideal`** — noiseless `AerSimulator`, no mitigation. The reference distribution.
- **`execute_noisy`** — one noisy backend, plain transpilation, no mitigation. What the algorithm actually does on hardware today.
- **`execute_optimized_mitigated`** — pick the lowest-error backend, search a good SABRE layout once, and run unmitigated *and* ZNE-mitigated on that same physical circuit.

All three go through `runtime.sample_counts`, so the sampler incantation, the
simulator-seeding rules and the job logging live in exactly one place. For
sweeps over *many* instances use `dgssp.experiments.run_batch` instead — it
batches every circuit into a single job.

## Contents

| Name | Kind |
|---|---|
| `ExecutionResult` | dataclass |
| `execute_ideal` | function |
| `execute_noisy` | function |
| `select_best_backend_by_error` | function |
| `execute_optimized_mitigated` | function |

---

### `ExecutionResult` — dataclass

**Fields**

- `backend` / `backend_name` — the backend used and its name.
- `counts: dict[str, int]` — raw measurement counts.
- `transpiled_circuit: QuantumCircuit | None` — the ISA-level circuit actually executed.
- `best_seed: int | None` — the transpiler seed, when a layout search was performed.
- `error_metrics: BackendErrorMetrics | None` — calibration error accumulation, when the backend exposes a target.
- `mitigated_distribution: dict[str, float] | None` — the ZNE result, when mitigation was requested.

---

### `execute_ideal(qc, *, shots=10000, seed_simulator=None, optimization_level=1) -> ExecutionResult`

- **Inputs:** the logical circuit; shot count; simulator seed; optimization level.
- **Output:** an `ExecutionResult` with `error_metrics` and `mitigated_distribution` left as `None` — there is no noise to characterise or mitigate.

---

### `execute_noisy(qc, backend, *, shots=10000, seed_transpiler=None, seed_simulator=None, optimization_level=1, mode="auto") -> ExecutionResult`

- **Inputs:** the logical circuit; a noisy simulator or real device; shot count; transpiler and simulator seeds; optimization level; `mode`, forwarded to `runtime.execution_mode` so hardware runs inside a `Batch`.
- **Output:** an `ExecutionResult` including calibration metrics when the backend has a target (and `None` for those metrics when it does not, rather than raising).

---

### `select_best_backend_by_error(qc, backends, *, optimization_level=1, layout_method="sabre", seed_transpiler=0) -> tuple[BackendV2 | None, BackendErrorMetrics | None]`

Picks the backend with the lowest accumulated calibration error. A single cheap
transpilation per backend is used for the comparison; the expensive seed sweep
happens only on the winner.

- **Inputs:** the logical circuit; the candidate backends; optimization level; layout method; a transpiler seed applied identically to every candidate so the comparison is fair.
- **Output:** `(backend, metrics)`, or `(None, None)` if the list was empty or no candidate exposed a target.

---

### `execute_optimized_mitigated(qc, backends_bundle, zne_cfg, *, shots_unmitigated=10000) -> dict[str, ExecutionResult | None]`

For each device class present in the bundle: pick the lowest-error backend,
search a SABRE layout **once**, sample that ISA circuit unmitigated, then reuse
the *same* ISA circuit for the ZNE folds — so the mitigated and unmitigated
numbers describe the same physical circuit and are actually comparable.

- **Inputs:** the logical circuit; `backends_bundle` (output of `backends.build_all_backends`); a `ZNESamplingConfig`; `shots_unmitigated`.
- **Output:** `{"best_fake": ..., "best_real": ...}`; a value is `None` when no backend of that class was available.
