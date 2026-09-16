# `dgssp.experiments`

## What the module does

The multi-instance batch runner — the module you use to produce thesis figures.

Given a list of instances it will, in one call:

1. solve each instance classically (exhaustive DP) to obtain **ground truth** — the set of correct bitstrings and their count `M`;
2. use `M` to pick the optimal Grover iteration count per instance;
3. build and transpile one circuit per instance;
4. submit **all of them as a single job** via `runtime.sample_counts`, so a sweep costs one queue wait rather than one per instance;
5. decode each result against its ground truth and return structured, serialisable records.

Because ground truth is computed inside the runner, every result carries a
measured `solution_probability`. Success is quantified automatically across the
whole sweep, which is what makes "P(solution) vs `n_items`", "vs backend" or
"vs ZNE on/off" a one-liner.

The `"optimized"` executor transpiles every instance once, then folds all of
them at all their scales — plus Mitiq's folds when the baseline is requested —
and submits everything through `mitigation.zne.run_zne_batch`: **one job per
backend**, so one job whenever `BatchConfig.backend` is given (the paper's
path). With a candidate pool, each instance picks its lowest-error backend and
there is one job per distinct backend chosen.

`run_iteration_sweep` puts every `r`-variant of one instance (and, for ZNE,
every fold of every variant) into one job.

## Contents

| Name | Kind |
|---|---|
| `BatchConfig` | dataclass |
| `InstanceResult` | dataclass |
| `BatchResult` | dataclass |
| `_InstanceMeta` | internal dataclass |
| `prepare_instances` | function |
| `_resolve_backend` | internal function |
| `run_batch` | function |
| `_run_batched` | internal function |
| `_run_optimized` | internal function |
| `run_random_batch` | function |
| `run_iteration_sweep` | function |

---

### `BatchConfig` — dataclass

**Fields**

- `dg_config: DGConfig` — solver configuration. With the default `iterations="auto"`, each instance gets its own optimal count from the DP ground truth.
- `executor: "ideal" | "noisy" | "optimized"` — noiseless Aer, one shared noisy backend, or per-instance best backend + ZNE.
- `shots: int = 10_000` — shots per instance.
- `backend: Any | None` — an explicit backend for the `"noisy"` executor.
- `backend_config: BackendSelectionConfig | None` — used to build a bundle when no backend is given.
- `service: Any | None` — an existing `QiskitRuntimeService`, required to reach real hardware.
- `zne_config: ZNESamplingConfig | None` — settings for `"optimized"`; defaults are supplied if omitted. Scale 1 is always measured (it is the unmitigated result) and is prepended when missing.
- `run_mitiq_baseline: bool = False` — also fold with Mitiq inside the same job and extrapolate offline with its factories. Requires the `dgssp[mitiq]` extra.
- `seed_simulator`, `seed_transpiler: int | None` — reproducibility seeds.
- `optimization_level: int = 1` — for the ideal/noisy paths.
- `mode: str = "auto"` — forwarded to `runtime.execution_mode`.
- `bootstrap_samples: int = 1000`, `bootstrap_seed: int = 0` — intervals on the unmitigated metrics.
- `compute_reference: bool = True` — compute each instance's exact noiseless distribution (`execution.exact_index_distribution`) so results carry distances to it.

---

### `InstanceResult` — dataclass

Everything measured for one instance.

**Fields**

- `instance`, `n_qubits`, `iterations` — what was run.
- `solution_bitstrings: list[str]`, `num_solutions: int` — the classical ground truth.
- `counts: dict[str, int]`, `distribution: dict[str, float]` — the raw and normalised outcome.
- `solution_probability: float` — the headline success metric.
- `mitigated_distribution: dict[str, float] | None`, `mitigated_solution_probability: float | None` — the library's ZNE result.
- `mitiq_solution_probability: float | None` — the Mitiq baseline.
- `error_metrics: BackendErrorMetrics | None`, `backend_name: str | None`.
- `depth: int | None`, `two_qubit_depth: int | None` — circuit depth overall and counting only two-qubit gates (barriers excluded).
- `two_qubit_count: int | None` — two-qubit gate applications in the executed circuit; the resource number the scaling analysis is built on.
- `physical_qubits: list[int] | None` — physical qubits the circuit landed on, in virtual-qubit order.
- `counts_per_scale: dict[float, dict[str, int]] | None` — raw counts keyed by **realised** ZNE noise scale, when mitigation ran. Archived because they cannot be reconstructed from the mitigated distribution, and resampling them is what bootstrap CIs need.
- `elapsed_s: float | None` — wall-clock seconds.
- WP4 additions (all defaulted): `best_seed`, `one_q_physical`, `two_q_physical`, `virtual_gate_count`, `survival` (from `backends.physical_gate_counts` and `metrics.survival_estimate`), `calibration_timestamp`, `job_id`, `p_uniform`, `reference_distribution`, `metrics_ci` (bootstrap intervals and distances to the reference), `zne_fit`, `pointwise`, `pathology`, `mitiq` (the blocks of `zne.analyse_zne_item`), `mitigation_mode`, `renorm`, `fold_seed`.

The `"optimized"` executor always measures scale 1 and uses those counts as the unmitigated result, so the unmitigated and mitigated numbers come from the same job.

**Methods**

- `to_dict() -> dict` — **Output:** a flat JSON-serialisable dictionary; nested dataclasses are expanded and the instance appears as its items, target and name.
- `transpiled_summary() -> dict` — the size metrics of `transpilation.transpiled_metrics` plus `one_q_physical`, `two_q_physical`, `virtual_gate_count`, `survival`, `best_seed`.
- `to_record(*, executor, shots, mitigation_mode=None, job_id=None, extra_metrics=None) -> ResultRecord` — the frozen archive schema (`dgssp.records`), `schema_version = 2`, no provenance (the paper scripts stamp it).

---

### `BatchResult` — dataclass

**Fields**

- `results: list[InstanceResult]` — one per input instance, in input order.
- `config: BatchConfig` — the settings used.
- `job_ids: list[str]` — for recovering or auditing a hardware run.

**Methods**

- `__len__() -> int` / `__iter__()` — the batch behaves like a sequence of results.
- `to_json(path) -> Path` — **Input:** a destination path (parent directories are created). **Output:** the path written, containing the config, job ids and every result.
- `to_dataframe() -> pandas.DataFrame` — **Output:** a tidy one-row-per-instance table with scalar columns only, so it plots directly. Error-metric fields are flattened with an `err_` prefix. **Raises:** `ImportError` naming the `dgssp[data]` extra if pandas is missing.
- `summary() -> dict` — **Output:** instance count, executor, shots, mean/min/max solution probability, and `top_outcome_success_rate` (the fraction of instances whose most likely outcome is correct). Includes the mean mitigated probability when mitigation ran.

---

### `prepare_instances(instances, config) -> list[_InstanceMeta]`

Solves each instance classically and builds its D-G circuit.

- **Inputs:** the instances; a `BatchConfig`.
- **Output:** one internal record per instance carrying its ground-truth bitstrings, iteration count and logical circuit.

---

### `run_batch(instances, config=None) -> BatchResult`

Builds, executes and decodes a whole set of instances.

- **Inputs:** `instances: Sequence[SubsetSumInstance]`; `config: BatchConfig | None` (defaults to the ideal executor).
- **Output:** a `BatchResult`.
- **Raises:** `ValueError` if `instances` is empty or the executor name is unknown.

For `"ideal"` and `"noisy"` every circuit is submitted in a **single** job.

---

### `run_random_batch(n_instances, n_items, *, max_value=20, seed=0, config=None) -> BatchResult`

Generates random feasible instances and runs them as one batch.

- **Inputs:** how many instances; items per instance; `max_value`; a base `seed` — instance *k* uses `seed + k`, so batches are reproducible and non-overlapping; an optional `BatchConfig`.
- **Output:** a `BatchResult`, as from `run_batch`.

---

### `run_iteration_sweep(instance, r_values, backend=None, *, executor="ideal", zne_config=None, shots=10000, best_seeds=None, seed_min=0, seed_max=32, optimization_level=3, scales_per_r=None, include_mitiq=False, seed_simulator=None, bootstrap_samples=1000, bootstrap_seed=0, mode="auto") -> dict`

`P(solution)` against the Grover iteration count, in **one** job.

- Every `r`-variant is built and transpiled once at `optimization_level`: with its recorded seed when `best_seeds` has one (the WP3 `best_seed` of the `r_opt` variant), otherwise by the `[seed_min, seed_max)` SABRE sweep (on a backend without a coupling map only `seed_min` is used).
- `executor`: `"ideal"` (noiseless Aer when `backend` is `None`), `"noisy"`, or `"zne"` (each variant folded at `scales_per_r.get(r, zne_config.scales)`, scale 1 always included, optionally with Mitiq's folds).
- **Output:** `{"instance", "M", "p_uniform", "solution_bitstrings", "executor", "backend", "job_id", "rows"}`; each row has `iterations`, `best_seed`, the transpiled sizes, `survival` (calibrated backends), `p_sol` and `p_sol_ci`, the analytic `sin²((2r+1)θ)` (`solvers.grover_success_probability`), the raw counts and, for ZNE, the scalar fit (`zne`), `pathology` and `mitiq`.
- Tested: on the ideal simulator every `r` matches the analytic curve within 4σ, in one `sample_counts` call.
