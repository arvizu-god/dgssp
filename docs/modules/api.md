# `dgssp.api`

## What the module does

The single-instance convenience layer. `run_dgssp` takes an instance and an
executor name and does everything in between: build the circuit, resolve a
backend, execute, and — on the `"optimized"` path — mitigate. It is the "just
run it" entry point; reach for `dgssp.experiments.run_batch` as soon as you
have more than one instance, since that batches them into a single job.

Unlike the previous version, the configuration object passed in is **never
mutated**: defaults are filled into local variables, so re-using one
`DGRunConfig` across calls cannot silently change its meaning.

## Contents

| Name | Kind |
|---|---|
| `DGRunConfig` | dataclass |
| `build_dg_circuit` | function |
| `_resolve_noisy_backend` | internal function |
| `run_dgssp` | function |
| `run_dgssp_ideal` | function |
| `run_dgssp_noisy` | function |
| `run_dgssp_optimized` | function |

---

### `DGRunConfig` — dataclass

**Fields**

- `dg_config: DGConfig | None` — solver configuration; `None` means defaults.
- `backend_config: BackendSelectionConfig | None` — used when no explicit backend is given.
- `backend_kind: "real" | "fake"` — which pool to draw from in `"noisy"` mode.
- `backend: BackendV2 | None` — an explicit backend, overriding `backend_kind` and `backend_config`.
- `service: Any | None` — an existing `QiskitRuntimeService`, required to reach hardware.
- `zne_config: ZNESamplingConfig | None` — settings for `"optimized"` mode; a sensible default is built if omitted.
- `shots: int`, `shots_unmitigated: int` — shot counts for the plain modes and for the unmitigated run inside `"optimized"`.
- `auto_num_solutions: bool = True` — when `iterations == "auto"`, run the DP solver first to obtain the true solution count `M`. Set `False` to assume `M = 1` and skip the (worst-case exponential) classical pass.
- `seed_simulator`, `seed_transpiler: int | None`.

---

### `build_dg_circuit(instance, dg_config=None, *, auto_num_solutions=True) -> tuple[QuantumCircuit, DGConfig]`

Builds the circuit, resolving `iterations="auto"` along the way.

- **Inputs:** the instance; a `DGConfig` or `None`; whether to run the DP solver to count solutions.
- **Output:** `(circuit, config_used)`.

---

### `run_dgssp(instance, *, executor="ideal", config=None) -> ExecutionResult | dict[str, ExecutionResult | None]`

- **Inputs:** the instance; `executor` — `"ideal"`, `"noisy"` or `"optimized"`; an optional `DGRunConfig`, which is never mutated.
- **Output:** a single `ExecutionResult` for `"ideal"` and `"noisy"`; for `"optimized"`, `{"best_fake": ..., "best_real": ...}`.
- **Raises:** `ValueError` for an unknown executor name or missing required configuration.

---

### `run_dgssp_ideal(instance, *, config=None) -> ExecutionResult`

Shortcut for `run_dgssp(..., executor="ideal")`. **Input:** the instance and an optional config. **Output:** the noiseless run.

---

### `run_dgssp_noisy(instance, *, config=None) -> ExecutionResult`

Shortcut for the `"noisy"` executor. The config must supply a backend or a backend config. **Output:** the noisy run.

---

### `run_dgssp_optimized(instance, *, config=None) -> dict[str, ExecutionResult | None]`

Shortcut for the `"optimized"` executor. The config must supply a backend or a backend config. **Output:** `{"best_fake": ..., "best_real": ...}`.
