# `dgssp.backends`

## What the module does

Backend discovery, noise-model construction and calibration-based scoring:

1. build an ideal (noiseless) `AerSimulator`;
2. list the real IBM devices an account can reach, filtered for usability;
3. derive a noisy `AerSimulator` from each real device;
4. score a *transpiled* circuit against a backend's calibration data, and rank backends by accumulated error or by measured solution probability.

Two deliberate changes from the previous version. Error metrics read
`backend.target` — the BackendV2 calibration store — rather than the removed V1
`properties()` API, and resolve physical qubits with
`QuantumCircuit.find_bit(...).index` instead of the private `Qubit._index`
attribute. And missing calibration entries are **counted**, not silently
treated as zero: silent zeros used to bias ranking toward devices with
incomplete data, making the "best backend" choice meaningless.

Credential handling lives in `dgssp.runtime`; this module only consumes a
service handed to it.

## Contents

| Name | Kind |
|---|---|
| `BackendLike` | type alias |
| `BackendSelectionConfig` | dataclass |
| `list_real_backends` | function |
| `build_ideal_aer_backend` | function |
| `build_fake_backends_from_real` | function |
| `build_all_backends` | function |
| `BackendErrorMetrics` | dataclass |
| `compute_accumulated_errors` | function |
| `BackendPerformance` | dataclass |
| `run_circuit_on_backend` | function |
| `evaluate_backends_for_circuit` | function |
| `select_best_backends` | function |

---

### `BackendLike` — type alias

Alias for `qiskit.providers.BackendV2`, used throughout the library to mean "a
modern Qiskit backend".

---

### `BackendSelectionConfig` — dataclass

**Fields**

- `min_qubits: int | None = None` — minimum device width for a real backend to be considered; `None` disables the filter.
- `seed_simulator: int | None = None` — seed applied to every simulator built here.
- `real: bool = False` — whether to include real hardware in the bundle.

Note what is *absent*: credential paths and `config_path` are gone, having
moved to `dgssp.runtime`.

---

### `list_real_backends(service, *, min_qubits=None) -> list[BackendV2]`

- **Inputs:** `service` (a `QiskitRuntimeService`); `min_qubits: int | None`.
- **Output:** non-simulator devices that are wide enough, operational and not under maintenance.

---

### `build_ideal_aer_backend(*, seed_simulator=None) -> AerSimulator`

- **Input:** `seed_simulator: int | None`.
- **Output:** a noiseless local simulator.

---

### `build_fake_backends_from_real(real_backends, *, seed_simulator=None) -> list[AerSimulator]`

- **Inputs:** `real_backends: Sequence[BackendV2]`; `seed_simulator: int | None`.
- **Output:** one noisy simulator per real device, cloning its noise model, in the same order.

---

### `build_all_backends(config, *, service=None) -> dict`

- **Inputs:** `config: BackendSelectionConfig`; `service` (an existing `QiskitRuntimeService`, or `None`).
- **Output:** `{"service", "ideal", "real_backends", "fake_backends"}`. `real_backends` is empty unless `config.real` is `True`. When `service is None`, **no remote lookup is attempted** and only the ideal simulator is returned — which keeps the function usable and offline in tests and CI.

---

### `BackendErrorMetrics` — dataclass

**Fields**

- `total_error: float` — sum of the three contributions below.
- `two_qubit_error: float` — summed error over every two-qubit gate application.
- `single_qubit_error: float` — summed error over every single-qubit application.
- `readout_error: float` — summed error over every measurement.
- `single_qubit_gate_count: int`, `two_qubit_gate_count: int` — application counts.
- `missing_calibration: int` — instructions for which the target held no error figure. A large value means the total is an *underestimate* and cross-backend comparison is unreliable.

**Methods**

- `to_dict() -> dict` — **Output:** a plain JSON-serialisable dictionary with one key per field.

---

### `compute_accumulated_errors(backend, qc) -> BackendErrorMetrics`

Accumulates calibration error over a transpiled circuit. The circuit **must**
already be transpiled for the backend: qubit positions are resolved with
`find_bit(...).index` and looked up in `backend.target`, so a logical circuit
would produce meaningless indices.

- **Inputs:** `backend: BackendV2` (must expose a calibrated `target`); `qc: QuantumCircuit` (ISA level).
- **Output:** a `BackendErrorMetrics`. Barriers and delays are skipped.
- **Raises:** `AttributeError` if the backend exposes no `target` — an explicit failure rather than a silently-zero score.

---

### `BackendPerformance` — dataclass

How one backend performed on one circuit.

**Fields:** `backend`, `backend_name`, `is_real`, `transpiled_circuit`,
`errors` (a `BackendErrorMetrics`), `counts`, `solution_probability`.

---

### `run_circuit_on_backend(backend, qc, *, shots=10000, seed_transpiler=None, optimization_level=0, seed_simulator=None) -> tuple[QuantumCircuit, dict[str, int]]`

- **Inputs:** the backend; the logical circuit; shot count; transpiler seed; optimization level; simulator seed.
- **Output:** `(transpiled_circuit, counts)`. Sampling goes through `runtime.sample_counts`, so the sampler incantation is not duplicated here.

---

### `evaluate_backends_for_circuit(instance, qc, *, real_backends=(), fake_backends=(), classical_solver=None, shots=10000, noise_seed=42) -> dict[str, list[BackendPerformance]]`

Scores a set of backends against classical ground truth. For each backend the
circuit is transpiled, scored against calibration data, executed, and the
measured probability of a DP-verified solution is recorded.

- **Inputs:** `instance` (supplies the ground truth); `qc` (the logical circuit); the two backend lists; `classical_solver` (defaults to an exhaustive DP solver); `shots`; `noise_seed` (applied identically to every backend so the comparison is fair).
- **Output:** `{"real": [...], "fake": [...]}` of `BackendPerformance`.

---

### `select_best_backends(real_performances, fake_performances) -> dict[str, BackendPerformance | None]`

- **Inputs:** the two result lists.
- **Output:** keys `best_real_by_error`, `best_fake_by_error`, `best_real_by_probability`, `best_fake_by_probability`. A value is `None` when the corresponding list is empty.
