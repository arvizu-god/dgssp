# `dgssp.transpilation`

## What the module does

Layout selection for noisy backends. The D-G circuit is dominated by two-qubit
gates — the controlled-phase adder and the multi-controlled gates — so *which
physical qubits the circuit lands on* matters more than almost anything else.
SABRE layout is stochastic in its seed, so this module sweeps seeds and keeps
the layout minimising the **calibration-weighted** two-qubit error rather than
the raw gate count.

The layout found here is meant to be searched **once** and then reused, notably
by `dgssp.mitigation.zne`, which folds the already-transpiled circuit instead
of re-searching a layout at every noise scale.

Errors are read from `backend.target` and qubit positions from
`find_bit(...).index`; the removed V1 `properties()` API and the private
`Qubit._index` attribute are gone. So is the `finding_best_seed` tuple-returning
wrapper.

## Contents

| Name | Kind |
|---|---|
| `BackendLike` | type alias |
| `TwoQubitErrorReport` | dataclass |
| `two_qubit_gate_errors_per_circuit_layout` | function |
| `BestSeedResult` | dataclass |
| `find_best_seed` | function |

---

### `BackendLike` — type alias

Alias for `qiskit.providers.BackendV2`.

---

### `TwoQubitErrorReport` — dataclass

Per-pair two-qubit error breakdown for one transpiled circuit.

**Fields**

- `accumulated_error: float` — total over every two-qubit application.
- `gate_count: int` — number of applications.
- `pairs: list[tuple[int, int]]` — the distinct physical qubit pairs used, in first-seen order.
- `error_per_pair: list[float]` — single-application error for each pair, parallel to `pairs`.
- `accumulated_per_pair: list[float]` — error summed over all uses of each pair, parallel to `pairs`.
- `missing_calibration: int` — applications with no calibration entry.

---

### `two_qubit_gate_errors_per_circuit_layout(circuit, backend) -> TwoQubitErrorReport`

- **Inputs:** `circuit: QuantumCircuit` (already transpiled for the backend); `backend: BackendV2`.
- **Output:** a `TwoQubitErrorReport`. If the backend has no target, an all-zero report is returned so seed sweeps degrade to "any layout" rather than crashing.

---

### `BestSeedResult` — dataclass

**Fields**

- `circuit: QuantumCircuit` — the transpiled circuit from the winning seed.
- `best_seed: int` — that seed.
- `total_two_qubit_error: float` — its accumulated two-qubit error.
- `two_qubit_gate_count: int` — its two-qubit gate count.
- `optimization_level: int` — the level used during the sweep, recorded so downstream code knows not to re-optimize.

---

### `find_best_seed(circuit, backend, *, seed_min=0, seed_max=128, optimization_level=3, layout_method="sabre") -> BestSeedResult`

- **Inputs:** the logical circuit; the backend; the half-open seed range `range(seed_min, seed_max)` (a range of one effectively disables the sweep); the optimization level; the layout method.
- **Output:** a `BestSeedResult`. Ties on accumulated error are broken by fewer two-qubit gates.
- **Raises:** `ValueError` if the seed range is empty — previously this silently returned nothing useful.
