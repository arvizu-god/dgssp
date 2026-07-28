# `dgssp.solvers.draper_grover`

## What the module does

Builds the Draper--Grover circuit. This is the core of the library and the one
module whose correctness everything else assumes.

The algorithm searches the space of *subsets* with Grover's algorithm. A Draper
(QFT) phase adder computes the sum of the selected items into an auxiliary
register, and a phase oracle marks the register value corresponding to the
target. One Grover iteration is:

1. **Add** — `QFT → controlled phase additions (+ constant offset) → QFT⁻¹` leaves the sum register holding `encoded_sum(i) = sum(i) + offset`.
2. **Oracle** — a multi-controlled Z conjugated by X gates flips the phase of `|encoded_target⟩`.
3. **Subtract** — the exact inverse of step 1, returning the sum register to `|0…0⟩` so it stays unentangled from the index register.
4. **Diffuser** — standard inversion about the mean on the index register.

Signed instances are handled entirely through `dgssp.encoding`: the offset is
added as an **uncontrolled** constant and removed by the subtractor, so the
uncomputation still lands exactly on zero.

Only the `FullQFT` assembly exists; the historical `HalfQFT` variant has been
removed. The module is side-effect free and returns a plain `QuantumCircuit`.

## Contents

| Name | Kind |
|---|---|
| `qft_no_swaps` | function |
| `optimal_iterations` | function |
| `DGConfig` | dataclass |
| `DGSSPSolver` | class |

---

### `qft_no_swaps(n) -> Gate`

Builds the swap-free Quantum Fourier Transform. The Draper adder assumes the
no-swap convention, in which the Fourier-basis phase carried by qubit *j* is
`2π·x / 2^(j+1)`. Constructing the transform explicitly, rather than relying on
a library class and a keyword whose home has moved between Qiskit releases,
pins that convention permanently.

- **Input:** `n: int`, number of qubits.
- **Output:** an `n`-qubit `Gate` labelled `"QFT"`.

---

### `optimal_iterations(n_ind, num_solutions) -> int`

The optimal Grover iteration count: `r = floor(π / (4θ))` with
`θ = arcsin(sqrt(M/N))`, `N = 2^n_ind`, `M = num_solutions`. Replaces the
hard-coded iteration counts of the previous version.

- **Inputs:** `n_ind: int` (index qubits, so `2^n_ind` candidate subsets); `num_solutions: int` (marked states `M`; values below 1 are clamped to 1).
- **Output:** an `int` ≥ 1. Returns `1` when `M ≥ N`, where the search is already saturated and amplification would over-rotate.

---

### `DGConfig` — dataclass

Solver configuration.

**Fields**

- `iterations: int | Literal["auto"] = "auto"` — an explicit positive count, or `"auto"` to derive it from `optimal_iterations` using the `num_solutions` passed to `build_circuit`.
- `measure: bool = True` — whether to append measurement of the index register. Set `False` to inspect the circuit with a statevector simulator.

Note there is no `assembly_type`: the `HalfQFT` option was deleted, so passing
it raises `TypeError`.

---

### `DGSSPSolver(BaseQuantumSSPSolver)` — class

The circuit builder. Constructed with an optional `DGConfig`.

#### Public methods

**`build_circuit(instance, *, num_solutions=None) -> QuantumCircuit`**

- **Inputs:** `instance: SubsetSumInstance`; `num_solutions: int | None`, used only when `iterations == "auto"` and normally supplied by the classical DP solver upstream. If omitted while `"auto"`, `M = 1` is assumed and a `UserWarning` is emitted.
- **Output:** a circuit named `"DGSSPSolve"` with index register `i`, sum register `s`, and (unless `measure=False`) classical register `c`. Its `metadata["dgssp_instance"]` records the items, target, register widths, offset, modulus, encoded target, iteration count and assembly type.
- **Raises:** `ValueError` if `iterations` is an integer below 1, or is neither an integer nor `"auto"`.
- **Warns:** `UserWarning` if the target lies outside the reachable sum range, since the oracle would then mark nothing.

**`build_step_circuit(instance) -> QuantumCircuit`**

- **Input:** `instance: SubsetSumInstance`.
- **Output:** a single add → oracle → subtract step as an `(n_ind + n_sum)`-qubit circuit with no measurements. Used by the uncomputation test and for diagrams.

**`encoding_for(instance) -> SumRegisterEncoding`**

- **Input:** `instance: SubsetSumInstance`.
- **Output:** the `SumRegisterEncoding` the solver will use, for callers that need to interpret the sum register directly.

#### Internal methods

**`_resolve_iterations(n_ind, num_solutions) -> int`** — turns `config.iterations` into a concrete positive integer, applying the `"auto"` rule and validating explicit values.

**`_sum_gate(enc) -> Gate`** — the Fourier-basis adder. **Input:** a `SumRegisterEncoding`. **Output:** an `(n_ind + n_sum)`-qubit gate labelled `"SumGate"`, applying `cp(2π·a mod M / 2^(j+1))` for each item and an *uncontrolled* `p(2π·offset / 2^(j+1))` for the constant. Applied between a QFT and an inverse QFT.

**`_sub_gate(enc) -> Gate`** — the exact inverse, using `(-a) mod M` per item and `(-offset) mod M` for the constant, so add ∘ subtract is the identity on the sum register. **Output:** a gate labelled `"SubGate"`.

**`_oracle_gate(enc) -> Gate`** — the phase oracle. **Input:** the encoding, which supplies the guaranteed-non-negative `encoded_target`. **Output:** an `n_sum`-qubit gate labelled `"OracleGate"`: X gates on the zero bits of the target, a multi-controlled Z, then the X gates again.

**`_grover_diffuser(enc) -> Gate`** — inversion about the mean. **Input:** the encoding, which supplies `n_ind`. **Output:** an `n_ind`-qubit gate labelled `"GroverDiffuser"`. Returns a `Gate` for consistency with the other helpers.

**`_multi_controlled_z(qc, qubits) -> None`** (static) — applies a multi-controlled Z in place via `H–MCX–H`, handling the degenerate 1- and 2-qubit widths that would otherwise make `mcx` fail with an empty control list. Uses the modern `mcx(controls, target)` signature; the removed `mode="noancilla"` keyword is gone. **Output:** `None`; modifies `qc`.

**`_step_gate(enc) -> Gate`** — assembles one full step (add → oracle → subtract). **Output:** an `(n_ind + n_sum)`-qubit gate labelled `"DGSSP_Step"`.
