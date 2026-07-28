# `dgssp.solvers.base`

## What the module does

Defines the abstract interfaces every solver implements, so that quantum and
classical approaches can be swapped where it makes sense while still being
honest about the fact that they return different things: a classical solver
returns an answer, a quantum solver returns a *circuit* that still has to be
executed.

## Contents

| Name | Kind |
|---|---|
| `BaseSSPSolver` | abstract base class |
| `BaseQuantumSSPSolver` | abstract base class |
| `BaseClassicalSSPSolver` | abstract base class |

---

### `BaseSSPSolver` — ABC

The common root. Subclasses decide what `solve` returns.

**Abstract methods**

- `solve(instance, *args, **kwargs) -> Any` — **Input:** a `SubsetSumInstance` plus implementation-specific options. **Output:** implementation-defined: a `DPResult` for classical solvers, a `QuantumCircuit` for quantum ones.

---

### `BaseQuantumSSPSolver(BaseSSPSolver)` — ABC

For solvers that build circuits. Execution — backend choice, shots, mitigation
— is deliberately *not* part of this interface; it belongs to
`dgssp.execution` and `dgssp.experiments`.

**Abstract methods**

- `build_circuit(instance, **kwargs) -> QuantumCircuit` — **Input:** the instance, plus solver-specific keyword arguments (the D-G solver accepts `num_solutions`). **Output:** a circuit ready for transpilation and execution.

**Concrete methods**

- `solve(instance, *args, **kwargs) -> QuantumCircuit` — default implementation that simply forwards to `build_circuit`, so a quantum solver satisfies the base interface without pretending to execute anything.

---

### `BaseClassicalSSPSolver(BaseSSPSolver)` — ABC

For solvers that compute an answer directly.

**Abstract methods**

- `solve(instance) -> DPResult` — **Input:** the instance. **Output:** a `DPResult` with the best solution, all solutions and metadata.
