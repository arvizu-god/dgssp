# `dgssp.instance`

## What the module does

Defines the problem data structures for the Subset Sum Problem and the helpers
that validate and generate them. It is deliberately free of any Qiskit
dependency so that the classical and quantum halves of the library share one
representation of a problem, and so that instances can be constructed,
serialised and reasoned about without a quantum stack installed.

An instance is a list of integer items plus a target sum; a solution is a
selection of item indices together with their values, their total, and whether
that total matches the target.

## Contents

| Name | Kind |
|---|---|
| `SubsetSumInstance` | dataclass |
| `SSPSolution` | dataclass |
| `DPResult` | dataclass |
| `validate_instance` | function |
| `random_instance` | function |

---

### `SubsetSumInstance` — dataclass

A single Subset Sum problem: choose a subset of `items` summing to `target`.
Validated on construction, and `items` is always coerced to a concrete `list`.

**Fields**

- `items: list[int]` — the integer weights. May contain negatives, zeros and duplicates.
- `target: int` — the sum to hit. May be negative.
- `name: str | None` — optional human-readable label, used in logs and batch reports.
- `metadata: dict[str, Any]` — free-form extra information (generator parameters, difficulty tags, provenance).

**Properties and methods**

- `n_items -> int` — number of items, which is also the width of the quantum index register.
- `subset_from_indices(indices: Sequence[int]) -> SSPSolution` — builds a solution record from 0-based item indices. **Input:** the selected indices. **Output:** an `SSPSolution` carrying those indices, the corresponding values, their total, and `is_exact = (total == target)`.

---

### `SSPSolution` — dataclass

A candidate or confirmed solution. Note that it does *not* hold a reference to
its instance; the solver or result object owns that association.

**Fields**

- `indices: list[int]` — 0-based indices of the selected items.
- `subset: list[int]` — the values of those items.
- `total: int` — their sum.
- `is_exact: bool` — whether `total` equals the instance target.

**Methods**

- `__repr__() -> str` — a compact one-line representation for debugging.

---

### `DPResult` — dataclass

The output of the classical dynamic-programming solver, used both as a
baseline and as the ground truth against which quantum runs are scored.

**Fields**

- `instance: SubsetSumInstance` — the instance that was solved.
- `best_solution: SSPSolution | None` — one solution, or `None` if the instance is infeasible.
- `all_solutions: list[SSPSolution]` — every exact solution found, when enumeration was requested.
- `optimal_value: int | None` — the achieved total, normally equal to the target.
- `success: bool` — whether the solver completed meaningfully.
- `metadata: dict[str, Any]` — algorithm name, item count, solution count.

---

### `validate_instance(instance) -> None` — function

Sanity-checks a newly constructed instance. Called automatically from
`SubsetSumInstance.__post_init__`, so you rarely call it yourself.

- **Input:** `instance: SubsetSumInstance`.
- **Output:** `None`.
- **Raises:** `ValueError` if the item list is empty, if any item is not an `int`, or if the target is not an `int`.
- **Warns:** `UserWarning` if the instance has more than 24 items, since the index register needs one qubit per item and such an instance is impractical to simulate. This is a warning, not a refusal.

---

### `random_instance(...) -> SubsetSumInstance` — function

Generates a random instance for testing and benchmarking.

**Inputs**

- `n_items: int` — how many items to generate; must be ≥ 1.
- `max_value: int = 20` — items are drawn uniformly from `[1, max_value]`.
- `target_strategy: str = "feasible"` — `"feasible"` samples a hidden subset and sets the target to its sum, guaranteeing at least one solution; `"random"` picks a target uniformly from `[1, n_items * max_value]`, which may be infeasible.
- `density: float = 0.5` — for the feasible strategy, the Bernoulli probability that each item joins the hidden subset. If nothing is selected, one item is forced in.
- `seed: int | None` — PRNG seed for reproducibility.
- `name: str | None` — label for the instance; defaults to `ssp_random_n{n_items}`.

**Output**

A `SubsetSumInstance` whose `metadata` records the generator, seed, `max_value`,
strategy and density, so any generated instance can be reproduced exactly.

**Raises:** `ValueError` for `n_items < 1`, `max_value < 1`, a density outside
`[0, 1]`, or an unknown strategy.
