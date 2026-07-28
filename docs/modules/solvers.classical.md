# `dgssp.solvers.classical`

## What the module does

An exhaustive dynamic-programming Subset Sum solver. It serves two purposes:
as a classical baseline to compare the quantum algorithm against, and — more
importantly in practice — as the **ground truth** that lets every quantum run
be scored automatically. `dgssp.experiments` calls it before every batch to
learn which bitstrings are correct and how many there are, which in turn sets
the optimal Grover iteration count.

The core is a DP over reachable sums that tracks item *indices*, so results map
back to concrete subsets. It handles negatives, zeros and duplicates: each
position in the item list is a distinct element.

## Contents

| Name | Kind |
|---|---|
| `DPConfig` | dataclass |
| `DPSSPSolver` | class |

---

### `DPConfig` — dataclass

**Fields**

- `enumerate_all: bool = True` — enumerate *every* subset summing to the target. Listing all solutions is inherently exponential in their number, which is fine for the instance sizes a quantum simulation can reach.

---

### `DPSSPSolver(BaseClassicalSSPSolver)` — class

Constructed with an optional `DPConfig`.

#### Public methods

**`solve(instance) -> DPResult`**

- **Input:** `instance: SubsetSumInstance`.
- **Output:** a `DPResult` whose `all_solutions` holds one `SSPSolution` per exact subset, `best_solution` is the first of them (or `None` if infeasible), `optimal_value` is the achieved total, and `metadata` records the algorithm name, item count and solution count.

#### Internal methods

**`_dp_all_index_subsets(items, target) -> list[list[int]]`**

The DP core.

- **Inputs:** `items: list[int]`; `target: int`.
- **Output:** a list of index lists, each summing to the target.

**How it works.** Maintain a dictionary mapping each reachable sum to the list
of index-subsets achieving it, starting from `{0: [[]]}` — sum zero via the
empty subset. For each item at position *i*, extend every existing subset with
*i*, producing new sums; merge those into the table alongside the subsets that
skip *i*. Because indices are processed left to right, each element is used at
most once.

**Complexity.** Time and space are exponential in the *number of solutions*,
which is unavoidable when the goal is to list them all, but the method is clean
and fast for the moderate instances used here.

**Edge cases.** If `target == 0` the empty subset is among the results.
Repeated values at different positions count as distinct solutions.
