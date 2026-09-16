# `dgssp.decoding`

## What the module does

Translates between measurement outcomes and Subset Sum objects. The D-G circuit
measures only the index register — one qubit per item, where qubit *k* set to 1
means item *k* is in the subset — and Qiskit reports outcomes most-significant
bit first.

Every helper here obeys one convention:

> index 0 ↔ least-significant bit ↔ **rightmost** character

Keeping all the bit-twiddling in one module means the solver, the backend
evaluators, the mitigation code and the batch runner cannot disagree about
which bitstring corresponds to which subset. The module is Qiskit-free and
depends only on `dgssp.instance`.

> **Deprecated (WP4).** `counts_to_probs`, `solution_probability` and
> `distribution_solution_probability` are metrics, not bit translation, and
> moved to [`dgssp.metrics`](metrics.md) so there is one implementation.
> `from dgssp.decoding import solution_probability` (or attribute access)
> still works, emits a `DeprecationWarning`, and returns the `dgssp.metrics`
> function itself — identical values by construction. They are no longer in
> `decoding.__all__`; `dgssp.solution_probability` and `dgssp.counts_to_probs`
> come from `metrics` and do not warn.

## Contents

| Name | Kind |
|---|---|
| `indices_to_bitstring` | function |
| `bitstring_to_indices` | function |
| `bitstring_to_subset` | function |
| `bitstring_to_solution` | function |
| `bit_is_solution` | function |
| `counts_to_probs` | deprecated → `metrics` |
| `solution_probability` | deprecated → `metrics` |
| `distribution_solution_probability` | deprecated → `metrics` |
| `exact_solution_bitstrings` | function |

---

### `indices_to_bitstring(indices, n_bits) -> str`

Converts selected item indices into a Qiskit-style bitstring.

- **Inputs:** `indices: Sequence[int]` (0-based selected items); `n_bits: int` (register width).
- **Output:** a length-`n_bits` string of `'0'`/`'1'`, most-significant bit first. For example `indices_to_bitstring([0], 3) == "001"`.
- **Raises:** `ValueError` if an index falls outside `[0, n_bits)`.

---

### `bitstring_to_indices(bitstring) -> list[int]`

The inverse translation.

- **Input:** `bitstring: str`, most-significant bit first. Spaces — which Qiskit inserts between classical registers — are ignored.
- **Output:** the sorted 0-based indices of the set bits.

---

### `bitstring_to_subset(bitstring, items) -> list[int]`

- **Inputs:** `bitstring: str`; `items: Sequence[int]` (the instance items).
- **Output:** the *values* of the selected items, in index order.

---

### `bitstring_to_solution(bitstring, instance) -> SSPSolution`

- **Inputs:** `bitstring: str`; `instance: SubsetSumInstance`.
- **Output:** a full `SSPSolution` with indices, values, total and the `is_exact` flag.

---

### `bit_is_solution(bitstring, items, target) -> bool`

- **Inputs:** `bitstring: str`; `items: Sequence[int]`; `target: int`.
- **Output:** `True` if the selected items sum exactly to the target.

---

### `counts_to_probs`, `solution_probability`, `distribution_solution_probability` — deprecated

Moved to [`dgssp.metrics`](metrics.md); see the note above. Signatures and
values are unchanged.

---

### `exact_solution_bitstrings(instance, dp_result) -> list[str]`

Extracts the ground-truth measurement outcomes from a classical DP result. This
is what turns "did the quantum run work?" into a number.

- **Inputs:** `instance: SubsetSumInstance` (supplies the register width); `dp_result: DPResult`.
- **Output:** sorted, de-duplicated bitstrings whose subsets sum exactly to the target; empty if the instance is infeasible.
