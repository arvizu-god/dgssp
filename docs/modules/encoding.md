# `dgssp.encoding`

## What the module does

Owns the arithmetic that maps (possibly negative) subset sums onto the unsigned
integer register the QFT adder manipulates. It is the single source of truth
for that mapping: the solver, the phase oracle and the decoder all read it from
here, so they cannot silently disagree.

Two ingredients make signed instances work:

**Offset.** Every reachable sum lies in `[sum_neg, sum_pos]`. Adding
`offset = -sum_neg` shifts that interval to `[0, range_len - 1]`, so the encoded
target is always a well-defined non-negative bit pattern and the oracle never
has to shift a negative Python integer.

**Guard bit.** The register is one bit wider than `range_len` strictly needs,
guaranteeing `modulus >= 2 * range_len`. Without that headroom, an instance
whose range is exactly a power of two fills the register completely, and any
off-by-one in the marked value wraps around and aliases a *different* reachable
sum.

The module contains no Qiskit code — it is pure integer arithmetic and is
independently unit-testable.

## Contents

| Name | Kind |
|---|---|
| `SumRegisterEncoding` | frozen dataclass |
| `build_encoding` | function |

---

### `SumRegisterEncoding` — frozen dataclass

An immutable record of how one instance's sums map onto the register.

**Fields**

- `items: list[int]` — the instance items, in order.
- `target: int` — the unencoded target (may be negative).
- `sum_neg: int` — sum of all strictly negative items (≤ 0); the lower bound of the reachable interval.
- `sum_pos: int` — sum of all strictly positive items (≥ 0); the upper bound.
- `offset: int` — `-sum_neg`, the non-negative shift constant.
- `range_len: int` — `sum_pos - sum_neg + 1`, the size of the reachable interval.
- `n_sum: int` — sum-register width *including* the guard bit.
- `modulus: int` — `2 ** n_sum`, the ring the phase adder works in.
- `encoded_target: int` — `target + offset`.

**Properties**

- `n_ind -> int` — index-register width, i.e. `len(items)`.
- `target_is_representable -> bool` — whether `encoded_target` lies in `[0, range_len - 1]`. A `False` here means no subset can reach the target, so the oracle marks nothing and the search cannot succeed.

**Methods**

- `encode(value: int) -> int` — **Input:** a signed subset sum. **Output:** `(value + offset) mod modulus`, the bit pattern the register holds for it.
- `decode(register_value: int) -> int` — **Input:** an integer read out of the register. **Output:** the signed sum it represents. Exact inverse of `encode` for reachable sums.

---

### `build_encoding(items, target) -> SumRegisterEncoding` — function

Computes the offset and register width for an instance.

**Inputs**

- `items: Sequence[int]` — the instance items; negatives, zeros and duplicates are all fine.
- `target: int` — the target sum; may be negative.

**Output**

A frozen `SumRegisterEncoding`. The width is computed as
`n_core = max(1, ceil(log2(range_len)))` followed by `n_sum = n_core + 1`, which
is what guarantees the `modulus >= 2 * range_len` headroom.

**Raises:** `ValueError` if `items` is empty, or if `range_len` is non-positive
(defensive — this cannot happen for a well-formed instance).
