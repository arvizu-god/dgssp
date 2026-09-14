# Resource model of the Draper–Grover SSP circuit

**Work package:** WP2, contribution **C1**.
**Scope:** the circuit that `dgssp.solvers.draper_grover.DGSSPSolver.build_circuit`
actually builds, at commit time of this note. Everything below is read off
`src/dgssp/solvers/draper_grover.py` and `src/dgssp/encoding.py`, not from the
thesis and not from the Draper adder literature. Where the implementation
differs from the textbook construction the difference is stated explicitly.

The closed forms are implemented in
`dgssp.solvers.draper_grover.resource_model()` and checked against the built
circuit, grid point by grid point, in `paper/scripts/resource_scaling.py`
(step 2) and in `tests/test_resource_model.py`. If a formula here and the
circuit ever disagree, the assert fires and **this note is what gets fixed**.

## 0. Notation

| symbol | meaning | source |
|---|---|---|
| `n` | index-register width = number of items | `SumRegisterEncoding.n_ind` |
| `b` | sum-register width, **including the guard bit** | `SumRegisterEncoding.n_sum` |
| `2^b` | modulus of the phase adder | `SumRegisterEncoding.modulus` |
| `r` | Grover iterations | `DGConfig.iterations` |
| `M` | number of exact solutions | classical DP ground truth |
| `a_k` | item `k` | `SumRegisterEncoding.items` |
| `ã_k` | `a_k mod 2^b`, the residue the adder actually applies | — |
| `n₊` | number of items with `ã_k ≠ 0` | `DGResourceModel.nonzero_items` |
| `ω` | `offset mod 2^b` | `SumRegisterEncoding.offset` |
| `z` | number of zero bits of `encoded_target` in the `b`-bit window | — |
| `v₂(x)` | 2-adic valuation of `x` (largest `v` with `2^v ∣ x`) | — |

Register width, from `encoding.build_encoding`:

```
sum_neg  = Σ_{a_k < 0} a_k          sum_pos = Σ_{a_k > 0} a_k
offset   = -sum_neg  ≥ 0            range_len = sum_pos - sum_neg + 1
b = ceil(log2(range_len)) + 1       (the +1 is the guard bit)
```

Note that `range_len` is **invariant under flipping the sign of any item**:
flipping `a_j > 0` moves `a_j` out of `sum_pos` and `-a_j` into `sum_neg`, and
`sum_pos - sum_neg` is unchanged. Signed instances therefore cost **no extra
sum-register width** — see §7.

## 1. What one Grover iteration contains

`DGSSPSolver.build_circuit` emits

```
H^{⊗n} on the index register
r × [ DGSSP_Step  ,  GroverDiffuser ]
measure index register
```

and `_step_gate` expands to

```
QFT_b · SumGate · QFT_b†      (add)
OracleGate                    (mark)
QFT_b · SubGate · QFT_b†      (uncompute)
```

So **one iteration contains exactly four QFT-type blocks on `b` qubits**: two
forward `QFT` and two inverse `QFT†`. This is the `FullQFT` assembly.

All four are **necessary**, and an earlier draft of this note was wrong to say
otherwise. The tempting saving is to cancel the `QFT†` that closes the add block
against the `QFT` that opens the uncompute block — but they are not adjacent:
`OracleGate` sits between them, and it needs the sum register in the
*computational* basis to recognise `|encoded_target⟩`. Being diagonal in that
basis does not let it commute past a QFT, so the pair cannot be removed.
Folding the target into the adder's constant (marking `|0…0⟩` instead of
`|t⟩`) does not help either: it moves X gates around and leaves all four QFT
blocks in place. The four blocks stand.

## 2. Controlled phases from the QFT blocks

`qft_no_swaps(b)` emits, for `j = b-1 … 0`: one `H` on qubit `j` and `cp` on
every pair `(k, j)` with `k < j`. Hence per block

```
#CP(QFT)  = Σ_{j=0}^{b-1} j = b(b-1)/2          #H(QFT) = b
```

and the inverse has the same counts. With four blocks per iteration:

> **#CP_QFT(b) = 4 · b(b-1)/2 = 2b(b-1)  per iteration**
> **#H_QFT(b)  = 4b                      per iteration**

## 3. Controlled phases from the controlled additions

`_sum_gate` is

```python
for k, a in enumerate(enc.items):
    a_mod = a % mod
    if a_mod == 0:
        continue
    for j in range(enc.n_sum):
        qc.cp((2*pi*a_mod) / 2**(j+1), ind[k], summ[j])
```

and `_sub_gate` is identical with `a_mod` replaced by `(-a) % mod`. The only
skip is at the **item** level: an item whose residue is zero contributes
nothing. Since `ã_k = 0 ⟺ (-a_k) mod 2^b = 0`, both blocks skip the same items.

> **#CP_add(n, b) = 2 · n₊ · b  per iteration**
> worst case `n₊ = n`: **#CP_add ≤ 2nb**, i.e. **≤ n·b per addition block**.

### 3.1 The implementation does not skip 2π multiples

The rotation applied by item `k` on sum qubit `j` is `2π·ã_k / 2^{j+1}`, which
is an exact multiple of `2π` — an identity — precisely when `2^{j+1} ∣ ã_k`,
i.e. for `j = 0 … v₂(ã_k) - 1`. **The code emits those gates anyway.** There is
no test on the angle in `_sum_gate` or `_sub_gate`, so `#CP_add` is *independent
of the binary expansion of the items* beyond the whole-item `ã_k = 0` test. The
"exact count as a function of the binary expansion" asked for by the work
package therefore does not exist for this implementation; what exists is the
count the implementation *would* have if the skip were added:

```
#CP_add_skippable = 2 · Σ_{k : ã_k ≠ 0} ( b - v₂(ã_k) )
```

(using `v₂((-a) mod 2^b) = v₂(a mod 2^b)` for `ã_k ≠ 0`, since
`2^b - ã_k` and `ã_k` share a valuation below `b`). The saving is
`2 Σ_k v₂(ã_k)` controlled phases per iteration — zero for all-odd items, and
up to `2 Σ_k (b-1)` for items that are large powers of two. Both numbers are
recorded per grid point in `paper/data/resources.json` as
`adder_cp_per_iteration` and `adder_cp_per_iteration_skippable`, and the
measured gap is what §8 quotes.

This is a **real but bounded** missed optimisation: the discarded gates are
identities, so the transpiler's `Optimize1qGatesDecomposition` /
`CommutativeCancellation` passes remove most of them at optimization level 3.
It matters for the *logical* count in the paper's table, and for anyone reading
the circuit, more than it matters for the hardware.

## 4. The uncontrolled offset addition

```python
off = enc.offset % mod
if off:
    for j in range(enc.n_sum):
        qc.p((2*pi*off) / 2**(j+1), summ[j])
```

Emitted once in `_sum_gate` and once (with `-offset`) in `_sub_gate`.

> **#P_offset(b) = 2b single-qubit phase gates per iteration when `ω ≠ 0`,
> and 0 otherwise. Two-qubit cost: zero.**

`ω ≠ 0` exactly when the instance has at least one negative item. The same 2π
remark as §3.1 applies: the skippable count is `2(b - v₂(ω))`.

The uncompute of the offset is not a separate structure — it is the `p` gates
inside `SubGate`, and it is exact because `(-offset) mod 2^b` is the modular
inverse of `offset mod 2^b` in the additive group the phase adder acts on. The
whole `SubGate` is the exact inverse of `SumGate`; `tests/test_sum_register_uncomputes.py`
checks that the sum register returns to `|0…0⟩`.

## 5. The two multi-controlled Z blocks

Both the target comparison and the diffuser go through the same helper:

```python
@staticmethod
def _multi_controlled_z(qc, qubits):
    if len(qubits) == 1:   qc.z(qubits[0])
    elif len(qubits) == 2: qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1]); qc.mcx(qubits[:-1], qubits[-1]); qc.h(qubits[-1])
```

- **Target comparison (`_oracle_gate`)**: `X` on every zero bit of
  `encoded_target`, one MCZ across all `b` sum qubits, `X` back. Width `b`,
  `2z` X gates.
- **Diffuser (`_grover_diffuser`)**: `H^{⊗n}`, `X^{⊗n}`, one MCZ across all `n`
  index qubits, `X^{⊗n}`, `H^{⊗n}`. Width `n`, `2n` X and `2n` H gates.

> **#MCX(n, b) = 2 per iteration**, of widths `b` and `n`.

### 5.1 Ancilla mode

`qc.mcx(controls, target)` is called with **no ancilla argument**, so the
circuit declares `MCXGate(k)` with `k = width - 1` controls and **no ancilla
register**: total qubits stay `n + b` (§6). What that gate *costs* is decided
later, by `HighLevelSynthesis` during transpilation, and this is the one place
where "the 2Q cost after decomposition" is not a property of the circuit alone:

1. Qiskit 2.5 chooses the cheaper of the `noaux_v24` and `noaux_hp24` plugins,
   by control count.
2. Inside the full circuit, `HighLevelSynthesis` **opportunistically borrows
   idle register qubits as ancillas**. During the oracle's MCZ the `n` index
   qubits are idle (dirty, in superposition); during the diffuser's MCZ the `b`
   sum qubits are idle (clean, restored to `|0…0⟩` by `SubGate`). A v-chain
   with ancillas costs `Θ(k)` instead of `Θ(k²)`, so the same gate in the same
   circuit costs a *different* amount depending on how many spare qubits happen
   to be around. Measured example: `mcx` with 5 controls costs 84 CX in a
   6-qubit circuit and 24 CX in an 8-qubit one.

A closed form needs a fixed synthesis. The paper therefore quotes logical
counts in a pinned **ancilla-free reference**:
`dgssp.transpilation.to_logical_basis` — optimization level 0, basis `{cx, u}`,
`HLSConfig(mcx=[("noaux_v24", {})])`. Deterministic, a function of `(n, b)`
alone, and an *upper* bound on what the transpiler does. The measured
ancilla-borrowing count is recorded alongside it as `two_q_logical_hls`.

### 5.2 Cost of the reference MCX

Measured (not hand-tabulated) by `transpilation.mcx_noaux_cx_cost(k)`, which
translates a bare `(k+1)`-qubit `mcx`; Qiskit 2.5.2:

| `k` (controls) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| CX | 1 | 6 | 14 | 36 | 84 | 140 | 220 | 324 | 444 | 580 |
| 1Q | 0 | 9 | 17 | 69 | 127 | 199 | 321 | 493 | 689 | 909 |

For `k ≥ 7` the CX count is **exactly** `f(k) = 8k² − 16k − 60` (verified to
`k = 15`), so the reference MCX is `Θ(k²)`, asymptotically `8k²` CX. The MCZ
cost as a function of *width* `w` is

```
g(w) = 0            w = 1   (a bare Z)
     = 1            w = 2   (a CZ)
     = f(w-1)       w ≥ 3
     = 8w² - 32w - 36   for w ≥ 8
```

Because these are measured at call time rather than hard-coded, the numbers in
this table are the ones for the pinned environment
(`paper/requirements-lock.txt`); a Qiskit upgrade will move them, and the
assert in `resource_scaling.py` step 2 will keep the note honest rather than
silently stale.

## 6. Qubits

```
n_ind = n          one qubit per item
n_sum = b          sum register, guard bit included
ancillas = 0
──────────────────────────────────────────
total = n + b
```

plus an `n`-bit classical register when `DGConfig.measure` is true. No ancilla
is *declared*; §5.1 explains that the transpiler nonetheless uses idle register
qubits as ancillas when synthesising the multi-controlled gates, which costs no
extra width.

## 7. Signed-encoding overhead

Two separable charges, both isolated in `encoding.build_encoding`:

**(a) The offset addition** — the price of allowing negative items. It is
`2b` single-qubit phase gates per iteration and **zero two-qubit gates**
(§4). Because `range_len` is invariant under sign flips (§0), the register
width is identical for `[a₁…aₙ]` and for the same list with any subset negated,
so the offset is the *entire* signed overhead. Recorded per grid point as
`signed_overhead` in `paper/data/resources.json`.

**(b) The guard bit** — charged unconditionally, to signed and all-positive
instances alike. It is not a signed-encoding cost, it is an aliasing-safety
cost, and it is by far the larger of the two. Costing it means asking what the
circuit would be at width `b-1` (`encoding_without_guard_bit`, a deliberately
invalid encoding used only for this difference):

```
ΔCP  = 2b(b-1) + 2n₊b  -  [ 2(b-1)(b-2) + 2n₊(b-1) ]
     = 4(b-1) + 2n₊                       per iteration
Δ2Q  = 2·ΔCP + g(b) - g(b-1)
Δqubits = 1
```

`g(b) - g(b-1)` is the dominant term at the top of the grid: `g(8) - g(7) = 80`
CX. Recorded per grid point as `guard_bit_overhead`.

## 8. Totals and asymptotics

Per iteration, exactly (this is what `DGResourceModel` returns):

```
#CP(n,b)   = 2b(b-1) + 2 n₊ b                    ≤ 2b(n + b - 1)
#P(b)      = 2b·[ω ≠ 0]
#H(n,b)    = 4b + 2n + 2·[b ≥ 3] + 2·[n ≥ 3]
#X(n,b)    = 2z + 2n
#MCX       = 2            (widths b and n)
```

Whole circuit with `r` iterations: multiply by `r`, then add the initial
Hadamard layer (`n` gates) and `n` measurements.

Two-qubit count in the pinned reference of §5.1 — each `cp` is 2 CX, each MCZ
is `g(·)`:

> **#2Q(n, b, r) = r · [ 4b(b-1) + 4 n₊ b + g(b) + g(n) ]**
> worst case `n₊ = n`: **#2Q ≤ r · [ 4b(n + b - 1) + g(b) + g(n) ]**

with the iteration count from `optimal_iterations`:

> **r_opt(n, M) = max(1, ⌊ π / (4 · arcsin √(M / 2ⁿ)) ⌋)**,
> returning 1 when `M ≥ 2ⁿ` (the search is saturated and further amplification
> over-rotates). For `M ≪ 2ⁿ`, `r_opt ≈ (π/4)√(2ⁿ/M) = Θ(2^{n/2}/√M)`.

**Asymptotics.** Take items bounded by a polynomial in `n`, so
`b = Θ(log n)`. Per iteration:

```
adder + QFT :  4b(n + b - 1) = Θ(n log n)
oracle MCZ  :  g(b) = Θ(b²)  = Θ(log² n)
diffuser MCZ:  g(n) = Θ(n²)
──────────────────────────────────────────
per iteration:               Θ(n²)
whole circuit: Θ( 2^{n/2} n² / √M )
```

**The ancilla-free diffuser dominates**, not the arithmetic. The Draper adder —
the part of this construction that is actually interesting — is `Θ(n log n)`
per iteration and is asymptotically free next to a `Θ(n²)` multi-controlled Z
that carries no problem structure at all. Two consequences for the paper:

1. Any statement of the form "the QFT adder is dominant" is true only over the
   measured window and false in the limit. With `g(·)` from §5.2, the diffuser
   MCZ overtakes the whole arithmetic block (`g(n) > 4b(n+b-1)`) at `n = 10`
   for `b = 6`, `n = 11` for `b = 8` and `n = 13` for `b = 10` — just past the
   top of this grid, where at `n = b = 8` the arithmetic is 480 CX against 220
   for each MCZ. So the measured table is still adder-dominated everywhere, and
   the asymptotic statement must be phrased as a limit, not read off the plot.
2. Rewriting the MCZs with an explicit dirty-ancilla v-chain (`Θ(n²) → Θ(n)`)
   would **buy nothing on hardware, because the transpiler already does it**.
   `HighLevelSynthesis` borrows the idle register as ancillas during synthesis,
   which is the 0.78x median in `two_q_logical_hls`; the transpiled counts then
   sit at 1.64x the *ancilla-free* reference, and 0.78 x 2.1 (routing) = 1.64
   exactly. The ancilla saving is therefore already inside every measured
   number in §10. The ancilla-free reference is kept as *the* closed form
   because it is the only synthesis that is a function of `(n, b)` alone
   (§5.1) — not because the paper believes the hardware pays it.

### 8.1 Not pursued in this paper

A third lever exists -- truncating the small rotations of the QFT blocks and the
controlled additions (the approximate-QFT / approximate-Draper-adder
construction). It is real, it is the only one a transpiler cannot take for you,
and it was measured: see `paper/notes/future_work.md`.

It is **deliberately out of scope**. C1 is a resource model of the algorithm as
designed, and changing the circuit partway through the paper would leave the
model describing something with no hardware data behind it. The construction in
this repository is frozen. Nothing in §§1-8 or §10 depends on that decision:
every number here describes the circuit `DGSSPSolver` builds today.

## 9. What is measured, and where

`paper/scripts/resource_scaling.py` builds the grid described in its module
docstring, asserts every formula above against `flatten_containers(circuit).count_ops()`
and against the pinned reference translation, transpiles at optimization level 3
on the heavy-hex fake twins of the devices on the Open plan, and writes
`paper/data/resources.json` + `paper/figures/fig_resources.pdf`. Fitted
effective exponents (`two_q_count ~ b^α` at fixed `n`, `~ n^β` at fixed `b`) go
to `paper/data/resources_fits.json`. None of the closed forms above is a power
law, so those exponents are *effective* exponents over the measured range and
are reported as such.

## 10. Measured summary (run of 2026-09-06, `paper/data/resources.json`)

432 records: 36 feasible `(n, b)` cells x 2 item families x {positive, negated}
x 3 devices (`fake_marrakesh`, `fake_fez`, `fake_torino`), one Grover
iteration, optimization level 3, SABRE seeds 0..31. All 144 circuits pass the
step-2 asserts, so every closed form above is confirmed against the built
circuit at every grid point.

**Routing overhead.** Transpiled 2Q count over the ancilla-free logical
reference: **1.25x to 2.08x, median 1.64x**. Heavy-hex degree-3 connectivity is
therefore worth roughly a factor 1.6 on this circuit family, not the factor
3-5 that a naive all-to-all-to-heavy-hex argument suggests — the adder's
controlled phases are all-to-all between two registers, but SABRE places the
registers interleaved and reuses the routes.

**Borrowed ancillas.** Qiskit's default `HighLevelSynthesis` (SS5.1) lands at
**0.63x to 1.00x of the ancilla-free reference, median 0.78x** — i.e. the idle
register borrowing is already recovering up to 37% of the two-qubit count
before routing. This is the empirical size of the lever in SS8, item 2.

**Signed encoding.** Negating one item changes the transpiled 2Q count by
**-15 to 0, median 0** across all 216 pairs — never a cost, occasionally a small
saving where the different phase angles let the optimiser cancel more. The
closed-form prediction (SS7a: zero two-qubit delta, `2b` single-qubit phases) is
confirmed: 6 to 16 phase gates across the grid, two-qubit delta identically 0.

**Guard bit.** +1 qubit, 12 to 44 controlled phases and 29 to 168 two-qubit
gates per iteration across the grid — an order of magnitude more than the
signed-encoding overhead it is often confused with.

**Item structure.** `sparse` (powers of two) minus `dense` (all odd) transpiled
2Q count: -272 to +90, median -40. The 2π-multiple gates the code does not skip
(SS3.1) are indeed mostly removed by optimization level 3, so the logical gap
between the families shrinks to a ~5% effect after transpilation.

**Effective exponents** (`paper/data/resources_fits.json`, transpiled 2Q count,
all-positive variants):

| slice | dense | sparse |
|---|---|---|
| `~ b^α` at fixed `n` | α = 2.36 (n=2) down to 1.42-1.74 (n=7,8) | 2.19 down to 1.17-1.45 |
| `~ n^β` at fixed `b` | β = 0.80-0.87 (b=4) down to 0.35-0.38 (b=8) | 0.91-1.01 down to 0.40-0.50 |

Standard errors are 0.03-0.26, `R² >= 0.88`. Neither trend is a power law
(SS8), and both effective exponents *fall* as the other parameter grows, because
the terms that do not depend on the swept variable — `2b(b-1)` and `g(b)` when
sweeping `n`, `g(n)` when sweeping `b` — become the constant part of the sum.
Quote these as descriptive fits over `2 <= n <= 8`, `3 <= b <= 8`, never as
asymptotics.

**Survival (C3 input).** `fake_marrakesh`, median 2Q error 3.29e-3, dense
all-positive, one iteration and extrapolated to `r_opt`:

| `n` | `b` | 2Q (r=1) | `r_opt` | S(r=1) | S(r_opt) |
|---|---|---|---|---|---|
| 2 | 3 | 83 | 1 | 0.73 | 0.73 |
| 2 | 8 | 839 | 1 | 0.042 | 0.042 |
| 5 | 5 | 493 | 1 | 0.16 | 0.16 |
| 5 | 8 | 1083 | 4 | 0.016 | 7.0e-8 |
| 8 | 5 | 618 | 1 | 0.094 | 0.094 |
| 8 | 8 | 1382 | 4 | 0.0053 | 7.6e-10 |

`S = (1 - ε₂)^{#2Q} · (1 - ε₁)^{#1Q_phys}`, where `#1Q_phys` counts only the
physical single-qubit gates (`sx`, `x`) — the virtual `rz` frame changes are
error-free and are excluded from the count exactly as they are excluded from the
median `ε₁` (`dgssp.backends.physical_gate_counts`). An earlier version of this
table charged `ε₁` to every `rz` as well, about a third of all single-qubit
operations, and understated `S` by a factor of 1.03–1.5 across the grid
(0.0037 → 0.0053 at `n = b = 8`); the conclusions below were unaffected.

Read this as an upper bound (SS `survival_estimate` docstring): it ignores
idling, crosstalk and readout, and it assumes every gate error is fatal. Even
so, **a single Grover iteration is already below 1% survival at `n = b = 8`, and
the optimal iteration count puts the full search nine orders of magnitude below
useful.** The hardware experiments in WP4 must therefore sit at the bottom-left
of this grid, and the Grover-iteration sweep of C3 is a sweep into a wall, not
around one.
