# The frozen instance family, the hardware subset and the sweep (WP3)

This note is the methodology section for the instances the paper reports on. It
changes no circuit construction, no encoding and no solver logic — the algorithm
is frozen. It records *which problems* the paper runs, *which of them get QPU
time*, and *why those and not others*, so that the selection is auditable rather
than asserted.

Artefacts:

| File | Written by | Contents |
|---|---|---|
| `paper/instances/family.json` | `paper/scripts/build_family.py` | 463 instances in 44 cells |
| `paper/instances/hardware_subset.json` | `paper/scripts/select_hardware_subset.py` | every instance evaluated, plus the selection |
| `paper/instances/sweep.json` | `paper/scripts/build_sweep.py` | the two iteration-sweep instances |

The appendix table is printed by `paper/scripts/family_table.py`.

---

## 1. The family

### 1.1 Two families, one register-width budget

**Family P** — `n ∈ {2,3,4,5,6}`, all items positive, drawn i.i.d. uniform from
`[1, a_max(n)]`. 20 instances per `(n, M-class)` cell.

**Family S** — `n ∈ {3,4,5}`, magnitudes drawn from the same distribution, then
exactly one item negated for half of each cell and exactly two for the other
half. 10 instances per `(n, M-class)` cell, 5 per sign count.

### 1.2 `a_max(n)`: the register width is guaranteed, not filtered

`dgssp.encoding.build_encoding` charges

```
range_len = sum_pos - sum_neg + 1
n_sum     = ceil(log2(range_len)) + 1        # the +1 is the guard bit
```

With every item non-zero, `range_len = Σ|a_i| + 1`. Therefore

```
n_sum ≤ 5   ⟺   Σ|a_i| ≤ 2^4 - 1 = 15
n_sum ≥ 3   ⟺   Σ|a_i| ≥ 2       (automatic for n ≥ 2)
```

so the design target `n_sum ∈ {3,4,5}` is met exactly by bounding the **item
magnitude**:

```
a_max(n) = floor(15 / n)
```

| `n` | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| `a_max` | 7 | 5 | 3 | 3 | 2 |
| `Σa` range | 2–14 | 3–15 | 4–12 | 5–15 | 6–12 |

This is a *guarantee*, not a rejection filter, and that distinction matters:
drawing from a wider range and discarding over-wide draws would condition the
item distribution on the sum, so the items would no longer be i.i.d. uniform and
the family could not be described in one line. The price is that `a_max(6) = 2`,
so the `n = 6` cells draw items from `{1,2}` only — visible in the empty-cell
report below, and acceptable because `n = 6` needs 11 qubits and is far outside
the hardware subset regardless.

Because `range_len` depends only on the magnitudes, **family S has the same
`n_sum` distribution as family P at the same `n`**. This is deliberate: it
isolates the *cost* of the signed encoding from the *width* it induces, which is
what WP2 measured (the offset costs `2b` single-qubit phases and exactly zero
two-qubit gates).

### 1.3 Targets and M-classes

For each drawn item set the entire sum spectrum is enumerated by
`dgssp.solvers.classical.subset_sum_multiplicities` — the same recurrence as
`DPSSPSolver`, counting rather than listing, so `M(t)` for *every* candidate `t`
costs one pass instead of one DP per candidate. A target is then drawn uniformly
from the sorted list of values realising the requested class:

- **M = 0** — unreachable, but strictly *inside* `[sum_neg, sum_pos]`: a **gap**,
  not an out-of-range value. This is the point of the class, see §1.5.
- **M = 1, 2, 3+** — exactly one, exactly two, at least three exact solutions.

Target `0` is excluded throughout. The empty subset always reaches it, which
would make the M=1 class artificially easy and needs no search to find.

`r_opt = optimal_iterations(n, M)`. For an M=0 instance the library clamps `M`
to 1, so the recorded `r_opt` is the iteration count an experimenter would use
**not knowing** the instance is infeasible. That is the right number: the whole
point of the infeasible cells is what happens when you run the algorithm you
would actually have run.

### 1.4 Cell census

463 instances in 44 cells; 40 cells are full. The four short cells are short for
structural reasons, and `tests/test_paper_instances.py` proves it by exhaustive
enumeration over `[1, a_max]^n` rather than trusting the generator's stall
counter — in every case the generator found **all** the instances that exist:

| Cell | Wanted | Produced | Why |
|---|---|---|---|
| P, n=2, M=2 | 20 | 7 | Two positive items give sums `{0, a, b, a+b}`; a doubly-reachable target needs `a = b`, and then only `t = a` works. Seven instances, one per `a ∈ [1,7]`. |
| P, n=2, M=3+ | 20 | **0** | Three of the four subset sums coincide only if an item is zero. The cell is empty for every positive item set — not a sampling failure. |
| P, n=3, M=3+ | 20 | 10 | With `a_max = 5` only ten `(items, target)` pairs reach a triply-degenerate target. |
| P, n=6, M=0 | 20 | 6 | With items in `{1,2}` every integer in `[0, Σa]` is reachable unless all items are 2, and then only the six odd targets `1,3,…,11` are gaps. |

Every family-S cell is full: sign flips create gaps readily, so the M=0 class is
much easier to populate for signed instances than for positive ones.

### 1.5 `target_is_representable` is not a feasibility test

`SumRegisterEncoding.target_is_representable` is `0 ≤ encoded_target <
range_len` — a statement about whether the target fits the register's window,
**not** about whether any subset reaches it. So

```
target_is_representable == (M > 0)          ← FALSE in general
```

and false by construction here. What holds, and what the tests assert:

- `M > 0` ⟹ representable (a reachable target is inside the interval);
- the converse fails, and the M=0 cells are 116 witnesses;
- **every** instance in `family.json` is representable, M=0 included.

That is exactly the design intent. An out-of-range target is rejected up front —
the solver warns, the oracle marks nothing, and the failure is trivial and
uninteresting. An in-range unreachable target produces a perfectly well-formed
circuit whose oracle happens to mark no state, which is the infeasible-instance
pathology contribution C3 is about and the case where ZNE on a sampling
distribution has nothing to extrapolate towards.

### 1.6 Reproducibility

Every draw comes from one `random.Random(20261020)` consumed in cell order, so
the instance list is a pure function of the seed. The provenance block carries a
timestamp and a hostname, so the file cannot be byte-identical across runs; the
deterministic part is hashed into `payload_sha256` and

```
python paper/scripts/build_family.py --check
```

regenerates and compares that hash without writing anything.

---

## 2. The hardware subset

### 2.1 Method

`select_hardware_subset.py` evaluates **all 463 instances** — no screening
heuristic, so the ranking carries no caveat. For each:

1. Build the circuit **at that instance's own `r_opt`**, not at one iteration.
   WP2's published survival figures are `r = 1` figures and are *not* reusable
   here; survival falls as the `r`-th power of the per-iteration factor.
2. Transpile once at optimization level 3 on the fake twin of the primary
   device, sweeping SABRE seeds 0–31 with `find_best_seed`. This is the paper's
   transpile-once step — the ISA circuit produced here is what ZNE would fold.
3. Survival from the device's median calibrated gate errors, via
   `dgssp.backends.median_gate_errors` and `resource_scaling.survival_estimate`.
   Neither is reimplemented. The gate counts fed to it come from
   `dgssp.backends.physical_gate_counts`, which counts only the gates the
   medians describe: the virtual `rz` frame changes (about a third of the
   single-qubit operations) are error-free and are excluded from the count just
   as they are excluded from `ε₁`. An earlier run charged `ε₁` to every `rz`
   and understated `S` by ×1.02–1.35 on the selected instances; every number
   below is from the corrected run.
4. `P_ideal`: noiseless Aer at `r_opt`, 20 000 shots, seeded, through
   `dgssp.execution.execute_ideal` and hence `dgssp.runtime.sample_counts`. For
   M=0 it is exactly 0 (no marked state) and is asserted rather than sampled;
   the file records which.

Ranking key is `(two_q_depth, two_q_count, name)` ascending — a total order, so
the result cannot depend on file or dict order.

**Device**: `fake_marrakesh`, the twin used throughout WP2. This is a
**placeholder** until `paper/hardware/account_facts.md` is filled. The real
primary device is whichever Heron the account exposes with the lowest median
two-qubit error and a tolerable queue; re-run with `--backend` once known.

### 2.2 The edge criterion

The edge instance is the largest one whose estimated success probability still
beats random guessing by a factor of two:

```
P_est = S·P_ideal + (1 − S)·P_uniform  ≥  2·P_uniform,     P_uniform = M / 2^n
```

The mixture is the honest reading of the survival estimate: with probability `S`
no gate errs and the circuit delivers its ideal distribution; otherwise the
outcome is treated as a uniform draw over the index register, which still lands
on a solution with probability `P_uniform`. Rearranged:

```
S ≥ P_uniform / (P_ideal − P_uniform)
```

so **the bar rises as `P_uniform` falls**. Comparing `S` against `P_uniform`
directly — the obvious-looking test — inverts this: that threshold falls
exponentially in `n` while `S` falls only as fast as the gate count grows, so
larger instances would pass *more* easily and the test would degenerate into
"return the biggest instance in the family". That test is wrong and is not used;
`tests/test_paper_hardware_subset.py` pins a case where the two disagree.

M=0 instances are excluded from edge candidacy: with `P_uniform = 0` the
criterion reads `0 ≥ 0`, true for every infeasible instance and meaningless.
They remain *in* the subset — the infeasible pathology is C3 — they are just not
eligible to define the edge.

Instances with `r_opt = 1` are excluded as well. The edge is meant to mark
where *search* stops working, and a one-iteration circuit barely searches:
without this rule the edge is `P-n6-M3p-00`, with `M = 11` of 64 states, which
beats guessing because a single oracle-plus-diffuser is cheap, not because
amplitude amplification survives at `n = 6`. Requiring `r_opt ≥ 2` keeps the
edge a statement about search.

"Largest" is `(n_items, n_sum, two_q_count)` descending.

### 2.3 Selected subset

Ten instances: the shallowest in each of the nine `n ∈ {2,3,4} × M ∈ {0,1,2}`
cells, plus the edge instance.

| name | n | b | M | r_opt | 2Q | 2Q depth | S | P_ideal | P_uniform | P_est | ≥2× |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P-n2-M0-06 | 2 | 4 | 0 | 1 | 137 | 112 | 5.95e−01 | 0.0000 | 0.0000 | 0.00e+00 | n/a |
| P-n2-M1-11 | 2 | 3 | 1 | 1 | 76 | 69 | 7.51e−01 | 1.0000 | 0.2500 | 8.13e−01 | yes |
| P-n2-M2-03 | 2 | 3 | 2 | 1 | 83 | 75 | 7.31e−01 | 0.5009 | 0.5000 | 5.01e−01 | no (null control) |
| P-n3-M0-06 | 3 | 4 | 0 | 2 | 370 | 279 | 2.45e−01 | 0.0000 | 0.0000 | 0.00e+00 | n/a |
| P-n3-M1-17 | 3 | 3 | 1 | 2 | 241 | 208 | 4.02e−01 | 0.9417 | 0.1250 | 4.54e−01 | yes |
| P-n3-M2-17 | 3 | 4 | 2 | 1 | 175 | 142 | 5.15e−01 | 1.0000 | 0.2500 | 6.36e−01 | yes |
| P-n4-M0-09 | 4 | 5 | 0 | 3 | 1206 | 794 | 1.04e−02 | 0.0000 | 0.0000 | 0.00e+00 | n/a |
| P-n4-M1-10 | 4 | 4 | 1 | 3 | 708 | 531 | 6.85e−02 | 0.9621 | 0.0625 | 1.24e−01 | no |
| P-n4-M2-02 | 4 | 4 | 2 | 2 | 465 | 327 | 1.70e−01 | 0.9448 | 0.1250 | 2.65e−01 | yes |
| **S-n4-g1-M2-00** | 4 | 4 | 2 | 2 | 488 | 335 | 1.57e−01 | 0.9448 | 0.1250 | 2.53e−01 | **edge** |

`P-n2-M2-03` cannot pass even in principle: at `M/N = 1/2` the Grover operator
gives `sin²(3π/4) = 1/2` for any `r`, so its ideal output *is* uniform. It is a
null control — any structure in its hardware output is noise — and is labelled
as such, not reported as a failure. Seven feasible instances in the family have
`M/N ≥ 1/2`; none can clear a 2× bar.

`P_ideal` was checked against the analytic `sin²((2r+1)θ)`, `θ = arcsin√(M/N)`,
on all 347 feasible instances: largest deviation 2.5σ at 20 000 shots, none
beyond 3σ.

### 2.4 What the table says

**Where the algorithm fails.** 55 of 347 feasible instances pass the 2× bar; 20
of them do so with `r_opt ≥ 2`. By width:

| `n` | passing / feasible | of which `r_opt ≥ 2` |
|---|---|---|
| 2 | 12 / 27 | 0 (every `n = 2` instance has `r_opt = 1`) |
| 3 | 23 / 80 | 11 |
| 4 | 18 / 90 | 9 |
| 5 | 1 / 90 | 0 |
| 6 | 1 / 60 | 0 |

**No genuine search passes beyond `n = 4`.** The only passers at `n = 5` and
`n = 6` are one-iteration instances on highly degenerate targets
(`P-n6-M3p-00`: `M = 11` of 64, `P_est` 0.361 against a bar of 0.344). This is
the C3 headline, now a measured statement at `r_opt` rather than an
extrapolation from WP2's `r = 1` numbers. By M-class, the widest passing
instance is:

| M-class | widest passing | `r_opt` | margin |
|---|---|---|---|
| 1 | `P-n3-M1-16`, n = 3 | 2 | `P_est` 0.312 vs 0.250 |
| 2 | `S-n4-g1-M2-00`, n = 4 | 2 | `P_est` 0.253 vs 0.250 |
| 3+ | `P-n6-M3p-00`, n = 6 | 1 | `P_est` 0.361 vs 0.344 |

**The edge is `S-n4-g1-M2-00`, and it sits on the boundary.** It is the largest
of the 20 edge-eligible instances, and its margin is 1.3% (`P_est` 0.2533 vs
0.2500) — well inside the error of a survival model that is itself an
estimate, optimistic because it ignores idling, crosstalk and leakage (and
readout). Report it as
"`n = 4, M = 2` is where search meets the 2× bar on this device twin", not as
a sharp threshold; on hardware it is expected to fall below.

**The signed encoding now gets hardware time.** The edge is a family-S instance.
It also forms a near-matched pair with the `n = 4, M = 2` cell winner
`P-n4-M2-02`: same `n`, `M`, `b = 4` and `r_opt = 2`, with 488 vs 465 two-qubit
gates and depth 335 vs 327. That pair is a direct hardware comparison of the
signed and positive encodings at fixed register width, which is exactly the C1
claim (the offset costs single-qubit phases only). Family P still wins all nine
cells on depth, by 1–5% at equal `b`, so without the edge rule the subset would
contain no signed instance. `select_hardware_subset.py --include-signed` remains
available, but it is no longer needed.

---

## 3. The sweep

Two instances from the hardware subset, both with `M = 1` and `r_opt ≥ 2`, at
distinct `n`, shallowest first within each width. `r_max = r_opt + 1`.

`M = 1` is required so the ideal curve has a single clean peak and `P_uniform` is
unambiguous; `r_opt ≥ 2` is required so there is a rise to observe at all (an
`r_opt = 1` instance has no interior). `r_max = r_opt + 1` is the cheapest
experiment that can separate over-rotation from decoherence: the noiseless curve
must fall at `r_opt + 1`, while the hardware curve may already have been falling
for several iterations. Going further multiplies QPU seconds for a curve that is
already monotonically decaying.

The pair is `P-n3-M1-17` (`r = 1..3`, `r_opt = 2`) and `P-n4-M1-10`
(`r = 1..4`, `r_opt = 3`); see `paper/instances/sweep.json`. They bracket the
edge. At `r_opt` the first has `P_est = 0.454` against a 2× bar of 0.250, so
the rise towards the peak should be visible on hardware. The second has
`P_est = 0.124` against 0.125: it misses the bar by 0.001 even under the
optimistic survival model, so its sweep is expected to show the success
probability decaying into the uniform floor before the ideal peak is reached.

---

## 4. Reproduction

```
python paper/scripts/build_family.py --check          # verifies family.json
python paper/scripts/select_hardware_subset.py --overwrite
python paper/scripts/build_sweep.py --overwrite
python paper/scripts/family_table.py --out paper/tex/family_table.tex
```

The selection run takes about eight minutes (463 circuits × 32 SABRE seeds at
optimization level 3, plus 347 noiseless 20 000-shot runs). It is deterministic:
`find_best_seed` sweeps a fixed seed range, the Aer runs are seeded in both the
simulator and the transpiler, and every ordering key is total.
