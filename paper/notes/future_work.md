# Out of scope for Paper A

Measured observations that are **not** part of this paper and must not enter a
work package, the contribution statement, or the manuscript's claims. The
algorithm in `src/dgssp/solvers/draper_grover.py` is frozen for C1: it is the
circuit the thesis designed, and the paper's job is to measure it, not to
improve it. Anything here is a candidate for a follow-up, or at most one
sentence in a closing "possible improvements" paragraph.

Recorded so the work is not lost and not re-derived.

## Rotation truncation (approximate QFT + approximate Draper adder)

Both the QFT blocks and the controlled additions emit rotations of angle
`2π/2^m` for `m` up to `b`. The smallest of them are below the device's own
gate error and contribute nothing but decoherence. Truncating at level `K`
(keep a rotation only when its reduced angle is at least `2π/2^K`) is the
standard approximate-QFT / approximate-Draper-adder construction, and unlike
the ancilla saving it is **not** something a transpiler can do for you: these
are genuine non-identity rotations, and only the algorithm designer may declare
them negligible.

Measured on `fake_marrakesh` (median 2Q error 3.29e-3), at `r = r_opt`, with
`P_ideal` from a noiseless Aer run at 20 000 shots and
`FOM = P_ideal x (1-ε₂)^{2Q}`:

| instance | `K` | CP | 2Q | 2Q depth | `P_ideal` | `S` | FOM |
|---|---|---|---|---|---|---|---|
| `[3,5,6,9] → 14` (n=4, b=6, M=2, r=2) | exact | 216 | 1174 | 735 | 0.947 | 0.021 | 0.020 |
| | 4 | 184 | 1064 | 675 | 0.949 | 0.030 | 0.028 |
| | **3** | **148** | **912** | **598** | **0.947** | **0.049** | **0.047** |
| | 2 | 96 | 616 | 427 | 0.779 | 0.131 | 0.102 |
| `[2,3,5,7,11] → 13` (n=5, b=6, M=1, r=4) | exact | 480 | 2914 | 1892 | 0.999 | 1.0e-4 | 7e-5 |
| | 4 | 408 | 2607 | 1622 | 0.989 | 2.0e-4 | 1.8e-4 |
| | **3** | **328** | **2281** | **1392** | **0.934** | **5.0e-4** | **5.0e-4** |
| | 2 | 216 | 1585 | 1100 | 0.292 | 5.3e-3 | 1.6e-3 |

**`K = 3` costs at most 6% of the ideal success probability and removes ~31% of
the controlled phases, ~22% of the transpiled two-qubit gates and ~20% of the
two-qubit depth**, for a 2.4x (n=4) to 7x (n=5) gain in the survival estimate.
`K = 2` gains more still on the FOM but breaks the algorithm's fidelity
(`P_ideal` falls to 0.29 at n=5), so it marks the edge of the useful range
rather than a recommended setting.

Caveat on the FOM: it multiplies a coherent approximation error by a stochastic
gate-error model, which is not rigorous. The 22% two-qubit reduction is
unambiguous; the survival factor is the order-of-magnitude claim of §10, not a
predicted success probability. Whether the gain survives on real hardware is
exactly what a paired exact-vs-truncated hardware run would settle. That
experiment belongs to a follow-up paper, not to this one: the algorithm is
frozen for Paper A (see the header), so no work package of this paper builds or
runs a truncated circuit.
