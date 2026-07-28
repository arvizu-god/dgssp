# `dgssp.mitigation.mitiq_baseline`

## What the module does

Applies Mitiq's ZNE to the D-G algorithm as an independent comparison
baseline.

Mitiq is built around an executor with signature `Circuit -> float`: it
mitigates an *expectation value*. The D-G algorithm does not produce one — it
produces a distribution over bitstrings. The adaptation is to define the
"observable" as a scalar functional of that distribution. The natural choice is

```
P(solution) = Σ over solution bitstrings of their sampled probability
```

which is exactly the expectation value of the indicator function
`F(b) = 1 if b is a solution else 0`. With that in hand, Mitiq's folding and
extrapolation machinery applies unchanged.

Why keep this alongside `dgssp.mitigation.zne`? The two answer different
questions, and comparing them is the point. The library's own ZNE extrapolates
*every bitstring probability*, yielding a full mitigated distribution — more
informative, but each bitstring is fitted from noisy, low-count data. This
module extrapolates *one aggregate scalar* with a third-party,
independently-validated implementation — less informative, but statistically far
better conditioned.

Mitiq is optional (`pip install "dgssp[mitiq]"`) and imported lazily.

## Contents

| Name | Kind |
|---|---|
| `_require_mitiq` | internal function |
| `make_sampling_executor` | function |
| `mitiq_zne_solution_probability` | function |
| `mitiq_is_available` | function |

---

### `_require_mitiq() -> module` — internal

- **Input:** none.
- **Output:** the imported `mitiq` module.
- **Raises:** `ImportError` with the exact install command in the message.

---

### `make_sampling_executor(backend, solution_bitstrings, *, shots=10000, seed_simulator=None) -> Callable[[QuantumCircuit], float]`

Builds the `Circuit -> float` executor Mitiq expects. The returned callable
executes whatever circuit Mitiq hands it — already folded — with no
transpilation or re-routing, and returns the sampled solution probability.
Transpile *before* calling Mitiq so the folds scale physical noise.

- **Inputs:** `backend` (noisy); `solution_bitstrings: Iterable[str]` (from the classical DP ground truth); `shots: int`; `seed_simulator: int | None` (applied only on simulators).
- **Output:** a callable taking a circuit and returning `P(solution)` in `[0, 1]`; `0.0` if no shots came back.

The executor runs `rebase_to_backend` on each circuit Mitiq hands it, because
Mitiq's folding inserts inverse gates that are not necessarily native — on IBM
devices `sx` inverts to `sxdg`, which will not execute.

---

### `mitiq_zne_solution_probability(tqc, backend, solution_bitstrings, *, shots=10000, scale_factors=(1., 3., 5.), factory=None, scale_noise=None, seed_simulator=None) -> float`

Mitigates `P(solution)` with Mitiq's ZNE.

- **Inputs:** `tqc` — an **already transpiled** (ISA-level) D-G circuit; passing a logical circuit would make Mitiq fold logical gates, which is not the noise you want to scale. Then `backend`; `solution_bitstrings`; `shots`; `scale_factors`; `factory` (a Mitiq inference factory, defaulting to `LinearFactory`); `scale_noise` (defaulting to `fold_gates_at_random`); `seed_simulator`.
- **Output:** the extrapolated probability of sampling a solution, as a `float`. Deliberately **not** clipped to `[0, 1]`: a value outside that interval is diagnostic information about the quality of the fit and is preserved.
- **Raises:** `ImportError` if Mitiq is not installed.

---

### `mitiq_is_available() -> bool`

- **Input:** none.
- **Output:** `True` if `import mitiq` succeeds. Lets calling code and tests skip the baseline cleanly.
