# `dgssp.mitigation.mitiq_baseline`

## What the module does

Applies Mitiq's ZNE to the D-G algorithm as an independent comparison
baseline.

Mitiq mitigates an *expectation value*; the D-G algorithm produces a
distribution. The adaptation is the observable

```
P(solution) = Σ over solution bitstrings of their sampled probability
```

the expectation of the indicator of the solution set. With that, Mitiq's
folding and extrapolation apply unchanged.

Two parts of Mitiq are used, and nothing else:

- **Folding** — `mitiq.zne.scaling.fold_gates_at_random`, Mitiq's public random
  local folding, handed the ISA circuit as a *native Cirq circuit* built by
  `mitiq_fold_isa`: each Qiskit instruction becomes an opaque Cirq gate on
  `LineQubit(physical index)` whose inverse is the Qiskit inverse. Mitiq's own
  Qiskit conversion goes through OpenQASM 2 and Cirq's QASM importer, which
  needs `ply` (absent from the paper environment — so the old Mitiq path had
  never actually run there) and renumbers qubits, taking an ISA circuit off
  its layout. The wrapper avoids both and is lossless. Barriers are dropped,
  as Mitiq's Qiskit path drops them.
- **Extrapolation** — the static `extrapolate` of `LinearFactory`,
  `PolyFactory(order=2)`, `RichardsonFactory` and `ExpFactory(asymptote=...)`,
  applied **offline** to measured values. Mitiq never executes a circuit: its
  folded circuits are sampled inside the caller's single job
  (`zne.run_zne_batch`).

Mitiq measures its scale factor on a fidelity-weighted count of *all* gates
(default weights `1 − 0.99^n_qubits`, virtual `rz` included), so the two-qubit
ratio its folds realise differs from the requested factor. `run_zne_batch`
asks Mitiq for the dgssp estimator's *realised* scales and records Mitiq's own
realised two-qubit ratios; Mitiq's extrapolation uses the requested factors,
as Mitiq itself would.

Mitiq is optional (`pip install "dgssp[mitiq]"`) and imported lazily.

## Contents

| Name | Kind |
|---|---|
| `_require_mitiq` | internal function |
| `mitiq_is_available` | function |
| `mitiq_fold_isa` | function |
| `mitiq_extrapolate` | function |
| `mitiq_bootstrap` | function |
| `make_sampling_executor` | function |
| `mitiq_zne_solution_probability` | function |

---

### `mitiq_fold_isa(tqc, scale_factor, *, seed=0) -> QuantumCircuit`

- **Inputs:** an ISA circuit with terminal measurements; Mitiq's scale factor (≥ 1); Mitiq's folding seed.
- **Output:** the folded circuit with the input's registers and layout, not yet rebased (Mitiq's inverses need `zne.rebase_to_backend`). After rebasing it is ISA-valid for the backend (tested on fake_marrakesh).
- **Raises:** `ImportError` without Mitiq; `ValueError` for resets or delays.

### `mitiq_extrapolate(scale_factors, values, method, *, asymptote=None) -> float`

Mitiq's zero-noise value through its factories' static `extrapolate`. `nan` if
Mitiq's fit fails (e.g. `ExpFactory`'s log-linear fit meeting a value on the
wrong side of the asymptote). Not clipped.

### `mitiq_bootstrap(counts_per_scale, observable_fn, *, method, asymptote=None, B=500, seed=0, alpha=0.05) -> dict`

Mitiq's extrapolation with a percentile-bootstrap interval (multinomial
resampling at every scale). Returns `{"method", "asymptote", "scales",
"values", "value", "lo", "hi", "se", "n_failed", "B", "seed"}`.

### `make_sampling_executor(backend, solution_bitstrings, *, shots=10000, seed_simulator=None)`

The `Circuit -> float` executor Mitiq's `execute_with_zne` expects; kept for
interactive use. The library no longer lets Mitiq drive execution — one job per
folded circuit is incompatible with the one-job hardware budget.

### `mitiq_zne_solution_probability(tqc, backend, solution_bitstrings, *, shots=10000, scale_factors=(1., 3., 5.), factory=None, scale_noise=None, seed_simulator=None, fold_seed=0) -> float`

Stand-alone Mitiq baseline in **one** job: every scale folded up front
(`mitiq_fold_isa`, or a custom `scale_noise(circuit, scale)`), all folded
circuits sampled in one `sample_counts` call, then `type(factory).extrapolate`
on the measured values (the factory instance's stored options, e.g.
`ExpFactory`'s asymptote, are passed through; default `LinearFactory`).
Deliberately **not** clipped to `[0, 1]`.

### `mitiq_is_available() -> bool`

`True` if `import mitiq` succeeds.
