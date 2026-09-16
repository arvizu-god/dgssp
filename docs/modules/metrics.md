# `dgssp.metrics`

## What the module does

The single implementation of every number the paper quotes about a
distribution: `P(solution)`, total-variation distance, classical fidelity,
Hellinger distance, the survival estimate, and the bootstrap confidence
intervals on all of them. The batch runner, the ZNE estimators and the paper
scripts call these functions; none re-derives them. NumPy only.

**Counts vs distributions.** Every distance accepts either. A mapping whose
values are all integers is treated as *counts* and normalised; a mapping of
floats is taken *as given* — a raw pointwise-ZNE output need not sum to one,
and silently renormalising it would hide exactly the defect the pathology
diagnostic measures.

`counts_to_probs`, `solution_probability` and
`distribution_solution_probability` moved here from `dgssp.decoding` in WP4
(the old imports still work and emit a `DeprecationWarning`). `survival_estimate`
moved here from `paper/scripts/resource_scaling.py`, which re-exports it.

## Contents

| Name | Kind |
|---|---|
| `counts_to_probs` | function |
| `as_distribution` | function |
| `uniform_distribution` | function |
| `solution_probability` | function |
| `distribution_solution_probability` | function |
| `proportion_sigma` | function |
| `total_variation` | function |
| `bhattacharyya_coefficient` | function |
| `classical_fidelity` | function |
| `hellinger` | function |
| `survival_estimate` | function |
| `resample_counts` | function |
| `percentile_interval` | function |
| `bootstrap_metric` | function |

---

### `counts_to_probs(counts) -> dict[str, float]`

Bitstring → empirical probability; all zeros when no shots were collected.

### `as_distribution(mapping) -> dict[str, float]`

Counts (all-integer values) are normalised; floats are returned as given.

### `uniform_distribution(n_bits) -> dict[str, float]`

`2**n_bits` keys, each `2**-n_bits`. The depolarising fixed point of the
measured register and the **exact** ideal output of every M = 0 instance and of
M/N = 1/2 at one iteration — the analytic reference of the pathology cases.

### `solution_probability(counts, solution_states) -> float`

Fraction of shots on a solution bitstring; `0.0` with no shots. The headline
success metric.

### `distribution_solution_probability(distribution, solution_states) -> float`

The same for an already-normalised (e.g. mitigated) distribution.

### `proportion_sigma(p, shots) -> float`

Binomial standard error `sqrt(p~(1-p~)/shots)` with the Jeffreys-smoothed
`p~ = (p·shots + 1/2)/(shots + 1)`. In the bulk it equals the plain binomial
SE — which is also the exact standard deviation of the multinomial bootstrap of
a proportion — to O(1/shots); at 0 and 1 it stays positive, so a weighted fit
never receives an infinite weight and a resolvability test never passes by
`0 >= 0`.

### `total_variation(p, q) -> float`

`½ Σ |p_b − q_b|` over the union of keys: 0 for identical, 1 for disjoint
supports.

### `bhattacharyya_coefficient(p, q)`, `classical_fidelity(p, q)`, `hellinger(p, q)`

`BC = Σ sqrt(p_b q_b)` (negative entries treated as zero);
`F = BC²` in [0, 1]; `H = sqrt(1 − BC)` in [0, 1]. The relations
`H² = 1 − sqrt(F)` and `D_TV ≥ H²` are tested.

### `survival_estimate(two_q, one_q, errors) -> dict`

`S = (1 − ε₂)^#2Q · (1 − ε₁)^#1Q_phys` from the median errors of
`backends.median_gate_errors`, with the counts from
`backends.physical_gate_counts` (virtual `rz` excluded, exactly as it is
excluded from the median). Returns `{"survival_two_q", "survival_one_q",
"survival"}`. A coherence-free **upper bound**.

### `resample_counts(counts, rng, size) -> (keys, draws)`

`size` multinomial resamples with the observed frequencies and total.

### `percentile_interval(samples, alpha=0.05) -> (lo, hi)`

Percentile interval of the finite replicates.

### `bootstrap_metric(counts, metric_fn, B=1000, seed=None, *, alpha=0.05) -> (point, lo, hi)`

Resamples `counts` multinomially `B` times, recomputes `metric_fn`, returns the
metric on the observed counts and the 95% percentile interval. Coverage on a
binomial `P_sol` is tested (≥ 35/40 seeds).
