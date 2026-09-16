"""
metrics.py

Scalar metrics on measured and mitigated distributions, the survival estimate,
and the multinomial bootstrap.

This is the single implementation of every number the paper quotes about a
distribution: ``P(solution)``, total-variation distance, classical fidelity,
Hellinger distance, and the confidence intervals on them.  Everything else in
the library -- the batch runner, the ZNE estimators, the paper scripts --
calls these functions rather than re-deriving them.

Inputs are plain mappings ``bitstring -> value``.  A mapping whose values are
all Python/NumPy integers is treated as *counts* and normalised; a mapping of
floats is taken as a distribution *as given* (a raw pointwise-ZNE output need
not sum to one, and silently renormalising it would hide exactly the defect
the pathology diagnostic measures).

The module depends on NumPy only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def counts_to_probs(counts: Mapping[str, int]) -> dict[str, float]:
    """
    Normalise a counts dictionary into a probability distribution.

    Parameters
    ----------
    counts:
        Mapping bitstring -> number of shots.

    Returns
    -------
    dict[str, float]
        Mapping bitstring -> empirical probability.  Returns all-zeros if the
        total shot count is zero.
    """
    total = sum(counts.values())
    if total == 0:
        return {b: 0.0 for b in counts}
    return {b: c / total for b, c in counts.items()}


def _is_counts(mapping: Mapping[str, Any]) -> bool:
    """True when every value is an integer (and not a bool): a counts dict."""
    return bool(mapping) and all(
        isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in mapping.values()
    )


def as_distribution(mapping: Mapping[str, float] | Mapping[str, int]) -> dict[str, float]:
    """
    View a counts dict or a distribution as a distribution.

    Parameters
    ----------
    mapping:
        Counts (all-integer values, normalised here) or probabilities (taken
        as given, *not* renormalised).

    Returns
    -------
    dict[str, float]
        Mapping bitstring -> probability.
    """
    if _is_counts(mapping):
        return counts_to_probs(mapping)  # type: ignore[arg-type]
    return {b: float(v) for b, v in mapping.items()}


def uniform_distribution(n_bits: int) -> dict[str, float]:
    """
    The uniform distribution over every ``n_bits``-bit outcome.

    This is the depolarising fixed point of the measured register, and the
    *exact* ideal output of every M = 0 instance (and of M/N = 1/2 at one
    iteration), which makes it the analytic reference of the pathology cases.

    Parameters
    ----------
    n_bits:
        Width of the measured register.

    Returns
    -------
    dict[str, float]
        ``2**n_bits`` keys, each ``2**-n_bits``.  Key formatting is Qiskit's
        (most-significant bit first); no item-to-bit translation is involved.
    """
    k = 2**n_bits
    return {format(i, f"0{n_bits}b"): 1.0 / k for i in range(k)}


# ---------------------------------------------------------------------------
# P(solution)
# ---------------------------------------------------------------------------


def solution_probability(counts: Mapping[str, int], solution_states: Iterable[str]) -> float:
    """
    Probability of sampling any of the given solution bitstrings.

    Parameters
    ----------
    counts:
        Mapping bitstring -> number of shots.
    solution_states:
        The bitstrings considered "correct".

    Returns
    -------
    float
        Fraction of shots landing on a solution bitstring; ``0.0`` if no shots
        were collected.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(counts.get(state, 0) for state in solution_states) / total


def distribution_solution_probability(
    distribution: Mapping[str, float], solution_states: Iterable[str]
) -> float:
    """
    Same as :func:`solution_probability` for an already-normalised distribution
    (e.g. a ZNE-mitigated one, whose "counts" are not integers).

    Parameters
    ----------
    distribution:
        Mapping bitstring -> probability.
    solution_states:
        The bitstrings considered "correct".

    Returns
    -------
    float
        Summed probability mass on the solution bitstrings.
    """
    return float(sum(distribution.get(state, 0.0) for state in solution_states))


def proportion_sigma(p: float, shots: int) -> float:
    """
    Binomial standard error of a sampled proportion, floored away from zero.

    ``sqrt(p~ (1 - p~) / shots)`` with the Jeffreys-smoothed
    ``p~ = (p * shots + 1/2) / (shots + 1)``.  Away from 0 and 1 this equals
    the plain ``sqrt(p (1 - p) / shots)`` to O(1/shots) -- which is also the
    *exact* standard deviation of the multinomial-bootstrap distribution of a
    proportion -- but it never returns zero, so a weighted fit never receives
    an infinite weight and a resolvability test ``|p - p0| >= 2 sigma`` is
    never passed trivially by ``0 >= 0``.

    Parameters
    ----------
    p:
        Observed proportion in ``[0, 1]``.
    shots:
        Number of shots it was measured from.

    Returns
    -------
    float
        The standard error.
    """
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}.")
    smoothed = (p * shots + 0.5) / (shots + 1.0)
    return float(np.sqrt(smoothed * (1.0 - smoothed) / shots))


# ---------------------------------------------------------------------------
# Distances between distributions
# ---------------------------------------------------------------------------


def _aligned(
    p: Mapping[str, float] | Mapping[str, int], q: Mapping[str, float] | Mapping[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Two distributions as arrays over the union of their keys."""
    pd, qd = as_distribution(p), as_distribution(q)
    keys = sorted(set(pd) | set(qd))
    return (
        np.array([pd.get(k, 0.0) for k in keys], dtype=float),
        np.array([qd.get(k, 0.0) for k in keys], dtype=float),
    )


def total_variation(
    p: Mapping[str, float] | Mapping[str, int], q: Mapping[str, float] | Mapping[str, int]
) -> float:
    """
    Total-variation distance ``D_TV = 1/2 sum_b |p_b - q_b|``.

    Parameters
    ----------
    p, q:
        Distributions or counts (counts are normalised).  Missing keys count
        as zero.

    Returns
    -------
    float
        In ``[0, 1]`` for two normalised distributions: ``0`` for identical
        ones, ``1`` for disjoint supports.
    """
    a, b = _aligned(p, q)
    return float(0.5 * np.abs(a - b).sum())


def bhattacharyya_coefficient(
    p: Mapping[str, float] | Mapping[str, int], q: Mapping[str, float] | Mapping[str, int]
) -> float:
    """
    ``BC = sum_b sqrt(p_b q_b)``; negative entries are treated as zero.

    Parameters
    ----------
    p, q:
        Distributions or counts.

    Returns
    -------
    float
        In ``[0, 1]`` for normalised distributions.
    """
    a, b = _aligned(p, q)
    return float(np.sqrt(np.clip(a, 0.0, None) * np.clip(b, 0.0, None)).sum())


def classical_fidelity(
    p: Mapping[str, float] | Mapping[str, int], q: Mapping[str, float] | Mapping[str, int]
) -> float:
    """
    Classical (Bhattacharyya) fidelity ``F = (sum_b sqrt(p_b q_b))**2``.

    Parameters
    ----------
    p, q:
        Distributions or counts.

    Returns
    -------
    float
        In ``[0, 1]``; ``1`` iff the distributions are identical.
    """
    return bhattacharyya_coefficient(p, q) ** 2


def hellinger(
    p: Mapping[str, float] | Mapping[str, int], q: Mapping[str, float] | Mapping[str, int]
) -> float:
    """
    Hellinger distance ``H = sqrt(1 - sum_b sqrt(p_b q_b))``.

    Parameters
    ----------
    p, q:
        Distributions or counts.

    Returns
    -------
    float
        In ``[0, 1]``: ``0`` for identical distributions, ``1`` for disjoint
        supports.  Floating-point excursions of ``BC`` above one are clipped.
    """
    return float(np.sqrt(max(0.0, 1.0 - bhattacharyya_coefficient(p, q))))


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------


def survival_estimate(two_q: int, one_q: int, errors: Mapping[str, Any]) -> dict[str, Any]:
    """
    Probability that no gate errs, from the device's median gate errors.

    ``S2 = (1 - eps2)**two_q`` is the primary figure -- two-qubit errors are an
    order of magnitude larger than single-qubit ones on every heavy-hex device
    in the fake provider -- and ``S1 = (1 - eps1)**one_q`` is the secondary
    factor.  This is a *coherence-free upper bound*: it ignores idling,
    crosstalk, readout and the fact that not every gate error is fatal to the
    marked amplitude.  It is used to say where the algorithm cannot work, not
    to predict a success probability.

    Parameters
    ----------
    two_q, one_q:
        *Physical* gate counts of the transpiled circuit, from
        :func:`dgssp.backends.physical_gate_counts` -- virtual ``rz`` frame
        changes excluded, exactly as they are excluded from the medians.
        Passing the raw single-qubit inventory instead charges the ``sx``
        error to every ``rz`` and understates survival.
    errors:
        Output of :func:`dgssp.backends.median_gate_errors`.

    Returns
    -------
    dict
        ``{"survival_two_q", "survival_one_q", "survival"}``; entries are
        ``None`` when the device carries no median for that gate class.
    """
    eps_two = errors.get("two_qubit")
    eps_one = errors.get("one_qubit")

    survival_two = (1.0 - eps_two) ** two_q if eps_two is not None else None
    survival_one = (1.0 - eps_one) ** one_q if eps_one is not None else None
    combined = (
        survival_two * survival_one
        if survival_two is not None and survival_one is not None
        else survival_two
    )
    return {
        "survival_two_q": survival_two,
        "survival_one_q": survival_one,
        "survival": combined,
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def resample_counts(
    counts: Mapping[str, int], rng: np.random.Generator, size: int
) -> tuple[list[str], np.ndarray]:
    """
    Multinomial resamples of a counts dictionary.

    Parameters
    ----------
    counts:
        The observed counts.
    rng:
        Random generator.
    size:
        Number of resamples.

    Returns
    -------
    tuple[list[str], numpy.ndarray]
        The outcome keys and a ``(size, len(keys))`` integer array; row ``i``
        is one resample with the same total shot count as ``counts``.
    """
    keys = list(counts)
    values = np.array([counts[k] for k in keys], dtype=float)
    total = int(values.sum())
    if total == 0:
        raise ValueError("Cannot resample an empty counts dictionary.")
    return keys, rng.multinomial(total, values / total, size=size)


def percentile_interval(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """
    Percentile bootstrap interval of the finite entries of ``samples``.

    Parameters
    ----------
    samples:
        Bootstrap replicates of a statistic.
    alpha:
        ``1 - confidence``; ``0.05`` gives the 95% interval.

    Returns
    -------
    tuple[float, float]
        ``(lo, hi)``; ``(nan, nan)`` if no replicate is finite.
    """
    finite = np.asarray(samples, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_metric(
    counts: Mapping[str, int],
    metric_fn: Callable[[Mapping[str, int]], float],
    B: int = 1000,
    seed: int | None = None,
    *,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Percentile-bootstrap confidence interval of a metric of sampled counts.

    The counts are resampled multinomially (same total shots, probabilities =
    observed frequencies), the metric is recomputed on each resample, and the
    ``alpha/2`` and ``1 - alpha/2`` percentiles are returned.

    Parameters
    ----------
    counts:
        Observed counts.
    metric_fn:
        Any function of a counts dictionary, e.g.
        ``lambda c: solution_probability(c, sols)`` or
        ``lambda c: total_variation(c, reference)``.
    B:
        Number of bootstrap resamples.
    seed:
        Seed of the resampling generator.
    alpha:
        ``1 - confidence``.

    Returns
    -------
    tuple[float, float, float]
        ``(point, lo, hi)``, where ``point`` is the metric on the observed
        counts (not the bootstrap mean).
    """
    point = float(metric_fn(counts))
    rng = np.random.default_rng(seed)
    keys, draws = resample_counts(counts, rng, B)
    replicates = np.array(
        [metric_fn(dict(zip(keys, (int(v) for v in row), strict=True))) for row in draws],
        dtype=float,
    )
    lo, hi = percentile_interval(replicates, alpha)
    return point, lo, hi


__all__ = [
    "counts_to_probs",
    "as_distribution",
    "uniform_distribution",
    "solution_probability",
    "distribution_solution_probability",
    "proportion_sigma",
    "total_variation",
    "bhattacharyya_coefficient",
    "classical_fidelity",
    "hellinger",
    "survival_estimate",
    "resample_counts",
    "percentile_interval",
    "bootstrap_metric",
]
