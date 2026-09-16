"""
Tests for ``dgssp.metrics`` and the deprecation of the decoding helpers.

(``tests/test_metrics.py`` covers the calibration error metrics of
``dgssp.backends``; this file covers the distribution metrics.)
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from dgssp import metrics
from dgssp.metrics import (
    bootstrap_metric,
    classical_fidelity,
    counts_to_probs,
    hellinger,
    proportion_sigma,
    solution_probability,
    survival_estimate,
    total_variation,
    uniform_distribution,
)

#: Fixed seed list shared with the ZNE coverage tests: 40 seeds, so a correct
#: 95% interval misses >= 6 of them with probability ~1.4%.
SEEDS = list(range(20261020, 20261060))


# ---------------------------------------------------------------------------
# Known values
# ---------------------------------------------------------------------------


def test_total_variation_known_values():
    """TV of identical distributions is 0, of disjoint ones 1."""
    p = {"00": 0.5, "11": 0.5}
    assert total_variation(p, p) == 0.0
    assert total_variation({"00": 1.0}, {"11": 1.0}) == 1.0
    assert total_variation({"0": 0.75, "1": 0.25}, {"0": 0.25, "1": 0.75}) == pytest.approx(0.5)


def test_fidelity_and_hellinger_ranges_and_extremes():
    """F in [0, 1], H in [0, 1], F = 1 / H = 0 iff identical, F = 0 / H = 1 if disjoint."""
    p = {"00": 0.5, "11": 0.5}
    assert classical_fidelity(p, p) == pytest.approx(1.0)
    assert hellinger(p, p) == pytest.approx(0.0, abs=1e-7)
    assert classical_fidelity({"0": 1.0}, {"1": 1.0}) == 0.0
    assert hellinger({"0": 1.0}, {"1": 1.0}) == 1.0

    rng = np.random.default_rng(7)
    for _ in range(200):
        a, b = rng.dirichlet(np.ones(8)), rng.dirichlet(np.ones(8))
        pa = {format(i, "03b"): v for i, v in enumerate(a)}
        pb = {format(i, "03b"): v for i, v in enumerate(b)}
        f, h = classical_fidelity(pa, pb), hellinger(pa, pb)
        assert 0.0 <= f <= 1.0 + 1e-12
        assert 0.0 <= h <= 1.0
        # H^2 = 1 - sqrt(F), and TV >= H^2 (Le Cam).
        assert h**2 == pytest.approx(1.0 - math.sqrt(f), abs=1e-12)
        assert total_variation(pa, pb) >= h**2 - 1e-12


def test_counts_are_normalised_but_distributions_are_not():
    """Integer values are counts; float values are taken as given."""
    assert total_variation({"0": 30, "1": 70}, {"0": 0.3, "1": 0.7}) == pytest.approx(0.0)
    # A raw pointwise-ZNE output that sums to 1.2 must not be renormalised.
    assert total_variation({"0": 1.2}, {"0": 1.0}) == pytest.approx(0.1)


def test_uniform_distribution():
    """2**n keys, equal mass, Qiskit key format."""
    u = uniform_distribution(3)
    assert len(u) == 8 and set(u) == {format(i, "03b") for i in range(8)}
    assert sum(u.values()) == pytest.approx(1.0)


def test_proportion_sigma_is_never_zero_and_matches_binomial_in_the_bulk():
    """The Jeffreys floor only matters at the edges."""
    assert proportion_sigma(0.0, 1000) > 0.0
    assert proportion_sigma(1.0, 1000) > 0.0
    assert proportion_sigma(0.3, 4000) == pytest.approx(math.sqrt(0.3 * 0.7 / 4000), rel=1e-3)


def test_survival_estimate_is_the_one_resource_scaling_used():
    """The moved function keeps its semantics and is re-exported by the script."""
    errors = {"two_qubit": 0.01, "one_qubit": 0.001}
    out = survival_estimate(100, 50, errors)
    assert out["survival_two_q"] == pytest.approx(0.99**100)
    assert out["survival"] == pytest.approx(0.99**100 * 0.999**50)
    assert survival_estimate(10, 10, {"two_qubit": 0.01})["survival"] == pytest.approx(0.99**10)

    scripts = Path(__file__).resolve().parents[1] / "paper" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    resource_scaling = pytest.importorskip("resource_scaling")
    assert resource_scaling.survival_estimate is survival_estimate


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_metric_coverage_on_a_binomial_proportion():
    """A 95% percentile interval on P_sol covers the truth in >= 35 of 40 seeds."""
    truth, shots = 0.37, 2000
    covered = 0
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        k = int(rng.binomial(shots, truth))
        counts = {"1": k, "0": shots - k}
        point, lo, hi = bootstrap_metric(
            counts, lambda c: solution_probability(c, ["1"]), B=1000, seed=seed
        )
        assert lo <= point <= hi
        covered += lo <= truth <= hi
    assert covered >= 35, covered


def test_bootstrap_metric_is_seeded():
    """Same seed, same interval."""
    counts = {"00": 400, "01": 300, "11": 300}
    ref = uniform_distribution(2)
    a = bootstrap_metric(counts, lambda c: total_variation(c, ref), B=200, seed=3)
    b = bootstrap_metric(counts, lambda c: total_variation(c, ref), B=200, seed=3)
    assert a == b


# ---------------------------------------------------------------------------
# Deprecated decoding imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["counts_to_probs", "solution_probability", "distribution_solution_probability"]
)
def test_deprecated_decoding_imports_warn_and_agree(name):
    """dgssp.decoding.<name> warns and *is* the dgssp.metrics function."""
    import dgssp.decoding as decoding

    with pytest.warns(DeprecationWarning, match="dgssp.metrics"):
        old = getattr(decoding, name)
    assert old is getattr(metrics, name)

    counts = {"011": 30, "000": 70}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if name == "counts_to_probs":
            assert old(counts) == counts_to_probs(counts)
        elif name == "solution_probability":
            assert old(counts, ["011"]) == solution_probability(counts, ["011"]) == 0.3
        else:
            dist = counts_to_probs(counts)
            assert old(dist, ["011"]) == pytest.approx(0.3)


def test_from_import_of_a_deprecated_name_warns():
    """``from dgssp.decoding import solution_probability`` still works, loudly."""
    with pytest.warns(DeprecationWarning):
        from dgssp.decoding import solution_probability as old  # noqa: F401


def test_top_level_names_do_not_warn():
    """``dgssp.solution_probability`` comes from metrics: no warning."""
    import dgssp

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert dgssp.solution_probability is solution_probability
        assert dgssp.counts_to_probs is counts_to_probs


def test_unknown_decoding_attribute_still_raises():
    """The module __getattr__ only knows the moved names."""
    import dgssp.decoding as decoding

    with pytest.raises(AttributeError):
        decoding.no_such_helper  # noqa: B018
