"""Tests for the Grover iteration-count helper."""

from __future__ import annotations

import math

import pytest

from dgssp import optimal_iterations


def reference(n_ind: int, m: int) -> int:
    """Textbook formula, used as the oracle for this test."""
    n = 2**n_ind
    theta = math.asin(math.sqrt(m / n))
    return max(1, math.floor(math.pi / (4 * theta)))


@pytest.mark.parametrize("n_ind", range(1, 12))
@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_matches_reference_formula(n_ind, m):
    """The helper reproduces floor(pi / (4 * asin(sqrt(M/N))))."""
    if m >= 2**n_ind:
        pytest.skip("saturated search handled separately")
    assert optimal_iterations(n_ind, m) == reference(n_ind, m)


def test_clamped_to_at_least_one():
    """Never returns zero, even for degenerate inputs."""
    assert optimal_iterations(1, 1) >= 1
    assert optimal_iterations(1, 0) >= 1
    assert optimal_iterations(2, -5) >= 1


def test_saturated_search_returns_one():
    """When M >= N there is nothing to amplify."""
    assert optimal_iterations(3, 8) == 1
    assert optimal_iterations(3, 100) == 1


def test_grows_as_solutions_become_rarer():
    """Fewer solutions in a bigger space means more iterations."""
    assert optimal_iterations(10, 1) > optimal_iterations(6, 1)
    assert optimal_iterations(10, 1) > optimal_iterations(10, 4)
