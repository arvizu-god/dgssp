"""Tests for bitstring <-> subset translation and distribution helpers."""

from __future__ import annotations

import itertools

import pytest

from dgssp import (
    DPSSPSolver,
    SubsetSumInstance,
    bit_is_solution,
    bitstring_to_indices,
    bitstring_to_solution,
    bitstring_to_subset,
    counts_to_probs,
    exact_solution_bitstrings,
    indices_to_bitstring,
    solution_probability,
)


@pytest.mark.parametrize("n_bits", [1, 2, 3, 5])
def test_roundtrip_over_all_subsets(n_bits):
    """indices -> bitstring -> indices is the identity."""
    for r in range(n_bits + 1):
        for combo in itertools.combinations(range(n_bits), r):
            bits = indices_to_bitstring(combo, n_bits)
            assert len(bits) == n_bits
            assert bitstring_to_indices(bits) == list(combo)


def test_index_zero_is_rightmost_bit():
    """The LSB-first convention is honoured."""
    assert indices_to_bitstring([0], 3) == "001"
    assert indices_to_bitstring([2], 3) == "100"
    assert bitstring_to_indices("101") == [0, 2]


def test_out_of_range_index_rejected():
    """An index beyond the register width is an error."""
    with pytest.raises(ValueError):
        indices_to_bitstring([3], 3)


def test_bitstring_to_subset_and_solution():
    """Values and totals are recovered from a bitstring."""
    instance = SubsetSumInstance(items=[3, -2, 4], target=1)
    assert bitstring_to_subset("011", [3, -2, 4]) == [3, -2]
    sol = bitstring_to_solution("011", instance)
    assert sol.total == 1 and sol.is_exact
    assert bit_is_solution("011", instance.items, 1)


def test_spaces_in_bitstrings_are_ignored():
    """Qiskit inserts spaces between classical registers."""
    assert bitstring_to_indices("10 1") == [0, 2]


def test_counts_to_probs_and_solution_probability():
    """Normalisation and solution mass are computed correctly."""
    counts = {"011": 30, "000": 70}
    probs = counts_to_probs(counts)
    assert probs["011"] == pytest.approx(0.3)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert solution_probability(counts, ["011"]) == pytest.approx(0.3)


def test_empty_counts_are_safe():
    """Zero shots produce zeros rather than a ZeroDivisionError."""
    assert counts_to_probs({}) == {}
    assert solution_probability({}, ["01"]) == 0.0


def test_exact_solution_bitstrings_matches_dp():
    """Ground-truth extraction agrees with the DP solver."""
    instance = SubsetSumInstance(items=[1, 2, 3], target=3)
    bits = exact_solution_bitstrings(instance, DPSSPSolver().solve(instance))
    assert set(bits) == {"100", "011"}
