"""Tests for the classical dynamic-programming baseline."""

from __future__ import annotations

import itertools

import pytest

from dgssp import DPConfig, DPSSPSolver, SubsetSumInstance


def brute_force(items, target):
    """All index subsets summing exactly to target, as sorted tuples."""
    out = set()
    for r in range(len(items) + 1):
        for combo in itertools.combinations(range(len(items)), r):
            if sum(items[i] for i in combo) == target:
                out.add(tuple(sorted(combo)))
    return out


CASES = [
    ([1, 2, 3], 5),
    ([1, 2, 3], 6),
    ([1, 2, 3], 0),
    ([3, -2, 4], 1),
    ([-1, -2, -3], -3),
    ([2, 2, 2], 4),
    ([5, 5, 5], 7),
    ([0, 1, 2], 1),
]


@pytest.mark.parametrize("items,target", CASES)
def test_dp_matches_brute_force(items, target):
    """DP finds all and only the correct subsets."""
    instance = SubsetSumInstance(items=items, target=target)
    result = DPSSPSolver(DPConfig(enumerate_all=True)).solve(instance)
    found = {tuple(sorted(s.indices)) for s in result.all_solutions if s.is_exact}
    assert found == brute_force(items, target)


def test_target_zero_includes_empty_subset():
    """Target 0 is always reachable via the empty subset."""
    instance = SubsetSumInstance(items=[1, 2, 3], target=0)
    result = DPSSPSolver().solve(instance)
    assert any(s.indices == [] for s in result.all_solutions)


def test_duplicates_are_distinct_elements():
    """Equal values at different positions count as different solutions."""
    instance = SubsetSumInstance(items=[2, 2], target=2)
    result = DPSSPSolver().solve(instance)
    exact = [s for s in result.all_solutions if s.is_exact]
    assert len(exact) == 2


def test_infeasible_instance_reports_no_solution():
    """An unreachable target yields no exact solutions."""
    instance = SubsetSumInstance(items=[2, 4], target=7)
    result = DPSSPSolver().solve(instance)
    assert [s for s in result.all_solutions if s.is_exact] == []
