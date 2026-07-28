"""
The gate test: does the Draper-Grover circuit actually amplify the solutions?

Everything else in the library assumes this works, so it is checked on an
ideal simulator against classical ground truth, for both non-negative and
negative-item instances.
"""

from __future__ import annotations

import pytest
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from dgssp import (
    DGConfig,
    DGSSPSolver,
    DPConfig,
    DPSSPSolver,
    SubsetSumInstance,
    build_ideal_aer_backend,
    counts_to_probs,
    exact_solution_bitstrings,
    optimal_iterations,
    sample_counts,
    solution_probability,
)

SHOTS = 8192

INSTANCES = [
    ([1, 2, 3], 5),
    ([1, 2, 3], 3),
    ([3, 5, 7, 10], 15),
    ([3, -2, 4], 1),        # negative item
    ([-1, 2, 3], 2),        # negative item
    ([4, -4, 2], 0),        # negative item, target 0
    ([2, 3, -5, 6], 1),     # negative item
]


def _run_ideal(instance: SubsetSumInstance):
    """Build, run on the ideal simulator and return (solution bits, probs)."""
    dp = DPSSPSolver(DPConfig(enumerate_all=True)).solve(instance)
    sol_bits = exact_solution_bitstrings(instance, dp)
    assert sol_bits, "test instance must be feasible"

    solver = DGSSPSolver(DGConfig(iterations="auto"))
    qc = solver.build_circuit(instance, num_solutions=len(sol_bits))

    backend = build_ideal_aer_backend(seed_simulator=1234)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    counts = sample_counts(
        backend, pm.run(qc), shots=SHOTS, seed_simulator=1234, run_log=None
    )[0]
    return sol_bits, counts


@pytest.mark.parametrize("items,target", INSTANCES)
def test_solutions_dominate_the_distribution(items, target):
    """Solution bitstrings carry more probability mass than everything else."""
    instance = SubsetSumInstance(items=items, target=target, name="t")
    sol_bits, counts = _run_ideal(instance)

    p_sol = solution_probability(counts, sol_bits)
    uniform = len(sol_bits) / 2**instance.n_items

    assert p_sol > uniform, "Grover did not amplify above the uniform baseline"
    assert p_sol > 0.5, f"solution mass {p_sol:.3f} is too low"


@pytest.mark.parametrize("items,target", INSTANCES)
def test_most_likely_outcome_is_a_solution(items, target):
    """The modal measurement outcome is a correct subset."""
    instance = SubsetSumInstance(items=items, target=target, name="t")
    sol_bits, counts = _run_ideal(instance)
    assert max(counts, key=lambda b: counts[b]) in sol_bits


def test_negative_instance_beats_the_same_circuit_without_grover():
    """One Grover iteration measurably improves on plain superposition."""
    instance = SubsetSumInstance(items=[3, -2, 4], target=1)
    sol_bits, counts = _run_ideal(instance)
    probs = counts_to_probs(counts)
    baseline = len(sol_bits) / 2**instance.n_items
    assert sum(probs.get(b, 0.0) for b in sol_bits) > 2 * baseline


def test_auto_iterations_matches_helper():
    """'auto' resolves to exactly optimal_iterations(n_ind, M)."""
    instance = SubsetSumInstance(items=[1, 2, 3, 4], target=5)
    dp = DPSSPSolver(DPConfig(enumerate_all=True)).solve(instance)
    m = len(exact_solution_bitstrings(instance, dp))

    qc = DGSSPSolver(DGConfig(iterations="auto")).build_circuit(
        instance, num_solutions=m
    )
    assert qc.metadata["dgssp_instance"]["iterations"] == optimal_iterations(
        instance.n_items, m
    )


def test_explicit_iterations_are_honoured():
    """An integer iteration count is used verbatim."""
    instance = SubsetSumInstance(items=[1, 2, 3], target=5)
    qc = DGSSPSolver(DGConfig(iterations=3)).build_circuit(instance)
    assert qc.metadata["dgssp_instance"]["iterations"] == 3


def test_invalid_iterations_rejected():
    """Zero or negative iteration counts are errors."""
    instance = SubsetSumInstance(items=[1, 2, 3], target=5)
    with pytest.raises(ValueError):
        DGSSPSolver(DGConfig(iterations=0)).build_circuit(instance)


def test_unreachable_target_warns():
    """Marking a state no subset can reach is reported to the user."""
    instance = SubsetSumInstance(items=[1, 2], target=99)
    with pytest.warns(UserWarning):
        DGSSPSolver(DGConfig(iterations=1)).build_circuit(instance)


def test_no_halfqft_option_remains():
    """The HalfQFT assembly was deleted, not merely deprecated."""
    with pytest.raises(TypeError):
        DGConfig(assembly_type="HalfQFT")  # type: ignore[call-arg]
