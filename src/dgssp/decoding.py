"""
decoding.py

Translation layer between measurement outcomes and Subset Sum objects.

The Draper--Grover circuit measures only the *index register*: one qubit per
item, where qubit ``k`` being ``1`` means "item ``k`` is in the subset".
Qiskit reports measurement outcomes as bitstrings ordered most-significant bit
first, i.e. ``q_{n-1} ... q_1 q_0``.  Every helper in this module obeys that
single convention:

    index 0  <->  least-significant bit  <->  rightmost character

Keeping all of the bit-twiddling in one place means the solver, the backend
evaluators, the mitigation code and the batch runner cannot silently disagree
about which bitstring corresponds to which subset.

The module is Qiskit-free; it only depends on :mod:`dgssp.instance`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from .instance import DPResult, SSPSolution, SubsetSumInstance

# ---------------------------------------------------------------------------
# Bitstring <-> subset
# ---------------------------------------------------------------------------


def indices_to_bitstring(indices: Sequence[int], n_bits: int) -> str:
    """
    Convert selected item indices into a Qiskit-style bitstring.

    Parameters
    ----------
    indices:
        0-based indices of the selected items.
    n_bits:
        Width of the index register (i.e. the number of items).

    Returns
    -------
    str
        A length-``n_bits`` string of ``'0'``/``'1'``, most-significant bit
        first.

    Raises
    ------
    ValueError
        If any index falls outside ``[0, n_bits)``.
    """
    bits_lsb_first = ["0"] * n_bits
    for idx in indices:
        if idx < 0 or idx >= n_bits:
            raise ValueError(f"Index {idx} is out of range for n_bits={n_bits}.")
        bits_lsb_first[idx] = "1"
    return "".join(reversed(bits_lsb_first))


def bitstring_to_indices(bitstring: str) -> list[int]:
    """
    Convert a Qiskit bitstring back into the list of selected item indices.

    Parameters
    ----------
    bitstring:
        Measurement outcome, most-significant bit first.  Spaces (which Qiskit
        inserts between classical registers) are ignored.

    Returns
    -------
    list[int]
        Sorted 0-based indices of the set bits.
    """
    clean = bitstring.replace(" ", "")
    lsb_first = clean[::-1]
    return [i for i, ch in enumerate(lsb_first) if ch == "1"]


def bitstring_to_subset(bitstring: str, items: Sequence[int]) -> list[int]:
    """
    Convert a bitstring into the corresponding list of item *values*.

    Parameters
    ----------
    bitstring:
        Measurement outcome, most-significant bit first.
    items:
        The instance items.

    Returns
    -------
    list[int]
        The values of the selected items, in index order.
    """
    return [items[i] for i in bitstring_to_indices(bitstring) if i < len(items)]


def bitstring_to_solution(
    bitstring: str, instance: SubsetSumInstance
) -> SSPSolution:
    """
    Turn a measurement outcome into a full :class:`~dgssp.instance.SSPSolution`.

    Parameters
    ----------
    bitstring:
        Measurement outcome, most-significant bit first.
    instance:
        The instance the circuit was built from.

    Returns
    -------
    SSPSolution
        Carries the indices, the item values, the total and whether the total
        equals the instance target.
    """
    return instance.subset_from_indices(bitstring_to_indices(bitstring))


def bit_is_solution(bitstring: str, items: Sequence[int], target: int) -> bool:
    """
    Test whether a measured bitstring encodes an exact solution.

    Parameters
    ----------
    bitstring:
        Measurement outcome, most-significant bit first.
    items:
        The instance items.
    target:
        The instance target sum.

    Returns
    -------
    bool
        ``True`` if the selected items sum exactly to ``target``.
    """
    return sum(bitstring_to_subset(bitstring, items)) == target


# ---------------------------------------------------------------------------
# Counts / distributions
# ---------------------------------------------------------------------------


def counts_to_probs(counts: Mapping[str, int]) -> dict[str, float]:
    """
    Normalize a counts dictionary into a probability distribution.

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


def solution_probability(
    counts: Mapping[str, int], solution_states: Iterable[str]
) -> float:
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
    Same as :func:`solution_probability` but for an already-normalized
    distribution (e.g. a ZNE-mitigated one, whose "counts" are not integers).

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


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def exact_solution_bitstrings(
    instance: SubsetSumInstance, dp_result: DPResult
) -> list[str]:
    """
    Extract the set of correct measurement outcomes from a classical DP result.

    Parameters
    ----------
    instance:
        The instance that was solved (supplies the register width).
    dp_result:
        Output of :class:`~dgssp.solvers.classical.DPSSPSolver`.

    Returns
    -------
    list[str]
        Sorted, de-duplicated bitstrings whose subsets sum exactly to the
        target.  Empty if the instance is infeasible.
    """
    exact = [s for s in dp_result.all_solutions if s.is_exact]
    if not exact and dp_result.best_solution is not None:
        if dp_result.best_solution.is_exact:
            exact = [dp_result.best_solution]
    return sorted(
        {indices_to_bitstring(sol.indices, instance.n_items) for sol in exact}
    )


__all__ = [
    "indices_to_bitstring",
    "bitstring_to_indices",
    "bitstring_to_subset",
    "bitstring_to_solution",
    "bit_is_solution",
    "counts_to_probs",
    "solution_probability",
    "distribution_solution_probability",
    "exact_solution_bitstrings",
]
