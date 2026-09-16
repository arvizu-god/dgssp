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

``counts_to_probs``, ``solution_probability`` and
``distribution_solution_probability`` moved to :mod:`dgssp.metrics`; importing
them from here still works and emits a ``DeprecationWarning``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

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
# Counts / distributions: moved to dgssp.metrics
# ---------------------------------------------------------------------------

#: Distribution helpers that used to live here.  They are metrics, not bit
#: translation, and now have exactly one implementation in
#: :mod:`dgssp.metrics`.  Importing them from this module still works but
#: emits a ``DeprecationWarning``.
_MOVED_TO_METRICS: frozenset[str] = frozenset(
    {"counts_to_probs", "solution_probability", "distribution_solution_probability"}
)


def __getattr__(name: str) -> Any:
    """
    Resolve the deprecated names in :data:`_MOVED_TO_METRICS`.

    Both ``from dgssp.decoding import solution_probability`` and
    ``dgssp.decoding.solution_probability`` land here, warn, and return the
    :mod:`dgssp.metrics` function itself, so the values are identical by
    construction.
    """
    if name in _MOVED_TO_METRICS:
        warnings.warn(
            f"dgssp.decoding.{name} is deprecated; import it from dgssp.metrics.",
            DeprecationWarning,
            stacklevel=2,
        )
        from . import metrics

        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "exact_solution_bitstrings",
]
