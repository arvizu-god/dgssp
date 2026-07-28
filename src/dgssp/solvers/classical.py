"""
classical.py

Classical dynamic-programming baseline solver for the Subset Sum Problem (SSP).

This module provides a DP-based solver which enumerates all subsets of the
instance items whose sum equals the target. It is mainly intended as a
reference / baseline (and source of ground truth) for the quantum
Draper-Grover solver.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..instance import DPResult, SSPSolution, SubsetSumInstance
from .base import BaseClassicalSSPSolver  # to be defined in solvers/base.py

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DPConfig:
    """
    Configuration options for the classical DP SSP solver.

    Attributes
    ----------
    enumerate_all:
        If True, enumerate all subsets that sum exactly to the target.
        (For most research / benchmarking use cases this is fine; if
        instances get very large, this can of course explode.)
    """

    enumerate_all: bool = True


# ---------------------------------------------------------------------------
# Solver implementation
# ---------------------------------------------------------------------------


class DPSSPSolver(BaseClassicalSSPSolver):
    """
    Classical dynamic-programming Subset Sum solver.

    Works on a SubsetSumInstance, tracks item *indices* so solutions map back
    to SSPSolution objects, and returns a DPResult with best_solution,
    all_solutions and metadata.

    The core idea:

      - Maintain a dict: sum_value -> list of subsets (each subset is a
        list of item indices) that achieve that sum.
      - Start with {0: [[]]} (sum 0 via the empty subset).
      - For each element a in A at index i, extend all existing subsets
        by adding index i, which creates new sums.
      - Works with negative numbers and duplicates: each position in A
        is treated as a distinct element.
    """

    def __init__(self, config: DPConfig | None = None) -> None:
        self.config = config or DPConfig()

    # Public API -------------------------------------------------------

    def solve(self, instance: SubsetSumInstance) -> DPResult:
        """
        Solve the given SubsetSumInstance using dynamic programming.

        Parameters
        ----------
        instance:
            SSP instance to solve.

        Returns
        -------
        DPResult
            Result object containing best_solution, all_solutions, etc.
        """
        index_solutions = self._dp_all_index_subsets(
            instance.items,
            instance.target,
        )

        # Convert index-based subsets into SSPSolution objects using the instance
        solutions: list[SSPSolution] = [
            instance.subset_from_indices(idxs) for idxs in index_solutions
        ]

        best_solution = solutions[0] if solutions else None
        optimal_value = best_solution.total if best_solution is not None else None

        metadata = {
            "algorithm": "classical_dp",
            "num_items": instance.n_items,
            "num_solutions": len(solutions),
        }

        return DPResult(
            instance=instance,
            best_solution=best_solution,
            all_solutions=solutions,
            optimal_value=optimal_value,
            success=True,
            metadata=metadata,
        )

    # Internal DP core -------------------------------------------------

    def _dp_all_index_subsets(
        self,
        items: list[int],
        target: int,
    ) -> list[list[int]]:
        """
        Dynamic-programming subset-sum over *indices*.

        Returns
        -------
        List[List[int]]
            A list of solutions, where each solution is a list of indices
            into `items` such that the corresponding values sum to `target`.

        Approach (DP over sums):

          - Maintain a dict: sum_value -> list of subsets (each subset is
            a list of indices) that achieve that sum.
          - Start with {0: [[]]} (sum 0 via the empty subset).
          - For each element a in items, at position i, extend all
            existing subsets by adding index i, which creates new sums.
          - Works with negative numbers and duplicates (each element is
            used at most once, since we process indices left to right).

        Complexity:

          - Time and space can be exponential in the number of solutions
            (unavoidable if you want to list all of them), but the method
            is efficient and clean for moderate instances.

        Edge cases:

          - If target == 0, the empty subset [] is included among the
            results (if reachable).
          - If items contains zeros or repeated values, distinct uses of
            identical values are considered distinct elements (as usual
            in subset-sum over a list).
        """
        # base case: sum 0 via the empty subset of indices
        sums_to_subsets: dict[int, list[list[int]]] = {0: [[]]}

        for idx, a in enumerate(items):
            additions: dict[int, list[list[int]]] = {}

            # For each currently reachable sum, adding 'a' creates new subsets
            for s, subsets in sums_to_subsets.items():
                ns = s + a
                new_subs = [sub + [idx] for sub in subsets]
                if ns in additions:
                    additions[ns].extend(new_subs)
                else:
                    additions[ns] = new_subs

            # Merge: keep old sums (not using 'a') and add new sums (using 'a')
            for ns, new_subs in additions.items():
                if ns in sums_to_subsets:
                    # If the sum already existed, append the new subsets
                    sums_to_subsets[ns] = sums_to_subsets[ns] + new_subs
                else:
                    sums_to_subsets[ns] = new_subs

        # Return all subsets (index lists) that sum to target
        return sums_to_subsets.get(target, [])


__all__ = [
    "DPConfig",
    "DPSSPSolver",
]
