"""
instance.py

Core data structures and helpers for the Subset Sum Problem (SSP).

This module is intentionally independent of Qiskit so it can be reused by
classical and quantum solvers alike.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# Above this many items the index register is impractically wide; we warn but
# do not refuse to build the instance.
_LARGE_INSTANCE_WARN_THRESHOLD = 24


@dataclass
class SubsetSumInstance:
    """
    Representation of a single Subset Sum Problem (SSP) instance.

    Attributes
    ----------
    items:
        List of integer weights a_i. The problem is to select a subset
        whose sum matches (or approximates) `target`.
    target:
        The target sum t for the subset.
    name:
        Optional human-readable identifier for the instance.
    metadata:
        Arbitrary additional information (e.g. source, difficulty tag,
        generation parameters, etc.).
    """

    items: list[int]
    target: int
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure we always work with a concrete list
        self.items = list(self.items)
        validate_instance(self)

    @property
    def n_items(self) -> int:
        """Number of items in the instance."""
        return len(self.items)

    def subset_from_indices(self, indices: Sequence[int]) -> SSPSolution:
        """
        Construct an SSPSolution from a sequence of item indices.

        Parameters
        ----------
        indices:
            Indices of selected items (0-based).

        Returns
        -------
        SSPSolution
            Solution object with subset values, total sum, and exactness flag.
        """
        idx = list(indices)
        subset = [self.items[i] for i in idx]
        total = sum(subset)
        is_exact = (total == self.target)
        return SSPSolution(indices=idx, subset=subset, total=total, is_exact=is_exact)


@dataclass
class SSPSolution:
    """
    A concrete solution (or candidate solution) to an SSP instance.

    Attributes
    ----------
    indices:
        List of indices of selected items (0-based).
    subset:
        The corresponding item values a_i.
    total:
        Sum of the selected subset.
    is_exact:
        Whether `total == target` for the associated instance.
        (Note: the instance is not stored here; the solver/result object
        is responsible for keeping that association.)
    """

    indices: list[int]
    subset: list[int]
    total: int
    is_exact: bool

    def __repr__(self) -> str:  # nice compact repr for debugging
        return (
            f"SSPSolution(indices={self.indices}, subset={self.subset}, "
            f"total={self.total}, is_exact={self.is_exact})"
        )


@dataclass
class DPResult:
    """
    Result of a classical dynamic-programming (DP) SSP solver.

    This is mainly used as a classical baseline and reference for the
    quantum solvers.

    Attributes
    ----------
    instance:
        The SSP instance being solved.
    best_solution:
        One (possibly optimal) solution according to the DP algorithm.
        May be None if no feasible subset was found under the chosen
        DP formulation.
    all_solutions:
        Optional list of all solutions found (can be left empty if the
        DP solver only returns a single best solution).
    optimal_value:
        The best (e.g. closest or exact) achievable total. Often equal
        to `instance.target` when an exact solution exists.
    success:
        Whether the algorithm completed successfully and produced a
        meaningful result.
    metadata:
        Additional information such as runtime, memory usage,
        internal DP table size, etc.
    """

    instance: SubsetSumInstance
    best_solution: SSPSolution | None
    all_solutions: list[SSPSolution] = field(default_factory=list)
    optimal_value: int | None = None
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def validate_instance(instance: SubsetSumInstance) -> None:
    """
    Basic sanity checks for a SubsetSumInstance.

    Raises
    ------
    ValueError
        If the instance is malformed (empty items, non-integers, etc.).
    """
    if not instance.items:
        raise ValueError("SubsetSumInstance must contain at least one item.")

    # Ensure all items are integers
    if any(not isinstance(x, int) for x in instance.items):
        raise ValueError("All items in SubsetSumInstance must be integers.")

    if not isinstance(instance.target, int):
        raise ValueError("Target in SubsetSumInstance must be an integer.")

    # Register-size sanity warning: the index register grows linearly with
    # n_items and the sum register logarithmically with the value range, so
    # very large instances become unsimulable rather than invalid.
    if instance.n_items > _LARGE_INSTANCE_WARN_THRESHOLD:
        warnings.warn(
            f"SubsetSumInstance has {instance.n_items} items; the Draper-Grover "
            "circuit needs one index qubit per item, so this instance may be "
            "impractical to simulate or run.",
            UserWarning,
            stacklevel=2,
        )


def random_instance(
    n_items: int,
    max_value: int = 20,
    *,
    target_strategy: str = "feasible",
    density: float = 0.5,
    seed: int | None = None,
    name: str | None = None,
) -> SubsetSumInstance:
    """
    Generate a random SubsetSumInstance for testing and benchmarking.

    Parameters
    ----------
    n_items:
        Number of items to generate (must be >= 1).
    max_value:
        Maximum absolute value of any item (values are in [1, max_value]).
    target_strategy:
        Strategy to choose the target:
        - "feasible": pick a random subset and set the target to its sum
                      (guarantees at least one exact solution).
        - "random": pick a random integer in [1, n_items * max_value].
    density:
        For the "feasible" strategy, expected fraction of items included
        in the hidden subset (between 0 and 1). Used as Bernoulli p.
    seed:
        Optional PRNG seed for reproducibility.
    name:
        Optional name for the generated instance.

    Returns
    -------
    SubsetSumInstance
        A randomly generated instance.

    Raises
    ------
    ValueError
        If parameters are inconsistent.
    """
    if n_items < 1:
        raise ValueError("n_items must be >= 1.")
    if max_value < 1:
        raise ValueError("max_value must be >= 1.")
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be in [0, 1].")

    rng = random.Random(seed)

    items = [rng.randint(1, max_value) for _ in range(n_items)]

    if target_strategy == "feasible":
        # Sample a subset using a Bernoulli(p = density) model
        selected_indices = [
            i for i in range(n_items) if rng.random() < density
        ]

        # In the rare case we selected nothing, force-select at least one item
        if not selected_indices:
            selected_indices = [rng.randrange(n_items)]

        target = sum(items[i] for i in selected_indices)

    elif target_strategy == "random":
        target = rng.randint(1, n_items * max_value)
    else:
        raise ValueError(
            f"Unknown target_strategy '{target_strategy}'. "
            "Supported: 'feasible', 'random'."
        )

    instance = SubsetSumInstance(
        items=items,
        target=target,
        name=name or f"ssp_random_n{n_items}",
        metadata={
            "generator": "dgssp.instance.random_instance",
            "seed": seed,
            "max_value": max_value,
            "target_strategy": target_strategy,
            "density": density,
        },
    )
    return instance


__all__ = [
    "SubsetSumInstance",
    "SSPSolution",
    "DPResult",
    "validate_instance",
    "random_instance",
]
