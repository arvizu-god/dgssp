"""
instance.py

Core data structures and helpers for the Subset Sum Problem (SSP).

This module is intentionally independent of Qiskit so it can be reused by
classical and quantum solvers alike.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable, Sequence
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



def max_item_value(n_items: int, *, max_n_sum: int = 5) -> int:
    """
    Largest item magnitude that keeps the sum register at or below ``max_n_sum``.

    The sum-register width is fixed by the *reachable interval*, not by the
    items individually.  With every item non-zero,
    ``range_len = sum(abs(a_i)) + 1`` (see :func:`dgssp.encoding.build_encoding`
    -- the interval spans ``[sum_neg, sum_pos]``, and flipping an item's sign
    moves both endpoints by the same amount while leaving the length alone),
    and the encoding charges

        n_sum = ceil(log2(range_len)) + 1        # the +1 is the guard bit

    so ``n_sum <= max_n_sum`` holds exactly when
    ``sum(abs(a_i)) <= 2**(max_n_sum - 1) - 1``.  Drawing every magnitude from
    ``[1, floor((2**(max_n_sum - 1) - 1) / n_items)]`` makes that bound hold for
    *every* draw, so the register width is guaranteed rather than filtered for.
    Guaranteeing it matters for the paper: rejecting over-wide draws would
    condition the item distribution on the sum and quietly stop the items being
    i.i.d. uniform.

    The lower end needs no work.  ``n_items >= 2`` forces
    ``sum(abs(a_i)) >= 2``, hence ``range_len >= 3`` and ``n_sum >= 3``.

    Parameters
    ----------
    n_items:
        Number of items.
    max_n_sum:
        Largest acceptable sum-register width, guard bit included.

    Returns
    -------
    int
        The inclusive upper bound ``a_max(n)`` on item magnitudes.

    Raises
    ------
    ValueError
        If ``n_items`` is below 1, if ``max_n_sum`` is below 2, or if no
        positive magnitude fits the budget (``n_items > 2**(max_n_sum-1) - 1``).

    Examples
    --------
    >>> [max_item_value(n) for n in range(2, 7)]
    [7, 5, 3, 3, 2]
    """
    if n_items < 1:
        raise ValueError("n_items must be >= 1.")
    if max_n_sum < 2:
        raise ValueError("max_n_sum must be >= 2.")

    budget = 2 ** (max_n_sum - 1) - 1
    a_max = budget // n_items
    if a_max < 1:
        raise ValueError(
            f"No item magnitude fits: {n_items} items of magnitude >= 1 already "
            f"exceed the sum budget {budget} implied by max_n_sum={max_n_sum}."
        )
    return a_max


def _draw_multiplicity_target(
    items: list[int],
    draw_items: Callable[[], list[int]],
    prng: random.Random,
    *,
    target_multiplicity: int | str | None,
    multiplicity_cap: int,
    allow_zero_target: bool,
    max_attempts: int,
) -> tuple[list[int], int]:
    """
    Redraw item sets until one admits a target in the requested M-class.

    A target is eligible when it lies inside the reachable interval
    ``[sum_neg, sum_pos]`` -- so an M=0 target is an unreachable *gap*, not a
    value the encoding would reject out of hand -- and its multiplicity falls
    in the requested class.  Among the eligible targets one is drawn uniformly,
    from the sorted candidate list, so the choice depends only on the PRNG
    state and not on dict iteration order.

    Parameters
    ----------
    items:
        The first item draw, already made by the caller.
    draw_items:
        Callable producing a further item draw, used on redraws.
    prng:
        The stream to draw from.
    target_multiplicity:
        An exact count (``int``), or a class string ``"0"``, ``"1"``, ``"2"``,
        ``"3+"``.  ``"k+"`` means "at least k".
    multiplicity_cap:
        Cap used to parse a ``"k+"`` string when it carries no explicit k.
    allow_zero_target:
        Whether target 0 (the empty subset) may be chosen.
    max_attempts:
        Item draws allowed before giving up.

    Returns
    -------
    tuple[list[int], int]
        The item set that worked and the chosen target.

    Raises
    ------
    ValueError
        If ``target_multiplicity`` is None or unparseable, or if no draw within
        ``max_attempts`` admitted a target of the requested class.
    """
    from .solvers.classical import subset_sum_multiplicities

    if target_multiplicity is None:
        raise ValueError(
            "target_strategy='multiplicity' requires target_multiplicity=..."
        )

    text = str(target_multiplicity)
    at_least = text.endswith("+")
    try:
        wanted = int(text[:-1] or multiplicity_cap) if at_least else int(text)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse target_multiplicity={target_multiplicity!r}; expected "
            "an int or a class string such as '0', '1', '2', '3+'."
        ) from exc
    if wanted < 0:
        raise ValueError("target_multiplicity must be >= 0.")
    if at_least and wanted == 0:
        raise ValueError("target_multiplicity='0+' matches every target.")

    current = items
    for _ in range(max(1, max_attempts)):
        spectrum = subset_sum_multiplicities(current)
        low = sum(a for a in current if a < 0)
        high = sum(a for a in current if a > 0)

        candidates = [
            t
            for t in range(low, high + 1)
            if (allow_zero_target or t != 0)
            and (
                spectrum.get(t, 0) >= wanted if at_least else spectrum.get(t, 0) == wanted
            )
        ]
        if candidates:
            return current, candidates[prng.randrange(len(candidates))]
        current = draw_items()

    raise ValueError(
        f"No item set with {len(items)} items admitted a target of class "
        f"{target_multiplicity!r} within {max_attempts} draws."
    )


def random_instance(
    n_items: int,
    max_value: int = 20,
    *,
    target_strategy: str = "feasible",
    density: float = 0.5,
    seed: int | None = None,
    name: str | None = None,
    n_negative: int = 0,
    target_multiplicity: int | str | None = None,
    multiplicity_cap: int = 3,
    allow_zero_target: bool = False,
    max_attempts: int = 500,
    rng: random.Random | None = None,
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
        - "multiplicity": pick a target uniformly among those whose number of
                      exact solutions falls in the class named by
                      `target_multiplicity`, redrawing the item set when the
                      class is unrealisable for it.
    density:
        For the "feasible" strategy, expected fraction of items included
        in the hidden subset (between 0 and 1). Used as Bernoulli p.
    seed:
        Optional PRNG seed for reproducibility. Ignored when `rng` is given.
    name:
        Optional name for the generated instance.
    n_negative:
        Number of items whose sign is flipped after magnitudes are drawn; the
        positions are sampled uniformly without replacement. Magnitudes always
        come from [1, max_value], so the reachable interval -- and hence the
        sum-register width -- is unaffected by this argument.
    target_multiplicity:
        Required M-class for the "multiplicity" strategy: an exact count, or a
        class string as produced by
        :func:`dgssp.solvers.classical.multiplicity_class` ("0", "1", "2",
        "3+"). A "k+" class means "at least k".
    multiplicity_cap:
        Cap used when a "+" class string carries no explicit count.
    allow_zero_target:
        Whether target 0 -- always reachable via the empty subset -- may be
        chosen by the "multiplicity" strategy. Off by default: the empty subset
        is a degenerate solution that makes the M=1 class artificially easy and
        needs no search to find.
    max_attempts:
        Item draws the "multiplicity" strategy may make before giving up. Some
        cells are genuinely unpopulatable -- no set of two positive items
        admits a target with three or more solutions -- so exhaustion raises
        rather than loops.
    rng:
        An explicit :class:`random.Random` to draw from, so a caller building a
        whole family can drive every draw from one master stream. Takes
        precedence over `seed`.

    Returns
    -------
    SubsetSumInstance
        A randomly generated instance.

    Raises
    ------
    ValueError
        If parameters are inconsistent, or if the "multiplicity" strategy
        exhausts `max_attempts` without realising the requested class.
    """
    if n_items < 1:
        raise ValueError("n_items must be >= 1.")
    if max_value < 1:
        raise ValueError("max_value must be >= 1.")
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be in [0, 1].")
    if not (0 <= n_negative <= n_items):
        raise ValueError(f"n_negative must be in [0, {n_items}], got {n_negative}.")

    prng = rng if rng is not None else random.Random(seed)

    def draw_items() -> list[int]:
        """Draw magnitudes in [1, max_value], then negate `n_negative` of them."""
        drawn = [prng.randint(1, max_value) for _ in range(n_items)]
        for idx in prng.sample(range(n_items), n_negative):
            drawn[idx] = -drawn[idx]
        return drawn

    items = draw_items()

    if target_strategy == "multiplicity":
        items, target = _draw_multiplicity_target(
            items,
            draw_items,
            prng,
            target_multiplicity=target_multiplicity,
            multiplicity_cap=multiplicity_cap,
            allow_zero_target=allow_zero_target,
            max_attempts=max_attempts,
        )

    elif target_strategy == "feasible":
        # Sample a subset using a Bernoulli(p = density) model
        selected_indices = [
            i for i in range(n_items) if prng.random() < density
        ]

        # In the rare case we selected nothing, force-select at least one item
        if not selected_indices:
            selected_indices = [prng.randrange(n_items)]

        target = sum(items[i] for i in selected_indices)

    elif target_strategy == "random":
        target = prng.randint(1, n_items * max_value)
    else:
        raise ValueError(
            f"Unknown target_strategy '{target_strategy}'. "
            "Supported: 'feasible', 'random', 'multiplicity'."
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
            "n_negative": n_negative,
            "target_multiplicity": target_multiplicity,
        },
    )
    return instance


__all__ = [
    "SubsetSumInstance",
    "SSPSolution",
    "DPResult",
    "validate_instance",
    "max_item_value",
    "random_instance",
]
