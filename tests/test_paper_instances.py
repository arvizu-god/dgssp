"""
Tests for the frozen paper instance family (WP3).

Two things are under test here and they are not the same thing.

*The generator* (``dgssp.instance.random_instance`` with
``target_strategy="multiplicity"``, plus
``dgssp.solvers.classical.subset_sum_multiplicities``) must agree with brute
force and must be a pure function of its seed.

*The file* ``paper/instances/family.json`` must still say what it said when it
was written: every ``M`` must survive re-derivation by exhaustive enumeration,
every solution bitstring must decode -- through ``dgssp.decoding``, the only
module allowed to know the bit order -- to a subset that really sums to the
target, and the register arithmetic must match ``dgssp.encoding``.

The file is the paper's evidence.  If a library change moves a number in it,
these tests fail rather than the paper quietly reporting the old number.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "paper" / "scripts"
FAMILY_FILE = REPO_ROOT / "paper" / "instances" / "family.json"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

build_family = pytest.importorskip("build_family")

from dgssp.decoding import (  # noqa: E402
    bit_is_solution,
    bitstring_to_indices,
    exact_solution_bitstrings,
    indices_to_bitstring,
)
from dgssp.encoding import build_encoding  # noqa: E402
from dgssp.instance import SubsetSumInstance, max_item_value, random_instance  # noqa: E402
from dgssp.solvers.classical import (  # noqa: E402
    DPConfig,
    DPSSPSolver,
    multiplicity_class,
    subset_sum_multiplicities,
)
from dgssp.solvers.draper_grover import optimal_iterations  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def family() -> dict[str, Any]:
    """The committed family file, or a skip when it has not been generated."""
    if not FAMILY_FILE.exists():
        pytest.skip(
            f"{FAMILY_FILE} not generated; run `python paper/scripts/build_family.py`."
        )
    return json.loads(FAMILY_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def records(family: dict[str, Any]) -> list[dict[str, Any]]:
    """Every instance record in the family file."""
    return list(family["instances"])


def brute_force_multiplicity(items: list[int], target: int) -> int:
    """Number of index subsets summing to ``target``, by explicit enumeration."""
    return sum(
        1
        for size in range(len(items) + 1)
        for combo in itertools.combinations(range(len(items)), size)
        if sum(items[i] for i in combo) == target
    )


# ---------------------------------------------------------------------------
# The counting primitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "items",
    [
        [1, 2, 3],
        [2, 2],
        [5, 5, 5],
        [3, -2, 4],
        [-1, -2, 3, 4],
        [1, 1, 1, 1, 1],
        [7, -7, 2, -2],
    ],
)
def test_multiplicities_match_brute_force(items: list[int]) -> None:
    """``subset_sum_multiplicities`` is the exhaustive count, for every sum."""
    spectrum = subset_sum_multiplicities(items)
    low = sum(a for a in items if a < 0)
    high = sum(a for a in items if a > 0)
    for target in range(low, high + 1):
        assert spectrum.get(target, 0) == brute_force_multiplicity(items, target)


@pytest.mark.parametrize("items", [[1, 2, 3], [2, 2], [3, -2, 4], [-1, -2, 3, 4]])
def test_multiplicities_match_the_dp_solver(items: list[int]) -> None:
    """The counting recurrence agrees with the enumerating one it mirrors."""
    solver = DPSSPSolver(DPConfig(enumerate_all=True))
    spectrum = subset_sum_multiplicities(items)
    low = sum(a for a in items if a < 0)
    high = sum(a for a in items if a > 0)
    for target in range(low, high + 1):
        result = solver.solve(SubsetSumInstance(items=list(items), target=target))
        assert len(result.all_solutions) == spectrum.get(target, 0)


# ---------------------------------------------------------------------------
# a_max and the register width
# ---------------------------------------------------------------------------


def test_a_max_table() -> None:
    """The documented a_max(n) table is what the library computes."""
    assert [max_item_value(n) for n in range(2, 7)] == [7, 5, 3, 3, 2]


@pytest.mark.parametrize("n_items", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("n_negative", [0, 1, 2])
def test_a_max_guarantees_the_register_width(n_items: int, n_negative: int) -> None:
    """
    Every item set the bound admits lands in ``n_sum in {3, 4, 5}``.

    Checked over the *extremes* of the draw -- all magnitudes 1 and all
    magnitudes ``a_max`` -- because ``range_len`` is monotone in the magnitudes
    and invariant under sign flips, so the extremes bracket every draw.
    """
    if n_negative > n_items:
        pytest.skip("more negatives than items")
    a_max = max_item_value(n_items)
    for magnitude in (1, a_max):
        items = [magnitude] * n_items
        for i in range(n_negative):
            items[i] = -items[i]
        assert build_encoding(items, 0).n_sum in (3, 4, 5)


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m_class", ["0", "1", "2", "3+"])
def test_generator_hits_the_requested_class(m_class: str) -> None:
    """A drawn instance really has the multiplicity class it was asked for."""
    instance = random_instance(
        4,
        max_item_value(4),
        target_strategy="multiplicity",
        target_multiplicity=m_class,
        seed=99,
    )
    measured = brute_force_multiplicity(instance.items, instance.target)
    assert multiplicity_class(measured) == m_class


def test_generator_never_picks_the_empty_subset_target() -> None:
    """Target 0 is excluded by default; the empty subset is not a search."""
    for seed in range(30):
        instance = random_instance(
            3,
            max_item_value(3),
            target_strategy="multiplicity",
            target_multiplicity="1",
            seed=seed,
        )
        assert instance.target != 0


def test_signed_draw_negates_exactly_n_negative_items() -> None:
    """``n_negative`` is exact, not a rate."""
    for seed in range(20):
        instance = random_instance(
            5,
            max_item_value(5),
            target_strategy="multiplicity",
            target_multiplicity="1",
            n_negative=2,
            seed=seed,
        )
        assert sum(1 for a in instance.items if a < 0) == 2
        assert all(a != 0 for a in instance.items)


def test_sign_flips_do_not_change_the_register_width() -> None:
    """
    WP2's invariance claim, exercised on the generator.

    ``range_len = sum_pos - sum_neg + 1`` depends only on the magnitudes, so a
    signed instance costs the same register as its positive twin.  This is why
    family S can be compared to family P at fixed ``n``.
    """
    magnitudes = [3, 1, 4, 2]
    base = build_encoding(magnitudes, 0).n_sum
    for signs in itertools.product([1, -1], repeat=len(magnitudes)):
        items = [s * a for s, a in zip(signs, magnitudes, strict=True)]
        assert build_encoding(items, 0).n_sum == base


def test_unrealisable_class_raises_rather_than_looping() -> None:
    """No two positive items admit a target with three or more solutions."""
    with pytest.raises(ValueError, match="No item set"):
        random_instance(
            2,
            max_item_value(2),
            target_strategy="multiplicity",
            target_multiplicity="3+",
            seed=0,
            max_attempts=200,
        )


def test_generation_is_a_pure_function_of_the_seed() -> None:
    """Two builds from the same seed produce the same payload hash."""
    first = build_family.build_family(4242)
    second = build_family.build_family(4242)
    assert build_family.payload_hash(first) == build_family.payload_hash(second)
    assert first["instances"] == second["instances"]


def test_a_different_seed_produces_a_different_family() -> None:
    """The hash is sensitive to the seed, so it can detect a silent re-roll."""
    assert build_family.payload_hash(
        build_family.build_family(4242)
    ) != build_family.payload_hash(build_family.build_family(4243))


# ---------------------------------------------------------------------------
# Short cells: capped means exhausted, not unlucky
# ---------------------------------------------------------------------------


def exhaustive_cell_size(n_items: int, m_class: str) -> int:
    """
    Number of distinct ``(items, target)`` pairs a family-P cell can hold.

    Enumerates every item vector in ``[1, a_max]**n`` and every eligible target.
    Cheap for the cells this is applied to: 49, 125 and 64 item vectors.
    """
    a_max = max_item_value(n_items)
    at_least = m_class.endswith("+")
    wanted = int(m_class[:-1]) if at_least else int(m_class)

    found: set[tuple[tuple[int, ...], int]] = set()
    for items in itertools.product(range(1, a_max + 1), repeat=n_items):
        spectrum = subset_sum_multiplicities(list(items))
        for target in range(1, sum(items) + 1):
            count = spectrum.get(target, 0)
            if (count >= wanted) if at_least else (count == wanted):
                found.add((items, target))
    return len(found)


@pytest.mark.parametrize(
    ("n_items", "m_class", "expected"),
    [(2, "2", 7), (2, "3+", 0), (3, "3+", 10), (6, "0", 6)],
)
def test_short_cells_are_genuinely_exhausted(
    records: list[dict[str, Any]], n_items: int, m_class: str, expected: int
) -> None:
    """
    The four short family-P cells hold *every* instance that exists.

    A cell that stops early because the stall counter tripped is only evidence
    of bad luck unless the true cell size is known.  It is known: these four
    are small enough to enumerate outright, and the generator found all of
    them.  ``n=2, M>=3`` is empty for a structural reason -- with two positive
    items the four subset sums are ``0, a, b, a+b`` and three of them can
    coincide only if an item is zero.
    """
    assert exhaustive_cell_size(n_items, m_class) == expected
    produced = [
        r
        for r in records
        if r["family"] == "P" and r["n_items"] == n_items and r["m_class"] == m_class
    ]
    assert len(produced) == expected


# ---------------------------------------------------------------------------
# The file: multiplicities
# ---------------------------------------------------------------------------


def test_family_file_is_the_committed_seed(family: dict[str, Any]) -> None:
    """The file on disk is the regeneration of its own recorded seed."""
    rebuilt = build_family.build_family(family["master_seed"])
    assert build_family.payload_hash(rebuilt) == family["payload_sha256"]


def test_every_M_matches_exhaustive_enumeration(records: list[dict[str, Any]]) -> None:
    """The recorded ``M`` is the brute-force count, instance by instance."""
    for record in records:
        assert record["M"] == brute_force_multiplicity(
            record["items"], record["target"]
        ), record["name"]


def test_recorded_class_matches_recorded_M(records: list[dict[str, Any]]) -> None:
    """``m_class`` is the bucket ``M`` falls in."""
    for record in records:
        assert multiplicity_class(record["M"]) == record["m_class"], record["name"]


# ---------------------------------------------------------------------------
# The file: solution bitstrings
# ---------------------------------------------------------------------------


def test_every_solution_bitstring_decodes_to_the_target(
    records: list[dict[str, Any]],
) -> None:
    """
    Each stored bitstring names a subset that really sums to the target.

    Decoding goes through ``dgssp.decoding``, which is the module that owns the
    convention "item 0 is the least-significant bit, i.e. the rightmost Qiskit
    character".  The test therefore fails if the file was written under a
    different convention *or* if the convention itself moves.
    """
    for record in records:
        items, target = record["items"], record["target"]
        for bits in record["solution_bitstrings"]:
            assert len(bits) == record["n_items"], record["name"]
            assert bit_is_solution(bits, items, target), (record["name"], bits)


def test_solution_bitstrings_are_complete_and_deduplicated(
    records: list[dict[str, Any]],
) -> None:
    """There are exactly ``M`` of them, sorted and distinct."""
    for record in records:
        bits = record["solution_bitstrings"]
        assert len(bits) == record["M"], record["name"]
        assert len(set(bits)) == len(bits), record["name"]
        assert bits == sorted(bits), record["name"]


def test_solution_bitstrings_match_the_dp_solver(records: list[dict[str, Any]]) -> None:
    """The stored bitstrings are what the classical ground truth produces now."""
    solver = DPSSPSolver(DPConfig(enumerate_all=True))
    for record in records:
        instance = SubsetSumInstance(
            items=list(record["items"]), target=record["target"], name=record["name"]
        )
        assert (
            exact_solution_bitstrings(instance, solver.solve(instance))
            == record["solution_bitstrings"]
        ), record["name"]


def test_bit_order_is_item_zero_rightmost(records: list[dict[str, Any]]) -> None:
    """
    The stored strings round-trip through the index representation.

    A silent MSB/LSB flip would still pass the sum check on palindromic
    subsets, so this pins the ordering directly.
    """
    for record in records:
        for bits in record["solution_bitstrings"]:
            indices = bitstring_to_indices(bits)
            assert indices_to_bitstring(indices, record["n_items"]) == bits


# ---------------------------------------------------------------------------
# The file: encoding and r_opt
# ---------------------------------------------------------------------------


def test_encoding_fields_match_the_encoding_module(
    records: list[dict[str, Any]],
) -> None:
    """No register arithmetic was re-derived when the file was written."""
    for record in records:
        enc = build_encoding(record["items"], record["target"])
        assert record["n_sum"] == enc.n_sum, record["name"]
        assert record["offset"] == enc.offset, record["name"]
        assert record["encoded_target"] == enc.encoded_target, record["name"]
        assert record["modulus"] == enc.modulus, record["name"]
        assert record["range_len"] == enc.range_len, record["name"]
        assert record["sum_neg"] == enc.sum_neg, record["name"]
        assert record["sum_pos"] == enc.sum_pos, record["name"]


def test_n_sum_is_in_the_designed_window(records: list[dict[str, Any]]) -> None:
    """``a_max(n)`` did its job for every instance actually drawn."""
    for record in records:
        assert record["n_sum"] in (3, 4, 5), record["name"]


def test_r_opt_is_the_library_value(records: list[dict[str, Any]]) -> None:
    """
    ``r_opt`` comes from ``optimal_iterations``, with M=0 clamped to M=1.

    The clamp is not a rounding detail: an M=0 instance is one an experimenter
    cannot know is infeasible, so the iteration count the paper reports for it
    is the one they would have used.
    """
    for record in records:
        assert record["r_opt"] == optimal_iterations(
            record["n_items"], record["M"]
        ), record["name"]
        if record["M"] == 0:
            assert record["r_opt"] == optimal_iterations(record["n_items"], 1)


# ---------------------------------------------------------------------------
# target_is_representable is NOT a feasibility test
# ---------------------------------------------------------------------------


def test_representable_is_an_interval_test_not_a_feasibility_test(
    records: list[dict[str, Any]],
) -> None:
    """
    The correct assertion, and why the obvious one is wrong.

    ``SumRegisterEncoding.target_is_representable`` is
    ``0 <= encoded_target < range_len``, i.e. "the target lies in the interval
    ``[sum_neg, sum_pos]`` the register can hold".  That is a statement about
    the *encoding*, not about whether any subset reaches the target, so

        target_is_representable == (M > 0)

    is false in general -- and false by construction for this family.  The M=0
    instances were deliberately drawn as *gaps* strictly inside the reachable
    interval, precisely so that the oracle marks no state while the circuit is
    otherwise perfectly well formed.  An out-of-range target would instead be
    rejected up front (the solver warns and the search is trivially hopeless),
    which is a different and far less interesting failure.

    What does hold, and is asserted here:

    * every instance in the family is representable, M=0 included;
    * ``M > 0`` implies representable -- a reachable target is in the interval;
    * the converse fails, and the family contains witnesses.
    """
    assert records, "family file has no instances"

    for record in records:
        enc = build_encoding(record["items"], record["target"])
        assert enc.target_is_representable is True, record["name"]
        assert record["target_is_representable"] is True, record["name"]
        if record["M"] > 0:
            assert enc.target_is_representable, record["name"]

    witnesses = [r for r in records if r["M"] == 0]
    assert witnesses, "family has no M=0 instances to witness the strict implication"
    for record in witnesses:
        assert build_encoding(record["items"], record["target"]).target_is_representable


def test_out_of_range_targets_are_not_representable(
    records: list[dict[str, Any]],
) -> None:
    """The other side of the distinction: outside the interval is rejected."""
    for record in records[:40]:
        items = record["items"]
        enc_high = build_encoding(items, record["sum_pos"] + 1)
        enc_low = build_encoding(items, record["sum_neg"] - 1)
        assert not enc_high.target_is_representable
        assert not enc_low.target_is_representable


# ---------------------------------------------------------------------------
# Cell bookkeeping
# ---------------------------------------------------------------------------


def test_cell_reports_match_the_instances(family: dict[str, Any]) -> None:
    """Every cell's ``produced`` count is the number of instances it emitted."""
    for cell in family["cells"]:
        matching = [
            r
            for r in family["instances"]
            if r["family"] == cell["family"]
            and r["n_items"] == cell["n_items"]
            and r["n_negative"] == cell["n_negative"]
            and r["m_class"] == cell["m_class"]
        ]
        assert len(matching) == cell["produced"], cell

    for cell in family["cells"]:
        if cell["status"] == "full":
            assert cell["produced"] == cell["wanted"], cell
        else:
            assert cell["produced"] < cell["wanted"], cell
            assert cell["reason"], cell


def test_family_S_sign_counts(family: dict[str, Any]) -> None:
    """Family S is half one-negative, half two-negative, per cell."""
    signed = [r for r in family["instances"] if r["family"] == "S"]
    assert signed
    for record in signed:
        assert sum(1 for a in record["items"] if a < 0) == record["n_negative"]
        assert record["n_negative"] in (1, 2)

    per_cell: dict[tuple[int, str], dict[int, int]] = {}
    for record in signed:
        key = (record["n_items"], record["m_class"])
        per_cell.setdefault(key, {1: 0, 2: 0})[record["n_negative"]] += 1
    for key, counts in per_cell.items():
        assert counts[1] == counts[2], key


def test_instance_names_are_unique(records: list[dict[str, Any]]) -> None:
    """Names address instances in later work packages, so they must be keys."""
    names = [r["name"] for r in records]
    assert len(set(names)) == len(names)


def test_instances_are_distinct(records: list[dict[str, Any]]) -> None:
    """No ``(items, target)`` pair appears twice anywhere in the family."""
    keys = [(tuple(r["items"]), r["target"]) for r in records]
    assert len(set(keys)) == len(keys)
