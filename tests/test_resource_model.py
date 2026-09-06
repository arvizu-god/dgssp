"""
Tests for the closed-form resource model of the D-G circuit (WP2 / C1).

The model in :func:`dgssp.solvers.draper_grover.resource_model` and the note in
``paper/notes/resource_model.md`` are the paper's claim about how the circuit
scales. These tests are what stops that claim from drifting away from the
circuit the solver actually builds: a change to the assembly, the adder or the
multi-controlled-Z helper fails here rather than in a referee's report.
"""

from __future__ import annotations

import pytest

from dgssp import (
    DGConfig,
    DGSSPSolver,
    SubsetSumInstance,
    build_encoding,
    encoding_without_guard_bit,
    flatten_containers,
    median_gate_errors,
    resource_model,
)
from dgssp.solvers.draper_grover import two_adic_valuation
from dgssp.transpilation import (
    logical_counts,
    mcx_noaux_1q_cost,
    mcx_noaux_cx_cost,
    mcz_noaux_1q_cost,
    mcz_noaux_cx_cost,
    to_logical_basis,
)

#: A spread of shapes: degenerate index register, all-positive, signed, items
#: with a zero residue, duplicated items, wide sum register.
CASES: list[tuple[list[int], int]] = [
    ([1, 1], 1),
    ([1, 2, 3], 5),
    ([1, 1, 1], 2),
    ([1, 2, 4, 8], 9),
    ([3, 5, 7, 10], 15),
    ([3, -2, 4], 1),
    ([1, 3, 5, -7], 4),
    ([16, 8, 4, 2, 1], 21),
    ([1, 3, 5, 7, 9, 11], 20),
]

#: Gate names the flattened circuit may contain.
GATE_LEVEL_OPS = {"cp", "p", "h", "x", "z", "cz", "ccx", "mcx", "barrier", "measure"}


def _circuit(items: list[int], target: int, iterations: int):
    """Build the D-G circuit for an instance at a fixed iteration count."""
    instance = SubsetSumInstance(items=items, target=target)
    return DGSSPSolver(DGConfig(iterations=iterations)).build_circuit(instance)


# ---------------------------------------------------------------------------
# Gate-level counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("items", "target"), CASES)
@pytest.mark.parametrize("iterations", [1, 3])
def test_gate_level_counts_match_the_circuit(items, target, iterations):
    """Every closed form reproduces ``count_ops()`` of the flattened circuit."""
    encoding = build_encoding(items, target)
    model = resource_model(encoding, iterations)
    flat = flatten_containers(_circuit(items, target, iterations))
    observed = {name: int(count) for name, count in flat.count_ops().items()}

    assert set(observed) <= GATE_LEVEL_OPS

    predicted = dict(model.gate_level_counts())
    for name, count in model.multi_controlled_op_counts().items():
        predicted[name] = predicted.get(name, 0) + count

    for name, count in predicted.items():
        assert observed.get(name, 0) == count, f"{name}: expected {count}"


@pytest.mark.parametrize(("items", "target"), CASES)
def test_two_and_one_qubit_reference_match_the_pinned_translation(items, target):
    """The ``{CX, 1Q}`` reference counts match the pinned ancilla-free basis."""
    encoding = build_encoding(items, target)
    model = resource_model(encoding, 1)
    measured = logical_counts(_circuit(items, target, 1))

    assert measured["two_q_count"] == model.two_qubit_reference(mcz_noaux_cx_cost)
    assert measured["one_q_count"] == model.one_qubit_reference(mcx_noaux_1q_cost)


def test_counts_are_linear_in_the_iteration_count():
    """Only the initial Hadamard layer sits outside the Grover loop."""
    encoding = build_encoding([1, 2, 3], 5)
    one = resource_model(encoding, 1)
    four = resource_model(encoding, 4)

    assert four.controlled_phase == 4 * one.controlled_phase
    assert four.pauli_x == 4 * one.pauli_x
    assert four.hadamard - encoding.n_ind == 4 * (one.hadamard - encoding.n_ind)


def test_qft_and_adder_terms_follow_the_note():
    """The two controlled-phase terms are ``2b(b-1)`` and ``2 n_plus b``."""
    encoding = build_encoding([3, 5, 7, 10], 15)
    n, b = encoding.n_ind, encoding.n_sum
    model = resource_model(encoding, 1)

    assert model.qft_cp_per_iteration == 2 * b * (b - 1)
    assert model.nonzero_items == n
    assert model.adder_cp_per_iteration == 2 * n * b
    assert model.mcz_widths == (b, n)
    assert model.num_qubits == n + b
    assert model.num_ancillas == 0


def test_only_a_literally_zero_item_has_a_zero_residue():
    """``n_plus < n`` happens exactly when an item is 0, never by wrap-around.

    The guard bit forces ``2**b >= 2 * range_len > 2 * |a_k|`` for every single
    item, so no non-zero item can be congruent to 0 modulo the adder's modulus.
    The ``a_mod == 0`` branch in ``_sum_gate`` is therefore reachable only for a
    zero item -- worth pinning, because it is what makes ``n_plus`` a property
    of the instance rather than of the register width.
    """
    encoding = build_encoding([0, 1, 2], 3)
    model = resource_model(encoding, 1)
    assert model.nonzero_items == 2
    assert model.adder_cp_per_iteration == 2 * 2 * encoding.n_sum

    for items, target in CASES:
        other = build_encoding(items, target)
        assert resource_model(other, 1).nonzero_items == sum(1 for a in items if a != 0)


# ---------------------------------------------------------------------------
# The 2*pi-multiple saving the implementation does not take
# ---------------------------------------------------------------------------


def test_all_odd_items_leave_nothing_to_skip():
    """With every item odd, no adder rotation is a multiple of ``2*pi``."""
    model = resource_model(build_encoding([1, 3, 5, 7], 8), 1)
    assert model.adder_cp_per_iteration_skippable == model.adder_cp_per_iteration


def test_powers_of_two_leave_the_most_to_skip():
    """Powers of two maximise the saving a valuation-aware adder would take."""
    encoding = build_encoding([1, 2, 4, 8], 9)
    b = encoding.n_sum
    model = resource_model(encoding, 1)
    expected = 2 * sum(b - v for v in (0, 1, 2, 3))
    assert model.adder_cp_per_iteration_skippable == expected
    assert model.adder_cp_per_iteration_skippable < model.adder_cp_per_iteration


def test_two_adic_valuation():
    """The valuation helper, including its refusal to answer for zero."""
    assert [two_adic_valuation(x) for x in (1, 2, 3, 4, 12, 96)] == [0, 1, 0, 2, 2, 5]
    with pytest.raises(ValueError, match="undefined"):
        two_adic_valuation(0)


# ---------------------------------------------------------------------------
# Signed encoding and the guard bit
# ---------------------------------------------------------------------------


def test_negating_an_item_costs_only_single_qubit_phases():
    """Sign flips leave ``b`` and every two-qubit count untouched (note SS7a)."""
    positive = build_encoding([1, 3, 5, 7], 8)
    negated = build_encoding([1, 3, 5, -7], 4)

    assert negated.n_sum == positive.n_sum
    assert negated.offset == 7 and positive.offset == 0

    model_pos = resource_model(positive, 1)
    model_neg = resource_model(negated, 1)

    assert model_neg.controlled_phase == model_pos.controlled_phase
    assert model_neg.two_qubit_reference(mcz_noaux_cx_cost) == model_pos.two_qubit_reference(
        mcz_noaux_cx_cost
    )
    assert model_pos.phase == 0
    assert model_neg.phase == 2 * negated.n_sum


def test_guard_bit_overhead_is_one_qubit_and_a_countable_delta():
    """The counterfactual encoding is one bit narrower and strictly cheaper."""
    encoding = build_encoding([3, 5, 7, 10], 15)
    narrow = encoding_without_guard_bit(encoding)

    assert narrow.n_sum == encoding.n_sum - 1
    assert narrow.modulus == encoding.modulus // 2

    wide_model = resource_model(encoding, 1)
    narrow_model = resource_model(narrow, 1)
    b, n_plus = encoding.n_sum, wide_model.nonzero_items

    assert wide_model.controlled_phase - narrow_model.controlled_phase == 4 * (b - 1) + 2 * n_plus
    assert wide_model.num_qubits - narrow_model.num_qubits == 1


def test_guard_bit_cannot_be_removed_from_a_one_qubit_register():
    """A one-qubit sum register has no guard bit left to price."""
    narrow = encoding_without_guard_bit(build_encoding([1], 1))
    assert narrow.n_sum == 1
    with pytest.raises(ValueError, match="guard bit"):
        encoding_without_guard_bit(narrow)


# ---------------------------------------------------------------------------
# Synthesis-pinned accounting
# ---------------------------------------------------------------------------


def test_mcz_cost_special_cases_the_degenerate_widths():
    """The solver emits ``Z`` at width 1 and ``CZ`` at width 2."""
    assert mcz_noaux_cx_cost(1) == 0
    assert mcz_noaux_cx_cost(2) == 1
    assert mcz_noaux_1q_cost(1) == 1
    for width in range(3, 8):
        assert mcz_noaux_cx_cost(width) == mcx_noaux_cx_cost(width - 1)
        assert mcz_noaux_1q_cost(width) == mcx_noaux_1q_cost(width - 1) + 2


def test_mcx_reference_cost_is_ancilla_free_and_grows_quadratically():
    """The pinned synthesis ignores idle qubits, so cost depends only on ``k``."""
    costs = [mcx_noaux_cx_cost(k) for k in range(1, 10)]
    assert costs == sorted(costs)
    # Exact for k >= 7 with the pinned plugin; see paper/notes/resource_model.md.
    for k in range(7, 12):
        assert mcx_noaux_cx_cost(k) == 8 * k * k - 16 * k - 60
    with pytest.raises(ValueError, match=">= 1"):
        mcx_noaux_cx_cost(0)


def test_to_logical_basis_is_deterministic():
    """Two translations of the same circuit give identical counts."""
    circuit = _circuit([1, 2, 4, 8], 9, 1)
    first = to_logical_basis(circuit).count_ops()
    second = to_logical_basis(circuit).count_ops()
    assert dict(first) == dict(second)


# ---------------------------------------------------------------------------
# Structural guards
# ---------------------------------------------------------------------------


def test_flatten_containers_leaves_only_primitive_gates():
    """No composite gate survives, and nothing unexpected appears."""
    flat = flatten_containers(_circuit([1, 3, 5, 7, 9], 16, 2))
    assert {instruction.operation.name for instruction in flat.data} <= GATE_LEVEL_OPS


def test_resource_model_rejects_a_non_positive_iteration_count():
    """A zero-iteration circuit is not a Grover search."""
    with pytest.raises(ValueError, match=">= 1"):
        resource_model(build_encoding([1, 2], 3), 0)


def test_resource_model_round_trips_through_json():
    """Every field the archive stores is a plain JSON type."""
    import json

    payload = resource_model(build_encoding([3, -2, 4], 1), 2).to_dict()
    assert json.loads(json.dumps(payload)) == payload


# ---------------------------------------------------------------------------
# Median calibration errors (the survival estimate's input)
# ---------------------------------------------------------------------------


def test_median_gate_errors_excludes_virtual_gates():
    """``rz`` has zero duration and zero error and must not enter the median."""
    pytest.importorskip("qiskit_ibm_runtime")
    from dgssp.backends import smallest_heavy_hex_fake_backend

    backend = smallest_heavy_hex_fake_backend()
    errors = median_gate_errors(backend)

    assert errors["one_qubit"] is not None and errors["one_qubit"] > 0
    assert errors["two_qubit"] is not None
    assert errors["two_qubit"] > errors["one_qubit"]
    assert "rz" not in errors["one_qubit_gates"]
    assert errors["num_two_qubit"] > 0


def test_median_gate_errors_needs_a_target():
    """A backend with no calibration store fails loudly rather than silently."""

    class Bare:
        name = "bare"
        target = None

    with pytest.raises(AttributeError, match="target"):
        median_gate_errors(Bare())
