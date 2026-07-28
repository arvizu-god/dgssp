"""
Statevector check that the sum register is returned to |0...0> after a step.

If the adder and subtractor are not exact inverses -- for example if the
constant offset is added but not removed -- the sum register stays entangled
with the index register, the diffuser operates on a mixed state, and Grover
silently stops working.  This test catches that directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from dgssp import DGConfig, DGSSPSolver, SubsetSumInstance, build_encoding
from dgssp.solvers.draper_grover import qft_no_swaps

INSTANCES = [
    ([1, 2, 3], 5),
    ([3, -2, 4], 1),
    ([-1, -2], -3),
    ([5, 5], 5),
]


@pytest.mark.parametrize("items,target", INSTANCES)
def test_sum_register_returns_to_zero(items, target):
    """After one add -> oracle -> subtract step the sum register is |0...0>."""
    instance = SubsetSumInstance(items=items, target=target)
    enc = build_encoding(items, target)
    solver = DGSSPSolver(DGConfig(iterations=1, measure=False))

    qc = QuantumCircuit(enc.n_ind + enc.n_sum)
    qc.h(range(enc.n_ind))  # superposition over every subset
    qc.compose(solver.build_step_circuit(instance), inplace=True)

    state = Statevector.from_instruction(qc)
    probs = np.asarray(state.probabilities())

    # Qiskit orders basis states with qubit 0 as the least significant bit, so
    # the sum register occupies the high bits.
    leaked = 0.0
    for basis_index, prob in enumerate(probs):
        if prob < 1e-12:
            continue
        sum_value = basis_index >> enc.n_ind
        if sum_value != 0:
            leaked += prob

    assert leaked < 1e-9, f"sum register retained {leaked:.3e} probability"


@pytest.mark.parametrize("items,target", INSTANCES)
def test_only_solutions_pick_up_a_phase(items, target):
    """The step applies a -1 phase to exactly the solution subsets."""
    instance = SubsetSumInstance(items=items, target=target)
    enc = build_encoding(items, target)
    solver = DGSSPSolver(DGConfig(iterations=1, measure=False))

    qc = QuantumCircuit(enc.n_ind + enc.n_sum)
    qc.h(range(enc.n_ind))
    qc.compose(solver.build_step_circuit(instance), inplace=True)

    amps = np.asarray(Statevector.from_instruction(qc).data)
    n_states = 2**enc.n_ind
    reference = 1.0 / np.sqrt(n_states)

    for index in range(n_states):
        subset_sum = sum(a for k, a in enumerate(items) if (index >> k) & 1)
        expected = -reference if subset_sum == target else reference
        assert amps[index] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("items,target", INSTANCES)
def test_adder_computes_the_encoded_sum(items, target):
    """The QFT adder alone lands on encode(sum) for each basis subset."""
    enc = build_encoding(items, target)
    solver = DGSSPSolver(DGConfig(measure=False))

    for index in range(2**enc.n_ind):
        qc = QuantumCircuit(enc.n_ind + enc.n_sum)
        for k in range(enc.n_ind):
            if (index >> k) & 1:
                qc.x(k)

        sum_qubits = list(range(enc.n_ind, enc.n_ind + enc.n_sum))
        qft = qft_no_swaps(enc.n_sum)
        qc.append(qft, sum_qubits)
        qc.append(solver._sum_gate(enc), list(range(enc.n_ind)) + sum_qubits)
        qc.append(qft.inverse(), sum_qubits)

        probs = np.asarray(Statevector.from_instruction(qc).probabilities())
        measured = int(np.argmax(probs))
        subset_sum = sum(a for k, a in enumerate(items) if (index >> k) & 1)

        assert probs[measured] == pytest.approx(1.0, abs=1e-9)
        assert (measured >> enc.n_ind) == enc.encode(subset_sum)
