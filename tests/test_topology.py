"""
Tests for heavy-hex device discovery and transpiled-circuit size metrics.

These guard the two things the paper's resource claims rest on: that "heavy-hex
device" means the same thing every time it is asserted, and that the four
numbers reported for an executed circuit have exactly one definition.
"""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from dgssp import (
    DGConfig,
    DGSSPSolver,
    SubsetSumInstance,
    build_ideal_aer_backend,
    is_heavy_hex,
    list_fake_backends,
    smallest_heavy_hex_fake_backend,
)
from dgssp.backends import (
    HEAVY_HEX_FAMILIES,
    HEAVY_HEX_MIN_QUBITS,
    calibration_timestamp,
    coupling_degrees,
    processor_family,
)
from dgssp.transpilation import (
    physical_qubits,
    transpiled_metrics,
    two_qubit_count,
    two_qubit_depth,
)


@pytest.fixture(scope="module")
def heavy_hex_backends():
    """Every bundled heavy-hex fake device, or skip if none ship."""
    pytest.importorskip("qiskit_ibm_runtime")
    backends = list_fake_backends(heavy_hex_only=True)
    if not backends:
        pytest.skip("installed qiskit-ibm-runtime ships no heavy-hex fake devices")
    return backends


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_heavy_hex_devices_are_discovered_not_hard_coded(heavy_hex_backends):
    """At least one heavy-hex device is found, and it is wide enough to be one."""
    assert len(heavy_hex_backends) >= 1
    for backend in heavy_hex_backends:
        assert backend.num_qubits >= HEAVY_HEX_MIN_QUBITS


def test_heavy_hex_devices_have_degree_at_most_three(heavy_hex_backends):
    """The defining topological property of the lattice."""
    for backend in heavy_hex_backends:
        degrees = coupling_degrees(backend)
        assert degrees, f"{backend.name} exposed no coupling map"
        assert max(degrees.values()) <= 3
        assert any(d == 3 for d in degrees.values())


def test_advertised_families_are_all_recognised(heavy_hex_backends):
    """Nothing slips through with an unknown processor family."""
    for backend in heavy_hex_backends:
        family = processor_family(backend)
        assert family is None or family in HEAVY_HEX_FAMILIES


def test_small_devices_are_not_heavy_hex():
    """A narrow Falcon fragment is not a heavy-hex lattice, family label or not."""
    pytest.importorskip("qiskit_ibm_runtime")
    for backend in list_fake_backends(max_qubits=HEAVY_HEX_MIN_QUBITS - 1):
        assert not is_heavy_hex(backend)


def test_ideal_simulator_is_not_heavy_hex():
    """An all-to-all simulator has no lattice at all."""
    assert not is_heavy_hex(build_ideal_aer_backend())


def test_selection_is_deterministic_and_wide_enough(heavy_hex_backends):
    """The narrowest matching device is picked, and the choice is stable."""
    first = smallest_heavy_hex_fake_backend(min_qubits=7)
    second = smallest_heavy_hex_fake_backend(min_qubits=7)
    assert first.name == second.name
    assert first.num_qubits == min(b.num_qubits for b in heavy_hex_backends)


def test_selection_raises_when_nothing_is_wide_enough():
    """An impossible width fails loudly rather than returning a narrow device."""
    pytest.importorskip("qiskit_ibm_runtime")
    with pytest.raises(RuntimeError, match="No heavy-hex fake backend"):
        smallest_heavy_hex_fake_backend(min_qubits=100_000)


def test_calibration_timestamp_present_on_devices_absent_on_simulators(
    heavy_hex_backends,
):
    """Provenance can date a device run, and honestly reports when it cannot."""
    stamp = calibration_timestamp(heavy_hex_backends[0])
    assert isinstance(stamp, str) and len(stamp) >= 10
    assert calibration_timestamp(build_ideal_aer_backend()) is None


# ---------------------------------------------------------------------------
# Transpiled metrics
# ---------------------------------------------------------------------------


def test_two_qubit_counters_ignore_barriers():
    """Barriers are two-or-more-qubit instructions but not gates."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.barrier()
    qc.cx(1, 0)

    assert two_qubit_count(qc) == 2
    assert two_qubit_depth(qc) == 2


def test_untranspiled_circuit_has_no_physical_qubits():
    """A circuit with no layout reports ``None`` rather than inventing indices."""
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    assert physical_qubits(qc) is None


def test_transpiled_metrics_match_the_device(heavy_hex_backends):
    """The recorded metrics describe the circuit that was actually submitted."""
    backend = heavy_hex_backends[0]
    instance = SubsetSumInstance(items=[1, 2, 3], target=5)
    qc = DGSSPSolver(DGConfig(iterations=1)).build_circuit(instance)
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=1, seed_transpiler=7
    )
    tqc = pm.run(qc)

    metrics = transpiled_metrics(tqc)

    assert set(metrics) == {"two_q_count", "depth", "two_q_depth", "physical_qubits"}
    assert metrics["depth"] == tqc.depth()
    assert metrics["two_q_count"] >= metrics["two_q_depth"] > 0
    assert metrics["depth"] >= metrics["two_q_depth"]

    qubits = metrics["physical_qubits"]
    assert len(qubits) == qc.num_qubits
    assert len(set(qubits)) == len(qubits)
    assert all(0 <= q < backend.num_qubits for q in qubits)


def test_instance_result_carries_the_same_metrics(heavy_hex_backends):
    """``run_batch`` records exactly what ``transpiled_metrics`` defines."""
    from qiskit_aer import AerSimulator

    from dgssp import BatchConfig, run_batch

    backend = AerSimulator.from_backend(heavy_hex_backends[0])
    instance = SubsetSumInstance(items=[1, 2, 3], target=5, name="t")
    result = run_batch(
        [instance],
        BatchConfig(
            executor="noisy",
            backend=backend,
            shots=128,
            seed_simulator=5,
            seed_transpiler=7,
            optimization_level=1,
        ),
    ).results[0]

    summary = result.transpiled_summary()
    assert summary["two_q_count"] == result.two_qubit_count
    assert summary["two_q_depth"] == result.two_qubit_depth
    assert summary["depth"] == result.depth
    assert len(summary["physical_qubits"]) == result.n_qubits
