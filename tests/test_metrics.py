"""
Tests for calibration-based error metrics.

The regression these guard against: the old implementation swallowed every
calibration lookup failure into a silent zero, so a backend with missing data
looked *better* than a well-characterised one and backend ranking became
meaningless.
"""

from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from dgssp import DGConfig, DGSSPSolver, SubsetSumInstance, compute_accumulated_errors
from dgssp.transpilation import find_best_seed, two_qubit_gate_errors_per_circuit_layout


@pytest.fixture
def fake_backend():
    """A bundled fake IBM device with real calibration data."""
    pytest.importorskip("qiskit_ibm_runtime")
    try:
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2

        return FakeManilaV2()
    except Exception:  # pragma: no cover
        pytest.skip("fake provider data unavailable")


@pytest.fixture
def transpiled(fake_backend):
    """A D-G circuit transpiled onto that fake device."""
    instance = SubsetSumInstance(items=[1, 2], target=3)
    qc = DGSSPSolver(DGConfig(iterations=1)).build_circuit(instance)
    pm = generate_preset_pass_manager(
        backend=fake_backend, optimization_level=1, seed_transpiler=7
    )
    return pm.run(qc)


def test_errors_are_positive_and_finite(fake_backend, transpiled):
    """Real calibration data yields nonzero, finite accumulated error."""
    metrics = compute_accumulated_errors(fake_backend, transpiled)

    assert math.isfinite(metrics.total_error)
    assert metrics.total_error > 0.0, "silent-zero regression"
    assert metrics.single_qubit_error > 0.0
    assert metrics.readout_error > 0.0
    assert metrics.single_qubit_gate_count > 0


def test_total_is_the_sum_of_its_parts(fake_backend, transpiled):
    """total_error is exactly 1q + 2q + readout."""
    m = compute_accumulated_errors(fake_backend, transpiled)
    assert m.total_error == pytest.approx(
        m.single_qubit_error + m.two_qubit_error + m.readout_error
    )


def test_missing_calibration_is_counted_not_hidden(fake_backend, transpiled):
    """Uncalibrated instructions are reported rather than scored as zero."""
    m = compute_accumulated_errors(fake_backend, transpiled)
    assert m.missing_calibration >= 0
    assert "missing_calibration" in m.to_dict()


def test_backend_without_target_is_an_explicit_error(transpiled):
    """A backend with no target raises instead of returning zeros."""

    class NoTarget:
        target = None
        name = "no_target"

    with pytest.raises(AttributeError):
        compute_accumulated_errors(NoTarget(), transpiled)


def test_two_qubit_report_is_consistent(fake_backend, transpiled):
    """The per-pair report agrees with the aggregate two-qubit error."""
    report = two_qubit_gate_errors_per_circuit_layout(transpiled, fake_backend)
    metrics = compute_accumulated_errors(fake_backend, transpiled)

    assert report.gate_count == metrics.two_qubit_gate_count
    assert report.accumulated_error == pytest.approx(metrics.two_qubit_error)
    assert len(report.pairs) == len(report.error_per_pair)


def test_find_best_seed_returns_a_physical_circuit(fake_backend):
    """The seed sweep produces a transpiled circuit and records its seed."""
    instance = SubsetSumInstance(items=[1, 2], target=3)
    qc = DGSSPSolver(DGConfig(iterations=1)).build_circuit(instance)

    best = find_best_seed(qc, fake_backend, seed_min=0, seed_max=4)
    assert isinstance(best.circuit, QuantumCircuit)
    assert 0 <= best.best_seed < 4
    assert best.circuit.num_qubits == fake_backend.num_qubits
    assert best.total_two_qubit_error >= 0.0


def test_empty_seed_range_rejected(fake_backend):
    """An empty sweep range is an error, not a silent no-op."""
    instance = SubsetSumInstance(items=[1, 2], target=3)
    qc = DGSSPSolver(DGConfig(iterations=1)).build_circuit(instance)
    with pytest.raises(ValueError):
        find_best_seed(qc, fake_backend, seed_min=5, seed_max=5)
