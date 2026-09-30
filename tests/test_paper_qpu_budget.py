"""
Tests for ``paper/scripts/qpu_budget.py``.

The estimate decides whether a month's allowance can hold a job shape, so the
parts worth pinning are the unit conversion (a factor of ``dt`` wrong is a
factor of 4e-9 wrong), the execution-time model, and the device limits that
make a shape illegal rather than merely expensive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "paper" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("scipy")
qpu_budget = pytest.importorskip("qpu_budget")

from dgssp.instance import SubsetSumInstance  # noqa: E402
from dgssp.solvers.draper_grover import DGConfig, DGSSPSolver  # noqa: E402


@pytest.fixture(scope="module")
def device():
    """A bundled heavy-hex fake device with calibrated durations."""
    pytest.importorskip("qiskit_ibm_runtime")
    try:
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    except Exception:  # pragma: no cover - provider data varies by release
        pytest.skip("FakeMarrakesh unavailable")
    return FakeMarrakesh()


@pytest.fixture(scope="module")
def isa_circuit(device):
    """One transpiled instance, the thing a duration is measured on."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    circuit = DGSSPSolver(DGConfig(iterations=1)).build_circuit(
        SubsetSumInstance(items=[2, 1], target=3, name="budget")
    )
    return generate_preset_pass_manager(
        backend=device, optimization_level=3, seed_transpiler=6
    ).run(circuit)


def test_a_duration_is_seconds_not_dt(device, isa_circuit):
    """
    The schedule comes back in ``dt``; reporting it unconverted would
    understate the cost by a factor of 4e-9.
    """
    seconds = qpu_budget.schedule_seconds(isa_circuit, device)
    # A two-item instance is tens of microseconds: longer than one readout
    # (2.6 us), far shorter than a millisecond.
    assert 5e-6 < seconds < 1e-3


def test_a_longer_circuit_takes_longer(device, isa_circuit):
    """Sanity: folding the circuit three times triples the work it schedules."""
    from dgssp.mitigation.zne import fold_isa, rebase_to_backend

    folded = rebase_to_backend(fold_isa(isa_circuit, 3.0, seed=0).circuit, device)
    plain = qpu_budget.schedule_seconds(isa_circuit, device)
    stretched = qpu_budget.schedule_seconds(folded, device)
    assert stretched > 2.0 * plain


def test_the_execution_model_is_shots_times_duration_plus_delay():
    """The arithmetic the whole estimate rests on."""
    assert qpu_budget.qpu_seconds(40e-6, 1000, 250e-6) == pytest.approx(0.29)
    # With no delay it is pure execution.
    assert qpu_budget.qpu_seconds(40e-6, 1000, 0.0) == pytest.approx(0.04)


def test_reset_delay_dominates_these_circuits():
    """
    The result that shapes a job: on circuits this short, most QPU time is
    spent waiting for qubits to relax, so shots are the expensive axis and
    depth is not.
    """
    circuits = [{"name": "a", "n_items": 3, "two_q": 300, "duration_s": 40e-6}] * 10
    stats = qpu_budget.summarise(circuits, shots=8192, rep_delay=250e-6)
    assert stats["n_circuits"] == 10
    assert stats["idle_fraction"] > 0.8
    assert stats["total_seconds"] == pytest.approx(10 * 8192 * 290e-6)
    assert stats["total_minutes"] == pytest.approx(stats["total_seconds"] / 60.0)


def test_halving_the_shots_halves_the_bill():
    """Shots scale the whole estimate linearly; nothing else does."""
    circuits = [{"name": "a", "n_items": 2, "two_q": 100, "duration_s": 30e-6}] * 4
    full = qpu_budget.summarise(circuits, shots=8192, rep_delay=250e-6)
    half = qpu_budget.summarise(circuits, shots=4096, rep_delay=250e-6)
    assert half["total_seconds"] == pytest.approx(full["total_seconds"] / 2.0)


def test_device_limits_are_read_from_the_device(device):
    """A job shape is checked against the real caps, not against assumptions."""
    limits = qpu_budget.device_limits(device)
    assert limits["max_circuits"] and limits["max_circuits"] > 0
    assert limits["max_shots"] and limits["max_shots"] > 0
    assert limits["dt"] == pytest.approx(4e-9)
    low, high = limits["rep_delay_range"]
    assert low <= limits["default_rep_delay"] <= high


def test_an_oversized_job_is_refused(device, monkeypatch, capsys, tmp_path):
    """
    Exceeding the per-job circuit limit is a failure, not a footnote: the job
    would be rejected at submission, after the queue wait.
    """
    monkeypatch.setattr(
        qpu_budget,
        "device_limits",
        lambda backend: {
            "max_shots": 100_000,
            "max_circuits": 4,
            "default_rep_delay": 250e-6,
            "rep_delay_range": [0.0, 5e-4],
            "dt": 4e-9,
        },
    )
    status = qpu_budget.main(
        ["--only", "P-n2-M1-11", "P-n2-M0-06", "--grid", "odd", "--no-mitiq"]
    )
    out = capsys.readouterr().out
    assert status == 1
    assert "exceeds this device's limit" in out


def test_the_estimate_is_reported_as_a_floor(capsys):
    """It counts execution only; saying otherwise would over-promise."""
    qpu_budget.main(["--only", "P-n2-M1-11", "--grid", "odd", "--no-mitiq"])
    out = capsys.readouterr().out
    assert "floor" in out
    assert "usage_estimation" in out
    assert "ESTIMATED QPU TIME" in out
