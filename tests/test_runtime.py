"""Tests for the runtime layer: batching, logging and import purity."""

from __future__ import annotations

import json

import pytest
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from dgssp import (
    DGConfig,
    DGSSPSolver,
    SubsetSumInstance,
    build_ideal_aer_backend,
    read_job_log,
    sample_counts,
)
from dgssp.runtime import backend_name, execution_mode, is_simulator


@pytest.fixture
def circuits():
    """Three transpiled circuits of differing widths."""
    backend = build_ideal_aer_backend(seed_simulator=3)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    built = []
    for items, target in ([1, 2], 3), ([1, 2, 3], 5), ([2, -1, 3], 2):
        instance = SubsetSumInstance(items=items, target=target)
        qc = DGSSPSolver(DGConfig(iterations=1)).build_circuit(instance)
        built.append(pm.run(qc))
    return backend, built


def test_batched_sampling_returns_one_dict_per_circuit(circuits):
    """A list of circuits comes back as a list of counts, in order."""
    backend, built = circuits
    counts_list = sample_counts(
        backend, built, shots=512, seed_simulator=3, run_log=None
    )

    assert len(counts_list) == 3
    for counts, qc in zip(counts_list, built, strict=True):
        assert sum(counts.values()) == 512
        assert all(len(bit.replace(" ", "")) == qc.num_clbits for bit in counts)


def test_single_circuit_still_returns_a_list(circuits):
    """The return type does not change with the input arity."""
    backend, built = circuits
    result = sample_counts(backend, built[0], shots=256, seed_simulator=3, run_log=None)
    assert isinstance(result, list) and len(result) == 1


def test_job_is_logged_to_the_ledger(circuits, tmp_path):
    """Each submission appends one JSON line with its metadata."""
    backend, built = circuits
    log_path = tmp_path / "runs" / "jobs.jsonl"

    sample_counts(
        backend, built, shots=128, seed_simulator=3, run_log=str(log_path),
        meta={"stage": "test"},
    )

    assert log_path.exists()
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    assert len(records) == 1
    assert records[0]["n_pubs"] == 3
    assert records[0]["shots"] == 128
    assert records[0]["stage"] == "test"

    assert read_job_log(str(log_path))[0]["n_pubs"] == 3


def test_read_job_log_missing_file_is_empty(tmp_path):
    """A missing ledger reads as an empty list, not an error."""
    assert read_job_log(str(tmp_path / "nope.jsonl")) == []


def test_simulator_detection_and_naming(circuits):
    """Aer backends are recognised as simulators and named consistently."""
    backend, _ = circuits
    assert is_simulator(backend)
    assert isinstance(backend_name(backend), str)


def test_execution_mode_passes_simulators_through(circuits):
    """Simulators are not wrapped in a Session or Batch."""
    backend, _ = circuits
    with execution_mode(backend) as mode:
        assert mode is backend


def test_importing_dgssp_has_no_side_effects():
    """No credentials are written and no service is built on import."""
    import importlib
    import sys
    from unittest import mock

    for name in [m for m in sys.modules if m.startswith("dgssp")]:
        del sys.modules[name]

    with mock.patch(
        "qiskit_ibm_runtime.QiskitRuntimeService.save_account"
    ) as save, mock.patch(
        "qiskit_ibm_runtime.QiskitRuntimeService.__init__", return_value=None
    ) as init:
        importlib.import_module("dgssp")
        save.assert_not_called()
        init.assert_not_called()
