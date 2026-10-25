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


def test_submit_then_decode_equals_sample_counts(circuits):
    """The non-blocking path and the blocking one give the same counts."""
    from dgssp.runtime import counts_from_result, submit_sampler_job

    backend, built = circuits
    job = submit_sampler_job(backend, built, shots=256, seed_simulator=11, run_log=None)
    split = counts_from_result(job.result())
    joined = sample_counts(backend, built, shots=256, seed_simulator=11, run_log=None)
    assert list(split) == list(joined)


def test_options_dict_reaches_the_sampler_and_keeps_the_seed(circuits):
    """Server-side options are accepted locally (with a warning) and the seed survives."""
    backend, built = circuits
    options = {
        "twirling": {"enable_gates": True, "num_randomizations": 4,
                     "shots_per_randomization": 64},
        "dynamical_decoupling": {"enable": True, "sequence_type": "XX"},
    }
    with pytest.warns(UserWarning, match="no effect in local testing mode"):
        with_options = sample_counts(
            backend, built, shots=256, seed_simulator=5, run_log=None, options=options
        )
    plain = sample_counts(backend, built, shots=256, seed_simulator=5, run_log=None)
    assert list(with_options) == list(plain)
    assert "simulator" not in options  # the caller's dict is not mutated


def test_preflight_accepts_isa_and_rejects_logical_circuits():
    """The ISA and instruction-in-target checks catch an untranspiled circuit."""
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    from dgssp.runtime import preflight_sampler_job

    device = FakeManilaV2()
    logical = DGSSPSolver(DGConfig(iterations=1)).build_circuit(
        SubsetSumInstance(items=[1, 1], target=1)
    )
    isa = generate_preset_pass_manager(
        backend=device, optimization_level=1, seed_transpiler=0
    ).run(logical)

    good = preflight_sampler_job(device, [isa], shots=128, options={"twirling": {"enable_gates": True}})
    assert good["passed"], good["checks"]
    assert good["program_inputs"] is not None

    bad = preflight_sampler_job(device, [logical], shots=128)
    assert not bad["passed"]
    assert bad["checks"]["isa"] != "ok"
    assert bad["checks"]["instructions_in_target"] != "ok"

    wrong = preflight_sampler_job(device, [isa], shots=128,
                                  options={"twirling": {"num_randomizations": 0}})
    assert wrong["checks"]["options"] != "ok"
