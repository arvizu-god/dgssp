"""
Tests for calibration snapshots: a live device's noise, frozen to JSON.

The property that matters for a hardware month is that the snapshot is a
*complete* stand-in for the device it came from: transpiling against it gives
the same ISA circuit, and sampling the simulator derived from it gives the
same counts, without the service being reachable.
"""

from __future__ import annotations

import json

import pytest

from dgssp import SubsetSumInstance
from dgssp.backends import (
    CALIBRATION_SCHEMA,
    backend_from_calibration,
    calibration_payload,
    calibration_timestamp,
    load_noise_model,
    median_gate_errors,
    save_noise_model,
)
from dgssp.solvers import DGConfig, DGSSPSolver


@pytest.fixture(scope="module")
def device():
    """A bundled heavy-hex fake device with a full calibration."""
    pytest.importorskip("qiskit_ibm_runtime")
    try:
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    except Exception:  # pragma: no cover - provider data varies by release
        pytest.skip("FakeMarrakesh unavailable")
    return FakeMarrakesh()


@pytest.fixture(scope="module")
def snapshot(device, tmp_path_factory):
    """The device saved and loaded back."""
    from qiskit_aer.noise import NoiseModel

    path = tmp_path_factory.mktemp("cal") / "calibration.json"
    save_noise_model(path, NoiseModel.from_backend(device), device)
    return load_noise_model(path)


def test_payload_is_plain_json_and_covers_every_instruction(device):
    """Nothing in the payload needs a custom encoder, and no gate is dropped."""
    payload = calibration_payload(device)
    json.dumps(payload)  # raises if a NumPy scalar or a Gate leaked in

    stored = set(payload["instructions"]) | set(payload["control_flow"])
    assert stored == set(device.target.operation_names)
    assert payload["num_qubits"] == device.num_qubits
    assert len(payload["qubit_properties"]) == device.num_qubits


def test_snapshot_reproduces_the_calibration_metrics(device, snapshot):
    """Medians, coupling map and calibration date survive the round trip."""
    assert snapshot.backend_name == device.name
    assert snapshot.calibration_timestamp == calibration_timestamp(device)
    assert calibration_timestamp(snapshot.backend) == calibration_timestamp(device)
    assert median_gate_errors(snapshot.backend) == median_gate_errors(device)
    assert snapshot.median_gate_errors == median_gate_errors(device)
    assert sorted(map(tuple, snapshot.backend.coupling_map)) == sorted(
        map(tuple, device.coupling_map)
    )


def test_snapshot_reproduces_transpilation_and_counts(device, snapshot):
    """The decisive property: same ISA circuit, same counts, no service."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator

    instance = SubsetSumInstance(items=[2, 1], target=3, name="snap")
    circuit = DGSSPSolver(DGConfig(iterations=1)).build_circuit(instance)

    live = generate_preset_pass_manager(
        backend=device, optimization_level=3, seed_transpiler=6
    ).run(circuit)
    frozen = generate_preset_pass_manager(
        backend=snapshot.backend, optimization_level=3, seed_transpiler=6
    ).run(circuit)
    assert live == frozen

    counts_live = (
        AerSimulator.from_backend(device)
        .run(live, shots=2000, seed_simulator=7)
        .result()
        .get_counts()
    )
    counts_frozen = (
        snapshot.simulator(seed_simulator=7)
        .run(frozen, shots=2000, seed_simulator=7)
        .result()
        .get_counts()
    )
    assert counts_live == counts_frozen


def test_noise_model_rebuilds_with_the_same_channels(device, snapshot):
    """The rebuilt model carries noise on the same instructions and qubits."""
    from qiskit_aer.noise import NoiseModel

    live = NoiseModel.from_backend(device)
    frozen = snapshot.noise_model
    assert sorted(frozen.noise_instructions) == sorted(live.noise_instructions)
    assert sorted(frozen.noise_qubits) == sorted(live.noise_qubits)
    assert snapshot.noise_model_summary["noise_instructions"] == sorted(
        live.noise_instructions
    )


def test_snapshot_is_evidence_and_is_not_clobbered(device, tmp_path):
    """A second save refuses; a snapshot is deleted deliberately."""
    path = tmp_path / "cal.json"
    save_noise_model(path, None, device)
    with pytest.raises(FileExistsError):
        save_noise_model(path, None, device)
    save_noise_model(path, None, device, overwrite=True)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == CALIBRATION_SCHEMA
    assert payload["noise_model_summary"] == {}


def test_a_snapshot_refuses_to_execute(snapshot):
    """It is calibration data, not a device."""
    with pytest.raises(NotImplementedError):
        snapshot.backend.run(None)


def test_loading_a_foreign_file_fails_loudly(tmp_path):
    """A JSON file that is not a snapshot is rejected by schema, not by luck."""
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"schema": "something/else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration-snapshot"):
        load_noise_model(path)


def test_rebuild_names_the_source_device(device):
    """The rebuilt backend can be named freely but remembers where it came from."""
    rebuilt = backend_from_calibration(
        calibration_payload(device), name="twin", source_name=device.name
    )
    assert rebuilt.name == "twin"
    assert rebuilt.source_name == device.name
    assert rebuilt.num_qubits == device.num_qubits


def test_opaque_device_instructions_survive_a_snapshot():
    """
    Current IBM targets (ibm_fez, fake_boston) advertise ``measure_2`` /
    ``reset_2`` as bare named instructions.  The snapshot keeps them by name and
    arity, so the rebuilt target and its noise model match the device's.
    """
    pytest.importorskip("qiskit_ibm_runtime")
    try:
        from qiskit_ibm_runtime.fake_provider import FakeBoston
    except Exception:  # pragma: no cover - provider data varies by release
        pytest.skip("FakeBoston unavailable")
    from qiskit_aer.noise import NoiseModel

    device = FakeBoston()
    payload = json.loads(json.dumps(calibration_payload(device)))
    assert payload["opaque_instructions"]["reset_2"] == {"num_qubits": 1, "num_clbits": 0}
    assert payload["opaque_instructions"]["measure_2"] == {"num_qubits": 1, "num_clbits": 1}

    rebuilt = backend_from_calibration(payload, name="fake_boston")
    assert sorted(rebuilt.target.operation_names) == sorted(device.target.operation_names)
    assert json.loads(json.dumps(calibration_payload(rebuilt))) == payload
    assert sorted(NoiseModel.from_backend(rebuilt).noise_instructions) == sorted(
        NoiseModel.from_backend(device).noise_instructions
    )


def test_an_instruction_with_meaning_beyond_its_name_is_still_refused(device):
    """A non-standard gate with parameters cannot be rebuilt from its name."""
    from qiskit.circuit import Gate, Parameter
    from qiskit.transpiler import Target

    target = Target(num_qubits=1)
    target.add_instruction(Gate("mystery", 1, [Parameter("t")]), {(0,): None})

    class Device:
        name = "toy"

        def __init__(self) -> None:
            self.target = target

    with pytest.raises(ValueError, match="no standard definition"):
        calibration_payload(Device())
