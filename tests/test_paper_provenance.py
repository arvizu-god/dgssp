"""
Tests for ``paper/scripts/_common.py``.

The paper's reproducibility claim reduces to two properties of this module:
every results file carries a filled provenance block, and no results file is
ever silently overwritten. Both are tested here rather than trusted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "paper" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

_common = pytest.importorskip("_common")

Record = _common.Record


@pytest.fixture
def record():
    """A minimal record with no provenance yet."""
    return Record(
        instance={"items": [1, 2, 3], "target": 5, "name": "t"},
        n_items=3,
        n_sum=4,
        iterations=2,
        executor="ideal",
        mitigation_mode="none",
        backend="aer_simulator",
        shots=1000,
        counts={"1": {"110": 955, "000": 45}},
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_has_every_field_the_paper_promises():
    """Commit, dirty flag, pinned versions, timestamp, host and Python."""
    provenance = _common.record_provenance()

    assert set(provenance) >= {
        "timestamp_utc",
        "git",
        "packages",
        "python",
        "platform",
        "host",
        "backend",
    }
    assert provenance["timestamp_utc"].endswith("Z")
    assert set(provenance["git"]) >= {"commit", "dirty", "branch"}
    assert set(provenance["packages"]) == set(_common.PINNED_PACKAGES)
    assert provenance["packages"]["qiskit"] is not None
    assert provenance["python"]["version"].count(".") == 2
    assert provenance["backend"] is None


def test_provenance_records_backend_and_calibration_when_given_one():
    """A device run is dated by the calibration snapshot it used."""
    pytest.importorskip("qiskit_ibm_runtime")
    from dgssp import list_fake_backends

    devices = list_fake_backends(heavy_hex_only=True)
    if not devices:
        pytest.skip("no heavy-hex fake devices installed")

    provenance = _common.record_provenance(devices[0])
    backend = provenance["backend"]

    assert backend["name"] == devices[0].name
    assert backend["heavy_hex"] is True
    assert backend["num_qubits"] == devices[0].num_qubits
    assert isinstance(backend["calibration_timestamp"], str)


def test_git_state_flags_a_dirty_tree_honestly():
    """``dirty`` is a bool inside a checkout; the flag is never silently absent."""
    state = _common.git_state()
    if state["commit"] is None:
        pytest.skip("not a git checkout")
    assert isinstance(state["dirty"], bool)
    assert len(state["commit"]) == 40


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def test_save_record_stamps_provenance_onto_a_bare_record(tmp_path, record):
    """A script physically cannot write an untraceable file."""
    path = _common.save_record(tmp_path / "r.json", record)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["provenance"]["packages"]["qiskit"] is not None
    assert payload["counts"]["1"]["110"] == 955
    assert payload["n_sum"] == 4


def test_save_record_never_overwrites(tmp_path, record):
    """A re-run adds a numeric suffix instead of destroying the previous result."""
    first = _common.save_record(tmp_path / "r.json", record)
    second = _common.save_record(tmp_path / "r.json", record)
    third = _common.save_record(tmp_path / "r.json", record)

    assert first.name == "r.json"
    assert second.name == "r_001.json"
    assert third.name == "r_002.json"
    assert first.exists() and second.exists() and third.exists()


def test_save_record_accepts_a_list_and_stamps_each_entry(tmp_path, record):
    """A batch is a list of records, each independently traceable."""
    path = _common.save_record(tmp_path / "batch.json", [record, record])
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(payload, list) and len(payload) == 2
    assert all(item["provenance"]["git"] for item in payload)


def test_load_records_flattens_files_and_tags_the_source(tmp_path, record):
    """Figure scripts get a flat table regardless of how runs were batched."""
    _common.save_record(tmp_path / "a.json", record)
    _common.save_record(tmp_path / "b.json", [record, record])

    loaded = _common.load_records(tmp_path / "*.json")

    assert len(loaded) == 3
    assert all(item["_source"].endswith(".json") for item in loaded)
    assert {Path(item["_source"]).name for item in loaded} == {"a.json", "b.json"}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_record_round_trips_through_json(record):
    """A record loaded back is the record that was saved."""
    restored = Record.from_dict(json.loads(json.dumps(record.to_dict())))
    assert restored == record


def test_record_from_dict_ignores_unknown_keys(record):
    """A file written by a later schema still loads."""
    payload = record.to_dict()
    payload["_source"] = "somewhere.json"
    payload["field_added_in_wp4"] = 42

    restored = Record.from_dict(payload)
    assert restored.n_items == 3


def test_record_from_instance_result_uses_the_library_encoding():
    """``n_sum`` comes from dgssp.encoding, not from a reimplementation here."""
    from dgssp import BatchConfig, SubsetSumInstance, build_encoding, run_batch

    instance = SubsetSumInstance(items=[3, -2, 4], target=1, name="neg")
    result = run_batch(
        [instance],
        BatchConfig(executor="ideal", shots=256, seed_simulator=1, seed_transpiler=1),
    ).results[0]

    built = Record.from_instance_result(
        result, executor="ideal", mitigation_mode="none", shots=256
    )

    assert built.n_sum == build_encoding([3, -2, 4], 1).n_sum
    assert built.n_items == 3
    assert built.instance == {"items": [3, -2, 4], "target": 1, "name": "neg"}
    assert list(built.counts) == ["1"]
    assert sum(built.counts["1"].values()) == 256
    assert set(built.transpiled) == {
        "two_q_count",
        "depth",
        "two_q_depth",
        "physical_qubits",
    }
    assert built.provenance["packages"]["qiskit"] is not None


def test_pinned_packages_and_lock_file_agree(tmp_path):
    """The lock file is generated from the same list the provenance uses."""
    path = _common.write_requirements_lock(tmp_path / "lock.txt")
    text = path.read_text(encoding="utf-8")

    versions = _common.package_versions()
    for name, version in versions.items():
        expected = f"{name}=={version}" if version else f"# {name} not installed"
        assert expected in text
