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


def test_git_state_records_dirty_paths_verbatim():
    """Every recorded path must exist on disk, character for character.

    Regression: ``_git`` used to ``strip()`` the whole porcelain output, which
    removed the leading space that ``git status --porcelain`` puts in front of
    an unstaged-only change (``" M path"``).  The fixed-width ``line[3:]`` slice
    then chopped the first character off the *first* dirty filename -- so every
    results file in ``paper/data`` carried a path that pointed nowhere.
    """
    state = _common.git_state()
    if state["commit"] is None:
        pytest.skip("not a git checkout")

    for relative in state["dirty_files"]:
        # Deletions and renames are reported too; a rename reads "old -> new".
        candidate = relative.split(" -> ")[-1].strip('"')
        assert not candidate.startswith(("/", " ")), candidate
        assert (_common.REPO_ROOT / candidate).parent.is_dir(), (
            f"{candidate!r} does not resolve under the repo root; the porcelain "
            "parse in git_state() has shifted again."
        )


def _fake_git(monkeypatch, *, status: str, untracked: str) -> list[tuple[str, ...]]:
    """
    Replace ``_git`` with a table lookup, and empty the per-process cache.

    A call ``git_state`` is not expected to make raises ``KeyError`` -- in
    particular a plain ``status --porcelain`` without ``--untracked-files=no``,
    which is the regression these tests guard.  The cache is reset through
    ``monkeypatch``, so the faked state is discarded when the test ends and
    never leaks into a test that probes the real checkout.

    Returns
    -------
    list
        Every call made, in order -- for asserting how often git ran.
    """
    responses = {
        ("rev-parse", "HEAD"): "a" * 40,
        ("rev-parse", "--abbrev-ref", "HEAD"): "main",
        ("status", "--porcelain", "--untracked-files=no"): status,
        ("ls-files", "--others", "--exclude-standard"): untracked,
    }
    calls: list[tuple[str, ...]] = []

    def fake(*args: str) -> str:
        calls.append(args)
        return responses[args]

    monkeypatch.setattr(_common, "_git", fake)
    monkeypatch.setattr(_common, "_GIT_STATE", None)
    return calls


def _no_outputs(**counts: int) -> dict[str, int]:
    """The per-directory count block, zero except where given."""
    block = {
        "paper/data": 0,
        "paper/figures": 0,
        "paper/instances": 0,
        "paper/hardware": 0,
        "paper/tex": 0,
    }
    for key, value in counts.items():
        block[f"paper/{key}"] = value
    return block


def test_untracked_files_do_not_make_the_tree_dirty(monkeypatch):
    """
    A pipeline's own untracked outputs must not taint the next file it stamps.

    Regression: ``git_state`` counted untracked files as dirty, contradicting
    its own docstring, so ``resources_fits.json`` -- stamped right after
    ``resources.json`` is written -- could never be clean.
    """
    _fake_git(
        monkeypatch,
        status="",
        untracked="paper/data/resources.json\npaper/instances/family.json\n",
    )
    state = _common.git_state()
    assert state["dirty"] is False
    assert state["dirty_files"] == []
    assert state["untracked_output_count"] == 2
    assert state["untracked_outputs_by_dir"] == _no_outputs(data=1, instances=1)
    assert "untracked_outputs" not in state


def test_untracked_code_still_makes_the_tree_dirty(monkeypatch):
    """
    An uncommitted new script is code the commit lacks: that *is* dirty.

    Only the output directories are exempt; a results file produced by a
    script that exists nowhere in git must never be stamped clean.
    """
    _fake_git(
        monkeypatch,
        status="",
        untracked="paper/scripts/new_analysis.py\nsrc/dgssp/new.py\npaper/data/out.json\n",
    )
    state = _common.git_state()
    assert state["dirty"] is True
    assert state["dirty_files"] == ["paper/scripts/new_analysis.py", "src/dgssp/new.py"]
    assert state["untracked_output_count"] == 1
    assert state["untracked_outputs_by_dir"] == _no_outputs(data=1)


def test_a_modified_tracked_file_still_makes_the_tree_dirty(monkeypatch):
    """The fix narrows the untracked rule; tracked modifications still count."""
    _fake_git(monkeypatch, status=" M src/dgssp/encoding.py", untracked="")
    state = _common.git_state()
    assert state["dirty"] is True
    assert state["dirty_files"] == ["src/dgssp/encoding.py"]
    assert state["untracked_output_count"] == 0
    assert state["untracked_outputs_by_dir"] == _no_outputs()


def test_a_sibling_directory_is_not_mistaken_for_an_output(monkeypatch):
    """``paper/data_old/`` shares a prefix with ``paper/data`` but is not one."""
    _fake_git(monkeypatch, status="", untracked="paper/data_old/x.json\n")
    state = _common.git_state()
    assert state["dirty_files"] == ["paper/data_old/x.json"]
    assert state["untracked_output_count"] == 0


def test_provenance_size_does_not_grow_with_the_files_already_written(monkeypatch):
    """
    The invariant the count exists for: record 2000 costs what record 1 did.

    Regression: the provenance block stored the full list of untracked
    outputs, and a campaign writes into those directories as it runs -- so
    every record embedded the names of all the records before it.  A late
    ZNE record carried 1179 paths (89 KB, of which 59 KB was the list).
    """
    sizes = {}
    for written in (0, 1, 50, 2000):
        names = "".join(f"paper/data/sim_zne_fake_odd/P-n6-{k:05d}.json\n" for k in range(written))
        _fake_git(monkeypatch, status="", untracked=names)
        sizes[written] = len(json.dumps(_common.record_provenance()))

    # Only the digits of the count may differ ("0" against "2000"), and the
    # count appears twice: the total and its directory's entry.  The list
    # this replaced would have added ~100 KB here.
    assert max(sizes.values()) - min(sizes.values()) <= 2 * len(str(2000)), sizes


def test_the_git_probe_runs_once_per_process(monkeypatch):
    """Two thousand records, one probe: the tree does not change mid-run."""
    calls = _fake_git(monkeypatch, status="", untracked="paper/data/a.json\n")
    first = _common.record_provenance()
    for _ in range(50):
        _common.record_provenance()
    assert len(calls) == 4  # rev-parse, status, ls-files, abbrev-ref: once each

    # Callers get copies, so one record's mutation cannot rewrite the next.
    first["git"]["dirty_files"].append("tampered")
    first["git"]["untracked_outputs_by_dir"]["paper/data"] = -1
    again = _common.git_state()
    assert again["dirty_files"] == []
    assert again["untracked_outputs_by_dir"]["paper/data"] == 1

    # An explicit refresh does probe again.
    _common.git_state(refresh=True)
    assert len(calls) == 8


def test_package_versions_are_looked_up_once_and_handed_out_as_copies():
    """The cache must not let one caller's edit leak into the next record."""
    versions = _common.package_versions()
    versions["qiskit"] = "tampered"
    assert _common.package_versions()["qiskit"] != "tampered"


def test_output_prefixes_are_the_paper_output_directories():
    """The exemption is exactly the five output directories, nothing broader."""
    assert set(_common._OUTPUT_PREFIXES) == {
        "paper/data/",
        "paper/figures/",
        "paper/instances/",
        "paper/hardware/",
        "paper/tex/",
    }


def test_live_output_summary_is_consistent():
    """
    In the real checkout, the counts add up and no untracked output is dirty.

    An untracked path is either counted as an output or listed as dirty,
    never both: no dirty path that git reports as untracked may lie in an
    output directory.
    """
    state = _common.git_state(refresh=True)
    if state["commit"] is None:
        pytest.skip("not a git checkout")
    by_dir = state["untracked_outputs_by_dir"]
    assert set(by_dir) == {p.rstrip("/") for p in _common._OUTPUT_PREFIXES}
    assert all(isinstance(v, int) and v >= 0 for v in by_dir.values())
    assert sum(by_dir.values()) == state["untracked_output_count"]

    untracked = set((_common._git("ls-files", "--others", "--exclude-standard") or "").splitlines())
    assert not {p for p in state["dirty_files"] if p in untracked and p.startswith(
        _common._OUTPUT_PREFIXES
    )}


# ---------------------------------------------------------------------------
# Files written before the count replaced the list
# ---------------------------------------------------------------------------

DATA_DIR = REPO_ROOT / "paper" / "data"


@pytest.mark.parametrize(
    "pattern", ["smoke*.json", "resources.json", "sim_*/*.json"]
)
def test_records_written_with_the_old_list_still_load(pattern):
    """
    ``smoke*``, ``resources.json`` and any WP5 records on disk carry the old
    ``untracked_outputs`` list; they must load and round-trip unchanged.

    ``sim_*`` is untracked, so it is checked when present and skipped in a
    fresh clone.
    """
    paths = sorted(DATA_DIR.glob(pattern))
    if not paths:
        pytest.skip(f"no {pattern} in paper/data")
    # Every file of a pattern, but at most 60 of a large stage directory.
    for path in paths[:: max(1, len(paths) // 60)]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload if isinstance(payload, list) else [payload]:
            record = Record.from_dict(item)
            assert record.provenance["git"]["commit"]
            assert Record.from_dict(record.to_dict()) == record


@pytest.mark.parametrize(
    "name",
    [
        "zne_synthetic.json",
        "zne_synthetic_001.json",
        "sweep_ideal.json",
        "resources_fits.json",
    ],
)
def test_payload_files_written_with_the_old_list_still_parse(name):
    """
    These are payload dicts, not ``ResultRecord`` rows, and never loaded
    through ``from_dict`` (it rejects them for missing ``instance`` etc.,
    as it always did).  What their readers rely on is ``provenance.git``.
    """
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not in paper/data")
    (loaded,) = _common.load_records(path)
    assert loaded["provenance"]["git"]["short_commit"]
    assert isinstance(loaded["provenance"]["git"]["dirty"], bool)


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


def test_save_record_never_leaves_a_truncated_record_under_its_name(
    tmp_path, record, monkeypatch
):
    """
    A process killed mid-write must not leave ``r.json`` behind: a resumed
    campaign would take a truncated record for finished work.
    """
    _common.save_record(tmp_path / "ok.json", record)
    assert not list(tmp_path.glob("*.tmp"))

    def killed(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(_common.os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        _common.save_record(tmp_path / "r.json", record)
    assert not (tmp_path / "r.json").exists()
    assert (tmp_path / "r.json.tmp").exists()  # what run_campaign clears


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
        "one_q_physical",
        "two_q_physical",
        "virtual_gate_count",
        "survival",
        "best_seed",
        "seed_source",
    }
    assert built.schema_version == 2
    assert built.metrics["solution_probability_ci"][0] <= built.metrics["solution_probability"]
    assert built.provenance["packages"]["qiskit"] is not None


def test_pinned_packages_and_lock_file_agree(tmp_path):
    """The lock file is generated from the same list the provenance uses."""
    path = _common.write_requirements_lock(tmp_path / "lock.txt")
    text = path.read_text(encoding="utf-8")

    versions = _common.package_versions()
    for name, version in versions.items():
        expected = f"{name}=={version}" if version else f"# {name} not installed"
        assert expected in text


# ---------------------------------------------------------------------------
# The cloud build input
# ---------------------------------------------------------------------------


def test_pins_are_read_by_normalised_name(tmp_path):
    """``Typing_Extensions`` and ``typing-extensions`` are one distribution."""
    path = tmp_path / "pins.txt"
    path.write_text(
        "# header: x\n\nTyping_Extensions==4.16.0\nqiskit==2.5.2  # trailing\n"
        "# mitiq not installed\n",
        encoding="utf-8",
    )
    assert _common.read_pins(path) == {"typing-extensions": "4.16.0", "qiskit": "2.5.2"}


def test_pin_drift_names_every_kind_of_difference():
    """Missing, extra and different versions are all failures, each named."""
    problems = _common.pin_drift(
        {"qiskit": "2.5.2", "mitiq": "1.1.0", "numpy": "2.2.6"},
        {"qiskit": "2.5.2", "mitiq": None, "NumPy": "2.3.0", "cirq_core": "1.6.1"},
    )
    assert problems == [
        "cirq-core==1.6.1 installed but not pinned",
        "mitiq==1.1.0 pinned but not installed",
        "numpy pinned 2.2.6, installed 2.3.0",
    ]
    assert _common.pin_drift({"qiskit": "2.5.2"}, {"Qiskit": "2.5.2"}) == []


def test_a_freeze_line_no_other_machine_can_install_is_refused(monkeypatch):
    """A conda build's ``name @ file:///...`` pins a path on this laptop."""

    class _Done:
        stdout = "numpy==2.2.6\nfoo @ file:///C:/conda/foo-1.0.tar.bz2\n"

    monkeypatch.setattr(_common.subprocess, "run", lambda *a, **k: _Done())
    with pytest.raises(ValueError, match="not installable elsewhere"):
        _common.pip_freeze()


def test_the_committed_cloud_pins_contain_the_lock():
    """
    The two committed files describe one environment: every lock pin appears
    in the cloud file at the same version -- Mitiq included, since a VM
    without it runs the campaign to completion with the C2 baseline missing.
    """
    cloud = _common.read_pins(_common.CLOUD_REQUIREMENTS_FILE)
    lock = _common.read_pins(_common.LOCK_FILE)
    assert {name: cloud.get(name) for name in lock} == lock
    assert {"mitiq", "cirq-core", "qiskit-aer", "rustworkx"} <= set(cloud)
    assert "dgssp" not in cloud  # installed editable from the checkout


def test_the_pin_file_header_is_readable(tmp_path):
    """``--verify`` compares the Python minor version against this header."""
    freeze_versions = pytest.importorskip("freeze_versions")
    path = tmp_path / "pins.txt"
    path.write_text(
        "# Some prose: with a colon\n# python:    3.11.16 (Windows-10)\n"
        "# pip:       26.2.1\n\nqiskit==2.5.2\n",
        encoding="utf-8",
    )
    header = freeze_versions.read_header(path)
    assert header["python"].startswith("3.11.16")
    assert header["pip"] == "26.2.1"
    assert "Some prose" not in header
