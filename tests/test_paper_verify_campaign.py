"""
Tests for ``paper/scripts/verify_campaign.py``.

A complete, valid three-instance slice of all six stages is written as
stand-in records; it must pass.  Every other test breaks exactly one thing
about it and expects a failure that names it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from dgssp.records import scale_key

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "paper" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("scipy")
verify = pytest.importorskip("verify_campaign")
sim_campaign = pytest.importorskip("sim_campaign")

COMMIT = "e" * 40
PACKAGES = {"qiskit": "2.5.2", "qiskit-aer": "0.17.2", "mitiq": "1.1.0"}
SEED = sim_campaign.SEED_SIMULATOR


def _provenance(**git) -> dict:
    return {
        "git": {"commit": COMMIT, "dirty": False, **git},
        "packages": dict(PACKAGES),
        "python": {"version": "3.11.2"},
        "host": "dgssp-campaign",
        "timestamp_utc": "2026-10-04T10:00:00Z",
    }


def _record(stage: str, grid: str | None, name: str) -> dict:
    """A minimal record as sim_campaign writes it for one stage."""
    shots = verify.configured_shots(stage)
    scales = [1.0, 3.0, 5.0] if stage == "zne" else [1.0]
    counts = {scale_key(s): {"00": shots - 1, "11": 1} for s in scales}
    campaign = {
        "stage": stage,
        "grid": grid,
        "noise": "none" if stage == "ideal" else "fake",
        "seed_simulator": SEED,
        "shots": shots,
    }
    if stage == "zne":
        campaign.update(nominal_scales=scales, include_mitiq=True)
    return {
        "schema_version": 2,
        "instance": {"name": name, "items": [1], "target": 1},
        "shots": shots,
        "counts": counts,
        "metrics": {"campaign": campaign},
        "mitiq": {"value": 0.5} if stage == "zne" else None,
        "provenance": _provenance(),
    }


def _sweep_payload(instance: str, executor: str, iterations: list[int]) -> dict:
    shots = verify.configured_shots("sweep", executor)
    scales = ["1", "3", "5"] if executor == "zne" else ["1"]
    return {
        "schema": verify.SWEEP_SCHEMA,
        "instance": {"name": instance},
        "executor": executor,
        "campaign": {"sweep_executor": executor, "seed_simulator": SEED},
        "rows": [
            {
                "iterations": r,
                "shots": shots,
                "counts": {k: {"0": shots} for k in scales},
                "mitiq": {"value": 0.4} if executor == "zne" else None,
            }
            for r in iterations
        ],
        "provenance": _provenance(),
    }


@pytest.fixture
def slice_dir(tmp_path):
    """A complete, valid --limit 3 campaign, and a lock that matches it."""
    data = tmp_path / "data"
    plan = verify.build_plan(
        [s.key for s in sim_campaign.CAMPAIGN],
        data_dir=data,
        noise="fake",
        limit=3,
        instances_dir=REPO_ROOT / "paper" / "instances",
    )
    _f, _s, sweep = sim_campaign.load_inputs(REPO_ROOT / "paper" / "instances")
    iterations = {row["name"]: row["iterations"] for row in sweep["instances"]}
    for stage in plan:
        stage.directory.mkdir(parents=True)
        for name in stage.expected:
            if stage.spec.stage == "sweep":
                instance, _, executor = name.rpartition("_")
                payload = _sweep_payload(instance, executor, iterations[instance])
            else:
                payload = _record(stage.spec.stage, stage.spec.grid, name)
            (stage.directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    lock = tmp_path / "lock.txt"
    lock.write_text("".join(f"{k}=={v}\n" for k, v in PACKAGES.items()), encoding="utf-8")
    return data, lock


def _verify(slice_dir, *extra) -> int:
    data, lock = slice_dir
    return verify.main(["--data-dir", str(data), "--limit", "3", "--lock", str(lock), *extra])


def _edit(path: Path, change) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _first(data: Path, directory: str) -> Path:
    return sorted((data / directory).glob("*.json"))[0]


def test_a_complete_campaign_passes(slice_dir, capsys):
    assert _verify(slice_dir) == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    for directory in ("sim_ideal", "sim_noisy_fake", "sim_zne_fake_odd",
                      "sim_zne_fake_planned", "sim_zne_fake_odd7", "sim_sweep_fake"):
        assert directory in out
    assert "KB/rec" in out and "MB" in out  # sizes on disk


@pytest.mark.parametrize(
    "breakage, message",
    [
        (lambda d: _first(d, "sim_noisy_fake").unlink(), "1 missing"),
        (lambda d: (_first(d, "sim_ideal").with_name(_first(d, "sim_ideal").stem + "_001.json")
                    ).write_text("{}", encoding="utf-8"), "duplicate"),
        (lambda d: (d / "sim_ideal" / "Z-n9-M0-00.json").write_text("{}", encoding="utf-8"),
         "not in the family"),
        (lambda d: (d / "sim_ideal" / "x.json.tmp").write_text("{", encoding="utf-8"),
         "partial write"),
        (lambda d: _first(d, "sim_ideal").write_text("{trunc", encoding="utf-8"), "unreadable"),
    ],
    ids=["missing", "duplicate", "extra", "partial", "unreadable"],
)
def test_what_is_on_disk_must_be_exactly_the_plan(slice_dir, capsys, breakage, message):
    breakage(slice_dir[0])
    assert _verify(slice_dir) == 1
    assert message in capsys.readouterr().out


@pytest.mark.parametrize(
    "directory, change, message",
    [
        ("sim_ideal", lambda p: p["provenance"]["git"].update(commit="f" * 40), "2 commits"),
        ("sim_zne_fake_odd", lambda p: p["provenance"]["git"].update(dirty=True), "dirty"),
        ("sim_noisy_fake", lambda p: p["provenance"]["packages"].update(qiskit="2.6.0"),
         "different package sets"),
        ("sim_noisy_fake", lambda p: p["provenance"]["python"].update(version="3.12.1"),
         "python"),
    ],
    ids=["two-commits", "dirty", "two-environments", "two-pythons"],
)
def test_one_dataset_is_one_commit_and_one_environment(
    slice_dir, capsys, directory, change, message
):
    _edit(_first(slice_dir[0], directory), change)
    assert _verify(slice_dir) == 1
    assert message in capsys.readouterr().out


@pytest.mark.parametrize(
    "directory, change, check",
    [
        ("sim_ideal", lambda p: p.update(schema_version=1), "failed schema"),
        ("sim_ideal", lambda p: p["counts"]["1"].update({"00": 0}), "failed shots"),
        ("sim_noisy_fake", lambda p: p.update(shots=8192), "failed shots"),
        ("sim_zne_fake_planned", lambda p: p["counts"].pop("5"), "failed shots"),
        ("sim_zne_fake_odd", lambda p: p.update(mitiq=None), "failed mitiq"),
        ("sim_zne_fake_odd", lambda p: p["metrics"]["campaign"].update(grid="planned"),
         "failed identity"),
        ("sim_ideal", lambda p: p["instance"].update(name="P-n2-M0-99"), "failed identity"),
        ("sim_noisy_fake", lambda p: p["metrics"]["campaign"].update(seed_simulator=1),
         "failed config"),
        ("sim_sweep_fake", lambda p: p["rows"][0].update(shots=100), "failed shots"),
        ("sim_sweep_fake", lambda p: p.update(schema="something/else"), "failed schema"),
    ],
    ids=["schema", "count-sum", "record-shots", "missing-scale", "no-mitiq",
         "misplaced", "wrong-name", "seed", "sweep-shots", "sweep-schema"],  # fmt: skip
)
def test_every_record_is_what_its_directory_says(slice_dir, capsys, directory, change, check):
    _edit(_first(slice_dir[0], directory), change)
    assert _verify(slice_dir) == 1
    assert check in capsys.readouterr().out


def test_a_deliberate_no_mitiq_run_can_still_pass(slice_dir):
    data = slice_dir[0]
    for path in (data / "sim_zne_fake_odd").glob("*.json"):
        _edit(path, lambda p: p.update(mitiq=None))
    assert _verify(slice_dir) == 1
    assert _verify(slice_dir, "--no-mitiq") == 0


def test_the_packages_must_be_the_locked_ones(slice_dir, tmp_path, capsys):
    data, _ = slice_dir
    other = tmp_path / "other_lock.txt"
    other.write_text("qiskit==2.0.2\nqiskit-aer==0.17.2\nmitiq==1.1.0\n", encoding="utf-8")
    assert verify.main(["--data-dir", str(data), "--limit", "3", "--lock", str(other)]) == 1
    assert "differ from other_lock.txt" in capsys.readouterr().out
    assert verify.main(
        ["--data-dir", str(data), "--limit", "3", "--lock", str(other), "--no-lock-check"]
    ) == 0


def test_the_launch_commit_can_be_asserted(slice_dir):
    assert _verify(slice_dir, "--commit", COMMIT[:7]) == 0
    assert _verify(slice_dir, "--commit", "abc1234") == 1


def test_a_slice_can_be_checked_stage_by_stage(slice_dir):
    """The local acceptance slice runs three stages, not six."""
    data = slice_dir[0]
    for directory in ("sim_zne_fake_odd", "sim_zne_fake_odd7", "sim_sweep_fake"):
        for path in (data / directory).iterdir():
            path.unlink()
    assert _verify(slice_dir) == 1
    assert _verify(slice_dir, "--stages", "ideal", "noisy", "zne-planned") == 0
