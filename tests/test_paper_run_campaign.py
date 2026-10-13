"""
Tests for ``paper/scripts/run_campaign.py``.

No stage runs here: the launcher is replaced by one that writes stand-in
records where ``sim_campaign.py`` would, so what is tested is the
orchestration -- the gate, the refusals, the resume, the log and the sync.
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

pytest.importorskip("scipy")
run_campaign = pytest.importorskip("run_campaign")
sim_campaign = pytest.importorskip("sim_campaign")

COMMIT = "c" * 40


@pytest.fixture
def clean(monkeypatch):
    """A clean tree at COMMIT, the pinned environment, Mitiq importable."""
    import dgssp.mitigation

    monkeypatch.setattr(
        run_campaign,
        "git_state",
        lambda: {
            "commit": COMMIT,
            "short_commit": COMMIT[:7],
            "branch": "main",
            "dirty": False,
            "dirty_files": [],
        },
    )
    monkeypatch.setattr(run_campaign, "environment_drift", lambda: [])
    monkeypatch.setattr(dgssp.mitigation, "mitiq_is_available", lambda: True)


def _stamp(commit: str = COMMIT, dirty: bool = False) -> dict:
    """The provenance block a stand-in record carries."""
    return {
        "git": {"commit": commit, "dirty": dirty},
        "packages": run_campaign.package_versions(),
        "python": {"version": "3.11.16"},
    }


def _plan(data_dir: Path, keys=None, limit=3):
    return run_campaign.build_plan(
        keys or [s.key for s in sim_campaign.CAMPAIGN],
        data_dir=data_dir,
        noise="fake",
        limit=limit,
        instances_dir=REPO_ROOT / "paper" / "instances",
    )


def _write(stage, names=None, **stamp) -> None:
    """Records for ``names`` (default: all expected) in a stage directory."""
    stage.directory.mkdir(parents=True, exist_ok=True)
    for name in stage.expected if names is None else names:
        (stage.directory / f"{name}.json").write_text(
            json.dumps({"provenance": _stamp(**stamp)}), encoding="utf-8"
        )


class FakeLauncher:
    """Stands in for ``sim_campaign.py``: writes each stage's records."""

    def __init__(self, data_dir: Path, *, exit_codes=None, partial=()):
        self.plan = {tuple(s.spec.argv()): s for s in _plan(data_dir)}
        self.exit_codes = exit_codes or {}
        self.partial = set(partial)
        self.calls: list[str] = []

    def __call__(self, command: list[str]) -> int:
        stage = next(s for argv, s in self.plan.items() if command[2 : 2 + len(argv)] == list(argv))
        self.calls.append(stage.key)
        code = self.exit_codes.get(stage.key, 0)
        if code == 0:
            keep = stage.expected[:1] if stage.key in self.partial else stage.expected
            _write(stage, keep)
        return code


class FakeUploader:
    def __init__(self, status="ok"):
        self.status = status
        self.calls = 0

    def __call__(self, data_dir, url):
        self.calls += 1
        return {"status": self.status, "seconds": 0.1, "commands": 1, "error": None
                if self.status == "ok" else "denied"}


def _run(tmp_path, *extra, launcher=None, uploader=None):
    data = tmp_path / "data"
    launcher = launcher or FakeLauncher(data)
    code = run_campaign.main(
        ["--data-dir", str(data), "--limit", "3", *extra],
        launcher=launcher,
        uploader=uploader or FakeUploader(),
    )
    log_path = data / run_campaign.LOG_NAME
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else None
    return code, launcher, log


# ---------------------------------------------------------------------------
# Plan and dry run
# ---------------------------------------------------------------------------


def test_the_dry_run_prints_six_stages_and_counts_and_runs_nothing(tmp_path, clean, capsys):
    launcher = FakeLauncher(tmp_path / "data")
    code = run_campaign.main(
        ["--dry-run", "--data-dir", str(tmp_path / "data")], launcher=launcher
    )
    out = capsys.readouterr().out
    assert code == 0 and launcher.calls == []
    for key, directory, count in [
        ("ideal", "sim_ideal", 463),
        ("noisy", "sim_noisy_fake", 463),
        ("zne-odd", "sim_zne_fake_odd", 463),
        ("zne-planned", "sim_zne_fake_planned", 463),
        ("zne-odd7", "sim_zne_fake_odd7", 197),
        ("sweep", "sim_sweep_fake", 6),
    ]:
        assert any(key in line and directory in line and str(count) in line
                   for line in out.splitlines()), key
    assert "2055" in out
    assert not (tmp_path / "data" / run_campaign.LOG_NAME).exists()


def test_stage_commands_pass_the_throttles_through(tmp_path):
    args = run_campaign.parse_args(
        ["--max-threads", "8", "--cooldown", "5", "--chunk", "40", "--backend", "fake_x",
         "--noise", "fake", "--limit", "20", "--no-mitiq", "--data-dir", str(tmp_path)]
    )  # fmt: skip
    plan = _plan(tmp_path)
    zne = run_campaign.stage_command(plan[2], args)
    assert zne[2:8] == ["--stage", "zne", "--family", "all", "--grid", "odd"]
    text = " ".join(zne)
    for fragment in ("--max-threads 8", "--cooldown 5", "--chunk 40", "--backend fake_x",
                     "--noise fake", "--limit 20", "--no-mitiq"):
        assert fragment in text, fragment
    assert "--limit" not in run_campaign.stage_command(plan[5], args)  # the sweep


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_dirty_tree_is_refused(tmp_path, clean, monkeypatch):
    monkeypatch.setattr(
        run_campaign, "git_state",
        lambda: {"commit": COMMIT, "short_commit": "c", "branch": "main", "dirty": True,
                 "dirty_files": ["src/dgssp/encoding.py"]},
    )  # fmt: skip
    code, launcher, log = _run(tmp_path)
    assert code == run_campaign.REFUSED and launcher.calls == [] and log is None


def test_a_zne_stage_without_mitiq_is_refused_unless_asked_for(tmp_path, clean, monkeypatch):
    import dgssp.mitigation

    monkeypatch.setattr(dgssp.mitigation, "mitiq_is_available", lambda: False)
    code, launcher, _ = _run(tmp_path)
    assert code == run_campaign.REFUSED and launcher.calls == []

    # Ideal and noisy alone do not need it.
    code, _, _ = _run(tmp_path, "--stages", "ideal", "noisy")
    assert code == run_campaign.OK

    code, launcher, _ = _run(tmp_path, "--no-mitiq")
    assert code == run_campaign.OK
    assert "zne-odd" in launcher.calls


def test_environment_drift_is_refused(tmp_path, clean, monkeypatch):
    monkeypatch.setattr(run_campaign, "environment_drift", lambda: ["lock: numpy pinned 2.2.6"])
    code, launcher, _ = _run(tmp_path)
    assert code == run_campaign.REFUSED and launcher.calls == []


@pytest.mark.parametrize(
    "stamp", [{"commit": "d" * 40}, {"dirty": True}],
    ids=["another-commit", "dirty-records"],
)
def test_records_from_another_run_are_not_resumed_onto(tmp_path, clean, capsys, stamp):
    """One dataset is one commit and one clean tree; a mix is refused."""
    (ideal, *_rest) = _plan(tmp_path / "data")
    _write(ideal, ideal.expected[:2], **stamp)
    code, launcher, _ = _run(tmp_path)
    assert code == run_campaign.REFUSED and launcher.calls == []
    assert "sim_ideal: 2 record(s) from" in capsys.readouterr().out


def test_a_duplicate_record_is_refused(tmp_path, clean):
    (ideal, *_rest) = _plan(tmp_path / "data")
    _write(ideal)
    (ideal.directory / f"{ideal.expected[0]}_001.json").write_text(
        json.dumps({"provenance": _stamp()}), encoding="utf-8"
    )
    code, launcher, _ = _run(tmp_path)
    assert code == run_campaign.REFUSED and launcher.calls == []


def test_downstream_stages_need_the_gate(tmp_path, clean):
    code, launcher, _ = _run(tmp_path, "--stages", "noisy")
    assert code == run_campaign.REFUSED and launcher.calls == []

    (ideal, *_rest) = _plan(tmp_path / "data")
    _write(ideal)
    code, launcher, _ = _run(tmp_path, "--stages", "noisy")
    assert code == run_campaign.OK and launcher.calls == ["noisy"]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def test_a_full_run_runs_every_stage_once_in_order_and_logs_it(tmp_path, clean):
    code, launcher, log = _run(tmp_path)
    assert code == run_campaign.OK
    assert launcher.calls == [s.key for s in sim_campaign.CAMPAIGN]

    (run,) = log["runs"]
    assert log["schema"] == run_campaign.LOG_SCHEMA
    assert run["status"] == "complete"
    assert run["options"]["limit"] == 3 and run["options"]["chunk"] == sim_campaign.DEFAULT_CHUNK
    assert run["provenance"]["git"]["commit"]
    assert [s["key"] for s in run["stages"]] == launcher.calls
    for stage in run["stages"]:
        assert stage["status"] == "ok" and stage["exit_code"] == 0
        assert stage["records_before"] == 0 and stage["records_after"] == stage["expected"]
        assert stage["wall_s"] >= 0 and stage["started_utc"] and stage["finished_utc"]


def test_a_second_run_skips_what_is_complete_and_appends_to_the_log(tmp_path, clean):
    _run(tmp_path)
    code, launcher, log = _run(tmp_path)
    assert code == run_campaign.OK and launcher.calls == []
    assert len(log["runs"]) == 2
    assert {s["status"] for s in log["runs"][1]["stages"]} == {"complete"}


def test_a_failed_ideal_gate_stops_everything(tmp_path, clean):
    data = tmp_path / "data"
    launcher = FakeLauncher(data, exit_codes={"ideal": 1})
    code, launcher, log = _run(tmp_path, "--keep-going", launcher=launcher)
    assert code == run_campaign.STAGE_FAILED
    assert launcher.calls == ["ideal"]  # --keep-going never passes the gate
    statuses = [s["status"] for s in log["runs"][0]["stages"]]
    assert statuses == ["failed"] + ["blocked"] * 5
    assert log["runs"][0]["status"] == "failed"


def test_exit_zero_with_records_missing_is_a_failure(tmp_path, clean):
    launcher = FakeLauncher(tmp_path / "data", partial={"noisy"})
    code, launcher, log = _run(tmp_path, launcher=launcher)
    assert code == run_campaign.STAGE_FAILED
    assert launcher.calls == ["ideal", "noisy"]
    assert log["runs"][0]["stages"][1]["status"] == "incomplete"


def test_keep_going_continues_past_a_downstream_failure(tmp_path, clean):
    launcher = FakeLauncher(tmp_path / "data", exit_codes={"zne-odd": 1})
    code, launcher, log = _run(tmp_path, "--keep-going", launcher=launcher)
    assert code == run_campaign.STAGE_FAILED
    assert launcher.calls == [s.key for s in sim_campaign.CAMPAIGN]
    assert log["runs"][0]["failed_stages"] == ["zne-odd"]


def test_partial_writes_are_cleared_before_resuming(tmp_path, clean):
    (ideal, *_rest) = _plan(tmp_path / "data")
    ideal.directory.mkdir(parents=True)
    stray = ideal.directory / f"{ideal.expected[0]}.json.tmp"
    stray.write_text("{trunc", encoding="utf-8")
    code, _, _ = _run(tmp_path)
    assert code == run_campaign.OK and not stray.exists()


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


def test_upload_happens_first_after_every_stage_and_last(tmp_path, clean, monkeypatch):
    monkeypatch.setattr(run_campaign.shutil, "which", lambda name: "gcloud")
    uploader = FakeUploader()
    code, launcher, log = _run(tmp_path, "--upload", "gs://bucket-x/wp5", uploader=uploader)
    assert code == run_campaign.OK
    assert uploader.calls == 1 + len(launcher.calls) + 1
    assert log["runs"][0]["initial_upload"]["status"] == "ok"
    assert all(s["upload"]["status"] == "ok" for s in log["runs"][0]["stages"])


def test_a_long_stage_is_also_synced_while_it_runs(tmp_path, clean, monkeypatch):
    """
    zne-odd ran 4.3 h on the cloud VM; a sync only between stages would let a
    preemption take all of it.  While a stage runs, the bucket is synced
    every --upload-every minutes as well.
    """
    import time as _time

    monkeypatch.setattr(run_campaign.shutil, "which", lambda name: "gcloud")
    data = tmp_path / "data"
    inner = FakeLauncher(data)

    def slow(command):
        _time.sleep(0.5)
        return inner(command)

    uploader = FakeUploader()
    code, _, log = _run(
        tmp_path, "--upload", "gs://bucket-x/wp5", "--upload-every", "0.002",
        "--stages", "ideal", launcher=slow, uploader=uploader,
    )  # fmt: skip
    assert code == run_campaign.OK
    (stage,) = log["runs"][0]["stages"]
    assert stage["periodic_uploads"]["count"] >= 2
    assert stage["periodic_uploads"]["failed"] == 0
    assert uploader.calls == 1 + stage["periodic_uploads"]["count"] + 1 + 1  # first, periodic, after, last

    # Without --upload there is nothing to sync and no thread.
    _, _, log = _run(tmp_path / "plain", "--stages", "ideal")
    assert "periodic_uploads" not in log["runs"][0]["stages"][0]


def test_an_unwritable_bucket_stops_the_run_before_any_compute(tmp_path, clean, monkeypatch):
    monkeypatch.setattr(run_campaign.shutil, "which", lambda name: "gcloud")
    code, launcher, log = _run(
        tmp_path, "--upload", "gs://bucket-x/wp5", uploader=FakeUploader("failed")
    )
    assert code == run_campaign.UPLOAD_FAILED and launcher.calls == []
    assert log["runs"][0]["status"] == "upload_failed"


def test_a_malformed_bucket_url_is_refused(tmp_path, clean, monkeypatch):
    monkeypatch.setattr(run_campaign.shutil, "which", lambda name: "gcloud")
    code, launcher, _ = _run(tmp_path, "--upload", "s3://elsewhere")
    assert code == run_campaign.REFUSED and launcher.calls == []


def test_upload_syncs_only_campaign_outputs_and_retries(tmp_path, monkeypatch):
    """Tracked files of paper/data never go to the bucket; failures retry."""
    monkeypatch.setattr(run_campaign.shutil, "which", lambda name: "/usr/bin/gcloud")
    (tmp_path / "sim_ideal").mkdir()
    (tmp_path / "sim_noisy_fake").mkdir()
    (tmp_path / "smoke.json").write_text("{}", encoding="utf-8")  # tracked: not synced
    (tmp_path / run_campaign.LOG_NAME).write_text("{}", encoding="utf-8")

    class Done:
        def __init__(self, code):
            self.returncode, self.stdout, self.stderr = code, "", "403 denied"

    calls: list[list[str]] = []
    outcomes = iter([1, 0, 0, 0])  # the first rsync fails once, then all succeed
    sleeps: list[float] = []

    def run(command, **kwargs):
        calls.append(command)
        return Done(next(outcomes))

    result = run_campaign.upload(tmp_path, "gs://b/p/", run=run, sleep=sleeps.append)
    assert result["status"] == "ok" and result["commands"] == 3
    assert sleeps == [run_campaign.UPLOAD_BACKOFF[0]]
    targets = [command[-1] for command in calls]
    assert targets == ["gs://b/p/sim_ideal", "gs://b/p/sim_ideal", "gs://b/p/sim_noisy_fake",
                       f"gs://b/p/{run_campaign.LOG_NAME}"]  # fmt: skip
    assert all("smoke.json" not in " ".join(c) for c in calls)
    assert all(r".*\.tmp$" in c for c in calls if "rsync" in c)

    always_failing = run_campaign.upload(
        tmp_path, "gs://b/p", run=lambda c, **k: Done(1), sleep=lambda s: None
    )
    assert always_failing["status"] == "failed" and "403" in always_failing["error"]
