"""
run_campaign.py

The whole WP5 simulation campaign, unattended: every stage of
:data:`sim_campaign.CAMPAIGN`, in order, one ``sim_campaign.py`` process each.

::

    ideal        --stage ideal --family all                  463 records
    noisy        --stage noisy --family all                  463
    zne-odd      --stage zne   --family all --grid odd       463
    zne-planned  --stage zne   --family all --grid planned   463
    zne-odd7     --stage zne   --family P   --grid odd7      197
    sweep        --stage sweep                                 6

What it guarantees
------------------

**It refuses to start** (exit 2, nothing computed) when the dataset it would
produce could not be described as one pass of one commit in one environment:

* the tree is dirty (``--allow-dirty`` overrides, for development only --
  ``verify_campaign.py`` rejects dirty records);
* the interpreter differs from ``requirements-lock.txt`` or
  ``paper/cloud/requirements-cloud.txt`` (``--allow-env-drift`` overrides);
* a ZNE or sweep stage is planned and Mitiq is not importable, unless
  ``--no-mitiq`` says the baseline is being left out on purpose;
* a stage directory already holds records from another commit, a dirty tree
  or other package versions, or a duplicate record -- resuming onto those
  would splice two datasets into one;
* a downstream stage is asked for while the ideal stage, the acceptance gate,
  is not complete;
* ``--upload`` is given and the bucket cannot be written.

**The ideal stage is the gate.**  If it fails its acceptance check (it then
writes nothing) the run stops, whatever ``--keep-going`` says.  Any other
stage that fails, or exits 0 with records missing, stops the run too unless
``--keep-going`` is given.

**It resumes.**  A complete stage is skipped; an incomplete one resumes where
it stopped, and ``sim_campaign.py`` re-runs whole the chunks that still hold a
missing instance, so a run interrupted any number of times gives the same
counts as one that was not (:func:`sim_campaign.chunks_to_run`).

**It syncs.**  With ``--upload gs://BUCKET/PATH`` every ``sim_*`` directory and
the logs are pushed with ``gcloud storage rsync`` once before the first stage
(which proves the bucket is writable) and after every stage, pass or fail, so
a preempted Spot VM loses at most the stage it was in.  Only campaign outputs
go to the bucket -- never the tracked files of ``paper/data`` -- so syncing
the bucket back down cannot touch anything git tracks.

**It logs.**  Timestamped progress to stdout, and ``paper/data/
campaign_log.json``: one entry per invocation with its options, provenance,
and per stage the status, exit code, wall clock, records before and after,
and the upload's result.  The log is rewritten in place after every stage
(the list of runs only grows); it is the one file in ``paper/data`` that is.

Exit codes: ``0`` complete; ``1`` a stage failed; ``2`` refused to start;
``3`` an upload failed (the data is complete on local disk); ``130``
interrupted.

Run::

    python paper/scripts/run_campaign.py --dry-run
    python paper/scripts/run_campaign.py
    python paper/scripts/run_campaign.py --upload gs://my-bucket/wp5 --keep-going

    # a slice, into a scratch directory, without the long stages
    python paper/scripts/run_campaign.py --limit 20 --stages ideal noisy zne-planned \\
        --data-dir C:/tmp/slice
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

import sim_campaign  # noqa: E402
from _common import (  # noqa: E402
    DATA_DIR,
    HARDWARE_DIR,
    INSTANCES_DIR,
    REPO_ROOT,
    environment_drift,
    git_state,
    package_versions,
    record_provenance,
    utc_timestamp,
)

#: The machine-readable run log, in the data directory.
LOG_NAME = "campaign_log.json"
LOG_SCHEMA = "dgssp/paper/campaign-log/1"

#: Written by ``paper/cloud/startup.sh`` (a ``tee`` of this script's stdout);
#: uploaded with the data when present.
STDOUT_LOG_NAME = "campaign_stdout.log"

#: Exit codes.
OK, STAGE_FAILED, REFUSED, UPLOAD_FAILED, INTERRUPTED = 0, 1, 2, 3, 130

#: Seconds to wait before the second and third upload attempts.
UPLOAD_BACKOFF: tuple[float, ...] = (15.0, 60.0)

_BUCKET_URL = re.compile(r"^gs://[a-z0-9][a-z0-9._-]*[a-z0-9](/[^\s]*)?$")


def say(message: str) -> None:
    """Print one timestamped orchestrator line, flushed."""
    print(f"[{utc_timestamp()}] {message}", flush=True)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass
class PlannedStage:
    """
    One campaign stage, resolved against the instance files and a data dir.

    Attributes
    ----------
    spec:
        The stage (:data:`sim_campaign.CAMPAIGN`).
    directory:
        Where its records go.
    expected:
        Every record name it writes when complete.
    """

    spec: sim_campaign.CampaignStage
    directory: Path
    expected: list[str]

    @property
    def key(self) -> str:
        """The stage's short name."""
        return self.spec.key

    def status(self) -> tuple[list[str], list[str], list[str]]:
        """``(missing, present, duplicated)`` names, from disk."""
        if not self.directory.is_dir():
            return list(self.expected), [], []
        return sim_campaign.pending(self.directory, self.expected)

    def complete(self) -> bool:
        """Every expected record is on disk."""
        return not self.status()[0]


def build_plan(
    keys: Sequence[str],
    *,
    data_dir: Path,
    noise: str,
    limit: int | None,
    instances_dir: Path,
) -> list[PlannedStage]:
    """
    Resolve stage keys to directories and expected record names.

    Parameters
    ----------
    keys:
        Stage keys; run in :data:`sim_campaign.CAMPAIGN` order whatever order
        they are given in.
    data_dir, noise, limit, instances_dir:
        As passed to ``sim_campaign.py``.

    Returns
    -------
    list[PlannedStage]
        In campaign order.
    """
    family, _subset, sweep = sim_campaign.load_inputs(instances_dir)
    wanted = set(keys)
    return [
        PlannedStage(
            spec,
            spec.directory(data_dir, noise),
            sim_campaign.expected_names(spec, family, sweep, limit),
        )
        for spec in sim_campaign.CAMPAIGN
        if spec.key in wanted
    ]


def stage_command(stage: PlannedStage, args: argparse.Namespace) -> list[str]:
    """
    The ``sim_campaign.py`` command line for one stage.

    Parameters
    ----------
    stage:
        The stage.
    args:
        This script's options; the pass-through ones are forwarded.

    Returns
    -------
    list[str]
        Ready for ``subprocess``.
    """
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "sim_campaign.py"),
        *stage.spec.argv(),
        "--noise", args.noise,
        "--chunk", str(args.chunk),
        "--cooldown", f"{args.cooldown:g}",
        "--data-dir", str(args.data_dir),
        "--instances-dir", str(args.instances_dir),
        "--hardware-dir", str(args.hardware_dir),
    ]  # fmt: skip
    if args.backend:
        command += ["--backend", args.backend]
    if args.calibration:
        command += ["--calibration", str(args.calibration)]
    if args.max_threads:
        command += ["--max-threads", str(args.max_threads)]
    if args.limit and stage.spec.stage != "sweep":
        command += ["--limit", str(args.limit)]
    if args.no_mitiq:
        command += ["--no-mitiq"]
    return command


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def read_identity(path: Path) -> dict[str, Any] | None:
    """
    The commit, dirty flag and package versions a results file was stamped with.

    Parameters
    ----------
    path:
        A record or payload written by ``save_record``.

    Returns
    -------
    dict or None
        ``{"commit", "dirty", "packages", "python"}``; ``None`` when the file
        is not readable JSON.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    provenance = payload.get("provenance") or {}
    git = provenance.get("git") or {}
    return {
        "commit": git.get("commit"),
        "dirty": git.get("dirty"),
        "packages": provenance.get("packages"),
        "python": (provenance.get("python") or {}).get("version"),
    }


def foreign_records(
    plan: Sequence[PlannedStage],
    *,
    commit: str | None,
    packages: dict[str, str | None],
    allow_dirty: bool = False,
) -> list[str]:
    """
    Records already on disk that this run must not resume onto.

    Parameters
    ----------
    plan:
        The stages about to run.
    commit:
        The commit this run will stamp.
    packages:
        The pinned versions this run will stamp.
    allow_dirty:
        Accept dirty records (a development run, never a dataset).

    Returns
    -------
    list[str]
        One line per stage directory and reason; empty when every record
        could have come from this run.
    """
    problems = []
    for stage in plan:
        if not stage.directory.is_dir():
            continue
        reasons: Counter[str] = Counter()
        for path in sorted(stage.directory.glob("*.json")):
            identity = read_identity(path)
            if identity is None:
                reasons["unreadable JSON"] += 1
            elif identity["commit"] != commit:
                reasons[f"commit {str(identity['commit'])[:7]}"] += 1
            elif identity["dirty"] is not False and not allow_dirty:
                reasons["a dirty tree"] += 1
            elif identity["packages"] != packages:
                reasons["other package versions"] += 1
        for reason, count in sorted(reasons.items()):
            problems.append(f"{stage.directory.name}: {count} record(s) from {reason}")
        duplicated = stage.status()[2]
        if duplicated:
            problems.append(
                f"{stage.directory.name}: {len(duplicated)} duplicate record(s) "
                f"({duplicated[:3]}): two runs wrote into it at once"
            )
    return problems


def preflight(
    plan: Sequence[PlannedStage],
    args: argparse.Namespace,
    *,
    ideal: PlannedStage,
) -> tuple[list[str], list[str]]:
    """
    Every reason this run must not start, and what was checked.

    Parameters
    ----------
    plan:
        The stages about to run.
    args:
        Options.
    ideal:
        The ideal stage for the same selection -- the gate, checked even when
        it is not in ``plan``.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(problems, passed)``: human-readable lines.
    """
    problems: list[str] = []
    passed: list[str] = []

    git = git_state()
    if git["commit"] is None:
        problems.append("git: not a git checkout -- no record could name its commit")
    elif git["dirty"] and not args.allow_dirty:
        problems.append(
            f"git: the tree is dirty ({git['dirty_files'][:4]}); commit or stash -- "
            "dirty records are not reproducible and verify_campaign.py rejects them"
        )
    else:
        passed.append(
            f"git: {git['short_commit']} ({git['branch']})"
            f"{', DIRTY (allowed)' if git['dirty'] else ', clean'}"
        )

    if args.allow_env_drift:
        passed.append("environment: not checked (--allow-env-drift)")
    else:
        drift = environment_drift()
        if drift:
            problems.append(
                f"environment: {len(drift)} difference(s) from the pinned files, e.g. "
                f"{drift[:3]}; run freeze_versions.py --verify"
            )
        else:
            passed.append("environment: matches requirements-lock.txt and requirements-cloud.txt")

    needs_mitiq = any(stage.spec.stage in ("zne", "sweep") for stage in plan)
    if needs_mitiq and not args.no_mitiq:
        from dgssp.mitigation import mitiq_is_available

        if mitiq_is_available():
            passed.append("mitiq: importable")
        else:
            problems.append(
                "mitiq: not importable, and a ZNE/sweep stage would run without the "
                "Mitiq baseline; install it or pass --no-mitiq on purpose"
            )
    elif needs_mitiq:
        passed.append("mitiq: OFF by request (--no-mitiq); the C2 baseline will be absent")

    gated = [*plan, *([] if any(s.key == ideal.key for s in plan) else [ideal])]
    foreign = foreign_records(
        gated,
        commit=git["commit"],
        packages=package_versions(),
        allow_dirty=args.allow_dirty,
    )
    if foreign:
        problems += [f"data: {line}" for line in foreign]
        problems.append(
            "data: move those directories out of the repository before resuming; "
            "a dataset that spans two commits or environments is not one dataset"
        )
    else:
        passed.append("data: every record on disk could have come from this run")

    downstream = [stage.key for stage in plan if stage.key != "ideal"]
    if downstream and ideal.key not in {s.key for s in plan} and not ideal.complete():
        problems.append(
            f"gate: {downstream} need a complete ideal stage in {ideal.directory.name}; "
            "include 'ideal' in --stages"
        )

    if args.upload:
        if not _BUCKET_URL.match(args.upload):
            problems.append(f"upload: {args.upload!r} is not a gs://BUCKET/PATH URL")
        elif shutil.which("gcloud") is None:
            problems.append("upload: gcloud is not on PATH")
        else:
            passed.append(f"upload: {args.upload} (checked by the first sync)")

    if args.noise == "live" and args.calibration is None:
        passed.append(
            "noise: live without --calibration asks the IBM service for the device "
            "(needs credentials on this machine)"
        )
    return problems, passed


def clear_partial_writes(plan: Sequence[PlannedStage]) -> list[Path]:
    """
    Delete ``*.json.tmp`` left by a process killed mid-write.

    ``save_record`` writes beside the target and renames into place, so a
    ``.tmp`` file is by construction a record that was never finished; the
    instance it belonged to is still missing and will be run again.

    Returns
    -------
    list[Path]
        The files removed.
    """
    removed = []
    for stage in plan:
        if stage.directory.is_dir():
            for path in stage.directory.glob("*.json.tmp"):
                path.unlink()
                removed.append(path)
    return removed


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload(
    data_dir: Path,
    url: str,
    *,
    run: Callable[..., Any] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """
    Push every ``sim_*`` directory and the logs to ``url``.

    ``gcloud storage rsync`` uploads only what changed and never deletes.
    ``*.tmp`` (a record being written right now) is excluded.  Each command
    is tried up to three times.

    Parameters
    ----------
    data_dir:
        The campaign's data directory.
    url:
        ``gs://BUCKET/PATH``.
    run, sleep:
        Injected for tests.

    Returns
    -------
    dict
        ``{"status": "ok" | "failed", "seconds", "commands", "error"}``.
    """
    gcloud = shutil.which("gcloud") or "gcloud"
    base = url.rstrip("/")
    commands = [
        [gcloud, "storage", "rsync", "--recursive", "--exclude", r".*\.tmp$",
         str(path), f"{base}/{path.name}"]
        for path in sorted(data_dir.glob("sim_*"))
        if path.is_dir()
    ]  # fmt: skip
    commands += [
        [gcloud, "storage", "cp", str(data_dir / name), f"{base}/{name}"]
        for name in (LOG_NAME, STDOUT_LOG_NAME)
        if (data_dir / name).exists()
    ]

    started = time.time()
    error = None
    for command in commands:
        for attempt in range(1 + len(UPLOAD_BACKOFF)):
            done = run(command, capture_output=True, text=True)
            if done.returncode == 0:
                error = None
                break
            error = f"{' '.join(command[1:4])} ... exit {done.returncode}: " + (
                (done.stderr or done.stdout or "").strip()[-400:]
            )
            if attempt < len(UPLOAD_BACKOFF):
                sleep(UPLOAD_BACKOFF[attempt])
        if error:
            break
    return {
        "status": "failed" if error else "ok",
        "seconds": round(time.time() - started, 1),
        "commands": len(commands),
        "error": error,
    }


# ---------------------------------------------------------------------------
# The run log
# ---------------------------------------------------------------------------


class CampaignLog:
    """
    ``campaign_log.json``: every invocation, appended, rewritten atomically.

    Attributes
    ----------
    path:
        The log file.
    run:
        This invocation's entry (mutate it, then :meth:`save`).
    """

    def __init__(self, path: Path, run: dict[str, Any]) -> None:
        self.path = path
        self.run = run
        payload: dict[str, Any] = {"schema": LOG_SCHEMA, "runs": []}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != LOG_SCHEMA:
                raise ValueError(f"{path} is not a {LOG_SCHEMA} file")
        payload["runs"].append(run)
        self._payload = payload
        self.save()

    def save(self) -> None:
        """Write the whole log, via a temporary file and a rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(self.path.name + ".tmp")
        partial.write_text(json.dumps(self._payload, indent=2, default=str), encoding="utf-8")
        os.replace(partial, self.path)


# ---------------------------------------------------------------------------
# Running a stage
# ---------------------------------------------------------------------------


def launch(command: list[str], *, cwd: Path = REPO_ROOT) -> int:
    """
    Run one stage, streaming its output with a timestamp on every line.

    Parameters
    ----------
    command:
        From :func:`stage_command`.
    cwd:
        Working directory (the repo root, so ``runs/jobs.jsonl`` lands in the
        ignored ``runs/``).

    Returns
    -------
    int
        The stage's exit code.
    """
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as process:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(f"  {time.strftime('%H:%M:%S', time.gmtime())} | {line.rstrip()}",
                      flush=True)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        return process.wait()


def _raise_interrupt(signum: int, frame: Any) -> None:
    """SIGTERM (a VM shutting down) takes the same path as Ctrl-C."""
    raise KeyboardInterrupt(f"signal {signum}")


def run_stages(
    plan: Sequence[PlannedStage],
    args: argparse.Namespace,
    log: CampaignLog,
    *,
    launcher: Callable[[list[str]], int] = launch,
    uploader: Callable[[Path, str], dict[str, Any]] = upload,
) -> int:
    """
    Run the plan in order, logging and syncing after every stage.

    Parameters
    ----------
    plan:
        The stages.
    args:
        Options.
    log:
        The run log; this invocation's entry is ``log.run``.
    launcher, uploader:
        Injected for tests.

    Returns
    -------
    int
        An exit code (see the module docstring).
    """
    status = OK
    upload_failed = False
    for number, stage in enumerate(plan, start=1):
        missing, present, _ = stage.status()
        entry: dict[str, Any] = {
            "key": stage.key,
            "directory": stage.directory.name,
            "expected": len(stage.expected),
            "records_before": len(present),
        }
        log.run["stages"].append(entry)
        header = f"stage {number}/{len(plan)} {stage.key} -> {stage.directory.name}"

        if status != OK:
            entry.update(status="blocked", records_after=len(present))
            say(f"{header}: not run (an earlier stage failed)")
            log.save()
            continue
        if not missing:
            entry.update(status="complete", records_after=len(present))
            say(f"{header}: complete ({len(present)}/{len(stage.expected)}), skipping")
            log.save()
            continue

        command = stage_command(stage, args)
        entry.update(status="running", started_utc=utc_timestamp(), command=command[1:])
        log.save()
        say(f"{header}: {len(missing)} to run, {len(present)} on disk")
        say(f"  {' '.join(command[1:])}")
        started = time.time()
        try:
            exit_code = launcher(command)
        except KeyboardInterrupt:
            entry.update(status="interrupted", wall_s=round(time.time() - started, 1))
            raise

        missing_after, present_after, duplicated = stage.status()
        if exit_code != 0:
            outcome = "failed"
        elif missing_after:
            outcome = "incomplete"
        else:
            outcome = "ok"
        entry.update(
            status=outcome,
            exit_code=exit_code,
            finished_utc=utc_timestamp(),
            wall_s=round(time.time() - started, 1),
            records_after=len(present_after),
            duplicates=len(duplicated),
        )
        say(
            f"{header}: {outcome.upper()} (exit {exit_code}) in "
            f"{entry['wall_s'] / 60:.1f} min, {len(present_after)}/{len(stage.expected)} records"
        )
        if outcome != "ok":
            if stage.key == "ideal":
                say("  the ideal stage is the acceptance gate: nothing downstream is "
                    "meaningful, stopping")
                status = STAGE_FAILED
            elif not args.keep_going:
                status = STAGE_FAILED
            else:
                say("  --keep-going: continuing with the remaining stages")
                log.run["failed_stages"] = [*log.run.get("failed_stages", []), stage.key]
        log.save()

        if args.upload:
            result = uploader(args.data_dir, args.upload)
            entry["upload"] = result
            upload_failed = result["status"] != "ok"
            say(f"  upload: {result['status']} in {result['seconds']}s"
                + (f" -- {result['error']}" if result["error"] else ""))
            log.save()

    if status == OK and log.run.get("failed_stages"):
        status = STAGE_FAILED
    if status == OK and upload_failed:
        status = UPLOAD_FAILED
    return status


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv:
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    keys = [stage.key for stage in sim_campaign.CAMPAIGN]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and the pre-flight checks; run nothing.")
    parser.add_argument("--stages", nargs="+", choices=keys, default=keys,
                        help="Run only these stages (always in campaign order).")
    parser.add_argument("--noise", choices=("fake", "live"), default="fake")
    parser.add_argument("--backend", default=None,
                        help="Backend name; defaults to hardware_subset.json's.")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Calibration snapshot for --noise live.")
    parser.add_argument("--chunk", type=int, default=sim_campaign.DEFAULT_CHUNK,
                        help="Circuits per job. Part of the result: keep it fixed.")
    parser.add_argument("--max-threads", type=int, default=None)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N instances of each stage (a slice).")
    parser.add_argument("--no-mitiq", action="store_true")
    parser.add_argument("--upload", default=None, metavar="gs://BUCKET/PATH",
                        help="Sync the campaign outputs here before and after every stage.")
    parser.add_argument("--keep-going", action="store_true",
                        help="Continue past a failed stage (never past the ideal gate).")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Run from a dirty tree (development only).")
    parser.add_argument("--allow-env-drift", action="store_true",
                        help="Skip the pinned-environment check (development only).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--instances-dir", type=Path, default=INSTANCES_DIR)
    parser.add_argument("--hardware-dir", type=Path, default=HARDWARE_DIR)
    return parser.parse_args(argv)


def print_plan(plan: Sequence[PlannedStage], args: argparse.Namespace, backend: str) -> None:
    """The stage table: what runs, where it lands, how many records."""
    print(
        f"campaign: {len(plan)} stage(s)  noise={args.noise}  backend={backend}  "
        f"chunk={args.chunk}  threads={args.max_threads or 'all'}  "
        f"cooldown={args.cooldown:g}s  limit={args.limit or 'none'}  "
        f"mitiq={'off' if args.no_mitiq else 'on'}  upload={args.upload or 'none'}"
    )
    print(f"data: {args.data_dir}\n")
    print(f"  {'#':>2}  {'stage':<12} {'directory':<22} {'expected':>8} {'on disk':>8}  arguments")
    total = on_disk = 0
    for number, stage in enumerate(plan, start=1):
        present = len(stage.status()[1])
        total += len(stage.expected)
        on_disk += present
        print(
            f"  {number:>2}  {stage.key:<12} {stage.directory.name:<22} "
            f"{len(stage.expected):>8} {present:>8}  {' '.join(stage.spec.argv())}"
        )
    print(f"  {'':>2}  {'total':<12} {'':<22} {total:>8} {on_disk:>8}\n")


def main(
    argv: list[str] | None = None,
    *,
    launcher: Callable[[list[str]], int] = launch,
    uploader: Callable[[Path, str], dict[str, Any]] = upload,
) -> int:
    """
    Run (or, with ``--dry-run``, describe) the campaign.

    Returns
    -------
    int
        An exit code (see the module docstring).
    """
    args = parse_args(argv)
    # Stages run with the repo root as their working directory.
    for name in ("data_dir", "instances_dir", "hardware_dir"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    _family, subset, _sweep = sim_campaign.load_inputs(args.instances_dir)
    backend = args.backend or subset["backend"]

    def plan_for(keys: Sequence[str]) -> list[PlannedStage]:
        return build_plan(
            keys,
            data_dir=args.data_dir,
            noise=args.noise,
            limit=args.limit,
            instances_dir=args.instances_dir,
        )

    plan = plan_for(args.stages)
    (ideal,) = plan_for(["ideal"])
    print_plan(plan, args, backend)

    problems, passed = preflight(plan, args, ideal=ideal)
    print("pre-flight:")
    for line in passed:
        print(f"  ok    {line}")
    for line in problems:
        print(f"  FAIL  {line}")
    print()

    if args.dry_run:
        print("dry run: nothing executed." + (" A real run would refuse to start."
                                             if problems else ""))
        return REFUSED if problems else OK
    if problems:
        say(f"refusing to start: {len(problems)} problem(s) above")
        return REFUSED

    for path in clear_partial_writes(plan):
        say(f"removed a partial write: {path}")

    log = CampaignLog(
        args.data_dir / LOG_NAME,
        {
            "started_utc": utc_timestamp(),
            "status": "running",
            "argv": sys.argv[1:] if argv is None else list(argv),
            "options": {
                "stages": [stage.key for stage in plan],
                "noise": args.noise,
                "backend": backend,
                "calibration": str(args.calibration) if args.calibration else None,
                "chunk": args.chunk,
                "max_threads": args.max_threads,
                "cooldown": args.cooldown,
                "limit": args.limit,
                "mitiq": not args.no_mitiq,
                "upload": args.upload,
                "keep_going": args.keep_going,
            },
            "plan": [
                {"key": s.key, "directory": s.directory.name, "expected": len(s.expected)}
                for s in plan
            ],
            "provenance": record_provenance(),
            "stages": [],
        },
    )

    if args.upload:
        first = uploader(args.data_dir, args.upload)
        log.run["initial_upload"] = first
        say(f"initial sync to {args.upload}: {first['status']} in {first['seconds']}s")
        if first["status"] != "ok":
            say(f"  {first['error']}")
            log.run.update(status="upload_failed", finished_utc=utc_timestamp())
            log.save()
            return UPLOAD_FAILED

    try:
        previous = signal.signal(signal.SIGTERM, _raise_interrupt)
    except (ValueError, OSError):  # pragma: no cover - not the main thread
        previous = None
    try:
        status = run_stages(plan, args, log, launcher=launcher, uploader=uploader)
    except KeyboardInterrupt:
        say("interrupted")
        log.run.update(status="interrupted", finished_utc=utc_timestamp())
        log.save()
        if args.upload:
            uploader(args.data_dir, args.upload)
        return INTERRUPTED
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)

    names = {OK: "complete", STAGE_FAILED: "failed", UPLOAD_FAILED: "upload_failed"}
    log.run.update(status=names[status], finished_utc=utc_timestamp())
    log.save()
    if args.upload:
        uploader(args.data_dir, args.upload)  # the log's final state
    say(f"campaign {names[status]} (exit {status}); log: {log.path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
