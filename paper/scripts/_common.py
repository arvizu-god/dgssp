"""
_common.py

Provenance, result schema and file I/O for every paper script.

The rule this module enforces is that a results file is self-describing: given
only ``paper/data/foo.json``, you can recover the git commit that produced it,
whether the tree was dirty at the time, which package versions were installed,
when and where it ran, and -- for anything touching a device -- which backend
and which calibration snapshot.  Nothing in ``paper/data/`` is written by hand,
so nothing in it is untraceable.

Nothing here re-derives library logic.  Register widths come from
:mod:`dgssp.encoding`, circuit size metrics from
:func:`dgssp.transpilation.transpiled_metrics`, calibration timestamps from
:func:`dgssp.backends.calibration_timestamp`.

Usage
-----
::

    from _common import Record, record_provenance, save_record, load_records

    record = Record.from_instance_result(result, executor="noisy",
                                         mitigation_mode="none", shots=1000)
    save_record(DATA_DIR / "foo.json", record)
"""

from __future__ import annotations

import copy
import getpass
import json
import platform
import re
import socket
import subprocess
import sys
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dgssp.records import ResultRecord

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: ``paper/scripts``.
SCRIPTS_DIR = Path(__file__).resolve().parent

#: ``paper``.
PAPER_ROOT = SCRIPTS_DIR.parent

#: The repository root (the directory holding ``pyproject.toml``).
REPO_ROOT = PAPER_ROOT.parent

DATA_DIR = PAPER_ROOT / "data"
FIGURES_DIR = PAPER_ROOT / "figures"
INSTANCES_DIR = PAPER_ROOT / "instances"
HARDWARE_DIR = PAPER_ROOT / "hardware"
NOTES_DIR = PAPER_ROOT / "notes"
TEX_DIR = PAPER_ROOT / "tex"

#: Where :func:`write_requirements_lock` writes the pinned versions.
LOCK_FILE = PAPER_ROOT / "requirements-lock.txt"

#: Where :func:`write_cloud_requirements` writes the full environment a cloud
#: VM installs (``paper/cloud/bootstrap.sh``).
CLOUD_REQUIREMENTS_FILE = PAPER_ROOT / "cloud" / "requirements-cloud.txt"

#: Directories that hold pipeline *outputs*.  An untracked file here is a
#: result, not code, and does not make the tree dirty (see :func:`git_state`).
OUTPUT_DIRS: tuple[Path, ...] = (DATA_DIR, FIGURES_DIR, INSTANCES_DIR, HARDWARE_DIR, TEX_DIR)

#: :data:`OUTPUT_DIRS` as repo-relative POSIX prefixes, the form git prints.
_OUTPUT_PREFIXES: tuple[str, ...] = tuple(
    d.relative_to(REPO_ROOT).as_posix() + "/" for d in OUTPUT_DIRS
)


# ---------------------------------------------------------------------------
# Pinned dependency set
# ---------------------------------------------------------------------------

#: Distributions whose versions are recorded in every results file and written
#: to ``paper/requirements-lock.txt``.  These are the packages whose behaviour
#: can move a number in the paper: the circuit builder, the transpiler, the
#: noise model, the mitigation baseline, the numerics and the plotting.
PINNED_PACKAGES: tuple[str, ...] = (
    "qiskit",
    "qiskit-ibm-runtime",
    "qiskit-aer",
    "mitiq",
    "numpy",
    "scipy",
    "matplotlib",
)


#: Per-process memo of :func:`package_versions`, keyed by the package tuple.
_PACKAGE_VERSIONS: dict[tuple[str, ...], dict[str, str | None]] = {}


def package_versions(
    packages: tuple[str, ...] = PINNED_PACKAGES,
) -> dict[str, str | None]:
    """
    Installed version of each pinned distribution.

    Looked up once per process and per ``packages`` tuple: the modules that
    ran are the ones imported when the process started, so a mid-run
    ``pip install`` would make a fresh lookup *less* accurate, not more --
    and the lookup costs 10 ms, paid once per record by a campaign that
    writes thousands of them.

    Parameters
    ----------
    packages:
        Distribution names, as they appear on PyPI.

    Returns
    -------
    dict[str, str | None]
        Version string per package; ``None`` when the package is not
        installed (Mitiq is an optional extra, so ``None`` is a legitimate
        value, not an error).  A fresh copy: callers may mutate it.
    """
    from importlib.metadata import PackageNotFoundError, version

    key = tuple(packages)
    if key not in _PACKAGE_VERSIONS:
        versions: dict[str, str | None] = {}
        for name in key:
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                versions[name] = None
        _PACKAGE_VERSIONS[key] = versions
    return dict(_PACKAGE_VERSIONS[key])


def write_requirements_lock(path: Path | str = LOCK_FILE) -> Path:
    """
    Write the installed versions of :data:`PINNED_PACKAGES` as ``name==version``.

    Parameters
    ----------
    path:
        Destination file; parent directories are created.

    Returns
    -------
    Path
        The path written.  Unlike :func:`save_record` this file *is* rewritten
        in place -- it is a snapshot of the current environment, not a result.
    """
    versions = package_versions()
    git = git_state()

    lines = [
        "# Pinned environment for Paper A (dgssp).",
        "# Regenerate with: python paper/scripts/freeze_versions.py",
        f"# generated: {utc_timestamp()}",
        f"# python:    {platform.python_version()} ({platform.platform()})",
        f"# commit:    {git['commit']}{' (dirty)' if git['dirty'] else ''}",
        "",
    ]
    for name, ver in versions.items():
        lines.append(f"{name}=={ver}" if ver else f"# {name} not installed")
    lines.append("")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def normalise_name(name: str) -> str:
    """A distribution name in PEP 503 form (``Typing_Extensions`` -> ``typing-extensions``)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pip_freeze() -> list[str]:
    """
    ``pip freeze --exclude-editable`` of the running interpreter.

    The editable install is ``dgssp`` itself, which a VM installs from its
    pinned checkout rather than from a requirements file.

    Returns
    -------
    list[str]
        One ``name==version`` line per distribution, in pip's order.

    Raises
    ------
    ValueError
        If a line is not a plain ``name==version`` pin -- a ``name @ file://``
        line (what a conda-built package can produce) names a path on this
        machine, and no other machine can install it.
    """
    out = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    pins = [
        line.strip()
        for line in out.stdout.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    unpinnable = [p for p in pins if "==" not in p or " @ " in p]
    if unpinnable:
        raise ValueError(
            f"{len(unpinnable)} distribution(s) are not installable elsewhere: "
            f"{unpinnable[:5]}. Reinstall them from PyPI with pip first."
        )
    return pins


def write_cloud_requirements(path: Path | str = CLOUD_REQUIREMENTS_FILE) -> Path:
    """
    Write the full environment, every distribution pinned, for a cloud VM.

    :data:`LOCK_FILE` pins the seven packages that can move a number and is
    the provenance record.  This file pins *everything* (``rustworkx``,
    ``cirq-core`` and the rest), because that is what a fresh machine has to
    install to reproduce the environment rather than merely resemble it.

    Parameters
    ----------
    path:
        Destination file; parent directories are created.

    Returns
    -------
    Path
        The path written (rewritten in place, like the lock file).
    """
    from importlib.metadata import version

    pins = pip_freeze()
    git = git_state()
    lines = [
        "# Build input for the cloud campaign VM (paper/cloud/bootstrap.sh): the full",
        "# `pip freeze --exclude-editable` of the environment of record.  dgssp itself is",
        "# installed editable from the pinned checkout, not from this file.",
        "# paper/requirements-lock.txt (7 packages) is the provenance record; this file",
        "# is what reproduces it on another machine.",
        "# Regenerate with: python paper/scripts/freeze_versions.py",
        f"# generated: {utc_timestamp()}",
        f"# python:    {platform.python_version()} ({platform.platform()})",
        f"# pip:       {version('pip')}",
        f"# commit:    {git['commit']}{' (dirty)' if git['dirty'] else ''}",
        "",
        *pins,
        "",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def read_pins(path: Path | str) -> dict[str, str]:
    """
    The ``name==version`` lines of a requirements file, keyed by normalised name.

    Comments (including the lock file's ``# mitiq not installed``) and blank
    lines are skipped.

    Parameters
    ----------
    path:
        A file written by :func:`write_requirements_lock` or
        :func:`write_cloud_requirements`.

    Returns
    -------
    dict[str, str]
        ``{normalised name: version}``.
    """
    pins: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        name, _, ver = text.partition("==")
        if not ver:
            raise ValueError(f"{path}: not a pin: {line!r}")
        pins[normalise_name(name)] = ver.strip()
    return pins


def pin_drift(expected: dict[str, str], installed: dict[str, str | None]) -> list[str]:
    """
    Every way an installed environment differs from a set of pins.

    Parameters
    ----------
    expected:
        Pins, as :func:`read_pins` returns them.
    installed:
        ``{name: version}`` actually present; ``None`` means not installed.
        Names are normalised here.

    Returns
    -------
    list[str]
        One line per difference -- missing, extra or a different version --
        empty when the two agree exactly.
    """
    have = {normalise_name(k): v for k, v in installed.items() if v is not None}
    problems = []
    for name in sorted(set(expected) | set(have)):
        want, got = expected.get(name), have.get(name)
        if want is None:
            problems.append(f"{name}=={got} installed but not pinned")
        elif got is None:
            problems.append(f"{name}=={want} pinned but not installed")
        elif want != got:
            problems.append(f"{name} pinned {want}, installed {got}")
    return problems


def environment_drift(
    lock: Path | str = LOCK_FILE,
    cloud: Path | str | None = CLOUD_REQUIREMENTS_FILE,
) -> list[str]:
    """
    How the running interpreter differs from the lock file and the cloud pins.

    Parameters
    ----------
    lock:
        :data:`LOCK_FILE`: the seven packages recorded in every result.
    cloud:
        :data:`CLOUD_REQUIREMENTS_FILE`, checked against the full
        ``pip freeze``; ``None`` skips it (it costs a ``pip`` subprocess).

    Returns
    -------
    list[str]
        Differences, each prefixed with the file it came from; empty when the
        environment is exactly the one of record.
    """
    problems = [f"lock: {p}" for p in pin_drift(read_pins(lock), package_versions())]
    if cloud is not None:
        frozen = dict(line.split("==", 1) for line in pip_freeze())
        problems += [f"cloud: {p}" for p in pin_drift(read_pins(cloud), dict(frozen))]
    return problems


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def utc_timestamp() -> str:
    """
    Current UTC time, ISO-8601 with a ``Z`` suffix and second precision.

    Returns
    -------
    str
        e.g. ``"2026-09-01T14:03:11Z"``.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _git(*args: str) -> str | None:
    """
    Run a git command in the repo root, or return ``None`` if it fails.

    Only *trailing* whitespace is stripped.  ``git status --porcelain`` encodes
    the status in the first two columns and pads unstaged-only changes with a
    leading space (``" M path"``), so an ``lstrip`` would silently shift the
    first line's path by one character -- which is exactly what it used to do.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.rstrip()


#: The working tree as this process first saw it (see :func:`git_state`).
_GIT_STATE: dict[str, Any] | None = None


def git_state(*, refresh: bool = False) -> dict[str, Any]:
    """
    Commit, branch and dirty flag of the working tree that produced a result.

    The probe runs **once per process** and is reused for every record the
    process stamps.  The code that ran is the code the process imported when
    it started, so the tree as it stood then is the state that describes the
    results; re-probing per record cost 95 ms each (four ``git`` calls
    scanning a directory the campaign itself keeps filling), about three
    minutes over a full campaign, and could only ever add noise -- such as a
    record flagged dirty because someone edited a file the running process
    had long since imported.  ``probed_utc`` says when the snapshot was taken.

    Parameters
    ----------
    refresh:
        Probe again and replace the cached state.

    Returns
    -------
    dict
        ``{"commit", "short_commit", "branch", "dirty", "dirty_files",
        "untracked_output_count", "untracked_outputs_by_dir", "probed_utc"}``,
        a fresh copy the caller may mutate.  ``commit`` is ``None`` outside a
        git checkout.  ``dirty`` is ``True`` when the code that ran is not
        exactly the code in ``HEAD`` -- a dirty result is not reproducible
        from the commit alone, so this flag is what tells you a number needs
        re-running before it goes in the paper.

        That means: any modified tracked file, and any untracked file *outside*
        the paper's output directories (:data:`OUTPUT_DIRS`) -- an uncommitted
        new script or module is code the commit does not contain.  Untracked
        files *inside* the output directories do not count.  They are results,
        not code, and counting them made clean results impossible: a stage that
        writes ``resources.json`` and then stamps ``resources_fits.json`` would
        always see its own first output as a dirty file.

        Untracked outputs are **counted, not listed**: a total, and the same
        count per output directory (all five keys, always present).  The list
        used to be stored verbatim, and since a campaign writes into those
        directories as it runs, every record embedded the names of all the
        records before it -- O(N^2) bytes, 60-73% of the WP5 data volume, and
        a provenance block that grew with the run.  Nothing ever read the
        list: untracked *inputs* stay traceable through the hashes downstream
        files carry (``family_payload_sha256``), and the per-directory counts
        still say whether an input directory (``paper/instances``,
        ``paper/hardware``) held anything untracked.  The block is now a
        fixed size.  A digest of the list was considered and rejected: no
        one keeps the list it would be checked against, and mid-campaign it
        fingerprints the campaign's own progress, not its inputs.
    """
    global _GIT_STATE
    if _GIT_STATE is None or refresh:
        _GIT_STATE = _probe_git()
    return copy.deepcopy(_GIT_STATE)


def _probe_git() -> dict[str, Any]:
    """The uncached probe behind :func:`git_state`."""
    by_dir = {prefix.rstrip("/"): 0 for prefix in _OUTPUT_PREFIXES}
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {
            "commit": None,
            "short_commit": None,
            "branch": None,
            "dirty": None,
            "dirty_files": [],
            "untracked_output_count": 0,
            "untracked_outputs_by_dir": by_dir,
            "probed_utc": utc_timestamp(),
        }

    # Porcelain v1: two status columns, a space, then the path.  Tracked files
    # only; untracked files are classified separately below.
    status = _git("status", "--porcelain", "--untracked-files=no") or ""
    dirty_files = [line[3:] for line in status.splitlines() if line[3:].strip()]

    # `ls-files` prints repo-relative POSIX paths, one per line, unquoted.
    untracked = _git("ls-files", "--others", "--exclude-standard") or ""
    for path in (line for line in untracked.splitlines() if line.strip()):
        prefix = next((p for p in _OUTPUT_PREFIXES if path.startswith(p)), None)
        if prefix is None:
            dirty_files.append(path)
        else:
            by_dir[prefix.rstrip("/")] += 1

    return {
        "commit": commit,
        "short_commit": commit[:7],
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "untracked_output_count": sum(by_dir.values()),
        "untracked_outputs_by_dir": by_dir,
        "probed_utc": utc_timestamp(),
    }


def record_provenance(backend: Any | None = None) -> dict[str, Any]:
    """
    Build the provenance block stamped onto every results file.

    Parameters
    ----------
    backend:
        Optional backend the run executed against.  Pass the *calibrated*
        backend (a real device or a bundled fake device) rather than the
        ``AerSimulator`` derived from it: an Aer simulator carries the noise
        model but not the calibration date.

    Returns
    -------
    dict
        ``{"timestamp_utc", "git", "packages", "python", "platform", "host",
        "user", "backend"}``.  ``backend`` is ``None`` when no backend was
        passed, otherwise ``{"name", "num_qubits", "processor_family",
        "heavy_hex", "calibration_timestamp"}``.  ``git`` and ``packages`` are
        probed once per process (:func:`git_state`, :func:`package_versions`),
        so the block is the same size for the first record of a run and the
        two-thousandth, and costs no subprocess after the first.
    """
    provenance: dict[str, Any] = {
        "timestamp_utc": utc_timestamp(),
        "git": git_state(),
        "packages": package_versions(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "user": _current_user(),
        "backend": None,
    }

    if backend is not None:
        from dgssp.backends import (
            calibration_timestamp,
            is_heavy_hex,
            processor_family,
        )
        from dgssp.runtime import backend_name

        provenance["backend"] = {
            "name": backend_name(backend),
            "num_qubits": getattr(backend, "num_qubits", None),
            "processor_family": processor_family(backend),
            "heavy_hex": is_heavy_hex(backend),
            "calibration_timestamp": calibration_timestamp(backend),
        }

    return provenance


def _current_user() -> str | None:
    """Login name, or ``None`` where the platform will not say."""
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - unusual environments
        return None


# ---------------------------------------------------------------------------
# Results schema
# ---------------------------------------------------------------------------


@dataclass
class Record(ResultRecord):
    """
    :class:`dgssp.records.ResultRecord` plus provenance stamping.

    The schema itself -- every field, the compatibility rules, the strict-JSON
    serialisation -- lives in the library (:mod:`dgssp.records`), because the
    library produces records (:meth:`dgssp.experiments.InstanceResult.
    to_record`) and cannot import this module.  This subclass adds nothing but
    the provenance block, which needs the git and environment probes that only
    the paper workspace has.  Old files (``smoke*.json``) load through
    :meth:`from_dict` unchanged: unknown keys are ignored and new keys take
    their defaults.
    """

    @classmethod
    def from_instance_result(
        cls,
        result: Any,
        *,
        executor: str,
        mitigation_mode: str | None = None,
        shots: int,
        job_id: str | None = None,
        backend: Any | None = None,
        extra_metrics: dict[str, Any] | None = None,
    ) -> Record:
        """
        Convert a :class:`dgssp.experiments.InstanceResult` into a stamped record.

        Parameters
        ----------
        result:
            An ``InstanceResult`` from :func:`dgssp.experiments.run_batch`.
        executor:
            Label for the execution path (``"ideal"``, ``"noisy"``,
            ``"optimized"``, ``"hardware"``).
        mitigation_mode:
            Label for the mitigation applied; defaults to the result's own.
        shots:
            Shots per circuit, per noise scale.
        job_id:
            Runtime job id; defaults to the one the result carries.
        backend:
            Calibrated backend object to stamp into the provenance block (pass
            the fake/real device, not the Aer simulator derived from it).
            Its calibration timestamp also fills ``calibration_timestamp``
            when the result carries none.
        extra_metrics:
            Additional entries merged into ``metrics``.

        Returns
        -------
        Record
            Fully populated, with provenance already filled in.
        """
        base = result.to_record(
            executor=executor,
            shots=shots,
            mitigation_mode=mitigation_mode,
            job_id=job_id,
            extra_metrics=extra_metrics,
        )
        record = cls(**{f.name: getattr(base, f.name) for f in fields(ResultRecord)})
        record.provenance = record_provenance(backend)
        stamped = record.provenance.get("backend") or {}
        if record.calibration_timestamp is None:
            record.calibration_timestamp = stamped.get("calibration_timestamp")
        return record


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _next_free_path(path: Path) -> Path:
    """First non-existent variant of ``path``, adding ``_001``, ``_002``, ..."""
    if not path.exists():
        return path
    for n in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{n:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Refusing to write: 1000 variants of {path} already exist.")


def save_record(path: Path | str, payload: Any) -> Path:
    """
    Write a record (or a list of records) to JSON, never overwriting.

    If the destination exists, a numeric suffix is appended
    (``smoke.json`` -> ``smoke_001.json``) and the new path is returned.  A
    results file is evidence: it is deleted deliberately, never clobbered by a
    re-run.

    Parameters
    ----------
    path:
        Intended destination.  Parent directories are created.
    payload:
        A :class:`Record`, a sequence of records, or any JSON-serialisable
        object.  Records without a provenance block get one stamped on here,
        so a script physically cannot write an untraceable file.

    Returns
    -------
    Path
        The path actually written, which may differ from ``path``.
    """
    obj = _serialisable(payload)

    target = _next_free_path(Path(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return target


def _serialisable(payload: Any) -> Any:
    """Normalise a payload to plain JSON types, filling missing provenance."""
    if isinstance(payload, ResultRecord):
        if not payload.provenance:
            payload.provenance = record_provenance()
        return payload.to_dict()

    if isinstance(payload, (list, tuple)):
        return [_serialisable(item) for item in payload]

    if isinstance(payload, dict):
        out = dict(payload)
        out.setdefault("provenance", record_provenance())
        return out

    return payload


def load_records(pattern: Path | str) -> list[dict[str, Any]]:
    """
    Load every JSON file matching a glob, newest-sorted by path.

    Parameters
    ----------
    pattern:
        A glob.  A bare name (``"smoke*.json"``) is resolved inside
        ``paper/data``; anything containing a separator is used as given.

    Returns
    -------
    list[dict]
        One entry per record.  Files holding a *list* of records are
        flattened, so ``load_records("zne_*.json")`` gives a flat table
        regardless of how the runs were batched into files.  Each entry gains
        a ``"_source"`` key naming the file it came from.
    """
    from glob import glob

    text = str(pattern)
    if "/" not in text and "\\" not in text:
        text = str(DATA_DIR / text)

    records: list[dict[str, Any]] = []
    for filename in sorted(glob(text)):
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict):
                item["_source"] = filename
                records.append(item)
    return records


__all__ = [
    "SCRIPTS_DIR",
    "PAPER_ROOT",
    "REPO_ROOT",
    "DATA_DIR",
    "FIGURES_DIR",
    "INSTANCES_DIR",
    "HARDWARE_DIR",
    "NOTES_DIR",
    "TEX_DIR",
    "LOCK_FILE",
    "CLOUD_REQUIREMENTS_FILE",
    "OUTPUT_DIRS",
    "PINNED_PACKAGES",
    "package_versions",
    "write_requirements_lock",
    "normalise_name",
    "pip_freeze",
    "write_cloud_requirements",
    "read_pins",
    "pin_drift",
    "environment_drift",
    "utc_timestamp",
    "git_state",
    "record_provenance",
    "ResultRecord",
    "Record",
    "save_record",
    "load_records",
]
