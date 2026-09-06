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

import getpass
import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def package_versions(
    packages: tuple[str, ...] = PINNED_PACKAGES,
) -> dict[str, str | None]:
    """
    Installed version of each pinned distribution.

    Parameters
    ----------
    packages:
        Distribution names, as they appear on PyPI.

    Returns
    -------
    dict[str, str | None]
        Version string per package; ``None`` when the package is not
        installed (Mitiq is an optional extra, so ``None`` is a legitimate
        value, not an error).
    """
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


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
    """Run a git command in the repo root, or return ``None`` if it fails."""
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
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    """
    Commit, branch and dirty flag of the working tree that produced a result.

    Returns
    -------
    dict
        ``{"commit", "short_commit", "branch", "dirty", "dirty_files"}``.
        ``commit`` is ``None`` outside a git checkout.  ``dirty`` is ``True``
        when tracked files differ from ``HEAD`` -- a dirty result is not
        reproducible from the commit alone, so this flag is what tells you a
        number needs re-running before it goes in the paper.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {
            "commit": None,
            "short_commit": None,
            "branch": None,
            "dirty": None,
            "dirty_files": [],
        }

    status = _git("status", "--porcelain") or ""
    dirty_files = [line[3:] for line in status.splitlines() if line.strip()]

    return {
        "commit": commit,
        "short_commit": commit[:7],
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
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
        "heavy_hex", "calibration_timestamp"}``.
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
class Record:
    """
    One executed configuration: an instance run one way, on one backend.

    A batch of instances becomes a *list* of records, not one record with
    lists inside it, so that a single row can always be traced to a single
    circuit execution.

    Attributes
    ----------
    instance:
        ``{"items", "target", "name"}`` -- the problem, verbatim.
    n_items:
        Index-register width (one qubit per item).
    n_sum:
        Sum-register width, from :func:`dgssp.encoding.build_encoding`
        (offset encoding plus the guard bit).
    iterations:
        Grover iterations actually applied.
    executor:
        Which path produced the counts: ``"ideal"``, ``"noisy"``,
        ``"optimized"`` or ``"hardware"``.
    mitigation_mode:
        ``"none"``, ``"zne_scalar"``, ``"zne_pointwise"`` or ``"mitiq"``.
    backend:
        Backend name the circuit ran on.
    shots:
        Shots per circuit, per noise scale.
    counts:
        Raw counts keyed by noise scale as a *string* (``"1"``, ``"3"``,
        ``"5"``).  An unmitigated run has the single key ``"1"``.  These are
        the only irrecoverable numbers in the file: everything else can be
        recomputed from them.
    metrics:
        Derived scalars (solution probability, mitigated probability, error
        accumulation, ...).  Free-form on purpose -- WP4 extends it.
    transpiled:
        ``{"two_q_count", "depth", "two_q_depth", "physical_qubits"}``, as
        produced by :func:`dgssp.transpilation.transpiled_metrics`.
    job_id:
        Runtime job id for a hardware run, else ``None``.
    provenance:
        Output of :func:`record_provenance`.
    """

    instance: dict[str, Any]
    n_items: int
    n_sum: int
    iterations: int
    executor: str
    mitigation_mode: str
    backend: str | None
    shots: int
    counts: dict[str, dict[str, int]]
    metrics: dict[str, Any] = field(default_factory=dict)
    transpiled: dict[str, Any] = field(default_factory=dict)
    job_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        JSON-serialisable view of the record.

        Returns
        -------
        dict
            All fields, with nothing dropped.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Record:
        """
        Rebuild a record from a loaded JSON object.

        Parameters
        ----------
        payload:
            A dictionary as written by :func:`save_record`.

        Returns
        -------
        Record
            Unknown keys are ignored, so a file written by a later version of
            the schema still loads.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    @classmethod
    def from_instance_result(
        cls,
        result: Any,
        *,
        executor: str,
        mitigation_mode: str = "none",
        shots: int,
        job_id: str | None = None,
        backend: Any | None = None,
        extra_metrics: dict[str, Any] | None = None,
    ) -> Record:
        """
        Convert a :class:`dgssp.experiments.InstanceResult` into a record.

        This is the only place the paper scripts translate library output into
        the archive schema, so a change to :class:`Record` never has to be
        chased through every script.

        Parameters
        ----------
        result:
            An ``InstanceResult`` from :func:`dgssp.experiments.run_batch`.
        executor:
            Label for the execution path (``"ideal"``, ``"noisy"``,
            ``"optimized"``, ``"hardware"``).
        mitigation_mode:
            Label for the mitigation applied.
        shots:
            Shots per circuit, per noise scale.
        job_id:
            Runtime job id, for hardware runs.
        backend:
            Calibrated backend object to stamp into the provenance block.
            Defaults to no backend information beyond ``result.backend_name``.
        extra_metrics:
            Additional scalars merged into ``metrics``.

        Returns
        -------
        Record
            Fully populated, with provenance already filled in.
        """
        from dgssp.encoding import build_encoding

        instance = result.instance
        encoding = build_encoding(list(instance.items), instance.target)

        if result.counts_per_scale:
            counts = {str(s): dict(c) for s, c in result.counts_per_scale.items()}
        else:
            counts = {"1": dict(result.counts)}

        metrics: dict[str, Any] = {
            "solution_bitstrings": list(result.solution_bitstrings),
            "num_solutions": result.num_solutions,
            "solution_probability": result.solution_probability,
            "mitigated_solution_probability": result.mitigated_solution_probability,
            "mitiq_solution_probability": result.mitiq_solution_probability,
            "n_qubits": result.n_qubits,
            "elapsed_s": result.elapsed_s,
        }
        if result.error_metrics is not None:
            metrics["error_metrics"] = result.error_metrics.to_dict()
        if result.mitigated_distribution is not None:
            metrics["mitigated_distribution"] = dict(result.mitigated_distribution)
        if extra_metrics:
            metrics.update(extra_metrics)

        return cls(
            instance={
                "items": list(instance.items),
                "target": instance.target,
                "name": instance.name,
            },
            n_items=instance.n_items,
            n_sum=encoding.n_sum,
            iterations=result.iterations,
            executor=executor,
            mitigation_mode=mitigation_mode,
            backend=result.backend_name,
            shots=shots,
            counts=counts,
            metrics=metrics,
            transpiled=result.transpiled_summary(),
            job_id=job_id,
            provenance=record_provenance(backend),
        )


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
    if isinstance(payload, Record):
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
    "PINNED_PACKAGES",
    "package_versions",
    "write_requirements_lock",
    "utc_timestamp",
    "git_state",
    "record_provenance",
    "Record",
    "save_record",
    "load_records",
]
