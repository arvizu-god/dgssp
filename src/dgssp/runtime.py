"""
runtime.py

Everything that talks to IBM Quantum: service creation, execution modes,
sampling and job bookkeeping.

This module is the *only* place in the library that constructs a
``QiskitRuntimeService`` or a ``SamplerV2``.  Centralising it has three
consequences that matter for real-hardware work:

* **No import-time side effects.**  Importing :mod:`dgssp` never touches the
  network and never rewrites your saved credentials.  Saving an account is an
  explicit, opt-in call to :func:`save_account`.
* **One batching primitive.**  :func:`sample_counts` accepts a *list* of
  circuits and submits them as a single job, so a whole sweep costs one queue
  wait instead of one per circuit.
* **Recoverable runs.**  Every submission is appended to a JSON-lines log with
  its job id, so a run interrupted mid-queue can be recovered later with
  :func:`fetch_result` instead of being re-executed.

The legacy ``ibm_quantum`` channel has been retired by IBM; this module uses
``ibm_quantum_platform`` with a token and an instance CRN.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

from qiskit import QuantumCircuit

logger = logging.getLogger("dgssp.runtime")

#: Default location of the job-id ledger, relative to the working directory.
DEFAULT_RUN_LOG = "runs/jobs.jsonl"

ExecutionModeName = Literal["auto", "backend", "session", "batch"]


# ---------------------------------------------------------------------------
# Service creation
# ---------------------------------------------------------------------------


def get_service(
    *,
    token: str | None = None,
    instance: str | None = None,
    name: str | None = None,
) -> Any:
    """
    Build a ``QiskitRuntimeService`` without side effects.

    Parameters
    ----------
    token:
        IBM Quantum Platform API token.  If given, an explicit service is
        constructed from it (nothing is written to disk).  If ``None``, the
        saved default account is loaded instead.
    instance:
        Instance CRN (or name) to scope the service to.  Only used together
        with ``token``.
    name:
        Name of a saved account to load, when not passing an explicit token.

    Returns
    -------
    QiskitRuntimeService
        A live service handle.

    Notes
    -----
    This function never calls ``save_account``.  Use :func:`save_account`
    explicitly if you want credentials persisted.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    if token is not None:
        return QiskitRuntimeService(
            channel="ibm_quantum_platform", token=token, instance=instance
        )
    return QiskitRuntimeService(name=name)


def save_account(
    token: str,
    instance: str | None = None,
    *,
    name: str | None = None,
    set_as_default: bool = True,
    overwrite: bool = True,
) -> None:
    """
    Persist IBM Quantum credentials to the local qiskit configuration.

    This is the opt-in counterpart to :func:`get_service`; nothing in the
    library calls it implicitly.

    Parameters
    ----------
    token:
        IBM Quantum Platform API token.
    instance:
        Instance CRN to associate with the saved account.
    name:
        Optional name under which to save the account.
    set_as_default:
        Whether the saved account becomes the default.
    overwrite:
        Whether to replace an existing entry with the same name.

    Returns
    -------
    None
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
        name=name,
        set_as_default=set_as_default,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# Backend introspection
# ---------------------------------------------------------------------------


def is_simulator(backend: Any) -> bool:
    """
    Best-effort test for whether a backend is a simulator.

    Prefers BackendV2 signals (Aer class name, ``target.description``) and
    falls back to the deprecated ``configuration().simulator`` flag only if
    nothing else is available.

    Parameters
    ----------
    backend:
        A Qiskit backend.

    Returns
    -------
    bool
        ``True`` for Aer / fake / simulated backends.
    """
    cls_name = type(backend).__name__.lower()
    if "aer" in cls_name or "simulator" in cls_name or "statevector" in cls_name:
        return True

    name = backend_name(backend).lower()
    if "simulator" in name or name.startswith("aer"):
        return True

    cfg = getattr(backend, "configuration", None)
    if callable(cfg):  # pragma: no cover - legacy V1 backends only
        try:
            return bool(getattr(cfg(), "simulator", False))
        except Exception:
            return False
    return False


def backend_name(backend: Any) -> str:
    """
    Read a backend's name across the V1 (method) / V2 (attribute) split.

    Parameters
    ----------
    backend:
        A Qiskit backend.

    Returns
    -------
    str
        The backend name, or ``str(backend)`` if it exposes none.
    """
    attr = getattr(backend, "name", None)
    if callable(attr):
        return str(attr())
    return str(attr) if attr else str(backend)


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------


@contextmanager
def execution_mode(
    backend: Any, *, mode: ExecutionModeName = "auto"
) -> Iterator[Any]:
    """
    Yield an object that ``SamplerV2`` accepts as its ``mode``.

    Simulators are yielded as-is (sessions are pointless locally).  Real
    hardware is wrapped in a ``Batch`` by default, which is the right container
    for a set of *independent* circuits: they share one queue reservation
    without the serialisation a ``Session`` imposes.

    Parameters
    ----------
    backend:
        The backend to execute on.
    mode:
        ``"auto"`` (simulator -> plain backend, hardware -> ``Batch``),
        ``"backend"`` (never wrap), ``"batch"`` or ``"session"``.

    Yields
    ------
    Any
        A backend, ``Batch`` or ``Session`` suitable for ``SamplerV2(mode=...)``.
    """
    if mode == "backend" or (mode == "auto" and is_simulator(backend)):
        yield backend
        return

    if mode == "session":
        from qiskit_ibm_runtime import Session

        with Session(backend=backend) as session:
            yield session
        return

    from qiskit_ibm_runtime import Batch

    with Batch(backend=backend) as batch:
        yield batch


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_counts(
    mode_or_backend: Any,
    circuits: QuantumCircuit | Sequence[QuantumCircuit],
    *,
    shots: int = 10_000,
    seed_simulator: int | None = None,
    run_log: str | None = DEFAULT_RUN_LOG,
    meta: dict[str, Any] | None = None,
) -> list[dict[str, int]]:
    """
    Run one or many circuits through ``SamplerV2`` and return counts.

    This is the library's single execution primitive and its batching
    entry point: passing a list of circuits submits them as one job with one
    queue wait, which is what makes the multi-instance sweeps in
    :mod:`dgssp.experiments` affordable on hardware.

    Parameters
    ----------
    mode_or_backend:
        A backend, ``Session`` or ``Batch`` (see :func:`execution_mode`).
    circuits:
        A single already-transpiled circuit, or a sequence of them.  Circuits
        of differing widths are fine -- each becomes its own pub.
    shots:
        Shots per circuit.
    seed_simulator:
        Seed forwarded to simulator backends for reproducibility; ignored by
        hardware.
    run_log:
        Path of the JSON-lines job ledger, or ``None`` to skip logging.
    meta:
        Extra fields to record in the ledger entry (e.g. instance names).

    Returns
    -------
    list[dict[str, int]]
        One counts dictionary per input circuit, in input order.
    """
    from qiskit_ibm_runtime import SamplerV2

    opts = (
        {"simulator": {"seed_simulator": int(seed_simulator)}}
        if seed_simulator is not None
        else None
    )
    sampler = SamplerV2(mode=mode_or_backend, options=opts)

    pubs = list(circuits) if isinstance(circuits, (list, tuple)) else [circuits]
    job = sampler.run(pubs, shots=shots)

    log_job(job, {"n_pubs": len(pubs), "shots": shots, **(meta or {})}, run_log=run_log)

    result = job.result()
    return [pub.join_data().get_counts() for pub in result]


# ---------------------------------------------------------------------------
# Job bookkeeping
# ---------------------------------------------------------------------------


def log_job(
    job: Any, meta: dict[str, Any], *, run_log: str | None = DEFAULT_RUN_LOG
) -> str | None:
    """
    Record a submitted job in the JSON-lines ledger and the Python logger.

    Parameters
    ----------
    job:
        The object returned by ``sampler.run(...)``.
    meta:
        Arbitrary JSON-serialisable metadata to store alongside the job id.
    run_log:
        Path of the ledger file.  Parent directories are created as needed.
        Pass ``None`` to log to the Python logger only.

    Returns
    -------
    str | None
        The job id, or ``None`` if the object did not expose one.
    """
    try:
        job_id = job.job_id() if callable(getattr(job, "job_id", None)) else None
    except Exception:  # pragma: no cover - defensive
        job_id = None

    record = {"job_id": job_id, "ts": time.time(), **meta}
    logger.info("submitted job %s (%s)", job_id, meta)

    if run_log is not None and job_id is not None:
        path = pathlib.Path(run_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    return job_id


def fetch_result(service: Any, job_id: str) -> Any:
    """
    Re-fetch the result of a previously submitted job.

    Hardware queues are long enough that a sweep may outlive the process that
    launched it; combined with the ledger written by :func:`log_job`, this
    makes runs recoverable.

    Parameters
    ----------
    service:
        A ``QiskitRuntimeService`` (see :func:`get_service`).
    job_id:
        The job identifier recorded in ``runs/jobs.jsonl``.

    Returns
    -------
    PrimitiveResult
        The completed job's result object.
    """
    return service.job(job_id).result()


def read_job_log(run_log: str = DEFAULT_RUN_LOG) -> list[dict[str, Any]]:
    """
    Load the job ledger written by :func:`log_job`.

    Parameters
    ----------
    run_log:
        Path of the JSON-lines ledger.

    Returns
    -------
    list[dict]
        One record per submitted job, oldest first.  Empty if the file does
        not exist.
    """
    path = pathlib.Path(run_log)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


__all__ = [
    "get_service",
    "save_account",
    "is_simulator",
    "backend_name",
    "execution_mode",
    "sample_counts",
    "log_job",
    "fetch_result",
    "read_job_log",
    "DEFAULT_RUN_LOG",
]
