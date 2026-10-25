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

import copy
import json
import logging
import pathlib
import time
from collections.abc import Iterator, Mapping, Sequence
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


def describe_account(service: Any) -> dict[str, Any]:
    """
    Summarise what an IBM Quantum account can actually reach.

    Written to answer, in one call, the questions that determine what a paper's
    hardware plan can contain: which plan is active, which instances the
    account owns, which devices those instances expose (and how wide / which
    processor family they are), and how much QPU time has been used.

    Every field is optional: the helper probes the API defensively and reports
    ``{"error": ...}`` for anything the installed ``qiskit-ibm-runtime`` or the
    active plan does not expose, rather than raising.

    Parameters
    ----------
    service:
        A ``QiskitRuntimeService`` (see :func:`get_service`).

    Returns
    -------
    dict
        ``{"channel", "active_account", "active_instance", "instances",
        "usage", "backends"}``.  ``backends`` is a list of
        ``{"name", "num_qubits", "processor_type", "simulator", "operational",
        "pending_jobs", "calibration_date"}`` dictionaries.

    Notes
    -----
    ``usage()`` reports usage for the *active instance* only, and its payload
    shape is defined by the IBM Quantum Platform API rather than by Qiskit, so
    treat its keys as data to be inspected, not as a stable schema.  Per-job
    QPU seconds come from ``job.usage()`` / ``job.metrics()`` instead; see
    https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time
    """

    def _probe(label: str, fn: Any) -> Any:
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - network / plan dependent
            logger.debug("describe_account: %s unavailable (%s)", label, exc)
            return {"error": f"{type(exc).__name__}: {exc}"}

    account = _probe("active_account", lambda: service.active_account())
    if isinstance(account, dict):
        # Never let a token reach a results file or a printed summary.
        account = {k: v for k, v in account.items() if k not in ("token", "proxies")}

    summary: dict[str, Any] = {
        "channel": getattr(service, "channel", None),
        "active_account": account,
        "active_instance": _probe(
            "active_instance", lambda: getattr(service, "active_instance", None)
        ),
        "instances": _probe("instances", lambda: list(service.instances())),
        "usage": _probe("usage", lambda: service.usage()),
    }

    backends = _probe("backends", lambda: list(service.backends()))
    if isinstance(backends, list):
        rows = []
        for backend in backends:
            status = None
            try:
                status = backend.status()
            except Exception:  # pragma: no cover - transient provider errors
                pass
            rows.append(
                {
                    "name": backend_name(backend),
                    "num_qubits": getattr(backend, "num_qubits", None),
                    "processor_type": getattr(backend, "processor_type", None),
                    "simulator": is_simulator(backend),
                    "operational": getattr(status, "operational", None),
                    "pending_jobs": getattr(status, "pending_jobs", None),
                    "calibration_date": _calibration_date(backend),
                }
            )
        summary["backends"] = rows
    else:
        summary["backends"] = backends

    return summary


def _calibration_date(backend: Any) -> str | None:
    """Calibration timestamp of a backend, or ``None`` (see dgssp.backends)."""
    props_fn = getattr(backend, "properties", None)
    if not callable(props_fn):
        return None
    try:
        stamp = getattr(props_fn(), "last_update_date", None)
    except Exception:  # pragma: no cover - network errors
        return None
    if stamp is None:
        return None
    iso = getattr(stamp, "isoformat", None)
    return iso() if callable(iso) else str(stamp)


def print_account_summary(service: Any) -> dict[str, Any]:
    """
    Print :func:`describe_account` as readable text and return the raw dict.

    Intended to be run once from a terminal to fill in
    ``paper/hardware/account_facts.md``.

    Parameters
    ----------
    service:
        A ``QiskitRuntimeService``.

    Returns
    -------
    dict
        The same payload :func:`describe_account` returns.
    """
    info = describe_account(service)

    print(f"channel         : {info['channel']}")
    print(f"active instance : {info['active_instance']}")
    print(f"account         : {info['active_account']}")
    print(f"instances       : {info['instances']}")
    print(f"usage           : {info['usage']}")

    rows = info.get("backends")
    if isinstance(rows, list):
        print(f"\nbackends ({len(rows)}):")
        for row in rows:
            family = row.get("processor_type") or {}
            family_str = (
                f"{family.get('family')} r{family.get('revision')}"
                if isinstance(family, dict) and family
                else "-"
            )
            print(
                f"  {row['name']:<24} {str(row['num_qubits']):>4}q  "
                f"{family_str:<14} sim={row['simulator']!s:<5} "
                f"operational={row['operational']!s:<5} "
                f"pending={row['pending_jobs']} "
                f"calibrated={row['calibration_date']}"
            )
    else:
        print(f"\nbackends: {rows}")

    return info


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


class CountsList(list):  # type: ignore[type-arg]
    """
    The list of counts dictionaries :func:`sample_counts` returns.

    A plain ``list`` in every respect, plus the ``job_id`` of the job that
    produced it, so a results record can name its job without a second
    channel.  ``job_id`` is ``None`` when the primitive exposed none.
    """

    job_id: str | None = None


def _sampler_options(
    options: Mapping[str, Any] | None, seed_simulator: int | None
) -> dict[str, Any] | None:
    """``options`` as a fresh dict, with the simulator seed merged in."""
    merged: dict[str, Any] = copy.deepcopy(dict(options)) if options else {}
    if seed_simulator is not None:
        simulator = dict(merged.get("simulator") or {})
        simulator["seed_simulator"] = int(seed_simulator)
        merged["simulator"] = simulator
    return merged or None


def submit_sampler_job(
    mode_or_backend: Any,
    circuits: QuantumCircuit | Sequence[QuantumCircuit],
    *,
    shots: int = 10_000,
    options: Mapping[str, Any] | None = None,
    seed_simulator: int | None = None,
    run_log: str | None = DEFAULT_RUN_LOG,
    meta: dict[str, Any] | None = None,
) -> Any:
    """
    Submit one or many circuits as ONE ``SamplerV2`` job, and return the job.

    This is the only place the library constructs a ``SamplerV2``.
    :func:`sample_counts` is this plus waiting for the result; a hardware
    submission that must not block (``paper/scripts/hardware_batch.py
    --submit``) calls this directly and exits, and the counts are read back
    later with :func:`fetch_result` and :func:`counts_from_result`.  A
    simulator dry run goes through :func:`sample_counts`, so both build the
    same PUBs from the same options.

    Parameters
    ----------
    mode_or_backend:
        A backend, ``Session`` or ``Batch`` (see :func:`execution_mode`).
    circuits:
        A single already-transpiled circuit, or a sequence of them.  Each
        becomes its own pub.
    shots:
        Shots per circuit.
    options:
        ``SamplerOptions`` fields as a nested dict (``{"twirling": {...},
        "dynamical_decoupling": {...}}``).  A dict rather than an options
        object so the exact value can be written to a manifest.  Options the
        local simulator path does not implement are ignored there with a
        ``UserWarning`` from ``qiskit-ibm-runtime``.
    seed_simulator:
        Seed forwarded to simulator backends; merged into ``options``.
    run_log:
        Path of the JSON-lines job ledger, or ``None`` to skip logging.
    meta:
        Extra fields to record in the ledger entry.

    Returns
    -------
    RuntimeJobV2
        The submitted job (a local job object on simulators).
    """
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=mode_or_backend, options=_sampler_options(options, seed_simulator))

    pubs = list(circuits) if isinstance(circuits, (list, tuple)) else [circuits]
    job = sampler.run(pubs, shots=shots)

    log_job(job, {"n_pubs": len(pubs), "shots": shots, **(meta or {})}, run_log=run_log)
    return job


def counts_from_result(result: Any) -> CountsList:
    """
    One counts dictionary per pub of a ``SamplerV2`` result, in pub order.

    The single decoder for sampler results, shared by :func:`sample_counts`
    and by results fetched back later with :func:`fetch_result`.  Bitstrings
    are Qiskit's (index-register bit 0 rightmost); translating them to subsets
    is :mod:`dgssp.decoding`'s job.

    Parameters
    ----------
    result:
        A ``PrimitiveResult`` of sampler pub results.

    Returns
    -------
    CountsList
        Counts per pub; ``job_id`` is left ``None`` for the caller to set.
    """
    return CountsList(pub.join_data().get_counts() for pub in result)


def sample_counts(
    mode_or_backend: Any,
    circuits: QuantumCircuit | Sequence[QuantumCircuit],
    *,
    shots: int = 10_000,
    seed_simulator: int | None = None,
    run_log: str | None = DEFAULT_RUN_LOG,
    meta: dict[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
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
    options:
        ``SamplerOptions`` as a nested dict (see :func:`submit_sampler_job`).

    Returns
    -------
    CountsList
        One counts dictionary per input circuit, in input order.  It is a
        ``list``; its ``job_id`` attribute carries the job id.
    """
    job = submit_sampler_job(
        mode_or_backend,
        circuits,
        shots=shots,
        options=options,
        seed_simulator=seed_simulator,
        run_log=run_log,
        meta=meta,
    )
    out = counts_from_result(job.result())
    out.job_id = _job_id(job)
    return out


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

#: Directives the device never executes, so the target does not list them.
_DIRECTIVES = frozenset({"barrier"})


def _unsupported_instructions(circuit: QuantumCircuit, target: Any) -> list[str]:
    """Every ``(name, physical qubits)`` of ``circuit`` the target lacks."""
    index = {qubit: i for i, qubit in enumerate(circuit.qubits)}
    missing: list[str] = []
    for instruction in circuit.data:
        name = instruction.operation.name
        if name in _DIRECTIVES:
            continue
        qargs = tuple(index[q] for q in instruction.qubits)
        if not target.instruction_supported(name, qargs):
            missing.append(f"{name}{list(qargs)}")
    return missing


def preflight_sampler_job(
    backend: Any,
    circuits: Sequence[QuantumCircuit],
    *,
    shots: int,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run every check ``SamplerV2.run`` would run, without submitting anything.

    The pinned ``qiskit-ibm-runtime`` (0.49.0) exposes no server-side dry run,
    so this replays the client-side validation of ``BasePrimitiveV2._run``:
    options parsing, classical-register, dynamic-circuit-with-DD and ``rzz``
    checks, the ISA check against ``backend.target`` and, on an
    ``IBMBackend``, the faulty-qubit check.  An independent check asserts that
    every instruction of every circuit, on its physical qubits, is in the
    target.  Server-side checks (plan entitlements, twirling support) are
    *not* covered: nothing short of a submission exercises them.

    Parameters
    ----------
    backend:
        The device the job is for.
    circuits:
        The circuits, in submission order.
    shots:
        Shots per circuit.
    options:
        ``SamplerOptions`` as a nested dict.

    Returns
    -------
    dict
        ``{"passed", "checks", "program_inputs", "n_pubs", "shots"}``.
        ``checks`` maps each check name to ``"ok"`` or its error message;
        ``program_inputs`` is what the client would send as primitive options.
    """
    from dataclasses import asdict

    from qiskit.primitives.containers.sampler_pub import SamplerPub
    from qiskit_ibm_runtime.options import SamplerOptions
    from qiskit_ibm_runtime.options.utils import merge_options_v2
    from qiskit_ibm_runtime.utils import (
        validate_classical_registers,
        validate_isa_circuits,
        validate_no_dd_with_dynamic_circuits,
        validate_rzz_pubs,
    )

    checks: dict[str, str] = {}
    program_inputs: Any = None

    def _check(name: str, fn: Any) -> Any:
        try:
            value = fn()
        except Exception as exc:
            checks[name] = f"{type(exc).__name__}: {exc}"
            return None
        checks[name] = "ok"
        return value

    opts = _check(
        "options",
        lambda: SamplerOptions(
            **merge_options_v2(SamplerOptions(), _sampler_options(options, None) or {})
        ),
    )
    if opts is not None:
        # VERIFY: private classmethod; it is what BasePrimitiveV2._run calls
        # to build the primitive inputs in 0.49.0.
        program_inputs = _check(
            "program_inputs",
            lambda: SamplerOptions._get_program_inputs(asdict(opts)),
        )

    pubs = _check("coerce_pubs", lambda: [SamplerPub.coerce(c, shots) for c in circuits])
    if pubs is not None:
        _check("classical_registers", lambda: validate_classical_registers(pubs))
        _check("rzz_angles", lambda: validate_rzz_pubs(pubs))
        if opts is not None:
            _check(
                "dd_with_dynamic_circuits",
                lambda: validate_no_dd_with_dynamic_circuits(
                    [p.circuit for p in pubs], opts
                ),
            )

    target = getattr(backend, "target", None)
    if target is None:
        checks["isa"] = "backend exposes no target"
        checks["instructions_in_target"] = "backend exposes no target"
    else:
        _check("isa", lambda: validate_isa_circuits(list(circuits), target))
        missing = {
            i: bad for i, c in enumerate(circuits) if (bad := _unsupported_instructions(c, target))
        }
        checks["instructions_in_target"] = (
            "ok"
            if not missing
            else "; ".join(f"pub {i}: {', '.join(bad[:5])}" for i, bad in missing.items())
        )

    check_faulty = getattr(backend, "check_faulty", None)
    if callable(check_faulty):
        _check("faulty_qubits", lambda: [check_faulty(c) for c in circuits])

    return {
        "passed": all(v == "ok" for v in checks.values()),
        "checks": checks,
        "program_inputs": program_inputs,
        "n_pubs": len(circuits),
        "shots": int(shots),
    }


# ---------------------------------------------------------------------------
# Job bookkeeping
# ---------------------------------------------------------------------------


def _job_id(job: Any) -> str | None:
    """A job's id, or ``None`` if it exposes none."""
    try:
        return job.job_id() if callable(getattr(job, "job_id", None)) else None
    except Exception:  # pragma: no cover - defensive
        return None


def job_usage(job: Any) -> dict[str, Any]:
    """
    Everything a runtime job reports about the QPU time it was charged.

    In the pinned ``qiskit-ibm-runtime`` (0.49.0) ``job.usage()`` returns
    ``metrics()["usage"]["qpu_charge_time_seconds"]`` once usage is final (``0``
    while pending), ``job.metrics()`` returns the job's metadata (with
    ``timestamps`` and ``usage``), and ``job.usage_estimation`` the
    pre-execution estimate ``{"quantum_seconds": ...}``.  Each is probed
    separately and an unavailable one is reported as ``{"error": ...}``.

    Parameters
    ----------
    job:
        A ``RuntimeJobV2`` (e.g. from ``service.job(job_id)``).

    Returns
    -------
    dict
        ``{"job_id", "status", "usage_seconds", "usage_estimation", "metrics"}``.

    Notes
    -----
    What ``qpu_charge_time_seconds`` counts is set by the platform, not the
    client: see https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time
    """

    def _probe(fn: Any) -> Any:
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - network / plan dependent
            return {"error": f"{type(exc).__name__}: {exc}"}

    status = _probe(lambda: job.status())
    return {
        "job_id": _job_id(job),
        "status": getattr(status, "name", status),
        "usage_seconds": _probe(lambda: job.usage()),
        "usage_estimation": _probe(lambda: job.usage_estimation),
        "metrics": _probe(lambda: job.metrics()),
    }


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
    job_id = _job_id(job)

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
    "describe_account",
    "print_account_summary",
    "is_simulator",
    "backend_name",
    "execution_mode",
    "CountsList",
    "submit_sampler_job",
    "counts_from_result",
    "sample_counts",
    "preflight_sampler_job",
    "job_usage",
    "log_job",
    "fetch_result",
    "read_job_log",
    "DEFAULT_RUN_LOG",
]
