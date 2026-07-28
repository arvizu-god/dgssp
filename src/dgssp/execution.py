"""
execution.py

Three single-circuit execution strategies, from cheapest to most involved.

* :func:`execute_ideal` -- noiseless ``AerSimulator``, no mitigation.  The
  reference distribution.
* :func:`execute_noisy` -- one noisy backend, plain transpilation, no
  mitigation.  What the algorithm actually does on hardware today.
* :func:`execute_optimized_mitigated` -- pick the lowest-error backend, search
  a good SABRE layout once, run unmitigated *and* ZNE-mitigated on that same
  physical circuit.  The best result the library can produce.

All three go through :func:`dgssp.runtime.sample_counts`, so the sampler
incantation, the simulator seeding rules and the job logging live in exactly
one place.  For sweeps over *many* instances, use
:func:`dgssp.experiments.run_batch` instead -- it batches every circuit into a
single job.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .backends import (
    BackendErrorMetrics,
    BackendLike,
    build_ideal_aer_backend,
    compute_accumulated_errors,
)
from .mitigation import ZNESamplingConfig, transpile_once, zne_mitigated_distribution
from .runtime import backend_name, execution_mode, sample_counts

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """
    Outcome of running one circuit on one backend.

    Attributes
    ----------
    backend:
        The backend used.
    backend_name:
        Its name.
    counts:
        Raw measurement counts.
    transpiled_circuit:
        The ISA-level circuit actually executed.
    best_seed:
        Transpiler seed used, when a layout search was performed.
    error_metrics:
        Calibration-based error accumulation, when the backend exposes a
        target.
    mitigated_distribution:
        ZNE-mitigated distribution, when mitigation was requested.
    """

    backend: BackendLike
    backend_name: str
    counts: dict[str, int]
    transpiled_circuit: QuantumCircuit | None = None
    best_seed: int | None = None
    error_metrics: BackendErrorMetrics | None = None
    mitigated_distribution: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# 1) Ideal
# ---------------------------------------------------------------------------


def execute_ideal(
    qc: QuantumCircuit,
    *,
    shots: int = 10_000,
    seed_simulator: int | None = None,
    optimization_level: int = 1,
) -> ExecutionResult:
    """
    Run a circuit on a noiseless ``AerSimulator``.

    Parameters
    ----------
    qc:
        The logical circuit.
    shots:
        Number of shots.
    seed_simulator:
        Seed for reproducible sampling.
    optimization_level:
        Preset optimization level for the local transpilation.

    Returns
    -------
    ExecutionResult
        With ``error_metrics`` and ``mitigated_distribution`` left as ``None``.
    """
    backend = build_ideal_aer_backend(seed_simulator=seed_simulator)
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=optimization_level
    )
    tqc = pm.run(qc)

    counts = sample_counts(
        backend,
        tqc,
        shots=shots,
        seed_simulator=seed_simulator,
        meta={"stage": "ideal"},
    )[0]

    return ExecutionResult(
        backend=backend,
        backend_name=backend_name(backend),
        counts=counts,
        transpiled_circuit=tqc,
    )


# ---------------------------------------------------------------------------
# 2) Noisy
# ---------------------------------------------------------------------------


def execute_noisy(
    qc: QuantumCircuit,
    backend: BackendLike,
    *,
    shots: int = 10_000,
    seed_transpiler: int | None = None,
    seed_simulator: int | None = None,
    optimization_level: int = 1,
    mode: str = "auto",
) -> ExecutionResult:
    """
    Run a circuit on a noisy backend with straightforward transpilation.

    Parameters
    ----------
    qc:
        The logical circuit.
    backend:
        A noisy simulator or a real device.
    shots:
        Number of shots.
    seed_transpiler:
        Seed for the preset pass manager.
    seed_simulator:
        Simulator seed (ignored by hardware).
    optimization_level:
        Preset optimization level.
    mode:
        Execution mode passed to :func:`dgssp.runtime.execution_mode`.

    Returns
    -------
    ExecutionResult
        Including calibration error metrics when the backend has a target.
    """
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    tqc = pm.run(qc)

    with execution_mode(backend, mode=mode) as exec_mode:  # type: ignore[arg-type]
        counts = sample_counts(
            exec_mode,
            tqc,
            shots=shots,
            seed_simulator=seed_simulator,
            meta={"stage": "noisy", "backend": backend_name(backend)},
        )[0]

    try:
        metrics: BackendErrorMetrics | None = compute_accumulated_errors(backend, tqc)
    except AttributeError:
        metrics = None

    return ExecutionResult(
        backend=backend,
        backend_name=backend_name(backend),
        counts=counts,
        transpiled_circuit=tqc,
        best_seed=seed_transpiler,
        error_metrics=metrics,
    )


# ---------------------------------------------------------------------------
# 3) Optimized + mitigated
# ---------------------------------------------------------------------------


def select_best_backend_by_error(
    qc: QuantumCircuit,
    backends: Sequence[BackendLike],
    *,
    optimization_level: int = 1,
    layout_method: str = "sabre",
    seed_transpiler: int = 0,
) -> tuple[BackendLike | None, BackendErrorMetrics | None]:
    """
    Pick the backend with the lowest accumulated calibration error.

    A single cheap transpilation per backend is used for the comparison; the
    expensive seed sweep happens only on the winner.

    Parameters
    ----------
    qc:
        The logical circuit.
    backends:
        Candidate backends.
    optimization_level:
        Optimization level for the preliminary transpilation.
    layout_method:
        Layout method for it.
    seed_transpiler:
        Seed, applied identically to every candidate so the comparison is fair.

    Returns
    -------
    tuple[BackendV2 | None, BackendErrorMetrics | None]
        The winning backend and its metrics, or ``(None, None)`` if the
        candidate list was empty or none exposed a target.
    """
    best_backend: BackendLike | None = None
    best_metrics: BackendErrorMetrics | None = None
    best_error = float("inf")

    for backend in backends:
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            layout_method=layout_method,
        )
        tqc = pm.run(qc)
        try:
            metrics = compute_accumulated_errors(backend, tqc)
        except AttributeError:
            continue
        if metrics.total_error < best_error:
            best_error = metrics.total_error
            best_backend = backend
            best_metrics = metrics

    return best_backend, best_metrics


def execute_optimized_mitigated(
    qc: QuantumCircuit,
    backends_bundle: dict[str, Any],
    zne_cfg: ZNESamplingConfig,
    *,
    shots_unmitigated: int = 10_000,
) -> dict[str, ExecutionResult | None]:
    """
    Best-backend + best-layout + ZNE, for the best fake and best real device.

    For each of the two device classes present in the bundle:

    1. pick the lowest-error backend,
    2. search a SABRE layout **once**,
    3. sample that ISA circuit unmitigated,
    4. reuse the *same* ISA circuit for the ZNE folds, so mitigated and
       unmitigated numbers describe the same physical circuit.

    Parameters
    ----------
    qc:
        The logical circuit.
    backends_bundle:
        Output of :func:`dgssp.backends.build_all_backends`.
    zne_cfg:
        ZNE settings, including the seed range for the layout search.
    shots_unmitigated:
        Shots for the unmitigated run on each winning backend.

    Returns
    -------
    dict[str, ExecutionResult | None]
        ``{"best_fake": ..., "best_real": ...}``; a value is ``None`` when no
        backend of that class was available.
    """
    results: dict[str, ExecutionResult | None] = {
        "best_fake": None,
        "best_real": None,
    }

    for key, bundle_key in (("best_fake", "fake_backends"), ("best_real", "real_backends")):
        candidates: list[BackendLike] = list(backends_bundle.get(bundle_key, []))
        if not candidates:
            continue

        backend, _ = select_best_backend_by_error(qc, candidates)
        if backend is None:
            continue

        # One layout search, reused for the unmitigated run and every fold.
        tqc = transpile_once(qc, backend, zne_cfg)

        with execution_mode(backend) as exec_mode:  # type: ignore[arg-type]
            counts = sample_counts(
                exec_mode,
                tqc,
                shots=shots_unmitigated,
                seed_simulator=zne_cfg.seed_simulator,
                meta={"stage": "optimized", "backend": backend_name(backend)},
            )[0]

        try:
            metrics: BackendErrorMetrics | None = compute_accumulated_errors(
                backend, tqc
            )
        except AttributeError:
            metrics = None

        mitigated = zne_mitigated_distribution(
            qc, backend, zne_cfg, transpiled=tqc
        )

        results[key] = ExecutionResult(
            backend=backend,
            backend_name=backend_name(backend),
            counts=counts,
            transpiled_circuit=tqc,
            error_metrics=metrics,
            mitigated_distribution=mitigated,
        )

    return results


__all__ = [
    "ExecutionResult",
    "execute_ideal",
    "execute_noisy",
    "select_best_backend_by_error",
    "execute_optimized_mitigated",
]
