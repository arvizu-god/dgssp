# src/ssp/execution.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple

from qiskit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit_ibm_runtime import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .backends import (
    BackendLike,                # alias = BackendV2
    BackendErrorMetrics,
    build_ideal_aer_backend,
    compute_accumulated_errors,
)
from .transpilation import find_best_seed, BestSeedResult
from .mitigation import ZNESamplingConfig, zne_mitigated_distribution


# ---------------------------------------------------------------------------
# Data container for execution results
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Summary of a single circuit execution on a given backend."""
    backend: BackendLike
    backend_name: str
    counts: Dict[str, int]
    transpiled_circuit: Optional[QuantumCircuit] = None
    best_seed: Optional[int] = None
    error_metrics: Optional[BackendErrorMetrics] = None
    mitigated_distribution: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Internal helper: sampler-based execution
# ---------------------------------------------------------------------------

def _backend_name(backend: BackendLike) -> str:
    name_attr = getattr(backend, "name", None)
    if callable(name_attr):
        return name_attr()
    return name_attr or str(backend)


def _execute_with_sampler(
    backend: BackendLike,
    circuit: QuantumCircuit,
    *,
    shots: int,
    seed_simulator: Optional[int] = None,
) -> Dict[str, int]:
    """Execute a transpiled circuit with SamplerV2 and return counts."""
    options = {}
    if seed_simulator is not None:
        options = {"simulator": {"seed_simulator": int(seed_simulator)}}

    sampler = SamplerV2(mode=backend, options=options or None)
    job = sampler.run([circuit], shots=shots)
    pub_result = job.result()[0]
    counts = pub_result.join_data().get_counts()
    return counts


# ---------------------------------------------------------------------------
# 1) Ideal executor (AerSimulator, no mitigation)
# ---------------------------------------------------------------------------

def execute_ideal(
    qc: QuantumCircuit,
    *,
    shots: int = 10_000,
    seed_simulator: Optional[int] = None,
    optimization_level: int = 1,
) -> ExecutionResult:
    """Execute a circuit on an ideal AerSimulator (no noise, no mitigation)."""
    backend = build_ideal_aer_backend(seed_simulator=seed_simulator)

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
    )
    tqc = pm.run(qc)

    counts = _execute_with_sampler(
        backend,
        tqc,
        shots=shots,
        seed_simulator=seed_simulator,
    )

    return ExecutionResult(
        backend=backend,
        backend_name=_backend_name(backend),
        counts=counts,
        transpiled_circuit=tqc,
        best_seed=None,
        error_metrics=None,
        mitigated_distribution=None,
    )


# ---------------------------------------------------------------------------
# 2) Noisy executor (fake or real backend, no optimization / mitigation)
# ---------------------------------------------------------------------------

def execute_noisy(
    qc: QuantumCircuit,
    backend: BackendLike,
    *,
    shots: int = 10_000,
    seed_transpiler: Optional[int] = None,
    seed_simulator: Optional[int] = None,
    optimization_level: int = 1,
) -> ExecutionResult:
    """
    Execute a circuit on a given noisy backend (fake or real) with a
    straightforward preset pass manager (no best-seed search, no ZNE).
    """
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    tqc = pm.run(qc)

    counts = _execute_with_sampler(
        backend,
        tqc,
        shots=shots,
        seed_simulator=seed_simulator,
    )

    # Optional: compute error metrics for the transpiled circuit
    metrics = compute_accumulated_errors(backend, tqc)

    return ExecutionResult(
        backend=backend,
        backend_name=_backend_name(backend),
        counts=counts,
        transpiled_circuit=tqc,
        best_seed=seed_transpiler,
        error_metrics=metrics,
        mitigated_distribution=None,
    )


# ---------------------------------------------------------------------------
# 3) Optimized / mitigated executor
#    - pick best fake & real backends by smallest accumulated error
#    - find best SABRE seed/layout on each
#    - execute with that layout (unmitigated counts)
#    - build ZNE-mitigated distribution for each best backend
# ---------------------------------------------------------------------------

def _select_best_backend_by_error(
    qc: QuantumCircuit,
    backends: List[BackendLike],
    *,
    optimization_level: int = 1,
    layout_method: str = "sabre",
    seed_transpiler: int = 0,
) -> Tuple[Optional[BackendLike], Optional[BackendErrorMetrics]]:
    """Select the backend with the smallest accumulated total error."""
    if not backends:
        return None, None

    best_backend: Optional[BackendLike] = None
    best_metrics: Optional[BackendErrorMetrics] = None
    best_error: float = float("inf")

    for backend in backends:
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            layout_method=layout_method,
        )
        tqc = pm.run(qc)
        metrics = compute_accumulated_errors(backend, tqc)
        if metrics.total_error < best_error:
            best_error = metrics.total_error
            best_backend = backend
            best_metrics = metrics

    return best_backend, best_metrics


def execute_optimized_mitigated(
    qc: QuantumCircuit,
    backends_bundle: Dict[str, Any],
    zne_cfg: ZNESamplingConfig,
    *,
    shots_unmitigated: int = 10_000,
) -> Dict[str, Optional[ExecutionResult]]:
    """
    Optimized / mitigated execution:

    1) From `backends_bundle` (output of build_all_backends), pick:
       - best fake backend,
       - best real backend,
       according to smallest accumulated error on a preliminary SABRE layout.

    2) For each of these best backends:
       - run `find_best_seed` to obtain the best SABRE layout / seed,
       - execute the circuit with that layout (unmitigated counts),
       - compute error metrics for the best-layout circuit,
       - run ZNE (local folding + curve fitting) to obtain a mitigated
         full probability distribution.

    Parameters
    ----------
    qc:
        Logical quantum circuit (e.g. DG-SSP circuit).
    backends_bundle:
        Dictionary returned by ssp.backends.build_all_backends:
        {
          "service": ...,
          "ideal": BackendLike,
          "real_backends": List[BackendLike],
          "fake_backends": List[BackendLike],
        }
    zne_cfg:
        ZNESamplingConfig specifying ZNE scales, fit method, etc.
    shots_unmitigated:
        Shots used for the unmitigated execution on each best backend.

    Returns
    -------
    Dict[str, Optional[ExecutionResult]]
        {
          "best_fake": ExecutionResult or None,
          "best_real": ExecutionResult or None,
        }
    """
    real_backends: List[BackendLike] = backends_bundle.get("real_backends", [])
    fake_backends: List[BackendLike] = backends_bundle.get("fake_backends", [])

    # 1) Select best fake and best real by accumulated error
    best_fake_backend, fake_pre_metrics = _select_best_backend_by_error(
        qc,
        fake_backends,
        optimization_level=1,
        layout_method="sabre",
        seed_transpiler=0,
    )
    best_real_backend, real_pre_metrics = _select_best_backend_by_error(
        qc,
        real_backends,
        optimization_level=1,
        layout_method="sabre",
        seed_transpiler=0,
    )

    results: Dict[str, Optional[ExecutionResult]] = {
        "best_fake": None,
        "best_real": None,
    }

    # 2) For best FAKE backend: best seed, unmitigated counts, ZNE distribution
    if best_fake_backend is not None:
        # Best seed / layout via SABRE
        fake_best = find_best_seed(
            qc,
            best_fake_backend,
            seed_min=0,
            seed_max=128,
            optimization_level=zne_cfg.optimization_level,
            layout_method=zne_cfg.layout_method,
        )
        fake_circuit = fake_best.circuit

        # Execute unmitigated with that layout
        fake_counts = _execute_with_sampler(
            best_fake_backend,
            fake_circuit,
            shots=shots_unmitigated,
            seed_simulator=zne_cfg.seed_simulator,
        )

        fake_metrics = compute_accumulated_errors(best_fake_backend, fake_circuit)

        # ZNE-mitigated distribution (starts from logical `qc`)
        fake_mitigated = zne_mitigated_distribution(
            qc,
            best_fake_backend,
            zne_cfg,
        )

        results["best_fake"] = ExecutionResult(
            backend=best_fake_backend,
            backend_name=_backend_name(best_fake_backend),
            counts=fake_counts,
            transpiled_circuit=fake_circuit,
            best_seed=fake_best.best_seed,
            error_metrics=fake_metrics,
            mitigated_distribution=fake_mitigated,
        )

    # 3) For best REAL backend: best seed, unmitigated counts, ZNE distribution
    if best_real_backend is not None:
        real_best = find_best_seed(
            qc,
            best_real_backend,
            seed_min=0,
            seed_max=128,
            optimization_level=zne_cfg.optimization_level,
            layout_method=zne_cfg.layout_method,
        )
        real_circuit = real_best.circuit

        real_counts = _execute_with_sampler(
            best_real_backend,
            real_circuit,
            shots=shots_unmitigated,
            seed_simulator=None,  # no simulator seed on real hardware
        )

        real_metrics = compute_accumulated_errors(best_real_backend, real_circuit)

        real_mitigated = zne_mitigated_distribution(
            qc,
            best_real_backend,
            zne_cfg,
        )

        results["best_real"] = ExecutionResult(
            backend=best_real_backend,
            backend_name=_backend_name(best_real_backend),
            counts=real_counts,
            transpiled_circuit=real_circuit,
            best_seed=real_best.best_seed,
            error_metrics=real_metrics,
            mitigated_distribution=real_mitigated,
        )

    return results


__all__ = [
    "ExecutionResult",
    "execute_ideal",
    "execute_noisy",
    "execute_optimized_mitigated",
]
