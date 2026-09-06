"""
mitiq_baseline.py

Mitiq's ZNE, adapted to a sampling algorithm, as an independent baseline.

Mitiq is built around an executor with signature ``Circuit -> float``: it
mitigates an *expectation value*.  The D-G algorithm does not produce one --
it produces a distribution over bitstrings.  The adaptation is to define the
"observable" as a scalar functional of that distribution.  The natural choice
is

    P(solution) = sum over solution bitstrings of their sampled probability

which is exactly the expectation value of the indicator function
``F(b) = 1 if b is a solution else 0``.  With that in hand Mitiq's folding and
extrapolation machinery applies unchanged.

Why keep this alongside :mod:`dgssp.mitigation.zne`?  The two answer different
questions and comparing them is the point:

* :mod:`~dgssp.mitigation.zne` extrapolates *every bitstring probability*,
  yielding a full mitigated distribution -- more informative, but each
  bitstring is fitted from noisy, low-count data.
* This module extrapolates *one aggregate scalar* with a third-party,
  independently-validated implementation -- less informative, but statistically
  far better conditioned.

Mitiq is an optional dependency; install it with ``pip install "dgssp[mitiq]"``.
It is imported lazily inside the functions that need it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeAlias

from qiskit import QuantumCircuit
from qiskit.providers import BackendV2

from ..runtime import is_simulator, sample_counts
from .zne import rebase_to_backend

BackendLike: TypeAlias = BackendV2

_MITIQ_HINT = (
    "Mitiq is required for the ZNE comparison baseline. "
    'Install it with: pip install "dgssp[mitiq]"'
)


def _require_mitiq() -> Any:
    """
    Import Mitiq or raise a clear, actionable error.

    Returns
    -------
    module
        The imported ``mitiq`` module.

    Raises
    ------
    ImportError
        If Mitiq is not installed, with the install command in the message.
    """
    try:
        import mitiq
    except ImportError as exc:  # pragma: no cover - exercised only without mitiq
        raise ImportError(_MITIQ_HINT) from exc
    return mitiq


def make_sampling_executor(
    backend: BackendLike,
    solution_bitstrings: Iterable[str],
    *,
    shots: int = 10_000,
    seed_simulator: int | None = None,
) -> Callable[[QuantumCircuit], float]:
    """
    Build the ``Circuit -> float`` executor Mitiq expects.

    The returned callable executes whatever circuit Mitiq hands it (already
    folded) at optimization level 0 -- no transpilation, no re-routing -- and
    returns the sampled probability of a solution bitstring.  Transpile
    *before* calling Mitiq so the folds scale physical noise.

    Parameters
    ----------
    backend:
        The noisy backend to execute on.
    solution_bitstrings:
        The outcomes counted as correct (from the classical DP ground truth).
    shots:
        Shots per executor call.
    seed_simulator:
        Simulator seed, applied only on simulators.

    Returns
    -------
    Callable[[QuantumCircuit], float]
        An executor returning ``P(solution)`` in ``[0, 1]``.
    """
    solutions = list(solution_bitstrings)
    seed = seed_simulator if is_simulator(backend) else None

    def executor(circuit: QuantumCircuit) -> float:
        # Mitiq's folding inserts inverse gates that need not be native (on IBM
        # devices sx inverts to sxdg), so rebase before submitting.
        counts = sample_counts(
            backend,
            rebase_to_backend(circuit, backend),
            shots=shots,
            seed_simulator=seed,
            meta={"stage": "mitiq_zne"},
        )[0]
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return sum(counts.get(bit, 0) for bit in solutions) / total

    return executor


def mitiq_zne_solution_probability(
    tqc: QuantumCircuit,
    backend: BackendLike,
    solution_bitstrings: Iterable[str],
    *,
    shots: int = 10_000,
    scale_factors: Sequence[float] = (1.0, 3.0, 5.0),
    factory: Any | None = None,
    scale_noise: Callable[..., QuantumCircuit] | None = None,
    seed_simulator: int | None = None,
) -> float:
    """
    Mitigate ``P(solution)`` with Mitiq's ZNE.

    Parameters
    ----------
    tqc:
        An **already transpiled** (ISA-level) D-G circuit.  Passing a logical
        circuit would make Mitiq fold logical gates, which is not the noise
        you want to scale.
    backend:
        The noisy backend to execute on.
    solution_bitstrings:
        Outcomes counted as correct.
    shots:
        Shots per executor call.
    scale_factors:
        Noise scale factors handed to the extrapolation factory.
    factory:
        A Mitiq inference factory.  Defaults to
        ``LinearFactory(scale_factors)``.
    scale_noise:
        Mitiq noise-scaling function.  Defaults to ``fold_gates_at_random``.
    seed_simulator:
        Simulator seed, applied only on simulators.

    Returns
    -------
    float
        The zero-noise-extrapolated probability of sampling a solution.  Not
        clipped to ``[0, 1]``: an extrapolated value outside that interval is
        diagnostic information about the fit and is deliberately preserved.

    Raises
    ------
    ImportError
        If Mitiq is not installed.
    """
    _require_mitiq()
    from mitiq.zne import execute_with_zne
    from mitiq.zne.inference import LinearFactory
    from mitiq.zne.scaling import fold_gates_at_random

    executor = make_sampling_executor(
        backend, solution_bitstrings, shots=shots, seed_simulator=seed_simulator
    )
    fac = factory or LinearFactory(scale_factors=list(scale_factors))
    noise_scaler = scale_noise or fold_gates_at_random

    return float(
        execute_with_zne(tqc, executor, factory=fac, scale_noise=noise_scaler)
    )


def mitiq_is_available() -> bool:
    """
    Whether the optional Mitiq dependency is importable.

    Returns
    -------
    bool
        ``True`` if ``import mitiq`` succeeds.
    """
    try:
        import mitiq  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = [
    "make_sampling_executor",
    "mitiq_zne_solution_probability",
    "mitiq_is_available",
]
