"""
mitiq_baseline.py

Mitiq's ZNE, adapted to a sampling algorithm, as an independent baseline.

Mitiq mitigates an *expectation value*.  The D-G algorithm produces a
distribution over bitstrings, so the "observable" is a scalar functional of
it, ``P(solution)`` -- the expectation of the indicator of the solution set.
With that, Mitiq's folding and extrapolation apply unchanged.

Two parts of Mitiq are used, and nothing else:

* **Folding** -- ``mitiq.zne.scaling.fold_gates_at_random``, Mitiq's public
  random local folding.  It is handed the ISA circuit as a *native Cirq
  circuit* built by :func:`mitiq_fold_isa`: each Qiskit instruction becomes an
  opaque Cirq gate on ``LineQubit(physical index)`` whose inverse is the Qiskit
  inverse.  Mitiq's own Qiskit conversion goes through OpenQASM 2 and Cirq's
  QASM importer, which (a) needs the ``ply`` package, absent from the paper
  environment, and (b) renumbers qubits, so an ISA circuit would come back off
  its layout.  The wrapper avoids both and is lossless: the circuit that
  comes back is the input circuit plus Mitiq's folds, on the same physical
  qubits.  Barriers are dropped, exactly as Mitiq's Qiskit path drops them.
* **Extrapolation** -- the static ``extrapolate`` methods of
  ``LinearFactory``, ``PolyFactory``, ``RichardsonFactory`` and
  ``ExpFactory``, applied *offline* to the measured values
  (:func:`mitiq_extrapolate`).  Mitiq never executes a circuit: the folded
  circuits are sampled inside the caller's single job
  (:func:`dgssp.mitigation.zne.run_zne_batch`).

Mitiq is an optional dependency; install it with ``pip install "dgssp[mitiq]"``.
It is imported lazily inside the functions that need it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any, TypeAlias

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2

from ..metrics import percentile_interval, resample_counts, solution_probability
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


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _isa_gate_class() -> type:
    """The Cirq gate that carries one Qiskit instruction (built lazily)."""
    import cirq

    class IsaGate(cirq.Gate):  # type: ignore[misc]
        """An opaque Cirq gate wrapping a Qiskit operation."""

        def __init__(self, operation: Any) -> None:
            self.operation = operation

        def _num_qubits_(self) -> int:
            return int(self.operation.num_qubits)

        def _has_unitary_(self) -> bool:
            return True

        def __pow__(self, exponent: Any) -> Any:
            if exponent == 1:
                return self
            if exponent == -1:
                return IsaGate(self.operation.inverse())
            return NotImplemented

        def _circuit_diagram_info_(self, args: Any) -> tuple[str, ...]:
            return (self.operation.name,) * self.operation.num_qubits

    return IsaGate


def mitiq_fold_isa(tqc: QuantumCircuit, scale_factor: float, *, seed: int | None = 0) -> QuantumCircuit:
    """
    Fold an ISA circuit with Mitiq's ``fold_gates_at_random``, losslessly.

    Parameters
    ----------
    tqc:
        An ISA circuit (qubit index = physical qubit) whose measurements are
        terminal.
    scale_factor:
        Mitiq's scale factor, ``>= 1``.  Mitiq measures it on a
        fidelity-weighted count of *all* gates (default weights
        ``1 - 0.99**n_qubits``, virtual ``rz`` included), so the realised
        two-qubit ratio differs from it; callers record that ratio.
    seed:
        Mitiq's folding seed.

    Returns
    -------
    QuantumCircuit
        Same registers and layout as ``tqc``; not yet rebased (Mitiq's
        inverses, e.g. ``sxdg``, need :func:`~dgssp.mitigation.zne.
        rebase_to_backend`).  Barriers are dropped, as in Mitiq's own Qiskit
        conversion.

    Raises
    ------
    ImportError
        If Mitiq is not installed.
    ValueError
        If the circuit contains resets or delays.
    """
    _require_mitiq()
    import cirq
    from mitiq.zne.scaling import fold_gates_at_random

    gate_cls = _isa_gate_class()
    line = cirq.LineQubit.range(tqc.num_qubits)
    operations = []
    for instruction in tqc.data:
        op = instruction.operation
        qubits = [line[tqc.find_bit(q).index] for q in instruction.qubits]
        if op.name == "barrier":
            continue
        if op.name == "measure":
            key = str(tqc.find_bit(instruction.clbits[0]).index)
            operations.append(cirq.measure(*qubits, key=key))
            continue
        if op.name in ("reset", "delay"):
            raise ValueError(f"mitiq_fold_isa cannot fold a circuit containing {op.name!r}.")
        operations.append(gate_cls(op).on(*qubits))

    folded = fold_gates_at_random(cirq.Circuit(operations), float(scale_factor), seed=seed)

    out = QuantumCircuit(*tqc.qregs, *tqc.cregs, name=f"{tqc.name}_mitiq_x{float(scale_factor):.4g}")
    for operation in folded.all_operations():
        qubits = [out.qubits[q.x] for q in operation.qubits]
        if cirq.is_measurement(operation):
            out.measure(qubits[0], out.clbits[int(operation.gate.key)])
        else:
            out.append(operation.gate.operation, qubits)
    if getattr(tqc, "layout", None) is not None:
        out._layout = tqc.layout  # noqa: SLF001 - no public setter exists
    return out


# ---------------------------------------------------------------------------
# Offline extrapolation
# ---------------------------------------------------------------------------


def mitiq_extrapolate(
    scale_factors: Sequence[float],
    values: Sequence[float],
    method: str,
    *,
    asymptote: float | None = None,
) -> float:
    """
    Mitiq's zero-noise value for measured data, via its static factories.

    Parameters
    ----------
    scale_factors:
        The scales the values were measured at (for Mitiq's folds, the scales
        Mitiq was asked for).
    values:
        Measured expectation values.
    method:
        ``"linear"`` -> ``LinearFactory``; ``"quadratic"`` ->
        ``PolyFactory(order=2)``; ``"richardson"`` -> ``RichardsonFactory``;
        ``"exponential"`` -> ``ExpFactory(asymptote=...)``.
    asymptote:
        Passed to ``ExpFactory``.

    Returns
    -------
    float
        The extrapolated value, or ``nan`` if Mitiq's fit fails (e.g.
        ``ExpFactory``'s log-linear fit meeting a value on the wrong side of
        the asymptote).  Not clipped.

    Raises
    ------
    ImportError
        If Mitiq is not installed.
    ValueError
        For an unknown method.
    """
    _require_mitiq()
    from mitiq.zne.inference import ExpFactory, LinearFactory, PolyFactory, RichardsonFactory

    scales = [float(s) for s in scale_factors]
    data = [float(v) for v in values]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if method == "linear":
                result = LinearFactory.extrapolate(scales, data)
            elif method == "quadratic":
                result = PolyFactory.extrapolate(scales, data, order=2)
            elif method == "richardson":
                result = RichardsonFactory.extrapolate(scales, data)
            elif method == "exponential":
                result = ExpFactory.extrapolate(scales, data, asymptote=asymptote)
            else:
                raise ValueError(f"Unknown method '{method}' for the Mitiq baseline.")
        except ValueError:
            raise
        except Exception:
            return float("nan")
    return float(result)


def mitiq_bootstrap(
    counts_per_scale: Mapping[float, Mapping[str, int]],
    observable_fn: Callable[[Mapping[str, int]], float],
    *,
    method: str,
    asymptote: float | None = None,
    B: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Mitiq's extrapolation with a percentile-bootstrap interval.

    Parameters
    ----------
    counts_per_scale:
        Counts of Mitiq's folded circuits, keyed by the scale Mitiq was asked
        for.
    observable_fn:
        The proportion to extrapolate (``P(solution)``).
    method, asymptote:
        See :func:`mitiq_extrapolate`.
    B, seed, alpha:
        Resamples, seed and ``1 - confidence``.

    Returns
    -------
    dict
        ``{"method", "asymptote", "scales", "values", "value", "lo", "hi",
        "se", "n_failed", "B", "seed"}``.
    """
    scales = sorted(counts_per_scale)
    values = [float(observable_fn(counts_per_scale[s])) for s in scales]
    point = mitiq_extrapolate(scales, values, method, asymptote=asymptote)

    rng = np.random.default_rng(seed)
    Y = np.empty((B, len(scales)))
    for j, scale in enumerate(scales):
        keys, draws = resample_counts(counts_per_scale[scale], rng, B)
        Y[:, j] = [observable_fn(dict(zip(keys, row.tolist(), strict=True))) for row in draws]
    replicates = np.array(
        [mitiq_extrapolate(scales, row.tolist(), method, asymptote=asymptote) for row in Y]
    )
    lo, hi = percentile_interval(replicates, alpha)
    finite = replicates[np.isfinite(replicates)]
    return {
        "method": method,
        "asymptote": asymptote,
        "scales": [float(s) for s in scales],
        "values": values,
        "value": point,
        "lo": lo,
        "hi": hi,
        "se": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
        "n_failed": int(replicates.size - finite.size),
        "B": B,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Stand-alone entry points
# ---------------------------------------------------------------------------


def make_sampling_executor(
    backend: BackendLike,
    solution_bitstrings: Iterable[str],
    *,
    shots: int = 10_000,
    seed_simulator: int | None = None,
) -> Callable[[QuantumCircuit], float]:
    """
    Build the ``Circuit -> float`` executor Mitiq's ``execute_with_zne`` expects.

    Kept for interactive use.  The library itself no longer lets Mitiq drive
    execution -- one job per folded circuit is incompatible with the one-job
    hardware budget -- and uses :func:`mitiq_zne_solution_probability` or
    :func:`dgssp.mitigation.zne.run_zne_batch` instead.

    Parameters
    ----------
    backend:
        The noisy backend to execute on.
    solution_bitstrings:
        The outcomes counted as correct.
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
        counts = sample_counts(
            backend,
            rebase_to_backend(circuit, backend),
            shots=shots,
            seed_simulator=seed,
            meta={"stage": "mitiq_zne"},
        )[0]
        return solution_probability(counts, solutions)

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
    fold_seed: int = 0,
) -> float:
    """
    Mitigate ``P(solution)`` with Mitiq's folding and extrapolation, one job.

    Every scale is folded up front (Mitiq's ``fold_gates_at_random`` through
    :func:`mitiq_fold_isa`), all folded circuits are sampled in a single
    :func:`dgssp.runtime.sample_counts` call, and the factory extrapolates the
    measured values offline.

    Parameters
    ----------
    tqc:
        An **already transpiled** (ISA-level) D-G circuit.
    backend:
        The noisy backend to execute on.
    solution_bitstrings:
        Outcomes counted as correct.
    shots:
        Shots per folded circuit.
    scale_factors:
        Noise scale factors.
    factory:
        A Mitiq factory *instance* whose class's static ``extrapolate`` is
        used (its stored options, e.g. ``ExpFactory``'s asymptote, are passed
        through).  Defaults to ``LinearFactory``.
    scale_noise:
        Custom ``(circuit, scale) -> circuit`` folding function.  Defaults to
        :func:`mitiq_fold_isa` with ``fold_seed``.
    seed_simulator:
        Simulator seed, applied only on simulators.
    fold_seed:
        Seed of the default folding.

    Returns
    -------
    float
        The zero-noise-extrapolated probability of sampling a solution.  Not
        clipped to ``[0, 1]``: a value outside that interval is diagnostic
        information about the fit and is deliberately preserved.

    Raises
    ------
    ImportError
        If Mitiq is not installed.
    """
    _require_mitiq()
    from mitiq.zne.inference import LinearFactory

    solutions = list(solution_bitstrings)
    scales = [float(s) for s in scale_factors]

    def default_folder(circuit: QuantumCircuit, scale: float) -> QuantumCircuit:
        return mitiq_fold_isa(circuit, scale, seed=fold_seed)

    folder = scale_noise or default_folder
    circuits = [rebase_to_backend(tqc if s == 1.0 else folder(tqc, s), backend) for s in scales]
    seed = seed_simulator if is_simulator(backend) else None
    counts_list = sample_counts(
        backend, circuits, shots=shots, seed_simulator=seed, meta={"stage": "mitiq_zne"}
    )
    values = [solution_probability(c, solutions) for c in counts_list]

    fac = factory or LinearFactory(scale_factors=scales)
    options = dict(getattr(fac, "_options", {}) or {})
    return float(type(fac).extrapolate(scales, values, **options))


__all__ = [
    "mitiq_is_available",
    "mitiq_fold_isa",
    "mitiq_extrapolate",
    "mitiq_bootstrap",
    "make_sampling_executor",
    "mitiq_zne_solution_probability",
]
