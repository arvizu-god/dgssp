"""
transpilation.py

Layout selection for noisy backends.

The D-G circuit is dominated by two-qubit gates (the controlled-phase adder
and the multi-controlled gates), so which physical qubits the circuit lands on
matters more than almost anything else.  SABRE layout is stochastic in its
seed, so this module sweeps seeds and keeps the layout that minimises the
*calibration-weighted* two-qubit error rather than the raw gate count.

The layout found here is meant to be searched **once** and then reused --
notably by the ZNE pipeline in :mod:`dgssp.mitigation.zne`, which folds the
already-transpiled circuit instead of re-searching a layout at every noise
scale.

Errors are read from ``backend.target`` and qubit positions from
``QuantumCircuit.find_bit(...).index``; the removed V1 ``properties()`` API and
the private ``Qubit._index`` attribute are no longer used.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

BackendLike = BackendV2


# ---------------------------------------------------------------------------
# Two-qubit error accounting
# ---------------------------------------------------------------------------


@dataclass
class TwoQubitErrorReport:
    """
    Per-pair two-qubit error breakdown for one transpiled circuit.

    Attributes
    ----------
    accumulated_error:
        Total two-qubit error over every two-qubit gate application.
    gate_count:
        Number of two-qubit gate applications.
    pairs:
        The distinct physical qubit pairs used, in first-seen order.
    error_per_pair:
        Single-application error for each pair (parallel to ``pairs``).
    accumulated_per_pair:
        Error summed over all uses of each pair (parallel to ``pairs``).
    missing_calibration:
        Number of two-qubit applications with no calibration entry.
    """

    accumulated_error: float
    gate_count: int
    pairs: list[tuple[int, int]]
    error_per_pair: list[float]
    accumulated_per_pair: list[float]
    missing_calibration: int = 0


def two_qubit_gate_errors_per_circuit_layout(
    circuit: QuantumCircuit, backend: BackendLike
) -> TwoQubitErrorReport:
    """
    Accumulate two-qubit calibration error for a transpiled circuit.

    Parameters
    ----------
    circuit:
        An ISA-level circuit already transpiled for ``backend``.
    backend:
        A BackendV2 exposing a calibrated ``target``.

    Returns
    -------
    TwoQubitErrorReport
        Total and per-pair error, gate count and missing-calibration count.
        If the backend has no target, an all-zero report is returned so that
        seed sweeps degrade to "any layout" rather than crashing.
    """
    target = getattr(backend, "target", None)
    if target is None:
        return TwoQubitErrorReport(0.0, 0, [], [], [], 0)

    pairs: list[tuple[int, int]] = []
    error_per_pair: list[float] = []
    accumulated_per_pair: list[float] = []
    index_of: dict[tuple[int, int], int] = {}
    gate_count = 0
    missing = 0

    for instruction in circuit.data:
        op = instruction.operation
        if op.num_qubits != 2 or op.name in ("barrier", "delay"):
            continue

        gate_count += 1
        pair = tuple(circuit.find_bit(q).index for q in instruction.qubits)

        error = 0.0
        props_map = target.get(op.name)
        props = props_map.get(pair) if props_map is not None else None
        raw = getattr(props, "error", None) if props is not None else None
        if raw is None:
            missing += 1
        else:
            error = float(raw)

        if pair not in index_of:
            index_of[pair] = len(pairs)
            pairs.append(pair)  # type: ignore[arg-type]
            error_per_pair.append(error)
            accumulated_per_pair.append(error)
        else:
            accumulated_per_pair[index_of[pair]] += error

    return TwoQubitErrorReport(
        accumulated_error=sum(accumulated_per_pair),
        gate_count=gate_count,
        pairs=pairs,  # type: ignore[arg-type]
        error_per_pair=error_per_pair,
        accumulated_per_pair=accumulated_per_pair,
        missing_calibration=missing,
    )


# ---------------------------------------------------------------------------
# Seed sweep
# ---------------------------------------------------------------------------


@dataclass
class BestSeedResult:
    """
    Outcome of a SABRE seed sweep.

    Attributes
    ----------
    circuit:
        The transpiled circuit produced by the winning seed.
    best_seed:
        The transpiler seed that produced it.
    total_two_qubit_error:
        Its accumulated two-qubit calibration error.
    two_qubit_gate_count:
        Its two-qubit gate count.
    optimization_level:
        The preset optimization level used during the sweep, recorded so that
        downstream code (e.g. ZNE) knows not to re-optimize.
    """

    circuit: QuantumCircuit
    best_seed: int
    total_two_qubit_error: float
    two_qubit_gate_count: int
    optimization_level: int = 3


def find_best_seed(
    circuit: QuantumCircuit,
    backend: BackendLike,
    *,
    seed_min: int = 0,
    seed_max: int = 128,
    optimization_level: int = 3,
    layout_method: str = "sabre",
) -> BestSeedResult:
    """
    Sweep transpiler seeds and keep the lowest-two-qubit-error layout.

    Parameters
    ----------
    circuit:
        The logical circuit to transpile.
    backend:
        The backend to transpile for.
    seed_min, seed_max:
        Half-open seed range ``range(seed_min, seed_max)``.  A range of one
        seed effectively disables the sweep.
    optimization_level:
        Preset optimization level.
    layout_method:
        Layout method passed to the preset pass manager.

    Returns
    -------
    BestSeedResult
        The winning circuit and its metrics.  Ties are broken by fewer
        two-qubit gates.

    Raises
    ------
    ValueError
        If the seed range is empty.
    """
    if seed_max <= seed_min:
        raise ValueError(
            f"Empty seed range: seed_min={seed_min}, seed_max={seed_max}."
        )

    best_circuit: QuantumCircuit | None = None
    best_seed: int | None = None
    best_error = float("inf")
    best_count = 0

    for seed in range(seed_min, seed_max):
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=seed,
            layout_method=layout_method,
        )
        candidate = pm.run(circuit)
        report = two_qubit_gate_errors_per_circuit_layout(candidate, backend)

        better = report.accumulated_error < best_error or (
            report.accumulated_error == best_error
            and best_circuit is not None
            and report.gate_count < best_count
        )
        if best_circuit is None or better:
            best_circuit = candidate
            best_seed = seed
            best_error = report.accumulated_error
            best_count = report.gate_count

    assert best_circuit is not None and best_seed is not None
    return BestSeedResult(
        circuit=best_circuit,
        best_seed=best_seed,
        total_two_qubit_error=best_error,
        two_qubit_gate_count=best_count,
        optimization_level=optimization_level,
    )


__all__ = [
    "BackendLike",
    "TwoQubitErrorReport",
    "BestSeedResult",
    "two_qubit_gate_errors_per_circuit_layout",
    "find_best_seed",
]
