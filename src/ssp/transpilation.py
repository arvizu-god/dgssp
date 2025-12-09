"""
transpilation.py

Transpilation utilities for the SSP library.

Main goal:
- Transpile a quantum circuit to a backend using SABRE layout.
- Sweep over transpiler seeds and select the seed that minimizes the
  accumulated two-qubit gate error, using backend-calibration data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from qiskit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Alias used throughout the library
BackendLike = BackendV2


# ---------------------------------------------------------------------------
# Two-qubit gate error accounting
# ---------------------------------------------------------------------------

def two_qubit_gate_errors_per_circuit_layout(
    circuit: QuantumCircuit,
    backend: BackendLike,
) -> Tuple[float, int, List[List[int]], List[float], List[float]]:
    """
    Calculate accumulated two-qubit gate errors and related metrics
    for a given circuit layout on a backend.

    Returns
    -------
    acc_two_qubit_error : float
        Total accumulated two-qubit gate error across the circuit.
    two_qubit_gate_count : int
        Number of two-qubit gate applications.
    pair_list : List[List[int]]
        Distinct qubit pairs used in the circuit.
    error_pair_list : List[float]
        Error per pair (single application).
    error_acc_pair_list : List[float]
        Accumulated error per pair across all uses in the circuit.
    """
    # If backend does not expose properties, we cannot compute anything meaningful
    if not hasattr(backend, "properties") or backend.properties() is None:
        return 0.0, 0, [], [], []

    properties = backend.properties()
    basis_gates = set(getattr(backend.configuration(), "basis_gates", []))

    # Prefer native 2q gate names used by IBM backends
    if "ecr" in basis_gates:
        default_two_qubit_gate = "ecr"
    elif "cz" in basis_gates:
        default_two_qubit_gate = "cz"
    else:
        default_two_qubit_gate = None  # will fall back to op.name

    pair_list: List[List[int]] = []
    error_pair_list: List[float] = []
    error_acc_pair_list: List[float] = []
    two_qubit_gate_count = 0

    for instruction in circuit.data:
        op = instruction.operation
        if op.num_qubits != 2:
            continue

        two_qubit_gate_count += 1

        # Physical qubit indices used by this gate in the transpiled circuit
        pair = [instruction.qubits[0]._index, instruction.qubits[1]._index]

        # Choose which gate name to query in properties
        gate_name_for_error = default_two_qubit_gate or op.name

        try:
            error_pair = properties.gate_error(gate=gate_name_for_error, qubits=pair)
        except Exception:
            # If gate_error is not defined, skip it
            error_pair = 0.0

        if pair not in pair_list:
            pair_list.append(pair)
            error_pair_list.append(error_pair)
            error_acc_pair_list.append(error_pair)
        else:
            pos = pair_list.index(pair)
            error_acc_pair_list[pos] += error_pair

    acc_two_qubit_error = sum(error_acc_pair_list)
    return (
        acc_two_qubit_error,
        two_qubit_gate_count,
        pair_list,
        error_pair_list,
        error_acc_pair_list,
    )


# ---------------------------------------------------------------------------
# Seed-sweep & SABRE-based best layout
# ---------------------------------------------------------------------------

@dataclass
class BestSeedResult:
    """
    Summary of the best-seed SABRE transpilation.

    Attributes
    ----------
    circuit:
        Transpiled circuit corresponding to the best seed.
    best_seed:
        Transpiler seed which minimized the accumulated two-qubit error.
    total_two_qubit_error:
        Accumulated two-qubit gate error for the best seed.
    two_qubit_gate_count:
        Number of two-qubit gates in the best-seed transpiled circuit.
    """

    circuit: QuantumCircuit
    best_seed: int
    total_two_qubit_error: float
    two_qubit_gate_count: int


def find_best_seed(
    circuit: QuantumCircuit,
    backend: BackendLike,
    *,
    seed_min: int = 0,
    seed_max: int = 500,
    optimization_level: int = 3,
    layout_method: str = "sabre",
) -> BestSeedResult:
    """
    Sweep over transpiler seeds to find the SABRE layout with minimal
    accumulated two-qubit gate error on the given backend.

    Parameters
    ----------
    circuit:
        The logical quantum circuit to transpile.
    backend:
        Backend on which the circuit will run (real or fake).
    seed_min, seed_max:
        Inclusive/exclusive bounds for the seed sweep range. The loop
        runs over seeds in range(seed_min, seed_max).
    optimization_level:
        Preset optimization level passed to `generate_preset_pass_manager`.
    layout_method:
        Layout method passed to the transpiler. For this library, "sabre"
        is the main choice.

    Returns
    -------
    BestSeedResult
        Dataclass containing the best transpiled circuit and metrics.
    """
    best_circuit: Optional[QuantumCircuit] = None
    best_seed: Optional[int] = None
    best_total_error: float = float("inf")
    best_two_qubit_gate_count: int = 0

    for seed_transpiler in range(seed_min, seed_max):
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            layout_method=layout_method,
        )

        # Transpile circuit for this seed
        circuit_opt_seed = pm.run(circuit)

        # Evaluate accumulated 2q gate error for this layout
        acc_total_error_seed, two_qubit_gate_count_seed, *_ = (
            two_qubit_gate_errors_per_circuit_layout(circuit_opt_seed, backend)
        )

        # Keep the seed with minimal total error (tie-breaker: fewer 2q gates)
        if acc_total_error_seed < best_total_error or (
            acc_total_error_seed == best_total_error
            and two_qubit_gate_count_seed < best_two_qubit_gate_count
        ):
            best_total_error = acc_total_error_seed
            best_circuit = circuit_opt_seed
            best_seed = seed_transpiler
            best_two_qubit_gate_count = two_qubit_gate_count_seed

    if best_circuit is None or best_seed is None:
        raise RuntimeError(
            "find_best_seed could not find a valid transpiled circuit. "
            "This is unexpected; check that the backend has valid properties."
        )

    return BestSeedResult(
        circuit=best_circuit,
        best_seed=best_seed,
        total_two_qubit_error=best_total_error,
        two_qubit_gate_count=best_two_qubit_gate_count,
    )


# Optional backwards-compatible wrapper (same signature/return as your original)
def finding_best_seed(
    circuit: QuantumCircuit,
    backend: BackendLike,
) -> Tuple[QuantumCircuit, int, float, int]:
    """
    Backwards-compatible wrapper around `find_best_seed`.

    Returns
    -------
    (circuit_opt_best_seed, best_seed_transpiler, min_err_acc_seed_loop, two_qubit_gate_count)
    """
    result = find_best_seed(circuit, backend)
    return (
        result.circuit,
        result.best_seed,
        result.total_two_qubit_error,
        result.two_qubit_gate_count,
    )


__all__ = [
    "BackendLike",
    "BestSeedResult",
    "two_qubit_gate_errors_per_circuit_layout",
    "find_best_seed",
    "finding_best_seed",
]
