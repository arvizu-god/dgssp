"""
backends.py

Backend construction and evaluation utilities for the SSP library.

This module:

1. Creates an ideal (noiseless) AerSimulator backend.
2. Lists all real IBM hardware backends visible to the user's account.
3. Builds a fake noisy AerSimulator for each real backend.
4. Provides tools to evaluate *fake* and *real* backends for a given
   quantum circuit using:
   - accumulated errors (1q, 2q, readout, total),
   - probability of sampling known solution bitstrings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Sequence, Tuple

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .instance import SubsetSumInstance
from .solvers.base import BaseClassicalSSPSolver
from .solvers.classical import DPSSPSolver, DPConfig

import numpy as np

# NOTE: ibm_service.py is assumed to be at project root (as in your setup)
from ibm_service import get_ibm_service


# ---------------------------------------------------------------------------
# Types & basic backend construction
# ---------------------------------------------------------------------------

BackendLike = BackendV2


@dataclass
class BackendSelectionConfig:
    """
    Configuration for building and listing backends.

    Attributes
    ----------
    config_path:
        Path to the JSON file with IBM Quantum credentials, used by
        `get_ibm_service`. Default: ./ibm-quantum-account.json
    min_qubits:
        Minimum number of qubits required for a device to be considered
        a "real" backend. If None, no filter on qubit count is applied.
    seed_simulator:
        Optional seed for simulators (ideal Aer + fake noisy Aers).
    """

    config_path: Path = Path("./ibm-quantum-account.json")
    min_qubits: Optional[int] = None
    seed_simulator: Optional[int] = None
    real: bool = False


def build_service(config: BackendSelectionConfig) -> QiskitRuntimeService:
    """Build a QiskitRuntimeService using credentials in `config.config_path`."""
    return get_ibm_service(
        config_path=config.config_path,
        ensure_saved=True,
        overwrite=True,
    )


def _is_simulator(backend: BackendLike) -> bool:
    """Best-effort check if a backend is a simulator."""
    config = backend.configuration()
    return getattr(config, "simulator", False)


def _has_qubits(backend: BackendLike, min_qubits: Optional[int]) -> bool:
    """Check if backend has at least `min_qubits` logical qubits."""
    if min_qubits is None:
        return True
    return getattr(backend.configuration(), "num_qubits", 0) >= min_qubits

def _is_backend_usable(backend: BackendLike) -> bool:
    """
    Heuristic for whether a backend is usable right now.

    - Must be operational.
    - Must not be in maintenance (status_msg contains 'maint').
    """
    status = backend.status()

    # Not functional
    if not getattr(status, "operational", False):
        return False

    # In maintenance (e.g. 'maintenance', 'under maintenance')
    msg = (getattr(status, "status_msg", "") or "").lower()
    if "maint" in msg:  # catches 'maintenance', 'maint', etc.
        return False

    return True


def list_real_backends(
    service: QiskitRuntimeService,
    *,
    min_qubits: Optional[int] = None,
) -> List[BackendLike]:
    """
    List all real IBM hardware backends visible to the given service,
    skipping simulators, devices with too few qubits, and devices that
    are non-functional or in maintenance.
    """
    backends: List[BackendLike] = []

    for backend in service.backends():
        # Skip simulators
        if _is_simulator(backend):
            continue

        # Skip devices with insufficient qubits
        if not _has_qubits(backend, min_qubits):
            continue

        # Skip devices that are not operational or in maintenance
        if not _is_backend_usable(backend):
            continue

        status = backend.status()

        if status.status_msg != 'maintenance':
            backends.append(backend)

    return backends



def build_ideal_aer_backend(
    *,
    seed_simulator: Optional[int] = None,
) -> BackendLike:
    """Build a noiseless local AerSimulator backend."""
    sim = AerSimulator()
    if seed_simulator is not None:
        sim.set_options(seed_simulator=int(seed_simulator))
    return sim


def build_fake_backends_from_real(
    real_backends: List[BackendLike],
    *,
    seed_simulator: Optional[int] = None,
) -> List[BackendLike]:
    """
    Build a fake noisy AerSimulator for each real backend.
    """
    fake_backends: List[BackendLike] = []

    for rb in real_backends:
        sim = AerSimulator.from_backend(rb)
        if seed_simulator is not None:
            sim.set_options(seed_simulator=int(seed_simulator))
        fake_backends.append(sim)

    return fake_backends


def build_all_backends(
    config: BackendSelectionConfig,
) -> Dict[str, Any]:
    """
    Build all relevant backends:

    - service: QiskitRuntimeService,
    - ideal: a single noiseless AerSimulator,
    - real_backends: list of real IBM devices accessible to the account,
    - fake_backends: list of noisy AerSimulators, one per real backend.

    This function does *not* pick a "best" backend; it just prepares
    everything so that higher-level logic can evaluate and choose.
    """
    service = build_service(config)
    real=config.real

    ideal_backend = build_ideal_aer_backend(
        seed_simulator=config.seed_simulator,
    )

    real_backends = list_real_backends(
            service,
            min_qubits=config.min_qubits,
    )

    fake_backends = build_fake_backends_from_real(
        real_backends,
        seed_simulator=config.seed_simulator,
    )

    backends_dict={}

    if real==True:
        backends_dict={"service": service,
        "ideal": ideal_backend,
        "real_backends": real_backends,
        "fake_backends": fake_backends,
        }
    else:
        backends_dict={
        "service": service,
        "ideal": ideal_backend,
        "real_backends": [],
        "fake_backends": fake_backends,
    }

    return backends_dict


# ---------------------------------------------------------------------------
# Error metrics for a transpiled circuit on a given backend
# ---------------------------------------------------------------------------

@dataclass
class BackendErrorMetrics:
    total_error: float
    two_qubit_error: float
    single_qubit_error: float
    readout_error: float
    single_qubit_gate_count: int
    two_qubit_gate_count: int


def compute_accumulated_errors(
    backend: BackendLike,
    qc: QuantumCircuit,
) -> BackendErrorMetrics:
    """
    Compute accumulated single-qubit, two-qubit and readout errors for a
    transpiled circuit on a given backend.

    This is essentially your `accumulated_errors` method, adapted to a
    functional style and made a bit more robust.
    """
    properties = backend.properties()

    #num_qubits = qc.num_qubits
    #layout = getattr(qc, "layout", None)

    # Try to get the physical qubit indices actually used in the layout
    #qubit_layout: List[int]
    #try:
        #if layout is not None and getattr(layout, "initial_layout", None) is not None:
            #phys = layout.initial_layout.get_physical_bits()
            #qubit_layout = list(phys.keys())[:num_qubits]
        #else:
            #qubit_layout = list(range(num_qubits))
    #except Exception:
        #qubit_layout = list(range(num_qubits))

    acc_single_qubit_error = 0.0
    acc_two_qubit_error = 0.0
    acc_readout_error = 0.0
    single_qubit_gate_count = 0
    two_qubit_gate_count = 0

    # Readout errors
    #for q in qubit_layout:
        #try:
            #acc_readout_error += properties.readout_error(q)
        #except Exception:
            # If for some reason readout_error is not available, skip it
            #pass

    measured_qubits = {
        qc.find_bit(qubit).index
        for instruction in qc.data
        if instruction.operation.name == 'measure'
        for qubit in instruction.qubits
    }
    acc_readout_error = sum(properties.readout_error(q) for q in measured_qubits)

    # Gate errors
    for instruction in qc.data:
        op = instruction.operation
        if op.num_qubits == 1 and op.name != "measure":
            index = instruction.qubits[0]._index
            try:
                acc_single_qubit_error += properties.gate_error(
                    gate=op.name,
                    qubits=[index],
                )
            except Exception:
                pass
            single_qubit_gate_count += 1
        elif op.num_qubits == 2:
            pair = [instruction.qubits[0]._index, instruction.qubits[1]._index]
            try:
                acc_two_qubit_error += properties.gate_error(
                    gate=op.name,
                    qubits=pair,
                )
            except Exception:
                pass
            two_qubit_gate_count += 1

    acc_total_error = acc_single_qubit_error + acc_two_qubit_error + acc_readout_error

    return BackendErrorMetrics(
        total_error=acc_total_error,
        two_qubit_error=acc_two_qubit_error,
        single_qubit_error=acc_single_qubit_error,
        readout_error=acc_readout_error,
        single_qubit_gate_count=single_qubit_gate_count,
        two_qubit_gate_count=two_qubit_gate_count,
    )


# ---------------------------------------------------------------------------
# Execution & solution probability
# ---------------------------------------------------------------------------

@dataclass
class BackendPerformance:
    backend: BackendLike
    backend_name: str
    is_real: bool
    transpiled_circuit: QuantumCircuit
    errors: BackendErrorMetrics
    counts: Dict[str, int]
    solution_probability: float


def run_circuit_on_backend(
    backend: BackendLike,
    qc: QuantumCircuit,
    *,
    shots: int = 10_000,
    seed_transpiler: Optional[int] = None,
) -> Tuple[QuantumCircuit, Dict[str, int]]:
    """
    Transpile a circuit for a backend and execute it using SamplerV2.
    Returns (transpiled_circuit, counts).
    """
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=0,
        seed_transpiler=seed_transpiler,
    )
    tqc = pm.run(qc)

    sampler = SamplerV2(mode=backend)
    job = sampler.run([tqc], shots=shots)
    pub_result = job.result()[0]
    counts = pub_result.join_data().get_counts()

    return tqc, counts


def solution_probability(
    counts: Dict[str, int],
    solution_states: Sequence[str],
) -> float:
    """
    Compute the probability of sampling one of the given solution bitstrings
    from the counts dictionary.
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    solution_hits = sum(counts.get(state, 0) for state in solution_states)
    return solution_hits / total_shots

def indices_to_bitstring(indices: Sequence[int], n_bits: int) -> str:
    """
    Convert a list of item indices into a Qiskit-style bitstring.

    Convention:
    - index 0 corresponds to the least significant qubit/bit.
    - Qiskit counts bitstrings are msb→lsb, so we build an lsb-first list,
      then reverse it before joining.
    """
    bits_lsb_first = ["0"] * n_bits
    for idx in indices:
        if idx < 0 or idx >= n_bits:
            raise ValueError(
                f"Index {idx} is out of range for n_bits={n_bits}."
            )
        bits_lsb_first[idx] = "1"
    return "".join(reversed(bits_lsb_first))



# ---------------------------------------------------------------------------
# High-level evaluation & "best backend" selection
# ---------------------------------------------------------------------------

def evaluate_backends_for_circuit(
    instance: SubsetSumInstance,
    qc: QuantumCircuit,
    *,
    real_backends: List[BackendLike],
    fake_backends: List[BackendLike],
    classical_solver: BaseClassicalSSPSolver | None = None,
    shots: int = 10_000,
    noise_seed: int = 42,
) -> Dict[str, List[BackendPerformance]]:
    """
    Evaluate real and fake backends for a given DG-SSP circuit and instance.

    Steps:
      1) Use a classical SSP solver (DP by default) to find all exact solutions
         for `instance`.
      2) Convert those solutions to measurement bitstrings for the DG circuit.
      3) For each backend:
         - transpile the circuit,
         - compute accumulated error metrics on the transpiled circuit,
         - execute it and obtain counts,
         - compute the probability of sampling a solution bitstring.

    Parameters
    ----------
    instance:
        The SubsetSumInstance corresponding to the DG circuit.
        (n_items determines the length of solution bitstrings.)
    qc:
        Quantum circuit (e.g. the DG-SSP circuit built from `instance`).
    real_backends:
        List of real IBM hardware backends.
    fake_backends:
        List of fake noisy simulators, one per real backend.
    classical_solver:
        Classical SSP solver to use. If None, a DPSSPSolver with
        enumerate_all=True is used.
    shots:
        Number of shots per execution.
    noise_seed:
        Seed used for the transpiler (and indirectly for simulators).

    Returns
    -------
    Dict[str, List[BackendPerformance]]
        {
          "real": [BackendPerformance, ...],
          "fake": [BackendPerformance, ...],
        }
    """
    # ------------------------------------------------------------------
    # 1) Run classical solver to obtain exact SSP solutions
    # ------------------------------------------------------------------
    if classical_solver is None:
        classical_solver = DPSSPSolver(DPConfig(enumerate_all=True))

    dp_result = classical_solver.solve(instance)

    # All exact solutions from DP
    exact_solutions = [sol for sol in dp_result.all_solutions if sol.is_exact]
    if not exact_solutions and dp_result.best_solution is not None:
        # Fallback: use best_solution if it is exact
        if dp_result.best_solution.is_exact:
            exact_solutions = [dp_result.best_solution]

    # Convert exact solutions to bitstrings understood by Qiskit counts
    solution_states: List[str] = [
        indices_to_bitstring(sol.indices, instance.n_items)
        for sol in exact_solutions
    ]
    # Remove duplicates, keep deterministic order
    solution_states = sorted(set(solution_states))

    # ------------------------------------------------------------------
    # 2) Evaluate each backend
    # ------------------------------------------------------------------
    performances_real: List[BackendPerformance] = []
    performances_fake: List[BackendPerformance] = []

    # Real hardware
    for rb in real_backends:
        name_attr = getattr(rb, "name", None)
        rb_name = name_attr() if callable(name_attr) else (name_attr or str(rb))

        tqc, counts = run_circuit_on_backend(
            rb,
            qc,
            shots=shots,
            seed_transpiler=noise_seed,
        )
        errors = compute_accumulated_errors(rb, tqc)
        prob = solution_probability(counts, solution_states)

        perf = BackendPerformance(
            backend=rb,
            backend_name=rb_name,
            is_real=True,
            transpiled_circuit=tqc,
            errors=errors,
            counts=counts,
            solution_probability=prob,
        )
        performances_real.append(perf)

    # Fake noisy simulators
    for fb in fake_backends:
        name_attr = getattr(fb, "name", None)
        fb_name = name_attr() if callable(name_attr) else (name_attr or str(fb))

        tqc, counts = run_circuit_on_backend(
            fb,
            qc,
            shots=shots,
            seed_transpiler=noise_seed,
        )
        errors = compute_accumulated_errors(fb, tqc)
        prob = solution_probability(counts, solution_states)

        perf = BackendPerformance(
            backend=fb,
            backend_name=fb_name,
            is_real=False,
            transpiled_circuit=tqc,
            errors=errors,
            counts=counts,
            solution_probability=prob,
        )
        performances_fake.append(perf)

    return {
        "real": performances_real,
        "fake": performances_fake,
    }


def select_best_backends(
    real_performances: List[BackendPerformance],
    fake_performances: List[BackendPerformance],
) -> Dict[str, Optional[BackendPerformance]]:
    """
    Select the best real and fake backends according to:

    - smallest total accumulated error,
    - largest solution probability.

    Returns
    -------
    Dict[str, Optional[BackendPerformance]]
        {
          "best_real_by_error": ... or None,
          "best_fake_by_error": ... or None,
          "best_real_by_probability": ... or None,
          "best_fake_by_probability": ... or None,
        }
    """
    best_real_by_error = (
        min(real_performances, key=lambda p: p.errors.total_error)
        if real_performances
        else None
    )
    best_fake_by_error = (
        min(fake_performances, key=lambda p: p.errors.total_error)
        if fake_performances
        else None
    )

    best_real_by_probability = (
        max(real_performances, key=lambda p: p.solution_probability)
        if real_performances
        else None
    )
    best_fake_by_probability = (
        max(fake_performances, key=lambda p: p.solution_probability)
        if fake_performances
        else None
    )

    return {
        "best_real_by_error": best_real_by_error,
        "best_fake_by_error": best_fake_by_error,
        "best_real_by_probability": best_real_by_probability,
        "best_fake_by_probability": best_fake_by_probability,
    }


__all__ = [
    "BackendLike",
    "BackendSelectionConfig",
    "build_service",
    "list_real_backends",
    "build_ideal_aer_backend",
    "build_fake_backends_from_real",
    "build_all_backends",
    "BackendErrorMetrics",
    "BackendPerformance",
    "compute_accumulated_errors",
    "run_circuit_on_backend",
    "solution_probability",
    "evaluate_backends_for_circuit",
    "select_best_backends",
]