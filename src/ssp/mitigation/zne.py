# src/ssp/mitigation/zne.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional, Mapping

import numpy as np
from scipy.optimize import curve_fit

from qiskit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit_ibm_runtime import SamplerV2

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from ..transpilation import find_best_seed
from ..solvers import DGSSPSolver, DGConfig
from ..instance import SubsetSumInstance

BackendLike = BackendV2


# ---------------------------------------------------------------------------
# 1) Local folding
# ---------------------------------------------------------------------------

from qiskit import QuantumCircuit

def fold_local_circuit(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """Perform Zero-Noise local folding on each instruction.

    - scale_factor must be an odd positive integer (1, 3, 5, ...).
    - Measurement and barrier instructions are NOT folded, but they are
      copied to the new circuit on the *same* logical qubits/clbits
      (via an explicit bit mapping).
    """
    if scale_factor % 2 == 0 or scale_factor <= 0:
        raise ValueError("scale_factor must be an odd positive integer (1, 3, 5, ...)")

    # No folding needed: just return the original circuit as-is
    if scale_factor == 1:
        return circuit

    n_repeat = (scale_factor - 1) // 2

    # Recreate circuit with same quantum / classical registers
    qc_folded = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    # Build explicit mapping old_bit -> new_bit
    qubit_map = {old_q: new_q for old_q, new_q in zip(circuit.qubits, qc_folded.qubits)}
    clbit_map = {old_c: new_c for old_c, new_c in zip(circuit.clbits, qc_folded.clbits)}

    # Iterate over instructions in original circuit
    for instr in circuit.data:
        op = instr.operation
        old_qargs = instr.qubits
        old_cargs = instr.clbits

        # Map qubits / clbits from old circuit to new circuit
        qargs = [qubit_map[q] for q in old_qargs]
        cargs = [clbit_map[c] for c in old_cargs]

        # Do not fold measurement or barrier: copy once, in order
        if op.name in ("measure", "barrier"):
            qc_folded.append(op, qargs, cargs)
            continue

        # Original gate
        qc_folded.append(op, qargs, cargs)

        # Local folding: (G · G^†)^n_repeat
        for _ in range(n_repeat):
            qc_folded.append(op, qargs, cargs)
            qc_folded.append(op.inverse(), qargs, cargs)

    return qc_folded

# ---------------------------------------------------------------------------
# 2) ZNE fitting models (per scalar value)
# ---------------------------------------------------------------------------

def _linear_model(x, a, b):
    return a * x + b


def _quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c


def _exponential_model(x, a, b, c):
    return a * np.exp(-b * x) + c


def zne_fit_single_value(
    method: str,
    xdata: Sequence[float],
    ydata: Sequence[float],
) -> Tuple[float, np.ndarray, np.ndarray, callable]:
    """Fit noisy values vs noise scale and extrapolate to zero noise.

    Parameters
    ----------
    method:
        'linear', 'quadratic', or 'exponential'.
    xdata:
        Noise scaling factors (e.g. [1, 3, 5]).
    ydata:
        Observed values at each scale (e.g. probabilities for a bitstring).

    Returns
    -------
    zero_val:
        Extrapolated value at x=0.
    ydata_arr:
        ydata as numpy array (for diagnostics).
    popt:
        Fitted model parameters.
    fit_fn:
        Model function (callable) used for the fit.
    """
    xdata_arr = np.asarray(xdata, dtype=float)
    ydata_arr = np.asarray(ydata, dtype=float)

    if method == "linear":
        popt, _ = curve_fit(_linear_model, xdata_arr, ydata_arr)
        zero_val = _linear_model(0, *popt)
        fit_fn = _linear_model

    elif method == "quadratic":
        popt, _ = curve_fit(_quadratic_model, xdata_arr, ydata_arr)
        zero_val = _quadratic_model(0, *popt)
        fit_fn = _quadratic_model

    elif method == "exponential":
        popt, _ = curve_fit(
            _exponential_model,
            xdata_arr,
            ydata_arr,
            p0=(1.0, 0.1, 0.0),
            maxfev=5000,
        )
        zero_val = _exponential_model(0, *popt)
        fit_fn = _exponential_model

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Use 'linear', 'quadratic', or 'exponential'."
        )

    return float(zero_val), ydata_arr, popt, fit_fn

def _is_simulator_backend(backend: BackendLike) -> bool:
    """Return True if backend is a simulator (e.g. Aer / fake), False for real hardware."""
    # Generic check via configuration
    try:
        cfg = backend.configuration()
        return bool(getattr(cfg, "simulator", False))
    except Exception:
        return False



# ---------------------------------------------------------------------------
# 3) Config for sampling ZNE
# ---------------------------------------------------------------------------

@dataclass
class ZNESamplingConfig:
    """Configuration for ZNE on sampling algorithms (DG-SSP)."""

    scales: Sequence[int]                    # e.g. [1, 3, 5]
    shots_per_scale: int = 10_000
    method: str = "linear"                   # 'linear', 'quadratic', 'exponential'
    clip: bool = True                        # clip negative probabilities to 0
    renormalize: bool = True                 # enforce sum=1
    # Best-seed search (SABRE). If you want to skip, you can set seed_min == seed_max.
    seed_min: int = 0
    seed_max: int = 128
    optimization_level: int = 3
    layout_method: str = "sabre"
    # Simulator seed (ignored on real hardware)
    seed_simulator: Optional[int] = None


# ---------------------------------------------------------------------------
# 4) Helper: folded-circuit sampling with best SABRE layout
# ---------------------------------------------------------------------------

def _run_folded_circuit_sampling(
    circuit: QuantumCircuit,
    backend: BackendLike,
    scale: int,
    zne_cfg: ZNESamplingConfig,
) -> Dict[str, int]:
    # 1) Local folding
    folded = fold_local_circuit(circuit, scale)

    # 2) Best seed / SABRE layout
    best = find_best_seed(
        folded,
        backend,
        seed_min=zne_cfg.seed_min,
        seed_max=zne_cfg.seed_max,
        optimization_level=zne_cfg.optimization_level,
        layout_method=zne_cfg.layout_method,
    )
    qc_best = best.circuit

    # 3) Sampler options: only pass simulator seed on simulators
    options = None
    if zne_cfg.seed_simulator is not None and _is_simulator_backend(backend):
        options = {"simulator": {"seed_simulator": int(zne_cfg.seed_simulator)}}

    sampler = SamplerV2(mode=backend, options=options)
    job = sampler.run([qc_best], shots=zne_cfg.shots_per_scale)
    pub_result = job.result()[0]
    counts = pub_result.join_data().get_counts()
    return counts



# ---------------------------------------------------------------------------
# 5) ZNE-mitigated probability distribution (generic sampling circuit)
# ---------------------------------------------------------------------------

def zne_mitigated_distribution(
    circuit: QuantumCircuit,
    backend: BackendLike,
    zne_cfg: ZNESamplingConfig,
) -> Dict[str, float]:
    """
    Compute a ZNE-mitigated probability distribution for a sampling algorithm
    (e.g. the DG-SSP circuit) using local folding and curve-fit extrapolation
    on each bitstring probability.

    Steps
    -----
    For each scale s in zne_cfg.scales:
      1) Fold circuit locally with scale s.
      2) Transpile using SABRE + best-seed search.
      3) Run with SamplerV2 → counts_s(bit).

    Then:
      4) Convert counts_s(bit) → probabilities p_s(bit).
      5) For each bitstring bit in the union of supports across all scales:
         - fit p_s(bit) vs s with the chosen model (linear, quadratic, exponential),
         - extrapolate the fit to s=0 to get p_0(bit).
      6) Optionally clip negative p_0(bit) to 0 and renormalize so Σ_bit p_0(bit) = 1.

    Returns
    -------
    Dict[str, float]
        ZNE-mitigated probability distribution over all bitstrings that
        appeared in any of the noisy distributions.
    """
    scales = list(zne_cfg.scales)
    if not scales:
        raise ValueError("ZNESamplingConfig.scales must be a non-empty sequence.")
    if any(s % 2 == 0 or s <= 0 for s in scales):
        raise ValueError("All scales must be odd positive integers: 1, 3, 5, ...")

    # 1) Run folded circuits at each scale
    counts_per_scale: Dict[int, Dict[str, int]] = {}
    for s in scales:
        counts = _run_folded_circuit_sampling(circuit, backend, s, zne_cfg)
        counts_per_scale[s] = counts

    # 2) Convert counts -> probabilities
    probs_per_scale: Dict[int, Dict[str, float]] = {}
    all_bitstrings: set[str] = set()

    for s in scales:
        counts = counts_per_scale[s]
        shots = sum(counts.values())
        if shots == 0:
            raise ValueError(f"No shots collected at ZNE scale {s}.")

        probs = {bit: c / shots for bit, c in counts.items()}
        probs_per_scale[s] = probs
        all_bitstrings.update(probs.keys())

    # 3) ZNE extrapolation per bitstring
    mitigated: Dict[str, float] = {}
    xdata = np.array(scales, dtype=float)

    for bit in all_bitstrings:
        # probabilities for this bitstring at each scale
        ydata = [probs_per_scale[s].get(bit, 0.0) for s in scales]

        # If all zero, keep zero
        if np.allclose(ydata, 0.0):
            mitigated[bit] = 0.0
            continue

        zero_val, _, _, _ = zne_fit_single_value(
            method=zne_cfg.method,
            xdata=xdata,
            ydata=ydata,
        )
        mitigated[bit] = float(zero_val)

    # 4) Optional clipping and renormalization
    if zne_cfg.clip:
        for bit, val in mitigated.items():
            if val < 0.0:
                mitigated[bit] = 0.0

    if zne_cfg.renormalize:
        Z = sum(mitigated.values())
        if Z > 0:
            for bit in mitigated:
                mitigated[bit] /= Z

    return mitigated


# ---------------------------------------------------------------------------
# 6) DG-SSP convenience: from instance + DG config to ZNE-mitigated distr.
# ---------------------------------------------------------------------------

def dgssp_zne_mitigated_distribution(
    instance: SubsetSumInstance,
    dg_config: DGConfig,
    backend: BackendLike,
    zne_cfg: ZNESamplingConfig,
) -> Dict[str, float]:
    """
    High-level helper: build the Draper-Grover circuit for a given SSP
    instance and DGConfig, then compute its ZNE-mitigated probability
    distribution on the given backend.

    Parameters
    ----------
    instance:
        SubsetSumInstance defining the SSP problem.
    dg_config:
        DGConfig configuration for DGSSPSolver (iterations, assembly_type, ...).
    backend:
        Noisy backend (real IBM device or noisy simulator).
    zne_cfg:
        ZNESamplingConfig specifying ZNE scales, fit method, etc.

    Returns
    -------
    Dict[str, float]
        ZNE-mitigated probability distribution over DG-SSP measurement outcomes.
    """
    solver = DGSSPSolver(dg_config)
    qc = solver.build_circuit(instance)
    return zne_mitigated_distribution(qc, backend, zne_cfg)


__all__ = [
    "ZNESamplingConfig",
    "fold_local_circuit",
    "zne_fit_single_value",
    "zne_mitigated_distribution",
    "dgssp_zne_mitigated_distribution",
]
