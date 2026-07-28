"""
zne.py

Zero-noise extrapolation for a *sampling* algorithm.

Standard ZNE mitigates an expectation value.  The D-G algorithm produces a
distribution over bitstrings, so this module extrapolates **each bitstring's
probability independently** to zero noise, then clips and renormalizes the
result into a valid distribution.  Every fitting model (linear, quadratic,
exponential) is available and the choice is explicit, which makes the
extrapolation auditable rather than magic.

Two ordering bugs from the previous implementation are fixed here, and the fix
is the whole point of the module:

* **Transpile first, then fold.**  Folding a *logical* circuit and then
  transpiling at optimization level 3 lets the gate-cancellation passes undo
  the folds -- the noise never actually gets scaled.  Here the circuit is
  transpiled once to ISA level and the *physical* gates are folded, then
  submitted at optimization level 0 so nothing is re-optimized or re-routed.
* **Search the layout once.**  Re-running a seed sweep per noise scale changes
  the physical circuit between scales, which silently invalidates the
  extrapolation: the points being fitted no longer lie on one noise curve.
  The layout is now searched a single time and reused for every scale.

All scales are submitted as a single batched job via
:func:`dgssp.runtime.sample_counts`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import curve_fit

from ..instance import SubsetSumInstance
from ..runtime import is_simulator, sample_counts
from ..solvers import DGConfig, DGSSPSolver
from ..transpilation import find_best_seed

BackendLike = BackendV2


# ---------------------------------------------------------------------------
# 1) Folding at ISA level
# ---------------------------------------------------------------------------


def fold_transpiled(tqc: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """
    Fold an already-transpiled circuit's gates as ``G (G^dag G)^n``.

    Because the input is ISA-level, the folds land on the physical gates that
    actually carry the noise, and the fixed layout is preserved (the returned
    circuit keeps the same registers and the input's ``layout`` attribute).

    Parameters
    ----------
    tqc:
        A transpiled circuit.
    scale_factor:
        An odd positive integer.  ``1`` returns the circuit unchanged; ``3``
        triples each gate; ``5`` quintuples it, and so on.

    Returns
    -------
    QuantumCircuit
        The folded circuit.  Measurements, barriers, delays and resets are
        copied through once and never folded.

    Raises
    ------
    ValueError
        If ``scale_factor`` is not an odd positive integer.
    """
    if scale_factor <= 0 or scale_factor % 2 == 0:
        raise ValueError(
            f"scale_factor must be an odd positive integer (1, 3, 5, ...), "
            f"got {scale_factor}."
        )
    if scale_factor == 1:
        return tqc.copy()

    n_repeat = (scale_factor - 1) // 2
    folded = QuantumCircuit(*tqc.qregs, *tqc.cregs, name=f"{tqc.name}_x{scale_factor}")

    qubit_map = dict(zip(tqc.qubits, folded.qubits, strict=True))
    clbit_map = dict(zip(tqc.clbits, folded.clbits, strict=True))

    for instruction in tqc.data:
        op = instruction.operation
        qargs = [qubit_map[q] for q in instruction.qubits]
        cargs = [clbit_map[c] for c in instruction.clbits]

        folded.append(op, qargs, cargs)
        if op.name in ("measure", "barrier", "delay", "reset"):
            continue

        for _ in range(n_repeat):
            folded.append(op.inverse(), qargs, cargs)
            folded.append(op, qargs, cargs)

    # Preserve the layout so downstream code still sees a physical circuit.
    if getattr(tqc, "layout", None) is not None:
        folded._layout = tqc.layout  # noqa: SLF001 - no public setter exists
    return folded


def rebase_to_backend(circuit: QuantumCircuit, backend: BackendLike) -> QuantumCircuit:
    """
    Rewrite a folded circuit back into the backend's native gate set.

    Folding inserts ``op.inverse()`` gates, and the inverse of a native gate is
    not necessarily native itself: on IBM devices ``sx`` inverts to ``sxdg``,
    which the device (and the Aer noise model derived from it) cannot execute.
    Without this step a folded circuit fails at submission with an
    "unknown instruction" error.

    Only gate *definitions* are translated. Layout and routing are untouched --
    which is the entire point, since re-routing between noise scales is exactly
    the bug the transpile-once design exists to avoid.

    Parameters
    ----------
    circuit:
        A folded ISA-level circuit.
    backend:
        The backend whose ``target`` defines the native gate set.

    Returns
    -------
    QuantumCircuit
        The circuit expressed in the backend's basis.  If the backend has no
        target, or translation fails, the input is returned unchanged so that
        permissive simulators keep working.
    """
    target = getattr(backend, "target", None)
    if target is None:
        return circuit

    try:
        from qiskit.circuit.equivalence_library import (
            SessionEquivalenceLibrary as equivalence_library,
        )
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import BasisTranslator

        basis = list(target.operation_names)
        return PassManager(
            [BasisTranslator(equivalence_library, target_basis=basis)]
        ).run(circuit)
    except Exception:  # pragma: no cover - permissive simulators need no rebase
        return circuit


# ---------------------------------------------------------------------------
# 2) Extrapolation models
# ---------------------------------------------------------------------------


def _linear_model(x, a, b):
    """Linear model ``a*x + b``."""
    return a * x + b


def _quadratic_model(x, a, b, c):
    """Quadratic model ``a*x^2 + b*x + c``."""
    return a * x**2 + b * x + c


def _exponential_model(x, a, b, c):
    """Decaying-exponential model ``a*exp(-b*x) + c``."""
    return a * np.exp(-b * x) + c


_MODELS: dict[str, tuple[Callable[..., float], tuple | None]] = {
    "linear": (_linear_model, None),
    "quadratic": (_quadratic_model, None),
    "exponential": (_exponential_model, (1.0, 0.1, 0.0)),
}


def zne_fit_single_value(
    method: str, xdata: Sequence[float], ydata: Sequence[float]
) -> tuple[float, np.ndarray, np.ndarray, Callable[..., float]]:
    """
    Fit one scalar against noise scale and extrapolate it to zero noise.

    Parameters
    ----------
    method:
        ``"linear"``, ``"quadratic"`` or ``"exponential"``.
    xdata:
        Noise scale factors, e.g. ``[1, 3, 5]``.
    ydata:
        The observed value at each scale (here, one bitstring's probability).

    Returns
    -------
    tuple
        ``(zero_value, ydata_array, popt, model_fn)`` -- the extrapolated value
        at zero noise, the inputs as an array, the fitted parameters, and the
        model function used (the latter three are for diagnostics and plots).

    Raises
    ------
    ValueError
        If ``method`` is not a known model, or there are fewer data points
        than the model has parameters.
    """
    if method not in _MODELS:
        raise ValueError(
            f"Unknown method '{method}'. Use 'linear', 'quadratic' or 'exponential'."
        )

    model, p0 = _MODELS[method]
    x = np.asarray(xdata, dtype=float)
    y = np.asarray(ydata, dtype=float)

    n_params = {"linear": 2, "quadratic": 3, "exponential": 3}[method]
    if x.size < n_params:
        raise ValueError(
            f"Method '{method}' needs at least {n_params} noise scales, got {x.size}."
        )

    try:
        popt, _ = curve_fit(model, x, y, p0=p0, maxfev=5000)
    except RuntimeError:
        # Fit did not converge; fall back to the least-noisy observation so a
        # single pathological bitstring cannot abort a whole run.
        popt = None

    if popt is None:
        return float(y[int(np.argmin(x))]), y, np.array([]), model

    return float(model(0.0, *popt)), y, popt, model


# ---------------------------------------------------------------------------
# 3) Configuration
# ---------------------------------------------------------------------------


@dataclass
class ZNESamplingConfig:
    """
    Configuration for per-bitstring ZNE on a sampling algorithm.

    Attributes
    ----------
    scales:
        Odd positive noise-scale factors, e.g. ``[1, 3, 5]``.
    shots_per_scale:
        Shots collected at each scale.
    method:
        Extrapolation model: ``"linear"``, ``"quadratic"`` or
        ``"exponential"``.
    clip:
        Clip negative extrapolated probabilities to zero.
    renormalize:
        Rescale the mitigated distribution to sum to one.
    seed_min, seed_max:
        Seed range for the *single* SABRE layout search.  Set
        ``seed_max = seed_min + 1`` to skip the sweep.
    optimization_level:
        Optimization level for that one transpilation.  The folded circuits
        are always submitted at level 0.
    layout_method:
        Layout method for the search.
    seed_simulator:
        Simulator seed, applied only on simulators.
    """

    scales: Sequence[int] = field(default_factory=lambda: [1, 3, 5])
    shots_per_scale: int = 10_000
    method: str = "linear"
    clip: bool = True
    renormalize: bool = True
    seed_min: int = 0
    seed_max: int = 64
    optimization_level: int = 3
    layout_method: str = "sabre"
    seed_simulator: int | None = None


# ---------------------------------------------------------------------------
# 4) Pipeline
# ---------------------------------------------------------------------------


def transpile_once(
    circuit: QuantumCircuit, backend: BackendLike, zne_cfg: ZNESamplingConfig
) -> QuantumCircuit:
    """
    Produce the single ISA-level circuit that every noise scale will fold.

    Parameters
    ----------
    circuit:
        The logical circuit.
    backend:
        The target backend.
    zne_cfg:
        Supplies the seed range, optimization level and layout method.

    Returns
    -------
    QuantumCircuit
        The transpiled circuit with the best layout found by the sweep (or a
        single-seed transpilation when the range has length one).
    """
    if zne_cfg.seed_max <= zne_cfg.seed_min + 1:
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=zne_cfg.optimization_level,
            seed_transpiler=zne_cfg.seed_min,
            layout_method=zne_cfg.layout_method,
        )
        return pm.run(circuit)

    return find_best_seed(
        circuit,
        backend,
        seed_min=zne_cfg.seed_min,
        seed_max=zne_cfg.seed_max,
        optimization_level=zne_cfg.optimization_level,
        layout_method=zne_cfg.layout_method,
    ).circuit


def run_zne_scales(
    tqc: QuantumCircuit, backend: BackendLike, zne_cfg: ZNESamplingConfig
) -> dict[int, dict[str, int]]:
    """
    Fold a transpiled circuit at every scale and sample all of them in one job.

    Parameters
    ----------
    tqc:
        An ISA-level circuit (see :func:`transpile_once`).
    backend:
        The backend to sample on.
    zne_cfg:
        Supplies the scales, shots and simulator seed.

    Returns
    -------
    dict[int, dict[str, int]]
        Counts keyed by noise scale.

    Raises
    ------
    ValueError
        If ``scales`` is empty or contains a non-odd / non-positive value.
    """
    scales = list(zne_cfg.scales)
    if not scales:
        raise ValueError("ZNESamplingConfig.scales must be non-empty.")
    if any(s <= 0 or s % 2 == 0 for s in scales):
        raise ValueError("All ZNE scales must be odd positive integers (1, 3, 5, ...).")

    folded = [rebase_to_backend(fold_transpiled(tqc, s), backend) for s in scales]
    seed = zne_cfg.seed_simulator if is_simulator(backend) else None

    counts_list = sample_counts(
        backend,
        folded,
        shots=zne_cfg.shots_per_scale,
        seed_simulator=seed,
        meta={"stage": "zne", "scales": scales},
    )
    return dict(zip(scales, counts_list, strict=True))


def extrapolate_distribution(
    counts_per_scale: dict[int, dict[str, int]], zne_cfg: ZNESamplingConfig
) -> dict[str, float]:
    """
    Extrapolate a per-bitstring distribution to zero noise.

    Parameters
    ----------
    counts_per_scale:
        Counts keyed by noise scale (see :func:`run_zne_scales`).
    zne_cfg:
        Supplies the model, clipping and renormalization flags.

    Returns
    -------
    dict[str, float]
        The mitigated distribution over every bitstring seen at any scale.

    Raises
    ------
    ValueError
        If any scale collected zero shots.
    """
    scales = sorted(counts_per_scale)
    probs_per_scale: dict[int, dict[str, float]] = {}
    all_bitstrings: set[str] = set()

    for scale in scales:
        counts = counts_per_scale[scale]
        shots = sum(counts.values())
        if shots == 0:
            raise ValueError(f"No shots collected at ZNE scale {scale}.")
        probs = {bit: c / shots for bit, c in counts.items()}
        probs_per_scale[scale] = probs
        all_bitstrings.update(probs)

    mitigated: dict[str, float] = {}
    xdata = np.array(scales, dtype=float)

    for bit in all_bitstrings:
        ydata = [probs_per_scale[s].get(bit, 0.0) for s in scales]
        if np.allclose(ydata, 0.0):
            mitigated[bit] = 0.0
            continue
        zero_val, _, _, _ = zne_fit_single_value(zne_cfg.method, xdata, ydata)
        mitigated[bit] = float(zero_val)

    if zne_cfg.clip:
        mitigated = {b: max(0.0, v) for b, v in mitigated.items()}

    if zne_cfg.renormalize:
        total = sum(mitigated.values())
        if total > 0:
            mitigated = {b: v / total for b, v in mitigated.items()}

    return mitigated


def zne_mitigated_distribution(
    circuit: QuantumCircuit,
    backend: BackendLike,
    zne_cfg: ZNESamplingConfig,
    *,
    transpiled: QuantumCircuit | None = None,
) -> dict[str, float]:
    """
    Full ZNE pipeline: transpile once, fold, batch-sample, extrapolate.

    Parameters
    ----------
    circuit:
        The logical circuit (e.g. a D-G circuit).
    backend:
        A noisy backend -- real hardware or a noisy simulator.
    zne_cfg:
        ZNE settings.
    transpiled:
        An ISA-level circuit to reuse instead of transpiling again.  Pass this
        when the caller has already searched a layout, so the unmitigated and
        mitigated runs share exactly the same physical circuit.

    Returns
    -------
    dict[str, float]
        The zero-noise-extrapolated probability distribution.
    """
    tqc = transpiled if transpiled is not None else transpile_once(
        circuit, backend, zne_cfg
    )
    counts_per_scale = run_zne_scales(tqc, backend, zne_cfg)
    return extrapolate_distribution(counts_per_scale, zne_cfg)


def dgssp_zne_mitigated_distribution(
    instance: SubsetSumInstance,
    dg_config: DGConfig,
    backend: BackendLike,
    zne_cfg: ZNESamplingConfig,
    *,
    num_solutions: int | None = None,
) -> dict[str, float]:
    """
    Convenience wrapper: build the D-G circuit for an instance, then mitigate.

    Parameters
    ----------
    instance:
        The Subset Sum instance.
    dg_config:
        Solver configuration.
    backend:
        A noisy backend.
    zne_cfg:
        ZNE settings.
    num_solutions:
        Number of exact solutions, used when ``dg_config.iterations == "auto"``.

    Returns
    -------
    dict[str, float]
        The mitigated distribution over index-register bitstrings.
    """
    qc = DGSSPSolver(dg_config).build_circuit(instance, num_solutions=num_solutions)
    return zne_mitigated_distribution(qc, backend, zne_cfg)


__all__ = [
    "ZNESamplingConfig",
    "fold_transpiled",
    "rebase_to_backend",
    "zne_fit_single_value",
    "transpile_once",
    "run_zne_scales",
    "extrapolate_distribution",
    "zne_mitigated_distribution",
    "dgssp_zne_mitigated_distribution",
]


# Backwards-compatibility shim -------------------------------------------------
def fold_local_circuit(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """
    Deprecated alias for :func:`fold_transpiled`.

    Kept so existing notebooks keep running, but note the semantic difference:
    folding is only meaningful on an ISA-level circuit.  Folding a logical
    circuit and transpiling afterwards lets the optimizer cancel the folds.

    Parameters
    ----------
    circuit:
        Circuit to fold (should already be transpiled).
    scale_factor:
        Odd positive integer.

    Returns
    -------
    QuantumCircuit
        The folded circuit.
    """
    import warnings

    warnings.warn(
        "fold_local_circuit is deprecated; use fold_transpiled on an "
        "ISA-level circuit instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fold_transpiled(circuit, scale_factor)


__all__.append("fold_local_circuit")
