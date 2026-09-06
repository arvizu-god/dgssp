"""
backends.py

Backend discovery, noise-model construction and calibration-based scoring.

Responsibilities:

1. Build an ideal (noiseless) ``AerSimulator``.
2. List the real IBM devices an account can reach, filtered for usability.
3. Derive a noisy ``AerSimulator`` from each real device.
4. Score a *transpiled* circuit against a backend's calibration data, and rank
   backends by accumulated error or by measured solution probability.

Two deliberate changes from the previous version:

* Error metrics read ``backend.target`` (the BackendV2 calibration store)
  rather than the removed V1 ``properties()`` API, and use
  ``QuantumCircuit.find_bit(...).index`` instead of the private ``_index``
  attribute to resolve physical qubits.
* Missing calibration entries are *counted*, not silently treated as zero
  error.  Silent zeros used to bias backend ranking toward devices with
  incomplete calibration data.

Credential handling lives in :mod:`dgssp.runtime`; this module only consumes a
service that was handed to it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from qiskit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from .decoding import exact_solution_bitstrings, solution_probability
from .instance import SubsetSumInstance
from .runtime import backend_name, is_simulator, sample_counts
from .solvers.base import BaseClassicalSSPSolver
from .solvers.classical import DPConfig, DPSSPSolver

#: Alias used throughout the library for "a modern Qiskit backend".
BackendLike: TypeAlias = BackendV2


# ---------------------------------------------------------------------------
# Selection configuration
# ---------------------------------------------------------------------------


@dataclass
class BackendSelectionConfig:
    """
    Which backends to build and how to filter them.

    Attributes
    ----------
    min_qubits:
        Minimum device width required for a real backend to be considered.
        ``None`` disables the filter.
    seed_simulator:
        Seed applied to every simulator built here, for reproducibility.
    real:
        Whether to include real hardware in the bundle.  When ``False``, only
        the ideal simulator and the noisy fake simulators are returned.
    """

    min_qubits: int | None = None
    seed_simulator: int | None = None
    real: bool = False


# ---------------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------------


def _has_qubits(backend: BackendLike, min_qubits: int | None) -> bool:
    """Whether ``backend`` is at least ``min_qubits`` wide (V2-first)."""
    if min_qubits is None:
        return True
    n = getattr(backend, "num_qubits", None)
    if n is None:  # pragma: no cover - legacy V1 fallback
        cfg = getattr(backend, "configuration", None)
        n = getattr(cfg(), "num_qubits", 0) if callable(cfg) else 0
    return int(n or 0) >= min_qubits


def _is_backend_usable(backend: BackendLike) -> bool:
    """
    Whether a device is operational and not in maintenance right now.

    Parameters
    ----------
    backend:
        A real IBM backend.

    Returns
    -------
    bool
        ``False`` if the device reports non-operational status or a status
        message mentioning maintenance.
    """
    try:
        status = backend.status()
    except Exception:  # pragma: no cover - transient provider errors
        return False

    if not getattr(status, "operational", False):
        return False

    msg = (getattr(status, "status_msg", "") or "").lower()
    return "maint" not in msg


def list_real_backends(
    service: Any, *, min_qubits: int | None = None
) -> list[BackendLike]:
    """
    List usable real devices visible to a service.

    Parameters
    ----------
    service:
        A ``QiskitRuntimeService`` (see :func:`dgssp.runtime.get_service`).
    min_qubits:
        Minimum device width, or ``None`` for no filter.

    Returns
    -------
    list[BackendV2]
        Non-simulator devices that are wide enough, operational and not under
        maintenance.
    """
    return [
        backend
        for backend in service.backends()
        if not is_simulator(backend)
        and _has_qubits(backend, min_qubits)
        and _is_backend_usable(backend)
    ]


def build_ideal_aer_backend(*, seed_simulator: int | None = None) -> BackendLike:
    """
    Build a noiseless local ``AerSimulator``.

    Parameters
    ----------
    seed_simulator:
        Optional RNG seed for reproducible sampling.

    Returns
    -------
    AerSimulator
        An ideal simulator backend.
    """
    sim = AerSimulator()
    if seed_simulator is not None:
        sim.set_options(seed_simulator=int(seed_simulator))
    return sim


def build_fake_backends_from_real(
    real_backends: Sequence[BackendLike], *, seed_simulator: int | None = None
) -> list[BackendLike]:
    """
    Derive one noisy ``AerSimulator`` per real device.

    Parameters
    ----------
    real_backends:
        Devices whose noise models should be cloned.
    seed_simulator:
        Optional RNG seed applied to each simulator.

    Returns
    -------
    list[AerSimulator]
        Noisy simulators, in the same order as the input.
    """
    fakes: list[BackendLike] = []
    for real in real_backends:
        sim = AerSimulator.from_backend(real)
        if seed_simulator is not None:
            sim.set_options(seed_simulator=int(seed_simulator))
        fakes.append(sim)
    return fakes


def build_all_backends(
    config: BackendSelectionConfig, *, service: Any | None = None
) -> dict[str, Any]:
    """
    Assemble the ideal / real / fake backend bundle used by the executors.

    Parameters
    ----------
    config:
        Filtering and seeding options.
    service:
        An existing ``QiskitRuntimeService``.  If ``None``, no remote lookup is
        attempted and only the ideal simulator is returned -- this keeps the
        function usable (and offline) in tests and CI.

    Returns
    -------
    dict
        ``{"service": ..., "ideal": AerSimulator, "real_backends": [...],
        "fake_backends": [...]}``.  ``real_backends`` is empty unless
        ``config.real`` is ``True``.
    """
    ideal = build_ideal_aer_backend(seed_simulator=config.seed_simulator)

    if service is None:
        return {
            "service": None,
            "ideal": ideal,
            "real_backends": [],
            "fake_backends": [],
        }

    reals = list_real_backends(service, min_qubits=config.min_qubits)
    fakes = build_fake_backends_from_real(
        reals, seed_simulator=config.seed_simulator
    )

    return {
        "service": service,
        "ideal": ideal,
        "real_backends": list(reals) if config.real else [],
        "fake_backends": fakes,
    }


# ---------------------------------------------------------------------------
# Topology / calibration introspection
# ---------------------------------------------------------------------------

#: IBM processor families laid out on the heavy-hexagonal lattice.  The older
#: Canary / Penguin families (<= 20 qubits) are *not* heavy-hex even though
#: they also cap out at degree three, which is why family metadata is preferred
#: over the topology heuristic in :func:`is_heavy_hex`.
HEAVY_HEX_FAMILIES = frozenset(
    {"falcon", "hummingbird", "eagle", "egret", "heron", "condor"}
)

#: Smallest device width at which IBM shipped heavy-hex hardware (Falcon r4).
#: Used as the cut-off for the topology fallback in :func:`is_heavy_hex`.
HEAVY_HEX_MIN_QUBITS = 27


def processor_family(backend: BackendLike) -> str | None:
    """
    Read a backend's processor family, if it advertises one.

    Parameters
    ----------
    backend:
        A Qiskit backend.  Real ``IBMBackend`` objects and the bundled fake
        devices expose ``processor_type = {"family": ..., "revision": ...}``;
        Aer simulators do not.

    Returns
    -------
    str | None
        The lower-cased family name (``"eagle"``, ``"heron"``, ...), or
        ``None`` when the backend advertises no family.
    """
    try:
        info = getattr(backend, "processor_type", None)
    except Exception:  # pragma: no cover - defensive against provider errors
        return None
    if not isinstance(info, dict):
        return None
    family = info.get("family")
    return str(family).lower() if family else None


def coupling_degrees(backend: BackendLike) -> dict[int, int]:
    """
    Undirected degree of every qubit in a backend's coupling map.

    Parameters
    ----------
    backend:
        A Qiskit backend.

    Returns
    -------
    dict[int, int]
        Qubit index -> number of distinct neighbours.  Empty when the backend
        exposes no coupling map (e.g. an all-to-all simulator).
    """
    cmap = getattr(backend, "coupling_map", None)
    if cmap is None:
        return {}
    try:
        edges = {tuple(sorted(edge)) for edge in cmap.get_edges()}
    except Exception:  # pragma: no cover - defensive
        return {}

    degrees: dict[int, int] = {}
    for a, b in edges:
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1
    return degrees


def is_heavy_hex(backend: BackendLike) -> bool:
    """
    Whether a backend sits on IBM's heavy-hexagonal lattice.

    Three conditions must all hold:

    1. at least :data:`HEAVY_HEX_MIN_QUBITS` qubits -- the small Falcon and
       Canary devices are sub-lattice fragments, not a heavy-hex lattice, and
       the ``Falcon`` family label alone does not distinguish a 7-qubit "H"
       from a 27-qubit heavy-hex unit cell;
    2. maximum coupling degree three, with at least one degree-three qubit --
       this rules out linear and ring topologies;
    3. if the backend advertises a processor family at all, that family is in
       :data:`HEAVY_HEX_FAMILIES` -- this rules out the 20-qubit Penguin-era
       devices, which also cap out at degree three.

    Conditions 2 and 3 are heuristics over the metadata Qiskit exposes, not a
    graph-isomorphism proof.

    Parameters
    ----------
    backend:
        A Qiskit backend.

    Returns
    -------
    bool
        ``True`` if the device is heavy-hex.
    """
    n_qubits = int(getattr(backend, "num_qubits", 0) or 0)
    if n_qubits < HEAVY_HEX_MIN_QUBITS:
        return False

    family = processor_family(backend)
    if family is not None and family not in HEAVY_HEX_FAMILIES:
        return False

    degrees = coupling_degrees(backend)
    if not degrees:
        return False
    return max(degrees.values()) <= 3 and any(d == 3 for d in degrees.values())


def calibration_timestamp(backend: BackendLike) -> str | None:
    """
    ISO-8601 timestamp of the calibration snapshot a backend carries.

    Recorded in every results file so a figure can be traced back to the
    calibration data it was produced under.

    Parameters
    ----------
    backend:
        A Qiskit backend.  Real devices and the bundled fake devices expose
        ``properties().last_update_date``; Aer simulators do not, so pass the
        *source* fake/real backend rather than the simulator derived from it.

    Returns
    -------
    str | None
        The calibration date as an ISO string, or ``None`` when the backend
        exposes no properties.
    """
    props_fn = getattr(backend, "properties", None)
    if not callable(props_fn):
        return None
    try:
        props = props_fn()
    except Exception:  # pragma: no cover - network / provider errors
        return None

    stamp = getattr(props, "last_update_date", None)
    if stamp is None:
        return None
    isoformat = getattr(stamp, "isoformat", None)
    return isoformat() if callable(isoformat) else str(stamp)


def list_fake_backends(
    *,
    min_qubits: int | None = None,
    max_qubits: int | None = None,
    heavy_hex_only: bool = False,
) -> list[BackendLike]:
    """
    List the BackendV2 fake devices bundled with ``qiskit-ibm-runtime``.

    Which fake devices ship changes between releases, so nothing in this
    repository hard-codes a device name: callers discover what the installed
    version actually provides.

    Parameters
    ----------
    min_qubits, max_qubits:
        Inclusive width bounds, or ``None`` for no bound.
    heavy_hex_only:
        Keep only devices for which :func:`is_heavy_hex` holds.

    Returns
    -------
    list[BackendV2]
        Matching fake backends, sorted by ``(num_qubits, name)`` so the choice
        is deterministic across runs and machines.

    Raises
    ------
    ImportError
        If ``qiskit_ibm_runtime.fake_provider`` is unavailable.
    """
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

    selected: list[BackendLike] = []
    for backend in FakeProviderForBackendV2().backends():
        n_qubits = int(getattr(backend, "num_qubits", 0) or 0)
        if min_qubits is not None and n_qubits < min_qubits:
            continue
        if max_qubits is not None and n_qubits > max_qubits:
            continue
        if heavy_hex_only and not is_heavy_hex(backend):
            continue
        selected.append(backend)

    return sorted(
        selected,
        key=lambda b: (int(getattr(b, "num_qubits", 0) or 0), backend_name(b)),
    )


def smallest_heavy_hex_fake_backend(*, min_qubits: int | None = None) -> BackendLike:
    """
    The narrowest bundled heavy-hex fake device, for local dry runs.

    Parameters
    ----------
    min_qubits:
        Require at least this many qubits (e.g. the width of the circuit under
        test).

    Returns
    -------
    BackendV2
        The narrowest matching heavy-hex fake backend.

    Raises
    ------
    RuntimeError
        If the installed ``qiskit-ibm-runtime`` ships no matching device.
    """
    candidates = list_fake_backends(min_qubits=min_qubits, heavy_hex_only=True)
    if not candidates:
        raise RuntimeError(
            f"No heavy-hex fake backend with >= {min_qubits} qubits in the "
            "installed qiskit-ibm-runtime."
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# Calibration-based error metrics
# ---------------------------------------------------------------------------


@dataclass
class BackendErrorMetrics:
    """
    Accumulated calibration error for a transpiled circuit on a backend.

    Attributes
    ----------
    total_error:
        Sum of the single-qubit, two-qubit and readout contributions.
    two_qubit_error:
        Summed error of every two-qubit gate application.
    single_qubit_error:
        Summed error of every single-qubit gate application.
    readout_error:
        Summed error of every measurement.
    single_qubit_gate_count:
        Number of single-qubit gate applications.
    two_qubit_gate_count:
        Number of two-qubit gate applications.
    missing_calibration:
        Number of instructions for which the target held no error figure.  A
        large value means the total is an *underestimate* and the comparison
        with other backends is unreliable.
    """

    total_error: float
    two_qubit_error: float
    single_qubit_error: float
    readout_error: float
    single_qubit_gate_count: int
    two_qubit_gate_count: int
    missing_calibration: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """
        Return the metrics as a plain JSON-serialisable dictionary.

        Returns
        -------
        dict
            One key per attribute.
        """
        return {
            "total_error": self.total_error,
            "two_qubit_error": self.two_qubit_error,
            "single_qubit_error": self.single_qubit_error,
            "readout_error": self.readout_error,
            "single_qubit_gate_count": self.single_qubit_gate_count,
            "two_qubit_gate_count": self.two_qubit_gate_count,
            "missing_calibration": self.missing_calibration,
        }


def _instruction_error(target: Any, name: str, qubits: tuple[int, ...]) -> float | None:
    """Look up an instruction's error in a ``Target``, or ``None`` if absent."""
    try:
        inst_map = target.get(name)
    except Exception:  # pragma: no cover - defensive
        return None
    if inst_map is None:
        return None
    props = inst_map.get(qubits)
    if props is None:
        return None
    err = getattr(props, "error", None)
    return float(err) if err is not None else None


def compute_accumulated_errors(
    backend: BackendLike, qc: QuantumCircuit
) -> BackendErrorMetrics:
    """
    Accumulate calibration error over a transpiled circuit.

    The circuit must already be transpiled for ``backend``: qubit positions are
    resolved with ``qc.find_bit(...).index`` and looked up in
    ``backend.target``, so a logical circuit would produce meaningless indices.

    Parameters
    ----------
    backend:
        A BackendV2 exposing a calibrated ``target``.
    qc:
        The transpiled (ISA-level) circuit to score.

    Returns
    -------
    BackendErrorMetrics
        Accumulated errors, gate counts and the number of instructions with no
        calibration entry.

    Raises
    ------
    AttributeError
        If the backend exposes no ``target``.
    """
    target = getattr(backend, "target", None)
    if target is None:
        raise AttributeError(
            f"Backend {backend_name(backend)!r} exposes no `target`; "
            "calibration-based metrics require a BackendV2."
        )

    acc_1q = acc_2q = acc_ro = 0.0
    n_1q = n_2q = 0
    missing = 0

    for instruction in qc.data:
        op = instruction.operation
        name = op.name
        if name in ("barrier", "delay"):
            continue

        qubits = tuple(qc.find_bit(q).index for q in instruction.qubits)

        if name == "measure":
            err = _instruction_error(target, "measure", qubits)
            if err is None:
                missing += 1
            else:
                acc_ro += err
            continue

        err = _instruction_error(target, name, qubits)
        if op.num_qubits == 1:
            n_1q += 1
            if err is None:
                missing += 1
            else:
                acc_1q += err
        elif op.num_qubits == 2:
            n_2q += 1
            if err is None:
                missing += 1
            else:
                acc_2q += err

    return BackendErrorMetrics(
        total_error=acc_1q + acc_2q + acc_ro,
        two_qubit_error=acc_2q,
        single_qubit_error=acc_1q,
        readout_error=acc_ro,
        single_qubit_gate_count=n_1q,
        two_qubit_gate_count=n_2q,
        missing_calibration=missing,
    )


# ---------------------------------------------------------------------------
# Execution & ranking
# ---------------------------------------------------------------------------


@dataclass
class BackendPerformance:
    """
    How one backend performed on one circuit.

    Attributes
    ----------
    backend:
        The backend object itself.
    backend_name:
        Its human-readable name.
    is_real:
        ``True`` for hardware, ``False`` for a noisy simulator.
    transpiled_circuit:
        The ISA-level circuit that was executed.
    errors:
        Accumulated calibration error for that circuit.
    counts:
        Raw measurement counts.
    solution_probability:
        Fraction of shots landing on a known-correct bitstring.
    """

    backend: BackendLike
    backend_name: str
    is_real: bool
    transpiled_circuit: QuantumCircuit
    errors: BackendErrorMetrics
    counts: dict[str, int]
    solution_probability: float


def run_circuit_on_backend(
    backend: BackendLike,
    qc: QuantumCircuit,
    *,
    shots: int = 10_000,
    seed_transpiler: int | None = None,
    optimization_level: int = 0,
    seed_simulator: int | None = None,
) -> tuple[QuantumCircuit, dict[str, int]]:
    """
    Transpile a circuit for a backend and sample it.

    Parameters
    ----------
    backend:
        Target backend.
    qc:
        Logical circuit.
    shots:
        Number of shots.
    seed_transpiler:
        Seed for the preset pass manager.
    optimization_level:
        Preset optimization level.
    seed_simulator:
        Seed forwarded to simulator backends.

    Returns
    -------
    tuple[QuantumCircuit, dict[str, int]]
        The transpiled circuit and its counts.
    """
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    tqc = pm.run(qc)
    counts = sample_counts(
        backend, tqc, shots=shots, seed_simulator=seed_simulator
    )[0]
    return tqc, counts


def evaluate_backends_for_circuit(
    instance: SubsetSumInstance,
    qc: QuantumCircuit,
    *,
    real_backends: Sequence[BackendLike] = (),
    fake_backends: Sequence[BackendLike] = (),
    classical_solver: BaseClassicalSSPSolver | None = None,
    shots: int = 10_000,
    noise_seed: int = 42,
) -> dict[str, list[BackendPerformance]]:
    """
    Score a set of backends on one D-G circuit against classical ground truth.

    For every backend the circuit is transpiled, scored against the backend's
    calibration data, executed, and the measured probability of landing on a
    DP-verified solution bitstring is recorded.

    Parameters
    ----------
    instance:
        The instance the circuit encodes (supplies the ground truth).
    qc:
        The logical D-G circuit.
    real_backends:
        Hardware devices to evaluate.
    fake_backends:
        Noisy simulators to evaluate.
    classical_solver:
        Solver used for ground truth; defaults to an exhaustive DP solver.
    shots:
        Shots per backend.
    noise_seed:
        Transpiler seed, applied identically to every backend so the
        comparison is fair.

    Returns
    -------
    dict[str, list[BackendPerformance]]
        ``{"real": [...], "fake": [...]}``.
    """
    solver = classical_solver or DPSSPSolver(DPConfig(enumerate_all=True))
    solution_states = exact_solution_bitstrings(instance, solver.solve(instance))

    def _evaluate(backend: BackendLike, is_real: bool) -> BackendPerformance:
        tqc, counts = run_circuit_on_backend(
            backend, qc, shots=shots, seed_transpiler=noise_seed
        )
        return BackendPerformance(
            backend=backend,
            backend_name=backend_name(backend),
            is_real=is_real,
            transpiled_circuit=tqc,
            errors=compute_accumulated_errors(backend, tqc),
            counts=counts,
            solution_probability=solution_probability(counts, solution_states),
        )

    return {
        "real": [_evaluate(b, True) for b in real_backends],
        "fake": [_evaluate(b, False) for b in fake_backends],
    }


def select_best_backends(
    real_performances: Sequence[BackendPerformance],
    fake_performances: Sequence[BackendPerformance],
) -> dict[str, BackendPerformance | None]:
    """
    Pick the best real and fake backend by each of two criteria.

    Parameters
    ----------
    real_performances:
        Results for hardware devices.
    fake_performances:
        Results for noisy simulators.

    Returns
    -------
    dict[str, BackendPerformance | None]
        Keys ``best_real_by_error``, ``best_fake_by_error``,
        ``best_real_by_probability`` and ``best_fake_by_probability``.  A value
        is ``None`` when the corresponding list is empty.
    """

    def _min_err(perfs: Sequence[BackendPerformance]):
        return min(perfs, key=lambda p: p.errors.total_error) if perfs else None

    def _max_prob(perfs: Sequence[BackendPerformance]):
        return max(perfs, key=lambda p: p.solution_probability) if perfs else None

    return {
        "best_real_by_error": _min_err(real_performances),
        "best_fake_by_error": _min_err(fake_performances),
        "best_real_by_probability": _max_prob(real_performances),
        "best_fake_by_probability": _max_prob(fake_performances),
    }


__all__ = [
    "BackendLike",
    "BackendSelectionConfig",
    "list_real_backends",
    "build_ideal_aer_backend",
    "build_fake_backends_from_real",
    "build_all_backends",
    "HEAVY_HEX_FAMILIES",
    "HEAVY_HEX_MIN_QUBITS",
    "processor_family",
    "coupling_degrees",
    "is_heavy_hex",
    "calibration_timestamp",
    "list_fake_backends",
    "smallest_heavy_hex_fake_backend",
    "BackendErrorMetrics",
    "BackendPerformance",
    "compute_accumulated_errors",
    "run_circuit_on_backend",
    "evaluate_backends_for_circuit",
    "select_best_backends",
]
