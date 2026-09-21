"""
zne.py

Zero-noise extrapolation for a *sampling* algorithm.

Standard ZNE mitigates an expectation value.  The D-G algorithm produces a
distribution over bitstrings, so this module offers two estimators and keeps
the choice explicit:

* **scalar** -- extrapolate one proportion, ``P(solution)``, by weighted least
  squares with binomial weights (linear, quadratic, Richardson or an
  exponential with a known asymptote).  Every fit carries its covariance,
  residuals, degrees of freedom, a bootstrap interval and a *resolvability*
  flag saying whether the folded signal was distinguishable from the
  random-guessing baseline at every scale.
* **pointwise** -- extrapolate *each bitstring's probability independently*,
  then map the result back to a distribution by clip-and-renormalise or by
  Euclidean projection onto the probability simplex.  This is the estimator
  whose failure on infeasible instances the paper characterises
  (:func:`pathology_diagnostic`).

Two ordering bugs from the thesis implementation stay fixed:

* **Transpile first, then fold.**  Folding a *logical* circuit and then
  transpiling at optimization level 3 lets the gate-cancellation passes undo
  the folds.  Here the circuit is transpiled once to ISA level, the *physical*
  gates are folded, and the folded circuits are submitted at optimization
  level 0 so nothing is re-optimised or re-routed.
* **Search the layout once.**  Re-running a seed sweep per noise scale changes
  the physical circuit between scales, and the fitted points no longer lie on
  one noise curve.

Noise scales are real numbers ``>= 1`` (:func:`fold_isa`): odd integers fold
every gate uniformly, fractional scales fold a seeded subset once more.  The
scale that is *fitted* is always the **realised** one -- the exact rational
ratio of two-qubit gate counts -- never the nominal one.

:func:`run_zne_batch` folds many instances at all their scales (plus Mitiq's
folds of the same circuits) and submits everything through one
:func:`dgssp.runtime.sample_counts` call: the monthly hardware run is one job.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import lru_cache
from typing import Any, Literal, TypeAlias

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import OptimizeWarning, curve_fit

from ..instance import SubsetSumInstance
from ..metrics import (
    counts_to_probs,
    distribution_solution_probability,
    percentile_interval,
    proportion_sigma,
    resample_counts,
    solution_probability,
    total_variation,
)
from ..runtime import backend_name, execution_mode, is_simulator, sample_counts
from ..solvers import DGConfig, DGSSPSolver
from ..transpilation import find_best_seed, two_qubit_count

BackendLike: TypeAlias = BackendV2

ZNEMode: TypeAlias = Literal["pointwise", "scalar"]
ZNEMethod: TypeAlias = Literal["linear", "quadratic", "richardson", "exponential"]
ZNERenorm: TypeAlias = Literal["clip", "simplex"]

ZNE_MODES: tuple[str, ...] = ("pointwise", "scalar")
ZNE_METHODS: tuple[str, ...] = ("linear", "quadratic", "richardson", "exponential")
ZNE_RENORMS: tuple[str, ...] = ("clip", "simplex")

#: Resolvability threshold, in standard errors: a scale is resolved when the
#: observable sits at least this far from the random-guessing baseline.
RESOLVABILITY_K: float = 2.0


# ---------------------------------------------------------------------------
# 1) Folding at ISA level
# ---------------------------------------------------------------------------

#: Instructions copied through once and never folded.
_UNFOLDABLE: frozenset[str] = frozenset({"measure", "barrier", "delay", "reset"})

#: Frame changes on IBM devices: zero duration, zero error.  They are folded
#: with the uniform layers (as they always were) but never picked for the
#: partial layer, because folding them adds no noise.  :func:`run_zne_batch`
#: takes the list from :func:`dgssp.backends.median_gate_errors` instead when
#: the backend is calibrated.
DEFAULT_VIRTUAL_GATES: tuple[str, ...] = ("rz",)

_SCALE_TOL = 1e-9


def _validate_scale(scale_factor: float) -> float:
    """A real scale factor ``>= 1``, or ``ValueError``."""
    lam = float(scale_factor)
    if not math.isfinite(lam) or lam < 1.0 - _SCALE_TOL:
        raise ValueError(f"scale_factor must be a real number >= 1, got {scale_factor}.")
    return max(lam, 1.0)


def fold_plan(scale_factor: float, n_gates: int) -> tuple[int, int]:
    """
    How a real scale is realised on a set of ``n_gates`` gates.

    Every gate is folded ``k = floor((lam - 1) / 2)`` times (``G (G^dag G)^k``,
    scale ``2k + 1``), then ``m = round(n_gates * (lam - 2k - 1) / 2)`` of them
    are folded once more.  The realised scale is ``2k + 1 + 2m / n_gates``,
    within ``1 / n_gates`` of ``lam``.

    Parameters
    ----------
    scale_factor:
        Nominal scale ``lam >= 1``.
    n_gates:
        Number of gates the scale is measured on.

    Returns
    -------
    tuple[int, int]
        ``(k, m)``.
    """
    lam = _validate_scale(scale_factor)
    k = int(math.floor((lam - 1.0) / 2.0 + _SCALE_TOL))
    frac = max(0.0, (lam - (2 * k + 1)) / 2.0)
    m = int(math.floor(frac * n_gates + 0.5))
    return k, min(m, max(n_gates, 0))


def realised_scale(scale_factor: float, n_gates: int) -> Fraction:
    """
    The exact scale :func:`fold_plan` realises on ``n_gates`` gates.

    Parameters
    ----------
    scale_factor:
        Nominal scale ``>= 1``.
    n_gates:
        Number of gates the scale is measured on (two-qubit gates for
        :func:`fold_isa`).  Must be positive.

    Returns
    -------
    fractions.Fraction
        ``(n_gates * (2k + 1) + 2m) / n_gates``.
    """
    if n_gates <= 0:
        raise ValueError(f"n_gates must be positive, got {n_gates}.")
    k, m = fold_plan(scale_factor, n_gates)
    return Fraction(n_gates * (2 * k + 1) + 2 * m, n_gates)


@dataclass(frozen=True)
class FoldResult:
    """
    A folded ISA circuit and the scale it actually realises.

    Attributes
    ----------
    circuit:
        The folded circuit (not yet rebased; see :func:`rebase_to_backend`).
    nominal_scale:
        The scale that was asked for.
    realised_scale:
        ``N_2q(folded) / N_2q(input)``, exact.  Falls back to the physical
        single-qubit ratio for a circuit with no two-qubit gates, and to
        ``2k + 1`` for a circuit with nothing foldable.  **This is the scale
        every fit uses.**
    realised_scale_1q:
        The same ratio on physical (non-virtual) single-qubit gates, or
        ``None`` when there are none.
    uniform_folds:
        ``k``: how many times every gate was folded.
    n_two_q, n_two_q_extra:
        Two-qubit gates in the input, and how many were folded once more.
    n_one_q_physical, n_one_q_extra:
        The same for physical single-qubit gates.
    seed:
        Seed of the partial-layer selection.
    """

    circuit: QuantumCircuit
    nominal_scale: float
    realised_scale: Fraction
    realised_scale_1q: Fraction | None
    uniform_folds: int
    n_two_q: int
    n_two_q_extra: int
    n_one_q_physical: int
    n_one_q_extra: int
    seed: int


def fold_isa(
    tqc: QuantumCircuit,
    scale_factor: float,
    *,
    seed: int = 0,
    virtual_gates: Sequence[str] = DEFAULT_VIRTUAL_GATES,
) -> FoldResult:
    """
    Fold an ISA circuit to a real noise scale and report the realised scale.

    The scale is measured on the **two-qubit** gate count, since ``eps2``
    dominates survival.  Every gate is folded ``k = floor((lam - 1) / 2)``
    times; then a seeded subset of ``m2 = round(N_2q (lam - 2k - 1) / 2)``
    two-qubit gates is folded once more.  So that single-qubit noise scales
    alongside, an independently seeded subset of the same *fraction* of the
    physical single-qubit gates is folded once more too; virtual gates
    (``rz``) are never in the partial layer.

    Odd-integer scales fold nothing partially and reproduce the previous
    integer-only implementation exactly.  For a fixed seed the partial subsets
    are prefixes of one permutation, so they are nested across scales: a
    larger scale folds a superset of a smaller one's gates.

    Parameters
    ----------
    tqc:
        A transpiled (ISA-level) circuit.
    scale_factor:
        Nominal scale ``lam >= 1``.
    seed:
        Seed of the partial-layer selection.  The output is a deterministic
        function of ``(tqc, scale_factor, seed)``.
    virtual_gates:
        Instruction names that are frame changes (never partially folded).

    Returns
    -------
    FoldResult
        The folded circuit and its realised scale.  Measurements, barriers,
        delays and resets are copied through once and never folded; the
        input's layout is preserved.

    Raises
    ------
    ValueError
        If ``scale_factor`` is not a real number ``>= 1``.
    """
    lam = _validate_scale(scale_factor)
    virtual = frozenset(virtual_gates)

    two: list[int] = []
    one: list[int] = []
    for idx, instruction in enumerate(tqc.data):
        op = instruction.operation
        if op.name in _UNFOLDABLE:
            continue
        if op.num_qubits == 2:
            two.append(idx)
        elif op.num_qubits == 1 and op.name not in virtual:
            one.append(idx)

    k, m2 = fold_plan(lam, len(two))
    _, m1 = fold_plan(lam, len(one))

    # Both permutations are always drawn, so the output depends only on
    # (circuit, lam, seed) and the subsets are nested across scales.
    rng = np.random.default_rng(seed)
    perm_two = rng.permutation(len(two))
    perm_one = rng.permutation(len(one))
    extra = {two[j] for j in perm_two[:m2]} | {one[j] for j in perm_one[:m1]}

    if len(two) > 0:
        realised = Fraction(len(two) * (2 * k + 1) + 2 * m2, len(two))
    elif len(one) > 0:
        realised = Fraction(len(one) * (2 * k + 1) + 2 * m1, len(one))
    else:
        realised = Fraction(2 * k + 1)
    realised_1q = (
        Fraction(len(one) * (2 * k + 1) + 2 * m1, len(one)) if len(one) > 0 else None
    )

    result = dict(
        nominal_scale=lam,
        realised_scale=realised,
        realised_scale_1q=realised_1q,
        uniform_folds=k,
        n_two_q=len(two),
        n_two_q_extra=m2,
        n_one_q_physical=len(one),
        n_one_q_extra=m1,
        seed=int(seed),
    )

    if k == 0 and not extra:
        return FoldResult(circuit=tqc.copy(), **result)  # type: ignore[arg-type]

    label = f"{2 * k + 1}" if not extra else f"{float(realised):.4g}"
    folded = QuantumCircuit(*tqc.qregs, *tqc.cregs, name=f"{tqc.name}_x{label}")

    qubit_map = dict(zip(tqc.qubits, folded.qubits, strict=True))
    clbit_map = dict(zip(tqc.clbits, folded.clbits, strict=True))

    for idx, instruction in enumerate(tqc.data):
        op = instruction.operation
        qargs = [qubit_map[q] for q in instruction.qubits]
        cargs = [clbit_map[c] for c in instruction.clbits]

        folded.append(op, qargs, cargs)
        if op.name in _UNFOLDABLE:
            continue

        for _ in range(k + (1 if idx in extra else 0)):
            folded.append(op.inverse(), qargs, cargs)
            folded.append(op, qargs, cargs)

    # Preserve the layout so downstream code still sees a physical circuit.
    if getattr(tqc, "layout", None) is not None:
        folded._layout = tqc.layout  # noqa: SLF001 - no public setter exists
    return FoldResult(circuit=folded, **result)  # type: ignore[arg-type]


def fold_transpiled(
    tqc: QuantumCircuit,
    scale_factor: float,
    *,
    seed: int = 0,
    virtual_gates: Sequence[str] = DEFAULT_VIRTUAL_GATES,
) -> QuantumCircuit:
    """
    Fold an already-transpiled circuit's gates as ``G (G^dag G)^n``.

    Because the input is ISA-level, the folds land on the physical gates that
    actually carry the noise, and the fixed layout is preserved.  This returns
    only the circuit; use :func:`fold_isa` to also get the realised scale,
    which is what a fit must use for fractional scales.

    Parameters
    ----------
    tqc:
        A transpiled circuit.
    scale_factor:
        A real number ``>= 1``.  ``1`` returns the circuit unchanged; odd
        integers fold every gate uniformly (``3`` triples each gate); other
        values add a seeded partial layer (see :func:`fold_isa`).
    seed:
        Seed of the partial-layer selection.
    virtual_gates:
        Frame-change gate names, never partially folded.

    Returns
    -------
    QuantumCircuit
        The folded circuit.  Measurements, barriers, delays and resets are
        copied through once and never folded.

    Raises
    ------
    ValueError
        If ``scale_factor`` is not a real number ``>= 1``.
    """
    return fold_isa(tqc, scale_factor, seed=seed, virtual_gates=virtual_gates).circuit


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
    the bug the transpile-once design exists to avoid.  The input's layout is
    re-attached to the output (the translation pass alone does not carry it).

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
        out = PassManager([BasisTranslator(equivalence_library, target_basis=basis)]).run(
            circuit
        )
    except Exception:  # pragma: no cover - permissive simulators need no rebase
        return circuit

    if getattr(out, "layout", None) is None and getattr(circuit, "layout", None) is not None:
        out._layout = circuit.layout  # noqa: SLF001 - no public setter exists
    return out


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


def _polynomial_model(x, *coefficients):
    """Polynomial with coefficients highest power first (Richardson's interpolant)."""
    return np.polyval(coefficients, x)


_MODELS: dict[str, tuple[Callable[..., float], tuple | None]] = {
    "linear": (_linear_model, None),
    "quadratic": (_quadratic_model, None),
    "exponential": (_exponential_model, (1.0, 0.1, 0.0)),
}

#: Number of free parameters per model (Richardson has one per scale).
_N_PARAMS: dict[str, int] = {"linear": 2, "quadratic": 3, "exponential": 3}

#: Polynomial degree for the models that are *linear in their parameters* and
#: therefore have a closed-form least-squares solution.
_POLYNOMIAL_DEGREE: dict[str, int] = {"linear": 1, "quadratic": 2}

#: Default upper bound on the exponential model's decay rate per unit noise
#: scale (:attr:`ZNESamplingConfig.exp_rate_max`).  ``exp(-10) = 4.5e-5`` per
#: unit scale is far below the smallest survival in the frozen family (0.0104
#: at n = 4, i.e. rate 4.6), so the bound never binds on an identified fit.
#:
#: **It is a display convention, not part of the estimator.**  When fewer
#: scales are resolved than the model has free parameters -- at n = 4 with
#: odd-integer folding only ``lambda = 1`` carries signal -- the amplitude and
#: the rate are not jointly identified: every pair obeying
#: ``b exp(-c lam_min) = obs(lam_min) - a`` fits the one informative point,
#: and once ``c`` is large they all predict the asymptote at every other
#: scale, so ``chi2`` is flat in ``c`` and its infimum sits at ``c -> inf``,
#: where the extrapolation ``a + b = a + d exp(c lam_min)`` diverges.  The
#: bound turns that divergence into a large finite number; ``rate_at_bound``
#: and ``identifiable`` record that it happened, and such a fit is never
#: reported (``reported_source == "min_scale"``).  Averaging the raw value
#: over such fits is meaningless: use the median, or condition on
#: ``resolved``.
EXP_RATE_MAX: float = 10.0

_GOLDEN: float = (math.sqrt(5.0) - 1.0) / 2.0


@lru_cache(maxsize=8)
def _rate_grid(rate_max: float) -> np.ndarray:
    """Rate search grid: zero plus 400 log-spaced points up to ``rate_max``."""
    return np.concatenate(([0.0], np.geomspace(1e-4, float(rate_max), 400)))


def richardson_weights(scales: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Lagrange weights of the degree ``len(scales) - 1`` extrapolation to zero.

    ``L_i(0) = prod_{j != i} x_j / (x_j - x_i)``, so the Richardson estimate is
    ``sum_i L_i(0) y_i``.

    Parameters
    ----------
    scales:
        At least two distinct noise scales.

    Returns
    -------
    numpy.ndarray
        One weight per scale; they sum to one.

    Raises
    ------
    ValueError
        With fewer than two scales, or repeated ones.
    """
    x = np.asarray(scales, dtype=float)
    if x.size < 2 or np.unique(x).size != x.size:
        raise ValueError(f"Richardson needs >= 2 distinct scales, got {list(x)}.")
    w = np.ones_like(x)
    for i in range(x.size):
        for j in range(x.size):
            if j != i:
                w[i] *= x[j] / (x[j] - x[i])
    return w


def _fit_polynomial(degree: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Ordinary least squares for a model that is linear in its parameters.

    ``np.vander(x, degree + 1)`` produces the columns ``[x**degree, ..., x, 1]``,
    so the returned coefficients are in the same order as the corresponding
    model function's arguments and ``popt`` keeps its meaning for callers.

    Parameters
    ----------
    degree:
        1 for the linear model, 2 for the quadratic one.
    x, y:
        Noise scales and observed values.

    Returns
    -------
    numpy.ndarray
        The fitted coefficients, highest power first.
    """
    design = np.vander(x, degree + 1)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return np.asarray(coefficients, dtype=float)


def zne_fit_single_value(
    method: str,
    xdata: Sequence[float] | np.ndarray,
    ydata: Sequence[float] | np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, Callable[..., float]]:
    """
    Fit one scalar against noise scale and extrapolate it to zero noise.

    Unweighted: this is the per-bitstring fit of the pointwise estimator.  The
    scalar estimator uses the weighted fits of :func:`fit_scalar_series`.

    Parameters
    ----------
    method:
        ``"linear"``, ``"quadratic"``, ``"richardson"`` or ``"exponential"``
        (free asymptote).
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

    Notes
    -----
    The linear and quadratic models are linear in their parameters, so they are
    solved in closed form by least squares rather than by a nonlinear
    optimiser.  That is exact, deterministic and independent of an initial
    guess -- which matters because this runs once *per bitstring*, on
    probabilities spanning several orders of magnitude, for every instance.
    Richardson is the Lagrange interpolant through every point, evaluated at
    zero.  Only the exponential model is genuinely nonlinear and still uses
    ``curve_fit``.

    With as many noise scales as the model has parameters the fit is exactly
    determined: the extrapolation is a valid Richardson estimator, but it has
    zero residual degrees of freedom, so the data say nothing about whether the
    model is right.  Use at least three scales for any claim about the
    extrapolation itself.
    """
    if method not in ZNE_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Use one of {', '.join(ZNE_METHODS)}."
        )

    x = np.asarray(xdata, dtype=float)
    y = np.asarray(ydata, dtype=float)

    if method == "richardson":
        weights = richardson_weights(x)
        coefficients = np.polyfit(x, y, x.size - 1)
        return float(weights @ y), y, np.asarray(coefficients), _polynomial_model

    model, p0 = _MODELS[method]
    n_params = _N_PARAMS[method]
    if x.size < n_params:
        raise ValueError(
            f"Method '{method}' needs at least {n_params} noise scales, got {x.size}."
        )

    degree = _POLYNOMIAL_DEGREE.get(method)
    if degree is not None:
        popt: np.ndarray | None = _fit_polynomial(degree, x, y)
    else:
        with warnings.catch_warnings():
            if x.size == n_params:
                # curve_fit always computes a parameter covariance, and warns
                # when it cannot.  With as many points as parameters it never
                # can -- there are no residual degrees of freedom.  We discard
                # the covariance anyway, so the warning reports on a quantity
                # that was never requested.  Suppressed *only* in that exactly
                # determined case, so a genuine fit failure at three or more
                # scales still surfaces.
                warnings.simplefilter("ignore", OptimizeWarning)
            try:
                popt, _ = curve_fit(model, x, y, p0=p0, maxfev=5000)
            except RuntimeError:
                # Fit did not converge; fall back to the least-noisy
                # observation so a single pathological bitstring cannot abort a
                # whole run.
                popt = None

    if popt is None:
        return float(y[int(np.argmin(x))]), y, np.array([]), model

    return float(model(0.0, *popt)), y, popt, model


def _safe_inverse(matrix: np.ndarray) -> np.ndarray | None:
    """
    Inverse of a normal matrix, or ``None`` when it is not a usable covariance.

    ``None`` for a singular matrix, non-finite entries, or a negative diagonal
    (numerically indefinite -- e.g. an exponential fit pinned at the rate
    bound, where the rate is not identified).
    """
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(inverse)) or np.any(np.diag(inverse) < 0):
        return None
    return inverse


def _exp_fixed_many(
    x: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    asymptote: float,
    rate_max: float = EXP_RATE_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted fit of ``a + b exp(-c x)`` with ``a`` fixed, for many series.

    Variable projection: for a fixed rate ``c`` the amplitude is linear,
    ``b(c) = S_yg / S_gg``, and ``chi2(c) = S_yy - S_yg^2 / S_gg``.  The
    one-dimensional minimisation over ``c in [0, rate_max]`` is a grid
    search followed by a vectorised golden-section refinement, so a whole
    bootstrap (thousands of series) is fitted in a few array operations and
    never depends on an initial guess.

    Parameters
    ----------
    x:
        Scales, shape ``(n,)``.
    Y:
        Observations, shape ``(B, n)``.
    W:
        Weights ``1 / sigma^2``, shape ``(B, n)`` or ``(1, n)``.
    asymptote:
        The fixed ``a``.
    rate_max:
        Upper bound of the rate search; see :data:`EXP_RATE_MAX`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Amplitudes ``b`` and rates ``c``, each of shape ``(B,)``.
    """
    W = np.broadcast_to(W, Y.shape)
    D = Y - asymptote
    WD = W * D
    Syy = np.sum(WD * D, axis=1)

    def chi2(c: np.ndarray) -> np.ndarray:
        g = np.exp(-np.outer(c, x))
        syg = np.sum(WD * g, axis=1)
        sgg = np.sum(W * g * g, axis=1)
        return Syy - syg**2 / sgg

    rates = _rate_grid(float(rate_max))
    G = np.exp(-np.outer(rates, x))
    grid = Syy[:, None] - (WD @ G.T) ** 2 / (W @ (G * G).T)
    best = np.argmin(grid, axis=1)
    last = rates.size - 1
    lo = rates[np.maximum(best - 1, 0)]
    hi = rates[np.minimum(best + 1, last)]
    for _ in range(60):
        c1 = hi - _GOLDEN * (hi - lo)
        c2 = lo + _GOLDEN * (hi - lo)
        left = chi2(c1) < chi2(c2)
        hi = np.where(left, c2, hi)
        lo = np.where(left, lo, c1)
    c = 0.5 * (lo + hi)
    grid_c = rates[best]
    c = np.where(chi2(c) <= chi2(grid_c), c, grid_c)

    g = np.exp(-np.outer(c, x))
    b = np.sum(WD * g, axis=1) / np.sum(W * g * g, axis=1)
    return b, c


def _fit_exponential_free(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray | None,
    rate_max: float = EXP_RATE_MAX,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """``curve_fit`` of ``amplitude * exp(-rate x) + asymptote``, or ``None``."""
    p0 = (float(y[0] - y[-1]) or 0.1, 0.5, float(y[-1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            popt, pcov = curve_fit(
                _exponential_model,
                x,
                y,
                p0=p0,
                sigma=sigma,
                absolute_sigma=sigma is not None,
                bounds=([-np.inf, 0.0, -np.inf], [np.inf, float(rate_max), np.inf]),
                maxfev=10_000,
            )
        except (RuntimeError, ValueError):
            return None
    cov = np.asarray(pcov, dtype=float) if np.all(np.isfinite(pcov)) else None
    return np.asarray(popt, dtype=float), cov


def _fit_series(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray | None = None,
    asymptote: float | None = None,
    rate_max: float = EXP_RATE_MAX,
) -> dict[str, Any]:
    """
    Fit one series and extrapolate it to zero noise, with full diagnostics.

    Weighted least squares when ``sigma`` is given (``chi2`` uses it), plain
    least squares otherwise (``cov`` and ``chi2`` are then ``None``).

    Returns
    -------
    dict
        ``{"method", "value", "value_se", "params", "param_names", "cov",
        "fitted", "residuals", "n_params", "dof", "chi2", "chi2_red",
        "fit_ok", "rate_at_bound"}``.
    """
    if method not in ZNE_METHODS:
        raise ValueError(f"Unknown method '{method}'. Use one of {', '.join(ZNE_METHODS)}.")
    n = x.size
    weights = None if sigma is None else 1.0 / sigma**2
    cov: np.ndarray | None = None
    value_se: float | None = None
    rate_at_bound: bool | None = None
    fit_ok = True

    if method in _POLYNOMIAL_DEGREE:
        degree = _POLYNOMIAL_DEGREE[method]
        n_params = degree + 1
        if n < n_params:
            raise ValueError(f"Method '{method}' needs >= {n_params} scales, got {n}.")
        X = np.vander(x, n_params)
        if weights is None:
            params = _fit_polynomial(degree, x, y)
        else:
            normal = X.T @ (weights[:, None] * X)
            params = np.linalg.solve(normal, X.T @ (weights * y))
            cov = _safe_inverse(normal)
        fitted = X @ params
        value = float(params[-1])
        names = ["slope", "intercept"] if degree == 1 else ["curvature", "slope", "intercept"]
        if cov is not None:
            value_se = float(np.sqrt(cov[-1, -1]))

    elif method == "richardson":
        n_params = n
        lagrange = richardson_weights(x)
        vander = np.vander(x, n)
        params = np.linalg.solve(vander, y)
        fitted = y.copy()
        value = float(lagrange @ y)
        names = [f"c{n - 1 - j}" for j in range(n)]
        if sigma is not None:
            inverse = np.linalg.inv(vander)
            cov = inverse @ np.diag(sigma**2) @ inverse.T
            value_se = float(np.sqrt(np.sum(lagrange**2 * sigma**2)))

    elif asymptote is not None:
        n_params = 2
        if n < n_params:
            raise ValueError(f"Exponential with a fixed asymptote needs >= 2 scales, got {n}.")
        w_row = (np.ones(n) if weights is None else weights)[None, :]
        b_arr, c_arr = _exp_fixed_many(x, y[None, :], w_row, float(asymptote), rate_max)
        b, c = float(b_arr[0]), float(c_arr[0])
        g = np.exp(-c * x)
        fitted = asymptote + b * g
        params = np.array([b, c])
        value = float(asymptote + b)
        names = ["amplitude", "rate"]
        rate_at_bound = bool(c >= rate_max * (1.0 - 1e-9))
        if sigma is not None:
            jac = np.column_stack([g, -b * x * g]) / sigma[:, None]
            cov = _safe_inverse(jac.T @ jac)
            if cov is not None:
                value_se = float(np.sqrt(cov[0, 0]))

    else:
        n_params = 3
        if n < n_params:
            raise ValueError(f"Exponential with a free asymptote needs >= 3 scales, got {n}.")
        fitted_free = _fit_exponential_free(x, y, sigma, rate_max)
        names = ["amplitude", "rate", "asymptote"]
        if fitted_free is None:
            fit_ok = False
            params = np.full(3, np.nan)
            fitted = np.full(n, np.nan)
            value = float("nan")
        else:
            params, cov = fitted_free
            fitted = _exponential_model(x, *params)
            value = float(params[0] + params[2])
            rate_at_bound = bool(params[1] >= rate_max * (1.0 - 1e-9))
            if cov is not None and sigma is not None:
                grad = np.array([1.0, 0.0, 1.0])
                value_se = float(np.sqrt(max(0.0, grad @ cov @ grad)))
            else:
                cov = None

    residuals = y - fitted
    dof = n - n_params
    chi2 = float(np.sum((residuals / sigma) ** 2)) if sigma is not None and fit_ok else None
    return {
        "method": method,
        "value": value,
        "value_se": value_se,
        "params": [float(p) for p in params],
        "param_names": names,
        "cov": None if cov is None else [[float(v) for v in row] for row in cov],
        "fitted": [float(v) for v in fitted],
        "residuals": [float(v) for v in residuals],
        "n_params": n_params,
        "dof": dof,
        "chi2": chi2,
        "chi2_red": chi2 / dof if (chi2 is not None and dof > 0) else None,
        "fit_ok": fit_ok,
        "rate_at_bound": rate_at_bound,
    }


def _sigma_matrix(Y: np.ndarray, shots: np.ndarray) -> np.ndarray:
    """Vectorised :func:`dgssp.metrics.proportion_sigma`."""
    smoothed = (Y * shots + 0.5) / (shots + 1.0)
    return np.sqrt(smoothed * (1.0 - smoothed) / shots)


def _extrapolate_many(
    method: str,
    x: np.ndarray,
    Y: np.ndarray,
    S: np.ndarray | None,
    asymptote: float | None,
    rate_max: float = EXP_RATE_MAX,
) -> np.ndarray:
    """
    Zero-noise values of many series at once (the bootstrap's inner loop).

    Same estimator as :func:`_fit_series`, row by row of ``Y`` (shape
    ``(B, n)``) with standard errors ``S`` (``None`` = unweighted).
    Non-converged free-asymptote fits come back as ``nan``.
    """
    if method in _POLYNOMIAL_DEGREE:
        X = np.vander(x, _POLYNOMIAL_DEGREE[method] + 1)
        if S is None:
            return Y @ np.linalg.pinv(X)[-1]
        W = 1.0 / S**2
        normal = np.einsum("bn,ni,nj->bij", W, X, X)
        rhs = np.einsum("bn,ni,bn->bi", W, X, Y)
        return np.linalg.solve(normal, rhs[..., None])[:, -1, 0]
    if method == "richardson":
        return Y @ richardson_weights(x)
    if asymptote is not None:
        W = np.ones_like(Y) if S is None else 1.0 / S**2
        b, _ = _exp_fixed_many(x, Y, W, float(asymptote), rate_max)
        return asymptote + b
    out = np.full(Y.shape[0], np.nan)
    for i, row in enumerate(Y):
        fitted = _fit_exponential_free(x, row, None if S is None else S[i], rate_max)
        if fitted is not None:
            out[i] = fitted[0][0] + fitted[0][2]
    return out


# ---------------------------------------------------------------------------
# 3) Configuration
# ---------------------------------------------------------------------------


@dataclass
class ZNESamplingConfig:
    """
    Configuration for ZNE on a sampling algorithm.

    Attributes
    ----------
    scales:
        Nominal noise scales, real numbers ``>= 1``, e.g. ``[1, 3, 5]`` or
        ``[1, 1.5, 2]``.  Fits always use the *realised* scales.
    shots_per_scale:
        Shots collected at each scale.
    method:
        Extrapolation model: ``"linear"``, ``"quadratic"``, ``"richardson"``
        (degree ``len(scales) - 1``, exact through every point) or
        ``"exponential"`` (``a + b exp(-c lam)``).
    clip, renormalize:
        Legacy flags of the pointwise ``renorm="clip"`` path: clip negative
        extrapolated probabilities to zero, then rescale to sum to one.
    seed_min, seed_max:
        Seed range for the *single* SABRE layout search.  Set
        ``seed_max = seed_min + 1`` to skip the sweep.  **The default range
        (0..63) is not the one WP3 ranked on (0..31)**: to reproduce a circuit
        recorded in ``paper/instances/hardware_subset.json``, pass
        ``seed_min=best_seed, seed_max=best_seed + 1``.
    optimization_level:
        Optimization level for that one transpilation.  The folded circuits
        are always submitted at level 0.
    layout_method:
        Layout method for the search.
    seed_simulator:
        Simulator seed, applied only on simulators.
    mode:
        ``"scalar"`` (extrapolate ``P(solution)``; the default) or
        ``"pointwise"`` (extrapolate every bitstring).  It decides which
        estimate a batch reports as *the* mitigated value; both are computed.
    renorm:
        How the pointwise estimate is mapped back to a distribution:
        ``"clip"`` (clip + renormalise) or ``"simplex"`` (Euclidean projection
        onto the probability simplex).  Pointwise only.
    asymptote:
        Exponential only.  A float fixes ``a`` (the paper uses ``P_uniform =
        M / 2**n``); ``"p_uniform"`` fixes it per instance to that instance's
        ``M / 2**n`` (the form a multi-instance batch needs); ``None`` fits it
        freely (needs >= 3 scales).  In pointwise mode any non-``None`` value
        fixes each bitstring's asymptote at the uniform ``2**-n``, the
        depolarising fixed point.
    fold_seed:
        Seed of the partial-folding selection (and of Mitiq's random folds).
    n_bootstrap:
        Bootstrap resamples per fit.
    bootstrap_seed:
        Seed of the bootstrap resampling.
    exp_rate_max:
        Upper bound of the exponential model's decay rate.  See
        :data:`EXP_RATE_MAX`: it bounds an estimator that is unbounded on a
        design resolving fewer scales than the model has parameters, and never
        binds otherwise.
    """

    scales: Sequence[float] = field(default_factory=lambda: [1, 3, 5])
    shots_per_scale: int = 10_000
    method: ZNEMethod = "linear"
    clip: bool = True
    renormalize: bool = True
    seed_min: int = 0
    seed_max: int = 64
    optimization_level: int = 3
    layout_method: str = "sabre"
    seed_simulator: int | None = None
    mode: ZNEMode = "scalar"
    renorm: ZNERenorm = "clip"
    asymptote: float | Literal["p_uniform"] | None = None
    fold_seed: int = 0
    n_bootstrap: int = 500
    bootstrap_seed: int = 0
    exp_rate_max: float = EXP_RATE_MAX

    def __post_init__(self) -> None:
        """Reject unknown names early, before any circuit is built."""
        if self.method not in ZNE_METHODS:
            raise ValueError(f"Unknown method '{self.method}'. Use one of {ZNE_METHODS}.")
        if self.mode not in ZNE_MODES:
            raise ValueError(f"Unknown mode '{self.mode}'. Use one of {ZNE_MODES}.")
        if self.renorm not in ZNE_RENORMS:
            raise ValueError(f"Unknown renorm '{self.renorm}'. Use one of {ZNE_RENORMS}.")
        if self.asymptote is not None and not (
            self.asymptote == "p_uniform" or isinstance(self.asymptote, (int, float))
        ):
            raise ValueError(
                f"asymptote must be a float, 'p_uniform' or None, got {self.asymptote!r}."
            )


def resolve_asymptote(cfg: ZNESamplingConfig, p_uniform: float | None) -> float | None:
    """
    The fixed asymptote a scalar exponential fit uses, or ``None``.

    Parameters
    ----------
    cfg:
        The config.  Only ``method == "exponential"`` has an asymptote.
    p_uniform:
        The instance's ``M / 2**n``; required when ``cfg.asymptote`` is
        ``"p_uniform"``.

    Returns
    -------
    float | None
        ``None`` for non-exponential methods and for a free asymptote.
    """
    if cfg.method != "exponential" or cfg.asymptote is None:
        return None
    if cfg.asymptote == "p_uniform":
        if p_uniform is None:
            raise ValueError("asymptote='p_uniform' needs the instance's p_uniform.")
        return float(p_uniform)
    return float(cfg.asymptote)


# ---------------------------------------------------------------------------
# 4) Scalar estimator
# ---------------------------------------------------------------------------


def resolved_scales_mask(
    values: Sequence[float],
    shots: Sequence[int],
    null_value: float,
    *,
    k: float = RESOLVABILITY_K,
) -> list[bool]:
    """
    Per scale: is the observable at least ``k`` standard errors from the null?

    How many entries are ``True`` is what decides whether a model is
    *identified*: a fit with fewer informative scales than free parameters is
    determined by noise, and the exponential is then not even bounded (see
    :data:`EXP_RATE_MAX`).

    Parameters
    ----------
    values:
        Observed proportions, one per scale.
    shots:
        Shots behind each.
    null_value:
        The random-guessing baseline ``P_uniform``.
    k:
        Threshold in standard errors.

    Returns
    -------
    list[bool]
        One flag per scale, in the order given.
    """
    return [
        abs(v - null_value) >= k * proportion_sigma(v, int(n))
        for v, n in zip(values, shots, strict=True)
    ]


def informative_scales_mask(
    values: Sequence[float],
    shots: Sequence[int],
    null_value: float,
    *,
    k: float = RESOLVABILITY_K,
) -> list[bool]:
    """
    Per scale: does it carry signal *in the direction the model decays from*?

    :func:`resolved_scales_mask` is two-sided -- it asks whether an
    observation is distinguishable from the null at all.  Identification asks
    something stricter: under a decaying model every scale lies on the same
    side of the asymptote as the least-noisy one, so a point that sits ``k``
    standard errors on the *other* side is a fluctuation, not information
    about the decay.  Counting it would make a fit look identified when the
    rate is in fact unconstrained (see :data:`EXP_RATE_MAX`).

    The reference direction is the sign of ``values[0] - null_value`` (the
    lowest scale, the least attenuated point).

    Parameters
    ----------
    values:
        Observed proportions, one per scale, lowest scale first.
    shots:
        Shots behind each.
    null_value:
        The random-guessing baseline ``P_uniform``.
    k:
        Threshold in standard errors.

    Returns
    -------
    list[bool]
        One flag per scale, in the order given.
    """
    if not values:
        return []
    sign = 1.0 if values[0] >= null_value else -1.0
    return [
        sign * (v - null_value) >= k * proportion_sigma(v, int(n))
        for v, n in zip(values, shots, strict=True)
    ]


def is_resolved(
    values: Sequence[float],
    shots: Sequence[int],
    null_value: float | None,
    *,
    k: float = RESOLVABILITY_K,
) -> bool | None:
    """
    Whether the observable is distinguishable from the null at every scale.

    ``resolved`` iff ``|obs(lam) - null| >= k * SE(lam)`` at *every* scale.
    ``SE`` is the binomial standard error of :func:`dgssp.metrics.
    proportion_sigma`, which is the exact standard deviation of the
    multinomial bootstrap of a proportion (Jeffreys-floored, so an observation
    of exactly the null never passes by ``0 >= 0``).  Using the closed form
    keeps the flag deterministic.

    Parameters
    ----------
    values:
        Observed proportions, one per scale.
    shots:
        Shots behind each.
    null_value:
        The random-guessing baseline ``P_uniform``.  ``None`` -> ``None``.
    k:
        Threshold in standard errors.

    Returns
    -------
    bool | None
        The flag, or ``None`` when no null was given.
    """
    if null_value is None:
        return None
    return all(resolved_scales_mask(values, shots, null_value, k=k))


def fit_scalar_series(
    scales: Sequence[float],
    values: Sequence[float],
    shots: Sequence[int],
    cfg: ZNESamplingConfig,
    *,
    p_uniform: float | None = None,
) -> dict[str, Any]:
    """
    Weighted fit of an observed proportion against noise scale.

    The core of the scalar estimator, on already-reduced data; see
    :func:`extrapolate_scalar` for the counts-level entry point.

    Parameters
    ----------
    scales:
        *Realised* noise scales.
    values:
        Observed proportion at each scale.
    shots:
        Shots behind each value; the fit weights are ``1 / sigma^2`` with the
        binomial ``sigma`` of :func:`dgssp.metrics.proportion_sigma`.
    cfg:
        Supplies ``method`` and ``asymptote``.
    p_uniform:
        The random-guessing baseline ``M / 2**n``: the null of the
        resolvability flag and the value of ``asymptote="p_uniform"``.

    Returns
    -------
    dict
        Everything :func:`_fit_series` returns (``value``, ``value_se``,
        ``params``, ``cov``, ``residuals``, ``dof``, ``chi2_red`` -- ``None``
        when ``dof == 0``, e.g. Richardson) plus ``scales``, ``observed``,
        ``sigma``, ``shots``, ``asymptote``, ``null_value``,
        ``resolvability_k``, ``value_at_min_scale`` and:

        ``resolved`` / ``resolved_scales`` / ``n_resolved_scales``
            Two-sided: is each observation distinguishable from the null?
            ``resolved`` is ``True`` only when every scale is.
        ``identifiable`` / ``informative_scales`` / ``n_informative_scales``
            Signed (:func:`informative_scales_mask`): are there at least as
            many scales carrying signal, on the same side of the asymptote as
            the lowest scale, as the model has free parameters?  A fit that is
            not identified is determined by noise, and an exponential one is
            not even bounded -- see :data:`EXP_RATE_MAX`.
        ``reported_value`` / ``reported_source``
            The extrapolation when the fit is resolved *and* identified (or no
            null was given), otherwise the lowest-scale observation.  An
            unresolved or unidentified fit still returns its raw numbers; the
            paper reports the ``lambda = 1`` value for it instead.
    """
    order = np.argsort(np.asarray(scales, dtype=float))
    x = np.asarray(scales, dtype=float)[order]
    y = np.asarray(values, dtype=float)[order]
    n = np.asarray(shots, dtype=float)[order]
    sigma = np.array([proportion_sigma(v, int(s)) for v, s in zip(y, n, strict=True)])
    asymptote = resolve_asymptote(cfg, p_uniform)

    fit = _fit_series(cfg.method, x, y, sigma, asymptote, cfg.exp_rate_max)
    if p_uniform is None:
        mask: list[bool] | None = None
        informative: list[bool] | None = None
        resolved: bool | None = None
        n_resolved: int | None = None
        n_informative: int | None = None
        identifiable: bool | None = None
    else:
        mask = resolved_scales_mask(y.tolist(), n.astype(int).tolist(), p_uniform)
        informative = informative_scales_mask(y.tolist(), n.astype(int).tolist(), p_uniform)
        resolved = all(mask)
        n_resolved = int(sum(mask))
        n_informative = int(sum(informative))
        identifiable = bool(n_informative >= fit["n_params"])
    use_fit = fit["fit_ok"] and resolved is not False and identifiable is not False
    fit.update(
        {
            "scales": x.tolist(),
            "observed": y.tolist(),
            "sigma": sigma.tolist(),
            "shots": n.astype(int).tolist(),
            "asymptote": asymptote,
            "null_value": p_uniform,
            "resolved": resolved,
            "resolved_scales": mask,
            "n_resolved_scales": n_resolved,
            "informative_scales": informative,
            "n_informative_scales": n_informative,
            "identifiable": identifiable,
            "resolvability_k": RESOLVABILITY_K,
            "exp_rate_max": cfg.exp_rate_max if cfg.method == "exponential" else None,
            "value_at_min_scale": float(y[0]),
            "reported_value": fit["value"] if use_fit else float(y[0]),
            "reported_source": "extrapolation" if use_fit else "min_scale",
        }
    )
    return fit


def _observed_series(
    counts_per_scale: Mapping[float, Mapping[str, int]],
    observable_fn: Callable[[Mapping[str, int]], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted scales, observed values and shot totals."""
    scales = sorted(counts_per_scale)
    shots = np.array([sum(counts_per_scale[s].values()) for s in scales], dtype=float)
    if np.any(shots == 0):
        bad = [s for s, n in zip(scales, shots, strict=True) if n == 0]
        raise ValueError(f"No shots collected at ZNE scale(s) {bad}.")
    values = np.array([float(observable_fn(counts_per_scale[s])) for s in scales])
    return np.asarray(scales, dtype=float), values, shots


def extrapolate_scalar(
    counts_per_scale: Mapping[float, Mapping[str, int]],
    observable_fn: Callable[[Mapping[str, int]], float],
    cfg: ZNESamplingConfig,
    *,
    p_uniform: float | None = None,
) -> dict[str, Any]:
    """
    Scalar ZNE: extrapolate one proportion to zero noise.

    Parameters
    ----------
    counts_per_scale:
        Counts keyed by *realised* scale (see :func:`run_zne_scales`).
    observable_fn:
        A proportion of the counts, typically
        ``lambda c: solution_probability(c, solutions)``.  The binomial
        weights assume it is the mass of a fixed outcome set.
    cfg:
        Supplies ``method`` and ``asymptote``.
    p_uniform:
        The random-guessing baseline (resolvability null; ``"p_uniform"``
        asymptote).

    Returns
    -------
    dict
        See :func:`fit_scalar_series`.

    Raises
    ------
    ValueError
        If any scale collected zero shots, or the method needs more scales.
    """
    x, y, n = _observed_series(counts_per_scale, observable_fn)
    return fit_scalar_series(x.tolist(), y.tolist(), n.astype(int).tolist(), cfg, p_uniform=p_uniform)


# ---------------------------------------------------------------------------
# 5) Pointwise estimator and renormalisation
# ---------------------------------------------------------------------------


def project_to_simplex(v: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto the probability simplex.

    Solves ``argmin_{p >= 0, sum p = 1} ||p - v||_2`` by the sorting-based
    ``O(K log K)`` algorithm of J. Duchi, S. Shalev-Shwartz, Y. Singer and
    T. Chandra, "Efficient projections onto the l1-ball for learning in high
    dimensions", ICML 2008: sort ``v`` descending as ``u``, let
    ``rho = max{j : u_j - (sum_{i<=j} u_i - 1) / j > 0}`` and
    ``theta = (sum_{i<=rho} u_i - 1) / rho``; then ``p = max(v - theta, 0)``.

    Because clip-and-renormalise also lands on the simplex, the projection is
    never farther from ``v`` in L2 than clip-and-renormalise is.  It is *not*
    guaranteed to be closer in total variation to any particular reference.

    Parameters
    ----------
    v:
        A vector, or a 2-D array projected row by row.

    Returns
    -------
    numpy.ndarray
        Same shape as ``v``; every row non-negative and summing to one.
    """
    arr = np.asarray(v, dtype=float)
    rows = np.atleast_2d(arr)
    k = rows.shape[1]
    u = -np.sort(-rows, axis=1)
    css = np.cumsum(u, axis=1) - 1.0
    idx = np.arange(1, k + 1)
    positive = u - css / idx > 0
    rho = k - np.argmax(positive[:, ::-1], axis=1)
    theta = css[np.arange(rows.shape[0]), rho - 1] / rho
    out = np.maximum(rows - theta[:, None], 0.0)
    return out[0] if arr.ndim == 1 else out


def clip_renormalise(
    raw: Mapping[str, float], *, clip: bool = True, renormalize: bool = True
) -> dict[str, float]:
    """
    Clip negative values to zero, then rescale to sum to one.

    Parameters
    ----------
    raw:
        An unconstrained pointwise estimate.
    clip, renormalize:
        The legacy flags of :class:`ZNESamplingConfig`.

    Returns
    -------
    dict[str, float]
        The adjusted estimate.  If every entry clips to zero there is nothing
        to renormalise and the all-zero vector is returned -- not a
        distribution, which is itself a pathology worth recording.
    """
    out = {b: max(0.0, v) for b, v in raw.items()} if clip else dict(raw)
    if renormalize:
        total = sum(out.values())
        if total > 0:
            out = {b: v / total for b, v in out.items()}
    return out


def simplex_renormalise(raw: Mapping[str, float]) -> dict[str, float]:
    """
    :func:`project_to_simplex` on a bitstring-keyed estimate.

    Parameters
    ----------
    raw:
        An unconstrained pointwise estimate.

    Returns
    -------
    dict[str, float]
        The closest distribution in L2.
    """
    keys = list(raw)
    projected = project_to_simplex(np.array([raw[k] for k in keys]))
    return {k: float(p) for k, p in zip(keys, projected, strict=True)}


def _probability_matrix(
    counts_per_scale: Mapping[float, Mapping[str, int]],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Sorted scales, sorted bitstring keys and the (n_scales, K) matrix."""
    scales = sorted(counts_per_scale)
    keys: set[str] = set()
    for scale in scales:
        if sum(counts_per_scale[scale].values()) == 0:
            raise ValueError(f"No shots collected at ZNE scale {scale}.")
        keys.update(counts_per_scale[scale])
    ordered = sorted(keys)
    probs = np.array(
        [[counts_to_probs(counts_per_scale[s]).get(k, 0.0) for k in ordered] for s in scales]
    )
    return np.asarray(scales, dtype=float), ordered, probs


def _pointwise_asymptote(cfg: ZNESamplingConfig, keys: Sequence[str]) -> float | None:
    """Per-bitstring asymptote: uniform ``2**-width`` when one is requested."""
    if cfg.method != "exponential" or cfg.asymptote is None or not keys:
        return None
    width = len(next(iter(keys)).replace(" ", ""))
    return 2.0**-width


def _pointwise_raw_matrix(x: np.ndarray, P: np.ndarray, cfg: ZNESamplingConfig, keys) -> np.ndarray:
    """
    Unconstrained per-bitstring extrapolation of stacked probability series.

    ``P`` has shape ``(..., n_scales, K)``; the result has shape ``(..., K)``.
    Bitstrings that are zero at every scale stay exactly zero.
    """
    lead = P.shape[:-2]
    n, k = P.shape[-2], P.shape[-1]
    series = np.moveaxis(P, -1, -2).reshape(-1, n)
    asymptote = _pointwise_asymptote(cfg, keys)
    if cfg.method == "exponential" and asymptote is None:
        raw = np.array(
            [
                0.0 if np.allclose(row, 0.0) else zne_fit_single_value("exponential", x, row)[0]
                for row in series
            ]
        )
    else:
        raw = _extrapolate_many(cfg.method, x, series, None, asymptote, cfg.exp_rate_max)
    raw = np.where(np.all(series == 0.0, axis=1), 0.0, raw)
    return raw.reshape(*lead, k)


def _renorm_rows(raw: np.ndarray, cfg: ZNESamplingConfig) -> np.ndarray:
    """Apply ``cfg.renorm`` to every row of ``raw``."""
    if cfg.renorm == "simplex":
        return project_to_simplex(raw)
    out = np.clip(raw, 0.0, None) if cfg.clip else raw.copy()
    if cfg.renormalize:
        totals = out.sum(axis=-1, keepdims=True)
        out = np.where(totals > 0, out / np.where(totals > 0, totals, 1.0), out)
    return out


def pointwise_raw(
    counts_per_scale: Mapping[float, Mapping[str, int]], cfg: ZNESamplingConfig
) -> dict[str, float]:
    """
    The unconstrained pointwise extrapolation, before any renormalisation.

    Parameters
    ----------
    counts_per_scale:
        Counts keyed by realised scale.
    cfg:
        Supplies ``method`` (and ``asymptote``, see :class:`ZNESamplingConfig`).

    Returns
    -------
    dict[str, float]
        One value per bitstring seen at any scale; may be negative and need
        not sum to one.
    """
    x, keys, probs = _probability_matrix(counts_per_scale)
    raw = _pointwise_raw_matrix(x, probs, cfg, keys)
    return {k: float(v) for k, v in zip(keys, raw, strict=True)}


def extrapolate_distribution(
    counts_per_scale: Mapping[float, Mapping[str, int]], zne_cfg: ZNESamplingConfig
) -> dict[str, float]:
    """
    Pointwise ZNE: extrapolate every bitstring's probability to zero noise.

    Parameters
    ----------
    counts_per_scale:
        Counts keyed by noise scale (see :func:`run_zne_scales`).
    zne_cfg:
        Supplies the model and ``renorm`` (``"clip"`` honours the legacy
        ``clip`` / ``renormalize`` flags; ``"simplex"`` projects).

    Returns
    -------
    dict[str, float]
        The mitigated distribution over every bitstring seen at any scale.

    Raises
    ------
    ValueError
        If any scale collected zero shots.
    """
    raw = pointwise_raw(counts_per_scale, zne_cfg)
    if zne_cfg.renorm == "simplex":
        return simplex_renormalise(raw)
    return clip_renormalise(raw, clip=zne_cfg.clip, renormalize=zne_cfg.renormalize)


# ---------------------------------------------------------------------------
# 6) Bootstrap
# ---------------------------------------------------------------------------


def zne_bootstrap(
    counts_per_scale: Mapping[float, Mapping[str, int]],
    cfg: ZNESamplingConfig,
    *,
    observable_fn: Callable[[Mapping[str, int]], float] | None = None,
    reference: Mapping[str, float] | None = None,
    solution_bitstrings: Sequence[str] | None = None,
    p_uniform: float | None = None,
    B: int | None = None,
    seed: int | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Bootstrap confidence intervals of a ZNE estimate.

    The counts at every scale are resampled multinomially (same shots, the
    observed frequencies), the *whole* estimator -- weights included -- is
    re-run on each resample, and percentile intervals are taken.

    Parameters
    ----------
    counts_per_scale:
        Counts keyed by realised scale.
    cfg:
        ``cfg.mode`` selects the estimator; ``method``, ``asymptote`` and
        ``renorm`` configure it; ``n_bootstrap`` and ``bootstrap_seed`` are
        the defaults of ``B`` and ``seed``.
    observable_fn:
        Scalar mode: the proportion to extrapolate (required).
    reference:
        Pointwise mode: a distribution to report ``D_TV`` against.
    solution_bitstrings:
        Pointwise mode: also report the interval on the pointwise
        ``P(solution)``.
    p_uniform:
        Scalar mode: the random-guessing baseline (see
        :func:`fit_scalar_series`).
    B, seed:
        Number of resamples and resampling seed.
    alpha:
        ``1 - confidence``.

    Returns
    -------
    dict
        Scalar: ``{"point", "lo", "hi", "se", "n_failed", "B", "seed",
        "alpha", "obs_se", "fit"}`` -- ``point`` is the fit on the observed
        data, ``obs_se`` the bootstrap standard error of the observable at
        each scale.  Pointwise: ``{"distribution", "intervals", "tv",
        "solution_probability", "renorm", "B", "seed", "alpha"}``, with
        ``intervals`` mapping each bitstring to ``[lo, hi]`` and ``tv`` /
        ``solution_probability`` as ``{"point", "lo", "hi"}`` (``None`` when
        no reference / solutions were given).
    """
    n_boot = cfg.n_bootstrap if B is None else int(B)
    boot_seed = cfg.bootstrap_seed if seed is None else int(seed)
    rng = np.random.default_rng(boot_seed)

    if cfg.mode == "scalar":
        if observable_fn is None:
            raise ValueError("Scalar-mode bootstrap needs an observable_fn.")
        x, _, shots = _observed_series(counts_per_scale, observable_fn)
        point = extrapolate_scalar(counts_per_scale, observable_fn, cfg, p_uniform=p_uniform)
        Y = np.empty((n_boot, x.size))
        for j, scale in enumerate(sorted(counts_per_scale)):
            keys, draws = resample_counts(counts_per_scale[scale], rng, n_boot)
            Y[:, j] = [
                observable_fn(dict(zip(keys, row.tolist(), strict=True))) for row in draws
            ]
        values = _extrapolate_many(
            cfg.method,
            x,
            Y,
            _sigma_matrix(Y, shots),
            resolve_asymptote(cfg, p_uniform),
            cfg.exp_rate_max,
        )
        lo, hi = percentile_interval(values, alpha)
        finite = values[np.isfinite(values)]
        return {
            "point": point["value"],
            "lo": lo,
            "hi": hi,
            "se": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
            "n_failed": int(values.size - finite.size),
            "B": n_boot,
            "seed": boot_seed,
            "alpha": alpha,
            "obs_se": Y.std(axis=0, ddof=1).tolist(),
            "fit": point,
        }

    x, keys, probs = _probability_matrix(counts_per_scale)
    point_dist = extrapolate_distribution(counts_per_scale, cfg)
    column = {k: i for i, k in enumerate(keys)}
    P = np.zeros((n_boot, x.size, len(keys)))
    for j, scale in enumerate(sorted(counts_per_scale)):
        counts = counts_per_scale[scale]
        draw_keys, draws = resample_counts(counts, rng, n_boot)
        cols = [column[k] for k in draw_keys]
        P[:, j, cols] = draws / float(sum(counts.values()))
    samples = _renorm_rows(_pointwise_raw_matrix(x, P, cfg, keys), cfg)

    intervals = {
        k: list(percentile_interval(samples[:, i], alpha)) for i, k in enumerate(keys)
    }

    tv_block: dict[str, float] | None = None
    if reference is not None:
        extra = sorted(set(reference) - set(keys))
        ref = np.array([reference.get(k, 0.0) for k in keys])
        ref_extra = np.array([reference[k] for k in extra])
        tv_samples = 0.5 * (np.abs(samples - ref).sum(axis=1) + ref_extra.sum())
        lo, hi = percentile_interval(tv_samples, alpha)
        tv_block = {"point": total_variation(point_dist, reference), "lo": lo, "hi": hi}

    sol_block: dict[str, float] | None = None
    if solution_bitstrings is not None:
        cols = [column[s] for s in solution_bitstrings if s in column]
        sol_samples = samples[:, cols].sum(axis=1) if cols else np.zeros(n_boot)
        lo, hi = percentile_interval(sol_samples, alpha)
        sol_block = {
            "point": distribution_solution_probability(point_dist, solution_bitstrings),
            "lo": lo,
            "hi": hi,
        }

    return {
        "distribution": point_dist,
        "intervals": intervals,
        "tv": tv_block,
        "solution_probability": sol_block,
        "renorm": cfg.renorm,
        "B": n_boot,
        "seed": boot_seed,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# 7) Scale planning and the pathology diagnostic
# ---------------------------------------------------------------------------


def plan_scales(
    survival: float,
    signal: float,
    shots: int,
    candidates: Sequence[float],
    k: float = 3.0,
    *,
    p_uniform: float | None = None,
    n_points: int | None = None,
) -> list[float]:
    """
    Pre-register the scale grid on which a folded signal stays resolvable.

    Under global depolarising noise the folded signal is
    ``S**lam * signal`` with ``signal = P_ideal - P_uniform``.  A candidate
    scale is admissible when that *predicted* signal is at least ``k`` shot-
    noise standard errors, ``sigma = sqrt(p (1 - p) / shots)`` at the
    predicted ``p = P_uniform + S**lam * signal``.  Because the signal falls
    monotonically with ``lam`` the admissible candidates form a prefix, and
    the grid is that prefix, starting at 1.

    ``survival`` is an *upper bound* (see
    :func:`dgssp.metrics.survival_estimate`), so the plan is optimistic; the
    resolvability flag of :func:`fit_scalar_series` checks it after the fact.

    Parameters
    ----------
    survival:
        Survival estimate ``S`` of the circuit at ``lam = 1``.
    signal:
        ``P_ideal - P_uniform``.
    shots:
        Planned shots per scale.
    candidates:
        Allowed scales (``1`` is always included).
    k:
        Required signal-to-noise ratio.
    p_uniform:
        ``P_uniform``, where ``sigma`` is evaluated; ``None`` uses the
        worst case ``p = 1/2``.
    n_points:
        Thin the grid to at most this many points, always keeping the first
        and last.  ``None`` keeps every admissible candidate.

    Returns
    -------
    list[float]
        Increasing scales.  ``[1.0]`` alone means not even ``lam = 1`` is
        resolvable: report the unmitigated value, do not extrapolate.  (The
        pathology diagnostic, which needs several scales even when there is no
        signal, is planned separately.)
    """
    if not 0.0 < survival <= 1.0:
        raise ValueError(f"survival must be in (0, 1], got {survival}.")
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}.")
    grid_all = sorted({1.0} | {float(c) for c in candidates if float(c) >= 1.0})

    admissible: list[float] = []
    for lam in grid_all:
        predicted = survival**lam * abs(signal)
        p = 0.5 if p_uniform is None else p_uniform + survival**lam * signal
        p = min(max(p, 1.0 / shots), 1.0 - 1.0 / shots)
        if predicted < k * math.sqrt(p * (1.0 - p) / shots):
            break
        admissible.append(lam)

    if not admissible:
        return [1.0]
    if n_points is not None and len(admissible) > n_points >= 2:
        picks = np.unique(np.round(np.linspace(0, len(admissible) - 1, n_points)).astype(int))
        admissible = [admissible[i] for i in picks]
    return admissible


def pathology_diagnostic(
    counts_per_scale: Mapping[float, Mapping[str, int]],
    cfg: ZNESamplingConfig,
    reference: Mapping[str, float],
) -> dict[str, Any]:
    """
    How far pointwise ZNE moves a distribution, and in which direction.

    Run on **every** instance, not only M = 0: for M = 0 (and for M/N = 1/2 at
    one iteration) the reference is analytically uniform, so any distance the
    mitigation *adds* is pure pathology; elsewhere the reference is the
    noiseless distribution.

    Parameters
    ----------
    counts_per_scale:
        Counts keyed by realised scale.
    cfg:
        Supplies the pointwise model.
    reference:
        The distribution the mitigated one should approach.

    Returns
    -------
    dict
        ``fraction_positive_slope`` (bitstrings whose least-squares linear
        slope in ``lam`` is positive -- a model-free descriptor, whatever the
        model), ``raw_total_mass`` (sum of the unconstrained estimate),
        ``clip_mass_removed`` (total negative mass clipped away),
        ``clip_mass_moved`` and ``simplex_mass_moved`` (L1 distance from the
        unconstrained estimate to each renormalised one), and
        ``tv_raw_min_scale`` / ``tv_pointwise_clip`` / ``tv_pointwise_simplex``
        (``D_TV`` to the reference of the lowest-scale -- unmitigated --
        distribution and of the two mitigated ones), with
        ``clip_worse_than_raw`` / ``simplex_worse_than_raw`` flags.
    """
    x, keys, probs = _probability_matrix(counts_per_scale)
    raw_vec = _pointwise_raw_matrix(x, probs, cfg, keys)
    raw = {k: float(v) for k, v in zip(keys, raw_vec, strict=True)}

    xc = x - x.mean()
    slopes = (xc @ (probs - probs.mean(axis=0))) / float(xc @ xc) if x.size > 1 else np.zeros(len(keys))

    clipped = clip_renormalise(raw)
    projected = simplex_renormalise(raw)
    lowest = dict(zip(keys, probs[0].tolist(), strict=True))

    tv_raw = total_variation(lowest, reference)
    tv_clip = total_variation(clipped, reference)
    tv_simplex = total_variation(projected, reference)
    ref_values = np.array(list(reference.values()), dtype=float)
    return {
        "method": cfg.method,
        "scales": x.tolist(),
        "n_bitstrings": len(keys),
        "fraction_positive_slope": float(np.mean(slopes > 0)),
        "raw_total_mass": float(raw_vec.sum()),
        "clip_mass_removed": float(np.clip(-raw_vec, 0.0, None).sum()),
        "clip_mass_moved": float(sum(abs(clipped[k] - raw[k]) for k in keys)),
        "simplex_mass_moved": float(sum(abs(projected[k] - raw[k]) for k in keys)),
        "tv_raw_min_scale": tv_raw,
        "tv_pointwise_clip": tv_clip,
        "tv_pointwise_simplex": tv_simplex,
        "clip_worse_than_raw": bool(tv_clip > tv_raw),
        "simplex_worse_than_raw": bool(tv_simplex > tv_raw),
        "reference_is_uniform": bool(
            ref_values.size > 0 and np.allclose(ref_values, ref_values.mean(), atol=1e-9)
        ),
        "raw_distribution": raw,
    }


# ---------------------------------------------------------------------------
# 8) Pipeline
# ---------------------------------------------------------------------------


def transpile_once_seeded(
    circuit: QuantumCircuit, backend: BackendLike, zne_cfg: ZNESamplingConfig
) -> tuple[QuantumCircuit, int]:
    """
    :func:`transpile_once`, also returning the seed that produced the circuit.

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
    tuple[QuantumCircuit, int]
        The ISA circuit and its transpiler seed.
    """
    if zne_cfg.seed_max <= zne_cfg.seed_min + 1:
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=zne_cfg.optimization_level,
            seed_transpiler=zne_cfg.seed_min,
            layout_method=zne_cfg.layout_method,
        )
        return pm.run(circuit), zne_cfg.seed_min

    best = find_best_seed(
        circuit,
        backend,
        seed_min=zne_cfg.seed_min,
        seed_max=zne_cfg.seed_max,
        optimization_level=zne_cfg.optimization_level,
        layout_method=zne_cfg.layout_method,
    )
    return best.circuit, best.best_seed


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
        Supplies the seed range, optimization level and layout method.  To
        reproduce a circuit ranked in WP3 pass ``seed_min=best_seed,
        seed_max=best_seed + 1``.

    Returns
    -------
    QuantumCircuit
        The transpiled circuit with the best layout found by the sweep (or a
        single-seed transpilation when the range has length one).
    """
    return transpile_once_seeded(circuit, backend, zne_cfg)[0]


def _virtual_gates(backend: BackendLike) -> tuple[str, ...]:
    """The backend's zero-duration gates, or :data:`DEFAULT_VIRTUAL_GATES`."""
    try:
        from ..backends import median_gate_errors

        names = median_gate_errors(backend).get("virtual_gates") or ()
    except Exception:
        return DEFAULT_VIRTUAL_GATES
    return tuple(names) or DEFAULT_VIRTUAL_GATES


def _checked_scales(scales: Sequence[float]) -> list[float]:
    """Non-empty list of valid scales."""
    out = [float(s) for s in scales]
    if not out:
        raise ValueError("ZNE scales must be non-empty.")
    for s in out:
        _validate_scale(s)
    return out


def run_zne_scales(
    tqc: QuantumCircuit, backend: BackendLike, zne_cfg: ZNESamplingConfig
) -> dict[float, dict[str, int]]:
    """
    Fold a transpiled circuit at every scale and sample all of them in one job.

    Parameters
    ----------
    tqc:
        An ISA-level circuit (see :func:`transpile_once`).
    backend:
        The backend to sample on.
    zne_cfg:
        Supplies the scales, shots, fold seed and simulator seed.

    Returns
    -------
    dict[float, dict[str, int]]
        Counts keyed by **realised** noise scale (equal to the nominal one for
        odd integers).

    Raises
    ------
    ValueError
        If ``scales`` is empty or contains a value below 1, or two nominal
        scales realise to the same scale.
    """
    batch = run_zne_batch(
        [ZNEItem(name=tqc.name, circuit=tqc, scales=zne_cfg.scales)],
        backend,
        zne_cfg,
        include_mitiq=False,
        mode="backend",
    )
    return dict(batch.items[0].counts_per_scale)


def zne_mitigated_distribution(
    circuit: QuantumCircuit,
    backend: BackendLike,
    zne_cfg: ZNESamplingConfig,
    *,
    transpiled: QuantumCircuit | None = None,
) -> dict[str, float]:
    """
    Full pointwise pipeline: transpile once, fold, batch-sample, extrapolate.

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
    tqc = transpiled if transpiled is not None else transpile_once(circuit, backend, zne_cfg)
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


# ---------------------------------------------------------------------------
# 9) One job for many instances
# ---------------------------------------------------------------------------


@dataclass
class ZNEItem:
    """
    One instance's contribution to a batched ZNE job.

    Attributes
    ----------
    name:
        Label used to map results back.
    circuit:
        The ISA circuit (transpiled once, for the batch's backend).
    scales:
        Nominal scales for this instance; ``None`` uses ``cfg.scales``.
    solution_bitstrings:
        Correct outcomes (kept with the item for the analysis step).
    """

    name: str
    circuit: QuantumCircuit
    scales: Sequence[float] | None = None
    solution_bitstrings: Sequence[str] = ()


@dataclass
class ZNEItemRun:
    """
    What one :class:`ZNEItem` produced inside a batched job.

    Attributes
    ----------
    name:
        The item's label.
    nominal_scales, realised_scales:
        Requested scales and the exact two-qubit ratios they realised (as
        floats; ``realised_scales_exact`` holds ``[numerator, denominator]``).
    realised_scales_1q:
        Physical single-qubit ratios, measured on the rebased circuits.
    counts_per_scale:
        Counts keyed by realised scale.
    mitiq_scales:
        Scales handed to Mitiq's ``fold_gates_at_random`` -- this item's
        realised scales, so both estimators fold to the same targets.
    mitiq_realised_scales:
        The two-qubit ratios Mitiq's folds actually realised (Mitiq measures
        its scale on a fidelity-weighted count of *all* gates, so these
        differ from ``mitiq_scales``; recorded for audit).
    mitiq_counts_per_scale:
        Counts of Mitiq's folded circuits keyed by ``mitiq_scales``.  The
        ``1.0`` entry reuses this item's own ``lam = 1`` counts: Mitiq's
        scale-1 circuit is the unfolded circuit.
    fold_seed:
        Seed of both folding procedures.
    pub_indices:
        ``{"dgssp": [...], "mitiq": [...]}`` positions in the submitted list.
    """

    name: str
    nominal_scales: list[float]
    realised_scales: list[float]
    realised_scales_exact: list[tuple[int, int]]
    realised_scales_1q: list[float | None]
    counts_per_scale: dict[float, dict[str, int]]
    mitiq_scales: list[float] = field(default_factory=list)
    mitiq_realised_scales: list[float] = field(default_factory=list)
    mitiq_counts_per_scale: dict[float, dict[str, int]] = field(default_factory=dict)
    fold_seed: int = 0
    pub_indices: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class ZNEBatchRun:
    """
    The single job behind a batch of ZNE items.

    Attributes
    ----------
    items:
        One :class:`ZNEItemRun` per input item, in input order.
    job_id:
        The runtime job id (``None`` on primitives that expose none).
    backend_name:
        Backend the job ran on.
    n_circuits:
        Number of pubs submitted.
    shots:
        Shots per pub.
    """

    items: list[ZNEItemRun]
    job_id: str | None
    backend_name: str
    n_circuits: int
    shots: int


def run_zne_batch(
    items: Sequence[ZNEItem],
    backend: BackendLike,
    cfg: ZNESamplingConfig,
    *,
    include_mitiq: bool = False,
    mode: str = "auto",
) -> ZNEBatchRun:
    """
    Fold many instances at all their scales and sample them in ONE job.

    For every item and every nominal scale the ISA circuit is folded with
    :func:`fold_isa` (seed ``cfg.fold_seed``) and rebased to the native
    basis.  With ``include_mitiq`` the same ISA circuit is also folded by
    Mitiq's ``fold_gates_at_random`` at this item's *realised* scales (same
    seed), except at scale 1, where Mitiq's circuit is the unfolded one and
    the item's own counts are reused.  Every circuit goes into a single
    :func:`dgssp.runtime.sample_counts` call, and the pubs are mapped back by
    position.  Mitiq never executes anything itself: its extrapolation runs
    offline (:func:`dgssp.mitigation.mitiq_baseline.mitiq_extrapolate`).

    Parameters
    ----------
    items:
        The instances.  Every circuit must already be transpiled for
        ``backend``.
    backend:
        The backend of the one job.
    cfg:
        Supplies default scales, ``shots_per_scale``, ``fold_seed`` and the
        simulator seed.
    include_mitiq:
        Also submit Mitiq's folds.
    mode:
        Execution mode for :func:`dgssp.runtime.execution_mode`.

    Returns
    -------
    ZNEBatchRun
        Per-item counts keyed by realised scale, plus the job id.

    Raises
    ------
    ValueError
        If there are no items, an item's scales are invalid, or two of its
        nominal scales realise to the same scale.
    """
    if not items:
        raise ValueError("run_zne_batch needs at least one item.")

    virtual = _virtual_gates(backend)
    if include_mitiq:
        from .mitiq_baseline import mitiq_fold_isa

    circuits: list[QuantumCircuit] = []
    plans: list[dict[str, Any]] = []
    for item in items:
        nominal = _checked_scales(item.scales if item.scales is not None else cfg.scales)
        base_two_q = two_qubit_count(item.circuit)
        folds = [
            fold_isa(item.circuit, lam, seed=cfg.fold_seed, virtual_gates=virtual)
            for lam in nominal
        ]
        realised = [f.realised_scale for f in folds]
        if len(set(realised)) != len(realised):
            raise ValueError(
                f"Item {item.name!r}: nominal scales {nominal} realise to "
                f"{[float(r) for r in realised]}, which collide."
            )

        plan: dict[str, Any] = {
            "item": item,
            "nominal": nominal,
            "folds": folds,
            "dgssp_idx": [],
            "mitiq_idx": [],
            "mitiq_scales": [],
            "mitiq_realised": [],
            "one_q_realised": [],
        }
        base_one_q = _physical_one_q(rebase_to_backend(item.circuit, backend), virtual)
        for fold in folds:
            rebased = rebase_to_backend(fold.circuit, backend)
            plan["dgssp_idx"].append(len(circuits))
            one_q = _physical_one_q(rebased, virtual)
            plan["one_q_realised"].append(one_q / base_one_q if base_one_q else None)
            circuits.append(rebased)

        if include_mitiq:
            for fold in folds:
                lam = float(fold.realised_scale)
                plan["mitiq_scales"].append(lam)
                if fold.realised_scale == 1:
                    plan["mitiq_idx"].append(None)
                    plan["mitiq_realised"].append(1.0)
                    continue
                mitiq_circuit = rebase_to_backend(
                    mitiq_fold_isa(item.circuit, lam, seed=cfg.fold_seed), backend
                )
                plan["mitiq_idx"].append(len(circuits))
                plan["mitiq_realised"].append(
                    two_qubit_count(mitiq_circuit) / base_two_q if base_two_q else float("nan")
                )
                circuits.append(mitiq_circuit)
        plans.append(plan)

    seed = cfg.seed_simulator if is_simulator(backend) else None
    with execution_mode(backend, mode=mode) as exec_mode:  # type: ignore[arg-type]
        counts_list = sample_counts(
            exec_mode,
            circuits,
            shots=cfg.shots_per_scale,
            seed_simulator=seed,
            meta={
                "stage": "zne_batch",
                "items": [p["item"].name for p in plans],
                "include_mitiq": include_mitiq,
            },
        )

    runs: list[ZNEItemRun] = []
    for plan in plans:
        folds = plan["folds"]
        counts_per_scale = {
            float(f.realised_scale): counts_list[i]
            for f, i in zip(folds, plan["dgssp_idx"], strict=True)
        }
        mitiq_counts: dict[float, dict[str, int]] = {}
        for lam, idx in zip(plan["mitiq_scales"], plan["mitiq_idx"], strict=True):
            mitiq_counts[lam] = counts_per_scale[lam] if idx is None else counts_list[idx]
        runs.append(
            ZNEItemRun(
                name=plan["item"].name,
                nominal_scales=plan["nominal"],
                realised_scales=[float(f.realised_scale) for f in folds],
                realised_scales_exact=[
                    (f.realised_scale.numerator, f.realised_scale.denominator) for f in folds
                ],
                realised_scales_1q=plan["one_q_realised"],
                counts_per_scale=counts_per_scale,
                mitiq_scales=plan["mitiq_scales"],
                mitiq_realised_scales=plan["mitiq_realised"],
                mitiq_counts_per_scale=mitiq_counts,
                fold_seed=cfg.fold_seed,
                pub_indices={
                    "dgssp": list(plan["dgssp_idx"]),
                    "mitiq": [i for i in plan["mitiq_idx"] if i is not None],
                },
            )
        )

    return ZNEBatchRun(
        items=runs,
        job_id=getattr(counts_list, "job_id", None),
        backend_name=backend_name(backend),
        n_circuits=len(circuits),
        shots=cfg.shots_per_scale,
    )


def _physical_one_q(circuit: QuantumCircuit, virtual: Sequence[str]) -> int:
    """Single-qubit gates that are neither virtual nor directives."""
    skip = _UNFOLDABLE | set(virtual)
    return sum(
        1
        for instruction in circuit.data
        if instruction.operation.num_qubits == 1 and instruction.operation.name not in skip
    )


def _scalar_block(
    run: ZNEItemRun,
    cfg: ZNESamplingConfig,
    observable_fn: Callable[[Mapping[str, int]], float],
    p_uniform: float,
) -> dict[str, Any]:
    """
    One method's scalar fit on an item's counts, with its bootstrap interval.

    Parameters
    ----------
    run:
        The item's counts.
    cfg:
        Estimator settings; ``cfg.method`` is the method fitted.
    observable_fn:
        The proportion to extrapolate.
    p_uniform:
        Resolvability null and ``"p_uniform"`` asymptote.

    Returns
    -------
    dict
        :func:`fit_scalar_series` output plus ``ci``, ``bootstrap_se``,
        ``n_failed``, ``B``, ``bootstrap_seed`` and the item's folding record.
    """
    boot = zne_bootstrap(
        run.counts_per_scale,
        _with_mode(cfg, "scalar"),
        observable_fn=observable_fn,
        p_uniform=p_uniform,
    )
    block = dict(boot["fit"])
    block.update(
        {
            "ci": [boot["lo"], boot["hi"]],
            "bootstrap_se": boot["se"],
            "n_failed": boot["n_failed"],
            "B": boot["B"],
            "bootstrap_seed": boot["seed"],
            "nominal_scales": list(run.nominal_scales),
            "realised_scales_exact": [list(f) for f in run.realised_scales_exact],
            "realised_scales_1q": list(run.realised_scales_1q),
            "fold_seed": run.fold_seed,
        }
    )
    return block


def analyse_zne_item(
    run: ZNEItemRun,
    cfg: ZNESamplingConfig,
    *,
    solution_bitstrings: Sequence[str],
    p_uniform: float,
    reference: Mapping[str, float] | None = None,
    methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Every estimator and diagnostic on one item's counts -- offline.

    Nothing here executes a circuit, so a hardware job fetched later with
    :func:`dgssp.runtime.fetch_result` is analysed by exactly this code.

    Parameters
    ----------
    run:
        The item's counts (see :func:`run_zne_batch`).
    cfg:
        Estimator settings.
    solution_bitstrings:
        Correct outcomes.
    p_uniform:
        ``M / 2**n``: resolvability null and ``"p_uniform"`` asymptote.
    reference:
        Noiseless reference distribution for the pointwise diagnostics.
    methods:
        Extrapolation methods to fit, all from the *same* counts -- nothing is
        re-sampled, so model comparison costs no shots.  ``None`` fits only
        ``cfg.method``.  ``cfg.method`` is always fitted, whether or not it
        appears here.

    Returns
    -------
    dict
        ``{"scalar", "scalars", "pointwise", "pathology", "mitiq"}``.
        ``scalar`` is the :func:`fit_scalar_series` dict for ``cfg.method``
        plus ``ci`` (``[lo, hi]``), ``bootstrap_se`` and ``n_failed``;
        ``scalars`` holds the same block for every requested method, keyed by
        method name (it always contains ``cfg.method``, and equals
        ``{cfg.method: scalar}`` when ``methods`` is ``None``); ``pointwise``
        holds both renormalised distributions, their ``P(solution)``, and
        bootstrap intervals for ``cfg.renorm``; ``pathology`` is
        :func:`pathology_diagnostic` (``None`` without a reference); ``mitiq``
        is ``None`` unless Mitiq folds were run.

    Raises
    ------
    ValueError
        If ``methods`` names an unknown extrapolation method.
    """
    sols = list(solution_bitstrings)

    def p_sol(counts: Mapping[str, int]) -> float:
        return solution_probability(counts, sols)

    wanted = [cfg.method] if methods is None else [cfg.method, *methods]
    for name in wanted:
        if name not in ZNE_METHODS:
            raise ValueError(f"Unknown method '{name}'. Use one of {', '.join(ZNE_METHODS)}.")
    # dict.fromkeys de-duplicates while keeping cfg.method first.
    scalars = {
        name: _scalar_block(run, replace(cfg, method=name), p_sol, p_uniform)  # type: ignore[arg-type]
        for name in dict.fromkeys(wanted)
    }
    scalar = scalars[cfg.method]

    pointwise: dict[str, Any] = {}
    for renorm in ZNE_RENORMS:
        dist = extrapolate_distribution(run.counts_per_scale, _with_renorm(cfg, renorm))
        pointwise[renorm] = {
            "distribution": dist,
            "solution_probability": distribution_solution_probability(dist, sols),
        }
    point_boot = zne_bootstrap(
        run.counts_per_scale,
        _with_mode(cfg, "pointwise"),
        reference=reference,
        solution_bitstrings=sols,
    )
    pointwise["bootstrap"] = {
        "renorm": cfg.renorm,
        "intervals": point_boot["intervals"],
        "tv": point_boot["tv"],
        "solution_probability": point_boot["solution_probability"],
        "B": point_boot["B"],
        "seed": point_boot["seed"],
    }

    pathology = (
        pathology_diagnostic(run.counts_per_scale, _with_mode(cfg, "pointwise"), reference)
        if reference is not None
        else None
    )

    mitiq: dict[str, Any] | None = None
    if run.mitiq_counts_per_scale:
        from .mitiq_baseline import mitiq_bootstrap

        mitiq = mitiq_bootstrap(
            run.mitiq_counts_per_scale,
            p_sol,
            method=cfg.method,
            asymptote=resolve_asymptote(cfg, p_uniform),
            B=cfg.n_bootstrap,
            seed=cfg.bootstrap_seed,
        )
        mitiq["realised_scales"] = list(run.mitiq_realised_scales)
        mitiq["fold_seed"] = run.fold_seed

    return {
        "scalar": scalar,
        "scalars": scalars,
        "pointwise": pointwise,
        "pathology": pathology,
        "mitiq": mitiq,
    }


def _with_mode(cfg: ZNESamplingConfig, mode: str) -> ZNESamplingConfig:
    """Copy of ``cfg`` with another ``mode``."""
    return replace(cfg, mode=mode)  # type: ignore[arg-type]


def _with_renorm(cfg: ZNESamplingConfig, renorm: str) -> ZNESamplingConfig:
    """Copy of ``cfg`` with another ``renorm``."""
    return replace(cfg, renorm=renorm)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 10) Synthetic depolarising model (no circuits)
# ---------------------------------------------------------------------------


def depolarised_probability(
    survival: float, lam: float, p_ideal: float, p_uniform: float
) -> float:
    """
    ``P(lam) = S**lam * P_ideal + (1 - S**lam) * P_uniform``.

    The global-depolarising model of a folded proportion: exponential in the
    scale, which is why polynomial extrapolations are biased under it.

    Parameters
    ----------
    survival:
        ``S`` at ``lam = 1``.
    lam:
        Noise scale.
    p_ideal, p_uniform:
        Noiseless value and depolarising fixed point.

    Returns
    -------
    float
        The expected proportion.
    """
    decay = survival**lam
    return decay * p_ideal + (1.0 - decay) * p_uniform


def synthetic_depolarising_counts(
    survival: float,
    scales: Sequence[float],
    p_ideal: float,
    p_uniform: float,
    shots: int,
    rng: np.random.Generator,
) -> dict[float, dict[str, int]]:
    """
    Binomial draws from :func:`depolarised_probability` at each scale.

    Parameters
    ----------
    survival, p_ideal, p_uniform:
        Model parameters.
    scales:
        (Realised) noise scales.
    shots:
        Shots per scale.
    rng:
        Random generator.

    Returns
    -------
    dict[float, dict[str, int]]
        ``{"1": successes, "0": failures}`` per scale; the observable is
        ``solution_probability(counts, ["1"])``.
    """
    out: dict[float, dict[str, int]] = {}
    for lam in scales:
        k = int(rng.binomial(shots, depolarised_probability(survival, lam, p_ideal, p_uniform)))
        out[float(lam)] = {"1": k, "0": shots - k}
    return out


def expected_extrapolation(
    cfg: ZNESamplingConfig,
    scales: Sequence[float],
    survival: float,
    p_ideal: float,
    p_uniform: float,
    shots: int,
) -> float:
    """
    What the scalar estimator converges to on noiseless-in-shots data.

    The fit applied to the *exact* model curve, with the exact binomial
    weights.  ``expected_extrapolation(...) - p_ideal`` is the estimator's
    analytic bias under global depolarising noise; the Monte-Carlo mean
    matches it to ``O(1 / shots)``.

    Parameters
    ----------
    cfg:
        Supplies ``method`` and ``asymptote``.
    scales:
        Realised scales.
    survival, p_ideal, p_uniform:
        Model parameters.
    shots:
        Shots per scale (sets the weights).

    Returns
    -------
    float
        The extrapolated value.
    """
    x = np.asarray(scales, dtype=float)
    y = np.array([depolarised_probability(survival, s, p_ideal, p_uniform) for s in x])
    sigma = np.sqrt(y * (1.0 - y) / shots)
    return float(
        _fit_series(
            cfg.method, x, y, sigma, resolve_asymptote(cfg, p_uniform), cfg.exp_rate_max
        )["value"]
    )


__all__ = [
    "ZNESamplingConfig",
    "ZNE_METHODS",
    "ZNE_MODES",
    "ZNE_RENORMS",
    "RESOLVABILITY_K",
    "DEFAULT_VIRTUAL_GATES",
    "EXP_RATE_MAX",
    "FoldResult",
    "fold_plan",
    "realised_scale",
    "fold_isa",
    "fold_transpiled",
    "rebase_to_backend",
    "richardson_weights",
    "zne_fit_single_value",
    "resolve_asymptote",
    "is_resolved",
    "informative_scales_mask",
    "resolved_scales_mask",
    "fit_scalar_series",
    "extrapolate_scalar",
    "project_to_simplex",
    "clip_renormalise",
    "simplex_renormalise",
    "pointwise_raw",
    "extrapolate_distribution",
    "zne_bootstrap",
    "plan_scales",
    "pathology_diagnostic",
    "transpile_once",
    "transpile_once_seeded",
    "run_zne_scales",
    "zne_mitigated_distribution",
    "dgssp_zne_mitigated_distribution",
    "ZNEItem",
    "ZNEItemRun",
    "ZNEBatchRun",
    "run_zne_batch",
    "analyse_zne_item",
    "depolarised_probability",
    "synthetic_depolarising_counts",
    "expected_extrapolation",
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
        Scale ``>= 1``.

    Returns
    -------
    QuantumCircuit
        The folded circuit.
    """
    warnings.warn(
        "fold_local_circuit is deprecated; use fold_transpiled on an "
        "ISA-level circuit instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fold_transpiled(circuit, scale_factor)


__all__.append("fold_local_circuit")
