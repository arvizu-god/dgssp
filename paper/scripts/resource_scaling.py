"""
resource_scaling.py

WP2 / contribution C1: the measured resource scaling of the Draper--Grover SSP
circuit on heavy-hex hardware.

What it does
------------
1. Builds a deterministic grid of instances covering ``n`` items in
   ``{2, ..., 8}`` and sum-register widths ``b`` in ``{3, ..., 8}``.
2. Checks every closed form in ``paper/notes/resource_model.md`` against the
   circuit the solver actually builds -- ``assert``, not ``warn``.  A failure
   means the note is wrong.
3. Transpiles each circuit at optimization level 3 on every requested heavy-hex
   fake device, sweeping SABRE seeds with
   :func:`dgssp.transpilation.find_best_seed`.
4. Turns the transpiled counts into a survival estimate from the device's
   median calibrated gate errors.
5. Fits effective power laws in ``b`` (at fixed ``n``) and in ``n`` (at fixed
   ``b``).
6. Writes ``paper/data/resources.json``, ``paper/data/resources_fits.json`` and
   ``paper/figures/fig_resources.pdf``.

Circuits are built with **one Grover iteration**.  The resource model is
per-iteration by construction (see the note), the iteration count enters only
as the multiplier ``r_opt(n, M)``, and building at ``r = 1`` keeps 144
optimization-level-3 transpilations x 32 seeds x 3 devices inside a coffee
break.  ``r_opt`` is recorded per grid point so the totals can be scaled.

How the items are chosen
------------------------
For a target ``(n, b)`` the register width is fixed by the item *sum*: with all
items positive, ``range_len = S + 1`` and ``b = ceil(log2(S+1)) + 1``, so

    b is hit exactly when   2**(b-2) <= S <= 2**(b-1) - 1.

Within that window two deterministic families bracket the cost of the
controlled additions (note SS3.1):

``dense``
    All items odd, so ``v2(a_k) = 0`` for every item and *no* rotation in the
    adder is a multiple of ``2*pi``.  This is the worst case: a
    valuation-aware adder would save nothing.  Items are the smallest distinct
    odd numbers ``1, 3, 5, ...`` when their sum fits the window, all ones when
    it does not, with the last item chosen odd to land ``S`` in the window.

``sparse``
    Every item a power of two, with valuations as large as the window allows.
    This is the best case: a valuation-aware adder would drop
    ``2 * sum_k v2(a_k)`` controlled phases per iteration.  ``S`` is taken at
    the bottom of the window and split into ``n`` powers of two by repeatedly
    halving the largest part.

Each family is built all-positive and with the **last item negated**.  Negating
an item leaves ``range_len`` -- and therefore ``b`` -- unchanged (note SS0), so
the signed variant is a clean isolation of the offset addition's cost.

The grid is not a full rectangle: ``n`` positive items force ``S >= n``, hence
``b >= ceil(log2(n+1)) + 1``.  Cells below that line (e.g. ``n = 8, b = 4``) do
not exist and are skipped, not faked.

Run
---
::

    python paper/scripts/resource_scaling.py
    python paper/scripts/resource_scaling.py --backends fake_marrakesh
    python paper/scripts/resource_scaling.py --logical-only     # no transpilation
    python paper/scripts/resource_scaling.py --figure-only      # replot existing data
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _common import DATA_DIR, Record, record_provenance, save_record
from dgssp import (
    DGConfig,
    DGSSPSolver,
    DPSSPSolver,
    SubsetSumInstance,
    build_encoding,
    encoding_without_guard_bit,
    flatten_containers,
    median_gate_errors,
    optimal_iterations,
    physical_gate_counts,
    resource_model,
)
from dgssp.backends import list_fake_backends
from dgssp.runtime import backend_name
from dgssp.transpilation import (
    LOGICAL_BASIS,
    find_best_seed,
    logical_counts,
    mcx_noaux_1q_cost,
    mcx_noaux_cx_cost,
    mcz_noaux_cx_cost,
    transpiled_metrics,
    two_qubit_count,
)

# --- Frozen sweep parameters -------------------------------------------------

#: Index-register widths swept.
N_VALUES: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)

#: Sum-register widths swept, guard bit included.
B_VALUES: tuple[int, ...] = (3, 4, 5, 6, 7, 8)

#: Heavy-hex fake devices used by default: the twins of the Heron devices on
#: the IBM Open plan.  Anything the installed ``qiskit-ibm-runtime`` does not
#: ship is reported and skipped rather than silently substituted.
DEFAULT_BACKENDS: tuple[str, ...] = ("fake_marrakesh", "fake_fez", "fake_torino")

#: SABRE seeds swept per (circuit, backend).  Fixed, so the script is
#: deterministic: the same seed range on the same package versions gives the
#: same layout and therefore the same counts.
SEED_MIN, SEED_MAX = 0, 32

#: Preset optimization level for the layout search.
OPTIMIZATION_LEVEL = 3

#: Grover iterations the measured circuits are built with.  See the module
#: docstring: the model is per-iteration and ``r_opt`` is recorded separately.
ITERATIONS = 1

#: Gate names allowed to survive :func:`flatten_containers`.  Anything else
#: means the solver grew a construct the resource model does not know about.
EXPECTED_GATE_LEVEL_OPS: frozenset[str] = frozenset(
    {"cp", "p", "h", "x", "z", "cz", "ccx", "mcx", "barrier", "measure"}
)


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def sum_window(b: int) -> tuple[int, int]:
    """
    Inclusive range of item sums that produce a ``b``-qubit sum register.

    Parameters
    ----------
    b:
        Target sum-register width, guard bit included.

    Returns
    -------
    tuple[int, int]
        ``(lowest, highest)`` admissible total for all-positive items.
    """
    return 2 ** (b - 2), 2 ** (b - 1) - 1


def dense_items(n: int, b: int) -> list[int] | None:
    """
    ``n`` odd items summing into :func:`sum_window`, or ``None`` if impossible.

    All-odd means every item has 2-adic valuation zero, so every one of the
    ``n * b`` controlled phases in an addition block is a genuine rotation.
    This is the worst case for the adder.

    Parameters
    ----------
    n, b:
        Grid coordinates.

    Returns
    -------
    list[int] | None
        The items, or ``None`` when no such set exists.
    """
    low, high = sum_window(b)
    base = [2 * i - 1 for i in range(1, n)]
    if sum(base) > high - 1:
        base = [1] * (n - 1)

    remainder_low = max(1, low - sum(base))
    remainder_high = high - sum(base)
    if remainder_high < 1:
        return None

    last = remainder_low if remainder_low % 2 == 1 else remainder_low + 1
    if last > remainder_high:
        return None
    return [*base, last]


def sparse_items(n: int, b: int) -> list[int] | None:
    """
    ``n`` powers of two summing into :func:`sum_window`, or ``None``.

    The total is taken at the bottom of the window (a single power of two) and
    split into ``n`` parts by repeatedly halving the largest, which keeps the
    valuations as high as the window allows.  This is the best case for a
    valuation-aware adder.

    Parameters
    ----------
    n, b:
        Grid coordinates.

    Returns
    -------
    list[int] | None
        The items in descending order, or ``None`` when no such set exists.
    """
    low, high = sum_window(b)
    total = max(low, n)
    if total > high:
        return None

    parts = [1 << exponent for exponent in range(total.bit_length()) if (total >> exponent) & 1]
    while len(parts) < n:
        parts.sort()
        largest = parts.pop()
        if largest == 1:
            return None
        parts += [largest // 2, largest // 2]
    return sorted(parts, reverse=True)


#: The two item families, in the order they appear in the figure legend.
FAMILIES = {"dense": dense_items, "sparse": sparse_items}


@dataclass(frozen=True)
class GridPoint:
    """
    One instance of the sweep: a family, a sign convention and a target.

    Attributes
    ----------
    n, b:
        Index- and sum-register widths the point was constructed for.  Checked
        against :func:`dgssp.encoding.build_encoding`, not assumed.
    family:
        ``"dense"`` or ``"sparse"``; see the module docstring.
    sign:
        ``"positive"`` or ``"negated"`` (the last item's sign flipped).
    items, target:
        The instance.
    """

    n: int
    b: int
    family: str
    sign: str
    items: tuple[int, ...]
    target: int

    @property
    def name(self) -> str:
        """A stable identifier, e.g. ``"n5_b7_dense_negated"``."""
        return f"n{self.n}_b{self.b}_{self.family}_{self.sign}"

    def instance(self) -> SubsetSumInstance:
        """The :class:`dgssp.SubsetSumInstance` this point describes."""
        return SubsetSumInstance(items=list(self.items), target=self.target, name=self.name)


def build_grid() -> list[GridPoint]:
    """
    Every feasible ``(n, b, family, sign)`` combination, in a fixed order.

    Returns
    -------
    list[GridPoint]
        The grid.  Cells whose ``(n, b)`` pair is arithmetically impossible are
        absent (see the module docstring).

    Raises
    ------
    AssertionError
        If a constructed item set does not in fact produce the ``(n, b)`` it
        was built for -- which would mean the window arithmetic here and
        :func:`dgssp.encoding.build_encoding` disagree.
    """
    grid: list[GridPoint] = []
    for n in N_VALUES:
        for b in B_VALUES:
            for family, make in FAMILIES.items():
                base = make(n, b)
                if base is None:
                    continue
                for sign in ("positive", "negated"):
                    items = list(base) if sign == "positive" else [*base[:-1], -base[-1]]
                    # The target is a subset sum by construction, so the oracle
                    # always marks at least one state.
                    target = sum(items[0::2])
                    encoding = build_encoding(items, target)
                    assert encoding.n_ind == n and encoding.n_sum == b, (
                        f"{family}/{sign} items {items} were built for (n={n}, b={b}) "
                        f"but encode to (n={encoding.n_ind}, b={encoding.n_sum})."
                    )
                    grid.append(
                        GridPoint(
                            n=n,
                            b=b,
                            family=family,
                            sign=sign,
                            items=tuple(items),
                            target=target,
                        )
                    )
    return grid


# ---------------------------------------------------------------------------
# Step 2: logical counts, checked against the closed forms
# ---------------------------------------------------------------------------


def measure_logical(point: GridPoint) -> dict[str, Any]:
    """
    Build one grid point's circuit and check every closed form against it.

    Parameters
    ----------
    point:
        The grid point.

    Returns
    -------
    dict
        The resource model, the gate-level ``count_ops``, the pinned
        ancilla-free reference counts, the two overhead deltas and the optimal
        iteration count.

    Raises
    ------
    AssertionError
        If any formula in ``paper/notes/resource_model.md`` disagrees with the
        circuit.  Fix the note, not the assert.
    """
    instance = point.instance()
    encoding = build_encoding(list(point.items), point.target)
    model = resource_model(encoding, ITERATIONS)

    circuit = DGSSPSolver(DGConfig(iterations=ITERATIONS)).build_circuit(instance)
    flat = flatten_containers(circuit)
    observed = {name: int(count) for name, count in flat.count_ops().items()}

    unexpected = set(observed) - EXPECTED_GATE_LEVEL_OPS
    assert not unexpected, (
        f"{point.name}: unexpected gate-level operations {sorted(unexpected)}; "
        "the resource model in paper/notes/resource_model.md does not cover them."
    )

    predicted = dict(model.gate_level_counts())
    for name, count in model.multi_controlled_op_counts().items():
        predicted[name] = predicted.get(name, 0) + count
    predicted = {name: count for name, count in predicted.items() if count}

    for name, count in predicted.items():
        assert observed.get(name, 0) == count, (
            f"{point.name} (n={point.n}, b={point.b}): predicted {count} x '{name}', "
            f"circuit has {observed.get(name, 0)}."
        )

    # The pinned ancilla-free reference translation (note SS5.1).
    reference = logical_counts(circuit)
    predicted_two_q = model.two_qubit_reference(mcz_noaux_cx_cost)
    predicted_one_q = model.one_qubit_reference(mcx_noaux_1q_cost)
    assert reference["two_q_count"] == predicted_two_q, (
        f"{point.name}: predicted {predicted_two_q} two-qubit gates in the "
        f"ancilla-free reference, measured {reference['two_q_count']}."
    )
    assert reference["one_q_count"] == predicted_one_q, (
        f"{point.name}: predicted {predicted_one_q} single-qubit gates in the "
        f"ancilla-free reference, measured {reference['one_q_count']}."
    )

    # What Qiskit's *default* synthesis does with the same circuit: same basis,
    # same optimization level, but HighLevelSynthesis is free to pick the
    # cheaper plugin and to borrow idle register qubits as ancillas (note SS5.1).
    from qiskit import transpile

    hls_two_q = two_qubit_count(
        transpile(flat, basis_gates=list(LOGICAL_BASIS), optimization_level=0)
    )

    # Signed-encoding overhead: the offset addition, single-qubit only.
    signed_overhead = {
        "phase_gates": model.phase,
        "phase_gates_skippable": model.phase_skippable,
        "two_q_gates": 0,
        "extra_sum_qubits": 0,
        "offset": encoding.offset,
    }

    # Guard-bit overhead: the same circuit one bit narrower (note SS7b).
    narrow = resource_model(encoding_without_guard_bit(encoding), ITERATIONS)
    guard_bit_overhead = {
        "controlled_phase": model.controlled_phase - narrow.controlled_phase,
        "two_q_reference": (
            predicted_two_q - narrow.two_qubit_reference(mcz_noaux_cx_cost)
        ),
        "qubits": 1,
    }

    num_solutions = len(DPSSPSolver().solve(instance).all_solutions)

    return {
        "resource_model": model.to_dict(),
        "gate_level_counts": observed,
        "logical": {
            "two_q_count": reference["two_q_count"],
            "one_q_count": reference["one_q_count"],
            "depth": reference["depth"],
            "two_q_depth": reference["two_q_depth"],
            "two_q_count_hls": hls_two_q,
        },
        "signed_overhead": signed_overhead,
        "guard_bit_overhead": guard_bit_overhead,
        "num_solutions": num_solutions,
        "r_opt": optimal_iterations(point.n, num_solutions),
        "encoded_target": encoding.encoded_target,
        "modulus": encoding.modulus,
    }


# ---------------------------------------------------------------------------
# Step 3: transpilation on real heavy-hex topologies
# ---------------------------------------------------------------------------


def resolve_backends(names: tuple[str, ...]) -> list[Any]:
    """
    Look up fake devices by name, keeping only the heavy-hex ones.

    Parameters
    ----------
    names:
        Backend names as ``qiskit-ibm-runtime`` spells them, e.g.
        ``"fake_marrakesh"``.

    Returns
    -------
    list
        The matching backends, in the order requested.  Names the installed
        release does not ship are reported on stderr and skipped.

    Raises
    ------
    RuntimeError
        If none of the requested names resolve.
    """
    available = {backend_name(backend): backend for backend in list_fake_backends(heavy_hex_only=True)}
    resolved = []
    for name in names:
        backend = available.get(name)
        if backend is None:
            print(f"  ! {name} is not a heavy-hex device in this qiskit-ibm-runtime; skipping", file=sys.stderr)
            continue
        resolved.append(backend)

    if not resolved:
        raise RuntimeError(
            f"None of {list(names)} resolved. Available heavy-hex fake devices: "
            f"{sorted(available)}"
        )
    return resolved


def measure_transpiled(
    point: GridPoint, backend: Any, logical_two_q: int, *, seed_max: int = SEED_MAX
) -> dict[str, Any]:
    """
    Transpile one grid point on one device and record its ISA-level size.

    Parameters
    ----------
    point:
        The grid point.
    backend:
        A calibrated heavy-hex backend.
    logical_two_q:
        The ancilla-free reference two-qubit count, used to infer the routing
        overhead when the transpiler exposes no SWAP statistics.
    seed_max:
        Exclusive upper bound of the SABRE seed sweep.

    Returns
    -------
    dict
        :func:`dgssp.transpilation.transpiled_metrics` plus the winning seed,
        the accumulated calibration error, the SWAP accounting and the survival
        estimate.
    """
    circuit = DGSSPSolver(DGConfig(iterations=ITERATIONS)).build_circuit(point.instance())

    started = time.time()
    best = find_best_seed(
        circuit,
        backend,
        seed_min=SEED_MIN,
        seed_max=seed_max,
        optimization_level=OPTIMIZATION_LEVEL,
    )
    elapsed = time.time() - started

    metrics = dict(transpiled_metrics(best.circuit))
    counts = {name: int(count) for name, count in best.circuit.count_ops().items()}
    one_q = sum(
        1
        for instruction in best.circuit.data
        if instruction.operation.num_qubits == 1
        and instruction.operation.name not in ("barrier", "delay", "measure")
    )

    # Qiskit's preset pass managers translate SWAPs into the device basis, so
    # an explicit `swap` count is normally absent.  Fall back to the excess
    # two-qubit gates over the ancilla-free logical reference, which is the
    # routing overhead plus whatever the optimiser saved -- a signed quantity,
    # reported as such rather than clamped.
    swap_count = counts.get("swap")
    metrics.update(
        {
            "one_q_count": one_q,
            "op_counts": counts,
            "best_seed": best.best_seed,
            "optimization_level": best.optimization_level,
            "accumulated_two_q_error": best.total_two_qubit_error,
            "swap_count": swap_count,
            "two_q_excess_over_logical": int(metrics["two_q_count"]) - logical_two_q,
            "transpile_seconds": round(elapsed, 3),
        }
    )

    # Survival is charged to *physical* gates only: the median excludes virtual
    # `rz`, so the count it multiplies must exclude them too.  `one_q_count`
    # above stays the full single-qubit inventory, which is a resource figure.
    errors = median_gate_errors(backend)
    physical = physical_gate_counts(best.circuit, errors)
    metrics["median_errors"] = errors
    metrics["one_q_physical"] = physical["one_q_physical"]
    metrics["two_q_physical"] = physical["two_q_physical"]
    metrics["virtual_gate_count"] = physical["virtual"]
    metrics.update(
        survival_estimate(physical["two_q_physical"], physical["one_q_physical"], errors)
    )
    return metrics


def survival_estimate(two_q: int, one_q: int, errors: dict[str, Any]) -> dict[str, Any]:
    """
    Probability that no gate errs, from the device's median gate errors.

    ``S2 = (1 - eps2)**two_q`` is the primary figure -- two-qubit errors are an
    order of magnitude larger than single-qubit ones on every heavy-hex device
    in the fake provider -- and ``S1 = (1 - eps1)**one_q`` is the secondary
    factor.  This is a *coherence-free upper bound*: it ignores idling,
    crosstalk, readout and the fact that not every gate error is fatal to the
    marked amplitude.  It is used to say where the algorithm cannot work, not
    to predict a success probability.

    Parameters
    ----------
    two_q, one_q:
        *Physical* gate counts of the transpiled circuit, from
        :func:`dgssp.backends.physical_gate_counts` -- virtual ``rz`` frame
        changes excluded, exactly as they are excluded from the medians.
        Passing the raw single-qubit inventory instead charges the ``sx``
        error to every ``rz`` and understates survival.
    errors:
        Output of :func:`dgssp.backends.median_gate_errors`.

    Returns
    -------
    dict
        ``{"survival_two_q", "survival_one_q", "survival"}``; entries are
        ``None`` when the device carries no median for that gate class.
    """
    eps_two = errors.get("two_qubit")
    eps_one = errors.get("one_qubit")

    survival_two = (1.0 - eps_two) ** two_q if eps_two is not None else None
    survival_one = (1.0 - eps_one) ** one_q if eps_one is not None else None
    combined = (
        survival_two * survival_one
        if survival_two is not None and survival_one is not None
        else survival_two
    )
    return {
        "survival_two_q": survival_two,
        "survival_one_q": survival_one,
        "survival": combined,
    }


# ---------------------------------------------------------------------------
# Step 5: effective power-law fits
# ---------------------------------------------------------------------------


def fit_power_law(xs: list[float], ys: list[float]) -> dict[str, Any] | None:
    """
    Least-squares fit of ``y = A * x**alpha`` in log-log space.

    Parameters
    ----------
    xs, ys:
        Strictly positive samples of equal length.

    Returns
    -------
    dict | None
        ``{"exponent", "exponent_stderr", "prefactor", "r_squared", "n_points"}``,
        or ``None`` when fewer than three distinct ``x`` values are available
        (two points fit any power law exactly and leave no residual).
    """
    from scipy import stats

    points = [(x, y) for x, y in zip(xs, ys, strict=True) if x > 0 and y > 0]
    if len({x for x, _ in points}) < 3:
        return None

    log_x = [math.log(x) for x, _ in points]
    log_y = [math.log(y) for _, y in points]
    fit = stats.linregress(log_x, log_y)
    return {
        "exponent": float(fit.slope),
        "exponent_stderr": float(fit.stderr),
        "prefactor": float(math.exp(fit.intercept)),
        "r_squared": float(fit.rvalue**2),
        "n_points": len(points),
    }


def fit_scaling(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Fit ``two_q_count ~ b**alpha`` at fixed ``n`` and ``~ n**beta`` at fixed ``b``.

    Fits are done per backend and per item family, on the all-positive
    variants only: the negated variants share the same two-qubit count by
    construction (note SS7a) and would only duplicate every point.

    None of the closed forms is a power law, so these are *effective* exponents
    over the measured range, not asymptotic ones.

    Parameters
    ----------
    records:
        Flattened rows from :func:`rows_from_records`.

    Returns
    -------
    dict
        ``{"vs_b": [...], "vs_n": [...]}``, each entry carrying its slice
        coordinates and the fit.
    """
    rows = [row for row in records if row["sign"] == "positive"]

    vs_b: list[dict[str, Any]] = []
    vs_n: list[dict[str, Any]] = []

    keys = sorted({(row["backend"], row["family"]) for row in rows})
    for backend, family in keys:
        subset = [r for r in rows if r["backend"] == backend and r["family"] == family]

        for n in sorted({r["n"] for r in subset}):
            slice_rows = sorted((r for r in subset if r["n"] == n), key=lambda r: r["b"])
            fit = fit_power_law(
                [float(r["b"]) for r in slice_rows],
                [float(r["two_q_count"]) for r in slice_rows],
            )
            if fit is not None:
                vs_b.append({"backend": backend, "family": family, "n": n, **fit})

        for b in sorted({r["b"] for r in subset}):
            slice_rows = sorted((r for r in subset if r["b"] == b), key=lambda r: r["n"])
            fit = fit_power_law(
                [float(r["n"]) for r in slice_rows],
                [float(r["two_q_count"]) for r in slice_rows],
            )
            if fit is not None:
                vs_n.append({"backend": backend, "family": family, "b": b, **fit})

    return {"vs_b": vs_b, "vs_n": vs_n}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def build_record(
    point: GridPoint,
    logical: dict[str, Any],
    backend: Any | None,
    transpiled: dict[str, Any] | None,
) -> Record:
    """
    One archive row: a grid point, on one backend.

    Parameters
    ----------
    point:
        The grid point.
    logical:
        Output of :func:`measure_logical`.
    backend:
        The device the circuit was transpiled on, or ``None`` for the
        logical-only row.
    transpiled:
        Output of :func:`measure_transpiled`, or ``None``.

    Returns
    -------
    Record
        With ``counts`` empty: nothing was executed, this is a size measurement.
    """
    metrics: dict[str, Any] = {
        "grid": {"n": point.n, "b": point.b, "family": point.family, "sign": point.sign},
        **logical,
    }
    if transpiled is not None:
        metrics["median_errors"] = transpiled.get("median_errors")
        metrics["survival"] = transpiled.get("survival")
        metrics["survival_two_q"] = transpiled.get("survival_two_q")
        metrics["survival_one_q"] = transpiled.get("survival_one_q")

    return Record(
        instance={"items": list(point.items), "target": point.target, "name": point.name},
        n_items=point.n,
        n_sum=point.b,
        iterations=ITERATIONS,
        executor="resource_model",
        mitigation_mode="none",
        backend=backend_name(backend) if backend is not None else None,
        shots=0,
        counts={},
        metrics=metrics,
        transpiled=transpiled or {},
        provenance=record_provenance(backend),
    )


def rows_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Flatten archive records into the plain table the figure and fits consume.

    Parameters
    ----------
    records:
        Records as written to ``paper/data/resources.json`` (or loaded back).

    Returns
    -------
    list[dict]
        One row per transpiled (grid point, backend) pair.  Logical-only rows
        (``backend is None``) are dropped: they are carried on every transpiled
        row already.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        transpiled = record.get("transpiled") or {}
        if not transpiled or record.get("backend") is None:
            continue
        grid = record["metrics"]["grid"]
        logical = record["metrics"]["logical"]
        rows.append(
            {
                "n": grid["n"],
                "b": grid["b"],
                "family": grid["family"],
                "sign": grid["sign"],
                "backend": record["backend"],
                "two_q_count": int(transpiled["two_q_count"]),
                "two_q_depth": int(transpiled["two_q_depth"]),
                "depth": int(transpiled["depth"]),
                "one_q_count": int(transpiled["one_q_count"]),
                "survival": transpiled.get("survival"),
                "logical_two_q": int(logical["two_q_count"]),
                "logical_two_q_hls": int(logical["two_q_count_hls"]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Step 6: the figure
# ---------------------------------------------------------------------------


def make_figure(rows: list[dict[str, Any]], *, family: str = "dense") -> list[Path]:
    """
    Build ``paper/figures/fig_resources.pdf`` from the flattened table.

    Panel (a): two-qubit count and two-qubit depth against ``n`` at the widest
    ``b`` every ``n`` in the grid supports.  Panel (b): the same against ``b``
    at the largest such ``n``.  Panel (c): survival estimate over the whole
    ``(n, b)`` grid for the first backend in the table.

    Parameters
    ----------
    rows:
        Output of :func:`rows_from_records`.
    family:
        Which item family to draw; the other is in the data either way.

    Returns
    -------
    list[Path]
        The files written.

    Raises
    ------
    ValueError
        If the table holds no rows for ``family``.
    """
    import numpy as np
    from matplotlib.lines import Line2D

    from plotting import OKABE_ITO, SEQUENTIAL_CMAP, backend_style, figure, save_figure

    table = [r for r in rows if r["family"] == family and r["sign"] == "positive"]
    if not table:
        raise ValueError(f"No rows for family={family!r}.")

    backends = sorted({r["backend"] for r in table})
    primary = backends[0]

    # Panel (a) fixes b at the largest value shared by the most n; panel (b)
    # fixes n at the largest n that reaches the top of the b range.
    b_counts = {b: len({r["n"] for r in table if r["b"] == b}) for b in {r["b"] for r in table}}
    b_fixed = max(b_counts, key=lambda b: (b_counts[b], b))
    n_candidates = {r["n"] for r in table if r["b"] == max(r2["b"] for r2 in table)}
    n_fixed = max(n_candidates)

    fig, axes = figure(width=6.9, aspect=0.36, ncols=3)
    ax_a, ax_b, ax_c = axes

    for ax, x_key, fixed_key, fixed_value, x_label in (
        (ax_a, "n", "b", b_fixed, "items $n$"),
        (ax_b, "b", "n", n_fixed, "sum-register width $b$"),
    ):
        twin = ax.twinx()
        twin.set_yscale("log")
        twin.grid(False)

        for backend in backends:
            series = sorted(
                (r for r in table if r["backend"] == backend and r[fixed_key] == fixed_value),
                key=lambda r: r[x_key],
            )
            if not series:
                continue
            style = backend_style(backend)
            xs = [r[x_key] for r in series]
            ax.plot(
                xs,
                [r["two_q_count"] for r in series],
                label=style.label,
                **style.line_kwargs(),
            )
            twin.plot(
                xs,
                [r["two_q_depth"] for r in series],
                **style.line_kwargs(linestyle=":", marker="None", alpha=0.55, zorder=1),
            )

        reference = sorted(
            {(r[x_key], r["logical_two_q"]) for r in table if r[fixed_key] == fixed_value}
        )
        if reference:
            style = backend_style("logical")
            ax.plot(
                [x for x, _ in reference],
                [y for _, y in reference],
                label=style.label,
                **style.line_kwargs(),
            )

        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("2Q gate count")
        # Only the right-hand panel carries the twin-axis label: with three
        # panels across a 6.9 in text width, a label on panel (a)'s twin lands
        # on top of panel (b)'s own y-label.
        if x_key == "b":
            twin.set_ylabel("2Q depth")
        ax.set_title(f"({'a' if x_key == 'n' else 'b'}) {fixed_key} = {fixed_value}", loc="left")

    depth_proxy = Line2D(
        [], [], color=OKABE_ITO["grey"], linestyle=":", alpha=0.8, label="2Q depth (right axis)"
    )
    # One legend for the whole figure, below the panels: the curves in (a) and
    # (b) rise steeply from the bottom-left to the top-right, so every in-axes
    # position either covers data or covers the logical reference.
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        [*handles, depth_proxy],
        [*labels, depth_proxy.get_label()],
        loc="outside lower center",
        ncol=len(handles) + 1,
        fontsize=7,
    )

    # Panel (c): survival heatmap on the primary backend.
    heat_rows = [r for r in table if r["backend"] == primary and r["survival"] is not None]
    ns = sorted({r["n"] for r in heat_rows})
    bs = sorted({r["b"] for r in heat_rows})
    grid = np.full((len(bs), len(ns)), np.nan)
    for row in heat_rows:
        grid[bs.index(row["b"]), ns.index(row["n"])] = math.log10(max(row["survival"], 1e-300))

    image = ax_c.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=SEQUENTIAL_CMAP,
        extent=(ns[0] - 0.5, ns[-1] + 0.5, bs[0] - 0.5, bs[-1] + 0.5),
    )
    ax_c.set_xticks(ns)
    ax_c.set_yticks(bs)
    ax_c.set_xlabel("items $n$")
    ax_c.set_ylabel("sum-register width $b$")
    ax_c.set_title(f"(c) {backend_style(primary).label}", loc="left")
    ax_c.grid(False)
    colorbar = fig.colorbar(image, ax=ax_c, pad=0.02)
    colorbar.set_label(r"$\log_{10} S$")

    return save_figure(fig, "fig_resources")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Command-line interface.

    Parameters
    ----------
    argv:
        Argument list, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_BACKENDS),
        help="Heavy-hex fake devices to transpile on.",
    )
    parser.add_argument(
        "--seed-max",
        type=int,
        default=SEED_MAX,
        help="Exclusive upper bound of the SABRE seed sweep.",
    )
    parser.add_argument(
        "--logical-only",
        action="store_true",
        help="Check the closed forms and stop; do not transpile.",
    )
    parser.add_argument(
        "--figure-only",
        action="store_true",
        help="Rebuild the figure from the newest paper/data/resources*.json.",
    )
    parser.add_argument(
        "--family",
        default="dense",
        choices=sorted(FAMILIES),
        help="Item family drawn in the figure.",
    )
    return parser.parse_args(argv)


def newest_resources_file() -> Path:
    """
    The most recent ``paper/data/resources*.json``.

    Returns
    -------
    Path
        The file.

    Raises
    ------
    FileNotFoundError
        If no such file exists yet.
    """
    candidates = sorted(DATA_DIR.glob("resources.json")) + sorted(DATA_DIR.glob("resources_[0-9]*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No resources*.json in {DATA_DIR}; run the script without --figure-only first."
        )
    return candidates[-1]


def main(argv: list[str] | None = None) -> int:
    """
    Run the sweep and write the data, the fits and the figure.

    Parameters
    ----------
    argv:
        Argument list, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success.
    """
    args = parse_args(argv)

    if args.figure_only:
        path = newest_resources_file()
        records = json.loads(path.read_text(encoding="utf-8"))
        written = make_figure(rows_from_records(records), family=args.family)
        print(f"read  {path}")
        for item in written:
            print(f"wrote {item}")
        return 0

    seed_max = int(args.seed_max)

    grid = build_grid()
    cells = sorted({(p.n, p.b) for p in grid})
    print(f"grid: {len(grid)} circuits over {len(cells)} (n, b) cells, {ITERATIONS} Grover iteration")
    print(f"      n in {list(N_VALUES)}, b in {list(B_VALUES)}, families {sorted(FAMILIES)}\n")

    print("step 2: checking the closed forms of paper/notes/resource_model.md")
    logical: dict[str, dict[str, Any]] = {}
    for point in grid:
        logical[point.name] = measure_logical(point)
    print(f"  ok: {len(grid)}/{len(grid)} grid points match the resource model\n")

    records: list[Record] = []
    if args.logical_only:
        for point in grid:
            records.append(build_record(point, logical[point.name], None, None))
    else:
        backends = resolve_backends(tuple(args.backends))
        print(f"step 3: transpiling at optimization level {OPTIMIZATION_LEVEL}, "
              f"seeds {SEED_MIN}..{seed_max - 1}, on {[backend_name(b) for b in backends]}")
        started = time.time()
        for index, point in enumerate(grid, start=1):
            reference_two_q = logical[point.name]["logical"]["two_q_count"]
            for backend in backends:
                transpiled = measure_transpiled(
                    point, backend, reference_two_q, seed_max=seed_max
                )
                records.append(build_record(point, logical[point.name], backend, transpiled))
            if index % 12 == 0 or index == len(grid):
                print(f"  {index:>3}/{len(grid)} circuits  ({time.time() - started:.0f}s)")
        print()

    payload = [record.to_dict() for record in records]
    data_path = save_record(DATA_DIR / "resources.json", payload)
    print(f"wrote {data_path}  ({len(records)} records)")

    rows = rows_from_records(payload)
    if rows:
        fits = fit_scaling(rows)
        fits_path = save_record(
            DATA_DIR / "resources_fits.json",
            {
                "source": data_path.name,
                "iterations": ITERATIONS,
                "optimization_level": OPTIMIZATION_LEVEL,
                "seed_range": [SEED_MIN, seed_max],
                "mcx_noaux_cx_cost": {str(k): mcx_noaux_cx_cost(k) for k in range(1, 12)},
                **fits,
            },
        )
        print(f"wrote {fits_path}")
        report_fits(fits)

        written = make_figure(rows, family=args.family)
        for item in written:
            print(f"wrote {item}")
    else:
        print("no transpiled rows: skipping fits and figure (--logical-only)")

    report_overheads(payload)
    return 0


def report_fits(fits: dict[str, Any]) -> None:
    """
    Print the fitted effective exponents.

    Parameters
    ----------
    fits:
        Output of :func:`fit_scaling`.

    Returns
    -------
    None
    """
    print("\neffective power-law exponents for the transpiled 2Q count")
    for label, key, coordinate in (
        ("two_q ~ b^alpha at fixed n", "vs_b", "n"),
        ("two_q ~ n^beta  at fixed b", "vs_n", "b"),
    ):
        entries = fits[key]
        if not entries:
            continue
        print(f"  {label}")
        for entry in entries:
            print(
                f"    {entry['backend']:<16} {entry['family']:<7} "
                f"{coordinate}={entry[coordinate]}  "
                f"exp = {entry['exponent']:.3f} +/- {entry['exponent_stderr']:.3f}  "
                f"R^2 = {entry['r_squared']:.4f}  ({entry['n_points']} pts)"
            )


def report_overheads(payload: list[dict[str, Any]]) -> None:
    """
    Print the signed-encoding and guard-bit deltas.

    Parameters
    ----------
    payload:
        The records written to ``paper/data/resources.json``.

    Returns
    -------
    None
    """
    negated = [r for r in payload if r["metrics"]["grid"]["sign"] == "negated"]
    if not negated:
        return
    phases = [r["metrics"]["signed_overhead"]["phase_gates"] for r in negated]
    two_q = {r["metrics"]["signed_overhead"]["two_q_gates"] for r in negated}
    guard_cp = [r["metrics"]["guard_bit_overhead"]["controlled_phase"] for r in negated]
    guard_2q = [r["metrics"]["guard_bit_overhead"]["two_q_reference"] for r in negated]
    print(
        f"\nsigned-encoding overhead (offset addition): "
        f"{min(phases)}-{max(phases)} single-qubit phase gates, "
        f"two-qubit delta {sorted(two_q)}"
    )
    print(
        f"guard-bit overhead: +1 qubit, "
        f"{min(guard_cp)}-{max(guard_cp)} controlled phases, "
        f"{min(guard_2q)}-{max(guard_2q)} two-qubit gates"
    )


if __name__ == "__main__":
    raise SystemExit(main())
