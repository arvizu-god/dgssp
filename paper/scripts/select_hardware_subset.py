"""
select_hardware_subset.py

WP3: which instances of the frozen family get hardware time, and why.

The QPU budget is one pre-validated job per month on the IBM Open plan, so the
subset is small and its choice has to be defensible in the paper rather than
"the ones that looked reasonable".  This script makes the choice mechanically
and writes down every number the choice was made from.

What it does
------------
1. Reads ``paper/instances/family.json`` and rebuilds each instance **at its own
   ``r_opt``** -- not at one iteration.  WP2's published survival figures are
   ``r = 1`` figures and are not reusable here; survival falls off as the
   ``r``-th power of the per-iteration factor, which is the whole point.
2. Transpiles each on the fake twin of the primary device at optimization level
   3, sweeping SABRE seeds with :func:`dgssp.transpilation.find_best_seed`.
   This is the paper's transpile-once step: the ISA circuit produced here is the
   one ZNE would later fold, so its metrics are the ones that matter.
3. Estimates survival from the device's median calibrated gate errors, reusing
   :func:`dgssp.backends.median_gate_errors` and
   :func:`dgssp.metrics.survival_estimate` -- neither is reimplemented.
4. Measures ``P_ideal``, the noiseless success probability at ``r_opt``, on a
   seeded ``AerSimulator`` through :func:`dgssp.execution.execute_ideal` (and so
   through :func:`dgssp.runtime.sample_counts`, the only sampler entry point).
5. Selects the shallowest instance in each ``n in {2,3,4}`` x
   ``M-class in {0,1,2}`` cell, by transpiled two-qubit depth.
6. Adds the *edge* instance, defined below.

The edge criterion
------------------
The edge instance is the largest one whose estimated success probability still
beats random guessing by a factor of two:

    P_est = S * P_ideal + (1 - S) * P_uniform  >=  2 * P_uniform,
    P_uniform = M / 2**n

The mixture is the honest reading of the survival estimate: with probability
``S`` no gate errs and the circuit delivers its ideal distribution; otherwise
the outcome is treated as a uniform draw over the index register, which still
lands on a solution with probability ``P_uniform``.  Rearranged, the criterion is

    S >= P_uniform / (P_ideal - P_uniform),

so the bar *rises* as ``P_uniform`` falls.  Comparing ``S`` against
``P_uniform`` directly -- the obvious-looking test -- inverts this: the
threshold falls off exponentially in ``n`` while ``S`` falls off only as fast as
the gate count grows, so bigger instances would pass more easily and the test
would simply return the largest instance in the family. That test is wrong and
is not used.

Instances with ``M = 0`` are excluded from edge candidacy: ``P_uniform = 0``
makes ``P_est >= 2 * P_uniform`` read ``0 >= 0``, true for every infeasible
instance and meaningless.  They remain in the hardware subset -- the infeasible
pathology is contribution C3 -- they are just not eligible to *define* the edge.

Instances with ``r_opt = 1`` are excluded too.  The edge is meant to mark where
*search* stops working, and a one-iteration circuit barely searches: it passes
the 2x bar on highly degenerate targets (M = 11 of 64 at n = 6) because a single
oracle-plus-diffuser is cheap, not because amplitude amplification survives at
that width.  Requiring ``r_opt >= 2`` keeps the edge a statement about search.

"Largest" is ordered by ``(n_items, n_sum, transpiled two-qubit count)``,
descending, so the edge is the widest instance first and the most expensive one
at that width second.

The primary device
------------------
Defaults to ``fake_marrakesh``, the twin used throughout WP2.  This is a
placeholder until ``paper/hardware/account_facts.md`` is filled: the real
primary device is whichever Heron the account actually exposes with the lowest
median two-qubit error and a tolerable queue.  Re-run with ``--backend`` once
that is known; the ranking will move.

Usage
-----
::

    python paper/scripts/select_hardware_subset.py
    python paper/scripts/select_hardware_subset.py --backend fake_fez --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import INSTANCES_DIR, record_provenance, save_record  # noqa: E402
from dgssp.backends import median_gate_errors, physical_gate_counts  # noqa: E402
from dgssp.execution import execute_ideal  # noqa: E402
from dgssp.instance import SubsetSumInstance  # noqa: E402
from dgssp.metrics import solution_probability, survival_estimate  # noqa: E402
from dgssp.runtime import backend_name  # noqa: E402
from dgssp.solvers.draper_grover import DGConfig, DGSSPSolver  # noqa: E402
from dgssp.transpilation import find_best_seed, transpiled_metrics  # noqa: E402
from resource_scaling import resolve_backends  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen selection parameters
# ---------------------------------------------------------------------------

SCHEMA = "dgssp/paper/hardware-subset/1"

#: Fake twin of the paper's primary device.  See the module docstring.
DEFAULT_BACKEND = "fake_marrakesh"

#: SABRE seed sweep, matching WP2 so the two tables are comparable.
SEED_MIN, SEED_MAX = 0, 32

#: Preset optimization level for the transpile-once step.
OPTIMIZATION_LEVEL = 3

#: Shots for the noiseless ``P_ideal`` measurement, and the seeds that make it
#: reproducible.  20 000 shots resolve a probability to about +/- 0.004, an
#: order of magnitude finer than the gaps the edge criterion has to resolve.
IDEAL_SHOTS = 20_000
IDEAL_SEED_SIMULATOR = 20261020
IDEAL_SEED_TRANSPILER = 0

#: Cells the hardware subset must cover: one instance per (n, M-class) pair.
SUBSET_N_VALUES: tuple[int, ...] = (2, 3, 4)
SUBSET_M_CLASSES: tuple[str, ...] = ("0", "1", "2")

#: Factor by which the edge instance must beat random guessing.
EDGE_FACTOR = 2.0

#: Fewest Grover iterations an instance needs to be edge-eligible.  See the
#: module docstring: below this the edge would measure one iteration's cost.
EDGE_MIN_R_OPT = 2


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------


def evaluate(
    record: dict[str, Any],
    backend: Any,
    errors: dict[str, Any],
    *,
    shots: int = IDEAL_SHOTS,
    seed_max: int = SEED_MAX,
) -> dict[str, Any]:
    """
    Transpile, estimate survival and measure ``P_ideal`` for one instance.

    Parameters
    ----------
    record:
        One entry from ``family.json``.
    backend:
        The calibrated fake twin to transpile against.
    errors:
        Output of :func:`dgssp.backends.median_gate_errors` for that backend,
        computed once by the caller rather than per instance.
    shots:
        Shots for the noiseless run.
    seed_max:
        Exclusive upper bound of the SABRE seed sweep.

    Returns
    -------
    dict
        The instance's identifying fields plus ``two_q_count``,
        ``two_q_depth``, ``depth``, ``physical_qubits``, ``best_seed``,
        ``survival``, ``p_ideal``, ``p_uniform`` and ``p_est``.
    """
    instance = SubsetSumInstance(
        items=list(record["items"]), target=record["target"], name=record["name"]
    )
    r_opt = int(record["r_opt"])
    circuit = DGSSPSolver(DGConfig(iterations=r_opt)).build_circuit(instance)

    started = time.time()
    best = find_best_seed(
        circuit,
        backend,
        seed_min=SEED_MIN,
        seed_max=seed_max,
        optimization_level=OPTIMIZATION_LEVEL,
    )
    transpile_seconds = time.time() - started

    metrics: dict[str, Any] = dict(transpiled_metrics(best.circuit))
    one_q = sum(
        1
        for instruction in best.circuit.data
        if instruction.operation.num_qubits == 1
        and instruction.operation.name not in ("barrier", "delay", "measure")
    )
    # Survival is charged to physical gates only; `one_q` above is the full
    # inventory, about a third of which is virtual `rz` and error-free.
    physical = physical_gate_counts(best.circuit, errors)
    survival = survival_estimate(
        physical["two_q_physical"], physical["one_q_physical"], errors
    )

    # A zero-solution instance has no marked state, so its noiseless success
    # probability is exactly zero.  Sampling it would consume simulator time to
    # confirm arithmetic, so it is asserted rather than measured, and the source
    # is recorded so the table cannot be misread as a measurement.
    if record["M"] == 0:
        p_ideal, p_ideal_source = 0.0, "exact (M=0: no marked state)"
    else:
        result = execute_ideal(
            circuit,
            shots=shots,
            seed_simulator=IDEAL_SEED_SIMULATOR,
            seed_transpiler=IDEAL_SEED_TRANSPILER,
        )
        p_ideal = solution_probability(result.counts, record["solution_bitstrings"])
        p_ideal_source = f"AerSimulator, {shots} shots, seed {IDEAL_SEED_SIMULATOR}"

    p_uniform = record["M"] / 2 ** record["n_items"]
    s = survival["survival"]
    p_est = (s * p_ideal + (1.0 - s) * p_uniform) if s is not None else None

    return {
        "name": record["name"],
        "family": record["family"],
        "n_items": record["n_items"],
        "n_negative": record["n_negative"],
        "n_sum": record["n_sum"],
        "items": list(record["items"]),
        "target": record["target"],
        "m_class": record["m_class"],
        "M": record["M"],
        "solution_bitstrings": list(record["solution_bitstrings"]),
        "r_opt": r_opt,
        "n_qubits": record["n_items"] + record["n_sum"],
        "two_q_count": int(metrics["two_q_count"]),
        "two_q_depth": int(metrics["two_q_depth"]),
        "depth": int(metrics["depth"]),
        "one_q_count": one_q,
        "one_q_physical": physical["one_q_physical"],
        "two_q_physical": physical["two_q_physical"],
        "virtual_gate_count": physical["virtual"],
        "physical_qubits": metrics["physical_qubits"],
        "best_seed": best.best_seed,
        "accumulated_two_q_error": best.total_two_qubit_error,
        "transpile_seconds": round(transpile_seconds, 3),
        "survival_two_q": survival["survival_two_q"],
        "survival_one_q": survival["survival_one_q"],
        "survival": s,
        "p_ideal": p_ideal,
        "p_ideal_source": p_ideal_source,
        "p_uniform": p_uniform,
        "p_est": p_est,
        # Undefined, not True, when M = 0: with P_uniform = 0 the criterion
        # reads 0 >= 0 for every infeasible instance and says nothing.
        "beats_random_by_2x": (
            None
            if (p_est is None or record["M"] == 0)
            else bool(p_est >= EDGE_FACTOR * p_uniform)
        ),
    }


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def rank_key(row: dict[str, Any]) -> tuple[int, int, str]:
    """
    Deterministic ranking key: shallowest first.

    Two-qubit depth is the primary figure -- it is what decoherence sees --
    with the two-qubit count as a tie-break and the instance name as a final
    one, so the ordering never depends on dict or file order.
    """
    return (row["two_q_depth"], row["two_q_count"], row["name"])


def size_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    """
    "Largest" ordering for the edge instance, ascending.

    Width first (``n_items``), then register width, then transpiled cost; the
    name breaks the remaining ties.  The maximum under this key is the edge.
    """
    return (row["n_items"], row["n_sum"], row["two_q_count"], row["name"])


def select_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Shallowest instance per ``(n, M-class)`` cell of the hardware grid.

    Both families compete in each cell: WP2 measured the signed encoding to
    cost zero extra two-qubit gates, so a family-S instance can legitimately
    win a cell on depth, and forcing a family would be picking the answer.
    ``n = 2`` cells can only be family P, which has no signed instances there.

    Parameters
    ----------
    rows:
        Evaluated candidates.

    Returns
    -------
    list[dict]
        One winner per populated cell, ordered by ``(n_items, m_class)``.
    """
    winners: list[dict[str, Any]] = []
    for n_items in SUBSET_N_VALUES:
        for m_class in SUBSET_M_CLASSES:
            cell = [
                r for r in rows if r["n_items"] == n_items and r["m_class"] == m_class
            ]
            if not cell:
                continue
            winner = min(cell, key=rank_key).copy()
            winner["selected_for"] = f"cell n={n_items} M={m_class}"
            winner["cell_size"] = len(cell)
            winner["cell_rank"] = 1
            winners.append(winner)
    return winners


def select_shallowest_signed(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Shallowest family-S instance inside the hardware grid, if one is wanted.

    Off by default, because the cell grid the paper specifies is
    ``n`` x ``M-class`` with no family dimension.  It exists because that grid
    has a measurable consequence: family P wins every cell on this device (a
    signed instance is 1-5% deeper at equal register width, and the guard bit
    occasionally costs it a whole bit), so the subset contains no signed
    instance at all -- and the signed encoding is half of contribution C1.
    Passing ``--include-signed`` adds one so the hardware run exercises it.

    Parameters
    ----------
    rows:
        Evaluated candidates.

    Returns
    -------
    dict | None
        The shallowest family-S candidate within the subset cells, or ``None``
        if there is none.
    """
    signed = [
        r
        for r in rows
        if r["family"] == "S"
        and r["n_items"] in SUBSET_N_VALUES
        and r["m_class"] in SUBSET_M_CLASSES
    ]
    if not signed:
        return None
    row = min(signed, key=rank_key).copy()
    row["selected_for"] = "shallowest signed instance (--include-signed)"
    return row


def select_edge(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Largest genuine search that still beats random guessing by :data:`EDGE_FACTOR`.

    Eligible instances have ``M > 0`` (the criterion is vacuous at M = 0) and
    ``r_opt >= EDGE_MIN_R_OPT`` (a one-iteration circuit is not a search).

    Parameters
    ----------
    rows:
        Evaluated candidates.

    Returns
    -------
    dict | None
        The edge instance, or ``None`` when nothing in the family passes -- a
        result worth reporting rather than working around.
    """
    eligible = [
        r
        for r in rows
        if r["M"] > 0
        and r["r_opt"] >= EDGE_MIN_R_OPT
        and r["p_est"] is not None
        and r["beats_random_by_2x"]
    ]
    if not eligible:
        return None
    edge = max(eligible, key=size_key).copy()
    edge["selected_for"] = (
        f"edge (P_est >= {EDGE_FACTOR:g} * P_uniform, r_opt >= {EDGE_MIN_R_OPT})"
    )
    return edge


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Command-line interface.

    Parameters
    ----------
    argv:
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description="Select the WP3 hardware subset.")
    parser.add_argument(
        "--family",
        type=Path,
        default=INSTANCES_DIR / "family.json",
        help="frozen family file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=INSTANCES_DIR / "hardware_subset.json",
        help="destination file",
    )
    parser.add_argument(
        "--backend", default=DEFAULT_BACKEND, help="fake twin of the primary device"
    )
    parser.add_argument("--shots", type=int, default=IDEAL_SHOTS, help="shots for P_ideal")
    parser.add_argument(
        "--seed-max",
        type=int,
        default=SEED_MAX,
        help="exclusive upper bound of the SABRE seed sweep",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=None,
        help="skip instances with more items than this (a fast dry run)",
    )
    parser.add_argument(
        "--include-signed",
        action="store_true",
        help=(
            "also select the shallowest family-S instance in the subset cells; "
            "without it the subset can end up with no signed instance at all"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing destination"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Evaluate the family, select the subset and write the file.

    Parameters
    ----------
    argv:
        Argument list.

    Returns
    -------
    int
        0 on success, 1 on a missing input or a refusal to overwrite.
    """
    args = parse_args(argv)

    if not args.family.exists():
        print(
            f"{args.family} not found; run `python paper/scripts/build_family.py` first.",
            file=sys.stderr,
        )
        return 1
    if args.out.exists() and not args.overwrite:
        print(f"Refusing to write: {args.out} exists. Use --overwrite.", file=sys.stderr)
        return 1

    family = json.loads(args.family.read_text(encoding="utf-8"))
    candidates = [
        r
        for r in family["instances"]
        if args.n_max is None or r["n_items"] <= args.n_max
    ]

    backend = resolve_backends((args.backend,))[0]
    errors = median_gate_errors(backend)
    print(f"backend       {backend_name(backend)}")
    print(f"median errors 2Q={errors.get('two_qubit')}  1Q={errors.get('one_qubit')}")
    print(f"candidates    {len(candidates)} of {len(family['instances'])}")

    rows: list[dict[str, Any]] = []
    started = time.time()
    for i, record in enumerate(candidates, start=1):
        rows.append(evaluate(record, backend, errors, shots=args.shots, seed_max=args.seed_max))
        if i % 25 == 0 or i == len(candidates):
            print(f"  {i}/{len(candidates)}  ({time.time() - started:.0f}s)")

    rows.sort(key=rank_key)

    selected = select_cells(rows)

    edge = select_edge(rows)
    signed = select_shallowest_signed(rows) if args.include_signed else None
    for extra, label in ((edge, "edge"), (signed, "signed")):
        if extra is None:
            continue
        existing = next((r for r in selected if r["name"] == extra["name"]), None)
        if existing is None:
            selected.append(extra)
        else:
            existing["selected_for"] += f" + {label}"

    report(rows, selected, edge)

    payload = {
        "schema": SCHEMA,
        "family_file": args.family.name,
        "family_payload_sha256": family.get("payload_sha256"),
        "family_master_seed": family.get("master_seed"),
        "backend": backend_name(backend),
        "median_gate_errors": errors,
        "policy": {
            "iterations": "r_opt per instance (NOT r=1; WP2 survival figures are r=1)",
            "optimization_level": OPTIMIZATION_LEVEL,
            "sabre_seeds": [SEED_MIN, args.seed_max],
            "rank_key": "(two_q_depth, two_q_count, name) ascending",
            "size_key": "(n_items, n_sum, two_q_count, name) descending",
            "subset_cells": {
                "n_items": list(SUBSET_N_VALUES),
                "m_classes": list(SUBSET_M_CLASSES),
            },
            "ideal_shots": args.shots,
            "ideal_seed_simulator": IDEAL_SEED_SIMULATOR,
            "ideal_seed_transpiler": IDEAL_SEED_TRANSPILER,
            "edge_criterion": (
                "P_est = S*P_ideal + (1-S)*P_uniform >= "
                f"{EDGE_FACTOR:g}*P_uniform, P_uniform = M/2**n, M > 0"
            ),
            "edge_excludes_m0": "P_uniform = 0 makes the criterion vacuous for M=0",
            "edge_min_r_opt": EDGE_MIN_R_OPT,
            "survival_gate_counts": (
                "physical gates only (dgssp.backends.physical_gate_counts); "
                "virtual rz excluded, matching the medians"
            ),
        },
        "selected": selected,
        "edge_instance": edge["name"] if edge else None,
        "signed_instance": signed["name"] if signed else None,
        "include_signed": bool(args.include_signed),
        "candidates": rows,
        "provenance": record_provenance(backend),
    }
    if args.out.exists():
        args.out.unlink()
    print(f"wrote {save_record(args.out, payload)}")
    return 0


def report(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    edge: dict[str, Any] | None,
) -> None:
    """
    Print the selection and the numbers behind it.

    Parameters
    ----------
    rows:
        Every evaluated candidate, already ranked.
    selected:
        The chosen subset.
    edge:
        The edge instance, or ``None``.
    """
    header = (
        f"{'name':<20} {'n':>2} {'b':>2} {'M':>3} {'r':>2} {'2Q':>6} {'2Qd':>6} "
        f"{'S':>10} {'P_ideal':>8} {'P_unif':>8} {'P_est':>10} {'2x':>3}"
    )
    print("\nselected subset")
    print(header)
    for row in sorted(selected, key=lambda r: (r["n_items"], r["m_class"], r["name"])):
        print(_line(row))

    print(f"\nedge instance  {edge['name'] if edge else 'NONE -- nothing passes the criterion'}")
    passing = [r for r in rows if r["beats_random_by_2x"]]
    print(f"candidates passing P_est >= 2*P_uniform: {len(passing)} of "
          f"{sum(1 for r in rows if r['M'] > 0)} feasible")
    if passing:
        widest = max(r["n_items"] for r in passing)
        print(f"widest passing instance has n = {widest}")


def _line(row: dict[str, Any]) -> str:
    """One formatted table row."""
    survival = "-" if row["survival"] is None else f"{row['survival']:.3e}"
    p_est = "-" if row["p_est"] is None else f"{row['p_est']:.3e}"
    verdict = {None: "n/a", True: "y", False: "n"}[row["beats_random_by_2x"]]
    return (
        f"{row['name']:<20} {row['n_items']:>2} {row['n_sum']:>2} {row['M']:>3} "
        f"{row['r_opt']:>2} {row['two_q_count']:>6} {row['two_q_depth']:>6} "
        f"{survival:>10} {row['p_ideal']:>8.4f} {row['p_uniform']:>8.4f} "
        f"{p_est:>10} {verdict:>3}"
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
