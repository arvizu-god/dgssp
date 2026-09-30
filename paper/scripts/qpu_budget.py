"""
qpu_budget.py

How many QPU minutes the monthly hardware job would cost -- computed offline,
from the device's own calibrated instruction durations.

Nothing here touches the network, submits anything or consumes allowance.  It
builds exactly the circuits the hardware run would submit (the same
``prepare_from_records`` + ``fold_isa`` path ``sim_campaign.py`` uses, so the
folded circuits are the same objects, not a re-implementation), schedules each
one against the backend's ``target``, and applies the execution-time model
below.

The model
---------

IBM charges for the time the QPU is dedicated to the job.  For a Sampler job
that is, per circuit::

    shots x (circuit_duration + rep_delay)

``circuit_duration`` is the ASAP-scheduled critical path of the ISA circuit,
which the target gives exactly: every calibrated instruction carries a
duration in seconds (``cz`` 68 ns, ``sx`` 36 ns, ``measure`` 2.58 us, ``rz``
0 -- it is a frame change).  ``rep_delay`` is the idle time between shots that
lets the qubits relax back to the ground state; the device advertises its
default and its permitted range.

**On these circuits ``rep_delay`` dominates.**  A three-item instance schedules
in about 40 us against a 250 us default delay, so roughly six of every seven
QPU seconds are spent waiting for qubits to reset, not computing.  That is the
lever worth knowing about before shaping a job: halving the shot count halves
the cost, while halving the circuit depth barely moves it.

What this is not
----------------

It is a *floor*, not the invoice.  It counts circuit execution only.  The
billed figure also carries per-job overhead that is not observable from here,
and the authoritative number is ``job.usage_estimation`` on a real submission
(see the ``[verify]`` items in ``paper/hardware/account_facts.md``).  Treat
this as the number that decides whether a job shape is plausible, then confirm
it against one real job before planning a month around it.

Run::

    python paper/scripts/qpu_budget.py
    python paper/scripts/qpu_budget.py --shots 4096 --grid odd
    python paper/scripts/qpu_budget.py --selected --shots 8192 --rep-delay 0.0005
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import HARDWARE_DIR, INSTANCES_DIR, save_record  # noqa: E402
from dgssp.experiments import prepare_from_records  # noqa: E402
from dgssp.mitigation.zne import (  # noqa: E402
    ZNESamplingConfig,
    fold_isa,
    rebase_to_backend,
)
from dgssp.runtime import backend_name  # noqa: E402
from sim_campaign import (  # noqa: E402
    FOLD_SEED,
    ZNE_SHOTS,
    resolve_fake_backend,
    scales_for,
)

#: Fallback inter-shot delay, in seconds, when the backend advertises none.
#: IBM's documented default for Heron-class devices.
DEFAULT_REP_DELAY = 250e-6

#: Open-plan monthly allowance, in minutes.  Confirm against the dashboard and
#: record it in ``paper/hardware/account_facts.md`` -- this is a placeholder for
#: arithmetic, not a fact about your account.
OPEN_PLAN_MINUTES = 10.0


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def schedule_seconds(circuit: Any, backend: Any) -> float:
    """
    ASAP-scheduled duration of an ISA circuit, in seconds.

    Parameters
    ----------
    circuit:
        A circuit already expressed in the backend's basis and layout.
    backend:
        The backend whose ``target`` supplies the instruction durations.

    Returns
    -------
    float
        Wall-clock length of one execution of the circuit.

    Raises
    ------
    ValueError
        If the scheduler returns a duration in units the target cannot
        convert -- which means the circuit contains an instruction the device
        has no calibrated duration for, and any estimate would be a guess.
    """
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ASAPScheduleAnalysis, PadDelay

    scheduled = PassManager(
        [ASAPScheduleAnalysis(target=backend.target), PadDelay(target=backend.target)]
    ).run(circuit)

    duration = getattr(scheduled, "duration", None)
    unit = getattr(scheduled, "unit", None)
    if duration is None:
        raise ValueError(
            f"{circuit.name!r} could not be scheduled against "
            f"{backend_name(backend)}; some instruction has no calibrated duration."
        )
    if unit == "dt":
        dt = backend.target.dt
        if dt is None:
            raise ValueError(
                f"{backend_name(backend)} reports a schedule in dt but no dt value."
            )
        return float(duration) * float(dt)
    if unit == "s":
        return float(duration)
    raise ValueError(f"Unexpected schedule unit {unit!r} for {circuit.name!r}.")


def device_limits(backend: Any) -> dict[str, Any]:
    """
    The execution limits a job shape has to respect.

    Parameters
    ----------
    backend:
        A backend exposing ``configuration()`` (real devices and fake twins do).

    Returns
    -------
    dict
        ``{"max_shots", "max_circuits", "default_rep_delay",
        "rep_delay_range", "dt"}``; entries the backend does not advertise are
        ``None``.
    """
    config = None
    if hasattr(backend, "configuration"):
        try:
            config = backend.configuration()
        except Exception:  # pragma: no cover - provider dependent
            config = None

    max_circuits = getattr(backend, "max_circuits", None)
    if max_circuits is None and config is not None:
        max_circuits = getattr(config, "max_experiments", None)

    return {
        "max_shots": getattr(config, "max_shots", None),
        "max_circuits": max_circuits,
        "default_rep_delay": getattr(config, "default_rep_delay", None),
        "rep_delay_range": getattr(config, "rep_delay_range", None),
        "dt": getattr(backend.target, "dt", None),
    }


def qpu_seconds(duration: float, shots: int, rep_delay: float) -> float:
    """
    Execution time of one circuit at one shot count.

    Parameters
    ----------
    duration:
        Scheduled circuit duration, in seconds.
    shots:
        Shots for this circuit.
    rep_delay:
        Idle time between shots, in seconds.

    Returns
    -------
    float
        Seconds of QPU time.
    """
    return shots * (duration + rep_delay)


# ---------------------------------------------------------------------------
# Job shape
# ---------------------------------------------------------------------------


def job_circuits(
    rows: Sequence[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    backend: Any,
    *,
    grid: str,
    shots: int,
    include_mitiq: bool,
) -> list[dict[str, Any]]:
    """
    Build every circuit the hardware job would submit, and time each one.

    This is the same path ``sim_campaign.py`` takes -- transpile once at the
    recorded ``best_seed``, fold the ISA circuit at the instance's scales,
    rebase to the native basis -- so the circuits timed here are the circuits
    that would run, not a model of them.

    Parameters
    ----------
    rows:
        ``family.json``-shaped instance records.
    candidates:
        ``hardware_subset.json`` rows, keyed by name.
    backend:
        The device (or its fake twin) to transpile and schedule against.
    grid:
        Scale grid name, as ``sim_campaign.scales_for`` understands it.
    shots:
        Shots per circuit.
    include_mitiq:
        Also count Mitiq's folds.  Mitiq reuses the unfolded circuit at
        ``lambda = 1``, so it adds one circuit per scale above 1.

    Returns
    -------
    list[dict]
        One entry per circuit: ``name``, ``scale``, ``source``
        (``"dgssp"`` / ``"mitiq"``), ``two_q``, ``duration_s``.
    """
    cfg = ZNESamplingConfig(optimization_level=3, fold_seed=FOLD_SEED)
    out: list[dict[str, Any]] = []

    for row in rows:
        candidate = candidates.get(row["name"])
        scales, _ = scales_for(row, candidate, grid, shots)
        prepared = prepare_from_records(
            row, candidate, backend, cfg, scales=scales, verify_solutions=False
        )
        for lam in prepared.scales:
            fold = fold_isa(prepared.transpiled, lam, seed=FOLD_SEED)
            circuit = rebase_to_backend(fold.circuit, backend)
            entry = {
                "name": prepared.name,
                "n_items": row["n_items"],
                "scale": float(fold.realised_scale),
                "source": "dgssp",
                "two_q": fold.n_two_q + fold.n_two_q_extra,
                "duration_s": schedule_seconds(circuit, backend),
            }
            out.append(entry)
            if include_mitiq and float(fold.realised_scale) != 1.0:
                # Mitiq folds to the same realised scale, so its circuit has
                # the same two-qubit count and, to within layout accidents,
                # the same schedule.  Counting it as a copy keeps this a
                # budget estimate rather than a second folding run.
                out.append({**entry, "source": "mitiq"})
    return out


def summarise(
    circuits: Sequence[dict[str, Any]], *, shots: int, rep_delay: float
) -> dict[str, Any]:
    """
    Total QPU time, and where it goes.

    Parameters
    ----------
    circuits:
        As :func:`job_circuits` returns.
    shots:
        Shots per circuit.
    rep_delay:
        Inter-shot delay, in seconds.

    Returns
    -------
    dict
        Totals, the split between computing and waiting, and per-``n``
        subtotals.
    """
    total = sum(qpu_seconds(c["duration_s"], shots, rep_delay) for c in circuits)
    compute = sum(c["duration_s"] * shots for c in circuits)
    by_n: dict[int, dict[str, Any]] = {}
    for circuit in circuits:
        cell = by_n.setdefault(
            circuit["n_items"], {"circuits": 0, "seconds": 0.0, "two_q": 0}
        )
        cell["circuits"] += 1
        cell["seconds"] += qpu_seconds(circuit["duration_s"], shots, rep_delay)
        cell["two_q"] += circuit["two_q"]

    return {
        "n_circuits": len(circuits),
        "shots": shots,
        "rep_delay_s": rep_delay,
        "total_seconds": total,
        "total_minutes": total / 60.0,
        "compute_seconds": compute,
        "idle_seconds": total - compute,
        "idle_fraction": (total - compute) / total if total else None,
        "median_duration_s": (
            sorted(c["duration_s"] for c in circuits)[len(circuits) // 2]
            if circuits
            else None
        ),
        "max_duration_s": max((c["duration_s"] for c in circuits), default=None),
        "by_n": {str(k): v for k, v in sorted(by_n.items())},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv:
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument(
        "--selected",
        action="store_true",
        help="Only the hardware subset's selected instances (the default is "
        "the whole subset file's selection).",
    )
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--grid", default="planned", choices=("planned", "odd", "odd7"))
    parser.add_argument("--shots", type=int, default=ZNE_SHOTS)
    parser.add_argument(
        "--rep-delay",
        type=float,
        default=None,
        help="Inter-shot delay in seconds; default is the device's own.",
    )
    parser.add_argument("--backend", default=None)
    parser.add_argument("--no-mitiq", action="store_true")
    parser.add_argument(
        "--allowance",
        type=float,
        default=OPEN_PLAN_MINUTES,
        help="Monthly QPU minutes, for the fraction-of-budget line.",
    )
    parser.add_argument("--instances-dir", type=Path, default=INSTANCES_DIR)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=f"Write the estimate here (bare names land in {HARDWARE_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Estimate the monthly job's QPU time.

    Parameters
    ----------
    argv:
        Command line; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        ``0`` on success, ``1`` if the job shape breaks a device limit.
    """
    args = parse_args(argv)
    subset = json.loads(
        (args.instances_dir / "hardware_subset.json").read_text(encoding="utf-8")
    )
    candidates = {row["name"]: row for row in subset["candidates"]}

    if args.only:
        rows = [candidates[name] for name in args.only if name in candidates]
    else:
        rows = list(subset["selected"])
    if not rows:
        print("No instances selected.")
        return 1

    backend = resolve_fake_backend(args.backend or subset["backend"])
    limits = device_limits(backend)
    rep_delay = (
        args.rep_delay
        if args.rep_delay is not None
        else (limits["default_rep_delay"] or DEFAULT_REP_DELAY)
    )

    from dgssp.mitigation import mitiq_is_available

    include_mitiq = (not args.no_mitiq) and mitiq_is_available()

    print(
        f"backend={backend_name(backend)}  grid={args.grid}  shots={args.shots}  "
        f"rep_delay={rep_delay * 1e6:.0f} us  mitiq={'on' if include_mitiq else 'off'}"
    )
    print(
        f"device limits: max_shots={limits['max_shots']}  "
        f"max_circuits/job={limits['max_circuits']}  "
        f"rep_delay_range={limits['rep_delay_range']}"
    )

    circuits = job_circuits(
        rows,
        candidates,
        backend,
        grid=args.grid,
        shots=args.shots,
        include_mitiq=include_mitiq,
    )
    stats = summarise(circuits, shots=args.shots, rep_delay=rep_delay)

    print(f"\n{len(rows)} instance(s) -> {stats['n_circuits']} circuits\n")
    print(f"{'n':>2}  {'circuits':>8}  {'2q gates':>9}  {'QPU s':>8}  {'share':>6}")
    for n_items, cell in stats["by_n"].items():
        share = cell["seconds"] / stats["total_seconds"] if stats["total_seconds"] else 0
        print(
            f"{n_items:>2}  {cell['circuits']:>8}  {cell['two_q']:>9}  "
            f"{cell['seconds']:>8.1f}  {share:>5.0%}"
        )

    print(
        f"\nmedian circuit {stats['median_duration_s'] * 1e6:.1f} us, "
        f"longest {stats['max_duration_s'] * 1e6:.1f} us"
    )
    print(
        f"computing {stats['compute_seconds']:.1f} s, "
        f"waiting for reset {stats['idle_seconds']:.1f} s "
        f"({stats['idle_fraction']:.0%} of the total)"
    )
    print(
        f"\nESTIMATED QPU TIME: {stats['total_seconds']:.1f} s "
        f"= {stats['total_minutes']:.2f} min "
        f"({stats['total_minutes'] / args.allowance:.0%} of a "
        f"{args.allowance:g}-minute monthly allowance)"
    )

    status = 0
    if limits["max_circuits"] and stats["n_circuits"] > limits["max_circuits"]:
        print(
            f"\nFAIL: {stats['n_circuits']} circuits exceeds this device's "
            f"limit of {limits['max_circuits']} per job. Split the job or drop "
            "a scale."
        )
        status = 1
    if limits["max_shots"] and args.shots > limits["max_shots"]:
        print(f"\nFAIL: {args.shots} shots exceeds max_shots={limits['max_shots']}.")
        status = 1

    print(
        "\nThis is a floor: circuit execution only, no per-job overhead. "
        "Confirm against job.usage_estimation on one real submission before "
        "planning a month around it."
    )

    if args.json:
        path = args.json if args.json.parent != Path(".") else HARDWARE_DIR / args.json.name
        payload = {
            "schema": "dgssp/paper/qpu-budget/1",
            "backend": backend_name(backend),
            "grid": args.grid,
            "include_mitiq": include_mitiq,
            "instances": [row["name"] for row in rows],
            "device_limits": limits,
            "model": (
                "shots * (ASAP-scheduled circuit duration + rep_delay), "
                "circuit execution only"
            ),
            **stats,
            "circuits": circuits,
        }
        print(f"\nwrote {save_record(path, payload)}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
