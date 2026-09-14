"""
build_sweep.py

WP3: the two instances the Grover-iteration sweep runs on (contribution C3).

The sweep asks where the algorithm stops working as iterations accumulate: it
runs the *same* instance at ``r = 1 .. r_max`` and watches the success
probability rise towards its ideal peak at ``r_opt`` and then be overtaken by
decoherence.  Two instances are enough to show the effect at two widths, and two
is what the QPU budget affords.

Selection rule
--------------
Both come from ``paper/instances/hardware_subset.json`` -- the sweep must not
introduce circuits the hardware plan has not already costed -- and both must
have

* ``M = 1``, so the ideal curve has a single clean peak rather than the flatter
  optimum a degenerate target produces, and ``P_uniform`` is unambiguous; and
* ``r_opt >= 2``, so there is a rise to observe at all. An ``r_opt = 1``
  instance has no interior to sweep.

Among the qualifying instances the two are taken at *distinct* ``n``, shallowest
first within each ``n``, so the pair brackets the width range instead of
sampling one width twice.

``r_max = r_opt + 1``
---------------------
One iteration past the optimum. That is the cheapest experiment that can
distinguish the two ways the curve can fail: if the ideal peak is real, the
noiseless probability falls at ``r_opt + 1`` (over-rotation) while the hardware
probability may already have been falling for several iterations. Going further
would multiply QPU seconds for a curve that is already monotonically decaying.

Usage
-----
::

    python paper/scripts/build_sweep.py
    python paper/scripts/build_sweep.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import INSTANCES_DIR, record_provenance, save_record  # noqa: E402
from dgssp.solvers.draper_grover import optimal_iterations  # noqa: E402

SCHEMA = "dgssp/paper/sweep/1"

#: How many instances the sweep runs on.
N_SWEEP_INSTANCES = 2

#: Minimum ``r_opt`` for an instance to have an interior worth sweeping.
MIN_R_OPT = 2


def qualifying(subset: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Hardware-subset instances eligible for the sweep.

    Parameters
    ----------
    subset:
        The loaded ``hardware_subset.json`` payload.

    Returns
    -------
    list[dict]
        Selected instances with ``M == 1`` and ``r_opt >= MIN_R_OPT``, ranked
        shallowest first by two-qubit depth with the name as a tie-break.
    """
    return sorted(
        (
            row
            for row in subset["selected"]
            if row["M"] == 1 and row["r_opt"] >= MIN_R_OPT
        ),
        key=lambda r: (r["two_q_depth"], r["two_q_count"], r["name"]),
    )


def choose(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Take the shallowest qualifying instance at each distinct ``n``.

    Falls back to the overall shallowest remainder if fewer distinct widths
    qualify than :data:`N_SWEEP_INSTANCES`, so the script still produces a
    usable pair rather than nothing.

    Parameters
    ----------
    candidates:
        Output of :func:`qualifying`, already ranked.

    Returns
    -------
    list[dict]
        At most :data:`N_SWEEP_INSTANCES` instances.
    """
    chosen: list[dict[str, Any]] = []
    widths: set[int] = set()
    for row in candidates:
        if row["n_items"] not in widths:
            chosen.append(row)
            widths.add(row["n_items"])
        if len(chosen) == N_SWEEP_INSTANCES:
            return chosen

    for row in candidates:
        if len(chosen) == N_SWEEP_INSTANCES:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen


def sweep_record(row: dict[str, Any]) -> dict[str, Any]:
    """
    Turn a selected instance into its sweep entry.

    Parameters
    ----------
    row:
        A row from ``hardware_subset.json``.

    Returns
    -------
    dict
        The instance, its ``r_opt``, ``r_max = r_opt + 1`` and the explicit
        iteration list the sweep submits.

    Raises
    ------
    AssertionError
        If the stored ``r_opt`` is not what ``optimal_iterations`` returns,
        which would mean the subset file and the library have drifted apart.
    """
    r_opt = int(row["r_opt"])
    assert r_opt == optimal_iterations(row["n_items"], row["M"]), row["name"]
    r_max = r_opt + 1
    return {
        "name": row["name"],
        "family": row["family"],
        "items": list(row["items"]),
        "target": row["target"],
        "M": row["M"],
        "solution_bitstrings": list(row["solution_bitstrings"]),
        "n_items": row["n_items"],
        "n_sum": row["n_sum"],
        "n_qubits": row["n_qubits"],
        "r_opt": r_opt,
        "r_max": r_max,
        "iterations": list(range(1, r_max + 1)),
        "p_uniform": row["p_uniform"],
        "p_ideal_at_r_opt": row["p_ideal"],
        "survival_at_r_opt": row["survival"],
        "two_q_count_at_r_opt": row["two_q_count"],
        "two_q_depth_at_r_opt": row["two_q_depth"],
    }


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
    parser = argparse.ArgumentParser(description="Pick the WP3 sweep instances.")
    parser.add_argument(
        "--subset", type=Path, default=INSTANCES_DIR / "hardware_subset.json"
    )
    parser.add_argument("--out", type=Path, default=INSTANCES_DIR / "sweep.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Choose the sweep instances and write ``sweep.json``.

    Parameters
    ----------
    argv:
        Argument list.

    Returns
    -------
    int
        0 on success, 1 on a missing input, a refusal to overwrite, or when
        fewer than two instances qualify.
    """
    args = parse_args(argv)
    if not args.subset.exists():
        print(
            f"{args.subset} not found; run "
            "`python paper/scripts/select_hardware_subset.py` first.",
            file=sys.stderr,
        )
        return 1
    if args.out.exists() and not args.overwrite:
        print(f"Refusing to write: {args.out} exists. Use --overwrite.", file=sys.stderr)
        return 1

    subset = json.loads(args.subset.read_text(encoding="utf-8"))
    candidates = qualifying(subset)
    chosen = choose(candidates)

    print(f"qualifying (M=1, r_opt >= {MIN_R_OPT}) in the hardware subset: "
          f"{len(candidates)}")
    for row in candidates:
        mark = "*" if row in chosen else " "
        print(f" {mark} {row['name']:<20} n={row['n_items']} b={row['n_sum']} "
              f"r_opt={row['r_opt']} 2Qd={row['two_q_depth']}")

    if len(chosen) < N_SWEEP_INSTANCES:
        print(
            f"Only {len(chosen)} instance(s) qualify; the sweep needs "
            f"{N_SWEEP_INSTANCES}. Widen the hardware subset first.",
            file=sys.stderr,
        )
        return 1

    records = [sweep_record(row) for row in chosen]
    payload = {
        "schema": SCHEMA,
        "source": args.subset.name,
        "family_payload_sha256": subset.get("family_payload_sha256"),
        "backend": subset.get("backend"),
        "policy": {
            "selected_from": "hardware_subset.json 'selected'",
            "requires": f"M == 1 and r_opt >= {MIN_R_OPT}",
            "prefers": "distinct n_items, shallowest two_q_depth within each",
            "r_max": "r_opt + 1",
        },
        "instances": records,
        "provenance": record_provenance(),
    }

    if args.out.exists():
        args.out.unlink()
    print(f"wrote {save_record(args.out, payload)}")
    for record in records:
        print(f"  {record['name']}: r = 1..{record['r_max']} (r_opt = {record['r_opt']})")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
