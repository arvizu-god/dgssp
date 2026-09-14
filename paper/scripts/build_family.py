"""
build_family.py

WP3: the frozen instance family the paper reports on.

Nothing here touches the circuit.  The algorithm is frozen; this script only
decides *which problems* the paper runs, and writes them down once so that
every later work package -- the hardware subset, the ZNE study, the iteration
sweep -- reads the same file instead of re-rolling its own dice.

Two families
------------
``P`` (positive)
    ``n`` in {2, 3, 4, 5, 6}, every item drawn i.i.d. uniform from
    ``[1, a_max(n)]``.  20 instances per ``(n, M-class)`` cell.

``S`` (signed)
    ``n`` in {3, 4, 5}, magnitudes drawn the same way, then exactly one item
    negated for half the cell and exactly two for the other half.  10 instances
    per ``(n, M-class)`` cell, i.e. 5 per sign count.

Why ``a_max(n) = floor(15 / n)``
-------------------------------
:func:`dgssp.instance.max_item_value` derives it: the encoding charges
``n_sum = ceil(log2(range_len)) + 1`` with ``range_len = sum(abs(a_i)) + 1``,
so ``n_sum <= 5`` holds exactly when ``sum(abs(a_i)) <= 15``, and ``n_sum >= 3``
holds automatically for ``n >= 2``.  Bounding the *magnitude* by ``15 / n``
makes the register width a guarantee rather than a rejection criterion, which
keeps the items genuinely i.i.d. uniform -- rejecting over-wide draws would
condition the item distribution on the sum.  The resulting bounds are

    n        2   3   4   5   6
    a_max    7   5   3   3   2

Because the same bound governs magnitudes in family S, and because ``range_len``
is invariant under sign flips (WP2, note SS4), family S has exactly the same
``n_sum`` distribution as family P at the same ``n``.  That is deliberate: it
isolates the *cost* of the signed encoding from the *width* it induces.

M-classes and targets
---------------------
For each drawn item set the full sum spectrum is enumerated with
:func:`dgssp.solvers.classical.subset_sum_multiplicities`, and a target is drawn
uniformly among those realising the requested class:

``0``
    Unreachable, but strictly *inside* the reachable interval
    ``[sum_neg, sum_pos]`` -- a gap, not an out-of-range value the encoding
    would reject on sight.  These are the infeasible instances contribution C3
    is about, and they are the reason the paper cannot use
    ``encoding.target_is_representable`` as a feasibility test: it is an
    interval test, and every instance in this file passes it.
``1``, ``2``, ``3+``
    Exactly one, exactly two, at least three exact solutions.

Target 0 is excluded throughout: the empty subset always reaches it, which
would make the M=1 class artificially easy and needs no search to find.

``r_opt`` is ``optimal_iterations(n, M)``.  For an M=0 instance
:func:`dgssp.solvers.draper_grover.optimal_iterations` clamps ``M`` to 1, so the
recorded ``r_opt`` is the iteration count an experimenter would use *not knowing*
the instance is infeasible.  That is the right number for the paper: the whole
point of the infeasible cells is what happens when you run the algorithm you
would actually have run.

Reproducibility
---------------
Every draw comes from one :class:`random.Random` seeded with ``--seed``, used
strictly in cell order, so the instance list is a pure function of the seed.
The provenance block carries a timestamp and therefore cannot be byte-identical
across runs; the deterministic part is hashed separately into
``payload_sha256``, and ``--check`` regenerates and compares that hash against
the file on disk without writing anything.

Usage
-----
::

    python paper/scripts/build_family.py
    python paper/scripts/build_family.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import INSTANCES_DIR, record_provenance, save_record  # noqa: E402
from dgssp.decoding import exact_solution_bitstrings  # noqa: E402
from dgssp.encoding import build_encoding  # noqa: E402
from dgssp.instance import SubsetSumInstance, max_item_value, random_instance  # noqa: E402
from dgssp.solvers.classical import (  # noqa: E402
    DPConfig,
    DPSSPSolver,
    subset_sum_multiplicities,
)
from dgssp.solvers.draper_grover import optimal_iterations  # noqa: E402

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

#: One master seed for the whole family.  Changing it changes every instance in
#: the paper, so it is a constant here rather than a default argument.
MASTER_SEED = 20261020

#: Schema tag written into the file, so a later reader can tell which layout it
#: is looking at without guessing.
SCHEMA = "dgssp/paper/instance-family/1"

#: Largest sum-register width the family admits, guard bit included.  Drives
#: :func:`dgssp.instance.max_item_value`.
MAX_N_SUM = 5

#: M-classes, in the order cells are filled.
M_CLASSES: tuple[str, ...] = ("0", "1", "2", "3+")

#: Family P: item counts and instances per ``(n, M-class)`` cell.
P_N_VALUES: tuple[int, ...] = (2, 3, 4, 5, 6)
P_PER_CELL = 20

#: Family S: item counts, negative-item counts, and instances per
#: ``(n, M-class)`` cell -- split evenly across the negative-item counts.
S_N_VALUES: tuple[int, ...] = (3, 4, 5)
S_N_NEGATIVE: tuple[int, ...] = (1, 2)
S_PER_CELL = 10

#: Draws :func:`dgssp.instance.random_instance` may make internally before it
#: declares a class unrealisable for a given ``n``.
DRAW_ATTEMPTS = 400

#: Consecutive duplicate draws that end a cell early.  Small cells are finite:
#: two positive items with a two-solution target means ``a == b`` and
#: ``t == a``, so the n=2/M=2 cell holds at most ``a_max`` distinct instances no
#: matter how long the loop runs.
STALL_LIMIT = 400


def cell_tag(m_class: str) -> str:
    """
    Filename-safe form of an M-class.

    Parameters
    ----------
    m_class:
        One of :data:`M_CLASSES`.

    Returns
    -------
    str
        ``"3+"`` becomes ``"3p"``; every other class is unchanged.
    """
    return m_class.replace("+", "p")


# ---------------------------------------------------------------------------
# One instance -> one record
# ---------------------------------------------------------------------------


def describe(instance: SubsetSumInstance, *, family: str, n_negative: int, m_class: str) -> dict[str, Any]:
    """
    Turn a generated instance into the JSON record the paper stores.

    Every derived quantity comes from the library: the register arithmetic from
    :func:`dgssp.encoding.build_encoding`, the solution bitstrings from
    :func:`dgssp.decoding.exact_solution_bitstrings` (the only place item index
    ordering is decided), the iteration count from
    :func:`dgssp.solvers.draper_grover.optimal_iterations`.

    Parameters
    ----------
    instance:
        The instance.
    family:
        ``"P"`` or ``"S"``.
    n_negative:
        How many items were negated, recorded rather than re-derived so the
        design of the draw stays visible in the file.
    m_class:
        The class the instance was drawn for; cross-checked against the
        measured multiplicity.

    Returns
    -------
    dict
        A JSON-serialisable instance record.

    Raises
    ------
    AssertionError
        If the measured multiplicity contradicts the requested class, or if the
        solution bitstrings disagree with it.
    """
    enc = build_encoding(instance.items, instance.target)
    multiplicity = subset_sum_multiplicities(instance.items).get(instance.target, 0)

    dp = DPSSPSolver(DPConfig(enumerate_all=True)).solve(instance)
    bitstrings = exact_solution_bitstrings(instance, dp)

    assert len(bitstrings) == multiplicity, (
        f"{instance.name}: {len(bitstrings)} solution bitstrings but "
        f"multiplicity {multiplicity}."
    )
    if m_class.endswith("+"):
        assert multiplicity >= int(m_class[:-1]), f"{instance.name}: M={multiplicity} not in {m_class}."
    else:
        assert multiplicity == int(m_class), f"{instance.name}: M={multiplicity} not in {m_class}."

    return {
        "name": instance.name,
        "family": family,
        "n_items": instance.n_items,
        "n_negative": n_negative,
        "items": list(instance.items),
        "target": instance.target,
        "m_class": m_class,
        "M": multiplicity,
        "solution_bitstrings": bitstrings,
        "n_sum": enc.n_sum,
        "sum_neg": enc.sum_neg,
        "sum_pos": enc.sum_pos,
        "range_len": enc.range_len,
        "offset": enc.offset,
        "modulus": enc.modulus,
        "encoded_target": enc.encoded_target,
        "target_is_representable": enc.target_is_representable,
        "r_opt": optimal_iterations(enc.n_ind, multiplicity),
    }


# ---------------------------------------------------------------------------
# Cell filling
# ---------------------------------------------------------------------------


def fill_cell(
    prng: random.Random,
    *,
    family: str,
    n_items: int,
    n_negative: int,
    m_class: str,
    wanted: int,
    name_prefix: str,
    seen: set[tuple[tuple[int, ...], int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Draw distinct instances for one cell until it is full or provably capped.

    Parameters
    ----------
    prng:
        The master stream.  Consumed in place, so cell order is part of the
        seed's meaning.
    family, n_items, n_negative, m_class:
        The cell coordinates.
    wanted:
        Target instance count.
    name_prefix:
        Prefix for the generated names; the index is appended.
    seen:
        ``(items, target)`` pairs already used anywhere in the family, so a
        cell cannot re-issue an instance another cell produced.

    Returns
    -------
    tuple[list[dict], dict]
        The instance records, and a report of what the cell managed:
        ``{"wanted", "produced", "status", "reason", "draws"}``.
    """
    a_max = max_item_value(n_items, max_n_sum=MAX_N_SUM)
    records: list[dict[str, Any]] = []
    draws = 0
    stall = 0

    while len(records) < wanted and stall < STALL_LIMIT:
        draws += 1
        try:
            instance = random_instance(
                n_items,
                a_max,
                target_strategy="multiplicity",
                target_multiplicity=m_class,
                n_negative=n_negative,
                allow_zero_target=False,
                max_attempts=DRAW_ATTEMPTS,
                rng=prng,
            )
        except ValueError as exc:
            # The generator exhausted its own redraw budget.  With no instance
            # at all the cell is unrealisable; with some, the cell is finite
            # and we found what there was.
            return records, {
                "wanted": wanted,
                "produced": len(records),
                "status": "empty" if not records else "exhausted",
                "reason": str(exc),
                "draws": draws,
            }

        key = (tuple(instance.items), instance.target)
        if key in seen:
            stall += 1
            continue

        seen.add(key)
        stall = 0
        instance.name = f"{name_prefix}-{len(records):02d}"
        records.append(
            describe(instance, family=family, n_negative=n_negative, m_class=m_class)
        )

    if len(records) == wanted:
        status, reason = "full", ""
    else:
        status = "capped"
        reason = (
            f"only {len(records)} distinct instances exist for n={n_items}, "
            f"a_max={a_max}, M-class {m_class!r}, n_negative={n_negative}: "
            f"{STALL_LIMIT} consecutive draws produced no new one."
        )
    return records, {
        "wanted": wanted,
        "produced": len(records),
        "status": status,
        "reason": reason,
        "draws": draws,
    }


def build_family(seed: int = MASTER_SEED) -> dict[str, Any]:
    """
    Generate the whole family, deterministically.

    Parameters
    ----------
    seed:
        Master seed.

    Returns
    -------
    dict
        The payload written to ``paper/instances/family.json``, minus the
        provenance block and the hash (both added by :func:`main`).
    """
    prng = random.Random(seed)
    seen: set[tuple[tuple[int, ...], int]] = set()
    instances: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []

    for n_items in P_N_VALUES:
        for m_class in M_CLASSES:
            tag = cell_tag(m_class)
            got, report = fill_cell(
                prng,
                family="P",
                n_items=n_items,
                n_negative=0,
                m_class=m_class,
                wanted=P_PER_CELL,
                name_prefix=f"P-n{n_items}-M{tag}",
                seen=seen,
            )
            instances.extend(got)
            cells.append(
                {
                    "family": "P",
                    "n_items": n_items,
                    "n_negative": 0,
                    "m_class": m_class,
                    "a_max": max_item_value(n_items, max_n_sum=MAX_N_SUM),
                    **report,
                }
            )

    per_sign = S_PER_CELL // len(S_N_NEGATIVE)
    for n_items in S_N_VALUES:
        for m_class in M_CLASSES:
            tag = cell_tag(m_class)
            for n_negative in S_N_NEGATIVE:
                got, report = fill_cell(
                    prng,
                    family="S",
                    n_items=n_items,
                    n_negative=n_negative,
                    m_class=m_class,
                    wanted=per_sign,
                    name_prefix=f"S-n{n_items}-g{n_negative}-M{tag}",
                    seen=seen,
                )
                instances.extend(got)
                cells.append(
                    {
                        "family": "S",
                        "n_items": n_items,
                        "n_negative": n_negative,
                        "m_class": m_class,
                        "a_max": max_item_value(n_items, max_n_sum=MAX_N_SUM),
                        **report,
                    }
                )

    return {
        "schema": SCHEMA,
        "master_seed": seed,
        "policy": {
            "max_n_sum": MAX_N_SUM,
            "m_classes": list(M_CLASSES),
            "a_max_rule": "floor((2**(max_n_sum - 1) - 1) / n_items)",
            "a_max": {
                str(n): max_item_value(n, max_n_sum=MAX_N_SUM)
                for n in sorted(set(P_N_VALUES) | set(S_N_VALUES))
            },
            "family_P": {
                "n_values": list(P_N_VALUES),
                "per_cell": P_PER_CELL,
                "n_negative": 0,
            },
            "family_S": {
                "n_values": list(S_N_VALUES),
                "per_cell": S_PER_CELL,
                "n_negative": list(S_N_NEGATIVE),
                "per_sign_count": per_sign,
            },
            "allow_zero_target": False,
            "m0_targets": "unreachable values strictly inside [sum_neg, sum_pos]",
            "r_opt": "optimal_iterations(n_items, M); M=0 is clamped to M=1 by the library",
            "draw_attempts": DRAW_ATTEMPTS,
            "stall_limit": STALL_LIMIT,
        },
        "cells": cells,
        "instances": instances,
    }


# ---------------------------------------------------------------------------
# Hashing and I/O
# ---------------------------------------------------------------------------


def payload_hash(payload: dict[str, Any]) -> str:
    """
    SHA-256 of the deterministic part of the payload.

    Provenance carries a timestamp, a hostname and installed versions, so the
    file cannot be byte-identical across runs.  What *must* be identical is
    everything the seed determines, and that is what this hashes: the payload
    with ``provenance`` and ``payload_sha256`` removed, serialised with sorted
    keys and no incidental whitespace.

    Parameters
    ----------
    payload:
        The family payload.

    Returns
    -------
    str
        Hex digest.
    """
    core = {k: v for k, v in payload.items() if k not in ("provenance", "payload_sha256")}
    blob = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--seed", type=int, default=MASTER_SEED, help="master seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=INSTANCES_DIR / "family.json",
        help="destination file",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate and compare payload_sha256 against the file on disk; write nothing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing destination instead of refusing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Build the family and write it, or verify an existing file.

    Parameters
    ----------
    argv:
        Argument list.

    Returns
    -------
    int
        Process exit status: 0 on success, 1 on a failed ``--check`` or a
        refusal to overwrite.
    """
    args = parse_args(argv)

    payload = build_family(args.seed)
    digest = payload_hash(payload)

    total = len(payload["instances"])
    short = [c for c in payload["cells"] if c["status"] != "full"]
    full = len(payload["cells"]) - len(short)

    print(f"seed          {args.seed}")
    print(f"instances     {total}")
    print(f"cells         {len(payload['cells'])} ({full} full, {len(short)} short)")
    print(f"payload_sha256 {digest}")

    for cell in short:
        print(
            f"  ! {cell['family']} n={cell['n_items']} g={cell['n_negative']} "
            f"M={cell['m_class']:>2}  {cell['status']}: "
            f"{cell['produced']}/{cell['wanted']} -- {cell['reason']}"
        )

    if args.check:
        if not args.out.exists():
            print(f"FAIL: {args.out} does not exist.", file=sys.stderr)
            return 1
        on_disk = json.loads(args.out.read_text(encoding="utf-8"))
        stored = on_disk.get("payload_sha256")
        if stored == digest:
            print(f"OK: {args.out.name} matches the seed-{args.seed} regeneration.")
            return 0
        print(f"FAIL: {args.out.name} has payload_sha256 {stored}, regenerated {digest}.",
              file=sys.stderr)
        return 1

    if args.out.exists() and not args.overwrite:
        print(
            f"Refusing to write: {args.out} exists. Use --check to verify it, or "
            "--overwrite to replace it.",
            file=sys.stderr,
        )
        return 1
    if args.out.exists():
        args.out.unlink()

    payload["payload_sha256"] = digest
    payload["provenance"] = record_provenance()
    written = save_record(args.out, payload)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
