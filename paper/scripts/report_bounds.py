"""
report_bounds.py

Step 5, part 1: how a reported scalar ZNE value outside the physical range is
handled.  Analysis only; nothing is wired in and no record is edited.

The physical range of the noiseless ``P(solution)`` is
:func:`dgssp.mitigation.zne.physical_bounds` -- ``[P_u, 1]`` for an instance
whose ideal lies above random guessing.  Three options
(:data:`~dgssp.mitigation.zne.REPORT_BOUNDS_OPTIONS`):

* ``flag``    -- the fit and the stored interval unchanged, plus range flags;
* ``clip``    -- point and interval clipped to the range after fitting;
* ``bounded`` -- ``a + b`` constrained to the range inside the fit, bootstrap
  re-run with the bound.

For every record with a fixed-asymptote exponential fit (simulation: odd,
odd7 and planned grids; hardware: both jobs):

1. Inventory: every *reported* extrapolation (resolved and identifiable,
   rule ``RESOLVED_RULE``) outside ``[0, 1]`` or outside its range, with its
   interval, and whether that interval still meets the range.
2. Checks: the bounded fit's resolved / identifiable / reported-source flags
   equal the re-derived ones on every record (the gate runs on the data); the
   unbounded bootstrap, re-run at the record's ``B`` and seed, reproduces the
   stored interval.
3. Comparison on every resolved fit, per grid and per n: median bias, RMSE and
   median absolute error against the exact ideal; coverage of the ideal by
   the 95 % interval (also split at ``P_ideal = 1``, where an interval ending
   on the bound covers by construction); median interval width; counts
   outside ``[0, 1]`` and outside the range.  Primary: the record's
   ``B = 500``, seed 0 (so ``flag`` *is* the stored interval); sensitivity:
   ``B = 2000``, seed 0, for all three options.

Output: ``paper/data/report_bounds.json`` (``save_record`` never overwrites).

Run::

    python paper/scripts/report_bounds.py
    python paper/scripts/report_bounds.py --out <scratch>/report_bounds.json
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import DATA_DIR, load_records, record_provenance, save_record  # noqa: E402
from aggregate import exact_p_ideal, rederive_flags  # noqa: E402
from dgssp.metrics import solution_probability  # noqa: E402
from dgssp.mitigation.zne import (  # noqa: E402
    REPORT_BOUNDS_OPTIONS,
    RESOLVED_RULE,
    SIGNAL_TOL,
    ZNESamplingConfig,
    extrapolate_scalar,
    fit_block_config,
    physical_bounds,
    report_bounded,
    report_clip,
    report_flag,
    signal_direction,
    zne_bootstrap,
)
from zne_theory_check import (  # noqa: E402
    HW_JOBS,
    SIM_GRIDS,
    counts_by_scale,
    load_sim,
    quantiles,
    record_key,
)

SCHEMA = "report_bounds/1"

#: Bootstrap of the sensitivity run (rule 5); the primary run uses each
#: record's own ``B`` and seed so ``flag`` reproduces the stored interval.
B_SENSITIVITY = 2000
SEED = 0
ALPHA = 0.05

#: ``lo <= P_ideal <= hi`` up to this (the exact ideal of a ``P = 1``
#: instance is a statevector sum, ``1 +- 1e-16``).
COVER_TOL = 1e-12

#: Stored vs recomputed interval: anything above this is a failed check.  The
#: campaigns ran on another platform; the golden-section search lands within
#: ~1e-8 of the stored value there (zne_theory.json saw the same).
REPRODUCE_TOL = 1e-6


# ---------------------------------------------------------------------------
# Per record
# ---------------------------------------------------------------------------


def _config(block: Mapping[str, Any], B: int, seed: int) -> ZNESamplingConfig:
    """The scalar configuration a stored block was fitted with, at bootstrap ``B``, ``seed``."""
    return replace(fit_block_config(block), n_bootstrap=int(B), bootstrap_seed=int(seed))


def _options(
    counts: Mapping[float, Mapping[str, int]],
    observable: Callable[[Mapping[str, int]], float],
    block: Mapping[str, Any],
    p_ideal: float | None,
    bounds: tuple[float, float],
    B: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], list[float]]:
    """The three reports of one resolved fit at one bootstrap setting."""
    p_u = float(block["null_value"])
    cfg = _config(block, B, seed)
    unbounded = zne_bootstrap(
        counts, cfg, observable_fn=observable, p_uniform=p_u, p_ideal=p_ideal, B=B, seed=seed,
        alpha=ALPHA,
    )
    ci = [float(unbounded["lo"]), float(unbounded["hi"])]
    value = float(unbounded["point"])
    out = {
        "flag": report_flag(value, ci, bounds),
        "clip": report_clip(value, ci, bounds),
        "bounded": report_bounded(
            counts, observable, cfg, p_uniform=p_u, p_ideal=p_ideal, B=B, seed=seed, alpha=ALPHA
        ),
    }
    return out, ci


def record_row(record: Mapping[str, Any], group: str) -> dict[str, Any] | None:
    """
    One record: inventory fields, the flag check, and the three reports.

    ``None`` for a record without a fixed-asymptote exponential fit.
    """
    block = record.get("zne_fit") or {}
    if (
        block.get("method") != "exponential"
        or block.get("asymptote") is None
        or not block.get("observed")
    ):
        return None
    p_ideal = exact_p_ideal(dict(record))
    p_u = float(block["null_value"])
    flags = rederive_flags(dict(block), p_ideal, report_bounds="flag")
    direction = signal_direction(p_ideal, p_u)
    bounds = physical_bounds(p_u, direction)
    counts = counts_by_scale(record)
    observable = partial(
        solution_probability, solution_states=record["metrics"]["solution_bitstrings"]
    )

    bounded_point = extrapolate_scalar(
        counts,
        observable,
        _config(block, int(block["B"]), int(block["bootstrap_seed"])),
        p_uniform=p_u,
        p_ideal=p_ideal,
        value_bounds=bounds,
    )
    flag_keys = ("resolved", "identifiable", "reported_source")
    flags_changed = [k for k in flag_keys if bounded_point[k] != flags[k]]

    lo, hi = (float(v) for v in block["ci"])
    row: dict[str, Any] = {
        "group": group,
        "name": record_key(record),
        "n": int(record["n_items"]),
        "M": int(record["metrics"]["num_solutions"]),
        "p_uniform": p_u,
        "p_ideal": p_ideal,
        "direction": direction,
        "bounds": list(bounds),
        "resolved": flags["resolved"],
        "identifiable": flags["identifiable"],
        "reported_source": flags["reported_source"],
        "reported": flags["reported_source"] == "extrapolation",
        "value": float(block["value"]),
        "ci": [lo, hi],
        "B": int(block["B"]),
        "seed": int(block["bootstrap_seed"]),
        "flags_changed_by_bound": flags_changed,
    }
    if not row["reported"]:
        return row

    primary, ci_recomputed = _options(
        counts, observable, block, p_ideal, bounds, row["B"], row["seed"]
    )
    sensitivity, _ = _options(counts, observable, block, p_ideal, bounds, B_SENSITIVITY, SEED)
    row["ci_reproduce_diff"] = max(abs(ci_recomputed[0] - lo), abs(ci_recomputed[1] - hi))
    row["options"] = {"primary": primary, "sensitivity": sensitivity}
    return row


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def _meets(ci: Sequence[float], bounds: Sequence[float]) -> bool:
    """Does the interval contain at least one value of the range?"""
    return bool(ci[1] >= bounds[0] and ci[0] <= bounds[1])


def inventory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reported values and intervals outside ``[0, 1]`` or the range."""
    reported = [r for r in rows if r["reported"]]

    def entry(r: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "name": r["name"],
            "n": r["n"],
            "M": r["M"],
            "value": r["value"],
            "ci": r["ci"],
            "p_uniform": r["p_uniform"],
            "p_ideal": r["p_ideal"],
            "ci_meets_range": _meets(r["ci"], r["bounds"]),
            "ci_covers_ideal": (
                None if r["p_ideal"] is None
                else bool(r["ci"][0] - COVER_TOL <= r["p_ideal"] <= r["ci"][1] + COVER_TOL)
            ),
        }

    outside_unit = [r for r in reported if not 0.0 <= r["value"] <= 1.0]
    outside_range = [r for r in reported if not r["bounds"][0] <= r["value"] <= r["bounds"][1]]
    below_pu = [r for r in reported if r["value"] < r["p_uniform"]]
    ci_out = [
        r for r in reported if r["ci"][0] < r["bounds"][0] or r["ci"][1] > r["bounds"][1]
    ]
    unreported_raw = [r for r in rows if not r["reported"] and not 0.0 <= r["value"] <= 1.0]
    return {
        "n_fits": len(rows),
        "n_signal": sum(1 for r in rows if r["direction"] != 0),
        "n_direction_negative": sum(1 for r in rows if r["direction"] < 0),
        "n_reported": len(reported),
        "n_points_outside_unit": len(outside_unit),
        "n_points_outside_range": len(outside_range),
        "n_points_below_p_uniform": len(below_pu),
        "points_outside_range": [entry(r) for r in outside_range],
        "points_outside_range_value_span": (
            [min(r["value"] for r in outside_range), max(r["value"] for r in outside_range)]
            if outside_range else None
        ),
        "n_points_outside_range_whose_ci_meets_range": sum(
            _meets(r["ci"], r["bounds"]) for r in outside_range
        ),
        "n_ci_outside_range": len(ci_out),
        "n_ci_above_one": sum(1 for r in reported if r["ci"][1] > 1.0),
        "n_ci_below_p_uniform": sum(1 for r in reported if r["ci"][0] < r["p_uniform"]),
        "ci_outside_range_max_hi": max((r["ci"][1] for r in ci_out), default=None),
        "ci_outside_range": [entry(r) for r in ci_out],
        "n_unreported_raw_outside_unit": len(unreported_raw),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _at_one(p_ideal: float) -> bool:
    return abs(p_ideal - 1.0) <= SIGNAL_TOL


def option_metrics(reports: Sequence[Mapping[str, Any]], ideals: Sequence[float]) -> dict[str, Any]:
    """Error, coverage, width and range counts of one option on a set of fits."""
    if not reports:
        return {"n": 0}
    value = np.array([r["value"] for r in reports])
    lo = np.array([r["ci"][0] for r in reports])
    hi = np.array([r["ci"][1] for r in reports])
    bounds = np.array([r["bounds"] for r in reports])
    ideal = np.array(ideals, dtype=float)
    err = value - ideal
    covered = (lo - COVER_TOL <= ideal) & (ideal <= hi + COVER_TOL)
    at_one = np.array([_at_one(p) for p in ideal])

    def frac(mask: np.ndarray) -> float | None:
        return float(np.mean(covered[mask])) if mask.any() else None

    return {
        "n": int(value.size),
        "median_bias": float(np.median(err)),
        "mean_bias": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "median_abs_error": float(np.median(np.abs(err))),
        "coverage": float(np.mean(covered)),
        "coverage_ideal_at_one": frac(at_one),
        "coverage_ideal_below_one": frac(~at_one),
        "n_ideal_at_one": int(at_one.sum()),
        "median_ci_width": float(np.median(hi - lo)),
        "n_outside_unit": int(np.sum((value < 0.0) | (value > 1.0))),
        "n_outside_range": int(np.sum((value < bounds[:, 0]) | (value > bounds[:, 1]))),
        "n_ci_outside_range": int(np.sum((lo < bounds[:, 0]) | (hi > bounds[:, 1]))),
        "n_value_at_bound": int(
            sum(bool(r.get("value_at_bound")) for r in reports)
        ),
    }


def compare(rows: Sequence[Mapping[str, Any]], setting: str) -> dict[str, Any]:
    """Every option's metrics on the resolved fits of ``rows``, overall and by n."""
    fits = [r for r in rows if r["reported"] and r["p_ideal"] is not None]

    def block(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        ideals = [r["p_ideal"] for r in members]
        return {
            option: option_metrics([r["options"][setting][option] for r in members], ideals)
            for option in REPORT_BOUNDS_OPTIONS
        }

    return {
        "overall": block(fits),
        "by_n": {str(n): block([r for r in fits if r["n"] == n]) for n in sorted({r["n"] for r in fits})},
    }


def flag_check(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Records whose flags differ between the bounded and the unbounded fit."""
    rows = list(rows)
    changed = [r["name"] for r in rows if r["flags_changed_by_bound"]]
    return {"n_records": len(rows), "n_changed": len(changed), "changed": changed}


def hardware_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each resolved hardware fit under each option (primary setting)."""
    out = []
    for r in rows:
        if not r["reported"]:
            continue
        entry: dict[str, Any] = {
            "group": r["group"], "name": r["name"], "p_uniform": r["p_uniform"],
            "p_ideal": r["p_ideal"],
        }
        for setting in ("primary", "sensitivity"):
            for option in REPORT_BOUNDS_OPTIONS:
                rep = r["options"][setting][option]
                entry[f"{setting}.{option}"] = {
                    "value": rep["value"],
                    "ci": rep["ci"],
                    "covers_ideal": bool(
                        rep["ci"][0] - COVER_TOL <= r["p_ideal"] <= rep["ci"][1] + COVER_TOL
                    ),
                    "error": rep["value"] - r["p_ideal"],
                    **({"value_at_bound": rep["value_at_bound"], "rate": rep["rate"]}
                       if option == "bounded" else {}),
                }
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _strip(row: Mapping[str, Any]) -> dict[str, Any]:
    """A row as stored in the JSON (the option reports in compact form)."""
    keep = {k: v for k, v in row.items() if k != "options"}
    if "options" in row:
        keep["options"] = {
            setting: {
                option: {k: rep[k] for k in ("value", "ci") if k in rep}
                | ({"value_at_bound": rep["value_at_bound"], "rate": rep["rate"]}
                   if option == "bounded" else {})
                for option, rep in reports.items()
            }
            for setting, reports in row["options"].items()
        }
    return keep


def _fmt(v: Any, d: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{d}f}" if math.isfinite(v) else str(v)
    return str(v)


def print_summary(payload: Mapping[str, Any]) -> None:
    """The acceptance numbers, on stdout."""
    for group, inv in payload["inventory"].items():
        print(
            f"{group:<10} fits={inv['n_fits']:<4} reported={inv['n_reported']:<4} "
            f"pt>1|<Pu={inv['n_points_outside_range']:<3} (ci meets range "
            f"{inv['n_points_outside_range_whose_ci_meets_range']}) "
            f"span={inv['points_outside_range_value_span']} ci out={inv['n_ci_outside_range']} "
            f"dir<0={inv['n_direction_negative']}"
        )
    print("flags changed by the bound:",
          {g: v["n_changed"] for g, v in payload["checks"]["flags_unchanged"].items()})
    print("max |stored ci - recomputed ci|:", payload["checks"]["max_ci_reproduce_diff"])
    for setting in ("primary", "sensitivity"):
        print(f"--- {setting}")
        for group, comp in payload["comparison"][setting].items():
            for option, m in comp["overall"].items():
                if not m.get("n"):
                    continue
                print(
                    f"{group:<10} {option:<8} n={m['n']:<4} bias={_fmt(m['median_bias'])} "
                    f"rmse={_fmt(m['rmse'])} mae={_fmt(m['median_abs_error'])} "
                    f"cov={_fmt(m['coverage'], 3)} (at1 {_fmt(m['coverage_ideal_at_one'], 3)}, "
                    f"<1 {_fmt(m['coverage_ideal_below_one'], 3)}) "
                    f"width={_fmt(m['median_ci_width'])} out01={m['n_outside_unit']} "
                    f"ci_out={m['n_ci_outside_range']} at_bound={m['n_value_at_bound']}"
                )


def main(argv: list[str] | None = None) -> int:
    """Inventory, checks and comparison; write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", type=Path, default=DATA_DIR / "report_bounds.json",
                        help="output JSON (never overwritten)")
    args = parser.parse_args(argv)
    started = time.time()

    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    for grid in SIM_GRIDS:
        rows_by_group[grid] = [
            row for record in load_sim(grid) if (row := record_row(record, grid)) is not None
        ]
    for job, (folder, job_id) in HW_JOBS.items():
        rows = []
        for record in load_records(DATA_DIR / folder / "*.json"):
            if record.get("job_id") != job_id:
                raise ValueError(f"{record['_source']}: job {record.get('job_id')} != {job_id}")
            row = record_row(record, f"hw_{job}")
            if row is not None:
                rows.append(row)
        rows_by_group[f"hw_{job}"] = rows
    pooled = {
        "sim_pooled": [r for g in SIM_GRIDS for r in rows_by_group[g]],
        "hw_pooled": [r for job in HW_JOBS for r in rows_by_group[f"hw_{job}"]],
    }
    groups = {**rows_by_group, **pooled}

    reproduce = [r["ci_reproduce_diff"] for rows in rows_by_group.values() for r in rows
                 if "ci_reproduce_diff" in r]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "parameters": {
            "options": list(REPORT_BOUNDS_OPTIONS),
            "resolved_rule": RESOLVED_RULE,
            "sim_grids": list(SIM_GRIDS),
            "hardware_jobs": {k: {"folder": v[0], "job_id": v[1]} for k, v in HW_JOBS.items()},
            "primary": "each record's stored B and bootstrap_seed (500, 0): flag = stored interval",
            "sensitivity": {"B": B_SENSITIVITY, "seed": SEED},
            "alpha": ALPHA,
            "ideal": "exact P(solution) of the record's reference distribution",
            "cover_tol": COVER_TOL,
            "ideal_at_one_tol": SIGNAL_TOL,
            "range": "physical_bounds(P_u, signal_direction): [P_u, 1] for direction +1",
        },
        "checks": {
            "flags_unchanged": {g: flag_check(rows) for g, rows in groups.items()},
            "max_ci_reproduce_diff": max(reproduce) if reproduce else None,
            "ci_reproduces_stored": bool(reproduce) and max(reproduce) <= REPRODUCE_TOL,
        },
        "inventory": {g: inventory(rows) for g, rows in groups.items()},
        "comparison": {
            setting: {g: compare(rows, setting) for g, rows in groups.items()}
            for setting in ("primary", "sensitivity")
        },
        "hardware_fits": hardware_table(pooled["hw_pooled"]),
        "rows": {g: [_strip(r) for r in rows] for g, rows in rows_by_group.items()},
        "quantiles_value_reported": {
            g: quantiles(r["value"] for r in rows if r["reported"]) for g, rows in groups.items()
        },
        "elapsed_s": None,
        "provenance": record_provenance(),
    }
    payload["elapsed_s"] = round(time.time() - started, 1)
    path = save_record(args.out, payload)
    print_summary(payload)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
