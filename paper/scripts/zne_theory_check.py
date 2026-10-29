"""
zne_theory_check.py

Test ``paper/notes/zne_theory.md`` on every record we have: the 463
simulated instances on each grid and both hardware jobs.  Reads records,
never edits them; no circuits, no QPU.

a. Hardware, both jobs: the delta-method SE of every scalar fit
   (:func:`dgssp.mitigation.zne.exp_fixed_se`) against the bootstrap SE taken
   from the stored 95 % interval, ``(hi - lo) / 3.92``, for resolved and for
   identified-but-not-resolved fits; and every record whose raw value equals
   the rate cap ``P_u + (P(lam_min) - P_u) exp(EXP_RATE_MAX lam_min)`` to
   ``1e-6`` relative.
b. Simulation, each grid: the same ratio on every resolved fit, recomputed
   from the per-scale counts; where it breaks (by n, observed z, rate,
   relative SE, number of scales); coverage of the exact ideal by
   ``+-1.96 SE_formula`` and by the bootstrap interval.
c. The grid rule: ``c = -ln kept`` per instance from the pooled distinct
   ``lam = 1`` counts of its records; the predicted SE of the odd, planned
   and :func:`~dgssp.mitigation.zne.plan_scales_spacing` grids; resolved
   fraction by predicted relative SE, by ``c * dlam`` and by the predicted
   gate; how well each predicts resolution.
d. Pointwise at M = 0 (simulation and the three hardware records): fraction
   of strings whose per-string fit sits at the rate cap, how often the
   simplex output is a single string, and the rank correlation between the
   cap fraction and D_TV after clipping.
e. Linearity (Liu-Cai): pointwise Richardson sums to one before
   renormalisation; pointwise exponential does not.

Plus the Monte Carlo of :func:`~dgssp.mitigation.zne.exp_fixed_monte_carlo`
on the designs of ``tests/test_zne_error.py`` (bias reported there, recorded
here), the optimal second-scale step for the hardware rates, and
``paper/figures/fig_zne_grid_theory.{pdf,png}``.

Output: ``paper/data/zne_theory.json`` (``save_record`` never overwrites).

Run::

    python paper/scripts/zne_theory_check.py
    python paper/scripts/zne_theory_check.py --figure-only paper/data/zne_theory.json
    python paper/scripts/zne_theory_check.py --out <scratch>/zne_theory.json --figures-dir <scratch>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import (  # noqa: E402
    DATA_DIR,
    FIGURES_DIR,
    INSTANCES_DIR,
    load_records,
    record_provenance,
    save_record,
)
from aggregate import exact_p_ideal, rederive_flags  # noqa: E402
from dgssp.metrics import proportion_sigma, solution_probability, total_variation  # noqa: E402
from dgssp.mitigation.zne import (  # noqa: E402
    EXP_RATE_MAX,
    OPTIMAL_STEP_LIMIT,
    RATE_CAP_RTOL,
    ZNESamplingConfig,
    at_rate_cap,
    exp_fixed_monte_carlo,
    exp_fixed_se,
    extrapolate_distribution,
    optimal_step,
    plan_scales,
    plan_scales_spacing,
    pointwise_fit_rates,
    pointwise_raw,
    predicted_se,
    signal_direction,
)
from sim_campaign import (  # noqa: E402
    PATHOLOGY_GRID,
    PLAN_CANDIDATES,
    PLAN_K,
    PLAN_POINTS,
    ZNE_SHOTS,
)

SCHEMA = "zne_theory/1"

#: Hardware jobs: folder and job id.  The ``_drift`` copies carry no fit.
HW_JOBS: dict[str, tuple[str, str]] = {
    "job1": ("hw_2026-10-25", "dbfeqr7gdj9s73bbgh6g"),
    "job2": ("hw_2026-10-28", "dbgkqq8te5oc738icnfg"),
}
SIM_GRIDS: tuple[str, ...] = ("odd", "odd7", "planned")
SIM_DIR = "sim_zne_fake_{grid}"

#: A 95 % percentile interval spans 2 x 1.96 standard errors.
Z95 = 1.96
CI_TO_SE = 2.0 * Z95

#: Monte Carlo designs (the same as tests/test_zne_error.py).
MC_P_UNIFORM = 0.125
MC_SIGNAL = 0.82
MC_RATES: tuple[float, ...] = (0.5, 1.0, 2.0)
MC_GRIDS: tuple[tuple[float, ...], ...] = ((1.0, 1.25, 1.5), (1.0, 1.5, 2.0), (1.0, 3.0, 5.0))
MC_SHOTS: tuple[int, ...] = (4096, 12_288)
MC_REPLICAS = 2000
MC_SEED = 0

#: Rates for the optimal-step table (the hardware rates are added).
STEP_RATES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0)

#: Hardware records whose rates draw the curves of Fig. (a): the resolved
#: fits, one per (instance, rate) -- n = 2 from job 1, both n = 3 from job 2.
FIGURE_RATE_RECORDS: tuple[tuple[str, str], ...] = (
    ("job1", "P-n2-M1-11"),
    ("job2", "P-n3-M1-17_r1"),
    ("job2", "P-n3-M1-17"),
)

#: The gentle grid of job 2 (nominal).
GENTLE_GRID: tuple[float, ...] = (1.0, 1.25, 1.5)
ODD_GRID: tuple[float, ...] = (1.0, 3.0, 5.0)

#: Bin edges.
Z_EDGES = (2.0, 3.0, 5.0, 10.0, math.inf)
RATE_EDGES = (0.0, 0.5, 1.0, 1.5, 2.5, math.inf)
REL_SE_EDGES = (0.0, 0.05, 0.1, 0.2, 0.3, math.inf)
PRED_REL_SE_EDGES = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, math.inf)
C_STEP_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, math.inf)

#: The claim of part c: spacing "near 1/c" and a "small" predicted SE.
NEAR_STEP = (0.5, 2.0)
SMALL_REL_SE = 0.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finite(values: Iterable[Any]) -> np.ndarray:
    arr = np.array([v for v in values if v is not None], dtype=float)
    return arr[np.isfinite(arr)]


def quantiles(values: Iterable[Any]) -> dict[str, Any]:
    """``n``, median, 5/95 % and the extremes of the finite values."""
    arr = _finite(values)
    if arr.size == 0:
        return {"n": 0}
    p5, p50, p95 = np.percentile(arr, [5, 50, 95])
    return {
        "n": int(arr.size),
        "median": float(p50),
        "p5": float(p5),
        "p95": float(p95),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _bin_label(lo: float, hi: float) -> str:
    return f"[{lo:g}, {hi:g})" if math.isfinite(hi) else f">= {lo:g}"


def binned(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    edges: Sequence[float],
    summary: Callable[[Sequence[Mapping[str, Any]]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply ``summary`` to the rows in each ``[lo, hi)`` bin of ``row[key]``."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        members = [
            r for r in rows if r.get(key) is not None and lo <= r[key] < hi
            or (r.get(key) == math.inf and hi == math.inf)
        ]
        out.append({"bin": _bin_label(lo, hi), "lo": lo, "hi": hi, **summary(members)})
    return out


def counts_by_scale(record: Mapping[str, Any]) -> dict[float, dict[str, int]]:
    """A record's counts keyed by realised scale (as floats)."""
    return {float(k): dict(v) for k, v in record["counts"].items()}


def observed_series(record: Mapping[str, Any]) -> tuple[list[float], list[float], list[int]]:
    """Scales, ``P(solution)`` and shots from the raw counts."""
    counts = counts_by_scale(record)
    sols = record["metrics"]["solution_bitstrings"]
    scales = sorted(counts)
    return (
        scales,
        [float(solution_probability(counts[s], sols)) for s in scales],
        [int(sum(counts[s].values())) for s in scales],
    )


def record_key(record: Mapping[str, Any]) -> str:
    """Hardware record key (sweep copies share an instance), else the instance name."""
    hardware = (record.get("metrics") or {}).get("hardware") or {}
    return str(hardware.get("record_key") or record["instance"]["name"])


def status_of(flags: Mapping[str, Any]) -> str:
    """``resolved`` (reported), ``identified_not_resolved`` or ``unidentified``."""
    if flags.get("reported_source") == "extrapolation":
        return "resolved"
    if flags.get("identifiable"):
        return "identified_not_resolved"
    return "unidentified"


def fit_row(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """
    One record's scalar fit, recomputed from its counts, beside the stored block.

    Returns ``None`` for a record without a fixed-asymptote exponential fit.
    """
    block = record.get("zne_fit") or {}
    if block.get("method") != "exponential" or not block.get("observed"):
        return None
    scales, observed, shots = observed_series(record)
    p_u = float(block["asymptote"])
    p_ideal = exact_p_ideal(dict(record))
    flags = rederive_flags(dict(block), p_ideal)
    fit = exp_fixed_se(scales, observed, shots, p_u)
    lo, hi = block["ci"]
    se_boot = (float(hi) - float(lo)) / CI_TO_SE
    direction = signal_direction(p_ideal, p_u)
    z = [
        (direction or 1) * (v - p_u) / proportion_sigma(v, n)
        for v, n in zip(observed, shots, strict=True)
    ]
    excess = fit["value"] - p_u
    cap = fit["cap_value"]
    return {
        "name": record_key(record),
        "n": int(record["n_items"]),
        "M": int(record["metrics"]["num_solutions"]),
        "scales": scales,
        "n_scales": len(scales),
        "observed": observed,
        "shots": shots,
        "p_uniform": p_u,
        "p_ideal": p_ideal,
        "status": status_of(flags),
        "value": fit["value"],
        "rate": fit["rate"],
        "se_formula": None if not math.isfinite(fit["se"]) else fit["se"],
        "se_known_f": fit["se_known_f"],
        "learning_cost": fit["learning_cost"] if math.isfinite(fit["learning_cost"]) else None,
        "se_bootstrap_ci": se_boot,
        "se_bootstrap_std": block.get("bootstrap_se"),
        "bootstrap_B": block.get("B"),
        "ci": [lo, hi],
        "ratio": fit["se"] / se_boot if se_boot > 0 and math.isfinite(fit["se"]) else None,
        "relative_se": (
            fit["se"] / abs(excess) if excess != 0 and math.isfinite(fit["se"]) else None
        ),
        "z": z,
        "z_min": min(z),
        "at_cap": fit["at_cap"],
        "rate_at_bound_stored": block.get("rate_at_bound"),
        "cap_value": cap,
        "cap_rel_diff": abs(fit["value"] - cap) / abs(cap) if cap != 0 else None,
        # The record as stored, for the consistency checks.
        "stored_value": block["value"],
        "stored_value_se": block.get("value_se"),
        "max_abs_observed_diff": float(
            np.max(np.abs(np.asarray(observed) - np.asarray(block["observed"])))
        ),
        "covered_formula": (
            None
            if p_ideal is None or not math.isfinite(fit["se"])
            else bool(abs(fit["value"] - p_ideal) <= Z95 * fit["se"])
        ),
        "covered_bootstrap": None if p_ideal is None else bool(lo <= p_ideal <= hi),
        "standardised_error": (
            (fit["value"] - p_ideal) / fit["se"]
            if p_ideal is not None and math.isfinite(fit["se"]) and fit["se"] > 0
            else None
        ),
    }


def ratio_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """SE ratio quantiles plus coverage, for a set of fit rows."""
    out = quantiles(r["ratio"] for r in rows)
    out["n"] = len(rows)
    cov_f = [r["covered_formula"] for r in rows if r["covered_formula"] is not None]
    cov_b = [r["covered_bootstrap"] for r in rows if r["covered_bootstrap"] is not None]
    out["coverage_formula"] = float(np.mean(cov_f)) if cov_f else None
    out["coverage_bootstrap"] = float(np.mean(cov_b)) if cov_b else None
    out["median_standardised_error"] = quantiles(
        r["standardised_error"] for r in rows
    ).get("median")
    out["median_bias"] = quantiles(
        r["value"] - r["p_ideal"] for r in rows if r["p_ideal"] is not None
    ).get("median")
    return out


# ---------------------------------------------------------------------------
# a) Hardware
# ---------------------------------------------------------------------------


def hardware_block() -> dict[str, Any]:
    """Per-fit SE comparison and cap identities on both jobs."""
    fits: list[dict[str, Any]] = []
    m0: list[dict[str, Any]] = []
    for job, (folder, job_id) in HW_JOBS.items():
        for record in load_records(DATA_DIR / folder / "*.json"):
            if record.get("job_id") != job_id:
                raise ValueError(f"{record['_source']}: job {record.get('job_id')} != {job_id}")
            row = fit_row(record)
            if row is None:
                continue
            row["job"] = job
            row["job_id"] = job_id
            fits.append(row)
            if row["M"] == 0:
                m0.append(record)
    by_status = {
        status: [
            {
                key: r[key]
                for key in (
                    "job", "name", "rate", "value", "se_formula", "se_bootstrap_ci",
                    "se_bootstrap_std", "ratio", "learning_cost", "z", "scales", "shots",
                )
            }
            for r in fits
            if r["status"] == status
        ]
        for status in ("resolved", "identified_not_resolved")
    }
    cap_rows = [
        {
            "job": r["job"],
            "name": r["name"],
            "value": r["value"],
            "cap_value": r["cap_value"],
            "rel_diff": r["cap_rel_diff"],
            "rate": r["rate"],
            "rate_at_bound_stored": r["rate_at_bound_stored"],
            "status": r["status"],
        }
        for r in fits
        if r["cap_rel_diff"] is not None and r["cap_rel_diff"] <= RATE_CAP_RTOL
    ]
    large = [r for r in fits if abs(r["value"]) > 1.0]
    return {
        "fits": fits,
        "by_status": by_status,
        "n_fits": len(fits),
        "n_by_status": dict(Counter(r["status"] for r in fits)),
        "cap_identities": cap_rows,
        "cap_rtol": RATE_CAP_RTOL,
        "large_values": [{"job": r["job"], "name": r["name"], "value": r["value"]} for r in large],
        "every_large_value_is_the_cap": all(
            r["cap_rel_diff"] is not None and r["cap_rel_diff"] <= RATE_CAP_RTOL for r in large
        ),
        "cap_missed_by_stored_flag": [
            r["name"] for r in fits if r["at_cap"] and not r["rate_at_bound_stored"]
        ],
        "recomputed_equals_stored": {
            "max_abs_observed_diff": max(r["max_abs_observed_diff"] for r in fits),
            "max_rel_value_diff": max(
                abs(r["value"] - r["stored_value"]) / max(abs(r["stored_value"]), 1e-300)
                for r in fits
            ),
            "max_rel_se_diff": max(
                (
                    abs(r["se_formula"] - r["stored_value_se"]) / r["stored_value_se"]
                    for r in fits
                    if r["se_formula"] is not None and r["stored_value_se"]
                ),
                default=None,
            ),
        },
        "_m0_records": m0,
    }


# ---------------------------------------------------------------------------
# b) Simulation: SE ratio
# ---------------------------------------------------------------------------


def load_sim(grid: str) -> list[dict[str, Any]]:
    """Every record of one simulated grid."""
    return load_records(DATA_DIR / SIM_DIR.format(grid=grid) / "*.json")


def simulation_se_block(fit_rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """SE_formula / SE_bootstrap on resolved simulated fits, and where it breaks."""
    out: dict[str, Any] = {}
    pooled: list[dict[str, Any]] = []
    for grid, rows in fit_rows.items():
        resolved = [r for r in rows if r["status"] == "resolved"]
        pooled += resolved
        out[grid] = {
            "n_records": len(rows),
            "n_resolved": len(resolved),
            "overall": ratio_summary(resolved),
            "ratio_vs_bootstrap_std": quantiles(
                r["se_formula"] / r["se_bootstrap_std"]
                for r in resolved
                if r["se_formula"] and r["se_bootstrap_std"]
            ),
            "by_n": {
                str(n): ratio_summary([r for r in resolved if r["n"] == n])
                for n in sorted({r["n"] for r in resolved})
            },
            "identified_not_resolved": quantiles(
                r["ratio"] for r in rows if r["status"] == "identified_not_resolved"
            ),
            "recomputed_equals_stored": {
                "max_abs_observed_diff": max(r["max_abs_observed_diff"] for r in rows),
                "max_rel_se_diff": max(
                    (
                        abs(r["se_formula"] - r["stored_value_se"]) / r["stored_value_se"]
                        for r in rows
                        if r["se_formula"] is not None and r["stored_value_se"]
                    ),
                    default=None,
                ),
            },
        }
    out["pooled"] = {
        "overall": ratio_summary(pooled),
        "by_z_min": binned(pooled, "z_min", Z_EDGES, ratio_summary),
        "by_rate": binned(pooled, "rate", RATE_EDGES, ratio_summary),
        "by_relative_se": binned(pooled, "relative_se", REL_SE_EDGES, ratio_summary),
        "by_n_scales": {
            str(k): ratio_summary([r for r in pooled if r["n_scales"] == k])
            for k in sorted({r["n_scales"] for r in pooled})
        },
        "by_n": {
            str(n): ratio_summary([r for r in pooled if r["n"] == n])
            for n in sorted({r["n"] for r in pooled})
        },
        "worst": sorted(
            (
                {
                    "grid": r["grid"],
                    "name": r["name"],
                    "ratio": r["ratio"],
                    "z": r["z"],
                    "rate": r["rate"],
                    "relative_se": r["relative_se"],
                    "se_formula": r["se_formula"],
                    "se_bootstrap_ci": r["se_bootstrap_ci"],
                }
                for r in pooled
                if r["ratio"] is not None
            ),
            key=lambda d: d["ratio"],
        )[:10],
    }
    return out


# ---------------------------------------------------------------------------
# c) The grid rule
# ---------------------------------------------------------------------------


def _resolved_fraction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    vals = [r["resolved"] for r in rows]
    return {"n": len(vals), "resolved_fraction": float(np.mean(vals)) if vals else None}


def _classifier(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    """How well a boolean prediction ``row[key]`` matches ``row['resolved']``."""
    pos = [r for r in rows if r[key]]
    neg = [r for r in rows if not r[key]]
    hits = sum(bool(r[key]) == bool(r["resolved"]) for r in rows)
    return {
        "n": len(rows),
        "accuracy": hits / len(rows) if rows else None,
        "n_predicted_resolved": len(pos),
        "resolved_given_predicted": float(np.mean([r["resolved"] for r in pos])) if pos else None,
        "resolved_given_not_predicted": (
            float(np.mean([r["resolved"] for r in neg])) if neg else None
        ),
    }


def grid_rule_block(
    sim: Mapping[str, Mapping[str, dict[str, Any]]],
    fit_rows: Mapping[str, Mapping[str, dict[str, Any]]],
    candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Predicted SE of each grid against what each grid actually resolved."""
    shots = ZNE_SHOTS
    rows: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    planned_reproduced = 0
    planned_checked = 0
    for name in sorted(sim["odd"]):
        base = sim["odd"][name]
        p_ideal = exact_p_ideal(base)
        p_u = float(base["metrics"]["p_uniform"])
        if p_ideal is None or signal_direction(p_ideal, p_u) == 0:
            continue
        signal = p_ideal - p_u
        sols = base["metrics"]["solution_bitstrings"]
        # Pool the distinct lambda = 1 count sets of every grid's record.
        distinct: list[dict[str, int]] = []
        for grid in SIM_GRIDS:
            rec = sim[grid].get(name)
            if rec is None:
                continue
            c1 = counts_by_scale(rec)[1.0]
            if c1 not in distinct:
                distinct.append(c1)
        hits = sum(sum(c.get(s, 0) for s in sols) for c in distinct)
        total = sum(sum(c.values()) for c in distinct)
        kept = ((hits / total) - p_u) / signal
        survival = min(kept, 1.0) if kept > 0 else 0.0
        rate = -math.log(survival) if survival > 0 else math.inf

        cand = candidates[name]
        planned_nominal = plan_scales(
            float(cand["survival"]), float(cand["p_ideal"]) - float(cand["p_uniform"]),
            shots, PLAN_CANDIDATES, PLAN_K, p_uniform=float(cand["p_uniform"]),
            n_points=PLAN_POINTS,
        )
        stored_nominal = sim["planned"][name]["zne_fit"]["nominal_scales"]
        grid_source = sim["planned"][name]["metrics"]["campaign"]["grid_source"]
        # A one-scale plan falls back to the pathology grid ("planned_floor").
        expected = [1.0] if grid_source == "planned_floor" else list(stored_nominal)
        planned_checked += 1
        planned_reproduced += planned_nominal == expected
        spacing = plan_scales_spacing(
            float(cand["survival"]), float(cand["p_ideal"]) - float(cand["p_uniform"]),
            shots, PLAN_CANDIDATES, PLAN_K, p_uniform=float(cand["p_uniform"]),
            n_points=PLAN_POINTS,
        )
        oracle = (
            plan_scales_spacing(
                survival, signal, shots, PLAN_CANDIDATES, PLAN_K, p_uniform=p_u,
                n_points=PLAN_POINTS,
            )
            if survival > 0
            else [1.0]
        )

        def pred(
            scales: Sequence[float],
            survival: float = survival,
            signal: float = signal,
            p_u: float = p_u,
        ) -> float:
            if len(scales) < 2:
                return math.inf
            return predicted_se(scales, survival, signal, p_u, shots)

        inst = {
            "name": name,
            "n": int(base["n_items"]),
            "kept": kept,
            "rate": rate,
            "lambda1_count_sets": len(distinct),
            "planned_nominal": list(stored_nominal),
            "planned_source": grid_source,
            "spacing_grid": spacing,
            "spacing_grid_oracle": oracle,
            "predicted_se_spacing": pred(spacing),
            "predicted_se_spacing_oracle": pred(oracle),
        }
        for grid in SIM_GRIDS:
            row = fit_rows[grid].get(name)
            if row is None:
                continue
            scales = row["scales"]
            se = pred(scales)
            z_pred = (
                min(
                    abs(signal) * survival**lam
                    / proportion_sigma(p_u + signal * survival**lam, shots)
                    for lam in scales
                )
                if survival > 0
                else 0.0
            )
            step = scales[1] - scales[0]
            entry = {
                "grid": grid,
                "name": name,
                "n": inst["n"],
                "resolved": row["status"] == "resolved",
                "rate": rate,
                "predicted_se": se,
                "predicted_rel_se": se / abs(signal),
                "c_step_first": rate * step,
                "c_span": rate * (scales[-1] - scales[0]),
                "predicted_z_min": z_pred,
                "source": grid_source if grid == "planned" else grid,
            }
            entry["near_step"] = NEAR_STEP[0] <= entry["c_step_first"] <= NEAR_STEP[1]
            entry["small_se"] = entry["predicted_rel_se"] <= SMALL_REL_SE
            entry["near_and_small"] = entry["near_step"] and entry["small_se"]
            entry["gate_expected"] = z_pred >= 2.0
            rows.append(entry)
            inst[f"predicted_se_{grid}"] = se
            inst[f"resolved_{grid}"] = entry["resolved"]
        instances.append(inst)

    per_grid = {}
    for grid in (*SIM_GRIDS, "planned_floor", "pooled"):
        if grid == "pooled":
            sub = rows
        elif grid == "planned_floor":
            sub = [r for r in rows if r["source"] == "planned_floor"]
        else:
            sub = [r for r in rows if r["grid"] == grid]
        per_grid[grid] = {
            **_resolved_fraction(sub),
            "by_predicted_rel_se": binned(
                sub, "predicted_rel_se", PRED_REL_SE_EDGES, _resolved_fraction
            ),
            "by_c_step_first": binned(sub, "c_step_first", C_STEP_EDGES, _resolved_fraction),
            "by_c_span": binned(sub, "c_span", C_STEP_EDGES, _resolved_fraction),
            "predictors": {
                key: _classifier(sub, key)
                for key in ("small_se", "near_step", "near_and_small", "gate_expected")
            },
        }

    both = [i for i in instances if math.isfinite(i["predicted_se_planned"])]
    ratio_sp = [
        i["predicted_se_spacing"] / i["predicted_se_planned"]
        for i in both
        if math.isfinite(i["predicted_se_spacing"])
    ]
    ratio_or = [
        i["predicted_se_spacing_oracle"] / i["predicted_se_planned"]
        for i in both
        if math.isfinite(i["predicted_se_spacing_oracle"])
    ]
    return {
        "shots": shots,
        "plan_candidates": list(PLAN_CANDIDATES),
        "plan_k": PLAN_K,
        "plan_points": PLAN_POINTS,
        "near_step": list(NEAR_STEP),
        "small_rel_se": SMALL_REL_SE,
        "n_instances_with_signal": len(instances),
        "planned_grid_reproduced": {"checked": planned_checked, "equal": planned_reproduced},
        "rate_from_lambda1": quantiles(i["rate"] for i in instances),
        "n_kept_nonpositive": sum(1 for i in instances if i["kept"] <= 0),
        "per_grid": per_grid,
        "spacing_vs_planned": {
            "differs_from_planned": sum(
                1 for i in instances if i["spacing_grid"] != i["planned_nominal"]
            ),
            "se_ratio_spacing_over_planned": quantiles(ratio_sp),
            "se_ratio_oracle_over_planned": quantiles(ratio_or),
            "grids_chosen": dict(Counter(str(i["spacing_grid"]) for i in instances)),
        },
        "planned_source_counts": dict(Counter(i["planned_source"] for i in instances)),
        "typical_planned_grid": Counter(
            str(i["planned_nominal"]) for i in instances if i["planned_source"] == "planned"
        ).most_common(3),
        "instances": instances,
    }


# ---------------------------------------------------------------------------
# d) Pointwise at M = 0;  e) linearity
# ---------------------------------------------------------------------------

POINTWISE_EXP = ZNESamplingConfig(method="exponential", asymptote="p_uniform")


def pointwise_m0_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Cap fraction and the renormalised outputs of one M = 0 record."""
    counts = counts_by_scale(record)
    n_bits = int(record["n_items"])
    rates = pointwise_fit_rates(counts, POINTWISE_EXP)
    capped = at_rate_cap(np.array(list(rates.values())), POINTWISE_EXP.exp_rate_max)
    reference = record["metrics"]["reference_distribution"]
    simplex = extrapolate_distribution(
        counts, ZNESamplingConfig(method="exponential", asymptote="p_uniform", renorm="simplex")
    )
    clip = extrapolate_distribution(
        counts, ZNESamplingConfig(method="exponential", asymptote="p_uniform", renorm="clip")
    )
    lowest = counts[min(counts)]
    shots = sum(lowest.values())
    tv_simplex = total_variation(simplex, reference)
    return {
        "name": record_key(record),
        "n": n_bits,
        "scales": sorted(counts),
        "n_strings_fitted": len(rates),
        "cap_fraction": float(np.mean(capped)),
        "n_capped": int(capped.sum()),
        "simplex_support": int(sum(v > 0 for v in simplex.values())),
        "simplex_single_string": sum(v > 0 for v in simplex.values()) == 1,
        "tv_simplex": tv_simplex,
        "tv_simplex_is_max": abs(tv_simplex - (1.0 - 2.0**-n_bits)) <= 1e-9,
        "tv_clip": total_variation(clip, reference),
        "tv_raw": total_variation({k: v / shots for k, v in lowest.items()}, reference),
        "stored_tv_simplex": (record.get("pathology") or {}).get("tv_pointwise_simplex"),
        "stored_tv_clip": (record.get("pathology") or {}).get("tv_pointwise_clip"),
    }


def _spearman(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    x = [r["cap_fraction"] for r in rows]
    y = [r["tv_clip"] for r in rows]
    if len(rows) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": len(rows), "rho": None, "p": None}
    res = spearmanr(x, y)
    return {"n": len(rows), "rho": float(res.statistic), "p": float(res.pvalue)}


def pointwise_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate of :func:`pointwise_m0_row` rows."""
    return {
        "n": len(rows),
        "cap_fraction": quantiles(r["cap_fraction"] for r in rows),
        "cap_fraction_mean": float(np.mean([r["cap_fraction"] for r in rows])) if rows else None,
        "fraction_with_any_cap": float(np.mean([r["n_capped"] > 0 for r in rows])) if rows else None,
        "fraction_simplex_single_string": (
            float(np.mean([r["simplex_single_string"] for r in rows])) if rows else None
        ),
        "fraction_tv_simplex_is_max": (
            float(np.mean([r["tv_simplex_is_max"] for r in rows])) if rows else None
        ),
        "tv_clip": quantiles(r["tv_clip"] for r in rows),
        "tv_clip_any_cap": quantiles(r["tv_clip"] for r in rows if r["n_capped"] > 0),
        "tv_clip_no_cap": quantiles(r["tv_clip"] for r in rows if r["n_capped"] == 0),
        "tv_raw": quantiles(r["tv_raw"] for r in rows),
        "spearman_cap_fraction_vs_tv_clip": _spearman(rows),
        "max_abs_diff_vs_stored_tv_simplex": max(
            (
                abs(r["tv_simplex"] - r["stored_tv_simplex"])
                for r in rows
                if r["stored_tv_simplex"] is not None
            ),
            default=None,
        ),
    }


def pointwise_block(
    sim: Mapping[str, Mapping[str, dict[str, Any]]], hw_m0: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Part d on every grid's M = 0 records and the hardware ones."""
    out: dict[str, Any] = {}
    for grid in SIM_GRIDS:
        rows = [pointwise_m0_row(r) for r in sim[grid].values()
                if r["metrics"]["num_solutions"] == 0]
        out[grid] = {
            **pointwise_summary(rows),
            "by_n": {
                str(n): pointwise_summary([r for r in rows if r["n"] == n])
                for n in sorted({r["n"] for r in rows})
            },
        }
    hw_rows = [pointwise_m0_row(r) for r in hw_m0]
    out["hardware"] = {**pointwise_summary(hw_rows), "records": hw_rows}
    return out


def linearity_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Sum of the unrenormalised pointwise estimates, Richardson and exponential."""
    counts = counts_by_scale(record)
    rich = pointwise_raw(counts, ZNESamplingConfig(method="richardson"))
    expo = pointwise_raw(counts, POINTWISE_EXP)
    return {
        "name": record_key(record),
        "richardson_sum_minus_one": float(sum(rich.values()) - 1.0),
        "exponential_sum": float(sum(expo.values())),
    }


def linearity_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Part e for one set of records."""
    dev = np.abs([r["richardson_sum_minus_one"] for r in rows])
    sums = np.array([r["exponential_sum"] for r in rows])
    return {
        "n": len(rows),
        "richardson_max_abs_sum_minus_one": float(dev.max()),
        "richardson_all_within_1e-12": bool(np.all(dev <= 1e-12)),
        "exponential_sum": quantiles(sums),
        "exponential_fraction_off_by_more_than_0.01": float(np.mean(np.abs(sums - 1) > 0.01)),
        "exponential_fraction_off_by_more_than_1": float(np.mean(np.abs(sums - 1) > 1.0)),
    }


def linearity_block(
    sim: Mapping[str, Mapping[str, dict[str, Any]]], hw_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Part e on every simulated record and every hardware record with a fit."""
    out: dict[str, Any] = {}
    for grid in SIM_GRIDS:
        out[grid] = linearity_summary([linearity_row(r) for r in sim[grid].values()])
    out["hardware"] = linearity_summary([linearity_row(r) for r in hw_records])
    return out


# ---------------------------------------------------------------------------
# Monte Carlo and optimal step
# ---------------------------------------------------------------------------


def monte_carlo_block() -> list[dict[str, Any]]:
    """The designs of tests/test_zne_error.py."""
    return [
        exp_fixed_monte_carlo(
            grid, math.exp(-rate), MC_SIGNAL, MC_P_UNIFORM, shots,
            n_replicas=MC_REPLICAS, seed=MC_SEED,
        )
        for rate in MC_RATES
        for grid in MC_GRIDS
        for shots in MC_SHOTS
    ]


def step_block(hw_rates: Mapping[str, float]) -> dict[str, Any]:
    """:func:`optimal_step` for a rate table and the hardware rates."""
    return {
        "limit": OPTIMAL_STEP_LIMIT,
        "table": [{"rate": c, **optimal_step(c)} for c in STEP_RATES],
        "hardware": {key: {"rate": c, **optimal_step(c)} for key, c in hw_rates.items()},
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(payload: Mapping[str, Any], directory: Path = FIGURES_DIR) -> list[Path]:
    """``fig_zne_grid_theory``: (a) relative SE vs step, (b) SE formula vs bootstrap."""
    from dgssp.mitigation.zne import two_point_relative_error
    from plotting import FULL_WIDTH_IN, OKABE_ITO, figure, save_figure, style_for

    fig, (ax_a, ax_b) = figure(width=FULL_WIDTH_IN, aspect=0.42, ncols=2)

    # (a) two-point relative error against the step, one curve per hardware rate.
    steps = np.geomspace(0.05, 6.0, 400)
    colours = (OKABE_ITO["blue"], OKABE_ITO["bluish_green"], OKABE_ITO["reddish_purple"])
    linestyles = ("-", "--", "-.")
    hw = payload["optimal_step"]["hardware"]
    for (key, info), colour, ls in zip(hw.items(), colours, linestyles, strict=True):
        rate = info["rate"]
        curve = [two_point_relative_error(float(d), rate) for d in steps]
        ax_a.plot(steps, curve, color=colour, linestyle=ls, label=f"$c = {rate:.2f}$ ({key})")
        ax_a.plot(info["step"], info["relative_error"], "o", color=colour, ms=4, zorder=5)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(steps[0], steps[-1])
    ax_a.set_ylim(1.5, 1e4)
    for i, mark in enumerate(payload["figure"]["grid_marks"]):
        y = 4000.0 / (2.2**i)
        offsets = mark["offsets"]
        ax_a.plot(offsets, [y] * len(offsets), "|", color=OKABE_ITO["grey"], ms=7, mew=1.2)
        # Right of the last tick, unless that runs into the steep end of the curves.
        left = offsets[-1] > 1.2
        ax_a.text(offsets[0] / 1.15 if left else offsets[-1] * 1.12, y, mark["label"],
                  va="center", ha="right" if left else "left", fontsize=6.5,
                  color=OKABE_ITO["grey"])
    ax_a.set_xlabel(r"step $\Delta\lambda = \lambda_2 - 1$")
    ax_a.set_ylabel(r"relative SE of $P(0)-P_u$  [$\sigma_1/D_1$]")
    ax_a.set_title("(a)", loc="left")
    ax_a.legend(loc="center left", fontsize=6.5)

    # (b) SE formula against the bootstrap SE.
    grid_style = {
        "odd": ("o", OKABE_ITO["grey"]),
        "odd7": ("s", OKABE_ITO["bluish_green"]),
        "planned": ("^", OKABE_ITO["blue"]),
    }
    points = payload["figure"]["se_points"]
    lo, hi = math.inf, 0.0
    for grid, (marker, colour) in grid_style.items():
        pts = np.array(points["simulation"].get(grid, []), dtype=float)
        if pts.size == 0:
            continue
        ax_b.plot(pts[:, 0], pts[:, 1], marker, mfc="none", mec=colour, ms=3.2, mew=0.7,
                  linestyle="none", label=f"sim, {grid}")
        lo, hi = min(lo, pts.min()), max(hi, pts.max())
    hw_style = style_for("hardware")
    for status, filled in (("resolved", True), ("identified_not_resolved", False)):
        pts = np.array(points["hardware"].get(status, []), dtype=float)
        if pts.size == 0:
            continue
        ax_b.plot(pts[:, 0], pts[:, 1], hw_style.marker, color=hw_style.color, ms=6,
                  mfc=hw_style.color if filled else "none", linestyle="none", zorder=6,
                  label="hardware" + ("" if filled else ", not resolved"))
        lo, hi = min(lo, pts.min()), max(hi, pts.max())
    span = [lo / 1.5, hi * 1.5]
    ax_b.plot(span, span, color=OKABE_ITO["black"], lw=0.7, zorder=1)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(span)
    ax_b.set_ylim(span)
    ax_b.set_xlabel(r"SE bootstrap, $(hi-lo)/3.92$")
    ax_b.set_ylabel("SE delta method")
    ax_b.set_title("(b)", loc="left")
    ax_b.legend(loc="upper left", fontsize=6.5)
    return save_figure(fig, "fig_zne_grid_theory", directory=directory)


def figure_data(
    hw: Mapping[str, Any],
    sim_fits: Mapping[str, Sequence[Mapping[str, Any]]],
    typical_planned: Sequence[Any],
) -> dict[str, Any]:
    """What the figure draws, kept in the JSON so the figure is reproducible from it."""
    planned = json.loads(typical_planned[0][0])

    def mark(label: str, grid: Sequence[float]) -> dict[str, Any]:
        return {"label": label, "grid": list(grid), "offsets": [s - 1.0 for s in grid[1:]]}

    return {
        "grid_marks": [
            mark("odd", ODD_GRID),
            mark("gentle", GENTLE_GRID),
            mark("planned (mode)", planned),
            mark("fallback", PATHOLOGY_GRID),
        ],
        "se_points": {
            "simulation": {
                grid: [
                    [r["se_bootstrap_ci"], r["se_formula"]]
                    for r in rows
                    if r["status"] == "resolved" and r["se_formula"] and r["se_bootstrap_ci"] > 0
                ]
                for grid, rows in sim_fits.items()
            },
            "hardware": {
                status: [
                    [r["se_bootstrap_ci"], r["se_formula"]]
                    for r in hw["fits"]
                    if r["status"] == status and r["se_formula"] and r["se_bootstrap_ci"] > 0
                ]
                for status in ("resolved", "identified_not_resolved")
            },
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_summary(payload: Mapping[str, Any]) -> None:
    """The acceptance numbers, on stdout."""
    hw = payload["hardware"]
    print("Hardware fits:", hw["n_by_status"])
    for status, rows in hw["by_status"].items():
        for r in rows:
            print(f"  {status:<24} {r['job']} {r['name']:<16} c={r['rate']:.3f} "
                  f"SE formula={r['se_formula'] or float('nan'):.4f} "
                  f"bootstrap={r['se_bootstrap_ci']:.4f} cost={r['learning_cost'] or 0:.2f}")
    for r in hw["cap_identities"]:
        print(f"  cap: {r['job']} {r['name']:<16} {r['value']:.6g} = {r['cap_value']:.6g} "
              f"(rel {r['rel_diff']:.1e}, stored rate_at_bound={r['rate_at_bound_stored']})")
    print("  every |value| > 1 is the cap:", hw["every_large_value_is_the_cap"])
    for grid in SIM_GRIDS:
        o = payload["simulation_se"][grid]["overall"]
        print(f"Sim {grid:<8} resolved={o['n']:<4} median ratio={o.get('median', float('nan')):.3f} "
              f"[{o.get('p5', float('nan')):.3f}, {o.get('p95', float('nan')):.3f}]  "
              f"coverage formula={o['coverage_formula']:.3f} bootstrap={o['coverage_bootstrap']:.3f}")
    for b in payload["simulation_se"]["pooled"]["by_relative_se"]:
        print(f"  rel SE {b['bin']:<12} n={b['n']:<4} median ratio={b.get('median', float('nan')):.3f}")
    rule = payload["grid_rule"]["per_grid"]["pooled"]["predictors"]
    for key, v in rule.items():
        print(f"Grid rule predictor {key:<15} accuracy={v['accuracy']:.3f} "
              f"P(res|pred)={v['resolved_given_predicted']} "
              f"P(res|not)={v['resolved_given_not_predicted']}")
    for grid in (*SIM_GRIDS, "hardware"):
        p = payload["pointwise_m0"][grid]
        print(f"M=0 {grid:<8} n={p['n']:<4} cap fraction median="
              f"{p['cap_fraction'].get('median')} simplex single={p['fraction_simplex_single_string']} "
              f"rho={p['spearman_cap_fraction_vs_tv_clip']['rho']}")
    for grid, v in payload["linearity"].items():
        print(f"Linearity {grid:<8} Richardson max|sum-1|={v['richardson_max_abs_sum_minus_one']:.1e} "
              f"exp sum median={v['exponential_sum']['median']:.3g}")


def main(argv: list[str] | None = None) -> int:
    """Run parts a-e, write the JSON, draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--figure-only", type=Path, default=None,
                        help="redraw the figure from an existing zne_theory JSON")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "zne_theory.json",
                        help="output JSON (never overwritten)")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    if args.figure_only is not None:
        stored = json.loads(args.figure_only.read_text(encoding="utf-8"))
        for path in make_figure(stored, args.figures_dir):
            print("wrote", path)
        return 0

    started = time.time()
    hw = hardware_block()
    hw_m0 = hw.pop("_m0_records")
    hw_records = [
        r for job, (folder, _) in HW_JOBS.items()
        for r in load_records(DATA_DIR / folder / "*.json") if r.get("zne_fit")
    ]

    sim = {grid: {r["instance"]["name"]: r for r in load_sim(grid)} for grid in SIM_GRIDS}
    sim_fit_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for grid, records in sim.items():
        sim_fit_rows[grid] = {}
        for name, record in records.items():
            row = fit_row(record)
            if row is not None:
                row["grid"] = grid
                sim_fit_rows[grid][name] = row
    fit_lists = {grid: list(rows.values()) for grid, rows in sim_fit_rows.items()}

    subset = json.loads((INSTANCES_DIR / "hardware_subset.json").read_text(encoding="utf-8"))
    candidates = {row["name"]: row for row in subset["candidates"]}

    hw_rates = {}
    for job, name in FIGURE_RATE_RECORDS:
        (row,) = [r for r in hw["fits"] if r["job"] == job and r["name"] == name]
        hw_rates[f"{name}, {job}"] = row["rate"]

    rule = grid_rule_block(sim, sim_fit_rows, candidates)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "parameters": {
            "hardware_jobs": {k: {"folder": v[0], "job_id": v[1]} for k, v in HW_JOBS.items()},
            "sim_grids": list(SIM_GRIDS),
            "ci_to_se": CI_TO_SE,
            "cap_rtol": RATE_CAP_RTOL,
            "exp_rate_max": EXP_RATE_MAX,
            "monte_carlo": {
                "p_uniform": MC_P_UNIFORM, "signal": MC_SIGNAL, "rates": list(MC_RATES),
                "grids": [list(g) for g in MC_GRIDS], "shots": list(MC_SHOTS),
                "replicas": MC_REPLICAS, "seed": MC_SEED,
            },
            "note": "SE_bootstrap is (hi - lo)/3.92 of the stored 95 % percentile interval "
                    "(B = 500, seed 0, as stored in every record).",
        },
        "hardware": hw,
        "simulation_se": simulation_se_block(fit_lists),
        "grid_rule": rule,
        "pointwise_m0": pointwise_block(sim, hw_m0),
        "linearity": linearity_block(sim, hw_records),
        "monte_carlo": monte_carlo_block(),
        "optimal_step": step_block(hw_rates),
        "figure": figure_data(hw, fit_lists, rule["typical_planned_grid"]),
        "elapsed_s": None,
        "provenance": record_provenance(),
    }
    payload["elapsed_s"] = round(time.time() - started, 1)
    path = save_record(args.out, payload)
    print_summary(payload)
    print("wrote", path)
    for fig_path in make_figure(payload, args.figures_dir):
        print("wrote", fig_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
