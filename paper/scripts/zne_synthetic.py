"""
zne_synthetic.py

The limits of ZNE before any hardware: C3's synthetic table.

Two studies, no circuits, seconds to run:

1. **Scalar estimators under global depolarising noise.**  Counts are drawn
   binomially from ``P(lam) = S**lam * P_ideal + (1 - S**lam) * P_uniform`` at
   ``S`` in {0.75, 0.40, 0.17} (the hardware subset's survival range), with
   ``P_ideal``, ``P_uniform`` and the two-qubit count taken from the subset
   instance whose survival sets that ``S``.  Scales are *realised* (the exact
   rational ratios partial folding produces on that gate count), on two grids:
   odd integers [1, 3, 5] and the grid :func:`dgssp.mitigation.zne.
   plan_scales` pre-registers.  For each method x S x grid the file records
   the Monte-Carlo bias and its standard error, the analytic bias (the
   estimator applied to the exact curve), bootstrap-interval coverage of the
   truth, median interval width, the fractions of fits flagged resolved and
   identifiable, and -- on the same counts, through Mitiq's own factories --
   the Mitiq baseline.

   **Robust statistics, and why.**  Where a design resolves fewer scales than
   the model has parameters (S = 0.17 on odd scales: only lambda = 1 carries
   signal), the exponential estimator is not identified and not bounded --
   chi2 is flat in the decay rate and its infimum sits at rate -> infinity,
   where the extrapolation diverges; only ``exp_rate_max`` keeps it finite,
   so its *mean* over seeds measures that bound, not the estimator.  The file
   therefore carries the median and the interquartile range beside the mean,
   and coverage/bias conditioned on the resolved seeds; the printed table
   shows the median and an ``UNRESOLVED`` marker for such cells.  Coverage
   there is vacuous rather than good: read it next to
   ``median_ci_width``.

2. **Pointwise renormalisation on near-uniform inputs** (the pathology
   regime).  A 4-bit register whose ideal output is uniform (every M = 0
   instance), sampled at each scale from a toy noise model -- shot noise only,
   or shot noise plus a relaxation bias toward low Hamming weight that grows
   with the scale.  Pointwise ZNE with clip and with simplex renormalisation
   is compared with the raw ``lam = 1`` distribution in total-variation
   distance to the uniform reference.  This is **recorded, not asserted**:
   that the simplex projection is closer in L2 is a theorem, that it is
   closer in D_TV is not.

Output: ``paper/data/zne_synthetic.json`` (``save_record`` never overwrites).

Run::

    python paper/scripts/zne_synthetic.py
    python paper/scripts/zne_synthetic.py --n-seeds 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import DATA_DIR, INSTANCES_DIR, save_record  # noqa: E402
from dgssp.metrics import solution_probability, uniform_distribution  # noqa: E402
from dgssp.mitigation import mitiq_is_available  # noqa: E402
from dgssp.mitigation.mitiq_baseline import mitiq_extrapolate  # noqa: E402
from dgssp.mitigation.zne import (  # noqa: E402
    ZNESamplingConfig,
    expected_extrapolation,
    pathology_diagnostic,
    plan_scales,
    realised_scale,
    synthetic_depolarising_counts,
    zne_bootstrap,
)

SCHEMA = "dgssp/paper/zne-synthetic/2"

#: Mitiq is an optional extra; without it the baseline columns are absent.
MITIQ = mitiq_is_available()

#: Survival grid and the subset instance that supplies P_ideal, P_uniform and
#: N_2q at each S.  The S values are the subset's range, rounded.
STUDY_CASES: tuple[tuple[float, str], ...] = (
    (0.75, "P-n2-M1-11"),
    (0.40, "P-n3-M1-17"),
    (0.17, "P-n4-M2-02"),
)

SHOTS = 4000
MASTER_SEED = 20261020
BOOTSTRAP_B = 500
METHODS: tuple[str, ...] = ("linear", "quadratic", "richardson", "exponential")
ODD_GRID: tuple[float, ...] = (1.0, 3.0, 5.0)
CANDIDATES: tuple[float, ...] = (1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5)
PLAN_K = 3.0
PLAN_POINTS = 3

#: Pointwise study: 4-bit register, relaxation-bias toy model.
POINTWISE_BITS = 4
POINTWISE_SURVIVAL = 0.6
POINTWISE_BETA = 1.0
POINTWISE_GAMMAS: tuple[float, ...] = (0.0, 0.3)
POINTWISE_GRIDS: dict[str, tuple[float, ...]] = {
    "odd": (1.0, 3.0, 5.0),
    "fractional": (1.0, 1.5, 2.0),
}
POINTWISE_METHODS: tuple[str, ...] = ("linear", "richardson")


def _observable(counts: Mapping[str, int]) -> float:
    return solution_probability(counts, ["1"])


def load_cases() -> list[dict[str, Any]]:
    """Look up each study instance in the frozen hardware subset."""
    subset = json.loads((INSTANCES_DIR / "hardware_subset.json").read_text(encoding="utf-8"))
    by_name = {r["name"]: r for r in subset["selected"]}
    cases = []
    for survival, name in STUDY_CASES:
        record = by_name[name]
        cases.append(
            {
                "survival": survival,
                "instance": name,
                "instance_survival": record["survival"],
                "p_ideal": record["p_ideal"],
                "p_uniform": record["p_uniform"],
                "two_q_count": record["two_q_count"],
            }
        )
    return cases


def scalar_cell(
    case: dict[str, Any], grid_name: str, nominal: list[float], method: str, seeds: list[int]
) -> dict[str, Any]:
    """One method x S x grid cell of the scalar study, with the Mitiq baseline."""
    s, p_ideal, p_u = case["survival"], case["p_ideal"], case["p_uniform"]
    scales = [float(realised_scale(lam, case["two_q_count"])) for lam in nominal]
    cfg = ZNESamplingConfig(
        method=method,  # type: ignore[arg-type]
        asymptote="p_uniform" if method == "exponential" else None,
        n_bootstrap=BOOTSTRAP_B,
    )
    asymptote = p_u if method == "exponential" else None

    points, reported, widths, mitiq_points = [], [], [], []
    covered, covered_resolved, resolved, identifiable, failed, at_bound = 0, 0, 0, 0, 0, 0
    resolved_points: list[float] = []
    for seed in seeds:
        counts = synthetic_depolarising_counts(
            s, scales, p_ideal, p_u, SHOTS, np.random.default_rng(seed)
        )
        boot = zne_bootstrap(counts, cfg, observable_fn=_observable, p_uniform=p_u, seed=seed)
        fit = boot["fit"]
        hit = boot["lo"] <= p_ideal <= boot["hi"]
        points.append(boot["point"])
        reported.append(fit["reported_value"])
        widths.append(boot["hi"] - boot["lo"])
        covered += hit
        resolved += bool(fit["resolved"])
        identifiable += bool(fit["identifiable"])
        failed += boot["n_failed"] > 0
        at_bound += bool(fit.get("rate_at_bound"))
        if fit["resolved"]:
            resolved_points.append(boot["point"])
            covered_resolved += hit
        if MITIQ:
            mitiq_points.append(
                mitiq_extrapolate(scales, fit["observed"], method, asymptote=asymptote)
            )

    arr = np.array(points)
    resolved_arr = np.array(resolved_points)
    analytic = expected_extrapolation(cfg, scales, s, p_ideal, p_u, SHOTS)
    signal = p_ideal - p_u
    q25, q75 = np.percentile(arr, [25, 75])

    cell = {
        "method": method,
        "survival": s,
        "instance": case["instance"],
        "grid": grid_name,
        "nominal_scales": list(nominal),
        "realised_scales": scales,
        "n_seeds": len(seeds),
        # Location: the mean is meaningless where the estimator is unbounded
        # (see the module docstring); the median is the headline number.
        "mean_estimate": float(np.mean(arr)),
        "median_estimate": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "bias_mc": float(np.mean(arr) - p_ideal),
        "bias_mc_se": float(np.std(arr, ddof=1) / np.sqrt(arr.size)),
        "bias_median": float(np.median(arr) - p_ideal),
        "bias_analytic": analytic - p_ideal,
        "signal_recovered_analytic": (analytic - p_u) / signal if signal else None,
        # Conditional on the fits the paper would actually report.
        "n_resolved_seeds": resolved,
        "resolved_fraction": resolved / len(seeds),
        "identifiable_fraction": identifiable / len(seeds),
        "bias_mc_resolved": (
            float(np.mean(resolved_arr) - p_ideal) if resolved_arr.size else None
        ),
        "median_estimate_resolved": (
            float(np.median(resolved_arr)) if resolved_arr.size else None
        ),
        "coverage": covered / len(seeds),
        "coverage_count": covered,
        "coverage_resolved": covered_resolved / resolved if resolved else None,
        "median_ci_width": float(np.median(widths)),
        "fraction_with_failed_replicates": failed / len(seeds),
        "fraction_rate_at_bound": at_bound / len(seeds) if method == "exponential" else None,
        "exp_rate_max": cfg.exp_rate_max if method == "exponential" else None,
        "mean_reported_value": float(np.mean(reported)),
        "bias_reported": float(np.mean(reported) - p_ideal),
    }

    if MITIQ:
        mitiq_arr = np.array(mitiq_points, dtype=float)
        finite = mitiq_arr[np.isfinite(mitiq_arr)]
        cell.update(
            {
                "mitiq_median_estimate": float(np.median(finite)) if finite.size else None,
                "mitiq_bias_median": (
                    float(np.median(finite) - p_ideal) if finite.size else None
                ),
                "mitiq_mean_estimate": float(np.mean(finite)) if finite.size else None,
                "mitiq_max_estimate": float(np.max(finite)) if finite.size else None,
                "mitiq_n_failed": int(mitiq_arr.size - finite.size),
            }
        )
    return cell


def _pointwise_counts(
    rng: np.random.Generator, gamma: float, scales: tuple[float, ...]
) -> dict[float, dict[str, int]]:
    """Uniform ideal output plus a scale-dependent relaxation bias, sampled."""
    k = 2**POINTWISE_BITS
    keys = [format(i, f"0{POINTWISE_BITS}b") for i in range(k)]
    weights = np.array([np.exp(-POINTWISE_BETA * key.count("1")) for key in keys])
    bias = weights / weights.sum()
    uniform = np.full(k, 1.0 / k)
    out: dict[float, dict[str, int]] = {}
    for lam in scales:
        w = gamma * (1.0 - POINTWISE_SURVIVAL**lam)
        draws = rng.multinomial(SHOTS, (1.0 - w) * uniform + w * bias)
        out[float(lam)] = {key: int(c) for key, c in zip(keys, draws, strict=True)}
    return out


def pointwise_cells(seeds: list[int]) -> list[dict[str, Any]]:
    """Clip vs simplex vs raw, in D_TV to the uniform reference."""
    reference = uniform_distribution(POINTWISE_BITS)
    cells = []
    for gamma in POINTWISE_GAMMAS:
        for grid_name, scales in POINTWISE_GRIDS.items():
            for method in POINTWISE_METHODS:
                cfg = ZNESamplingConfig(method=method, mode="pointwise")  # type: ignore[arg-type]
                raw, clip, simplex = [], [], []
                clip_removed, simplex_moved, positive = [], [], []
                for seed in seeds:
                    counts = _pointwise_counts(np.random.default_rng(seed), gamma, scales)
                    diag = pathology_diagnostic(counts, cfg, reference)
                    raw.append(diag["tv_raw_min_scale"])
                    clip.append(diag["tv_pointwise_clip"])
                    simplex.append(diag["tv_pointwise_simplex"])
                    clip_removed.append(diag["clip_mass_removed"])
                    simplex_moved.append(diag["simplex_mass_moved"])
                    positive.append(diag["fraction_positive_slope"])
                raw_a, clip_a, simplex_a = np.array(raw), np.array(clip), np.array(simplex)
                cells.append(
                    {
                        "model": "shot_noise_only" if gamma == 0 else "relaxation_bias",
                        "gamma": gamma,
                        "survival": POINTWISE_SURVIVAL,
                        "beta": POINTWISE_BETA,
                        "grid": grid_name,
                        "scales": list(scales),
                        "method": method,
                        "n_seeds": len(seeds),
                        "tv_raw_mean": float(raw_a.mean()),
                        "tv_clip_mean": float(clip_a.mean()),
                        "tv_simplex_mean": float(simplex_a.mean()),
                        "fraction_clip_worse_than_raw": float(np.mean(clip_a > raw_a)),
                        "fraction_simplex_worse_than_raw": float(np.mean(simplex_a > raw_a)),
                        "fraction_simplex_closer_than_clip": float(np.mean(simplex_a < clip_a)),
                        "clip_mass_removed_mean": float(np.mean(clip_removed)),
                        "simplex_mass_moved_mean": float(np.mean(simplex_moved)),
                        "fraction_positive_slope_mean": float(np.mean(positive)),
                    }
                )
    return cells


def main(argv: list[str] | None = None) -> int:
    """Run both studies and write ``paper/data/zne_synthetic.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n-seeds", type=int, default=200, help="Monte-Carlo seeds per cell")
    args = parser.parse_args(argv)

    started = time.time()
    seeds = list(range(MASTER_SEED, MASTER_SEED + args.n_seeds))
    cases = load_cases()

    scalar: list[dict[str, Any]] = []
    grids: dict[str, dict[str, list[float]]] = {}
    for case in cases:
        planned = plan_scales(
            case["survival"],
            case["p_ideal"] - case["p_uniform"],
            SHOTS,
            CANDIDATES,
            PLAN_K,
            p_uniform=case["p_uniform"],
            n_points=PLAN_POINTS,
        )
        grids[case["instance"]] = {"odd": list(ODD_GRID), "planned": planned}
        for grid_name, nominal in grids[case["instance"]].items():
            for method in METHODS:
                cell = scalar_cell(case, grid_name, list(nominal), method, seeds)
                scalar.append(cell)
                # Where most fits are unresolved the mean is a function of
                # exp_rate_max, not of the estimator: show the median.
                unresolved = cell["resolved_fraction"] < 0.5
                mitiq = cell.get("mitiq_bias_median")
                print(
                    f"S={cell['survival']:<4} {grid_name:<7} {str(nominal):<18} {method:<11} "
                    f"bias_med={cell['bias_median']:+.3f} "
                    f"(analytic {cell['bias_analytic']:+.3f}"
                    + (f", mitiq {mitiq:+.3f}" if mitiq is not None else "")
                    + ")  "
                    f"coverage={cell['coverage']:.2f}  "
                    f"resolved={cell['resolved_fraction']:.2f} "
                    f"ident={cell['identifiable_fraction']:.2f}"
                    + ("   UNRESOLVED: mean is unbounded" if unresolved else "")
                )

    pointwise = pointwise_cells(seeds)
    for cell in pointwise:
        print(
            f"{cell['model']:<16} {cell['grid']:<10} {cell['method']:<10} "
            f"TV raw={cell['tv_raw_mean']:.4f} clip={cell['tv_clip_mean']:.4f} "
            f"simplex={cell['tv_simplex_mean']:.4f}  "
            f"simplex<clip in {cell['fraction_simplex_closer_than_clip']:.2f}"
        )

    payload = {
        "schema": SCHEMA,
        "parameters": {
            "shots_per_scale": SHOTS,
            "master_seed": MASTER_SEED,
            "n_seeds": args.n_seeds,
            "bootstrap_B": BOOTSTRAP_B,
            "confidence": 0.95,
            "methods": list(METHODS),
            "candidates": list(CANDIDATES),
            "plan_k": PLAN_K,
            "plan_points": PLAN_POINTS,
            "resolvability_k": 2.0,
            "mitiq_available": MITIQ,
            "mitiq_note": (
                "same counts, same realised scales, Mitiq's own factory "
                "extrapolation (LinearFactory / PolyFactory(2) / "
                "RichardsonFactory / ExpFactory(asymptote=P_uniform))"
            ),
            "model": "P(lam) = S**lam * P_ideal + (1 - S**lam) * P_uniform",
            "exponential_asymptote": "P_uniform (fixed)",
            "cases": cases,
            "grids": grids,
            "pointwise_model": (
                "p(lam) = (1 - w) * uniform + w * bias, w = gamma * (1 - S**lam), "
                "bias ~ exp(-beta * HammingWeight)"
            ),
        },
        "scalar": scalar,
        "pointwise": pointwise,
        "elapsed_s": round(time.time() - started, 2),
    }
    path = save_record(DATA_DIR / "zne_synthetic.json", payload)
    git = json.loads(path.read_text(encoding="utf-8"))["provenance"]["git"]
    print(f"\nwrote {path}  commit={git['short_commit']} dirty={git['dirty']}")
    print(f"total {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
