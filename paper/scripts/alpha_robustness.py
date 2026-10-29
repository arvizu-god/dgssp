"""
alpha_robustness.py

How robust is "the device behaves like its calibration with every error times
alpha"?  Offline, from files under ``paper/data`` and ``paper/hardware`` that
other scripts wrote; it simulates nothing and never contacts the service.

a. Which errors carry alpha.  The ``all`` / ``2q`` / ``readout`` fits of job 1
   (``noise_model_check.py --fit-alpha --scale-only <class>``): alpha, CI,
   chi2/dof, leave-one-record-out range, the lambda = 1 and lambda = 3 fits;
   and each class's out-of-sample prediction of job 2
   (``noise_model_check.py --predict --scale-only <class>`` on job 2's plan
   calibration, the one the pre-registered ``all`` prediction used):
   per-record lambda = 1 deviation, RMS, max.
b. Depth.  Per record (its lambda = 1 PUB and its drift copy, pooled), alpha_r
   fitted on the cached alpha grid of its own job (bootstrap B = 2000, seed 0),
   and the exponent ``ln(kept) / ln(S)`` with a delta-method SE; weighted
   regressions of both on the two-qubit count, and the <= / > 230 2Q split.
   Over-rotated variants (r > r_opt), alpha_r at the grid edge and
   non-monotone model curves are excluded from the trend, with the reason.
c. Residuals of job 1 at alpha_hat, lambda = 1 PUBs: table with z, grouped by
   sweep variant; RMS and the chi2/dof they imply.
d. Day to day: alpha_r of P-n2-M1-11 on each day and the difference.
e. Layered error: IBM's layer fidelity (``lf_<N>`` in the properties frozen
   with each job's execution calibration), EPLG, against alpha_hat times the
   median CZ error.  A ``paper/hardware/layer_fidelity_<date>.json`` from
   ``describe_layer_fidelity.py`` is listed too, flagged as not the run day.

Output: ``paper/data/alpha_robustness.json`` (``save_record`` never
overwrites) and ``paper/figures/fig_alpha_depth.{pdf,png}``.

Run::

    python paper/scripts/alpha_robustness.py
    python paper/scripts/alpha_robustness.py --figure-only paper/data/alpha_robustness.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import (  # noqa: E402
    DATA_DIR,
    FIGURES_DIR,
    HARDWARE_DIR,
    record_provenance,
    save_record,
    utc_timestamp,
)
from dgssp.backends import (  # noqa: E402
    calibration_payload,
    layer_fidelity,
    load_noise_model,
    median_gate_errors,
)
from dgssp.metrics import proportion_sigma, solution_probability  # noqa: E402
from hardware_batch import Paths, load_manifest, rel  # noqa: E402
from noise_model_check import (  # noqa: E402
    _sigma2,
    alpha_cache_dir,
    class_suffix,
    fit_alpha,
    hardware_counts,
    parse_utc,
    signal_pubs,
    used_qubit_errors,
    used_qubits,
)

SCHEMA = "dgssp/paper/alpha-robustness/1"

#: Job of record (odd grid; alpha fitted here) and job 2 (gentle grid).
JOB1, JOB2 = "2026-10-25", "2026-10-28"

#: The alpha caches, both on each job's execution-time calibration.  Job 2's is
#: labelled apart: a fit.json in hw_alpha_exec_<date> would become that job's
#: (in-sample) model in hw_aggregate.py, replacing the out-of-sample prediction.
CACHE_LABELS = {JOB1: "exec", JOB2: "exec_insample"}

#: Job 2's pre-registered prediction used the plan calibration of job 2; the
#: per-class predictions use the same one, so they compare like for like.
PREDICTION_LABEL = "plan"

#: The error classes fitted (``1q`` is selectable but not fitted: its errors
#: are ~10x below the CZ errors and cannot move P(solution) by themselves).
VARIANTS: tuple[str, ...] = ("all", "2q", "readout")

#: Per-record fits: bootstrap resamples and seed (the paper's convention for
#: new numbers; the job-level fits keep the B of the paper's alpha fit).
RECORD_B = 2000
RECORD_SEED = 0

#: Split of the depth trend, from ``hw_review_wp7.md`` §2 F2 -- chosen after
#: seeing the data, so the split test is descriptive, not a pre-registered test.
DEPTH_SPLIT_2Q = 230

#: The record whose alpha is compared between days (same instance, same
#: transpiled 2Q count, a different layout).
DAY_RECORD = "P-n2-M1-11"

#: The day-2 record at random on every circuit (hw_results.md §4): an
#: out-of-sample failure, not a depth effect.
DAY2_FAILURE = "P-n3-M2-17"

Z95 = 1.959963984540054

OUT = DATA_DIR / "alpha_robustness.json"
FIGURE = "fig_alpha_depth"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def base_key(key: str) -> str:
    """A drift copy belongs to its record: ``X_drift`` -> ``X``."""
    return key.removesuffix("_drift")


def read_cached_counts(cache: Path, alpha: float, which: str) -> list[dict[str, int]]:
    """One alpha's cached counts; never simulates (a missing file is an error)."""
    path = cache / f"counts_alpha_{alpha:g}.json"
    if not path.exists():
        raise SystemExit(f"{rel(path)} missing: the alpha grid of {rel(cache)} is incomplete "
                         "(noise_model_check.py --fit-alpha builds it).")
    stored = _load_json(path)
    if stored.get("which", "all") != which:
        raise SystemExit(f"{rel(path)} holds '{stored.get('which', 'all')}' counts, not '{which}'.")
    return [dict(c) for c in stored["counts"]]


def load_job(date: str, which: str = "all") -> dict[str, Any]:
    """
    One job's PUBs with signal, hardware and cached-model P(solution).

    Returns
    -------
    dict
        ``paths``, ``manifest``, ``shots``, ``alphas``, ``fit`` (the cache's
        ``fit.json``), ``pubs`` (:func:`noise_model_check.signal_pubs` rows
        plus ``base``, ``n_items``, ``iterations``, ``r_opt``, ``role``,
        ``physical_qubits``), ``hw`` ``(n_pub,)`` and ``sim``
        ``(n_alpha, n_pub)``.
    """
    paths = Paths(date, HARDWARE_DIR, DATA_DIR)
    manifest, _, _ = load_manifest(paths.manifest)
    cache = alpha_cache_dir(DATA_DIR, CACHE_LABELS[date], which, date)
    fit_path = cache / "fit.json"
    if not fit_path.exists():
        raise SystemExit(f"{rel(fit_path)} missing: run noise_model_check.py --fit-alpha "
                         f"--date {date} --label {CACHE_LABELS[date]}"
                         + ("" if which == "all" else f" --scale-only {which}"))
    fit = _load_json(fit_path)
    alphas = [float(a) for a in fit["alphas"]]
    records = {r["key"]: r for r in manifest["records"]}
    circuits = {int(c["pub_index"]): c for c in manifest["circuits"]}
    pubs = []
    for p in signal_pubs(manifest):
        rec = records[p["key"]]
        pubs.append({**p, "base": base_key(p["key"]), "n_items": int(rec["n_items"]),
                     "iterations": int(rec["iterations"]), "r_opt": int(rec["r_opt"]),
                     "role": rec["role"],
                     "physical_qubits": [int(q) for q in circuits[p["pub"]]["physical_qubits"]]})
    hw_counts = hardware_counts(paths)
    hw = np.array([solution_probability(hw_counts[p["pub"]], p["solution_bitstrings"])
                   for p in pubs])
    sim = np.empty((len(alphas), len(pubs)))
    for i, alpha in enumerate(alphas):
        counts = read_cached_counts(cache, alpha, which)
        sim[i] = [solution_probability(counts[p["pub"]], p["solution_bitstrings"]) for p in pubs]
    return {"date": date, "paths": paths, "manifest": manifest, "cache": cache,
            "shots": int(manifest["settings"]["shots"]), "alphas": alphas, "fit": fit,
            "pubs": pubs, "hw": hw, "sim": sim}


def summary_path(date: str) -> Path:
    """``hw_aggregate.py``'s summary of a job (the job of record has no suffix)."""
    return DATA_DIR / ("hw_summary.json" if date == JOB1 else f"hw_summary_{date}.json")


def survival_records(date: str) -> dict[str, dict[str, Any]]:
    """``survival.records`` of the job's summary, by record key."""
    return {r["key"]: r for r in _load_json(summary_path(date))["survival"]["records"]}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wls(x: Sequence[float], y: Sequence[float], se: Sequence[float]) -> dict[str, Any] | None:
    """
    Weighted least squares ``y = a + b x`` with weights ``1 / se**2``.

    The covariance is scaled by ``max(1, chi2/dof)`` -- the same correction
    the alpha fit's ``ci_scaled`` applies -- and the 95 % intervals use
    Student's t on ``n - 2`` degrees of freedom.
    """
    from scipy.stats import t as student_t

    xa, ya, sa = (np.asarray(v, dtype=float) for v in (x, y, se))
    n = xa.size
    if n < 3:
        return None
    w = 1.0 / sa**2
    design = np.column_stack([np.ones(n), xa])
    normal = design.T @ (w[:, None] * design)
    beta = np.linalg.solve(normal, design.T @ (w * ya))
    resid = ya - design @ beta
    chi2 = float(np.sum(w * resid**2))
    dof = n - 2
    scale = max(1.0, chi2 / dof)
    cov = np.linalg.inv(normal) * scale
    tq = float(student_t.ppf(0.975, dof))
    se_b = math.sqrt(cov[1, 1])
    return {
        "n": int(n),
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "slope_se": se_b,
        "slope_ci": [float(beta[1] - tq * se_b), float(beta[1] + tq * se_b)],
        "slope_per_100_2q": float(100 * beta[1]),
        "slope_per_100_2q_ci": [float(100 * (beta[1] - tq * se_b)),
                                float(100 * (beta[1] + tq * se_b))],
        "chi2": chi2,
        "dof": dof,
        "chi2_red": chi2 / dof,
        "covariance_scale": scale,
        "t_quantile": tq,
    }


def weighted_mean(y: Sequence[float], se: Sequence[float]) -> dict[str, Any] | None:
    """Inverse-variance mean, its SE scaled by ``sqrt(max(1, chi2/dof))``."""
    ya, sa = np.asarray(y, dtype=float), np.asarray(se, dtype=float)
    if ya.size == 0:
        return None
    w = 1.0 / sa**2
    mean = float(np.sum(w * ya) / np.sum(w))
    chi2 = float(np.sum(w * (ya - mean) ** 2))
    dof = ya.size - 1
    scale = max(1.0, chi2 / dof) if dof > 0 else 1.0
    return {"n": int(ya.size), "mean": mean, "se": math.sqrt(scale / np.sum(w)),
            "chi2_red": chi2 / dof if dof > 0 else None}


def split_test(rows: Sequence[Mapping[str, Any]], value: str, se: str) -> dict[str, Any]:
    """Weighted means of ``value`` at <= and > :data:`DEPTH_SPLIT_2Q`, and their difference."""
    from scipy.stats import norm

    lo = [r for r in rows if r["two_q"] <= DEPTH_SPLIT_2Q]
    hi = [r for r in rows if r["two_q"] > DEPTH_SPLIT_2Q]
    m_lo = weighted_mean([r[value] for r in lo], [r[se] for r in lo])
    m_hi = weighted_mean([r[value] for r in hi], [r[se] for r in hi])
    out: dict[str, Any] = {"threshold_2q": DEPTH_SPLIT_2Q, "post_hoc": True,
                           "shallow": m_lo, "deep": m_hi,
                           "shallow_keys": [r["id"] for r in lo], "deep_keys": [r["id"] for r in hi]}
    if m_lo and m_hi:
        diff = m_hi["mean"] - m_lo["mean"]
        se_d = math.hypot(m_lo["se"], m_hi["se"])
        out |= {"difference": diff, "difference_se": se_d,
                "difference_ci": [diff - Z95 * se_d, diff + Z95 * se_d],
                "z": diff / se_d, "p_two_sided": float(2 * norm.sf(abs(diff / se_d)))}
    return out


def model_at(alphas: Sequence[float], sim: np.ndarray, alpha: float) -> np.ndarray:
    """Each PUB's predicted P(solution) at ``alpha``: the fit's monotone cubic interpolation."""
    from scipy.interpolate import PchipInterpolator

    return np.asarray(PchipInterpolator(np.asarray(alphas), sim, axis=0)(alpha))


def monotone(curve: np.ndarray, shots: int) -> bool:
    """Model P(solution) never rises with alpha by more than 2 sigma of simulation noise."""
    steps = np.diff(curve)
    noise = np.sqrt(_sigma2(curve[:-1], shots) + _sigma2(curve[1:], shots))
    return bool(np.all(steps <= 2.0 * noise))


def rms(values: Sequence[float]) -> float | None:
    return float(math.sqrt(np.mean(np.square(values)))) if len(values) else None


# ---------------------------------------------------------------------------
# a. Which errors carry alpha
# ---------------------------------------------------------------------------


def fit_summary(fit: Mapping[str, Any]) -> dict[str, Any]:
    """The numbers of one subset's alpha fit a note quotes."""
    loo = list((fit.get("loo") or {}).values())
    return {
        "alpha": fit["alpha"],
        "ci": fit["ci"],
        "ci_scaled": fit["ci_scaled"],
        "chi2_min": fit["chi2_min"],
        "dof": fit["dof"],
        "chi2_red": fit["chi2_red"],
        "n_pubs": fit["n_pubs"],
        "at_grid_edge": fit["at_grid_edge"],
        "loo_range": [min(loo), max(loo)] if loo else None,
    }


def prediction_dir(which: str, alpha: float) -> Path:
    """Where ``--predict --scale-only <which> --alpha <alpha>`` puts job 2's prediction."""
    return DATA_DIR / f"hw_predict_{PREDICTION_LABEL}_x{alpha:g}{class_suffix(which)}_{JOB2}"


def predict_command(which: str, alpha: float) -> str:
    return (f"python paper/scripts/noise_model_check.py --predict --date {JOB2} "
            f"--calibration paper/hardware/calibration_{JOB2}.json --label {PREDICTION_LABEL} "
            f"--alpha {alpha:g}" + ("" if which == "all" else f" --scale-only {which}"))


def _p_by_scale(record: Mapping[str, Any]) -> list[tuple[float, float]]:
    """``(realised scale, P(solution))`` of a stored record, in increasing scale."""
    sols = record["metrics"]["solution_bitstrings"]
    return [(float(k), solution_probability(v, sols))
            for k, v in sorted(record["counts"].items(), key=lambda kv: float(kv[0]))]


def out_of_sample(which: str, alpha_fit: float) -> dict[str, Any]:
    """Job 2 against the ``which`` model at the job-1 alpha (rounded as the paper rounds it)."""
    alpha = round(alpha_fit, 3)
    directory = prediction_dir(which, alpha)
    if not directory.exists():
        return {"alpha_used": alpha, "path": rel(directory), "missing": True,
                "command": predict_command(which, alpha)}
    hw_dir = DATA_DIR / f"hw_{JOB2}"
    per_record: list[dict[str, Any]] = []
    per_pub: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        pred = _load_json(path)
        info = pred["metrics"]["prediction"]
        if not math.isclose(float(info["alpha"]), alpha, abs_tol=1e-9) or \
                info["scale_info"].get("which", "all") != which:
            raise SystemExit(f"{rel(path)} is alpha {info['alpha']} "
                             f"({info['scale_info'].get('which', 'all')}), expected {alpha} ({which}).")
        hw = _load_json(hw_dir / path.name)
        shots = int(hw["shots"])
        p_u = float(hw["metrics"]["p_uniform"])
        for scale_index, ((lam, p_pred), (_, p_hw)) in enumerate(
                zip(_p_by_scale(pred), _p_by_scale(hw), strict=True)):
            sigma = math.hypot(proportion_sigma(p_hw, shots), proportion_sigma(p_pred, shots))
            row: dict[str, Any] = {"key": path.stem, "scale": lam, "hardware": p_hw, "predicted": p_pred,
                   "deviation": p_hw - p_pred, "sigma": sigma, "z": (p_hw - p_pred) / sigma,
                   "p_uniform": p_u}
            per_pub.append(row)
            if scale_index == 0:
                per_record.append(row)
    main = [r for r in per_record if not r["key"].endswith("_drift")]
    worst = max(main, key=lambda r: abs(r["deviation"]))
    return {
        "alpha_used": alpha,
        "path": rel(directory),
        "calibration": rel(HARDWARE_DIR / f"calibration_{JOB2}.json"),
        "per_record_lambda1": per_record,
        "lambda1_records": [r["key"] for r in main],
        "lambda1_rms": rms([r["deviation"] for r in main]),
        "lambda1_rms_without_failure": rms([r["deviation"] for r in main
                                            if r["key"] != DAY2_FAILURE]),
        "lambda1_max_abs": {"key": worst["key"], "deviation": worst["deviation"],
                            "z": worst["z"]},
        "all_pubs_rms": rms([r["deviation"] for r in per_pub]),
        "all_pubs_rms_without_failure": rms([r["deviation"] for r in per_pub
                                             if base_key(r["key"]) != DAY2_FAILURE]),
        "n_pubs": len(per_pub),
    }


def variants_block() -> dict[str, Any]:
    """Each error class: its job-1 fit and its job-2 prediction."""
    out: dict[str, Any] = {}
    for which in VARIANTS:
        path = alpha_cache_dir(DATA_DIR, CACHE_LABELS[JOB1], which, JOB1) / "fit.json"
        if not path.exists():
            out[which] = {"missing": True, "path": rel(path)}
            continue
        doc = _load_json(path)
        fits = doc["fits"]
        out[which] = {
            "path": rel(path),
            "what_alpha_scales": doc.get("what_alpha_scales"),
            "alphas": doc["alphas"],
            "all": fit_summary(fits["all"]),
            "lambda1": fit_summary(fits["lambda1"]),
            "lambda3": fit_summary(fits["lambda3"]),
            "lambda5": fit_summary(fits["lambda5"]),
            "out_of_sample_job2": out_of_sample(which, float(fits["all"]["alpha"])),
        }
    if "all" in out and not out["all"].get("missing"):
        base = out["all"]["all"]["chi2_min"]
        for which in VARIANTS:
            if not out[which].get("missing"):
                out[which]["delta_chi2_vs_all"] = out[which]["all"]["chi2_min"] - base
    return out


# ---------------------------------------------------------------------------
# b. Depth: per-record alpha and exponent
# ---------------------------------------------------------------------------


def record_rows(job: Mapping[str, Any], *, keep_draws_for: str | None = None) -> list[dict[str, Any]]:
    """Per record: alpha_r on its lambda = 1 PUBs (drift copy pooled), exponent, exclusions."""
    pubs, hw, sim, shots, alphas = job["pubs"], job["hw"], job["sim"], job["shots"], job["alphas"]
    survival = survival_records(job["date"])
    rows = []
    for base in dict.fromkeys(p["base"] for p in pubs):
        idx = [j for j, p in enumerate(pubs) if p["base"] == base and p["scale"] == 1.0]
        main = next(j for j in idx if pubs[j]["key"] == base)
        p = pubs[main]
        fit = fit_alpha(alphas, sim[:, idx], shots, hw[idx], shots, B=RECORD_B, seed=RECORD_SEED,
                        keep_draws=base == keep_draws_for)
        surv = survival[base]
        s1, two_q = float(surv["survival_alpha_1"]), int(surv["two_q_physical"])
        p_ideal, p_u = p["p_ideal"], p["p_uniform"]
        kept = (hw[main] - p_u) / (p_ideal - p_u)
        kept_se = proportion_sigma(float(hw[main]), shots) / abs(p_ideal - p_u)
        stored = surv["kept_observed"][0]
        if stored is None or not math.isclose(kept, stored, rel_tol=1e-9, abs_tol=1e-12):
            raise SystemExit(f"{base}: kept {kept} != stored {stored} in {rel(summary_path(job['date']))}")
        exponent = math.log(kept) / math.log(s1) if 0 < kept < 1 else None
        exponent_se = kept_se / (kept * abs(math.log(s1))) if exponent is not None else None
        curves_monotone = all(monotone(sim[:, j], shots) for j in idx)
        reasons = []
        if p["iterations"] > p["r_opt"]:
            reasons.append("over-rotated (r > r_opt)")
        if fit["at_grid_edge"]:
            reasons.append("alpha_r at the grid edge")
        if not curves_monotone:
            reasons.append("model P(solution) not monotone in alpha")
        if job["date"] == JOB2 and base == DAY2_FAILURE:
            reasons.append("day-2 out-of-sample failure (at random on every circuit)")
        row = {
            "id": f"{base}@{job['date']}",
            "key": base,
            "job": job["date"],
            "n_items": p["n_items"],
            "iterations": p["iterations"],
            "r_opt": p["r_opt"],
            "role": p["role"],
            "two_q": two_q,
            "pubs": [pubs[j]["pub"] for j in idx],
            "shots": shots,
            "hardware": [float(hw[j]) for j in idx],
            "alpha_r": fit["alpha"],
            "alpha_r_ci": fit["ci"],
            "alpha_r_ci_scaled": fit["ci_scaled"],
            "alpha_r_se": fit["bootstrap_se"],
            "alpha_r_chi2_red": fit["chi2_red"],
            "at_grid_edge": fit["at_grid_edge"],
            "model_monotone": curves_monotone,
            "survival_alpha_1": s1,
            "kept": kept,
            "kept_se": kept_se,
            "exponent": exponent,
            "exponent_se": exponent_se,
            "exponent_stored": surv.get("exponent"),
            "in_trend": not reasons,
            "excluded_because": reasons,
        }
        if "draws" in fit:
            row["_draws"] = fit["draws"]
        rows.append(row)
    return rows


def depth_block(rows: Sequence[Mapping[str, Any]], alpha_hat: Mapping[str, Any]) -> dict[str, Any]:
    """Trends of alpha_r and of the exponent with the 2Q count, on the rows in the trend."""
    trend = [r for r in rows if r["in_trend"]]
    exp_rows = [dict(r) for r in trend if r["exponent"] is not None and r["exponent_se"]]
    return {
        "alpha_hat": alpha_hat,
        "B": RECORD_B,
        "seed": RECORD_SEED,
        "pooling": "each record's lambda = 1 PUB and its drift copy (same circuit) are fitted "
                   "together: one alpha_r per distinct circuit and day",
        "exclusion_rule": "r > r_opt (over-rotated: past r_opt noise can raise P(solution), so "
                          "P(alpha) and kept are not a survival); alpha_r at the grid edge; "
                          "non-monotone model curve; the day-2 failure record",
        "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "trend_ids": [r["id"] for r in trend],
        "alpha_r_vs_2q": wls([r["two_q"] for r in trend], [r["alpha_r"] for r in trend],
                             [r["alpha_r_se"] for r in trend]),
        "alpha_r_split": split_test(trend, "alpha_r", "alpha_r_se"),
        "alpha_r_job1_only_vs_2q": wls(
            [r["two_q"] for r in trend if r["job"] == JOB1],
            [r["alpha_r"] for r in trend if r["job"] == JOB1],
            [r["alpha_r_se"] for r in trend if r["job"] == JOB1]),
        "exponent_vs_2q": wls([r["two_q"] for r in exp_rows], [r["exponent"] for r in exp_rows],
                              [r["exponent_se"] for r in exp_rows]),
        "exponent_split": split_test(exp_rows, "exponent", "exponent_se"),
    }


# ---------------------------------------------------------------------------
# c. Residuals at alpha_hat
# ---------------------------------------------------------------------------


def variant_group(p: Mapping[str, Any]) -> str:
    """Sweep variant of a PUB, for grouping residuals."""
    if p["key"].endswith("_drift"):
        return "drift copy (r = r_opt)"
    if p["iterations"] == p["r_opt"]:
        return "r = r_opt"
    side = "r < r_opt" if p["iterations"] < p["r_opt"] else "r > r_opt"
    return f"n = {p['n_items']}, {side}"


def residual_block(job: Mapping[str, Any]) -> dict[str, Any]:
    """Job 1, lambda = 1 PUBs: hardware minus the model at the paper's alpha_hat."""
    alpha_hat = float(job["fit"]["fits"]["all"]["alpha"])
    shots = job["shots"]
    idx = [j for j, p in enumerate(job["pubs"]) if p["scale"] == 1.0]
    model = model_at(job["alphas"], job["sim"][:, idx], alpha_hat)
    hw = job["hw"][idx]
    sigma = np.sqrt(_sigma2(hw, shots) + _sigma2(model, shots))
    rows = []
    for k, j in enumerate(idx):
        p = job["pubs"][j]
        rows.append({"pub": p["pub"], "key": p["key"], "n_items": p["n_items"],
                     "iterations": p["iterations"], "r_opt": p["r_opt"],
                     "group": variant_group(p), "hardware": float(hw[k]),
                     "model": float(model[k]), "residual": float(hw[k] - model[k]),
                     "sigma": float(sigma[k]), "z": float((hw[k] - model[k]) / sigma[k])})
    z = np.array([r["z"] for r in rows])
    groups: dict[str, Any] = {}
    for name in dict.fromkeys(r["group"] for r in rows):
        g = [r for r in rows if r["group"] == name]
        groups[name] = {"n": len(g), "keys": [r["key"] for r in g],
                        "mean_residual": float(np.mean([r["residual"] for r in g])),
                        "mean_z": float(np.mean([r["z"] for r in g])),
                        "n_above_model": sum(r["residual"] > 0 for r in g),
                        "n_below_model": sum(r["residual"] < 0 for r in g)}
    chi2 = float(np.sum(z**2))
    return {
        "alpha_hat": alpha_hat,
        "fit_file": rel(job["cache"] / "fit.json"),
        "sigma": "sqrt(var_hw + var_model), Jeffreys-smoothed binomial at the job's shots "
                 "(the alpha fit's weights)",
        "rows": rows,
        "n": len(rows),
        "rms": rms([r["residual"] for r in rows]),
        "max_abs": max(rows, key=lambda r: abs(r["residual"]))["residual"],
        "chi2": chi2,
        "dof": len(rows) - 1,
        "chi2_red": chi2 / (len(rows) - 1),
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# d. Day to day
# ---------------------------------------------------------------------------


def day_block(rows1: Sequence[Mapping[str, Any]], rows2: Sequence[Mapping[str, Any]],
              job1: Mapping[str, Any], job2: Mapping[str, Any]) -> dict[str, Any]:
    """alpha_r of every record run on both days; the headline is :data:`DAY_RECORD`."""
    by1, by2 = {r["key"]: r for r in rows1}, {r["key"]: r for r in rows2}
    summaries = {d: _load_json(summary_path(d))["survival"]["medians"]["alpha_1"]["two_qubit"]
                 for d in (JOB1, JOB2)}

    def qubits(job: Mapping[str, Any], key: str) -> list[int]:
        return next(p["physical_qubits"] for p in job["pubs"]
                    if p["key"] == key and p["scale"] == 1.0)

    shared = []
    for key in by2:
        if key not in by1:
            continue
        a, b = by1[key], by2[key]
        q1, q2 = qubits(job1, key), qubits(job2, key)
        shared.append({"key": key, "two_q": [a["two_q"], b["two_q"]],
                       "alpha_r": [a["alpha_r"], b["alpha_r"]],
                       "alpha_r_ci_scaled": [a["alpha_r_ci_scaled"], b["alpha_r_ci_scaled"]],
                       "difference": b["alpha_r"] - a["alpha_r"],
                       "z": (b["alpha_r"] - a["alpha_r"]) / math.hypot(a["alpha_r_se"], b["alpha_r_se"]),
                       "same_layout": q1 == q2, "shared_qubits": len(set(q1) & set(q2)),
                       "n_qubits": len(q1), "excluded_because": [a["excluded_because"],
                                                                 b["excluded_because"]]})
    a, b = by1[DAY_RECORD], by2[DAY_RECORD]
    diff_draws = np.asarray(b["_draws"]) - np.asarray(a["_draws"])
    lo, hi = np.percentile(diff_draws, [2.5, 97.5])
    diff = b["alpha_r"] - a["alpha_r"]
    se = math.hypot(a["alpha_r_se"], b["alpha_r_se"])
    return {
        "record": DAY_RECORD,
        "note": "each day's alpha_r is relative to that day's execution calibration; the "
                "layouts differ; shots 4096 (day 1) and 12288 (day 2)",
        "alpha_r": {JOB1: a["alpha_r"], JOB2: b["alpha_r"]},
        "alpha_r_ci_scaled": {JOB1: a["alpha_r_ci_scaled"], JOB2: b["alpha_r_ci_scaled"]},
        "difference": diff,
        "difference_ci_bootstrap": [float(lo), float(hi)],
        "difference_se": se,
        "z": diff / se,
        "relative_change": diff / a["alpha_r"],
        "two_q": [a["two_q"], b["two_q"]],
        "median_cz_error": summaries,
        "shared_records": shared,
    }


# ---------------------------------------------------------------------------
# e. Layered error
# ---------------------------------------------------------------------------


def _cz_errors(payload: Mapping[str, Any]) -> dict[frozenset[int], float]:
    return {frozenset(r["qubits"]): float(r["error"]) for r in payload["instructions"].get("cz", [])
            if r["error"] is not None and r["qubits"] is not None}


def layered_day(date: str, alpha_all: float, alpha_2q: float | None) -> dict[str, Any]:
    """The layer fidelity frozen with one job's execution calibration, against alpha * eps_CZ."""
    paths = Paths(date, HARDWARE_DIR, DATA_DIR)
    calibration = HARDWARE_DIR / f"calibration_exec_{date}.json"
    raw = _load_json(calibration).get("raw_properties")
    lf = layer_fidelity(raw)
    if not lf:
        return {"calibration": rel(calibration), "layer_fidelity": None}
    snapshot = load_noise_model(calibration)
    payload = calibration_payload(snapshot.backend)
    cz = _cz_errors(payload)
    device_cz = float(median_gate_errors(snapshot.backend)["two_qubit"])
    manifest, _, _ = load_manifest(paths.manifest)
    job_cz = used_qubit_errors(payload, used_qubits(manifest))["cz"]["median"]

    n_max = max(lf)
    top = lf[n_max]
    chain = top["qubits"] or []
    chain_errs = [cz.get(frozenset(pair)) for pair in zip(chain[:-1], chain[1:], strict=True)]
    found = [e for e in chain_errs if e is not None]
    chain_median = float(np.median(found)) if found else None
    # The isolated errors' own prediction of the chain's EPLG: every CZ of the
    # chain once, as average gate infidelity and as process infidelity
    # (d = 4: 1 - F_pro = (5/4) (1 - F_avg)); single-qubit gates left out.
    iso_avg = 1.0 - float(np.prod([1 - e for e in found])) ** (1 / len(found)) if found else None
    iso_pro = 1.0 - float(np.prod([1 - 1.25 * e for e in found])) ** (1 / len(found)) if found else None

    jobs = _load_json(paths.jobs)["jobs"]
    job_id = jobs[0]["job_id"]
    running = _load_json(paths.usage_json)[job_id]["metrics"]["timestamps"]["running"]
    lf_date = top["date"]
    age_h = None
    if lf_date:
        age_h = (parse_utc(running) - parse_utc(str(lf_date).replace(" ", "T"))).total_seconds() / 3600
    eplg = top["eplg"]
    return {
        "calibration": rel(calibration),
        "calibration_timestamp": snapshot.calibration_timestamp,
        "job_id": job_id,
        "job_running_utc": running,
        "lf_date": lf_date,
        "lf_age_at_run_h": age_h,
        "chain_lengths": [min(lf), n_max],
        "lf": top["lf"],
        "n_2q": top["n_2q"],
        "eplg": eplg,
        "eplg_by_n": {str(n): v["eplg"] for n, v in lf.items()},
        "chain_qubits": chain,
        "chain_cz_found": len(found),
        "chain_cz_median": chain_median,
        "chain_isolated_eplg_avg": iso_avg,
        "chain_isolated_eplg_process": iso_pro,
        "device_cz_median": device_cz,
        "job_qubits_cz_median": job_cz,
        "alpha_hat_all": alpha_all,
        "alpha_hat_2q": alpha_2q,
        "alpha_all_x_device_cz": alpha_all * device_cz,
        "alpha_2q_x_device_cz": alpha_2q * device_cz if alpha_2q is not None else None,
        "ratio_eplg_to_device_cz": eplg / device_cz,
        "ratio_eplg_to_chain_cz": eplg / chain_median if chain_median else None,
        "ratio_eplg_to_chain_isolated_process": eplg / iso_pro if iso_pro else None,
    }


def layered_block(variants: Mapping[str, Any]) -> dict[str, Any]:
    alpha_all = float(variants["all"]["all"]["alpha"])
    alpha_2q = (float(variants["2q"]["all"]["alpha"])
                if not variants.get("2q", {}).get("missing") else None)
    today = []
    for path in sorted(HARDWARE_DIR.glob("layer_fidelity_*.json")):
        doc = _load_json(path)
        today.append({"path": rel(path), "fetched_utc": doc.get("fetched_utc"),
                      "eplg": doc.get("eplg"), "lf_date": doc.get("lf_date"),
                      "note": "fetched after the jobs: not the run day's calibration"})
    return {
        "definition": "EPLG = 1 - LF**(1/(N-1)) for the longest chain N reported (lf_<N> in "
                      "BackendProperties.general, chain in general_qlists); a process "
                      "infidelity per layered gate, single-qubit gates and crosstalk included",
        "alpha_from": "the job-1 'all' and '2q' fits",
        "days": {JOB1: layered_day(JOB1, alpha_all, alpha_2q),
                 JOB2: layered_day(JOB2, alpha_all, alpha_2q)},
        "fetched_later": today,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(payload: Mapping[str, Any], directory: Path = FIGURES_DIR) -> list[Path]:
    """``fig_alpha_depth``: (a) alpha_r and (b) the exponent against the 2Q count."""
    from plotting import FULL_WIDTH_IN, OKABE_ITO, figure, save_figure

    depth = payload["depth"]
    rows = depth["rows"]
    band = depth["alpha_hat"]
    fig, (ax_a, ax_b) = figure(width=FULL_WIDTH_IN, aspect=0.42, ncols=2)
    style = {JOB1: ("o", OKABE_ITO["blue"], "job 1 (odd grid)"),
             JOB2: ("s", OKABE_ITO["vermillion"], "job 2 (gentle grid)")}

    for ax, value, ci_of in ((ax_a, "alpha_r", "alpha_r_ci_scaled"), (ax_b, "exponent", None)):
        ax.axhspan(band["ci_scaled"][0], band["ci_scaled"][1], color=OKABE_ITO["grey"],
                   alpha=0.18, lw=0, zorder=0)
        ax.axhline(band["alpha"], color=OKABE_ITO["grey"], lw=0.8, zorder=1,
                   label=rf"$\hat\alpha = {band['alpha']:.2f}$ (job 1, all PUBs)")
        ax.axvline(DEPTH_SPLIT_2Q, color=OKABE_ITO["black"], lw=0.5, ls=":", zorder=1)
        for job, (marker, colour, label) in style.items():
            for in_trend in (True, False):
                pts = [r for r in rows if r["job"] == job and r["in_trend"] == in_trend
                       and r[value] is not None]
                if not pts:
                    continue
                x = np.array([r["two_q"] for r in pts], dtype=float)
                y = np.array([r[value] for r in pts], dtype=float)
                if ci_of:
                    lo = y - np.array([r[ci_of][0] for r in pts])
                    hi = np.array([r[ci_of][1] for r in pts]) - y
                else:
                    se = np.array([r["exponent_se"] for r in pts])
                    lo = hi = Z95 * se
                ax.errorbar(x, y, yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)], fmt=marker,
                            color=colour, ms=4, lw=0.8, capsize=1.5,
                            mfc=colour if in_trend else "none", zorder=4 if in_trend else 3,
                            label=label if in_trend else f"{label}, excluded")
        trend = depth["alpha_r_vs_2q" if value == "alpha_r" else "exponent_vs_2q"]
        if trend:
            xs = np.array([50.0, 1000.0])
            ax.plot(xs, trend["intercept"] + trend["slope"] * xs, color=OKABE_ITO["black"],
                    lw=0.8, ls="--", zorder=2, label="weighted fit (in trend)")
        ax.set_xscale("log")
        ax.set_xlim(50, 1100)
        ax.set_xlabel(r"two-qubit gates ($\lambda = 1$ circuit)")
    for ax, value, top in ((ax_a, "alpha_r", 6.5), (ax_b, "exponent", 6.0)):
        # Points above the axis (alpha_r at the grid edge): an arrow on the top edge.
        for job, (_, colour, _) in style.items():
            off = [r["two_q"] for r in rows if r["job"] == job and r[value] is not None
                   and r[value] > top]
            if off:
                ax.plot(off, [top - 0.12] * len(off), "^", color=colour, mfc="none", ms=5,
                        zorder=4, clip_on=False)
    ax_a.set_ylim(0.5, 6.5)
    ax_b.set_ylim(0.0, 6.0)
    ax_a.set_ylabel(r"$\alpha_r$ (per-record fit)")
    ax_b.set_ylabel(r"exponent $\ln(\mathrm{kept}) / \ln S$")
    ax_a.set_title("(a)", loc="left")
    ax_b.set_title("(b)", loc="left")
    ax_a.legend(loc="lower left", fontsize=6.0)
    return save_figure(fig, FIGURE, directory=directory)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build() -> dict[str, Any]:
    job1, job2 = load_job(JOB1), load_job(JOB2)
    variants = variants_block()
    rows1 = record_rows(job1, keep_draws_for=DAY_RECORD)
    rows2 = record_rows(job2, keep_draws_for=DAY_RECORD)
    alpha_hat = fit_summary(job1["fit"]["fits"]["all"])
    payload = {
        "schema": SCHEMA,
        "jobs": {JOB1: {"cache": rel(job1["cache"]), "shots": job1["shots"]},
                 JOB2: {"cache": rel(job2["cache"]), "shots": job2["shots"],
                        "fit_all_lambda1": fit_summary(job2["fit"]["fits"]["lambda1"])}},
        "variants": variants,
        "depth": depth_block(rows1 + rows2, alpha_hat),
        "residuals": residual_block(job1),
        "day_to_day": day_block(rows1, rows2, job1, job2),
        "layered": layered_block(variants),
        "created_utc": utc_timestamp(),
        "provenance": record_provenance(),
    }
    return payload


def _f(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_summary(payload: Mapping[str, Any]) -> None:
    print("a. which errors carry alpha (job 1 fit; job 2 out of sample)")
    for which, v in payload["variants"].items():
        if v.get("missing"):
            print(f"  {which:<8} missing: {v['path']}")
            continue
        f, o = v["all"], v["out_of_sample_job2"]
        oos = ("missing: " + o["command"]) if o.get("missing") else (
            f"job2 lam1 RMS {_f(o['lambda1_rms'])} (w/o failure {_f(o['lambda1_rms_without_failure'])}), "
            f"max {o['lambda1_max_abs']['key']} {_f(o['lambda1_max_abs']['deviation'])}")
        print(f"  {which:<8} alpha {_f(f['alpha'])} [{_f(f['ci_scaled'][0])}, {_f(f['ci_scaled'][1])}] "
              f"chi2/dof {_f(f['chi2_red'], 2)} dchi2 {_f(v.get('delta_chi2_vs_all'), 1)} "
              f"loo {_f(f['loo_range'][0], 2)}..{_f(f['loo_range'][1], 2)} "
              f"lam1 {_f(v['lambda1']['alpha'], 2)} lam3 {_f(v['lambda3']['alpha'], 2)}"
              f"{'  EDGE' if f['at_grid_edge'] else ''}\n           {oos}")
    d = payload["depth"]
    print("\nb. depth (alpha_r per record; exponent)")
    for r in d["rows"]:
        print(f"  {r['id']:<26} 2Q {r['two_q']:>4}  alpha_r {_f(r['alpha_r'], 2)} "
              f"[{_f(r['alpha_r_ci_scaled'][0], 2)}, {_f(r['alpha_r_ci_scaled'][1], 2)}]  "
              f"exp {_f(r['exponent'], 2)} +- {_f(r['exponent_se'], 2)}  "
              f"{'; '.join(r['excluded_because'])}")
    for name in ("alpha_r_vs_2q", "exponent_vs_2q"):
        t = d[name]
        if t:
            print(f"  {name}: slope/100 2Q {_f(t['slope_per_100_2q'])} "
                  f"[{_f(t['slope_per_100_2q_ci'][0])}, {_f(t['slope_per_100_2q_ci'][1])}] "
                  f"n {t['n']} chi2/dof {_f(t['chi2_red'], 2)}")
    for name in ("alpha_r_split", "exponent_split"):
        s = d[name]
        if "difference" in s:
            print(f"  {name}: shallow {_f(s['shallow']['mean'], 2)} (n {s['shallow']['n']}), "
                  f"deep {_f(s['deep']['mean'], 2)} (n {s['deep']['n']}), diff "
                  f"{_f(s['difference'], 2)} [{_f(s['difference_ci'][0], 2)}, "
                  f"{_f(s['difference_ci'][1], 2)}] z {_f(s['z'], 1)}")
    r = payload["residuals"]
    print(f"\nc. residuals at alpha_hat {_f(r['alpha_hat'])}: RMS {_f(r['rms'])} "
          f"chi2/dof {_f(r['chi2_red'], 2)} (n {r['n']})")
    for name, g in r["groups"].items():
        print(f"  {name:<24} n {g['n']}  mean resid {_f(g['mean_residual'])}  mean z "
              f"{_f(g['mean_z'], 1)}  above/below {g['n_above_model']}/{g['n_below_model']}")
    dd = payload["day_to_day"]
    print(f"\nd. {dd['record']}: alpha_r {_f(dd['alpha_r'][JOB1], 2)} -> {_f(dd['alpha_r'][JOB2], 2)}, "
          f"diff {_f(dd['difference'], 2)} [{_f(dd['difference_ci_bootstrap'][0], 2)}, "
          f"{_f(dd['difference_ci_bootstrap'][1], 2)}], z {_f(dd['z'], 1)}")
    print("\ne. layered error")
    for date, day in payload["layered"]["days"].items():
        if day.get("layer_fidelity", True) is None:
            print(f"  {date}: no lf entries")
            continue
        print(f"  {date}: LF_{day['chain_lengths'][1]} {_f(day['lf'], 4)} EPLG {_f(day['eplg'], 5)} "
              f"(measured {day['lf_date']}, {_f(day['lf_age_at_run_h'], 1)} h before the run); "
              f"median CZ device {_f(day['device_cz_median'], 5)}, chain {_f(day['chain_cz_median'], 5)}; "
              f"EPLG/CZ {_f(day['ratio_eplg_to_device_cz'], 2)}; "
              f"alpha_all*CZ {_f(day['alpha_all_x_device_cz'], 5)}, "
              f"alpha_2q*CZ {_f(day['alpha_2q_x_device_cz'], 5)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness of the alpha result (offline).")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--figure-only", type=Path, default=None,
                        help="Redraw the figure from an existing alpha_robustness JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.figure_only:
        payload = _load_json(args.figure_only)
    else:
        payload = build()
        written = save_record(args.out, payload)
        print(f"wrote {rel(written)}")
    for path in make_figure(payload, args.figures_dir):
        print(f"wrote {rel(path)}")
    print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
