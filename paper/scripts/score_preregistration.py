"""
score_preregistration.py

Score a design job against its pre-registration, exactly as declared.

Reads ``paper/hardware/preregistration_<date>.json`` (written and pushed
before the job; its sha256 is printed here and compared with the one
``--submit`` recorded in ``jobs_<date>.json``), the manifest, and the job's
records (``paper/data/hw_<date>/``).  Every threshold, view and verdict rule
comes from the pre-registration's copy of the design's criteria -- none from
code constants -- and every named grid is fitted on its own from the same
counts (``hw_aggregate.grid_fit``, the design's bootstrap).

Q6 needs alpha_r on the job's day: per record, step 4's fit
(``alpha_robustness.alpha_r_rows``) on an alpha-grid cache of this date
(label ``exec_insample`` by default, on the execution calibration).  The
cache is built by ``noise_model_check.py --fit-alpha``; without it Q6 is
"pending" and the exact command is printed.

Also reported, untested: the day's layer fidelity (EPLG, EPLG / median CZ)
from the execution calibration's ``raw_properties``, beside days 1-2.

Writes ``paper/data/scorecard_<date>.json`` and ``.md``: per question the
prediction, the observation, the verdict and the sentence the design maps it
to, and a ``deviations`` list that stays empty unless an entry (with a
reason) is added to the JSON by hand -- re-running keeps those entries.

Run::

    python paper/scripts/score_preregistration.py --date 2026-11-12
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

from _common import DATA_DIR, HARDWARE_DIR, record_provenance, utc_timestamp  # noqa: E402
from dgssp.metrics import proportion_sigma  # noqa: E402
from hardware_batch import (  # noqa: E402
    Paths,
    load_manifest,
    preregistration_path,
    rel,
    resolve,
    sha256_file,
    write_json,
)
from hw_aggregate import grid_rows, load_records  # noqa: E402

SCHEMA = "dgssp/paper/scorecard/1"

#: The alpha cache of the job's own day (step 4's name for an in-sample cache
#: that must not become hw_aggregate's model).
DEFAULT_ALPHA_LABEL = "exec_insample"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Keep(dict):  # type: ignore[type-arg]
    """``str.format_map`` mapping that leaves unknown fields as they are."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def sentence(template: str | None, **values: Any) -> str | None:
    """A design sentence with the observed numbers filled in."""
    if template is None:
        return None
    try:
        return template.format_map(_Keep(values))
    except (ValueError, TypeError):
        return template


def rms(values: Sequence[float]) -> float | None:
    """Root mean square, ``None`` for no values."""
    return float(math.sqrt(sum(v * v for v in values) / len(values))) if values else None


def p_sol(record: Mapping[str, Any] | None) -> float | None:
    """A record's lambda = 1 ``P(solution)``."""
    value = ((record or {}).get("metrics") or {}).get("solution_probability")
    return float(value) if value is not None else None


def pooled(hw: Mapping[str, Mapping[str, Any]], key: str) -> tuple[float | None, int]:
    """lambda = 1 ``P(solution)`` of a record pooled with its drift copy, and the PUB count."""
    values = [v for v in (p_sol(hw.get(key)), p_sol(hw.get(f"{key}_drift"))) if v is not None]
    return (float(np.mean(values)), len(values)) if values else (None, 0)


# ---------------------------------------------------------------------------
# The questions
# ---------------------------------------------------------------------------


def score_q1(c: Mapping[str, Any], pred: Mapping[str, Any], obs: Mapping[str, Any]) -> dict[str, Any]:
    """Does alpha_hat hold: per circuit set, every |P_hw - P_pred| within the threshold."""
    threshold = float(c["threshold_abs"])
    out: dict[str, Any] = {"threshold_abs": threshold, "sets": {}}
    for circuit_set in c["sets"]:
        rows = {k: {"predicted": v["all"], "observed": obs[k],
                    "deviation": obs[k] - v["all"]}
                for k, v in pred["lambda1"].items()
                if v["circuit_set"] == circuit_set and obs.get(k) is not None}
        devs = [r["deviation"] for r in rows.values()]
        n_out = sum(1 for d in devs if abs(d) > threshold)
        verdict = "held" if devs and n_out == 0 else ("not held" if devs else "not tested")
        values = {"threshold_abs": threshold, "rms": rms(devs) or float("nan"),
                  "max_abs": max((abs(d) for d in devs), default=float("nan")),
                  "n_out": n_out, "n": len(devs)}
        out["sets"][circuit_set] = {"records": rows, **values, "verdict": verdict,
                                    "sentence": sentence(c["sentences"].get(
                                        f"{circuit_set}.{verdict}"), **values)}
    return out


def score_q2(c: Mapping[str, Any], pred: Mapping[str, Any], obs: Mapping[str, Any]) -> dict[str, Any]:
    """All errors or CZ only: lambda = 1 RMS of each model, per set, against the margin."""
    margin = float(c["margin"])
    out: dict[str, Any] = {"margin": margin, "primary_set": c["primary_set"], "sets": {}}
    for circuit_set in c["sets"]:
        keys = [k for k, v in pred["lambda1"].items()
                if v["circuit_set"] == circuit_set and obs.get(k) is not None]
        by_model = {m: rms([obs[k] - pred["lambda1"][k][m] for k in keys]) for m in c["models"]}
        decid = pred["q2"]["sets"][circuit_set]
        a, b = c["models"]
        rms_a, rms_b = by_model[a], by_model[b]
        if not decid["decidable"]:
            verdict = "not decidable by design"
        elif rms_a is None or rms_b is None:
            verdict = "not tested"
        elif rms_a <= rms_b - margin:
            verdict = f"{a} better"
        elif rms_b <= rms_a - margin:
            verdict = f"{b} better"
        else:
            verdict = "indistinguishable"
        values = {f"rms_{m}": (v if v is not None else float("nan")) for m, v in by_model.items()}
        values.update(margin=margin,
                      predicted_rms_difference=decid["predicted_rms_difference"] or float("nan"))
        out["sets"][circuit_set] = {
            "records": keys, "rms": by_model, "verdict": verdict,
            "predicted_rms_difference": decid["predicted_rms_difference"],
            "baseline": {m: pred["q2"]["baselines"][m][circuit_set] for m in c["models"]},
            "sentence": sentence(c["sentences"].get(verdict), **values)}
    out["primary_verdict"] = out["sets"][c["primary_set"]]["verdict"]
    order = [c["primary_set"]] + [s for s in out["sets"] if s != c["primary_set"]]
    out["verdict"] = "; ".join(f"{s} {out['sets'][s]['verdict']}" for s in order)
    return out


def score_q3(c: Mapping[str, Any], pred: Mapping[str, Any],
             fits: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    """The grid alone: odd fails and gentle resolves on the same circuit and counts."""
    odd = fits.get((c["record"], c["odd_grid"]))
    gentle = fits.get((c["record"], c["gentle_grid"]))
    if odd is None or gentle is None:
        return {"verdict": "not tested", "predicted": pred.get("q3")}
    ok = odd["status"] in c["odd_statuses"] and gentle["status"] == c["gentle_status"]
    verdict = "confirmed" if ok else "not confirmed"
    return {"predicted": pred.get("q3"), "odd_status": odd["status"],
            "gentle_status": gentle["status"], "verdict": verdict,
            "sentence": sentence(c["sentences"][verdict], odd_status=odd["status"],
                                 gentle_status=gentle["status"])}


def score_q4(c: Mapping[str, Any], pred: Mapping[str, Any],
             fits: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    """The error formula: Q4a (tested), Q4b (reported), Q4c (tested)."""
    parts = c["parts"]
    a = parts["Q4a"]
    lo, hi = (float(x) for x in a["window"])
    rows = []
    for key, grids in pred["fits"].items():
        for gname, p in grids.items():
            f = fits.get((key, gname))
            if f is None:
                continue
            eligible = p["se_pred"] is not None and f["status"] == "resolved"
            ratio_a = (f["se_boot"] / f["se_formula"]
                       if f["se_boot"] and f["se_formula"] else None)
            ratio_b = f["se_boot"] / p["all"]["se_formula_value"] \
                if f["se_boot"] and p["all"]["se_formula_value"] else None
            rows.append({"key": key, "grid": gname, "eligible": eligible,
                         "predicted_relative_se": p["relative_se"],
                         "observed_status": f["status"], "se_boot": f["se_boot"],
                         "se_formula_observed": f["se_formula"], "se_pred": p["se_pred"],
                         "se_pred_value": p["all"]["se_formula_value"],
                         "se_band": p["se_band"], "ratio_q4a": ratio_a, "ratio_q4b": ratio_b,
                         "inside_window": (ratio_a is not None and lo <= ratio_a <= hi)
                         if eligible else None})
    tested = [r for r in rows if r["eligible"]]
    n_out = sum(1 for r in tested if not r["inside_window"])
    verdict_a = "not tested" if not tested else ("held" if n_out == 0 else "not held")
    ratios = ", ".join(f"{r['key']}/{r['grid']} {r['ratio_q4a']:.2f}" for r in tested
                       if r["ratio_q4a"] is not None)
    q4a = {"window": [lo, hi], "rows": rows, "n": len(tested), "n_out": n_out,
           "verdict": verdict_a,
           "sentence": sentence(c["sentences"].get(f"Q4a.{verdict_a}"), ratios=ratios or "-",
                                n=len(tested), n_out=n_out,
                                eligible_max_predicted_relative_se=a[
                                    "eligible_max_predicted_relative_se"])}

    g = parts["Q4c"]
    num = fits.get((g["record"], g["numerator_grid"]))
    den = fits.get((g["record"], g["denominator_grid"]))
    predicted = (pred.get("q4c") or {}).get("ratio")
    reason = None
    if num is None or den is None:
        reason = "a grid was not run"
    elif num["status"] != "resolved" or den["status"] != "resolved":
        reason = f"not both resolved ({num['status']}, {den['status']})"
    elif not num["se_boot"] or not den["se_boot"] or not predicted:
        reason = "no bootstrap SE or no predicted ratio"
    if reason is not None:
        q4c = {"verdict": "not tested", "reason": reason, "predicted": predicted,
               "sentence": sentence(c["sentences"]["Q4c.not tested"], reason=reason)}
    else:
        assert num is not None and den is not None and predicted
        observed = num["se_boot"] / den["se_boot"]
        verdict_c = "held" if abs(observed / predicted - 1) <= float(g["tolerance_rel"]) \
            else "not held"
        q4c = {"observed": observed, "predicted": predicted,
               "tolerance_rel": g["tolerance_rel"], "verdict": verdict_c,
               "sentence": sentence(c["sentences"][f"Q4c.{verdict_c}"], observed=observed,
                                    predicted=predicted)}
    return {"Q4a": q4a, "Q4b": {"tested": False, "rows": [
        {k: r[k] for k in ("key", "grid", "ratio_q4b", "se_pred_value", "se_band", "se_boot")}
        for r in rows]}, "Q4c": q4c, "verdict": f"Q4a {verdict_a}; Q4c {q4c['verdict']}"}


def score_q5(c: Mapping[str, Any], pred: Mapping[str, Any],
             fits: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    """Predicted vs observed status per record x grid; matches against their expectation."""
    rows = []
    for key, grids in pred["fits"].items():
        for gname, p in grids.items():
            f = fits.get((key, gname))
            if f is None:
                continue
            rows.append({"key": key, "grid": gname, "predicted": p["status"],
                         "p_predicted": p["status_probability"][p["status"]],
                         "observed": f["status"],
                         "p_observed": p["status_probability"].get(f["status"]),
                         "match": p["status"] == f["status"]})
    matches = sum(1 for r in rows if r["match"])
    expected = sum(r["p_predicted"] for r in rows)
    return {"rows": rows, "matches": matches, "n": len(rows), "expected": expected,
            "verdict": "reported",
            "sentence": sentence(c["sentences"]["reported"], matches=matches, n=len(rows),
                                 expected=expected)}


def score_q6(c: Mapping[str, Any], pred: Mapping[str, Any],
             day3: Sequence[Mapping[str, Any]] | None, pending: str | None) -> dict[str, Any]:
    """alpha_r a circuit property: differences after a shift per reference day, and scatter."""
    refs = pred["q6"]["references"]
    if day3 is None:
        return {"verdict": "pending", "how": pending, "references": refs}
    by_key = {r["key"]: r for r in day3}
    rows = []
    for key, ref in refs.items():
        now = by_key.get(key)
        if ref is None or now is None:
            continue
        excluded = ([f"reference: {x}" for x in ref["excluded_because"]]
                    + [f"day 3: {x}" for x in now["excluded_because"]])
        rows.append({"key": key, "day": ref["day"], "alpha_ref": ref["alpha_r"],
                     "se_ref": ref["alpha_r_se"], "alpha_day3": now["alpha_r"],
                     "se_day3": now["alpha_r_se"], "difference": now["alpha_r"] - ref["alpha_r"],
                     "excluded_because": excluded, "included": not excluded})
    used = [r for r in rows if r["included"]]
    days = sorted({r["day"] for r in used})
    chi2_d, shifts = 0.0, {}
    for day in days:
        group = [r for r in used if r["day"] == day]
        w = np.array([1.0 / (r["se_ref"] ** 2 + r["se_day3"] ** 2) for r in group])
        d = np.array([r["difference"] for r in group])
        shift = float(np.sum(w * d) / np.sum(w))
        shifts[day] = {"shift": shift, "se": float(1 / math.sqrt(np.sum(w))), "n": len(group)}
        chi2_d += float(np.sum(w * (d - shift) ** 2))
    dof_d = len(used) - len(days)
    w3 = np.array([1.0 / r["se_day3"] ** 2 for r in used]) if used else np.array([])
    a3 = np.array([r["alpha_day3"] for r in used]) if used else np.array([])
    mean3 = float(np.sum(w3 * a3) / np.sum(w3)) if used else float("nan")
    chi2_b = float(np.sum(w3 * (a3 - mean3) ** 2)) if used else float("nan")
    dof_b = len(used) - 1
    red_d = chi2_d / dof_d if dof_d > 0 else None
    red_b = chi2_b / dof_b if dof_b > 0 else None
    spearman = None
    if len(used) >= 3:
        from scipy.stats import spearmanr

        result = spearmanr([r["alpha_ref"] for r in used], [r["alpha_day3"] for r in used])
        spearman = {"rho": float(result.statistic), "p": float(result.pvalue)}
    if red_d is None or red_b is None:
        verdict = "not tested"
    elif red_d > float(c["max_chi2_red_differences"]):
        verdict = "not reproducible"
    elif red_b > float(c["min_chi2_red_between_day3"]):
        verdict = "reproducible per circuit"
    else:
        verdict = "no scatter to reproduce"
    shift_text = ", ".join(f"{v['shift']:+.2f} (vs {d})" for d, v in shifts.items()) or "-"
    return {"rows": rows, "n": len(used), "shifts": shifts, "chi2_differences": chi2_d,
            "dof_differences": dof_d, "chi2_red_differences": red_d,
            "weighted_mean_day3": mean3, "chi2_between_day3": chi2_b, "dof_between_day3": dof_b,
            "chi2_red_between": red_b, "spearman": spearman, "verdict": verdict,
            "sentence": sentence(c["sentences"].get(verdict), n=len(used),
                                 chi2_red_differences=red_d if red_d is not None else float("nan"),
                                 chi2_red_between=red_b if red_b is not None else float("nan"),
                                 shifts=shift_text)}


def score_q7(c: Mapping[str, Any], pred: Mapping[str, Any],
             hw: Mapping[str, Mapping[str, Any]], shots: int) -> dict[str, Any]:
    """P-n3-M2-17 on both layouts: random / degraded / fine, and the map's sentence."""
    k, fine = float(c["random_k_sigma"]), float(c["fine_threshold"])
    layouts: dict[str, Any] = {}
    for layout, key in c["records"].items():
        p_pred = pred["q7"].get(layout)
        p_hw, n = pooled(hw, key)
        if p_pred is None or p_hw is None:
            layouts[layout] = {"category": "not run"}
            continue
        sigma = proportion_sigma(p_hw, shots * n)
        if not p_pred["decidable"]:
            category = "calibration predicts it"
        elif p_hw < p_pred["p_uniform"] + k * sigma:
            category = "random"
        elif p_hw >= p_pred["p_pred"] - fine:
            category = "fine"
        else:
            category = "degraded"
        layouts[layout] = {"record": key, "observed": p_hw, "n_pubs": n, "sigma": sigma,
                           "predicted": p_pred["p_pred"], "p_uniform": p_pred["p_uniform"],
                           "z_above_random": (p_hw - p_pred["p_uniform"]) / sigma,
                           "category": category}
    cell = f"L2={layouts.get('L2', {}).get('category')},L1={layouts.get('L1', {}).get('category')}"
    text = c["map"].get(cell)
    return {"layouts": layouts, "cell": cell, "verdict": cell if text else "outside the map",
            "sentence": text}


def score_q8(c: Mapping[str, Any], pred: Mapping[str, Any], hw: Mapping[str, Mapping[str, Any]],
             fits: Mapping[tuple[str, str], Mapping[str, Any]],
             trimmed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Gentle-grid ZNE at n = 4: resolved or not, and whether the interval holds the ideal."""
    record = c["record"]
    if any(t["record"] == record for t in trimmed) or (record, c["grid"]) not in fits:
        return {"verdict": "not tested", "predicted": pred.get("q8"),
                "sentence": sentence(c["sentences"]["not tested"], record=record)}
    f = fits[(record, c["grid"])]
    verdict = "reaches" if f["status"] == "resolved" else "does not reach"
    covers = (f["reported_lo"] <= f["p_ideal"] <= f["reported_hi"]
              if verdict == "reaches" else None)
    return {"predicted": pred.get("q8"), "status": f["status"], "reported_value":
            f["reported_value"], "reported_ci": [f["reported_lo"], f["reported_hi"]],
            "p_ideal": f["p_ideal"], "interval_contains_ideal": covers, "verdict": verdict,
            "sentence": sentence(c["sentences"][verdict], value=f["reported_value"],
                                 lo=f["reported_lo"] or float("nan"),
                                 hi=f["reported_hi"] or float("nan"), p_ideal=f["p_ideal"],
                                 status=f["status"], p1=p_sol(hw.get(record)) or float("nan"))}


def score_q9(c: Mapping[str, Any], pred: Mapping[str, Any], hw: Mapping[str, Mapping[str, Any]],
             shots: int) -> dict[str, Any]:
    """The P-n4-M1-10_r2 residual at lambda = 1, day 3 beside day 1 (reported)."""
    day1 = pred["q9"]["day1"]
    p_hw, n = pooled(hw, c["record"])
    p_pred = pred["q9"]["predicted_lambda1"]
    if p_hw is None or p_pred is None:
        return {"verdict": "not run", "day1": day1}
    residual = p_hw - p_pred
    z = residual / proportion_sigma(p_hw, shots * n)
    return {"day1": day1, "observed": p_hw, "predicted": p_pred, "residual": residual, "z": z,
            "verdict": "reported",
            "sentence": sentence(c["sentences"]["reported"], day1=day1["residual"],
                                 day1_z=day1["z"], day3=residual, day3_z=z)}


# ---------------------------------------------------------------------------
# Untested: layer fidelity
# ---------------------------------------------------------------------------


def layered(args: argparse.Namespace, prereg: Mapping[str, Any]) -> dict[str, Any]:
    """Day-3 EPLG and EPLG / median CZ beside days 1-2 (alpha_robustness.json)."""
    from alpha_robustness import layered_day

    alphas = prereg["alphas"]
    robustness = json.loads(resolve(alphas["all"]["file"]).read_text(encoding="utf-8"))
    keys = ("eplg", "ratio_eplg_to_device_cz", "device_cz_median", "lf_age_at_run_h")
    days = {d: {k: v.get(k) for k in keys}
            for d, v in robustness["layered"]["days"].items()}
    calibration = args.hardware_dir / f"calibration_exec_{args.date}.json"
    if not calibration.exists():
        day3: Any = (f"not available: {rel(calibration)} missing (noise_model_check.py "
                     f"--fetch-calibration --date {args.date})")
    else:
        full = layered_day(args.date, float(alphas["all"]["value"]),
                           float(alphas["2q"]["value"]) if "2q" in alphas else None,
                           hardware_dir=args.hardware_dir, data_dir=args.data_dir)
        day3 = {k: full.get(k) for k in keys}
    return {"tested": False, "days_1_2": days, "day_3": day3}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def day3_alpha_rows(args: argparse.Namespace, prereg: Mapping[str, Any],
                    k: float) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Q6's day-3 alpha_r per record, or why not yet (with the command that builds it)."""
    from alpha_robustness import alpha_r_rows, load_job
    from noise_model_check import alpha_cache_dir

    cache = alpha_cache_dir(args.data_dir, args.alpha_label, "all", args.date)
    calibration = args.hardware_dir / f"calibration_exec_{args.date}.json"
    command = ("python paper/scripts/noise_model_check.py --fit-alpha --date "
               f"{args.date} --label {args.alpha_label} --calibration {rel(calibration)} "
               "--alphas " + " ".join(f"{a:g}" for a in prereg["alpha_r_grid"]))
    if not (cache / "fit.json").exists():
        return None, f"{rel(cache / 'fit.json')} missing: {command}"
    job = load_job(args.date, label=args.alpha_label, hardware_dir=args.hardware_dir,
                   data_dir=args.data_dir)
    return alpha_r_rows(job, random_k=k), None


def build(args: argparse.Namespace) -> dict[str, Any]:
    """The scorecard."""
    paths = Paths(args.date, args.hardware_dir, args.data_dir)
    prereg_file = preregistration_path(paths)
    prereg = json.loads(prereg_file.read_text(encoding="utf-8"))
    prereg_sha = sha256_file(prereg_file)
    manifest, manifest_sha, _ = load_manifest(paths.manifest)
    if prereg["manifest_sha256"] != manifest_sha:
        raise SystemExit("The pre-registration was made from another manifest.")
    at_submit = None
    if paths.jobs.exists():
        at_submit = (json.loads(paths.jobs.read_text(encoding="utf-8")).get("preregistration")
                     or {}).get("sha256")
    records_dir = Path(args.records_dir) if args.records_dir else paths.hardware_out_dir
    hw = load_records(records_dir)
    if not hw:
        raise SystemExit(f"no records in {rel(records_dir)}.")
    criteria = {c["id"]: c for c in prereg["criteria"]}
    pred = prereg["predictions"]
    shots = int(prereg["shots"])
    boot = prereg["design"].get("grid_fit_bootstrap") or {}
    rows = grid_rows(manifest, hw, report_bounds=prereg["report_bounds"], B=boot.get("B"),
                     seed=boot.get("seed"), predicted=pred["fits"])
    fits = {(r["key"], r["grid_name"]): r for r in rows}
    obs = {k: pooled(hw, k)[0] for k in pred["lambda1"]}
    day3, pending = day3_alpha_rows(args, prereg, float(criteria["Q6"]["random_k_sigma"]))

    questions = {
        "Q1": score_q1(criteria["Q1"], pred, obs),
        "Q2": score_q2(criteria["Q2"], pred, obs),
        "Q3": score_q3(criteria["Q3"], pred, fits),
        "Q4": score_q4(criteria["Q4"], pred, fits),
        "Q5": score_q5(criteria["Q5"], pred, fits),
        "Q6": score_q6(criteria["Q6"], pred, day3, pending),
        "Q7": score_q7(criteria["Q7"], pred, hw, shots),
        "Q8": score_q8(criteria["Q8"], pred, hw, fits, prereg.get("trimmed") or []),
        "Q9": score_q9(criteria["Q9"], pred, hw, shots),
    }
    questions["Q1"]["verdict"] = "; ".join(f"{s} {v['verdict']}"
                                           for s, v in questions["Q1"]["sets"].items())
    for cid, q in questions.items():
        q["question"] = criteria[cid]["question"]
        q["tested"] = criteria[cid].get("tested")

    out_json = args.out_dir / f"scorecard_{args.date}.json"
    old = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    return {
        "schema": SCHEMA,
        "date": args.date,
        "created_utc": utc_timestamp(),
        "preregistration": {"file": rel(prereg_file), "sha256": prereg_sha,
                            "sha256_recorded_at_submit": at_submit,
                            "matches_submit": None if at_submit is None
                            else at_submit == prereg_sha},
        "manifest_sha256": manifest_sha,
        "records_dir": rel(records_dir),
        "executor": sorted({str(r.get("executor")) for r in hw.values()}),
        "report_bounds": prereg["report_bounds"],
        "grid_fits": rows,
        "alpha_r_day3": day3,
        "questions": questions,
        "reported_untested": {"layer_fidelity": layered(args, prereg)},
        "deviations": old.get("deviations", []),
        "provenance": record_provenance(),
    }


def _f(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(card: Mapping[str, Any]) -> str:
    """The scorecard as a note: verdicts first, then what each rests on."""
    q = card["questions"]
    lines = [
        f"# Scorecard — {card['date']}",
        "",
        "Written by `paper/scripts/score_preregistration.py` from the pre-registration "
        f"`{card['preregistration']['file']}` (sha256 `{card['preregistration']['sha256']}`; "
        f"recorded at submission: {card['preregistration']['sha256_recorded_at_submit']}) and "
        f"the records in `{card['records_dir']}` (executor {', '.join(card['executor'])}).",
        "",
        "| ID | Question | Tested | Verdict | Sentence |",
        "|---|---|---|---|---|",
    ]
    for cid, v in q.items():
        text = v.get("sentence")
        if cid == "Q1":
            text = " ".join(s["sentence"] or "" for s in v["sets"].values())
        if cid == "Q2":
            text = " ".join(f"[{s}] {x['sentence']}" for s, x in v["sets"].items())
        if cid == "Q4":
            text = f"{v['Q4a']['sentence']} {v['Q4c']['sentence']}"
        cells = [cid, v["question"], "yes" if v.get("tested") else "reported",
                 v.get("verdict"), text or v.get("how") or "-"]
        lines.append("| " + " | ".join(str(c).replace("|", r"\|") for c in cells) + " |")
    lines += ["", "## Named grids, each fitted on the same counts", "",
              "| Record | Grid | Status (pred.) | Reported [CI] | SE_boot | SE_formula | SE_pred |",
              "|---|---|---|---|---|---|---|"]
    for r in card["grid_fits"]:
        lines.append(f"| {r['key']} | {r['grid_name']} | {r['status']} ({r['predicted_status']}) "
                     f"| {_f(r['reported_value'])} [{_f(r['reported_lo'])}, {_f(r['reported_hi'])}]"
                     f" | {_f(r['se_boot'])} | {_f(r['se_formula'])} | {_f(r['se_pred'])} |")
    lines += ["", "## lambda = 1 against the prediction (Q1, Q2)", "",
              "| Record | Set | Observed | Pred. all | Dev. all |", "|---|---|---|---|---|"]
    for s, v in q["Q1"]["sets"].items():
        for key, r in v["records"].items():
            lines.append(f"| {key} | {s} | {_f(r['observed'])} | {_f(r['predicted'])} | "
                         f"{r['deviation']:+.3f} |")
    q6 = q["Q6"]
    if q6["verdict"] != "pending":
        lines += ["", "## alpha_r (Q6)", "", "| Record | Ref. day | alpha_r ref | alpha_r day 3 "
                  "| Difference | Included |", "|---|---|---|---|---|---|"]
        for r in q6["rows"]:
            lines.append(f"| {r['key']} | {r['day']} | {_f(r['alpha_ref'], 2)}±{_f(r['se_ref'], 2)}"
                         f" | {_f(r['alpha_day3'], 2)}±{_f(r['se_day3'], 2)} | "
                         f"{r['difference']:+.2f} | "
                         f"{'yes' if r['included'] else '; '.join(r['excluded_because'])} |")
    q7 = q["Q7"]
    lines += ["", f"Q7 cell `{q7['cell']}`: " + ", ".join(
        f"{lay} {v.get('category')} (P {_f(v.get('observed'))} vs pred. {_f(v.get('predicted'))}, "
        f"{_f(v.get('z_above_random'), 1)} sigma above random)" for lay, v in q7["layouts"].items())]
    lf = card["reported_untested"]["layer_fidelity"]
    lines += ["", "## Reported, not tested: layer fidelity", "",
              "Days 1-2: " + "; ".join(f"{d} EPLG {_f(v['eplg'], 5)}, EPLG / median CZ "
                                      f"{_f(v['ratio_eplg_to_device_cz'], 2)}"
                                      for d, v in lf["days_1_2"].items())
              + ". Day 3: " + (lf["day_3"] if isinstance(lf["day_3"], str) else
                               f"EPLG {_f(lf['day_3']['eplg'], 5)}, EPLG / median CZ "
                               f"{_f(lf['day_3']['ratio_eplg_to_device_cz'], 2)}") + "."]
    lines += ["", "## Deviations", ""]
    lines += [f"- {d}" for d in card["deviations"]] or ["None."]
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Score a design job against its "
                                                 "pre-registration.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--records-dir", default=None,
                        help="Records to score (default paper/data/hw_<date>).")
    parser.add_argument("--alpha-label", default=DEFAULT_ALPHA_LABEL,
                        help="Label of the day's alpha cache (Q6).")
    parser.add_argument("--hardware-dir", type=Path, default=HARDWARE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where scorecard_<date>.{json,md} go (default --data-dir).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Score, write, print."""
    args = parse_args(argv)
    args.out_dir = args.out_dir or args.data_dir
    card = build(args)
    out = write_json(args.out_dir / f"scorecard_{args.date}.json", card)
    markdown = render_markdown(card)
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(f"pre-registration sha256 {card['preregistration']['sha256']}"
          + ("" if card["preregistration"]["matches_submit"] is None else
             f" (recorded at submit: {'MATCHES' if card['preregistration']['matches_submit'] else 'DIFFERS'})"))
    print(markdown)
    print(f"wrote {rel(out)}\n      {rel(out.with_suffix('.md'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
