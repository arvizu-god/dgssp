"""
preregister.py

Freeze, before a design job is submitted, what every pre-declared question
(Q1-Q9 of its design file) predicts.  Offline: it simulates on the plan's
calibration snapshot and never contacts the service.

For the manifest ``paper/hardware/manifest_<date>.json`` (made by
``hardware_batch.py --plan --design <file>``):

* every PUB is simulated with the full noise model of the plan calibration,
  every error times each named alpha of the design (``all``: alpha_hat;
  ``2q``: the CZ-only fit, ``--scale-only 2q``), through
  ``noise_model_check.py --predict`` -- so the prediction records land in
  ``paper/data/hw_predict_plan_x<alpha>[_2q]_<date>/`` exactly as job 2's did
  -- and at the two alpha_r quantiles of the design's ``alpha_r_band`` (Q4b's
  band; in memory only, so ``hw_aggregate.py`` never mistakes them for the
  job's model);
* per record x named grid: predicted ``P(lambda)``; the predicted status (the
  gate on the expected values at the planned shots) and its probability (a
  seeded parametric bootstrap of binomial counts); the predicted reported
  value under ``REPORT_BOUNDS``; the predicted SE (unclipped,
  :func:`dgssp.mitigation.zne.exp_fixed_se` on the expected values) where
  its relative SE is at most the design's cut, "formula not applicable" with
  the value elsewhere; the SE band at the alpha_r quantiles; and the
  gentle/rule SE ratio of Q4c;
* the references the questions are scored against: day-1/day-2 alpha_r of
  every pinned circuit (Q6), each model's lambda = 1 RMS on the same circuits
  on their own day (Q2), the day-1 residual of Q9, and Q2/Q7's decidability
  as the design defines it.

Writes ``paper/hardware/preregistration_<date>.json`` (the design's criteria
copied verbatim, the manifest's sha256, the calibration stamp, the code
commit and every prediction) and ``preregistration_<date>.md`` (the Q1-Q9
table and the outcome -> sentence map), and prints the JSON's sha256.  An
existing pre-registration is never replaced (``--force`` before it is
committed, if the plan was redone).

Run::

    python paper/scripts/preregister.py --date 2026-11-12
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
    HARDWARE_DIR,
    git_state,
    record_provenance,
    utc_timestamp,
)
from dgssp.execution import exact_index_distribution  # noqa: E402
from dgssp.instance import SubsetSumInstance  # noqa: E402
from dgssp.metrics import (  # noqa: E402
    distribution_solution_probability,
    proportion_sigma,
    solution_probability,
)
from dgssp.mitigation.zne import (  # noqa: E402
    REPORT_BOUNDS,
    RESOLVED_RULE,
    exp_fixed_se,
    report_extrapolation,
    scalar_flags,
)
from hardware_batch import (  # noqa: E402
    PREREG_SCHEMA,
    Paths,
    load_design,
    load_manifest,
    parse_pin,
    preregistration_path,
    rel,
    resolve,
    sha256_file,
    write_json,
)
from hw_aggregate import alpha_predictions  # noqa: E402

#: Normal quantile of the predicted 95 % interval (as the bootstrap's).
Z95 = 1.959963984540054

#: Status words, as ``hw_aggregate.status_of`` gives them.
RESOLVED, IDENTIFIED, NOT_IDENTIFIABLE = "resolved", "identified, not resolved", "not identifiable"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def criteria_by_id(design: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The design's criteria keyed by id (``Q1`` ...)."""
    return {c["id"]: dict(c) for c in design["criteria"]}


def _node(spec: Mapping[str, Any], key: str) -> Any:
    """``spec[key]`` (a key path) of the JSON file ``spec["file"]``."""
    node: Any = json.loads(resolve(spec["file"]).read_text(encoding="utf-8"))
    for part in spec[key]:
        node = node[part]
    return node


def band_alphas(design: Mapping[str, Any]) -> dict[str, Any]:
    """Q4b's alpha_r quantiles, from the rows the design names."""
    spec = design["alpha_r_band"]
    rows = _node(spec, "rows")
    values = [float(r["alpha_r"]) for r in rows
              if r.get("in_trend") or not spec.get("in_trend_only", True)]
    q = [float(x) for x in spec["quantiles"]]
    lo, hi = (float(v) for v in np.quantile(values, q))
    return {"file": spec["file"], "rows": list(spec["rows"]), "n": len(values),
            "quantiles": q, "alphas": [lo, hi]}


def record_info(row: Mapping[str, Any]) -> dict[str, Any]:
    """``p_ideal`` (exact) and ``p_uniform`` of a manifest record."""
    instance = SubsetSumInstance(items=list(row["items"]), target=row["target"],
                                 name=row["instance"])
    reference = exact_index_distribution(instance, int(row["iterations"]))
    return {"p_ideal": distribution_solution_probability(reference,
                                                         list(row["solution_bitstrings"])),
            "p_uniform": float(row["p_uniform"])}


def simulate_alpha(
    manifest: Mapping[str, Any],
    circuits: Sequence[Any],
    calibration: Path,
    alpha: float,
    which: str,
    max_threads: int | None,
) -> list[dict[str, int]]:
    """The manifest's PUBs on the plan calibration with the ``which`` errors x ``alpha``."""
    from noise_model_check import noise_device, simulate

    _, device, _ = noise_device(calibration, alpha, which)  # type: ignore[arg-type]
    counts, _ = simulate(manifest, circuits, device, max_threads=max_threads,
                         meta={"stage": "preregistration", "alpha": alpha, "which": which})
    return counts


def pub_probabilities(
    manifest: Mapping[str, Any], counts: Sequence[Mapping[str, int]]
) -> dict[str, dict[float, float]]:
    """``{record key: {nominal scale: P(solution)}}`` from per-PUB counts."""
    out: dict[str, dict[float, float]] = {}
    for row in manifest["records"]:
        sols = list(row["solution_bitstrings"])
        out[row["key"]] = {
            float(scale): solution_probability(counts[int(pub)], sols)
            for scale, pub in zip(row["nominal_scales"], row["pub_indices"], strict=True)
        }
    return out


def realised_scales(manifest: Mapping[str, Any]) -> dict[tuple[str, float], float]:
    """``{(record key, nominal): realised scale}`` from the manifest's circuit rows."""
    return {(c["key"], float(c["nominal_scale"])): float(c["realised_scale"])
            for c in manifest["circuits"]}


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def status_from(flags: Mapping[str, Any]) -> str:
    """A status word from :func:`scalar_flags` (fit assumed converged)."""
    if flags["use_fit"]:
        return RESOLVED
    return IDENTIFIED if flags["identifiable"] else NOT_IDENTIFIABLE


def predicted_fit(
    realised: Sequence[float],
    p: Sequence[float],
    shots: int,
    info: Mapping[str, Any],
    *,
    max_relative_se: float,
) -> dict[str, Any]:
    """
    One grid's prediction at one alpha: status, value, SE, reported value.

    The gate is applied to the expected values (``P(lambda)`` as simulated)
    at the planned shots; the fit and its delta-method SE are those values'
    (:func:`exp_fixed_se`).  The SE is stated only where its relative SE is
    at most ``max_relative_se`` (zne_theory.md section 7: the formula holds
    there and runs low above it); elsewhere "formula not applicable" with the
    value, for the record.
    """
    p_u, p_ideal = float(info["p_uniform"]), float(info["p_ideal"])
    flags = scalar_flags(list(p), [shots] * len(p), p_u, n_params=2, fit_ok=True,
                         p_ideal=p_ideal)
    status = status_from(flags)
    fit = exp_fixed_se(realised, p, [shots] * len(p), p_u)
    se = fit["se"] if math.isfinite(fit["se"]) else None
    excess = fit["value"] - p_u
    relative = se / abs(excess) if se is not None and excess else None
    applicable = relative is not None and relative <= max_relative_se
    if status == RESOLVED and se is not None:
        report = report_extrapolation(float(fit["value"]),
                                      [fit["value"] - Z95 * se, fit["value"] + Z95 * se],
                                      p_u, p_ideal, option=REPORT_BOUNDS)
        reported, reported_ci = report["value"], report["ci"]
    else:
        reported, reported_ci = float(p[0]), None
    return {
        "p": [float(x) for x in p],
        "status": status,
        "value": float(fit["value"]),
        "rate": float(fit["rate"]),
        "se_formula_value": se,
        "relative_se": relative,
        "se_pred": se if applicable else None,
        "se_note": None if applicable else (
            f"formula not applicable (predicted relative SE {relative:.3f} > {max_relative_se})"
            if relative is not None else "formula not applicable (fit not identified)"),
        "reported_value": reported,
        "reported_ci": reported_ci,
        "report_bounds": REPORT_BOUNDS,
    }


def status_probabilities(
    record_p: Mapping[float, float],
    grids: Mapping[str, Sequence[float]],
    shots: int,
    info: Mapping[str, Any],
    *,
    B: int,
    seed: int,
) -> dict[str, Any]:
    """
    P(status) of every named grid of a record, from the same draws.

    Each replicate draws binomial counts at the predicted ``P(lambda)`` of
    every scale the record runs, once, and every grid's gate is applied to
    those draws -- so joint events (Q3: odd fails *and* gentle resolves) come
    out of the same replicates, as on hardware, where the grids share
    lambda = 1.
    """
    rng = np.random.default_rng(seed)
    scales = sorted(record_p)
    draws = rng.binomial(shots, [record_p[s] for s in scales], size=(B, len(scales))) / shots
    col = {s: i for i, s in enumerate(scales)}
    p_u, p_ideal = float(info["p_uniform"]), float(info["p_ideal"])
    statuses: dict[str, list[str]] = {}
    for name, grid in grids.items():
        cols = [col[float(s)] for s in grid]
        statuses[name] = [
            status_from(scalar_flags(draws[b, cols].tolist(), [shots] * len(cols), p_u,
                                     n_params=2, fit_ok=True, p_ideal=p_ideal))
            for b in range(B)
        ]
    per_grid = {
        name: {s: sum(1 for x in st if x == s) / B for s in (RESOLVED, IDENTIFIED,
                                                             NOT_IDENTIFIABLE)}
        for name, st in statuses.items()
    }
    return {"B": B, "seed": seed, "per_grid": per_grid, "_draws": statuses}


def pooled_p(probs: Mapping[str, Mapping[float, float]], key: str) -> float:
    """lambda = 1 ``P(solution)`` of a record pooled with its drift copy (equal shots)."""
    values = [probs[key][1.0]]
    if f"{key}_drift" in probs:
        values.append(probs[f"{key}_drift"][1.0])
    return float(np.mean(values))


def rms(values: Sequence[float]) -> float | None:
    """Root mean square, ``None`` for no values."""
    return float(math.sqrt(sum(v * v for v in values) / len(values))) if values else None


def day_baselines(design_records: Sequence[Mapping[str, Any]],
                  alphas: Mapping[str, Mapping[str, Any]],
                  sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """
    Q2's baselines: each model's lambda = 1 RMS on the same circuits on their own day.

    ``fitted`` (job-1 circuits): day 1, from the job-1 alpha cache of each
    error class at that class's alpha (in-sample).  ``out_of_sample`` (job-2
    circuits): day 2, the prediction made before job 2.  Both files are the
    design's (``Q2.baselines``); records are pooled with their drift copy,
    as on day 3.
    """
    out: dict[str, Any] = {}
    for name, alpha in alphas.items():
        which = alpha["scale_only"]
        suffix = "" if which == "all" else f"_{which}"
        fit_doc = json.loads(resolve(sources["fitted"]["fit_file"].format(suffix=suffix))
                             .read_text(encoding="utf-8"))
        oos = sources["out_of_sample"]
        oos_rows = _node({"file": oos["file"],
                          "path": [part.format(which=which) for part in oos["path"]]}, "path")
        pred = alpha_predictions(fit_doc, float(alpha["value"]))
        hw = {(r["key"], float(r["scale"])): float(r["hardware"]) for r in fit_doc["per_pub"]}
        per_set: dict[str, Any] = {}
        for circuit_set in ("fitted", "out_of_sample"):
            devs = {}
            for record in design_records:
                if record.get("circuit_set") != circuit_set:
                    continue
                pin = parse_pin(record["source"])
                base = pin["key"]
                if circuit_set == "fitted":
                    keys = [k for k in (base, f"{base}_drift") if (k, 1.0) in hw]
                    devs[record["key"]] = float(np.mean([hw[(k, 1.0)] - pred[(k, 1.0)]
                                                         for k in keys]))
                else:
                    vals = [r["deviation"] for r in oos_rows
                            if r["key"] in (base, f"{base}_drift")]
                    devs[record["key"]] = float(np.mean(vals))
            per_set[circuit_set] = {"deviations": devs, "rms": rms(list(devs.values())),
                                    "day": sources[circuit_set]["day"]}
        out[name] = per_set
    return out


def q6_references(design_records: Sequence[Mapping[str, Any]],
                  spec: Mapping[str, Any]) -> dict[str, Any]:
    """Each pinned circuit's alpha_r on its own day (step 4), with the exclusion flags."""
    rows = {r["id"]: r for r in _node(spec, "rows")}
    out = {}
    for record in design_records:
        pin = parse_pin(record["source"])
        ref = rows.get(f"{pin['key']}@{pin['date']}")
        out[record["key"]] = None if ref is None else {
            "id": ref["id"], "day": pin["date"], "alpha_r": ref["alpha_r"],
            "alpha_r_se": ref["alpha_r_se"], "alpha_r_ci": ref["alpha_r_ci"],
            "alpha_r_ci_scaled": ref["alpha_r_ci_scaled"],
            "excluded_because": ref["excluded_because"], "in_trend": ref["in_trend"]}
    return out


def q9_reference(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The day-1 lambda = 1 residual of the Q9 record (alpha_robustness.json residuals)."""
    rows = _node(spec, "rows")
    row = next(r for r in rows if r["key"] == spec["key"])
    return {"key": row["key"], "pub": row["pub"], "residual": row["residual"], "z": row["z"],
            "sigma": row["sigma"], "file": spec["file"]}


def build(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """Every prediction, and the Markdown that reads them."""
    paths = Paths(args.date, args.hardware_dir, args.data_dir)
    manifest, manifest_sha, circuits = load_manifest(paths.manifest)
    block = manifest.get("design")
    if not block:
        raise SystemExit(f"{rel(paths.manifest)} was not planned from a design file.")
    design_path = resolve(block["file"])
    if sha256_file(design_path) != block["sha256"]:
        raise SystemExit(f"{rel(design_path)} changed after --plan; re-plan.")
    design = load_design(design_path)
    criteria = criteria_by_id(design)
    calibration = resolve(manifest["backend"]["calibration_file"])
    shots = int(manifest["settings"]["shots"])
    alphas = block["alphas"]
    band = band_alphas(design)

    # The prediction records of each named alpha (as job 2's were made), then
    # every alpha's per-PUB counts for the tables below (same simulator, same seed).
    import noise_model_check as nmc

    prediction_dirs = {}
    for name, alpha in alphas.items():
        argv = ["--predict", "--date", args.date, "--calibration", str(calibration),
                "--alpha", repr(float(alpha["value"])), "--scale-only", alpha["scale_only"],
                "--hardware-dir", str(args.hardware_dir), "--data-dir", str(args.data_dir)]
        if args.max_threads:
            argv += ["--max-threads", str(args.max_threads)]
        if nmc.main(argv) != 0:
            raise SystemExit(f"noise_model_check.py --predict failed for alpha {name}.")
        suffix = "" if alpha["scale_only"] == "all" else f"_{alpha['scale_only']}"
        label = f"{nmc.calibration_label(calibration, args.date)}_x{float(alpha['value']):g}{suffix}"
        directory = args.data_dir / f"hw_predict_{label}_{args.date}"
        prediction_dirs[name] = {"dir": rel(directory), "records": {
            rel(p): sha256_file(p) for p in sorted(directory.glob("*.json"))}}

    probs: dict[str, dict[str, dict[float, float]]] = {}
    for name, alpha in alphas.items():
        print(f"simulating {len(circuits)} PUBs at {name} = {alpha['value']:g}", flush=True)
        probs[name] = pub_probabilities(manifest, simulate_alpha(
            manifest, circuits, calibration, float(alpha["value"]), alpha["scale_only"],
            args.max_threads))
    for i, value in enumerate(band["alphas"]):
        name = f"band_{'lo' if i == 0 else 'hi'}"
        print(f"simulating {len(circuits)} PUBs at {name} = {value:g}", flush=True)
        probs[name] = pub_probabilities(manifest, simulate_alpha(
            manifest, circuits, calibration, value, "all", args.max_threads))

    realised = realised_scales(manifest)
    records = {r["key"]: r for r in manifest["records"]}
    info = {k: record_info(r) for k, r in records.items()}
    q4 = criteria["Q4"]["parts"]
    cut = float(q4["Q4a"]["eligible_max_predicted_relative_se"])
    q5 = criteria["Q5"]["bootstrap"]

    fits: dict[str, dict[str, Any]] = {}
    joint: dict[str, Any] = {}
    for key, row in records.items():
        grids = row.get("grids") or {}
        if not grids:
            continue
        fits[key] = {}
        for gname, grid in grids.items():
            real = [realised[(key, float(s))] for s in grid]
            entry: dict[str, Any] = {"grid": [float(s) for s in grid], "realised": real}
            for name in probs:
                p = [probs[name][key][float(s)] for s in grid]
                entry[name] = predicted_fit(real, p, shots, info[key], max_relative_se=cut)
            main = entry["all"]
            entry.update({k: main[k] for k in ("status", "se_pred", "se_note", "relative_se",
                                                "reported_value", "reported_ci")})
            # Each end under the same cut as SE_pred: None where the formula
            # does not apply at that alpha.
            entry["se_band"] = [entry["band_lo"]["se_pred"], entry["band_hi"]["se_pred"]]
            fits[key][gname] = entry
        boot = status_probabilities(probs["all"][key], grids, shots, info[key],
                                    B=int(q5["B"]), seed=int(q5["seed"]))
        draws = boot.pop("_draws")
        for gname in grids:
            fits[key][gname]["status_probability"] = boot["per_grid"][gname]
        joint[key] = draws

    # Q3: both halves from the same replicates.
    c3 = criteria["Q3"]
    d3 = joint.get(c3["record"])
    q3 = None
    if d3 is not None:
        both = [o in c3["odd_statuses"] and g == c3["gentle_status"]
                for o, g in zip(d3[c3["odd_grid"]], d3[c3["gentle_grid"]], strict=True)]
        q3 = {"record": c3["record"],
              "odd_status": fits[c3["record"]][c3["odd_grid"]]["status"],
              "gentle_status": fits[c3["record"]][c3["gentle_grid"]]["status"],
              "p_confirmed": sum(both) / len(both)}

    # Q4c: the predicted gentle / rule SE ratio.
    c4 = q4["Q4c"]
    q4c = None
    pair = fits.get(c4["record"], {})
    if c4["numerator_grid"] in pair and c4["denominator_grid"] in pair:
        num = pair[c4["numerator_grid"]]["all"]["se_formula_value"]
        den = pair[c4["denominator_grid"]]["all"]["se_formula_value"]
        q4c = {"record": c4["record"], "numerator": num, "denominator": den,
               "ratio": num / den if num and den else None,
               "relative_se": [pair[c4["numerator_grid"]]["relative_se"],
                               pair[c4["denominator_grid"]]["relative_se"]]}

    # Q1 / Q2: lambda = 1 per record (drift copy pooled), per circuit set.
    lambda1 = {}
    for record in design["records"]:
        key = record["key"]
        if key not in records:
            continue
        lambda1[key] = {"circuit_set": record.get("circuit_set"),
                        **{name: pooled_p(probs[name], key) for name in alphas},
                        "p_uniform": info[key]["p_uniform"], "p_ideal": info[key]["p_ideal"]}
    c2 = criteria["Q2"]
    q2 = {}
    for circuit_set in c2["sets"]:
        diffs = [v["all"] - v["2q"] for v in lambda1.values() if v["circuit_set"] == circuit_set]
        diff_rms = rms(diffs)
        q2[circuit_set] = {"n": len(diffs), "predicted_rms_difference": diff_rms,
                           "decidable": diff_rms is not None and diff_rms
                           >= float(c2["decidable_min_predicted_rms_difference"])}

    # Q7: each layout's prediction and decidability.
    c7 = criteria["Q7"]
    q7 = {}
    for layout, key in c7["records"].items():
        if key not in lambda1:
            continue
        n_pubs = 2 if f"{key}_drift" in records else 1
        p_pred, p_u = lambda1[key]["all"], lambda1[key]["p_uniform"]
        sigma = proportion_sigma(p_pred, shots * n_pubs)
        margin = float(c7["fine_threshold"]) + float(c7["random_k_sigma"]) * sigma
        q7[layout] = {"record": key, "p_pred": p_pred, "p_uniform": p_u, "sigma": sigma,
                      "margin_needed": margin, "decidable": p_pred - p_u >= margin}

    c8 = criteria["Q8"]
    q8 = None
    if c8["record"] in fits and c8["grid"] in fits[c8["record"]]:
        f8 = fits[c8["record"]][c8["grid"]]
        q8 = {"record": c8["record"], "status": f8["status"],
              "p_resolved": f8["status_probability"][RESOLVED],
              "reported_value": f8["reported_value"], "reported_ci": f8["reported_ci"],
              "p_ideal": info[c8["record"]]["p_ideal"]}

    c9 = criteria["Q9"]
    q9 = {"day1": q9_reference(c9["day1"]),
          "predicted_lambda1": lambda1.get(c9["record"], {}).get("all")}

    q5_rows = [{"key": k, "grid": g, "status": e["status"],
                "probability_of_that_status": e["status_probability"][e["status"]]}
               for k, gs in fits.items() for g, e in gs.items()]

    state = git_state(refresh=True)
    doc: dict[str, Any] = {
        "schema": PREREG_SCHEMA,
        "date": args.date,
        "created_utc": utc_timestamp(),
        "design": {"file": block["file"], "sha256": block["sha256"], "name": block["name"],
                   "grid_fit_bootstrap": design.get("grid_fit_bootstrap")},
        "manifest": rel(paths.manifest),
        "manifest_sha256": manifest_sha,
        "circuits_sha256": manifest["circuits_sha256"],
        "calibration": {"file": rel(calibration), "sha256": sha256_file(calibration),
                        "timestamp": manifest["backend"]["calibration_timestamp"]},
        "code": {"commit": state["commit"], "dirty": state["dirty"],
                 "dirty_files": state.get("dirty_files")},
        "report_bounds": REPORT_BOUNDS,
        "resolved_rule": RESOLVED_RULE,
        "shots": shots,
        "alphas": alphas,
        "alpha_r_band": band,
        "alpha_r_grid": [float(a) for a in _node(design["alpha_r_grid"], "path")],
        "rule_grids": block["rule_grids"],
        "trimmed": block["trimmed"],
        "criteria": design["criteria"],
        "reported_untested": design.get("reported_untested", []),
        "prediction_dirs": prediction_dirs,
        "predictions": {
            "lambda1": lambda1,
            "fits": fits,
            "q2": {"sets": q2, "baselines": day_baselines(design["records"], alphas,
                                                          c2["baselines"])},
            "q3": q3,
            "q4c": q4c,
            "q5": {"rows": q5_rows,
                   "expected_matches_if_most_likely": sum(r["probability_of_that_status"]
                                                          for r in q5_rows)},
            "q6": {"references": q6_references(design["records"],
                                               criteria["Q6"]["reference_rows"])},
            "q7": q7,
            "q8": q8,
            "q9": q9,
        },
        "provenance": record_provenance(),
    }
    return doc, render_markdown(doc)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _f(value: Any, digits: int = 3) -> str:
    """A number for the tables; ``-`` for none."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(doc: Mapping[str, Any]) -> str:
    """The readable pre-registration: hashes, the Q1-Q9 table, predictions, sentence map."""
    pred = doc["predictions"]
    crit = {c["id"]: c for c in doc["criteria"]}
    lines = [
        f"# Pre-registration — {doc['design']['name']}, plan date {doc['date']}",
        "",
        "Written by `paper/scripts/preregister.py`; do not edit by hand. The JSON beside this "
        "file is the record; its sha256 is printed by the script and goes on the checklist.",
        "",
        f"- manifest `{doc['manifest']}`, sha256 `{doc['manifest_sha256']}`",
        f"- design `{doc['design']['file']}`, sha256 `{doc['design']['sha256']}`",
        f"- calibration `{doc['calibration']['file']}`, stamp {doc['calibration']['timestamp']}",
        f"- code commit `{doc['code']['commit']}` (dirty: {doc['code']['dirty']})",
        f"- REPORT_BOUNDS = {doc['report_bounds']}, RESOLVED_RULE = {doc['resolved_rule']}, "
        f"{doc['shots']} shots per PUB",
        "- alphas: " + ", ".join(f"{n} = {a['value']:.4g} ({a['scale_only']} errors)"
                                 for n, a in doc["alphas"].items())
        + f"; Q4b band alpha_r = {_f(doc['alpha_r_band']['alphas'][0])}, "
          f"{_f(doc['alpha_r_band']['alphas'][1])} (quantiles "
          f"{doc['alpha_r_band']['quantiles']} of {doc['alpha_r_band']['n']} records)",
        "",
        "## Questions, rules and what is predicted",
        "",
        "| ID | Question | Tested | Rule | Prediction |",
        "|---|---|---|---|---|",
    ]

    def row(cid: str, prediction: str) -> None:
        c = crit[cid]
        rule = c.get("verdict_rule") or "; ".join(
            f"{k}: {v.get('verdict_rule') or v.get('what', '')}" for k, v in
            (c.get("parts") or {}).items()) or "; ".join(
            f"{k}: {v}" for k, v in (c.get("categories") or {}).items()) or c.get("view", "")
        cells = [cid, c["question"], "yes" if c.get("tested") else "reported", rule, prediction]
        lines.append("| " + " | ".join(str(x).replace("|", r"\|") for x in cells) + " |")

    l1 = pred["lambda1"]
    for cid in ("Q1",):
        parts = []
        for circuit_set in crit[cid]["sets"]:
            keys = [k for k, v in l1.items() if v["circuit_set"] == circuit_set]
            parts.append(f"{circuit_set}: " + ", ".join(f"{k} {_f(l1[k]['all'])}" for k in keys))
        row(cid, "; ".join(parts))
    q2 = pred["q2"]["sets"]
    row("Q2", "; ".join(f"{s}: predicted RMS(P_all - P_2q) {_f(v['predicted_rms_difference'])}"
                        f" -> {'decidable' if v['decidable'] else 'NOT DECIDABLE BY DESIGN'}"
                        for s, v in q2.items()))
    q3 = pred["q3"]
    row("Q3", "-" if not q3 else f"odd {q3['odd_status']}, gentle {q3['gentle_status']}; "
                                 f"P(confirmed) = {_f(q3['p_confirmed'], 2)}")
    q4c = pred["q4c"]
    row("Q4", "Q4a eligible: " + (", ".join(
        f"{k}/{g}" for k, gs in pred["fits"].items() for g, e in gs.items() if e["se_pred"])
        or "none") + (f"; Q4c ratio {_f(q4c['ratio'], 2)}" if q4c else ""))
    row("Q5", f"{len(pred['q5']['rows'])} fits; expected matches "
              f"{_f(pred['q5']['expected_matches_if_most_likely'], 1)}")
    refs = pred["q6"]["references"]
    row("Q6", ", ".join(f"{k} {_f(v['alpha_r'], 2)}±{_f(v['alpha_r_se'], 2)}"
                        + (" (excl.)" if v["excluded_because"] else "")
                        for k, v in refs.items() if v))
    q7 = pred["q7"]
    row("Q7", "; ".join(f"{lay}: P_pred {_f(v['p_pred'])}, P_u {_f(v['p_uniform'])}, "
                        f"{'decidable' if v['decidable'] else 'calibration predicts it'}"
                        for lay, v in q7.items()))
    q8 = pred["q8"]
    row("Q8", "-" if not q8 else f"{q8['status']}, P(resolved) = {_f(q8['p_resolved'], 2)}")
    q9 = pred["q9"]
    row("Q9", f"day 1 residual {q9['day1']['residual']:+.3f} (z {q9['day1']['z']:+.1f}); "
              f"day 3 P_pred {_f(q9['predicted_lambda1'])}")

    lines += [
        "",
        "## Per record x grid (alpha_all, plan calibration)",
        "",
        "| Record | Grid | lambda | P(lambda) predicted | Status | P(status) | Reported "
        f"({doc['report_bounds']}) | SE_pred | SE band (alpha_r) | 2q status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, grids in pred["fits"].items():
        for gname, e in grids.items():
            ci = e["reported_ci"]
            rep = (f"{_f(e['reported_value'])} [{_f(ci[0])}, {_f(ci[1])}]" if ci
                   else f"{_f(e['reported_value'])} (lambda = 1)")
            se = _f(e["se_pred"]) if e["se_pred"] is not None else f"n/a ({_f(e['relative_se'], 2)})"
            band = [_f(v) if v is not None else f"n/a ({_f(e[end]['relative_se'], 2)})"
                    for v, end in zip(e["se_band"], ("band_lo", "band_hi"), strict=True)]
            lines.append(
                f"| {key} | {gname} | {', '.join(f'{s:g}' for s in e['grid'])} | "
                f"{', '.join(_f(x) for x in e['all']['p'])} | {e['status']} | "
                f"{_f(e['status_probability'][e['status']], 2)} | {rep} | {se} | "
                f"{band[0]}–{band[1]} | {e['2q']['status']} |")
    lines += [
        "",
        "## lambda = 1 per record (drift copies pooled)",
        "",
        "| Record | Set | P_pred (all) | P_pred (2q) | P_u | Ideal | Reference alpha_r |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, v in l1.items():
        ref = refs.get(key)
        lines.append(f"| {key} | {v['circuit_set']} | {_f(v['all'])} | {_f(v['2q'])} | "
                     f"{_f(v['p_uniform'])} | {_f(v['p_ideal'])} | "
                     f"{_f(ref['alpha_r'], 2) + ' (' + ref['day'] + ')' if ref else '-'} |")
    base = pred["q2"]["baselines"]
    lines += ["", "Q2 baselines (each model's lambda = 1 RMS on these circuits on their own day): "
              + "; ".join(f"{m}: " + ", ".join(f"{s} {_f(v['rms'])} ({v['day']})"
                                               for s, v in sets.items())
                          for m, sets in base.items()), ""]
    lines += ["## Outcome -> sentence", ""]
    for c in doc["criteria"]:
        for outcome, sentence in (c.get("sentences") or {}).items():
            lines.append(f"- **{c['id']} {outcome}**: {sentence}")
        for outcome, sentence in (c.get("map") or {}).items():
            lines.append(f"- **{c['id']} {outcome}**: {sentence}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Pre-register a design job's predictions.")
    parser.add_argument("--date", required=True, help="The manifest's date.")
    parser.add_argument("--max-threads", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Replace an existing pre-registration (only before committing it).")
    parser.add_argument("--hardware-dir", type=Path, default=HARDWARE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Simulate, predict, write the pre-registration, print its sha256."""
    args = parse_args(argv)
    out = preregistration_path(Paths(args.date, args.hardware_dir, args.data_dir))
    if out.exists() and not args.force:
        print(f"REFUSED: {rel(out)} exists; a pre-registration is never replaced "
              "(--force only before it is committed).")
        return 1
    doc, markdown = build(args)
    write_json(out, doc)
    md = out.with_suffix(".md")
    md.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {rel(out)}\n      {rel(md)}")
    print(f"\nsha256 {sha256_file(out)}  {rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
