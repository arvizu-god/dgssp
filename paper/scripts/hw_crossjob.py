"""
hw_crossjob.py

What jobs 1-3 say together.  Offline, from files under ``paper/data`` and
``paper/hardware`` that other scripts wrote; it simulates nothing and never
contacts the service.

a. alpha-hat out of sample on two days (jobs 2 and 3): per-record lambda = 1
   deviations from the prediction made before each job, RMS with and without
   P-n3-M2-17, for the all-errors and the CZ-only model.
b. alpha_r day 1 vs day 3 per pinned circuit (Q6 of the scorecard), with the
   instance's day-2 value (another layout) where step 4 has one.
c. P-n3-M2-17 on days 1, 2 and 3, on both layouts (Q7).
d. The error formula on hardware: every resolved fit of jobs 1-3, SE_boot
   against ``exp_fixed_se`` at the observed points, split into post hoc (jobs
   1-2) and pre-registered (job 3), with job 3's forecast SE_pred beside it.
e. Total QPU seconds over all jobs and probes.
f. **Post hoc, not pre-registered:** which qubits and couplers separate the job
   3 circuits below their prediction from those above it, and what Q1, Q2 and
   Q6 would read without the circuits on those qubits.  It never replaces a
   verdict: the scorecard is the record of what was declared.

Output: ``paper/data/hw_crossjob.json`` (``save_record`` never overwrites).

Run::

    python paper/scripts/hw_crossjob.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import (  # noqa: E402
    DATA_DIR,
    HARDWARE_DIR,
    record_provenance,
    save_record,
    utc_timestamp,
)
from dgssp.metrics import proportion_sigma  # noqa: E402
from hardware_batch import Paths, load_manifest, rel  # noqa: E402
from hw_aggregate import grid_fit  # noqa: E402

SCHEMA = "dgssp/paper/hw-crossjob/1"

JOB1 = "2026-10-25"
JOB2 = "2026-10-28"
JOB3 = "2026-10-29"

#: The design's bootstrap (``grid_fit_bootstrap``), so jobs 1-2 compare with job 3.
FIT_B = 2000
FIT_SEED = 0

#: The layout-test instance, left out of the "without" RMS (Q7 asks about it).
LAYOUT_TEST = "P-n3-M2-17"

#: Q4a's window and eligibility, read from the scorecard for job 3; here they
#: only count how many post hoc fits fall inside the same window.
RATIO_WINDOW = (0.75, 1.33)
ELIGIBLE_RELATIVE_SE = 0.2

#: "At random" as Q7 declares it: lambda = 1 less than this many sigma above P_u.
RANDOM_K = 2.0

OUT = DATA_DIR / "hw_crossjob.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rms(values: Iterable[float]) -> float | None:
    """Root mean square, ``None`` for no values."""
    items = list(values)
    return math.sqrt(sum(v * v for v in items) / len(items)) if items else None


def base_key(key: str) -> str:
    """The instance of a record key: ``P-n3-M1-17_r1_L2`` -> ``P-n3-M1-17_r1``."""
    return key.removesuffix("_drift").removesuffix("_L1").removesuffix("_L2")


# ---------------------------------------------------------------------------
# a. Out of sample
# ---------------------------------------------------------------------------


def job2_rows(robustness: Mapping[str, Any], variant: str) -> list[dict[str, Any]]:
    """Job 2's lambda = 1 records against one model's prediction, drift copies pooled."""
    pooled: dict[str, list[Mapping[str, Any]]] = {}
    for row in robustness["variants"][variant]["out_of_sample_job2"]["per_record_lambda1"]:
        pooled.setdefault(base_key(row["key"]), []).append(row)
    return [{"key": key,
             "observed": statistics.fmean(r["hardware"] for r in rows),
             "predicted": statistics.fmean(r["predicted"] for r in rows),
             "deviation": statistics.fmean(r["deviation"] for r in rows)}
            for key, rows in pooled.items()]


def job3_rows(card: Mapping[str, Any], prereg: Mapping[str, Any],
              variant: str) -> list[dict[str, Any]]:
    """Job 3's lambda = 1 records against one model's pre-registered prediction."""
    observed: dict[str, float] = {}
    for block in card["questions"]["Q1"]["sets"].values():
        observed.update({k: r["observed"] for k, r in block["records"].items()})
    for layout in card["questions"]["Q7"]["layouts"].values():
        observed[layout["record"]] = layout["observed"]
    rows = []
    for key, pred in prereg["predictions"]["lambda1"].items():
        rows.append({"key": key, "circuit_set": pred["circuit_set"], "observed": observed[key],
                     "predicted": pred[variant], "deviation": observed[key] - pred[variant]})
    return rows


def rms_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-record deviations and the RMS with and without the layout-test instance."""
    return {"rows": [dict(r) for r in rows],
            "rms": rms(r["deviation"] for r in rows),
            "rms_without_layout_test": rms(r["deviation"] for r in rows
                                           if base_key(r["key"]) != LAYOUT_TEST),
            "max_abs": max(abs(r["deviation"]) for r in rows)}


def out_of_sample_block(robustness: Mapping[str, Any], card: Mapping[str, Any],
                        prereg: Mapping[str, Any]) -> dict[str, Any]:
    """alpha-hat out of sample: job 2, and job 3's circuits of each origin."""
    out: dict[str, Any] = {"what": "lambda = 1, prediction made before each job on its plan "
                                   "calibration; drift copies pooled"}
    for variant in ("all", "2q"):
        third = job3_rows(card, prereg, variant)
        out[variant] = {
            "alpha": prereg["alphas"][variant]["value"],
            JOB2: rms_block(job2_rows(robustness, variant)),
            f"{JOB3}_day2_circuits": rms_block([r for r in third if r["key"].endswith("_L2")]),
            f"{JOB3}_day1_circuits": rms_block([r for r in third if r["key"].endswith("_L1")]),
        }
    return out


# ---------------------------------------------------------------------------
# b. alpha_r per circuit, c. P-n3-M2-17
# ---------------------------------------------------------------------------


def alpha_r_block(robustness: Mapping[str, Any], card: Mapping[str, Any]) -> dict[str, Any]:
    """Q6's rows, with the instance's day-2 value (another layout) where step 4 has one."""
    day2 = {r["key"]: r for r in robustness["depth"]["rows"] if r["job"] == JOB2}
    q6 = card["questions"]["Q6"]
    rows = []
    for row in q6["rows"]:
        other = day2.get(base_key(row["key"]))
        rows.append({**row,
                     "alpha_r_day2_other_layout": (other or {}).get("alpha_r")
                     if row["day"] != JOB2 else None})
    return {"rows": rows, "verdict": q6["verdict"], "shifts": q6["shifts"],
            "chi2_red_differences": q6["chi2_red_differences"],
            "chi2_red_between_day3": q6["chi2_red_between"], "spearman": q6["spearman"]}


def layout_row(day: str, layout: str, observed: float, shots: int, p_uniform: float,
               predicted: float | None) -> dict[str, Any]:
    """One day and layout of the layout-test instance, classified as Q7 declares."""
    sigma = proportion_sigma(observed, shots)
    z = (observed - p_uniform) / sigma
    return {"day": day, "layout": layout, "observed": observed, "shots_pooled": shots,
            "sigma": sigma, "p_uniform": p_uniform, "z_above_random": z,
            "at_random": z < RANDOM_K, "predicted": predicted}


def layout_block(robustness: Mapping[str, Any], card: Mapping[str, Any]) -> dict[str, Any]:
    """P-n3-M2-17 on days 1, 2, 3 and both layouts (Q7's cell comes from the scorecard)."""
    q7 = card["questions"]["Q7"]
    depth = {r["id"]: r for r in robustness["depth"]["rows"]}
    day2_pred = next(r["predicted"] for r in job2_rows(robustness, "all")
                     if r["key"] == LAYOUT_TEST)
    rows = []
    for day, layout, predicted in ((JOB1, "L1", None), (JOB2, "L2", day2_pred)):
        row = depth[f"{LAYOUT_TEST}@{day}"]
        rows.append(layout_row(day, layout, statistics.fmean(row["hardware"]),
                               row["shots"] * len(row["hardware"]),
                               q7["layouts"]["L1"]["p_uniform"], predicted))
    for layout in ("L1", "L2"):
        block = q7["layouts"][layout]
        rows.append({**layout_row(JOB3, layout, block["observed"],
                                  card_shots(card) * block["n_pubs"], block["p_uniform"],
                                  block["predicted"]),
                     "category": block["category"]})
    return {"rows": rows, "cell": q7["cell"], "sentence": q7["sentence"]}


def card_shots(card: Mapping[str, Any]) -> int:
    """The shots per PUB of the scored job."""
    return int(card["grid_fits"][0]["shots"][0])


# ---------------------------------------------------------------------------
# d. The error formula on hardware
# ---------------------------------------------------------------------------


def fit_row(job: str, key: str, grid_name: str, fit: Mapping[str, Any]) -> dict[str, Any]:
    """The common columns of one fit."""
    se_boot, se_formula = fit["se_boot"], fit["se_formula"]
    lo, hi = fit["reported_lo"], fit["reported_hi"]
    ideal = fit["p_ideal"]
    return {"job": job, "key": key, "grid_name": grid_name, "grid": list(fit["grid"]),
            "shots": fit["shots"][0], "status": fit["status"], "p_ideal": ideal,
            "reported_value": fit["reported_value"], "reported_lo": lo, "reported_hi": hi,
            "interval_contains_ideal": lo <= ideal <= hi or math.isclose(ideal, hi, abs_tol=1e-9),
            "se_boot": se_boot, "se_formula": se_formula,
            "relative_se_formula": fit.get("relative_se_formula"),
            "ratio": se_boot / se_formula}


def post_hoc_fit_rows(job: str, data_dir: Path) -> list[dict[str, Any]]:
    """Every resolved fit of a job without named grids: the grid it ran, the design's bootstrap."""
    rows = []
    for path in sorted((data_dir / f"hw_{job}").glob("*.json")):
        record = _load_json(path)
        stored = record.get("zne_fit")
        if not stored or len(stored["nominal_scales"]) < 2:
            continue
        grid = sorted(float(s) for s in stored["nominal_scales"])
        fit = grid_fit(record, grid, B=FIT_B, seed=FIT_SEED)
        if fit["status"] == "resolved":
            rows.append(fit_row(job, path.stem, "as run", {**fit, "grid": grid}))
    return rows


def formula_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How the ratios of a set of fits sit against Q4a's window."""
    ratios = [r["ratio"] for r in rows]
    if not ratios:
        return {"n": 0}
    low, high = RATIO_WINDOW
    return {"n": len(ratios), "ratio_median": statistics.median(ratios),
            "ratio_min": min(ratios), "ratio_max": max(ratios),
            "n_inside_window": sum(low <= r <= high for r in ratios),
            "n_interval_contains_ideal": sum(bool(r["interval_contains_ideal"]) for r in rows)}


def formula_block(card: Mapping[str, Any], data_dir: Path) -> dict[str, Any]:
    """SE_boot against the formula: post hoc (jobs 1-2), pre-registered (job 3)."""
    post_hoc = post_hoc_fit_rows(JOB1, data_dir) + post_hoc_fit_rows(JOB2, data_dir)
    q4a = {(r["key"], r["grid"]): r for r in card["questions"]["Q4"]["Q4a"]["rows"]}
    third = []
    for fit in card["grid_fits"]:
        if fit["status"] != "resolved":
            continue
        declared = q4a[(fit["key"], fit["grid_name"])]
        third.append({**fit_row(JOB3, fit["key"], fit["grid_name"], fit),
                      "eligible": declared["eligible"], "se_pred": declared["se_pred"],
                      "se_pred_value": declared["se_pred_value"],
                      "ratio_to_forecast": declared["ratio_q4b"]})
    eligible = [r for r in third if r["eligible"]]
    return {"window": list(RATIO_WINDOW), "bootstrap": {"B": FIT_B, "seed": FIT_SEED},
            "post_hoc_jobs_1_2": {"rows": post_hoc, "summary": formula_summary(post_hoc)},
            "pre_registered_job_3": {
                "rows": third, "summary": formula_summary(third),
                "summary_eligible": formula_summary(eligible),
                "forecast_ratios_eligible": {f"{r['key']}/{r['grid_name']}": r["ratio_to_forecast"]
                                             for r in eligible}}}


# ---------------------------------------------------------------------------
# e. QPU seconds
# ---------------------------------------------------------------------------


def qpu_block(usage: Mapping[str, Any]) -> dict[str, Any]:
    """Every job's charged seconds and their total."""
    jobs = {job_id: row["usage_seconds"] for job_id, row in usage.items()}
    return {"jobs": jobs, "total_seconds": sum(jobs.values())}


# ---------------------------------------------------------------------------
# f. Post hoc: what separates the circuits below their prediction
# ---------------------------------------------------------------------------


def circuit_couplers(manifest: Mapping[str, Any], circuits: Sequence[Any]) -> dict[str, set[tuple[int, int]]]:
    """The physical CZ couplers of each record's lambda = 1 circuit, drift copies left out."""
    out: dict[str, set[tuple[int, int]]] = {}
    for row in manifest["circuits"]:
        key = row["key"]
        if float(row["nominal_scale"]) != 1.0 or key.endswith("_drift"):
            continue
        circuit = circuits[row["pub_index"]]
        pairs: set[tuple[int, int]] = set()
        for inst in circuit.data:
            if inst.operation.name == "cz":
                first, second = sorted(circuit.find_bit(q).index for q in inst.qubits)
                pairs.add((first, second))
        out[key] = pairs
    return out


def separators(below: Sequence[set[Any]], above: Sequence[set[Any]]) -> list[Any]:
    """Elements in every set of ``below`` and in no set of ``above``."""
    if not below:
        return []
    shared = set.intersection(*below)
    return sorted(shared - set().union(*above))


def shifted_chi2(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """
    Q6's first statistic: chi2/dof of the differences after one shift per reference day.

    ``rows`` carry ``day``, ``difference``, ``se_ref`` and ``se_day3``; the dof
    are the number of rows minus the number of reference days.
    """
    days: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        days.setdefault(row["day"], []).append(row)
    chi2 = 0.0
    shifts = {}
    for day, group in days.items():
        weights = [1.0 / (r["se_ref"] ** 2 + r["se_day3"] ** 2) for r in group]
        shift = sum(w * r["difference"] for w, r in zip(weights, group, strict=True)) / sum(weights)
        shifts[day] = shift
        chi2 += sum(w * (r["difference"] - shift) ** 2
                    for w, r in zip(weights, group, strict=True))
    dof = len(rows) - len(days)
    return {"n": len(rows), "dof": dof, "chi2": chi2, "chi2_red": chi2 / dof if dof > 0 else None,
            "shifts": shifts}


def between_chi2(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Q6's second statistic: chi2/dof of the day-3 alpha_r about their weighted mean."""
    if len(rows) < 2:
        return None
    weights = [1.0 / r["se_day3"] ** 2 for r in rows]
    mean = sum(w * r["alpha_day3"] for w, r in zip(weights, rows, strict=True)) / sum(weights)
    return sum(w * (r["alpha_day3"] - mean) ** 2
               for w, r in zip(weights, rows, strict=True)) / (len(rows) - 1)


def calibration_entries(pairs: Iterable[tuple[int, int]], qubits: Iterable[int],
                        hardware_dir: Path) -> dict[str, Any]:
    """The candidates' CZ errors and T2 in each job's execution calibration."""
    out: dict[str, Any] = {}
    for day in (JOB1, JOB2, JOB3):
        snapshot = _load_json(hardware_dir / f"calibration_exec_{day}.json")
        cz = {tuple(sorted(e["qubits"])): e["error"]
              for e in snapshot["calibration"]["instructions"]["cz"]}
        props = snapshot["raw_properties"]["qubits"]
        t2 = {q: next((p["value"] for p in props[q] if p["name"] == "T2"), None) for q in qubits}
        out[day] = {"calibration_timestamp": snapshot["calibration_timestamp"],
                    "device_cz_median": snapshot["median_gate_errors"]["two_qubit"],
                    "cz": {f"{a}-{b}": cz.get((a, b)) for a, b in pairs},
                    "t2_us": {str(q): t2[q] for q in qubits}}
    return out


def post_hoc_block(robustness: Mapping[str, Any], card: Mapping[str, Any],
                   prereg: Mapping[str, Any], hardware_dir: Path, data_dir: Path) -> dict[str, Any]:
    """The pattern behind the misses, clearly apart from the verdicts."""
    manifest, _, circuits = load_manifest(Paths(JOB3, hardware_dir, data_dir).manifest)
    couplers = circuit_couplers(manifest, circuits)
    rows = job3_rows(card, prereg, "all")
    below = sorted(r["key"] for r in rows if r["deviation"] < 0)
    above = sorted(r["key"] for r in rows if r["deviation"] >= 0)
    qubits_of = {k: {q for pair in v for q in pair} for k, v in couplers.items()}
    q_sep = separators([qubits_of[k] for k in below], [qubits_of[k] for k in above])
    c_sep = separators([couplers[k] for k in below], [couplers[k] for k in above])
    touching = {k for k, q in qubits_of.items() if set(q_sep) & q}

    q6_rows = [r for r in card["questions"]["Q6"]["rows"] if r["included"]]
    check = shifted_chi2(q6_rows)
    kept_q6 = [r for r in q6_rows if r["key"] not in touching]
    without_q6 = shifted_chi2(kept_q6)

    variants: dict[str, Any] = {}
    for variant in ("all", "2q"):
        third = {r["key"]: r["deviation"] for r in job3_rows(card, prereg, variant)}
        variants[variant] = {
            name: {"records": sorted(k for k in third
                                     if k.endswith(suffix) and k not in touching
                                     and base_key(k) != LAYOUT_TEST),
                   "rms": rms(d for k, d in third.items()
                              if k.endswith(suffix) and k not in touching
                              and base_key(k) != LAYOUT_TEST)}
            for name, suffix in (("day2_circuits", "_L2"), ("day1_circuits", "_L1"))}
    day2 = {r["key"]: r["deviation"] for r in job2_rows(robustness, "all")}
    return {
        "status": "post hoc: chosen after seeing job 3; a hypothesis, not a verdict",
        "below_prediction": below, "at_or_above_prediction": above,
        "deviations_all": {r["key"]: r["deviation"] for r in rows},
        "couplers": {k: [f"{a}-{b}" for a, b in sorted(v)] for k, v in couplers.items()},
        "qubits_in_every_below_and_no_above": q_sep,
        "couplers_in_every_below_and_no_above": [f"{a}-{b}" for a, b in c_sep],
        "calibration_of_candidates": calibration_entries(c_sep, q_sep, hardware_dir),
        "same_qubits_on_day_2": {
            "P-n2-M1-11_L2_deviation": day2["P-n2-M1-11"],
            "P-n3-M2-17_L2_deviation": day2[LAYOUT_TEST],
            "note": "both circuits use these couplers; on day 2 only P-n3-M2-17_L2 failed",
        },
        "q6": {"reproduces_scorecard_chi2_red": check["chi2_red"],
               "without_records_on_candidates": {
                   **without_q6, "records": sorted(r["key"] for r in kept_q6),
                   "chi2_red_between_day3": between_chi2(kept_q6)}},
        "q1_q2_rms_without_records_on_candidates": variants,
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(data_dir: Path = DATA_DIR, hardware_dir: Path = HARDWARE_DIR) -> dict[str, Any]:
    """Every block, from the files the other scripts wrote."""
    robustness = _load_json(data_dir / "alpha_robustness.json")
    card = _load_json(data_dir / f"scorecard_{JOB3}.json")
    prereg = _load_json(hardware_dir / f"preregistration_{JOB3}.json")
    usage = _load_json(hardware_dir / "usage.json")
    return {
        "schema": SCHEMA,
        "jobs": [JOB1, JOB2, JOB3],
        "out_of_sample": out_of_sample_block(robustness, card, prereg),
        "alpha_r": alpha_r_block(robustness, card),
        "layout_test": layout_block(robustness, card),
        "error_formula": formula_block(card, data_dir),
        "qpu": qpu_block(usage),
        "post_hoc": post_hoc_block(robustness, card, prereg, hardware_dir, data_dir),
        "created_utc": utc_timestamp(),
        "provenance": record_provenance(),
    }


def _f(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_summary(payload: Mapping[str, Any]) -> None:
    """The blocks, for the terminal."""
    print("\nalpha-hat out of sample (lambda = 1): RMS, RMS without P-n3-M2-17")
    for variant in ("all", "2q"):
        block = payload["out_of_sample"][variant]
        for name in (JOB2, f"{JOB3}_day2_circuits", f"{JOB3}_day1_circuits"):
            print(f"  {variant:3} {name:26} {_f(block[name]['rms'])}  "
                  f"{_f(block[name]['rms_without_layout_test'])}")
    print("\nP-n3-M2-17:")
    for row in payload["layout_test"]["rows"]:
        print(f"  {row['day']} {row['layout']}  P {row['observed']:.3f}  "
              f"z above random {row['z_above_random']:+.1f}  predicted {_f(row['predicted'])}")
    print("\nerror formula (SE_boot / SE_formula at the observed points):")
    formula = payload["error_formula"]
    print(f"  post hoc      {formula['post_hoc_jobs_1_2']['summary']}")
    print(f"  pre-registered {formula['pre_registered_job_3']['summary']}")
    print(f"  forecast ratios {formula['pre_registered_job_3']['forecast_ratios_eligible']}")
    print(f"\nQPU seconds, all jobs and probes: {payload['qpu']['total_seconds']}")
    post = payload["post_hoc"]
    print("\npost hoc, NOT a verdict:")
    print(f"  below prediction: {post['below_prediction']}")
    print(f"  qubits separating them: {post['qubits_in_every_below_and_no_above']}")
    print(f"  couplers separating them: {post['couplers_in_every_below_and_no_above']}")
    print(f"  Q6 chi2/dof without them: "
          f"{_f(post['q6']['without_records_on_candidates']['chi2_red'], 2)} "
          f"(scorecard {_f(post['q6']['reproduces_scorecard_chi2_red'], 2)})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="What jobs 1-3 say together (offline).")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--hardware-dir", type=Path, default=HARDWARE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build(args.data_dir, args.hardware_dir)
    written = save_record(args.out, payload)
    print(f"wrote {rel(written)}")
    print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
