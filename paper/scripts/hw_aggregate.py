"""
hw_aggregate.py

Turn one hardware job's records into the paper's hardware tables.

Reads, for one job date (nothing is simulated, transpiled or submitted):

``paper/hardware/manifest_<date>.json``, ``usage.json``, ``calibration_exec_<date>_fetch.json``
    What was sent, what it cost, and the calibration in force while it ran.
``paper/data/hw_<date>/``
    The hardware records (executor ``hardware``).
``paper/data/hw_dryrun_<date>/``
    The same circuits on the noise model of the *plan* calibration.
``paper/data/hw_predict_exec_<date>/``
    The same circuits on the noise model of the *execution* calibration.
``paper/data/hw_alpha_exec_<date>/fit.json``
    The fitted error scale alpha, and each PUB's prediction on the alpha grid.

and writes:

``paper/data/hw_records.csv``
    One row per record: ``P(solution)`` on hardware and on each model,
    "signal kept", the ZNE estimate with its status, Mitiq, the ``M = 0``
    distances.
``paper/data/hw_summary.json``
    The same per record plus what the figures need (each scale's
    observations, the fitted curve, the worked example's distributions), the
    drift check, the alpha fit, the run configuration, the two-qubit band
    where ZNE stops working (:func:`limit_band`), and provenance.
    ``make_figures.py --set hw`` and ``resource_scaling.py --figure-only``
    read this file and nothing else from the job.
``paper/tex/hw_table_results.tex``
    Table "hardware results per record".
``paper/tex/hw_table_config.tex``
    Table "hardware run configuration".

The flags are re-derived, not read: ``resolved``, ``identifiable`` and the
reported value follow the library's current rule via
:func:`aggregate.rederive_flags`, exactly as the simulation tables do.

"Signal kept" is ``(P_sol - P_uniform) / (P_ideal - P_uniform)`` at
``lambda = 1``: 1 is the noiseless answer, 0 is random guessing.  It is
undefined for ``M = 0`` and ``M = N/2``, whose ideal *is* random guessing.

Run::

    python paper/scripts/hw_aggregate.py
    python paper/scripts/hw_aggregate.py --date 2026-10-25

Without ``--date`` the job of record (:data:`JOB_OF_RECORD`) is aggregated.
Another job's files carry its date (``hw_summary_<date>.json``, ...), so a
later job never overwrites the paper's tables.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import DATA_DIR, HARDWARE_DIR, TEX_DIR, record_provenance  # noqa: E402
from aggregate import exact_p_ideal, rederive_flags, write_csv  # noqa: E402
from dgssp.mitigation.zne import RESOLVED_RULE  # noqa: E402
from hardware_batch import (  # noqa: E402
    Paths,
    drift_metrics,
    latest_manifest_date,
    rel,
    sha256_file,
    write_json,
)
from noise_model_check import signal_kept  # noqa: E402

SUMMARY_SCHEMA = "dgssp/paper/hardware-summary/1"

#: The job the paper's hardware tables and figures are of.  Without ``--date``
#: this one is aggregated, not the newest manifest: planning a later job must
#: not change the paper.  Any other date writes its files with a ``_<date>``
#: suffix (``hw_summary_<date>.json``, ...), beside the paper's.
JOB_OF_RECORD = "2026-10-25"

#: The label ``noise_model_check.py`` gives the execution-time calibration.
EXEC_LABEL = "exec"

#: Which alpha fit the tables quote: every PUB with signal, all scales.
ALPHA_GROUP = "all"

#: Status of a record's scalar ZNE estimate, in the order the tables explain.
STATUSES: tuple[str, ...] = (
    "resolved",
    "identified, not resolved",
    "not identifiable",
    "no signal",
)

#: Points on the fitted curve the folding figure draws (lambda = 0 .. 5.5).
CURVE_LAMBDAS: tuple[float, ...] = tuple(float(x) for x in np.linspace(0.0, 5.5, 56))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class HwInputs:
    """Every file one job's tables are built from."""

    def __init__(self, date: str, hardware_dir: Path, data_dir: Path) -> None:
        self.paths = Paths(date, hardware_dir, data_dir)
        self.date = date
        self.predict_dir = data_dir / f"hw_predict_{EXEC_LABEL}_{date}"
        self.alpha_fit = data_dir / f"hw_alpha_{EXEC_LABEL}_{date}" / "fit.json"
        self.alpha_fit_plan = data_dir / f"hw_alpha_plan_{date}" / "fit.json"
        self.fetch_report = hardware_dir / f"calibration_{EXEC_LABEL}_{date}_fetch.json"

    def required(self) -> list[Path]:
        """The files without which no table can be built."""
        return [self.paths.manifest, self.paths.hardware_out_dir]

    def hashed(self) -> dict[str, str | None]:
        """``{repo path: sha256}`` of every input file (directories: their records)."""
        files = [self.paths.manifest, self.paths.usage_json, self.fetch_report,
                 self.alpha_fit, self.alpha_fit_plan]
        for directory in (self.paths.hardware_out_dir, self.paths.dryrun_dir, self.predict_dir):
            files += sorted(directory.glob("*.json")) if directory.exists() else []
        return {rel(p): (sha256_file(p) if p.exists() else None) for p in files}


def _load_json(path: Path) -> Any:
    """A JSON file, or ``None`` when it does not exist."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_records(directory: Path) -> dict[str, dict[str, Any]]:
    """
    The records of one directory, keyed by record key (the file stem).

    Parameters
    ----------
    directory:
        ``hw_<date>``, ``hw_dryrun_<date>`` or ``hw_predict_exec_<date>``.

    Returns
    -------
    dict
        Empty when the directory does not exist (a stage not run yet).
    """
    if not directory.exists():
        return {}
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    }


# ---------------------------------------------------------------------------
# Per-record quantities
# ---------------------------------------------------------------------------


def _get(record: Mapping[str, Any] | None, *path: str) -> Any:
    """Nested ``dict.get`` that tolerates a missing record or level."""
    node: Any = record
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def status_of(flags: Mapping[str, Any]) -> str | None:
    """
    One word for what the scalar estimate is, from :func:`aggregate.rederive_flags`.

    Returns
    -------
    str | None
        One of :data:`STATUSES`, or ``None`` when there is no fit.
    """
    if flags.get("resolved") is None and flags.get("identifiable") is None:
        return None
    if flags.get("signal_direction") == 0:
        return "no signal"
    if flags.get("reported_source") == "extrapolation":
        return "resolved"
    if flags.get("identifiable"):
        return "identified, not resolved"
    return "not identifiable"


def exponential_curve(
    fit: Mapping[str, Any] | None, lambdas: Sequence[float]
) -> list[float] | None:
    """
    The stored exponential fit evaluated on ``lambdas``.

    The scalar fit is ``P(lambda) = asymptote + b * exp(-c * lambda)``, with the
    asymptote fixed at ``P_uniform`` and ``params = [b, c]``; its value at
    ``lambda = 0`` is the extrapolation.

    Returns
    -------
    list[float] | None
        ``None`` for any other model or a fit without parameters.
    """
    if not fit or fit.get("method") != "exponential":
        return None
    params, asymptote = fit.get("params"), fit.get("asymptote")
    if params is None or len(params) != 2 or asymptote is None:
        return None
    b, c = float(params[0]), float(params[1])
    return [float(asymptote) + b * math.exp(-c * x) for x in lambdas]


def alpha_predictions(fit_doc: Mapping[str, Any] | None, alpha: float | None) -> dict[
    tuple[str, float], float
]:
    """
    Each PUB's model ``P(solution)`` at ``alpha``, interpolated as the fit does.

    The fit interpolates each PUB's prediction across the alpha grid with a
    monotone cubic (PCHIP); the same interpolation at the fitted alpha is what
    the fit says the device looks like.

    Returns
    -------
    dict
        ``{(record key, scale): P(solution)}``; empty without a fit.
    """
    if not fit_doc or alpha is None:
        return {}
    from scipy.interpolate import PchipInterpolator

    out: dict[tuple[str, float], float] = {}
    for row in fit_doc.get("per_pub", []):
        grid = sorted((float(a), float(p)) for a, p in row["predicted"].items())
        xs, ys = np.array([g[0] for g in grid]), np.array([g[1] for g in grid])
        out[(row["key"], float(row["scale"]))] = float(PchipInterpolator(xs, ys)(alpha))
    return out


def probabilities(counts: Mapping[str, int] | None) -> dict[str, float] | None:
    """Counts as probabilities, or ``None``."""
    if not counts:
        return None
    total = sum(counts.values())
    return {k: v / total for k, v in sorted(counts.items())}


def _scale_key(scale: float) -> str:
    """How the records key their counts by nominal scale (``"1"``, ``"3"``)."""
    return f"{scale:g}"


def record_entry(
    row: Mapping[str, Any],
    hw: Mapping[str, Any],
    plan: Mapping[str, Any] | None,
    exec_: Mapping[str, Any] | None,
    alpha_pred: Mapping[tuple[str, float], float],
) -> dict[str, Any]:
    """
    Everything the tables and figures say about one ``zne`` record.

    Parameters
    ----------
    row:
        The record's entry in the manifest's ``records``.
    hw, plan, exec_:
        The hardware record, and the same circuits on the plan / execution
        noise models (either may be missing).
    alpha_pred:
        :func:`alpha_predictions`.

    Returns
    -------
    dict
        Flat scalars (the CSV row) plus ``"detail"`` (lists for the figures).
    """
    key = row["key"]
    p_uniform = float(row["p_uniform"])
    p_ideal = exact_p_ideal(dict(hw))
    fit = hw.get("zne_fit") or {}
    flags = rederive_flags(fit, p_ideal)
    status = status_of(flags)
    scales = [float(s) for s in row["nominal_scales"]]

    def p_sol(record: Mapping[str, Any] | None) -> float | None:
        value = _get(record, "metrics", "solution_probability")
        return float(value) if value is not None else None

    def kept(value: float | None) -> float | None:
        return signal_kept(value, p_ideal, p_uniform) if p_ideal is not None else None

    hw_ci = _get(hw, "metrics", "solution_probability_ci") or [None, None]
    alpha_1 = alpha_pred.get((key, scales[0]))
    use_fit = status == "resolved"
    zne_ci = fit.get("ci") if use_fit else None
    mitiq = hw.get("mitiq") or {}
    pathology = hw.get("pathology") or {}

    flat: dict[str, Any] = {
        "key": key,
        "instance": row["instance"],
        "role": row["role"],
        "family": key.split("-", 1)[0],
        "n_items": int(row["n_items"]),
        "M": int(row["M"]),
        "iterations": int(row["iterations"]),
        "r_opt": row.get("r_opt"),
        "two_q_count": _get(hw, "transpiled", "two_q_count"),
        "two_q_depth": _get(hw, "transpiled", "two_q_depth"),
        "seed_source": _get(hw, "transpiled", "seed_source"),
        "p_uniform": p_uniform,
        "p_ideal": p_ideal,
        "signal_direction": flags["signal_direction"],
        # P(solution) at lambda = 1
        "hardware": p_sol(hw),
        "hardware_lo": hw_ci[0],
        "hardware_hi": hw_ci[1],
        "model_plan": p_sol(plan),
        "model_exec": p_sol(exec_),
        "model_alpha": alpha_1,
        "kept_hardware": kept(p_sol(hw)),
        "kept_model_plan": kept(p_sol(plan)),
        "kept_model_exec": kept(p_sol(exec_)),
        "kept_model_alpha": kept(alpha_1),
        # scalar ZNE (ours), flags under the current rule
        "resolved_rule": RESOLVED_RULE,
        "zne_status": status,
        "resolved": flags["resolved"],
        "identifiable": flags["identifiable"],
        "reported_source": flags["reported_source"],
        "zne_reported": flags["reported_value"],
        "zne_reported_lo": zne_ci[0] if zne_ci else None,
        "zne_reported_hi": zne_ci[1] if zne_ci else None,
        "zne_value": fit.get("value"),
        "zne_chi2_red": fit.get("chi2_red"),
        "n_informative_scales": flags["n_informative_scales"],
        # Mitiq, offline on the same folded counts
        "mitiq": mitiq.get("value"),
        "mitiq_lo": mitiq.get("lo"),
        "mitiq_hi": mitiq.get("hi"),
        # pointwise ZNE and the M = 0 distances
        "pointwise_clip": _get(hw, "pointwise", "clip", "solution_probability"),
        "pointwise_simplex": _get(hw, "pointwise", "simplex", "solution_probability"),
        "tv_raw": pathology.get("tv_raw_min_scale"),
        "tv_pointwise_clip": pathology.get("tv_pointwise_clip"),
        "tv_pointwise_simplex": pathology.get("tv_pointwise_simplex"),
    }
    detail = {
        "scales": scales,
        "hardware": list(fit.get("observed") or []),
        "hardware_sigma": list(fit.get("sigma") or []),
        "model_plan": list(_get(plan, "zne_fit", "observed") or []),
        "model_exec": list(_get(exec_, "zne_fit", "observed") or []),
        "model_alpha": [alpha_pred.get((key, s)) for s in scales],
        "fit_curve": {
            "lambda": list(CURVE_LAMBDAS),
            "p": exponential_curve(fit, CURVE_LAMBDAS),
        },
        "fit_curve_model_exec": {
            "lambda": list(CURVE_LAMBDAS),
            "p": exponential_curve(_get(exec_, "zne_fit"), CURVE_LAMBDAS),
        },
        "solution_bitstrings": list(_get(hw, "metrics", "solution_bitstrings") or []),
        "distributions": {
            "ideal": _get(hw, "metrics", "reference_distribution"),
            "model_exec": probabilities(_get(exec_, "counts", _scale_key(scales[0]))),
            "hardware": probabilities(_get(hw, "counts", _scale_key(scales[0]))),
            "hardware_zne_pointwise": _get(hw, "pointwise", "clip", "distribution"),
        },
    }
    return {**flat, "detail": detail}


def drift_entries(
    manifest: Mapping[str, Any], hw: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """
    The drift check: each scale-1 repeat against its first copy.

    Uses :func:`hardware_batch.drift_metrics` (two-proportion z-test and the
    total-variation distance between the two histograms).
    """
    key_of_pub = {int(c["pub_index"]): c["key"] for c in manifest["circuits"]}
    out = []
    for row in manifest["records"]:
        if row["kind"] != "drift" or row["key"] not in hw:
            continue
        first_key = key_of_pub[int(row["repeat_of_pub"])]
        first = _get(hw.get(first_key), "counts", "1")
        repeat = _get(hw[row["key"]], "counts", "1")
        if not first or not repeat:
            continue
        metrics = drift_metrics(first, repeat, list(row["solution_bitstrings"]))
        out.append({"key": row["key"], "repeat_of": first_key, **metrics})
    return out


def limit_band(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """
    The two-qubit counts over which ZNE on this device stops working.

    Read off the statuses, with no threshold of its own: the band runs from
    the largest circuit whose estimate was *resolved* to the smallest larger
    one that was *not identifiable*.  Records with no signal say nothing about
    the limit and are left out.  Records that sit on the wrong side (a
    resolved or identified estimate above the band, a not-identifiable one
    below it) are listed as exceptions rather than moving the edges.

    Returns
    -------
    dict | None
        ``{"lower", "lower_key", "upper", "upper_key", "exceptions"}``, or
        ``None`` when no record was resolved or none above it failed.
    """
    rows = [r for r in records
            if r.get("zne_status") not in (None, "no signal") and r.get("two_q_count")]
    resolved = [r for r in rows if r["zne_status"] == "resolved"]
    if not resolved:
        return None
    lower = max(resolved, key=lambda r: int(r["two_q_count"]))
    failed = [r for r in rows if r["zne_status"] == "not identifiable"
              and int(r["two_q_count"]) > int(lower["two_q_count"])]
    if not failed:
        return None
    upper = min(failed, key=lambda r: int(r["two_q_count"]))
    exceptions = sorted(
        r["key"] for r in rows
        if (r["zne_status"] != "not identifiable"
            and int(r["two_q_count"]) > int(upper["two_q_count"]))
        or (r["zne_status"] == "not identifiable"
            and int(r["two_q_count"]) < int(lower["two_q_count"]))
    )
    return {
        "lower": int(lower["two_q_count"]),
        "lower_key": lower["key"],
        "upper": int(upper["two_q_count"]),
        "upper_key": upper["key"],
        "exceptions": exceptions,
    }


def alpha_block(fit_doc: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The alpha fit's numbers the text quotes, per fitted group."""
    if not fit_doc:
        return None
    return {
        "calibration_file": fit_doc.get("calibration_file"),
        "what_alpha_scales": fit_doc.get("what_alpha_scales"),
        "alphas": fit_doc.get("alphas"),
        "fits": {
            group: {
                k: fit.get(k)
                for k in ("alpha", "ci", "ci_scaled", "chi2_red", "dof", "n_pubs",
                          "at_grid_edge", "loo")
            }
            for group, fit in (fit_doc.get("fits") or {}).items()
        },
    }


def config_block(
    manifest: Mapping[str, Any],
    usage: Mapping[str, Any] | None,
    fetch: Mapping[str, Any] | None,
    hw: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """
    The run configuration (Table "hardware run configuration"), from the files.

    Every entry is read from the manifest, ``usage.json``, the calibration
    fetch report or a hardware record, never typed in.
    """
    settings = manifest["settings"]
    options = manifest["sampler"]["options"]
    twirling = options.get("twirling") or {}
    any_record = next(iter(hw.values()))
    job_id = any_record.get("job_id")
    job_usage = (usage or {}).get(str(job_id)) or {}
    timestamps = _get(job_usage, "metrics", "timestamps") or {}
    used = sorted({int(q) for c in manifest["circuits"] for q in c["physical_qubits"]})
    roles: dict[str, int] = dict(manifest["totals"].get("by_role") or {})
    estimate = manifest.get("estimate") or {}
    return {
        "backend": manifest["backend"]["name"],
        "processor_family": manifest["backend"].get("processor_family"),
        "num_qubits": manifest["backend"].get("num_qubits"),
        "heavy_hex": manifest["backend"].get("heavy_hex"),
        "qubits_used": len(used),
        "execution_mode": manifest["sampler"].get("execution_mode_default"),
        "job_id": job_id,
        "created_utc": timestamps.get("created"),
        "running_utc": timestamps.get("running"),
        "finished_utc": timestamps.get("finished"),
        "calibration_plan": manifest["backend"].get("calibration_timestamp"),
        "calibration_exec": (fetch or {}).get("calibration_timestamp"),
        "optimization_level": settings.get("optimization_level"),
        "layout_method": settings.get("layout_method"),
        "seed_min": settings.get("seed_min"),
        "seed_max_inclusive": int(settings["seed_max"]) - 1,
        "sweep_layout": sorted({
            str(_get(r, "transpiled", "seed_source")) for r in hw.values()
        }),
        "scales": settings.get("scales"),
        "fold": "ISA-level gate folding (dgssp fold_isa), submitted without re-transpiling",
        "shots": settings.get("shots"),
        "circuits": manifest["totals"].get("circuits"),
        "circuits_by_role": roles,
        "total_shots": manifest["totals"].get("shots"),
        "gate_twirling": bool(twirling.get("enable_gates")),
        "measure_twirling": bool(twirling.get("enable_measure")),
        "twirling_randomizations": twirling.get("num_randomizations"),
        "twirling_shots_per_randomization": twirling.get("shots_per_randomization"),
        "twirling_strategy": twirling.get("strategy"),
        "dynamical_decoupling": bool((options.get("dynamical_decoupling") or {}).get("enable")),
        "zne_method": settings["zne"].get("method"),
        "zne_asymptote": settings["zne"].get("asymptote"),
        "zne_bootstrap": settings["zne"].get("n_bootstrap"),
        "qpu_estimate_s": estimate.get("estimated_seconds"),
        "qpu_estimate_safety_factor": estimate.get("safety_factor"),
        "qpu_charged_s": job_usage.get("usage_seconds"),
        "packages": _get(any_record, "provenance", "packages"),
        "code_commit": _get(any_record, "provenance", "git", "short_commit"),
        "code_dirty": _get(any_record, "provenance", "git", "dirty"),
    }


def build_summary(inputs: HwInputs) -> dict[str, Any]:
    """
    Every number the hardware tables and figures need, from the files alone.

    Raises
    ------
    FileNotFoundError
        Without the manifest or the hardware records.
    """
    for path in inputs.required():
        if not path.exists():
            raise FileNotFoundError(f"{rel(path)} not found; run hardware_batch.py first.")
    manifest = json.loads(inputs.paths.manifest.read_text(encoding="utf-8"))
    hw = load_records(inputs.paths.hardware_out_dir)
    plan = load_records(inputs.paths.dryrun_dir)
    exec_ = load_records(inputs.predict_dir)
    fit_doc = _load_json(inputs.alpha_fit)
    alpha_hat = _get(fit_doc, "fits", ALPHA_GROUP, "alpha")
    alpha_pred = alpha_predictions(fit_doc, alpha_hat)

    records = [
        record_entry(row, hw[row["key"]], plan.get(row["key"]), exec_.get(row["key"]),
                     alpha_pred)
        for row in manifest["records"]
        if row["kind"] == "zne" and row["key"] in hw
    ]
    statuses = {s: sum(1 for r in records if r["zne_status"] == s) for s in STATUSES}
    return {
        "schema": SUMMARY_SCHEMA,
        "date": inputs.date,
        "resolved_rule": RESOLVED_RULE,
        "alpha_group": ALPHA_GROUP,
        "alpha_hat": alpha_hat,
        "records": records,
        "statuses": statuses,
        "limit_band": limit_band(records),
        "drift": drift_entries(manifest, hw),
        "alpha": {
            EXEC_LABEL: alpha_block(fit_doc),
            "plan": alpha_block(_load_json(inputs.alpha_fit_plan)),
        },
        "config": config_block(manifest, _load_json(inputs.paths.usage_json),
                               _load_json(inputs.fetch_report), hw),
        "inputs": inputs.hashed(),
        "provenance": record_provenance(),
    }


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


def _tex(text: Any) -> str:
    """Escape the LaTeX specials a record key or setting can contain."""
    return str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace(
        "&", "\\&").replace("%", "\\%").replace("#", "\\#")


def _when(text: Any, *, utc: bool = False) -> str:
    """An ISO time as ``2026-10-26 06:05:15`` (``UTC`` appended when ``utc``)."""
    if not text:
        return "---"
    parsed = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if utc:
        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    offset = parsed.strftime("%z")
    offset = f"{offset[:3]}:{offset[3:]}" if offset else ""
    return parsed.strftime("%Y-%m-%d %H:%M:%S") + (f" (UTC{offset})" if offset else "")


def _num(value: Any, digits: int = 3) -> str:
    """A number at fixed precision, ``---`` when missing or not finite."""
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "---"
    return f"{value:.{digits}f}"


#: Short forms of :data:`STATUSES` for the table, explained in its footer.
STATUS_SHORT: dict[str, str] = {
    "resolved": "res.",
    "identified, not resolved": "id.",
    "not identifiable": "n.i.",
    "no signal": "---",
}


def results_table(summary: Mapping[str, Any]) -> str:
    """
    Table "hardware results per record", as LaTeX.

    Columns: record, ``n``, ``M``, ``r``, two-qubit gates, ``P_uniform``,
    ideal, the execution-calibration model, hardware at ``lambda = 1`` with
    its 95 % interval, our scalar ZNE (the extrapolation with its interval
    when resolved, else the ``lambda = 1`` value it falls back to), its status,
    and Mitiq on the same counts.
    """
    def order(r: Mapping[str, Any]) -> tuple[Any, ...]:
        return (r["role"] != "subset", r["n_items"], r["instance"], r["M"], r["iterations"])

    lines = [
        "% Generated by paper/scripts/hw_aggregate.py -- do not edit.",
        f"% job date: {summary['date']}; flags under rule {summary['resolved_rule']}",
        "\\begin{tabular}{lrrrrrrrlllr}",
        "\\hline",
        "Record & $n$ & $M$ & $r$ & 2q & $P_\\mathrm{unif}$ & Ideal & Model "
        "& Hardware $\\lambda{=}1$ & ZNE (ours) & & Mitiq \\\\",
        "\\hline",
    ]
    role = None
    for r in sorted(summary["records"], key=order):
        if role is not None and r["role"] != role:
            lines.append("\\hline")
        role = r["role"]
        hardware = (f"{_num(r['hardware'])} [{_num(r['hardware_lo'])}, "
                    f"{_num(r['hardware_hi'])}]")
        if r["zne_status"] == "resolved":
            zne = (f"{_num(r['zne_reported'])} [{_num(r['zne_reported_lo'])}, "
                   f"{_num(r['zne_reported_hi'])}]")
        else:
            zne = f"{_num(r['zne_reported'])}$^\\dagger$"
        mitiq = r["mitiq"]
        mitiq_text = _num(mitiq)
        if isinstance(mitiq, (int, float)) and not 0.0 <= mitiq <= 1.0:
            mitiq_text += "$^\\ddagger$"
        lines.append(
            " & ".join([
                _tex(r["key"]), str(r["n_items"]), str(r["M"]), str(r["iterations"]),
                str(r["two_q_count"]), _num(r["p_uniform"]), _num(r["p_ideal"]),
                _num(r["model_exec"]), hardware, zne,
                STATUS_SHORT.get(str(r["zne_status"]), "---"), mitiq_text,
            ])
            + " \\\\"
        )
    alpha = summary.get("alpha_hat")
    lines += [
        "\\hline",
        "\\end{tabular}",
        "% Model: the same circuits on the noise model of the calibration in force",
        "% during the job (errors x 1"
        + (f"; the fitted scale is alpha = {alpha:.2f})." if alpha else ")."),
        "% res. = resolved and identified: the extrapolation is reported, with its",
        "%   bootstrap 95% interval. id. = identified but not resolved; n.i. = not",
        "%   identifiable; --- = no signal (M = 0 or M = N/2).",
        "% dagger = reported at lambda = 1 (the estimator declines to extrapolate).",
        "% double dagger = outside [0, 1].",
    ]
    return "\n".join(lines) + "\n"


def config_table(summary: Mapping[str, Any]) -> str:
    """Table "hardware run configuration", as LaTeX (two columns)."""
    c = summary["config"]
    roles = c.get("circuits_by_role") or {}
    role_text = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in roles.items())
    packages = c.get("packages") or {}
    software = ", ".join(
        f"{name} {packages[name]}"
        for name in ("qiskit", "qiskit-ibm-runtime", "qiskit-aer", "mitiq")
        if name in packages
    )
    alpha = (summary.get("alpha") or {}).get(EXEC_LABEL) or {}
    fit = (alpha.get("fits") or {}).get(summary.get("alpha_group") or ALPHA_GROUP) or {}
    ci = fit.get("ci_scaled") or [None, None]
    family = str(c.get("processor_family") or "?").capitalize()
    device = f"{_tex(c['backend'])} ({family}, {c.get('num_qubits')} qubits"
    device += ", heavy-hex)" if c.get("heavy_hex") else ")"
    twirl = (
        f"gate twirling, {c.get('twirling_randomizations')} randomizations $\\times$ "
        f"{c.get('twirling_shots_per_randomization')} shots ({_tex(c.get('twirling_strategy'))})"
        if c.get("gate_twirling") else "no gate twirling"
    )
    twirl += "; measurement twirling " + ("on" if c.get("measure_twirling") else "off")
    twirl += "; dynamical decoupling " + ("on" if c.get("dynamical_decoupling") else "off")
    twirl += "; no readout mitigation"
    layouts = c.get("sweep_layout") or []
    layout_text = (
        f"optimization level {c.get('optimization_level')}, {c.get('layout_method')} "
        f"layout, seeds {c.get('seed_min')}--{c.get('seed_max_inclusive')} "
        "(best estimated survival), once per instance on the live target"
    )
    if "sweep_fixed_layout" in layouts:
        layout_text += "; iteration-sweep variants pinned to the $r_\\mathrm{opt}$ layout"
    scales = ", ".join(f"{s:g}" for s in c.get("scales") or [])
    rows = [
        ("Device", device),
        ("Qubits used", f"{c.get('qubits_used')} physical qubits across all circuits"),
        ("Job", f"{_tex(c.get('job_id'))}, {_tex(c.get('execution_mode'))} mode; ran "
                f"{_when(c.get('running_utc'), utc=True)} to "
                f"{_when(c.get('finished_utc'), utc=True)}"),
        ("Calibration", f"plan {_when(c.get('calibration_plan'))}; in force during the job "
                        f"{_when(c.get('calibration_exec'))}"),
        ("Transpilation", layout_text),
        ("Noise scaling", f"$\\lambda \\in \\{{{scales}\\}}$, {_tex(c.get('fold'))}"),
        ("Circuits and shots", f"{c.get('circuits')} circuits ({role_text}) $\\times$ "
                               f"{c.get('shots')} shots = {c.get('total_shots')} shots"),
        ("Error suppression", twirl),
        ("Extrapolation", f"{_tex(c.get('zne_method'))}, asymptote fixed at "
                          f"$P_\\mathrm{{unif}}$; {c.get('zne_bootstrap')} bootstrap resamples; "
                          f"flags rule {summary['resolved_rule']}"),
        ("QPU time", f"estimated {_num(c.get('qpu_estimate_s'), 1)} s (safety factor "
                     f"{c.get('qpu_estimate_safety_factor')}); charged {c.get('qpu_charged_s')} s"),
        ("Noise-model gap", f"errors $\\times\\,\\hat\\alpha$ = {_num(fit.get('alpha'), 2)} "
                            f"[{_num(ci[0], 2)}, {_num(ci[1], 2)}] reproduce the hardware"),
        ("Software", software),
        ("Code", f"commit {_tex(c.get('code_commit'))}"
                 + (" (dirty)" if c.get("code_dirty") else "")),
    ]
    lines = [
        "% Generated by paper/scripts/hw_aggregate.py -- do not edit.",
        f"% job date: {summary['date']}",
        "\\begin{tabular}{lp{0.72\\linewidth}}",
        "\\hline",
    ]
    lines += [f"{name} & {value} \\\\" for name, value in rows]
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def default_date(hardware_dir: Path) -> str | None:
    """The job of record if its manifest is there, else the newest manifest."""
    if (hardware_dir / f"manifest_{JOB_OF_RECORD}.json").exists():
        return JOB_OF_RECORD
    return latest_manifest_date(hardware_dir)


def output_suffix(date: str) -> str:
    """``""`` for the job of record (the paper's files), ``"_<date>"`` otherwise."""
    return "" if date == JOB_OF_RECORD else f"_{date}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--date", default=None,
                        help=f"job date (default: the job of record, {JOB_OF_RECORD}, "
                             "else the newest manifest)")
    parser.add_argument("--hardware-dir", type=Path, default=HARDWARE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--tex-dir", type=Path, default=TEX_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Build the hardware tables.

    Returns
    -------
    int
        ``0`` on success, ``1`` when there is no job to aggregate.
    """
    args = parse_args(argv)
    date = args.date or default_date(args.hardware_dir)
    if date is None:
        print(f"no manifest_<date>.json in {args.hardware_dir}; nothing to aggregate")
        return 1
    inputs = HwInputs(date, args.hardware_dir, args.data_dir)
    try:
        summary = build_summary(inputs)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    missing = [rel(p) for p in (inputs.paths.dryrun_dir, inputs.predict_dir, inputs.alpha_fit)
               if not p.exists()]
    rows = [{k: v for k, v in r.items() if k != "detail"} for r in summary["records"]]
    suffix = output_suffix(date)
    written = [
        write_csv(args.data_dir / f"hw_records{suffix}.csv", rows),
        write_json(args.data_dir / f"hw_summary{suffix}.json", summary),
    ]
    args.tex_dir.mkdir(parents=True, exist_ok=True)
    for name, text in ((f"hw_table_results{suffix}.tex", results_table(summary)),
                       (f"hw_table_config{suffix}.tex", config_table(summary))):
        path = args.tex_dir / name
        path.write_text(text, encoding="utf-8")
        written.append(path)

    print(f"job {date}: {len(summary['records'])} records; statuses {summary['statuses']}")
    alpha = summary.get("alpha_hat")
    print("alpha_hat = " + (f"{alpha:.3f}" if alpha else "n/a (no fit.json)"))
    print(f"{'record':<16} {'2q':>5} {'ideal':>6} {'model':>6} {'hw':>6} "
          f"{'kept hw':>7} {'ZNE':>7}  status")
    for r in summary["records"]:
        kept = r["kept_hardware"]
        print(f"{r['key']:<16} {r['two_q_count']:>5} {_num(r['p_ideal']):>6} "
              f"{_num(r['model_exec']):>6} {_num(r['hardware']):>6} "
              f"{_num(kept) if kept is not None else '-':>7} {_num(r['zne_reported']):>7}  "
              f"{r['zne_status']}")
    band = summary["limit_band"]
    if band:
        print(f"limit band: {band['lower']} ({band['lower_key']}) to {band['upper']} "
              f"({band['upper_key']}) two-qubit gates; exceptions {band['exceptions']}")
    for d in summary["drift"]:
        print(f"drift {d['key']}: {d['p_first']:.4f} -> {d['p_repeat']:.4f} "
              f"(z = {d['z']:+.2f}, p = {d['p_value']:.3f})")
    if missing:
        print("  ! not found (columns left empty): " + ", ".join(missing))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
