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
    The fitted error scale alpha, and each PUB's prediction on the alpha grid
    (the grid's cached counts, ``counts_alpha_*.json``, give each PUB's whole
    distribution at alpha-hat).
``paper/data/hw_predict_<label>_x<alpha>_<date>/``
    Only for a job without its own alpha fit (a later job): the prediction at
    a scaled alpha made for it, which then stands in the "model at alpha"
    columns -- an out-of-sample test of alpha-hat.

When ``hw_predict_exec_<date>`` does not exist (the calibration in force while
the job ran was not fetched), the "model" column falls back to the plan's
calibration (the dry run) and says so (``model_source``).

and writes:

``paper/data/hw_records.csv``
    One row per record: ``P(solution)`` on hardware and on each model,
    "signal kept", the ZNE estimate with its status, Mitiq, the ``M = 0``
    distances.
``paper/data/hw_summary.json``
    The same per record plus what the figures need (each scale's
    observations, the fitted curve, the worked example's distributions), the
    drift check with a bootstrap interval and a shot-noise null on the
    distance between the repeats (:func:`tv_bootstrap`), the alpha fit, the
    run configuration, the two-qubit band where ZNE stops working
    (:func:`limit_band`), model vs hardware on every PUB at alpha = 1 and at
    alpha-hat (:func:`pub_rows`, :func:`deviation_stats`), the survival
    estimate against the measured signal and the band it predicts
    (:func:`survival_block`), and provenance.
    ``make_figures.py --set hw`` and ``resource_scaling.py --figure-only``
    read this file and nothing else from the job.
``paper/tex/hw_table_results.tex``
    Table "hardware results per record".
``paper/tex/hw_table_config.tex``
    Table "hardware run configuration".

With ``--grids <date>`` it writes only ``paper/tex/hw_table_grids.tex``
(:func:`grid_table`): a later job's records beside the same records of the job
of record, and, with ``--grids-prediction``, the model prediction made for the
later job before it ran.

The flags are re-derived, not read: ``resolved``, ``identifiable`` and the
reported value follow the library's current rule via
:func:`aggregate.rederive_flags`, exactly as the simulation tables do.

"Signal kept" is ``(P_sol - P_uniform) / (P_ideal - P_uniform)`` at
``lambda = 1``: 1 is the noiseless answer, 0 is random guessing.  It is
undefined for ``M = 0`` and ``M = N/2``, whose ideal *is* random guessing.

Run::

    python paper/scripts/hw_aggregate.py
    python paper/scripts/hw_aggregate.py --date 2026-10-25
    python paper/scripts/hw_aggregate.py --grids 2026-10-28 --grids-prediction paper/data/hw_predict_plan_x3.176_2026-10-28

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
from dgssp.backends import ERROR_CLASSES  # noqa: E402
from dgssp.metrics import (  # noqa: E402
    distribution_solution_probability,
    percentile_interval,
    survival_estimate,
    total_variation,
)
from dgssp.mitigation.zne import (  # noqa: E402
    RESOLVABILITY_K,
    RESOLVED_RULE,
    plan_scales,
    signal_direction,
)
from hardware_batch import (  # noqa: E402
    Paths,
    drift_metrics,
    latest_manifest_date,
    rel,
    sha256_file,
    write_json,
)
from noise_model_check import median_errors, signal_kept  # noqa: E402

SUMMARY_SCHEMA = "dgssp/paper/hardware-summary/1"

#: The job the paper's hardware tables and figures are of.  Without ``--date``
#: this one is aggregated, not the newest manifest: planning a later job must
#: not change the paper.  Any other date writes its files with a ``_<date>``
#: suffix (``hw_summary_<date>.json``, ...), beside the paper's.
JOB_OF_RECORD = "2026-10-25"

#: The supplement table that sets a later job beside the job of record.
GRID_TABLE = "hw_table_grids.tex"

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

#: Resamples (and seed) of the total-variation checks: the drift interval and
#: null, and the shot-noise floor of D_TV(hardware, model).
TV_BOOTSTRAP_B = 2000
TV_BOOTSTRAP_SEED = 0


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
        self.alpha_cache = self.alpha_fit.parent
        self.fetch_report = hardware_dir / f"calibration_{EXEC_LABEL}_{date}_fetch.json"
        self.exec_calibration = hardware_dir / f"calibration_{EXEC_LABEL}_{date}.json"

    def required(self) -> list[Path]:
        """The files without which no table can be built."""
        return [self.paths.manifest, self.paths.hardware_out_dir]

    def model_source(self) -> tuple[str, Path]:
        """
        The model that stands beside the hardware at ``alpha = 1``.

        The execution calibration's prediction when it was made, else the
        plan calibration's dry run.  One source for every record of the job,
        never a mix.
        """
        if self.predict_dir.exists():
            return EXEC_LABEL, self.predict_dir
        return "plan", self.paths.dryrun_dir

    def calibration(self) -> Path:
        """The snapshot the survival estimate reads, chosen as :meth:`model_source` is."""
        return self.exec_calibration if self.predict_dir.exists() else self.paths.calibration

    def alpha_prediction_dirs(self) -> list[Path]:
        """
        ``hw_predict_<label>_x<alpha>_<date>``: predictions made at a scaled alpha.

        Every-error predictions only.  A ``--scale-only <class>`` prediction
        (``..._x<alpha>_<class>_<date>``) is a robustness check of which errors
        carry alpha (``alpha_robustness.py``), not this job's model.
        """
        data_dir = self.paths.data_dir
        classes = tuple(f"_{c}_{self.date}" for c in ERROR_CLASSES if c != "all")
        return sorted(p for p in data_dir.glob(f"hw_predict_*_x*_{self.date}")
                      if p.is_dir() and not p.name.endswith(classes))

    def hashed(self) -> dict[str, str | None]:
        """``{repo path: sha256}`` of every input file (directories: their records)."""
        files = [self.paths.manifest, self.paths.usage_json, self.fetch_report,
                 self.alpha_fit, self.alpha_fit_plan, self.calibration()]
        files += sorted(self.alpha_cache.glob("counts_alpha_*.json"))
        for directory in (self.paths.hardware_out_dir, self.paths.dryrun_dir, self.predict_dir,
                          *self.alpha_prediction_dirs()):
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


def _counts_at(record: Mapping[str, Any] | None, scale: float) -> dict[str, int] | None:
    """
    A record's counts at one *realised* scale.

    The records key their counts by the realised scale as text: ``"3"`` on
    the odd grid, but ``"1.2535211267605635"`` where a fractional fold could
    not hit the nominal ``1.25`` exactly.  So the lookup is by value.
    """
    for text, counts in (_get(record, "counts") or {}).items():
        if math.isclose(float(text), scale, rel_tol=1e-9):
            return dict(counts)
    return None


def _realised(circuit: Mapping[str, Any]) -> float:
    """A manifest circuit's realised scale (its nominal one when not recorded)."""
    return float(circuit.get("realised_scale") or circuit["nominal_scale"])


def record_entry(
    row: Mapping[str, Any],
    hw: Mapping[str, Any],
    plan: Mapping[str, Any] | None,
    exec_: Mapping[str, Any] | None,
    alpha_pred: Mapping[tuple[str, float], float],
    *,
    model_source: str = EXEC_LABEL,
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
    model_source:
        Which of ``plan`` / ``exec_`` is "the model" (:meth:`HwInputs.model_source`).

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
    model = p_sol(exec_ if model_source == EXEC_LABEL else plan)
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
        "model": model,
        "model_source": model_source,
        "kept_hardware": kept(p_sol(hw)),
        "kept_model_plan": kept(p_sol(plan)),
        "kept_model_exec": kept(p_sol(exec_)),
        "kept_model_alpha": kept(alpha_1),
        "kept_model": kept(model),
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


def prediction_entries(
    manifest: Mapping[str, Any], predicted: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """
    A model run's ``zne`` records, flagged exactly as the hardware's are.

    Parameters
    ----------
    manifest:
        The job's manifest (its ``records`` say which keys are ``zne``).
    predicted:
        :func:`load_records` of ``hw_predict_<label>_<date>``.

    Returns
    -------
    dict
        ``{record key: record_entry}`` with the model record in the hardware
        slot, so its status and interval follow the same rule.
    """
    return {
        row["key"]: record_entry(row, predicted[row["key"]], None, None, {})
        for row in manifest["records"]
        if row["kind"] == "zne" and row["key"] in predicted
    }


def drift_entries(
    manifest: Mapping[str, Any], hw: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """
    The drift check: each scale-1 repeat against its first copy.

    Uses :func:`hardware_batch.drift_metrics` (two-proportion z-test and the
    total-variation distance between the two histograms) and
    :func:`tv_bootstrap` (that distance's interval and its shot-noise null).
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
        out.append({"key": row["key"], "repeat_of": first_key, **metrics,
                    **tv_bootstrap(first, repeat)})
    return out


# ---------------------------------------------------------------------------
# Total-variation distance with its shot noise
# ---------------------------------------------------------------------------


def _tv_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Total-variation distance between aligned probability vectors (last axis)."""
    return np.asarray(0.5 * np.abs(p - q).sum(axis=-1))


def tv_null(
    probs: np.ndarray, n1: int, n2: int, rng: np.random.Generator, size: int
) -> np.ndarray:
    """
    Distances between two independent samples of one distribution.

    What D_TV is when nothing differs: two histograms of ``n1`` and ``n2``
    shots drawn from ``probs`` are still a positive distance apart.

    Returns
    -------
    numpy.ndarray
        ``size`` draws of the distance.
    """
    a = rng.multinomial(n1, probs, size=size) / n1
    b = rng.multinomial(n2, probs, size=size) / n2
    return _tv_rows(a, b)


def tv_bootstrap(
    first: Mapping[str, int],
    second: Mapping[str, int],
    *,
    B: int = TV_BOOTSTRAP_B,
    seed: int = TV_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """
    The distance between two histograms, its interval, and what shot noise alone gives.

    D_TV between two finite samples is positive even when nothing changed,
    and its bootstrap interval sits above zero for the same reason, so the
    interval alone cannot say "no drift".  The null answers that: both
    histograms drawn from their pooled distribution (nothing changed), and
    the share of those draws at least as far apart as the observed pair.

    Returns
    -------
    dict
        ``tv`` (as :func:`hardware_batch.drift_metrics`' ``tv_between``),
        ``tv_ci`` (percentile bootstrap 95 %, each histogram resampled from
        itself), ``tv_null_mean``, ``tv_null_q95``, ``tv_p_value``
        (``(1 + #null >= tv) / (B + 1)``), ``tv_B``, ``tv_seed``.
    """
    keys = sorted(set(first) | set(second))
    c1 = np.array([first.get(k, 0) for k in keys], dtype=float)
    c2 = np.array([second.get(k, 0) for k in keys], dtype=float)
    n1, n2 = int(c1.sum()), int(c2.sum())
    tv = float(_tv_rows(c1 / n1, c2 / n2))
    rng = np.random.default_rng(seed)
    boot = _tv_rows(rng.multinomial(n1, c1 / n1, size=B) / n1,
                    rng.multinomial(n2, c2 / n2, size=B) / n2)
    null = tv_null((c1 + c2) / (n1 + n2), n1, n2, rng, B)
    lo, hi = percentile_interval(boot)
    return {
        "tv": tv,
        "tv_ci": [lo, hi],
        "tv_null_mean": float(null.mean()),
        "tv_null_q95": float(np.percentile(null, 95)),
        "tv_p_value": float((1 + np.count_nonzero(null >= tv)) / (B + 1)),
        "tv_B": B,
        "tv_seed": seed,
    }


# ---------------------------------------------------------------------------
# Model vs hardware, every PUB
# ---------------------------------------------------------------------------


def record_distributions(
    manifest: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]
) -> dict[tuple[str, float], dict[str, float]]:
    """``{(record key, nominal scale): distribution}`` of every PUB a record directory holds."""
    out: dict[tuple[str, float], dict[str, float]] = {}
    for c in manifest["circuits"]:
        dist = probabilities(_counts_at(records.get(c["key"]), _realised(c)))
        if dist:
            out[(c["key"], float(c["nominal_scale"]))] = dist
    return out


def alpha_distributions(
    manifest: Mapping[str, Any], cache: Path, alpha: float | None
) -> dict[tuple[str, float], dict[str, float]]:
    """
    Each PUB's model distribution at ``alpha``, between the alpha fit's grid points.

    Every outcome's probability is interpolated in alpha as
    :func:`alpha_predictions` interpolates ``P(solution)`` (monotone cubic
    through the cached grid ``counts_alpha_*.json``), clipped at zero and
    renormalised.  No new simulation: alpha-hat is not a grid point.

    Returns
    -------
    dict
        ``{(record key, nominal scale): distribution}``; empty without at
        least two cached grid points or without ``alpha``.
    """
    docs = [_load_json(p) for p in sorted(cache.glob("counts_alpha_*.json"))]
    if alpha is None or len(docs) < 2:
        return {}
    from scipy.interpolate import PchipInterpolator

    docs.sort(key=lambda d: float(d["alpha"]))
    alphas = np.array([float(d["alpha"]) for d in docs])
    out: dict[tuple[str, float], dict[str, float]] = {}
    for c in manifest["circuits"]:
        per_alpha = [d["counts"][int(c["pub_index"])] for d in docs]
        keys = sorted({k for counts in per_alpha for k in counts})
        grid = np.array([[counts.get(k, 0) / sum(counts.values()) for k in keys]
                         for counts in per_alpha])
        p = np.clip(PchipInterpolator(alphas, grid, axis=0)(alpha), 0.0, None)
        out[(c["key"], float(c["nominal_scale"]))] = dict(zip(keys, (p / p.sum()).tolist(),
                                                               strict=True))
    return out


def alpha_model(
    inputs: HwInputs, manifest: Mapping[str, Any], fit_doc: Mapping[str, Any] | None
) -> tuple[dict[str, Any] | None, dict[tuple[str, float], float],
           dict[tuple[str, float], dict[str, float]]]:
    """
    The model at a scaled alpha that stands beside this job.

    The job's own alpha fit when there is one (in-sample: alpha-hat was fitted
    to these very PUBs); otherwise the single ``hw_predict_*_x<alpha>_<date>``
    prediction made for it (out-of-sample: alpha came from another job).

    Returns
    -------
    tuple
        ``(info, {(key, scale): P(solution)}, {(key, scale): distribution})``;
        ``(None, {}, {})`` when there is neither, or several predictions.
    """
    alpha_hat = _get(fit_doc, "fits", ALPHA_GROUP, "alpha")
    if fit_doc and alpha_hat is not None:
        info = {"source": "fit", "alpha": alpha_hat, "in_sample": True,
                "path": rel(inputs.alpha_fit), "calibration": fit_doc.get("calibration_file")}
        return (info, alpha_predictions(fit_doc, alpha_hat),
                alpha_distributions(manifest, inputs.alpha_cache, alpha_hat))
    dirs = inputs.alpha_prediction_dirs()
    if len(dirs) != 1:
        return None, {}, {}
    records = load_records(dirs[0])
    first = next(iter(records.values()), None)
    prediction = _get(first, "metrics", "prediction") or {}
    sols = {row["key"]: list(row["solution_bitstrings"]) for row in manifest["records"]}
    dists = record_distributions(manifest, records)
    info = {"source": "prediction", "alpha": prediction.get("alpha"), "in_sample": False,
            "path": rel(dirs[0]), "calibration": prediction.get("calibration_file"),
            "made_utc": _get(first, "provenance", "timestamp_utc")}
    p_sol = {k: distribution_solution_probability(d, sols[k[0]]) for k, d in dists.items()}
    return info, p_sol, dists


def _tv(p: Mapping[str, float] | None, q: Mapping[str, float] | None) -> float | None:
    """:func:`dgssp.metrics.total_variation`, ``None`` when either side is missing."""
    return total_variation(p, q) if p and q else None


def pub_rows(
    manifest: Mapping[str, Any],
    hw: Mapping[str, Mapping[str, Any]],
    model_1: Mapping[tuple[str, float], Mapping[str, float]],
    alpha_p_sol: Mapping[tuple[str, float], float],
    alpha_dists: Mapping[tuple[str, float], Mapping[str, float]],
    *,
    B: int = TV_BOOTSTRAP_B,
    seed: int = TV_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """
    Model vs hardware on every PUB, folded scales and drift repeats included.

    Parameters
    ----------
    manifest, hw:
        The job's manifest and hardware records.
    model_1:
        :func:`record_distributions` of the ``alpha = 1`` model
        (:meth:`HwInputs.model_source`).
    alpha_p_sol, alpha_dists:
        The model at alpha (:func:`alpha_model`).  ``P(solution)`` comes from
        ``alpha_p_sol`` -- for a fitted alpha, exactly the fit's interpolation
        -- and the distances from ``alpha_dists``.

    Returns
    -------
    list[dict]
        One row per PUB: ``P(solution)`` on hardware and both models and the
        deviations ``hardware - model`` (``dev_1``, ``dev_alpha``); D_TV to
        the ideal distribution of each and its deviation (``dtv_1``,
        ``dtv_alpha``); D_TV between hardware and each model with its
        shot-noise floor (mean distance of two samples of the model's
        distribution, :func:`tv_null`); ``signal`` (False for ``M = 0`` and
        ``M = N/2``, where ``P(solution)`` says nothing about the noise).
    """
    rng = np.random.default_rng(seed)
    rows_by_key = {row["key"]: row for row in manifest["records"]}

    def floor(dist: Mapping[str, float] | None, shots: int) -> float | None:
        if not dist:
            return None
        probs = np.array(list(dist.values()), dtype=float)
        return float(tv_null(probs / probs.sum(), shots, shots, rng, B).mean())

    def diff(a: float | None, b: float | None) -> float | None:
        return a - b if a is not None and b is not None else None

    out: list[dict[str, Any]] = []
    for c in sorted(manifest["circuits"], key=lambda c: int(c["pub_index"])):
        key, scale = c["key"], float(c["nominal_scale"])
        record = hw.get(key)
        counts = _counts_at(record, _realised(c))
        if not counts or record is None:
            continue
        row = rows_by_key[key]
        sols = list(row["solution_bitstrings"])
        p_uniform, p_ideal = float(row["p_uniform"]), exact_p_ideal(dict(record))
        ideal = _get(record, "metrics", "reference_distribution")
        shots = int(sum(counts.values()))
        hw_dist = probabilities(counts) or {}
        m1, ma = model_1.get((key, scale)), alpha_dists.get((key, scale))

        p_hw = distribution_solution_probability(hw_dist, sols)
        p_1 = distribution_solution_probability(m1, sols) if m1 else None
        p_a = alpha_p_sol.get((key, scale))
        tv_hw, tv_1, tv_a = (_tv(d, ideal) for d in (hw_dist, m1, ma))
        out.append({
            "pub": int(c["pub_index"]),
            "key": key,
            "scale": scale,
            "realised_scale": _realised(c),
            "two_q_count": c.get("two_q_count"),
            "shots": shots,
            "signal": p_ideal is not None and signal_direction(p_ideal, p_uniform) != 0,
            "p_uniform": p_uniform,
            "p_ideal": p_ideal,
            "hardware": p_hw,
            "model_1": p_1,
            "model_alpha": p_a,
            "dev_1": diff(p_hw, p_1),
            "dev_alpha": diff(p_hw, p_a),
            "tv_ideal_hardware": tv_hw,
            "tv_ideal_model_1": tv_1,
            "tv_ideal_model_alpha": tv_a,
            "dtv_1": diff(tv_hw, tv_1),
            "dtv_alpha": diff(tv_hw, tv_a),
            "tv_hardware_model_1": _tv(hw_dist, m1),
            "tv_hardware_model_alpha": _tv(hw_dist, ma),
            "tv_floor_1": floor(m1, shots),
            "tv_floor_alpha": floor(ma, shots),
        })
    return out


def deviation_stats(
    rows: Sequence[Mapping[str, Any]], field: str, *, signal_only: bool
) -> dict[str, dict[str, Any]]:
    """
    Mean absolute and mean signed ``field`` per nominal scale and over all PUBs.

    ``n_negative`` / ``n_positive`` and a two-sided sign test say whether the
    deviation has a systematic sign.

    Returns
    -------
    dict
        ``{"all" | "lambda=<scale>": {"n", "mean_abs", "mean_signed",
        "n_negative", "n_positive", "sign_test_p"}}``; groups without values
        are left out.
    """
    from scipy.stats import binomtest

    groups: dict[str, list[float]] = {"all": []}
    for r in rows:
        value = r.get(field)
        if value is None or (signal_only and not r["signal"]):
            continue
        groups["all"].append(float(value))
        groups.setdefault(f"lambda={r['scale']:g}", []).append(float(value))
    out: dict[str, dict[str, Any]] = {}
    for name, values in groups.items():
        if not values:
            continue
        v = np.array(values)
        neg, pos = int(np.count_nonzero(v < 0)), int(np.count_nonzero(v > 0))
        out[name] = {
            "n": int(v.size),
            "mean_abs": float(np.abs(v).mean()),
            "mean_signed": float(v.mean()),
            "n_negative": neg,
            "n_positive": pos,
            "sign_test_p": float(binomtest(neg, neg + pos).pvalue) if neg + pos else None,
        }
    return out


def model_vs_hardware(
    rows: Sequence[Mapping[str, Any]],
    model_source: str,
    alpha_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    The comparison the text quotes: :func:`deviation_stats` of every deviation.

    ``P(solution)`` only on PUBs with signal (``M = 0`` has no solution to
    count and ``M = N/2`` sits at ``P_uniform`` whatever the noise); the
    distances on every PUB.  Deviations are ``hardware - model``: negative
    ``P(solution)`` and positive D_TV-to-ideal mean the device is noisier
    than the model.
    """
    def mean_of(field: str) -> float | None:
        values = [float(r[field]) for r in rows if r.get(field) is not None]
        return float(np.mean(values)) if values else None

    return {
        "model_1": {"source": model_source, "alpha": 1.0},
        "model_alpha": dict(alpha_info) if alpha_info else None,
        "deviation": "hardware - model",
        "p_solution": {
            "alpha_1": deviation_stats(rows, "dev_1", signal_only=True),
            "alpha_hat": deviation_stats(rows, "dev_alpha", signal_only=True),
        },
        "tv_to_ideal": {
            "alpha_1": deviation_stats(rows, "dtv_1", signal_only=False),
            "alpha_hat": deviation_stats(rows, "dtv_alpha", signal_only=False),
        },
        "tv_hardware_model": {
            "alpha_1": {"mean": mean_of("tv_hardware_model_1"),
                        "mean_shot_floor": mean_of("tv_floor_1")},
            "alpha_hat": {"mean": mean_of("tv_hardware_model_alpha"),
                          "mean_shot_floor": mean_of("tv_floor_alpha")},
        },
        "pubs": [dict(r) for r in rows],
    }


# ---------------------------------------------------------------------------
# Survival estimate vs measured signal
# ---------------------------------------------------------------------------


def survival_rows(
    records: Sequence[Mapping[str, Any]],
    hw: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    errors: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    WP2's survival estimate per record beside the signal the hardware kept.

    ``S = (1 - eps2)**n2 * (1 - eps1)**n1`` (:func:`dgssp.metrics.survival_estimate`,
    physical gate counts of the ``lambda = 1`` circuit, median errors) is the
    signal kept a global-depolarising device would leave; a folded circuit
    keeps ``S**lambda`` (realised scale).  From ``S`` the pre-registration
    rule :func:`dgssp.mitigation.zne.plan_scales` (same shots, same
    resolvability ``k`` as the stored fit) gives the scales on which the
    signal should stay resolvable; two or more is what identifiability needs.

    Parameters
    ----------
    records:
        :func:`record_entry` of every ``zne`` record (no-signal ones skipped).
    hw, manifest:
        The hardware records and the manifest.
    errors:
        ``{label: median_gate_errors}``, e.g. ``{"alpha_1": ..., "alpha_hat": ...}``.

    Returns
    -------
    list[dict]
        Per record and label: ``survival_<label>``, ``kept_predicted_<label>``
        (per scale), ``planned_scales_<label>``,
        ``predicted_identifiable_<label>``; and the measured ``kept_observed``
        (per scale), ``n_informative_scales``, ``zne_status``, and
        ``exponent`` = ``ln(kept) / ln(S_alpha_1)`` at ``lambda = 1`` (the
        "signal = S**3" rule, per record).
    """
    realised = {(c["key"], float(c["nominal_scale"])): _realised(c) for c in manifest["circuits"]}
    out: list[dict[str, Any]] = []
    for r in records:
        if r.get("signal_direction") in (0, None) or r["key"] not in hw:
            continue
        record = hw[r["key"]]
        two_q, one_q = (_get(record, "transpiled", k) for k in ("two_q_physical", "one_q_physical"))
        if two_q is None or one_q is None:
            continue
        p_ideal, p_uniform = float(r["p_ideal"]), float(r["p_uniform"])
        scales = [float(s) for s in r["detail"]["scales"]]
        lams = [realised.get((r["key"], s), s) for s in scales]
        k = float(_get(record, "zne_fit", "resolvability_k") or RESOLVABILITY_K)
        shots = int(record.get("shots") or 0)
        kept_obs = [signal_kept(p, p_ideal, p_uniform) for p in r["detail"]["hardware"]]
        entry: dict[str, Any] = {
            "key": r["key"],
            "two_q_count": r["two_q_count"],
            "two_q_physical": two_q,
            "one_q_physical": one_q,
            "survival_stored": _get(record, "transpiled", "survival"),
            "scales": scales,
            "realised_scales": lams,
            "kept_observed": kept_obs,
            "n_informative_scales": r["n_informative_scales"],
            "zne_status": r["zne_status"],
        }
        for label, errs in errors.items():
            s = survival_estimate(int(two_q), int(one_q), errs)["survival"]
            planned = (plan_scales(s, p_ideal - p_uniform, shots, scales, k=k,
                                   p_uniform=p_uniform)
                       if s is not None and 0.0 < s <= 1.0 and shots > 0 else None)
            entry[f"survival_{label}"] = s
            entry[f"kept_predicted_{label}"] = [s**lam for lam in lams] if s is not None else None
            entry[f"planned_scales_{label}"] = planned
            entry[f"predicted_identifiable_{label}"] = (len(planned) >= 2
                                                       if planned is not None else None)
        s1, kept1 = entry.get("survival_alpha_1"), kept_obs[0] if kept_obs else None
        entry["exponent"] = (math.log(kept1) / math.log(s1)
                             if s1 is not None and 0.0 < s1 < 1.0
                             and kept1 is not None and 0.0 < kept1 < 1.0 else None)
        out.append(entry)
    return out


def survival_block(
    rows: Sequence[Mapping[str, Any]],
    errors: Mapping[str, Mapping[str, Any]],
    *,
    calibration: str,
    alpha: float | None,
    measured_band: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    :func:`survival_rows` plus the two-qubit band each survival label predicts.

    The predicted band is :func:`limit_band` run on the survival plan's
    verdicts instead of the measured statuses (identifiable stands in for
    resolved: it is what resolution needs at the least), so the two bands
    are read off by the same rule.
    """
    bands: dict[str, Any] = {"measured": dict(measured_band) if measured_band else None}
    agree: dict[str, Any] = {}
    for label in errors:
        verdicts = [{"key": r["key"], "two_q_count": r["two_q_count"],
                     "zne_status": "resolved" if r[f"predicted_identifiable_{label}"]
                     else "not identifiable"}
                    for r in rows if r.get(f"predicted_identifiable_{label}") is not None]
        bands[label] = limit_band(verdicts)
        agree[label] = sum(
            1 for r in rows if r.get(f"predicted_identifiable_{label}") is not None
            and r[f"predicted_identifiable_{label}"] == (r["n_informative_scales"] >= 2))
    return {
        "calibration": calibration,
        "alpha_hat": alpha,
        "medians": {label: {k: e.get(k) for k in ("two_qubit", "one_qubit", "readout")}
                    for label, e in errors.items()},
        "bands": bands,
        "identifiable_agreement": {label: {"agree": n, "of": len(rows)}
                                   for label, n in agree.items()},
        "records": [dict(r) for r in rows],
    }


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
    model_source, model_dir = inputs.model_source()
    fit_doc = _load_json(inputs.alpha_fit)
    alpha_hat = _get(fit_doc, "fits", ALPHA_GROUP, "alpha")
    alpha_info, alpha_pred, alpha_dists = alpha_model(inputs, manifest, fit_doc)

    records = [
        record_entry(row, hw[row["key"]], plan.get(row["key"]), exec_.get(row["key"]),
                     alpha_pred, model_source=model_source)
        for row in manifest["records"]
        if row["kind"] == "zne" and row["key"] in hw
    ]
    statuses = {s: sum(1 for r in records if r["zne_status"] == s) for s in STATUSES}
    band = limit_band(records)
    pubs = pub_rows(manifest, hw, record_distributions(manifest, load_records(model_dir)),
                    alpha_pred, alpha_dists)

    calibration = inputs.calibration()
    survival = None
    if calibration.exists():
        alpha = (alpha_info or {}).get("alpha")
        errors = {"alpha_1": median_errors(calibration, 1.0)}
        if alpha is not None:
            errors["alpha_hat"] = median_errors(calibration, float(alpha))
        survival = survival_block(survival_rows(records, hw, manifest, errors), errors,
                                  calibration=rel(calibration), alpha=alpha,
                                  measured_band=band)
    return {
        "schema": SUMMARY_SCHEMA,
        "date": inputs.date,
        "resolved_rule": RESOLVED_RULE,
        "alpha_group": ALPHA_GROUP,
        "alpha_hat": alpha_hat,
        "model_source": model_source,
        "model_alpha": alpha_info,
        "records": records,
        "statuses": statuses,
        "limit_band": band,
        "model_vs_hardware": model_vs_hardware(pubs, model_source, alpha_info),
        "survival": survival,
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


def _order(r: Mapping[str, Any]) -> tuple[Any, ...]:
    """Table order: instances before sweep variants, then by ``n``, instance, ``M``, ``r``."""
    return (r["role"] != "subset", r["n_items"], r["instance"], r["M"], r["iterations"])


def results_table(summary: Mapping[str, Any]) -> str:
    """
    Table "hardware results per record", as LaTeX.

    Columns: record, ``n``, ``M``, ``r``, two-qubit gates, ``P_uniform``,
    ideal, the execution-calibration model, hardware at ``lambda = 1`` with
    its 95 % interval, our scalar ZNE (the extrapolation with its interval
    when resolved, else the ``lambda = 1`` value it falls back to), its status,
    and Mitiq on the same counts.
    """
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
    for r in sorted(summary["records"], key=_order):
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
                _num(r["model"]), hardware, zne,
                STATUS_SHORT.get(str(r["zne_status"]), "---"), mitiq_text,
            ])
            + " \\\\"
        )
    alpha = summary.get("alpha_hat")
    model_note = (
        ["% Model: the same circuits on the noise model of the calibration in force",
         "% during the job (errors x 1"
         + (f"; the fitted scale is alpha = {alpha:.2f})." if alpha else ").")]
        if summary.get("model_source", EXEC_LABEL) == EXEC_LABEL else
        ["% Model: the same circuits on the noise model of the plan's calibration",
         "% (errors x 1; the calibration in force during the job was not fetched)."]
    )
    lines += [
        "\\hline",
        "\\end{tabular}",
        *model_note,
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
    if "pinned_layout" in layouts:
        layout_text += "; records suffixed with a layout tag pinned to an earlier job's layout and seed"
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


def _estimate(r: Mapping[str, Any] | None) -> str:
    """A ZNE cell: the extrapolation with its interval when resolved, else the status."""
    if r is None:
        return "---"
    if r["zne_status"] == "resolved":
        return (f"{_num(r['zne_reported'])} [{_num(r['zne_reported_lo'])}, "
                f"{_num(r['zne_reported_hi'])}]")
    return STATUS_SHORT.get(str(r["zne_status"]), "---")


def _grid(summary: Mapping[str, Any]) -> str:
    """A job's date and fold grid, as a column-group heading."""
    scales = ", ".join(f"{s:g}" for s in (summary.get("config") or {}).get("scales") or [])
    return f"{summary['date']}, $\\lambda \\in \\{{{scales}\\}}$"


def grid_table(
    of_record: Mapping[str, Any],
    later: Mapping[str, Any],
    predicted: Mapping[str, Mapping[str, Any]],
    *,
    prediction_source: str | None = None,
) -> str:
    """
    Table "fold grid: job of record vs a later job", as LaTeX (supplement).

    One row per record of ``later``; each job contributes its two-qubit count
    (the layout is searched again on each day's calibration), the signal kept
    at ``lambda = 1`` and the ZNE cell of :func:`_estimate`.  The last column
    is ``predicted``'s ZNE cell for the later job.

    Parameters
    ----------
    of_record, later:
        :func:`build_summary` of the two jobs.
    predicted:
        :func:`prediction_entries` for the later job (may be empty).
    prediction_source:
        One line saying what the prediction is, for the footer.
    """
    def kept(r: Mapping[str, Any] | None) -> str:
        text = _num(r["kept_hardware"], 2) if r else "---"
        return f"${text}$" if text.startswith("-") and text != "---" else text

    base = {r["key"]: r for r in of_record["records"]}
    lines = [
        "% Generated by paper/scripts/hw_aggregate.py --grids -- do not edit.",
        f"% jobs: {of_record['date']} (of record) and {later['date']}; "
        f"flags under rule {later['resolved_rule']}",
        "\\begin{tabular}{lrrrrrlrrll}",
        "\\hline",
        f" & & & & \\multicolumn{{3}}{{c}}{{{_grid(of_record)}}} "
        f"& \\multicolumn{{3}}{{c}}{{{_grid(later)}}} & Predicted \\\\",
        "Record & $M$ & $r$ & Ideal & 2q & Kept & ZNE & 2q & Kept & ZNE & ZNE \\\\",
        "\\hline",
    ]
    role = None
    for r in sorted(later["records"], key=_order):
        if role is not None and r["role"] != role:
            lines.append("\\hline")
        role = r["role"]
        old = base.get(r["key"])
        lines.append(
            " & ".join([
                _tex(r["key"]), str(r["M"]), str(r["iterations"]), _num(r["p_ideal"]),
                str(old["two_q_count"]) if old else "---", kept(old), _estimate(old),
                str(r["two_q_count"]), kept(r), _estimate(r),
                _estimate(predicted.get(r["key"])),
            ])
            + " \\\\"
        )
    lines += [
        "\\hline",
        "\\end{tabular}",
        "% Kept: signal kept at lambda = 1, (P - P_unif) / (P_ideal - P_unif).",
        "% ZNE: the extrapolation with its bootstrap 95% interval when resolved;",
        "%   id. = identified but not resolved; n.i. = not identifiable.",
        "% 2q differs between jobs: the layout is searched again on each day's calibration.",
        "% Predicted: " + (prediction_source or "none"),
    ]
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
    parser.add_argument("--grids", default=None, metavar="DATE",
                        help=f"write only tex/{GRID_TABLE}: this later job beside --date's")
    parser.add_argument("--grids-prediction", type=Path, default=None, metavar="DIR",
                        help="hw_predict_<label>_<date> made for the --grids job")
    return parser.parse_args(argv)


def prediction_note(
    predicted: Mapping[str, Mapping[str, Any]], directory: Path, running_utc: str | None
) -> str:
    """
    The footer line for a prediction: its model, calibration, and when it was made.

    "before the job ran" is stated only when the records' own timestamp is
    earlier than the job's start.
    """
    first = next(iter(predicted.values()))
    made = _get(first, "provenance", "timestamp_utc")
    note = (f"{first.get('backend')} noise model of the calibration "
            f"{_when(first.get('calibration_timestamp'))}, made {_when(made, utc=True)} "
            f"({rel(directory)})")
    if made and running_utc:
        start = datetime.fromisoformat(str(running_utc).replace("Z", "+00:00"))
        if datetime.fromisoformat(str(made).replace("Z", "+00:00")) < start:
            note += f"; before the job ran ({_when(running_utc, utc=True)})"
    return note


def write_grid_table(args: argparse.Namespace, date: str) -> int:
    """``--grids``: the later job beside the job ``date``; writes only :data:`GRID_TABLE`."""
    later_inputs = HwInputs(args.grids, args.hardware_dir, args.data_dir)
    try:
        of_record = build_summary(HwInputs(date, args.hardware_dir, args.data_dir))
        later = build_summary(later_inputs)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    predicted: dict[str, dict[str, Any]] = {}
    note = None
    if args.grids_prediction:
        raw = load_records(args.grids_prediction)
        if not raw:
            print(f"no records in {args.grids_prediction}")
            return 1
        manifest = json.loads(later_inputs.paths.manifest.read_text(encoding="utf-8"))
        predicted = prediction_entries(manifest, raw)
        note = prediction_note(raw, args.grids_prediction, later["config"].get("running_utc"))

    args.tex_dir.mkdir(parents=True, exist_ok=True)
    path = args.tex_dir / GRID_TABLE
    path.write_text(grid_table(of_record, later, predicted, prediction_source=note),
                    encoding="utf-8")
    base = {r["key"]: r for r in of_record["records"]}
    print(f"grids: {date} (of record) vs {args.grids}")
    for r in sorted(later["records"], key=_order):
        old = base.get(r["key"])
        print(f"  {r['key']:<16} {old['zne_status'] if old else '-':<26} -> "
              f"{r['zne_status']:<26} predicted {_estimate(predicted.get(r['key']))}")
    print(f"  predicted: {note or 'none'}")
    print(f"wrote {path}")
    return 0


def print_model_vs_hardware(summary: Mapping[str, Any]) -> None:
    """The deviation table of :func:`model_vs_hardware`, one line per group."""
    mvh = summary.get("model_vs_hardware") or {}
    a = mvh.get("model_alpha") or {}
    print(f"model vs hardware (hardware - model; alpha-hat {a.get('alpha')} from "
          f"{a.get('source')}, in-sample {a.get('in_sample')}; alpha = 1 model: "
          f"{(mvh.get('model_1') or {}).get('source')})")
    for quantity in ("p_solution", "tv_to_ideal"):
        for label, groups in (mvh.get(quantity) or {}).items():
            for group, s in groups.items():
                p = s["sign_test_p"]
                print(f"  {quantity:<11} {label:<9} {group:<11} n {s['n']:>2}  mean|d| "
                      f"{s['mean_abs']:.4f}  mean d {s['mean_signed']:+.4f}  "
                      f"neg/pos {s['n_negative']}/{s['n_positive']}  sign p "
                      + (f"{p:.2g}" if p is not None else "-"))
    for label, s in (mvh.get("tv_hardware_model") or {}).items():
        print(f"  D_TV(hw, model) {label:<9} mean {_num(s['mean'], 4)}  "
              f"shot floor {_num(s['mean_shot_floor'], 4)}")


def print_records(summary: Mapping[str, Any]) -> None:
    """The per-record table and the limit band, as the console shows them."""
    alpha = summary.get("alpha_hat")
    print("alpha_hat = " + (f"{alpha:.3f}" if alpha else "n/a (no fit.json)"))
    print(f"{'record':<16} {'2q':>5} {'ideal':>6} {'model':>6} {'hw':>6} "
          f"{'kept hw':>7} {'ZNE':>7}  status")
    for r in summary["records"]:
        kept = r["kept_hardware"]
        print(f"{r['key']:<16} {r['two_q_count']:>5} {_num(r['p_ideal']):>6} "
              f"{_num(r['model']):>6} {_num(r['hardware']):>6} "
              f"{_num(kept) if kept is not None else '-':>7} {_num(r['zne_reported']):>7}  "
              f"{r['zne_status']}")
    band = summary["limit_band"]
    if band:
        print(f"limit band: {band['lower']} ({band['lower_key']}) to {band['upper']} "
              f"({band['upper_key']}) two-qubit gates; exceptions {band['exceptions']}")


def print_survival(block: Mapping[str, Any] | None) -> None:
    """:func:`survival_block`, one line per record and the predicted bands."""
    if not block:
        print("survival: no calibration snapshot")
        return
    labels = list(block["medians"])
    print(f"survival ({block['calibration']}; medians "
          + "; ".join(f"{k}: 2q {v['two_qubit']:.2e}, 1q {v['one_qubit']:.2e}"
                      for k, v in block["medians"].items()) + ")")
    for r in block["records"]:
        cells = "  ".join(
            f"S[{label}] {_num(r[f'survival_{label}'])} "
            f"{'id' if r[f'predicted_identifiable_{label}'] else 'n.i.'}" for label in labels)
        print(f"  {r['key']:<16} {r['two_q_count']:>4}  kept {_num(r['kept_observed'][0])}  "
              f"{cells}  exponent {_num(r['exponent'], 2)}  measured {r['zne_status']}")
    for label, band in block["bands"].items():
        print(f"  band {label:<9} " + (f"{band['lower']} ({band['lower_key']}) to "
                                       f"{band['upper']} ({band['upper_key']}), exceptions "
                                       f"{band['exceptions']}" if band else "none"))
    for label, a in block["identifiable_agreement"].items():
        print(f"  identifiable predicted = measured ({label}): {a['agree']} of {a['of']}")


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
    if args.grids:
        return write_grid_table(args, date)
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
    print_records(summary)
    for d in summary["drift"]:
        print(f"drift {d['key']}: {d['p_first']:.4f} -> {d['p_repeat']:.4f} "
              f"(z = {d['z']:+.2f}, p = {d['p_value']:.3f}); D_TV {d['tv']:.4f} "
              f"[{d['tv_ci'][0]:.4f}, {d['tv_ci'][1]:.4f}], shot-noise null mean "
              f"{d['tv_null_mean']:.4f}, 95th pct {d['tv_null_q95']:.4f}, p = {d['tv_p_value']:.3f}")
    print_model_vs_hardware(summary)
    print_survival(summary.get("survival"))
    if missing:
        print("  ! not found (columns left empty): " + ", ".join(missing))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
