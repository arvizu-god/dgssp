"""
aggregate.py

Turn the campaign's per-instance records into the paper's tables.

Reads ``paper/data/sim_*`` (and nothing else -- no backend is imported, no
circuit is built) and writes:

``paper/data/sim_per_instance.csv``
    One row per (instance, grid): every estimator's ``P(solution)``, its
    distance to the exact reference, and the flags that say whether the fit
    means anything.
``paper/data/sim_cells.csv``
    One row per (family, n, M-class, grid, series): the cell statistic, its
    standard error, the number of instances behind it, and
    ``fraction_resolved`` / ``fraction_identifiable``, over every instance and
    (``*_with_signal``) over those whose ideal answer is not random guessing.
``paper/data/sim_pathology.csv``
    Every ``M = 0`` instance: ``D_TV`` to the uniform reference before and
    after each renormalisation.
``paper/data/sim_models.csv``
    The ``odd7`` model comparison: median absolute error to the ideal
    ``P(solution)``, over identified fits only.
``paper/data/sim_summary.json``
    Everything above in machine-readable form, plus the headline numbers the
    text quotes.  This is what ``make_figures.py`` and ``sim_results.md`` read.
``paper/tex/sim_table.tex``
    The per-cell table, ready to ``\\input``.

Two statistical rules are enforced here rather than left to the reader:

*Never average a raw extrapolation over unresolved fits.*  An exponential fit
that the design does not identify is not merely noisy, it is unbounded: its
chi-square is flat in the decay rate, so the fitted value tracks whatever
``exp_rate_max`` allows and a *mean* over such a cell measures that bound, not
the physics.  Wherever ``fraction_resolved < 0.5`` the unbounded series
(``zne_scalar_value``, ``zne_scalar_reported``, ``mitiq``) are summarised by the
**median**, with a robust standard error (``1.2533 * 1.4826 * MAD / sqrt(N)``);
bounded series keep the mean.  Every cell records which statistic it used.

*The raw value and the reported value are different numbers.*  They are carried
in separate columns and differ exactly where a fit is unresolved or
unidentified -- which is the WP4 result the paper is about, so collapsing them
would hide it.

*The flags are re-derived, not read.*  ``resolved``, ``identifiable`` and the
reported value follow the library's current rule
(:data:`dgssp.mitigation.zne.RESOLVED_RULE`, via
:func:`dgssp.mitigation.zne.scalar_flags`), computed here from the
observations, shots and ``P_uniform`` every record stores, with the side of
the null taken from the record's exact reference.  The WP5 records were
written under rule ``"v1"``, which was two-sided and so counted a series
sitting *below* random guessing as resolved; the records stay as written, and
the stored v1 flags are kept beside the v2 ones in ``*_v1`` columns.

*A reported extrapolation is reported inside its physical range.*  The noiseless
``P(solution)`` lies in ``[P_u, 1]``; an extrapolation that lands outside is
handled by :data:`dgssp.mitigation.zne.REPORT_BOUNDS` (``"clip"``: point and
interval clipped after fitting; ``--report-bounds flag`` reproduces the tables
before Step 5).  ``zne_scalar_value`` / ``zne_scalar_lo`` / ``zne_scalar_hi``
stay the fit as fitted; ``zne_scalar_reported`` and
``zne_scalar_reported_lo`` / ``_hi`` carry the option, and ``out_of_range`` /
``ci_out_of_range`` say where it acted.  The model-comparison columns
(``fit_<method>*``) are the fits as fitted.

Run::

    python paper/scripts/aggregate.py
    python paper/scripts/aggregate.py --grids odd planned odd7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections.abc import Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import DATA_DIR, TEX_DIR, load_records, record_provenance  # noqa: E402
from dgssp.metrics import solution_probability  # noqa: E402
from dgssp.mitigation.zne import (  # noqa: E402
    REPORT_BOUNDS,
    REPORT_BOUNDS_OPTIONS,
    RESOLVABILITY_K,
    RESOLVED_RULE,
    fit_block_config,
    report_extrapolation,
    scalar_flags,
)

#: The grids the campaign may have run.
DEFAULT_GRIDS: tuple[str, ...] = ("odd", "planned", "odd7")

#: Extrapolation methods compared on ``odd7``.
METHODS: tuple[str, ...] = ("linear", "quadratic", "richardson", "exponential")

#: Series whose value is not bounded, and which therefore get the median
#: treatment in a cell most of whose fits are unresolved.
UNBOUNDED_SERIES: frozenset[str] = frozenset(
    {"zne_scalar_value", "zne_scalar_reported", "mitiq"}
)

#: ``P(solution)`` series reported per cell, in table order.
PSOL_SERIES: tuple[str, ...] = (
    "ideal",
    "noisy",
    "zne_scalar_value",
    "zne_scalar_reported",
    "pointwise_clip",
    "pointwise_simplex",
    "mitiq",
)

#: Distance series reported per cell.
TV_SERIES: tuple[str, ...] = ("tv_noisy", "tv_pointwise_clip", "tv_pointwise_simplex")

#: Bias series: each estimator minus the *same instance's* ideal run.
BIAS_SERIES: tuple[str, ...] = (
    "bias_noisy",
    "bias_zne_lambda1",
    "bias_zne_scalar_value",
    "bias_zne_scalar_reported",
    "bias_pointwise_clip",
    "bias_pointwise_simplex",
    "bias_mitiq",
    *(f"bias_fit_{method}" for method in METHODS),
)

#: The bias of an unbounded estimator is unbounded too.
UNBOUNDED_SERIES = UNBOUNDED_SERIES | {
    f"bias_{name}" for name in UNBOUNDED_SERIES
} | {f"bias_fit_{method}" for method in METHODS}

#: Below this fraction of resolved fits, a cell is summarised by the median.
MEDIAN_BELOW = 0.5

#: 1.2533 = sqrt(pi/2), the asymptotic efficiency factor of the median;
#: 1.4826 converts a median absolute deviation into a standard deviation.
MEDIAN_SE_FACTOR = 1.2533
MAD_TO_SIGMA = 1.4826


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def stage_records(data_dir: Path, name: str) -> dict[str, dict[str, Any]]:
    """
    Load one stage directory, keyed by instance name.

    Parameters
    ----------
    data_dir:
        ``paper/data``.
    name:
        Directory name, e.g. ``"sim_zne_fake_planned"``.

    Returns
    -------
    dict[str, dict]
        Empty when the directory does not exist -- a stage that was not run is
        simply absent from the tables, not an error.
    """
    directory = data_dir / name
    if not directory.is_dir():
        return {}
    return {
        record["instance"]["name"]: record
        for record in load_records(directory / "*.json")
    }


def discover_noise(data_dir: Path) -> str:
    """
    Which noise label the campaign used (``"fake"`` or ``"live"``).

    Parameters
    ----------
    data_dir:
        ``paper/data``.

    Returns
    -------
    str
        The label found on disk; ``"fake"`` when nothing has been run yet.
    """
    for candidate in ("fake", "live"):
        if any(data_dir.glob(f"sim_*_{candidate}*")):
            return candidate
    return "fake"


# ---------------------------------------------------------------------------
# Per-instance extraction
# ---------------------------------------------------------------------------


def _get(record: dict[str, Any] | None, *path: str) -> Any:
    """
    Follow a key path through a record, returning ``None`` at the first gap.

    Parameters
    ----------
    record:
        A loaded record, or ``None``.
    *path:
        Keys to follow.

    Returns
    -------
    Any
        The value, or ``None``.
    """
    node: Any = record
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def exact_p_ideal(record: dict[str, Any] | None) -> float | None:
    """
    ``P(solution)`` of a record's exact reference distribution.

    Parameters
    ----------
    record:
        A loaded record, or ``None``.

    Returns
    -------
    float | None
        ``None`` when the record carries no reference or no solution list.
    """
    reference = _get(record, "metrics", "reference_distribution")
    sols = _get(record, "metrics", "solution_bitstrings")
    if not isinstance(reference, dict) or sols is None:
        return None
    return float(sum(float(reference.get(s, 0.0)) for s in sols))


def rederive_flags(
    block: dict[str, Any] | None,
    p_ideal: float | None,
    *,
    report_bounds: str | None = None,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    A stored scalar-fit block's flags and reported value under the current rule.

    Parameters
    ----------
    block:
        One ``zne_fit`` / ``zne_fits[method]`` block as stored.
    p_ideal:
        The instance's exact ``P(solution)`` (:func:`exact_p_ideal`).
    report_bounds:
        How a reported extrapolation outside the physical range is handled
        (:func:`dgssp.mitigation.zne.report_extrapolation`); ``None`` ->
        :data:`~dgssp.mitigation.zne.REPORT_BOUNDS`.  The flags do not depend
        on it.
    record:
        The record the block belongs to: ``"bounded"`` re-fits its counts.

    Returns
    -------
    dict
        ``resolved``, ``identifiable``, ``n_resolved_scales``,
        ``n_informative_scales``, ``signal_direction``, ``reported_value``,
        ``reported_source``, and ``report_bounds``, ``reported_ci`` (the
        interval of a reported extrapolation, else ``None``), ``out_of_range``
        / ``ci_out_of_range`` (of the unconstrained fit; ``None`` when no
        extrapolation is reported).  A missing or not-applicable block keeps
        its stored values (it has no fit to report).
    """
    option = REPORT_BOUNDS if report_bounds is None else report_bounds
    keys = ("resolved", "identifiable", "n_resolved_scales", "n_informative_scales",
            "reported_value", "reported_source")
    not_reported = {"report_bounds": option, "reported_ci": None, "out_of_range": None,
                    "ci_out_of_range": None}
    if not block or block.get("applicable") is False or not block.get("observed"):
        out = {key: (block or {}).get(key) for key in keys}
        out["signal_direction"] = None
        return {**out, **not_reported}
    flags = scalar_flags(
        block["observed"],
        block["shots"],
        block.get("null_value"),
        n_params=int(block["n_params"]),
        fit_ok=bool(block["fit_ok"]),
        p_ideal=p_ideal,
        k=float(block.get("resolvability_k") or RESOLVABILITY_K),
    )
    use_fit = flags["use_fit"]
    out = {
        "resolved": flags["resolved"],
        "identifiable": flags["identifiable"],
        "n_resolved_scales": flags["n_resolved_scales"],
        "n_informative_scales": flags["n_informative_scales"],
        "signal_direction": flags["signal_direction"],
        "reported_value": block["value"] if use_fit else block["value_at_min_scale"],
        "reported_source": "extrapolation" if use_fit else "min_scale",
        **not_reported,
    }
    if use_fit and block.get("ci") and block.get("null_value") is not None:
        bounded_inputs: dict[str, Any] = {}
        if option == "bounded":
            if record is None:
                raise ValueError("report_bounds='bounded' needs the record (its counts).")
            bounded_inputs = {
                "counts_per_scale": {float(k): dict(v) for k, v in record["counts"].items()},
                "observable_fn": partial(
                    solution_probability,
                    solution_states=record["metrics"]["solution_bitstrings"],
                ),
                "cfg": fit_block_config(block),
            }
        report = report_extrapolation(
            float(block["value"]), block["ci"], float(block["null_value"]), p_ideal,
            option=option, **bounded_inputs,
        )
        out.update(
            reported_value=report["value"],
            reported_ci=report["ci"],
            out_of_range=report["out_of_range"],
            ci_out_of_range=report["ci_out_of_range"],
        )
    return out


def instance_row(
    name: str,
    ideal: dict[str, Any] | None,
    noisy: dict[str, Any] | None,
    zne: dict[str, Any] | None,
    grid: str,
    report_bounds: str | None = None,
) -> dict[str, Any]:
    """
    One row of the per-instance table.

    Parameters
    ----------
    name:
        Instance name.
    ideal, noisy, zne:
        That instance's records from the three stages; any may be ``None``.
    grid:
        The scale grid the ZNE record used.
    report_bounds:
        :func:`rederive_flags`; ``None`` -> ``REPORT_BOUNDS``.

    Returns
    -------
    dict
        Flat, CSV-ready.
    """
    source = zne or noisy or ideal or {}
    campaign = _get(source, "metrics", "campaign") or {}
    fit = _get(zne, "zne_fit") or {}
    p_ideal_exact = exact_p_ideal(zne)
    flags = rederive_flags(fit, p_ideal_exact, report_bounds=report_bounds, record=zne)
    pathology = _get(zne, "pathology") or {}

    row: dict[str, Any] = {
        "name": name,
        "family": campaign.get("family"),
        "n_items": source.get("n_items"),
        "m_class": campaign.get("m_class"),
        "M": _get(source, "metrics", "num_solutions"),
        "iterations": source.get("iterations"),
        "grid": grid,
        "grid_source": campaign.get("grid_source"),
        "p_uniform": _get(source, "metrics", "p_uniform"),
        "two_q_count": _get(source, "transpiled", "two_q_count"),
        "survival": _get(zne or noisy, "transpiled", "survival"),
        "seed_source": _get(source, "transpiled", "seed_source"),
        # P(solution)
        "ideal": _get(ideal, "metrics", "solution_probability"),
        "noisy": _get(noisy, "metrics", "solution_probability"),
        "zne_lambda1": _get(zne, "metrics", "solution_probability"),
        "zne_scalar_value": fit.get("value"),
        "zne_scalar_reported": flags["reported_value"],
        "zne_scalar_se": fit.get("bootstrap_se"),
        "zne_scalar_lo": (fit.get("ci") or [None, None])[0],
        "zne_scalar_hi": (fit.get("ci") or [None, None])[1],
        # the interval of the reported value: REPORT_BOUNDS applied to the
        # extrapolation's interval; empty when the lambda = 1 value is reported
        "zne_scalar_reported_lo": (flags["reported_ci"] or [None, None])[0],
        "zne_scalar_reported_hi": (flags["reported_ci"] or [None, None])[1],
        "pointwise_clip": _get(zne, "pointwise", "clip", "solution_probability"),
        "pointwise_simplex": _get(zne, "pointwise", "simplex", "solution_probability"),
        "mitiq": _get(zne, "mitiq", "value"),
        # flags (re-derived under RESOLVED_RULE; see the module docstring)
        "resolved_rule": RESOLVED_RULE if fit else None,
        "p_ideal_exact": p_ideal_exact,
        "signal_direction": flags["signal_direction"],
        "resolved": flags["resolved"],
        "identifiable": flags["identifiable"],
        "reported_source": flags["reported_source"],
        "n_informative_scales": flags["n_informative_scales"],
        "n_resolved_scales": flags["n_resolved_scales"],
        "report_bounds": flags["report_bounds"] if fit else None,
        "out_of_range": flags["out_of_range"],
        "ci_out_of_range": flags["ci_out_of_range"],
        # the same flags as the record stores them (rule v1), for audit
        "resolved_v1": fit.get("resolved"),
        "identifiable_v1": fit.get("identifiable"),
        "reported_source_v1": fit.get("reported_source"),
        "zne_scalar_reported_v1": fit.get("reported_value"),
        "chi2_red": fit.get("chi2_red"),
        "dof": fit.get("dof"),
        "method": fit.get("method"),
        "scales": json.dumps(fit.get("scales") or campaign.get("nominal_scales")),
        # distances
        "tv_ideal": _get(ideal, "metrics", "tv_to_reference", "point"),
        "tv_noisy": _get(noisy, "metrics", "tv_to_reference", "point"),
        "tv_zne_lambda1": pathology.get("tv_raw_min_scale"),
        "tv_pointwise_clip": pathology.get("tv_pointwise_clip"),
        "tv_pointwise_simplex": pathology.get("tv_pointwise_simplex"),
        "clip_worse_than_raw": pathology.get("clip_worse_than_raw"),
        "simplex_worse_than_raw": pathology.get("simplex_worse_than_raw"),
        "fraction_positive_slope": pathology.get("fraction_positive_slope"),
    }
    for method in METHODS:
        block = _get(zne, "metrics", "zne_fits", method) or {}
        # The model comparison compares the fits as fitted.
        method_flags = rederive_flags(block, p_ideal_exact, report_bounds="flag")
        row[f"fit_{method}"] = block.get("value")
        # Records written before the not-applicable entry existed lack the key.
        row[f"fit_{method}_applicable"] = block.get("applicable", True if block else None)
        row[f"fit_{method}_identifiable"] = method_flags["identifiable"]
        row[f"fit_{method}_resolved"] = method_flags["resolved"]
        row[f"fit_{method}_reported"] = method_flags["reported_value"]
        row[f"fit_{method}_identifiable_v1"] = block.get("identifiable")

    # Bias against the same instance's ideal run, which is what every
    # extrapolation is trying to recover.  Computed per instance and only then
    # aggregated: a difference of cell means would hide the pairing.
    # The base series only, in a fixed order.  (Iterating the module-level
    # UNBOUNDED_SERIES, which also holds the ``bias_*`` names, wrote two
    # ``bias_bias_*`` columns whose values followed the set's hash order.)
    for series in ("noisy", "zne_lambda1", "zne_scalar_value", "zne_scalar_reported", "mitiq",
                   "pointwise_clip", "pointwise_simplex"):
        value, reference = row.get(series), row["ideal"]
        row[f"bias_{series}"] = (
            float(value) - float(reference)
            if isinstance(value, (int, float))
            and isinstance(reference, (int, float))
            and math.isfinite(value)
            else None
        )
    for method in METHODS:
        value, reference = row[f"fit_{method}"], row["ideal"]
        row[f"bias_fit_{method}"] = (
            float(value) - float(reference)
            if isinstance(value, (int, float))
            and isinstance(reference, (int, float))
            and math.isfinite(value)
            else None
        )
    return row


def build_rows(
    data_dir: Path, grids: Sequence[str], noise: str, report_bounds: str | None = None
) -> list[dict[str, Any]]:
    """
    The per-instance table for every grid that was run.

    An instance appears once per grid; instances with no ZNE record at all
    still appear once, under the grid ``"none"``, so the ideal and noisy
    columns are never silently dropped.

    Parameters
    ----------
    data_dir:
        ``paper/data``.
    grids:
        Grids to look for.
    noise:
        Noise label in the directory names.
    report_bounds:
        :func:`rederive_flags`; ``None`` -> ``REPORT_BOUNDS``.

    Returns
    -------
    list[dict]
        Rows in (grid, name) order.
    """
    ideal = stage_records(data_dir, "sim_ideal")
    noisy = stage_records(data_dir, f"sim_noisy_{noise}")

    rows: list[dict[str, Any]] = []
    covered: set[str] = set()
    for grid in grids:
        zne = stage_records(data_dir, f"sim_zne_{noise}_{grid}")
        for name in sorted(zne):
            rows.append(
                instance_row(name, ideal.get(name), noisy.get(name), zne[name], grid,
                             report_bounds)
            )
            covered.add(name)
    for name in sorted(set(ideal) | set(noisy)):
        if name not in covered:
            rows.append(instance_row(name, ideal.get(name), noisy.get(name), None, "none"))
    return rows


# ---------------------------------------------------------------------------
# Cell statistics
# ---------------------------------------------------------------------------


def _finite(values: Iterable[Any]) -> list[float]:
    """Numeric, finite entries of a column."""
    out = []
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(value):
                out.append(float(value))
    return out


def summarise(values: Sequence[float], *, use_median: bool) -> dict[str, Any]:
    """
    Point estimate and standard error of one cell's values.

    Parameters
    ----------
    values:
        The finite values in the cell.
    use_median:
        Use the median and a MAD-based standard error instead of the mean.

    Returns
    -------
    dict
        ``{"statistic", "value", "se", "n", "min", "max"}``; ``value`` is
        ``None`` for an empty cell.
    """
    n = len(values)
    if n == 0:
        return {"statistic": "median" if use_median else "mean", "value": None,
                "se": None, "n": 0, "min": None, "max": None}
    if use_median:
        point = statistics.median(values)
        mad = statistics.median([abs(v - point) for v in values])
        se = MEDIAN_SE_FACTOR * MAD_TO_SIGMA * mad / math.sqrt(n) if n > 1 else None
        statistic = "median"
    else:
        point = statistics.fmean(values)
        se = statistics.stdev(values) / math.sqrt(n) if n > 1 else None
        statistic = "mean"
    return {
        "statistic": statistic,
        "value": point,
        "se": se,
        "n": n,
        "min": min(values),
        "max": max(values),
    }


def fraction(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    """
    Fraction of rows whose boolean flag is true, over rows that carry it.

    Parameters
    ----------
    rows:
        The cell's rows.
    key:
        Flag name.

    Returns
    -------
    float | None
        ``None`` when no row carries the flag.
    """
    present = [r[key] for r in rows if isinstance(r.get(key), bool)]
    return sum(present) / len(present) if present else None


def with_signal(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    The rows whose noiseless answer differs from random guessing.

    ``M = 0`` (``P_ideal = 0 = P_uniform``) and ``M = N/2``
    (``P_ideal = P_uniform``) have ``signal_direction == 0``: rule v2 can never
    resolve them, so a fraction over every instance partly counts how many of
    them the family contains.  The fraction over this subset measures ZNE.

    Parameters
    ----------
    rows:
        Per-instance rows.

    Returns
    -------
    list[dict]
        The rows with ``signal_direction`` of +1 or -1.
    """
    return [r for r in rows if r.get("signal_direction") in (1, -1)]


def signal_fractions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    ``fraction_resolved`` / ``fraction_identifiable`` over the rows with signal.

    Parameters
    ----------
    rows:
        Per-instance rows.

    Returns
    -------
    dict
        ``n_with_signal``, ``fraction_resolved_with_signal``,
        ``fraction_identifiable_with_signal``.
    """
    subset = with_signal(rows)
    return {
        "n_with_signal": len(subset),
        "fraction_resolved_with_signal": fraction(subset, "resolved"),
        "fraction_identifiable_with_signal": fraction(subset, "identifiable"),
    }


def _cell_members(
    rows: Sequence[dict[str, Any]],
    family: str,
    n_items: int,
    m_class: str,
    grid: str,
) -> list[dict[str, Any]]:
    """
    The instances behind one cell.

    Two pooled ``M``-classes exist beside the four real ones: ``"all"``
    (every instance at that ``n``) and ``"feasible"`` (``M >= 1`` only).
    ``P(solution)`` is identically zero for an infeasible instance, so pooling
    it with the rest would drag every mean toward zero for reasons that have
    nothing to do with the mitigation -- the figures use ``"feasible"``.

    Parameters
    ----------
    rows:
        The per-instance table.
    family, n_items, m_class, grid:
        The cell.

    Returns
    -------
    list[dict]
        Its rows.
    """
    at_n = [
        r
        for r in rows
        if (r["family"], r["n_items"], r["grid"]) == (family, n_items, grid)
    ]
    if m_class == "all":
        return at_n
    if m_class == "feasible":
        return [r for r in at_n if (r["M"] or 0) >= 1]
    return [r for r in at_n if r["m_class"] == m_class]


def build_cells(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Group the per-instance rows into (family, n, M-class, grid) cells.

    Parameters
    ----------
    rows:
        The per-instance table.

    Returns
    -------
    list[dict]
        One row per cell and series, including the pooled ``"all"`` and
        ``"feasible"`` ``M``-classes the figures plot.
    """
    keys = sorted(
        {
            (r["family"], r["n_items"], m_class, r["grid"])
            for r in rows
            if r["family"] is not None
            for m_class in (r["m_class"], "all", "feasible")
        },
        key=lambda k: (str(k[3]), str(k[0]), k[1] or 0, str(k[2])),
    )

    cells: list[dict[str, Any]] = []
    for family, n_items, m_class, grid in keys:
        members = _cell_members(rows, family, n_items, m_class, grid)
        if not members:
            continue
        resolved = fraction(members, "resolved")
        identifiable = fraction(members, "identifiable")
        use_median = resolved is not None and resolved < MEDIAN_BELOW
        signal = signal_fractions(members)

        for series in (*PSOL_SERIES, *TV_SERIES, *BIAS_SERIES):
            values = _finite(r.get(series) for r in members)
            stats = summarise(
                values, use_median=use_median and series in UNBOUNDED_SERIES
            )
            cells.append(
                {
                    "family": family,
                    "n_items": n_items,
                    "m_class": m_class,
                    "grid": grid,
                    "series": series,
                    "n_instances": len(members),
                    **stats,
                    "fraction_resolved": resolved,
                    "fraction_identifiable": identifiable,
                    **signal,
                }
            )
    return cells


# ---------------------------------------------------------------------------
# Pathology and model comparison
# ---------------------------------------------------------------------------


def build_pathology(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    The ``M = 0`` rows, where the exact reference is uniform.

    An ``M = 0`` instance has no marked state, so its noiseless index
    distribution is exactly uniform and *any* distance mitigation adds is
    pathology, not residual error.

    Parameters
    ----------
    rows:
        The per-instance table.

    Returns
    -------
    list[dict]
        One row per ``M = 0`` instance and grid.
    """
    return [
        {
            key: row[key]
            for key in (
                "name",
                "family",
                "n_items",
                "m_class",
                "grid",
                "grid_source",
                "scales",
                "survival",
                "tv_zne_lambda1",
                "tv_pointwise_clip",
                "tv_pointwise_simplex",
                "clip_worse_than_raw",
                "simplex_worse_than_raw",
                "fraction_positive_slope",
            )
        }
        for row in rows
        if row["M"] == 0 and row["tv_zne_lambda1"] is not None
    ]


def build_model_comparison(
    rows: Sequence[dict[str, Any]], grid: str
) -> list[dict[str, Any]]:
    """
    Median absolute error of each extrapolation model, identified fits only.

    The error is measured against the *ideal* ``P(solution)`` of the same
    instance, which is the quantity the extrapolation is trying to recover.
    Fits the design does not identify are excluded and counted, not silently
    averaged in: an unidentified exponential has no finite expectation.

    Parameters
    ----------
    rows:
        The per-instance table.
    grid:
        Which grid to compare on.

    Returns
    -------
    list[dict]
        One row per method.
    """
    members = [r for r in rows if r["grid"] == grid and (r["M"] or 0) >= 1]
    out: list[dict[str, Any]] = []
    for method in METHODS:
        errors: list[float] = []
        excluded = {"no_ideal": 0, "not_applicable": 0, "not_identifiable": 0, "not_finite": 0}
        for row in members:
            if row["ideal"] is None:
                excluded["no_ideal"] += 1
                continue
            if row.get(f"fit_{method}_applicable") is False:
                excluded["not_applicable"] += 1
                continue
            if row.get(f"fit_{method}_identifiable") is not True:
                excluded["not_identifiable"] += 1
                continue
            value = row.get(f"fit_{method}")
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                excluded["not_finite"] += 1
                continue
            errors.append(abs(float(value) - float(row["ideal"])))
        out.append(
            {
                "grid": grid,
                "method": method,
                "n_used": len(errors),
                "n_candidates": len(members),
                "median_abs_error": statistics.median(errors) if errors else None,
                "mean_abs_error": statistics.fmean(errors) if errors else None,
                "max_abs_error": max(errors) if errors else None,
                **{f"excluded_{k}": v for k, v in excluded.items()},
            }
        )
    return out


# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------


def headline(
    rows: Sequence[dict[str, Any]], cells: Sequence[dict[str, Any]], grids: Sequence[str]
) -> dict[str, Any]:
    """
    The numbers the paper's text quotes, computed once, here.

    Parameters
    ----------
    rows:
        The per-instance table.
    cells:
        The per-cell table.
    grids:
        Grids present.

    Returns
    -------
    dict
        Nested by grid, plus the grid comparison.
    """
    out: dict[str, Any] = {"per_grid": {}, "grid_comparison": {}}

    for grid in grids:
        members = [r for r in rows if r["grid"] == grid]
        if not members:
            continue
        feasible = [r for r in members if (r["M"] or 0) >= 1]
        recovery = [
            (r["zne_scalar_reported"] - r["zne_lambda1"]) / (r["ideal"] - r["zne_lambda1"])
            for r in feasible
            if None not in (r["zne_scalar_reported"], r["zne_lambda1"], r["ideal"])
            and abs(r["ideal"] - r["zne_lambda1"]) > 1e-9
            and math.isfinite(r["zne_scalar_reported"])
        ]
        pathology = [r for r in members if r["M"] == 0 and r["tv_zne_lambda1"] is not None]
        out["per_grid"][grid] = {
            "n_instances": len(members),
            "fraction_resolved": fraction(members, "resolved"),
            "fraction_identifiable": fraction(members, "identifiable"),
            **signal_fractions(members),
            "fraction_reported_from_extrapolation": (
                sum(1 for r in members if r["reported_source"] == "extrapolation")
                / len(members)
            ),
            "n_reported_from_extrapolation": sum(
                1 for r in members if r["reported_source"] == "extrapolation"
            ),
            # Of the unconstrained fit, before REPORT_BOUNDS is applied.
            "n_reported_out_of_range": sum(1 for r in members if r["out_of_range"]),
            "n_reported_ci_out_of_range": sum(1 for r in members if r["ci_out_of_range"]),
            "signal_recovery_median": (
                statistics.median(recovery) if recovery else None
            ),
            "signal_recovery_n": len(recovery),
            "pathology": {
                "n_m0": len(pathology),
                "median_tv_lambda1": (
                    statistics.median([r["tv_zne_lambda1"] for r in pathology])
                    if pathology
                    else None
                ),
                "median_tv_clip": (
                    statistics.median(_finite(r["tv_pointwise_clip"] for r in pathology))
                    if pathology
                    else None
                ),
                "median_tv_simplex": (
                    statistics.median(
                        _finite(r["tv_pointwise_simplex"] for r in pathology)
                    )
                    if pathology
                    else None
                ),
                "fraction_clip_worse": fraction(pathology, "clip_worse_than_raw"),
                "fraction_simplex_worse": fraction(pathology, "simplex_worse_than_raw"),
            },
        }

    by_n: dict[str, dict[str, Any]] = {}
    for grid in ("odd", "planned"):
        members = [r for r in rows if r["grid"] == grid]
        for n_items in sorted({r["n_items"] for r in members if r["n_items"]}):
            subset = [r for r in members if r["n_items"] == n_items]
            by_n.setdefault(str(n_items), {})[grid] = {
                "n_instances": len(subset),
                "fraction_resolved": fraction(subset, "resolved"),
                "fraction_identifiable": fraction(subset, "identifiable"),
                **signal_fractions(subset),
            }
    out["grid_comparison"] = by_n
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """
    Write a list of flat dictionaries as a CSV, union of keys in first-seen order.

    Unlike :func:`_common.save_record` this *does* overwrite: a CSV is a
    derived view of the records, not evidence, and is regenerated whenever the
    records change.

    Parameters
    ----------
    path:
        Destination.
    rows:
        The rows.

    Returns
    -------
    Path
        The file written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fmt(value: Any, digits: int = 3) -> str:
    """A number for LaTeX, or an em dash when there is none."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "---"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_latex(path: Path, cells: Sequence[dict[str, Any]], grid: str) -> Path:
    """
    The per-cell ``P(solution)`` table for one grid.

    Parameters
    ----------
    path:
        Destination ``.tex`` file.
    cells:
        The per-cell table.
    grid:
        Which grid to tabulate.

    Returns
    -------
    Path
        The file written.
    """
    # The pooled classes exist for the figures; the table lists the real ones.
    real = [c for c in cells if c["grid"] == grid and c["m_class"] not in ("all", "feasible")]
    index: dict[tuple[Any, ...], dict[str, Any]] = {
        (c["family"], c["n_items"], c["m_class"], c["series"]): c for c in real
    }
    keys = sorted(
        {(c["family"], c["n_items"], c["m_class"]) for c in real},
        key=lambda k: (str(k[0]), k[1] or 0, str(k[2])),
    )

    shown = ("ideal", "noisy", "zne_scalar_reported", "pointwise_clip", "mitiq")
    header = ("Ideal", "Noisy", "ZNE (rep.)", "Pointwise", "Mitiq")

    lines = [
        "% Generated by paper/scripts/aggregate.py -- do not edit.",
        f"% grid: {grid}",
        "\\begin{tabular}{llr" + "r" * len(shown) + "rr}",
        "\\hline",
        "Fam. & $n$ & $M$-class & "
        + " & ".join(header)
        + " & $f_\\mathrm{res}$ & $f_\\mathrm{id}$ \\\\",
        "\\hline",
    ]
    for family, n_items, m_class in keys:
        first = index.get((family, n_items, m_class, "ideal"), {})
        cell_values = []
        for series in shown:
            cell = index.get((family, n_items, m_class, series))
            if cell is None or cell["value"] is None:
                cell_values.append("---")
                continue
            marker = "*" if cell["statistic"] == "median" else ""
            cell_values.append(
                f"{_fmt(cell['value'])}{marker} $\\pm$ {_fmt(cell['se'])}"
            )
        lines.append(
            f"{family} & {n_items} & {m_class} & "
            + " & ".join(cell_values)
            + f" & {_fmt(first.get('fraction_resolved'), 2)}"
            + f" & {_fmt(first.get('fraction_identifiable'), 2)} \\\\"
        )
    lines += [
        "\\hline",
        "\\end{tabular}",
        "% * = median (the cell's fits are mostly unresolved, and an "
        "unidentified exponential has no finite mean).",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Notes template
# ---------------------------------------------------------------------------


def fill_notes(template: Path, summary: dict[str, Any]) -> str:
    """
    Substitute the computed numbers into the results-notes template.

    Placeholders are ``{{path.to.value}}``, resolved against ``summary``; an
    unresolvable placeholder is left visible as ``??`` rather than silently
    dropped, so an un-run stage is obvious in the notes rather than reading as
    a measured zero.

    Parameters
    ----------
    template:
        ``paper/notes/sim_results.template.md``.  It is read, never written:
        the filled copy goes to a separate file, so the placeholders survive
        to be filled again next time.
    summary:
        The summary dictionary.

    Returns
    -------
    str
        The filled text.
    """
    import re

    def resolve(match: re.Match[str]) -> str:
        node: Any = summary
        for key in match.group(1).split("."):
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return "??"
        if isinstance(node, float):
            return f"{node:.4g}"
        return "??" if node is None else str(node)

    return re.sub(r"\{\{([A-Za-z0-9_.]+)\}\}", resolve, template.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv:
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--tex-dir", type=Path, default=TEX_DIR)
    parser.add_argument("--noise", choices=("fake", "live"), default=None)
    parser.add_argument("--grids", nargs="+", default=list(DEFAULT_GRIDS))
    parser.add_argument(
        "--report-bounds",
        choices=REPORT_BOUNDS_OPTIONS,
        default=REPORT_BOUNDS,
        help=f"How a reported extrapolation outside [P_u, 1] is handled (default {REPORT_BOUNDS}).",
    )
    parser.add_argument(
        "--notes-template",
        type=Path,
        default=None,
        help="Template to fill (default: paper/notes/sim_results.template.md).",
    )
    parser.add_argument(
        "--notes",
        type=Path,
        default=None,
        help="Where the filled notes go (default: paper/notes/sim_results.md).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Build every table from the records on disk.

    Parameters
    ----------
    argv:
        Command line; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        ``0`` on success, ``1`` when there is nothing to aggregate.
    """
    args = parse_args(argv)
    noise = args.noise or discover_noise(args.data_dir)

    rows = build_rows(args.data_dir, args.grids, noise, args.report_bounds)
    if not rows:
        print(
            f"Nothing to aggregate: no sim_* records under {args.data_dir}. "
            "Run sim_campaign.py first."
        )
        return 1

    grids_present = sorted({r["grid"] for r in rows} - {"none"})
    cells = build_cells(rows)
    pathology = build_pathology(rows)
    models = [
        row
        for grid in grids_present
        for row in build_model_comparison(rows, grid)
    ]

    summary = {
        "schema": "dgssp/paper/sim-summary/1",
        "noise": noise,
        "resolved_rule": RESOLVED_RULE,
        "report_bounds": args.report_bounds,
        "grids": grids_present,
        "n_instances": len({r["name"] for r in rows}),
        "stages_present": {
            "ideal": sum(1 for r in rows if r["ideal"] is not None),
            "noisy": sum(1 for r in rows if r["noisy"] is not None),
            "zne": sum(1 for r in rows if r["zne_scalar_value"] is not None),
            "mitiq": sum(1 for r in rows if r["mitiq"] is not None),
        },
        **headline(rows, cells, grids_present),
        "cells": cells,
        "models": models,
        "provenance": record_provenance(),
    }

    written = [
        write_csv(args.data_dir / "sim_per_instance.csv", rows),
        write_csv(args.data_dir / "sim_cells.csv", cells),
    ]
    if pathology:
        written.append(write_csv(args.data_dir / "sim_pathology.csv", pathology))
    if models:
        written.append(write_csv(args.data_dir / "sim_models.csv", models))

    summary_path = args.data_dir / "sim_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    written.append(summary_path)

    for grid in grids_present:
        written.append(write_latex(args.tex_dir / f"sim_table_{grid}.tex", cells, grid))

    notes_dir = args.data_dir.parent / "notes"
    template = args.notes_template or (notes_dir / "sim_results.template.md")
    notes = args.notes or (notes_dir / "sim_results.md")
    if template.exists():
        notes.parent.mkdir(parents=True, exist_ok=True)
        notes.write_text(fill_notes(template, summary), encoding="utf-8")
        written.append(notes)
    else:
        print(f"  ! no notes template at {template}; skipping the results note")

    print(f"instances: {summary['n_instances']}  grids: {grids_present}  "
          f"report bounds: {args.report_bounds}")
    print(f"rows carrying each stage's value: {summary['stages_present']} "
          "(an instance appears once per grid)")
    for grid, block in summary["per_grid"].items():
        print(
            f"  [{grid}] resolved={_fmt(block['fraction_resolved'], 2)} "
            f"identifiable={_fmt(block['fraction_identifiable'], 2)} "
            f"(with signal, N={block['n_with_signal']}: "
            f"{_fmt(block['fraction_resolved_with_signal'], 2)} / "
            f"{_fmt(block['fraction_identifiable_with_signal'], 2)}) "
            f"extrapolated={_fmt(block['fraction_reported_from_extrapolation'], 2)} "
            f"signal recovery (median)={_fmt(block['signal_recovery_median'], 3)} "
            f"out of range: point {block['n_reported_out_of_range']} / interval "
            f"{block['n_reported_ci_out_of_range']} of {block['n_reported_from_extrapolation']}"
        )
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
