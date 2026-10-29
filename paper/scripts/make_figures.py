"""
make_figures.py

Build the paper's figures from ``paper/data`` alone.

Nothing here imports a backend, builds a circuit or samples anything: every
number was measured by ``sim_campaign.py`` and reduced by ``aggregate.py``, and
a figure that cannot be rebuilt from the archived tables is a figure whose
provenance is broken.  Run ``aggregate.py`` first; this script reads its CSVs,
``sim_summary.json`` and the sweep records.

``--set sim`` builds five figures:

``fig_sim_psol``
    ``P(solution)`` against ``n``, for the ideal, noisy, scalar-ZNE,
    pointwise-ZNE and Mitiq estimators, over the *feasible* instances
    (``M >= 1``; an infeasible instance's ``P(solution)`` is identically zero
    and pooling it would drag every curve toward zero for arithmetic reasons).
    Cells whose fits are mostly unresolved are marked: those points are the
    ``lambda = 1`` observation, not an extrapolation.

``fig_sim_dtv``
    ``D_TV`` to the exact reference against ``n`` for the unmitigated and
    pointwise-mitigated distributions, and -- third panel -- the ``M = 0``
    pathology one point per instance: distance before mitigation against
    distance after, with the diagonal marking "no change".  Points above the
    diagonal are instances the mitigation made *worse*.

``fig_sim_grid``
    The partial-folding result: resolved and identifiable fractions, and the
    bias of the scalar estimator, for the odd-integer grid against the planned
    one, at matched shots.

``fig_sim_sweep``
    ``P(solution))`` against the Grover iteration count for the two sweep
    instances, with ``sin^2((2r+1) theta)`` as the analytic line.

``fig_sim_models``
    Model comparison on the ``odd7`` grid: per-instance residuals and the
    median absolute error of each extrapolation, over identified fits only.

``--set hw`` builds the hardware figures from ``hw_summary.json`` (run
``hw_aggregate.py`` first) and, for the comparisons, the simulation tables:

``fig_hw_example``
    The worked example: one instance's output distribution, ideal / noise
    model / hardware / hardware with pointwise ZNE.

``fig_hw_signal``
    Signal kept at ``lambda = 1`` against two-qubit gate count: the WP5
    simulation as a cloud, the device's own noise model, the same model with
    every error scaled by the fitted alpha, and the hardware.

``fig_hw_folding``
    Why the odd grid fails on hardware: ``P(solution)`` against ``lambda`` with
    the fitted exponential, for one record of each estimator status.

``fig_hw_zne``
    Our scalar ZNE against Mitiq on the same counts, per record, with the
    ideal value, the status of our estimate, and values outside ``[0, 1]``
    marked.

``fig_hw_sweep`` / ``fig_hw_dtv``
    ``fig_sim_sweep`` and ``fig_sim_dtv`` with the hardware points added (the
    sweep instances' records; the ``M = 0`` distances).

Run::

    python paper/scripts/make_figures.py --set sim
    python paper/scripts/make_figures.py --set hw
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import DATA_DIR, FIGURES_DIR  # noqa: E402
from aggregate import rederive_flags  # noqa: E402
from plotting import (  # noqa: E402
    COLUMN_WIDTH_IN,
    FULL_WIDTH_IN,
    OKABE_ITO,
    figure,
    figure_legend,
    legend_below,
    save_figure,
    style_for,
)

#: Grid whose cells the main figures plot.
MAIN_GRID = "planned"

#: Grid the model comparison lives on.
MODEL_GRID = "odd7"

#: The M-class pooling the figures use (see ``aggregate._cell_members``).
FEASIBLE = "feasible"
ALL = "all"

#: Extrapolation models are not executors, so they are not in
#: ``plotting.EXECUTORS``; they get their own consistent identities, drawn from
#: the same Okabe--Ito palette.
METHOD_STYLE: dict[str, tuple[str, str]] = {
    "linear": (OKABE_ITO["sky_blue"], "o"),
    "quadratic": (OKABE_ITO["orange"], "s"),
    "richardson": (OKABE_ITO["reddish_purple"], "^"),
    "exponential": (OKABE_ITO["bluish_green"], "D"),
}

#: Below this resolved fraction a cell is reported at ``lambda = 1``; the
#: figures mark those points rather than letting them read as extrapolations.
RESOLVED_MARK = 0.5

#: The worked example of ``fig_hw_example``: the one instance whose hardware
#: estimate is resolved.  Any record key from ``hw_summary.json`` works.
EXAMPLE_RECORD = "P-n2-M1-11"

#: ``fig_hw_folding`` shows one record per status, the shallowest of each.
FOLDING_STATUSES: tuple[str, ...] = (
    "resolved",
    "identified, not resolved",
    "not identifiable",
)

#: ``fig_hw_zne`` draws ``P(solution)`` up to here; larger values (Mitiq's
#: unbounded extrapolations) sit on the edge with their value written.
ZNE_AXIS_MAX = 1.25

#: Half-width of a 95 % normal interval, for the per-scale hardware points.
Z95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _number(value: str) -> float | None:
    """A CSV field as a float, or ``None`` when it is blank or not numeric."""
    if value in ("", "None", "nan"):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _flag(value: str) -> bool | None:
    """A CSV field as a boolean, or ``None`` when blank."""
    return {"True": True, "False": False}.get(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    """
    Read a CSV written by ``aggregate.py``.

    Parameters
    ----------
    path:
        The file.

    Returns
    -------
    list[dict]
        Empty when the file does not exist, so a partial campaign still
        produces the figures it can.
    """
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def cell_series(
    cells: Sequence[dict[str, str]],
    *,
    family: str,
    series: str,
    grid: str,
    m_class: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One series of one family as arrays over ``n``.

    Parameters
    ----------
    cells:
        Rows of ``sim_cells.csv``.
    family:
        ``"P"`` or ``"S"``.
    series:
        Series name.
    grid:
        Scale grid.
    m_class:
        ``M``-class or a pooled class.

    Returns
    -------
    tuple of ndarray
        ``(n, value, se, fraction_resolved)``, sorted by ``n``; ``se`` and
        ``fraction_resolved`` hold NaN where the table has none.
    """
    rows = [
        row
        for row in cells
        if row["family"] == family
        and row["series"] == series
        and row["grid"] == grid
        and row["m_class"] == m_class
        and _number(row["value"]) is not None
    ]
    rows.sort(key=lambda r: int(r["n_items"]))
    n = np.array([int(r["n_items"]) for r in rows], dtype=float)
    value = np.array([_number(r["value"]) for r in rows], dtype=float)
    se = np.array(
        [np.nan if _number(r["se"]) is None else _number(r["se"]) for r in rows],
        dtype=float,
    )
    resolved = np.array(
        [
            np.nan
            if _number(r["fraction_resolved"]) is None
            else _number(r["fraction_resolved"])
            for r in rows
        ],
        dtype=float,
    )
    return n, value, se, resolved


def _integer_xticks(ax: Any, values: Sequence[float]) -> None:
    """
    Put ticks on the integers present, so ``n`` never reads as 2.025.

    Parameters
    ----------
    ax:
        The axes.
    values:
        The ``n`` values plotted (may be empty, in which case nothing is done).
    """
    ticks = sorted({int(v) for v in values})
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xlim(min(ticks) - 0.4, max(ticks) + 0.4)


def _plot_series(
    ax: Any,
    n: np.ndarray,
    value: np.ndarray,
    se: np.ndarray,
    key: str,
    *,
    label: str | None = None,
) -> None:
    """Draw one executor's series with its fixed identity and SE bars."""
    if n.size == 0:
        return
    style = style_for(key)
    ax.errorbar(
        n,
        value,
        yerr=np.nan_to_num(se, nan=0.0),
        capsize=2,
        elinewidth=0.8,
        label=label if label is not None else style.label,
        **style.line_kwargs(markersize=4),
    )


# ---------------------------------------------------------------------------
# fig_sim_psol
# ---------------------------------------------------------------------------


def fig_sim_psol(cells: Sequence[dict[str, str]], grid: str, out: Path) -> list[Path]:
    """
    ``P(solution)`` against ``n``, one panel per instance family.

    Parameters
    ----------
    cells:
        Rows of ``sim_cells.csv``.
    grid:
        Which grid's ZNE columns to plot.
    out:
        Figures directory.

    Returns
    -------
    list[Path]
        Files written.
    """
    fig, axes = figure(width=FULL_WIDTH_IN, aspect=0.42, ncols=2, sharey=True)
    plotted = ("ideal", "noisy", "zne_scalar", "zne_pointwise", "mitiq")
    columns = {
        "ideal": "ideal",
        "noisy": "noisy",
        "zne_scalar": "zne_scalar_reported",
        "zne_pointwise": "pointwise_clip",
        "mitiq": "mitiq",
    }

    for ax, family, title in zip(
        axes, ("P", "S"), ("Family P (all positive)", "Family S (signed)"), strict=True
    ):
        for key in plotted:
            n, value, se, resolved = cell_series(
                cells, family=family, series=columns[key], grid=grid, m_class=FEASIBLE
            )
            _plot_series(ax, n, value, se, key)
            if key == "zne_scalar" and n.size:
                # A point whose cell is mostly unresolved is the lambda = 1
                # observation; a filled marker says "not extrapolated".
                mask = np.nan_to_num(resolved, nan=0.0) < RESOLVED_MARK
                if mask.any():
                    ax.plot(
                        n[mask],
                        value[mask],
                        linestyle="none",
                        marker=style_for(key).marker,
                        markersize=4,
                        color=style_for(key).color,
                        markerfacecolor=style_for(key).color,
                        zorder=style_for(key).zorder + 1,
                        label="reported at $\\lambda=1$",
                    )
        ax.set_title(title)
        ax.set_xlabel("$n$ (items)")
        _integer_xticks(
            ax, [int(r["n_items"]) for r in cells if r["family"] == family]
        )
    axes[0].set_ylabel("$P(\\mathrm{solution})$, feasible instances")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].annotate(
        f"grid: {grid}",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize="x-small",
        color=OKABE_ITO["grey"],
    )
    legend_below(axes[0], ncol=3)
    return save_figure(fig, "fig_sim_psol", directory=out)


# ---------------------------------------------------------------------------
# fig_sim_dtv
# ---------------------------------------------------------------------------


def fig_sim_dtv(
    cells: Sequence[dict[str, str]],
    pathology: Sequence[dict[str, str]],
    grid: str,
    out: Path,
    *,
    hardware: Sequence[dict[str, Any]] | None = None,
    name: str = "fig_sim_dtv",
) -> list[Path]:
    """
    Distance to the exact reference, and the ``M = 0`` pathology.

    Parameters
    ----------
    cells:
        Rows of ``sim_cells.csv``.
    pathology:
        Rows of ``sim_pathology.csv``.
    grid:
        Which grid's mitigated columns to plot.
    out:
        Figures directory.
    hardware:
        ``hw_summary.json`` records; their ``M = 0`` distances are added to
        the third panel as filled hardware markers.
    name:
        Output file stem.

    Returns
    -------
    list[Path]
        Files written.
    """
    fig, axes = figure(width=FULL_WIDTH_IN, aspect=0.40, ncols=3)
    columns = {
        "noisy": "tv_noisy",
        "zne_pointwise": "tv_pointwise_clip",
    }

    for ax, family in zip(axes[:2], ("P", "S"), strict=True):
        for key, column in columns.items():
            n, value, se, _ = cell_series(
                cells, family=family, series=column, grid=grid, m_class=ALL
            )
            _plot_series(ax, n, value, se, key)
        n, value, se, _ = cell_series(
            cells,
            family=family,
            series="tv_pointwise_simplex",
            grid=grid,
            m_class=ALL,
        )
        if n.size:
            style = style_for("zne_pointwise")
            ax.errorbar(
                n,
                value,
                yerr=np.nan_to_num(se, nan=0.0),
                capsize=2,
                elinewidth=0.8,
                label="ZNE (pointwise, simplex)",
                **style.line_kwargs(marker="P", linestyle=":", markersize=4),
            )
        ax.set_title(f"Family {family}")
        ax.set_xlabel("$n$ (items)")
        _integer_xticks(
            ax, [int(r["n_items"]) for r in cells if r["family"] == family]
        )
    axes[0].set_ylabel("$D_{\\mathrm{TV}}$ to exact reference")

    ax = axes[2]
    before = np.array([_number(r["tv_zne_lambda1"]) or np.nan for r in pathology])
    for column, key, label in (
        ("tv_pointwise_clip", "zne_pointwise", "clip"),
        ("tv_pointwise_simplex", "mitiq", "simplex"),
    ):
        after = np.array([_number(r[column]) or np.nan for r in pathology])
        style = style_for(key)
        ax.plot(
            before,
            after,
            linestyle="none",
            marker=style.marker,
            markersize=3,
            alpha=0.6,
            color=style.color,
            markerfacecolor="none",
            label=label,
        )
    hw_m0 = [
        r for r in hardware or []
        if r.get("M") == 0 and isinstance(r.get("tv_raw"), (int, float))
    ]
    hw_values: list[float] = []
    for column, marker, label in (
        ("tv_pointwise_clip", None, "hardware, clip"),
        ("tv_pointwise_simplex", "P", "hardware, simplex"),
    ):
        points = [(r["tv_raw"], r[column]) for r in hw_m0
                  if isinstance(r.get(column), (int, float))]
        if not points:
            continue
        style = style_for("hardware")
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            linestyle="none",
            marker=marker or style.marker,
            markersize=6,
            color=style.color,
            markerfacecolor=style.color if marker is None else "none",
            zorder=style.zorder,
            label=label,
        )
        hw_values += [v for p in points for v in p]
    # Log--log with matched limits, so the diagonal really is "no change" and
    # the distance above it is the factor by which mitigation made it worse.
    after_all = np.concatenate(
        [
            np.array([_number(r[column]) or np.nan for r in pathology])
            for column in ("tv_pointwise_clip", "tv_pointwise_simplex")
        ]
    ) if pathology else np.array([np.nan])
    finite = np.concatenate([before, after_all, np.array(hw_values, dtype=float)])
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size:
        lo, hi = float(finite.min()) / 2.0, float(finite.max()) * 2.0
        ax.plot([lo, hi], [lo, hi], color=OKABE_ITO["grey"], linewidth=0.8, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$D_{\\mathrm{TV}}$ before ($\\lambda=1$)")
    ax.set_ylabel("$D_{\\mathrm{TV}}$ after")
    title = f"$M=0$ ({len(pathology)} inst."
    title += f", {len(hw_m0)} on hardware)" if hw_m0 else ")"
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize="x-small")
    figure_legend(fig, axes[:2], ncol=3)
    return save_figure(fig, name, directory=out)


# ---------------------------------------------------------------------------
# fig_sim_grid
# ---------------------------------------------------------------------------


def fig_sim_grid(cells: Sequence[dict[str, str]], out: Path) -> list[Path]:
    """
    Odd-integer against planned grid: identifiability, resolvability and bias.

    Parameters
    ----------
    cells:
        Rows of ``sim_cells.csv``.
    out:
        Figures directory.

    Returns
    -------
    list[Path]
        Files written.
    """
    fig, axes = figure(width=FULL_WIDTH_IN, aspect=0.42, ncols=2)
    grids = {"odd": ("--", OKABE_ITO["vermillion"]), "planned": ("-", OKABE_ITO["blue"])}

    ax = axes[0]
    for grid, (linestyle, color) in grids.items():
        for column, marker, name in (
            ("fraction_resolved", "s", "resolved"),
            ("fraction_identifiable", "^", "identifiable"),
        ):
            rows = [
                r
                for r in cells
                if r["grid"] == grid
                and r["m_class"] == ALL
                and r["series"] == "ideal"
                and _number(r[column]) is not None
            ]
            # One point per n, pooling both families at that n.
            by_n: dict[int, list[float]] = {}
            for row in rows:
                by_n.setdefault(int(row["n_items"]), []).append(
                    float(_number(row[column]) or 0.0)
                )
            if not by_n:
                continue
            n = np.array(sorted(by_n), dtype=float)
            value = np.array([float(np.mean(by_n[int(k)])) for k in n])
            ax.plot(
                n,
                value,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                markerfacecolor="none",
                color=color,
                label=f"{grid}: {name}",
            )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("$n$ (items)")
    ax.set_ylabel("fraction of instances")
    ax.set_title("Does the design identify the fit?")
    _integer_xticks(ax, [int(r["n_items"]) for r in cells])
    ax.legend(fontsize="x-small", loc="best")

    ax = axes[1]
    for grid, (linestyle, color) in grids.items():
        rows = [
            r
            for r in cells
            if r["grid"] == grid
            and r["m_class"] == FEASIBLE
            and r["series"] == "bias_zne_scalar_reported"
            and _number(r["value"]) is not None
        ]
        by_n_value: dict[int, list[float]] = {}
        by_n_se: dict[int, list[float]] = {}
        for row in rows:
            by_n_value.setdefault(int(row["n_items"]), []).append(
                float(_number(row["value"]) or 0.0)
            )
            by_n_se.setdefault(int(row["n_items"]), []).append(
                float(_number(row["se"]) or 0.0)
            )
        if not by_n_value:
            continue
        n = np.array(sorted(by_n_value), dtype=float)
        value = np.array([float(np.mean(by_n_value[int(k)])) for k in n])
        se = np.array([float(np.mean(by_n_se[int(k)])) for k in n])
        ax.errorbar(
            n,
            value,
            yerr=se,
            linestyle=linestyle,
            marker="o",
            markersize=4,
            markerfacecolor="none",
            capsize=2,
            elinewidth=0.8,
            color=color,
            label=f"{grid} grid",
        )
    ax.axhline(0.0, color=OKABE_ITO["grey"], linewidth=0.8, zorder=1)
    ax.set_xlabel("$n$ (items)")
    ax.set_ylabel("bias of reported $P(\\mathrm{solution})$")
    ax.set_title("What the reporting rule delivers")
    _integer_xticks(ax, [int(r["n_items"]) for r in cells])
    ax.legend(fontsize="x-small", loc="best")
    return save_figure(fig, "fig_sim_grid", directory=out)


# ---------------------------------------------------------------------------
# fig_sim_sweep
# ---------------------------------------------------------------------------


def load_sweeps(data_dir: Path) -> dict[str, dict[str, Any]]:
    """
    The iteration sweeps, keyed by ``"<instance>_<executor>"``.

    Parameters
    ----------
    data_dir:
        ``paper/data``.

    Returns
    -------
    dict
        Empty when the sweep stage has not run.
    """
    out: dict[str, dict[str, Any]] = {}
    for directory in sorted(data_dir.glob("sim_sweep_*")):
        for path in sorted(directory.glob("*.json")):
            out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


def fig_sim_sweep(
    sweeps: dict[str, dict[str, Any]],
    out: Path,
    *,
    hardware: Sequence[dict[str, Any]] | None = None,
    name: str = "fig_sim_sweep",
) -> list[Path]:
    """
    ``P(solution)`` against the Grover iteration count.

    Parameters
    ----------
    sweeps:
        As :func:`load_sweeps` returns.
    out:
        Figures directory.
    hardware:
        ``hw_summary.json`` records; those of a swept instance (its ``r_opt``
        record and its sweep variants) are added at ``lambda = 1`` with their
        95 % intervals.
    name:
        Output file stem.

    Returns
    -------
    list[Path]
        Files written; empty when nothing has been swept.
    """
    names = sorted({key.rsplit("_", 1)[0] for key in sweeps})
    if not names:
        return []

    fig, axes = figure(
        width=FULL_WIDTH_IN if len(names) > 1 else COLUMN_WIDTH_IN,
        aspect=0.5,
        ncols=max(1, len(names)),
        squeeze=False,
    )
    flat = axes.ravel()

    for ax, instance in zip(flat, names, strict=False):
        _sweep_panel(ax, sweeps, instance)
        points = sorted(
            (r for r in hardware or [] if r.get("instance") == instance
             and isinstance(r.get("hardware"), (int, float))),
            key=lambda r: r["iterations"],
        )
        if points:
            style = style_for("hardware")
            value = np.array([r["hardware"] for r in points], dtype=float)
            lo = np.array([r["hardware_lo"] for r in points], dtype=float)
            hi = np.array([r["hardware_hi"] for r in points], dtype=float)
            ax.errorbar(
                [r["iterations"] for r in points],
                value,
                yerr=np.abs(np.vstack([value - lo, hi - value])),
                capsize=2,
                elinewidth=0.8,
                label=style.label,
                **style.line_kwargs(linestyle="none", markersize=6,
                                    markerfacecolor=style.color),
            )
    flat[0].set_ylabel("$P(\\mathrm{solution})$")
    figure_legend(fig, flat, ncol=5 if hardware else 4)
    return save_figure(fig, name, directory=out)


def _sweep_panel(ax: Any, sweeps: dict[str, dict[str, Any]], name: str) -> None:
    """One instance's simulated sweep: analytic curve, ideal, noisy, ZNE."""
    analytic_drawn = False
    for executor, key in (
        ("ideal", "ideal"),
        ("noisy", "noisy"),
        ("zne", "zne_scalar"),
    ):
        payload = sweeps.get(f"{name}_{executor}")
        if payload is None:
            continue
        rows = payload["rows"]
        r = np.array([row["iterations"] for row in rows], dtype=float)
        if not analytic_drawn:
            # The analytic success probability is smooth in r, so it is
            # drawn as a curve rather than by joining the sampled points:
            # sin^2((2r+1) theta), theta = arcsin(sqrt(M / 2**n)).  The
            # measured points must land *on* it, which is the check this
            # panel makes; interpolating between them would hide a miss.
            n_items = len(payload["instance"]["items"])
            theta = math.asin(math.sqrt(payload["M"] / 2**n_items))
            fine = np.linspace(0.0, float(r.max()) + 0.3, 200)
            ax.plot(
                fine,
                np.sin((2.0 * fine + 1.0) * theta) ** 2,
                color=OKABE_ITO["grey"],
                linewidth=1.0,
                zorder=1,
                label="$\\sin^2((2r{+}1)\\theta)$",
            )
            analytic_drawn = True
        if executor == "zne":
            # Flags and reported value under the current rule (the analytic
            # line is the exact P_ideal) and REPORT_BOUNDS; an interval is
            # drawn only for a reported extrapolation -- a lambda = 1 fallback
            # has the shot interval of that observation, not the fit's
            # (unbounded) one.  The sweep rows carry no counts, so
            # REPORT_BOUNDS = "bounded" is refused here (rederive_flags raises).
            flags = [
                rederive_flags(row["zne"], row.get("analytic")) if row.get("zne") else {}
                for row in rows
            ]
            value = np.array(
                [f.get("reported_value", np.nan) for f in flags], dtype=float
            )
            ci = np.array(
                [
                    (f.get("reported_ci") or [np.nan, np.nan])
                    if f.get("reported_source") == "extrapolation"
                    else [np.nan, np.nan]
                    for f in flags
                ],
                dtype=float,
            )
            lo, hi = ci[:, 0], ci[:, 1]
        else:
            value = np.array([row["p_sol"] for row in rows], dtype=float)
            ci = np.array([row["p_sol_ci"] for row in rows], dtype=float)
            lo, hi = ci[:, 0], ci[:, 1]
        errors = np.vstack(
            [
                np.nan_to_num(value - lo, nan=0.0),
                np.nan_to_num(hi - value, nan=0.0),
            ]
        )
        style = style_for(key)
        ax.errorbar(
            r,
            value,
            yerr=np.abs(errors),
            capsize=2,
            elinewidth=0.8,
            label=style.label,
            **style.line_kwargs(linestyle="none", markersize=5),
        )
        ax.axhline(
            payload["p_uniform"],
            color=OKABE_ITO["grey"],
            linestyle=":",
            linewidth=0.8,
            zorder=1,
        )
    ax.set_title(name, fontsize="small")
    ax.set_xlabel("Grover iterations $r$")
    ax.set_xticks(
        sorted(
            {
                row["iterations"]
                for key, payload in sweeps.items()
                if key.startswith(name)
                for row in payload["rows"]
            }
        )
    )


# ---------------------------------------------------------------------------
# fig_sim_models
# ---------------------------------------------------------------------------


def fig_sim_models(
    rows: Sequence[dict[str, str]],
    models: Sequence[dict[str, str]],
    grid: str,
    out: Path,
) -> list[Path]:
    """
    Model comparison on the four-point grid, identified fits only.

    Parameters
    ----------
    rows:
        Rows of ``sim_per_instance.csv``.
    models:
        Rows of ``sim_models.csv``.
    grid:
        The grid to compare on.
    out:
        Figures directory.

    Returns
    -------
    list[Path]
        Files written; empty when that grid was not run.
    """
    members = [r for r in rows if r["grid"] == grid and (_number(r["M"]) or 0) >= 1]
    summary = [r for r in models if r["grid"] == grid]
    if not members or not summary:
        return []

    fig, axes = figure(width=FULL_WIDTH_IN, aspect=0.42, ncols=2)

    ax = axes[0]
    for index, method in enumerate(METHOD_STYLE):
        residuals = [
            float(_number(row[f"fit_{method}"]) or 0.0) - float(_number(row["ideal"]) or 0.0)
            for row in members
            if _flag(row[f"fit_{method}_identifiable"]) is True
            and _number(row[f"fit_{method}"]) is not None
            and _number(row["ideal"]) is not None
        ]
        if not residuals:
            continue
        colour, marker = METHOD_STYLE[method]
        jitter = np.random.default_rng(index).normal(0.0, 0.06, len(residuals))
        ax.plot(
            index + jitter,
            residuals,
            linestyle="none",
            marker=marker,
            markersize=3,
            alpha=0.5,
            color=colour,
            markerfacecolor="none",
        )
        ax.plot(
            [index - 0.25, index + 0.25],
            [float(np.median(residuals))] * 2,
            color=colour,
            linewidth=1.6,
        )
    ax.axhline(0.0, color=OKABE_ITO["grey"], linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(METHOD_STYLE)))
    ax.set_xticklabels(list(METHOD_STYLE), rotation=20, ha="right")
    ax.set_ylabel("fit $-$ ideal $P(\\mathrm{solution})$")
    ax.set_title(f"Residuals on the {grid} grid")

    ax = axes[1]
    order = [row for row in summary if row["method"] in METHOD_STYLE]
    positions = np.arange(len(order))
    values = [float(_number(row["median_abs_error"]) or 0.0) for row in order]
    colours = [METHOD_STYLE[row["method"]][0] for row in order]
    ax.bar(positions, values, color=colours, edgecolor="black", linewidth=0.5)
    for position, row, value in zip(positions, order, values, strict=True):
        ax.annotate(
            f"N={row['n_used']}",
            (position, value),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize="xx-small",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([row["method"] for row in order], rotation=20, ha="right")
    ax.set_ylabel("median $|$fit $-$ ideal$|$")
    ax.set_title("Identified fits only")
    return save_figure(fig, "fig_sim_models", directory=out)


# ---------------------------------------------------------------------------
# Hardware figures (``--set hw``)
# ---------------------------------------------------------------------------


def load_hw_summary(data_dir: Path) -> dict[str, Any] | None:
    """``hw_summary.json`` as ``hw_aggregate.py`` wrote it, or ``None``."""
    path = data_dir / "hw_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _with_signal(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records whose ideal answer is not random guessing (direction +1 / -1)."""
    return [r for r in records if r.get("signal_direction") in (1, -1)]


def fig_hw_example(summary: dict[str, Any], out: Path, key: str = EXAMPLE_RECORD) -> list[Path]:
    """
    The worked example: one instance's output distribution four ways.

    Parameters
    ----------
    summary:
        ``hw_summary.json``.
    out:
        Figures directory.
    key:
        Record to show.

    Returns
    -------
    list[Path]
        Files written; empty when the record is not in the summary.
    """
    record = next((r for r in summary["records"] if r["key"] == key), None)
    if record is None:
        return []
    detail = record["detail"]
    dists = detail["distributions"]
    series = (
        ("ideal", "ideal", "Ideal"),
        ("model_exec", "noise_model", "Noise model ($\\lambda=1$)"),
        ("hardware", "hardware", "Hardware ($\\lambda=1$)"),
        ("hardware_zne_pointwise", "zne_pointwise", "Hardware + pointwise ZNE"),
    )
    present = [(name, style, label) for name, style, label in series if dists.get(name)]
    bits = sorted({b for name, _, _ in present for b in dists[name]})
    solutions = set(detail.get("solution_bitstrings") or [])

    fig, ax = figure(width=COLUMN_WIDTH_IN, aspect=0.62)
    width = 0.8 / max(1, len(present))
    x = np.arange(len(bits))
    for index, (name, key_style, label) in enumerate(present):
        style = style_for(key_style)
        values = [float(dists[name].get(b, 0.0)) for b in bits]
        ax.bar(
            x + (index - (len(present) - 1) / 2) * width,
            values,
            width,
            label=label,
            **style.bar_kwargs(edgecolor="black", linewidth=0.4),
        )
    ax.axhline(record["p_uniform"], color=OKABE_ITO["grey"], linestyle=":",
               linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{b}\n(solution)" if b in solutions else b for b in bits], fontsize="small"
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("probability")
    ax.set_title(f"{record['key']}: $n={record['n_items']}$, $M={record['M']}$, "
                 f"{record['two_q_count']} two-qubit gates", fontsize="small")
    legend_below(ax, ncol=2)
    return save_figure(fig, "fig_hw_example", directory=out)


def fig_hw_signal(
    summary: dict[str, Any], sim_rows: Sequence[dict[str, str]], out: Path, grid: str
) -> list[Path]:
    """
    Signal kept at ``lambda = 1`` against two-qubit gate count.

    Parameters
    ----------
    summary:
        ``hw_summary.json``.
    sim_rows:
        Rows of ``sim_per_instance.csv``; the ``grid`` rows give each WP5
        instance once (its unmitigated observation does not depend on the grid).
    out:
        Figures directory.
    grid:
        Which grid's rows to take the WP5 instances from.

    Returns
    -------
    list[Path]
        Files written.
    """
    fig, ax = figure(width=COLUMN_WIDTH_IN, aspect=0.85)

    cloud = []
    for row in sim_rows:
        if row["grid"] != grid or row.get("signal_direction") not in ("1", "-1"):
            continue
        gates, noisy = _number(row["two_q_count"]), _number(row["noisy"])
        ideal, uniform = _number(row["p_ideal_exact"]), _number(row["p_uniform"])
        if None in (gates, noisy, ideal, uniform) or math.isclose(ideal or 0, uniform or 0):
            continue
        cloud.append((gates, ((noisy or 0) - (uniform or 0)) / ((ideal or 0) - (uniform or 0))))
    if cloud:
        # The WP5 campaign is the backdrop here, so it is drawn in the
        # reference grey: in its own "noisy" blue, faded, it reads as the
        # sky-blue device noise model drawn on top of it.
        ax.plot(
            [c[0] for c in cloud],
            [c[1] for c in cloud],
            linestyle="none",
            marker=style_for("noisy").marker,
            markersize=2.5,
            alpha=0.5,
            color=OKABE_ITO["grey"],
            markerfacecolor="none",
            zorder=1,
            label=f"Simulation, WP5 ({len(cloud)} inst.)",
        )

    records = sorted(_with_signal(summary["records"]), key=lambda r: r["two_q_count"])
    alpha = summary.get("alpha_hat")
    record_gates = [r["two_q_count"] for r in records]
    for column, key, label, filled in (
        ("kept_model_exec", "noise_model", "Device noise model", False),
        ("kept_model_alpha", "noise_model",
         f"Noise model, errors $\\times\\,{alpha:.2f}$" if alpha else None, True),
        ("kept_hardware", "hardware", "Hardware", True),
    ):
        if label is None:
            continue
        values = [r.get(column) for r in records]
        if not any(isinstance(v, (int, float)) for v in values):
            continue
        style = style_for(key)
        ax.plot(
            record_gates,
            [np.nan if v is None else v for v in values],
            linestyle="none",
            marker="o" if (key == "noise_model" and filled) else style.marker,
            markersize=5 if key == "hardware" else 4,
            color=style.color,
            markerfacecolor=style.color if filled else "none",
            zorder=style.zorder,
            label=label,
        )
    # Each record's model -> hardware drop, so the gap reads per circuit.
    for r in records:
        if isinstance(r.get("kept_model_exec"), (int, float)) and isinstance(
            r.get("kept_hardware"), (int, float)
        ):
            ax.plot([r["two_q_count"]] * 2, [r["kept_model_exec"], r["kept_hardware"]],
                    color=OKABE_ITO["grey"], linewidth=0.5, zorder=1)
    ax.axhline(0.0, color=OKABE_ITO["grey"], linestyle=":", linewidth=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel("two-qubit gates (transpiled, $\\lambda=1$)")
    ax.set_ylabel("signal kept, $(P-P_\\mathrm{unif})/(P_\\mathrm{ideal}-P_\\mathrm{unif})$")
    legend_below(ax, ncol=2)
    return save_figure(fig, "fig_hw_signal", directory=out)


def _folding_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The shallowest record of each status in :data:`FOLDING_STATUSES`."""
    chosen = []
    for status in FOLDING_STATUSES:
        candidates = [r for r in records if r.get("zne_status") == status]
        if candidates:
            chosen.append(min(candidates, key=lambda r: r["two_q_count"]))
    return chosen


def fig_hw_folding(summary: dict[str, Any], out: Path) -> list[Path]:
    """
    ``P(solution)`` against ``lambda`` with the fitted exponential.

    One panel per estimator status (resolved; identified, not resolved; not
    identifiable), each the shallowest record with that status.  Hardware
    points carry 95 % intervals; the device noise model's points and fit are
    drawn beside them; ``P_uniform`` is the dotted line the folded points
    collapse onto.

    Returns
    -------
    list[Path]
        Files written; empty when no record has a fit.
    """
    chosen = _folding_records(summary["records"])
    if not chosen:
        return []
    fig, axes = figure(width=FULL_WIDTH_IN, aspect=0.36, ncols=len(chosen),
                       sharey=True, squeeze=False)
    flat = axes.ravel()
    hw, model = style_for("hardware"), style_for("noise_model")
    for ax, r in zip(flat, chosen, strict=True):
        d = r["detail"]
        scales = np.array(d["scales"], dtype=float)
        observed = np.array(d["hardware"], dtype=float)
        sigma = np.array(d["hardware_sigma"], dtype=float)
        curve = d["fit_curve"]
        if curve.get("p"):
            reported = r["zne_status"] == "resolved"
            ax.plot(curve["lambda"], curve["p"], color=hw.color, linewidth=1.0,
                    linestyle="-" if reported else "--", zorder=2,
                    label="hardware fit" if reported else "hardware fit (not reported)")
        if d.get("model_exec"):
            ax.plot(scales, d["model_exec"], linestyle="none", markersize=5,
                    label="device noise model",
                    **{k: v for k, v in model.line_kwargs().items() if k != "linestyle"})
            mcurve = d.get("fit_curve_model_exec") or {}
            if mcurve.get("p"):
                ax.plot(mcurve["lambda"], mcurve["p"], color=model.color, linewidth=0.8,
                        linestyle=model.linestyle, zorder=1)
        ax.errorbar(scales, observed, yerr=Z95 * sigma, capsize=2, elinewidth=0.8,
                    label="hardware",
                    **hw.line_kwargs(linestyle="none", markersize=6,
                                     markerfacecolor=hw.color))
        ax.axhline(r["p_uniform"], color=OKABE_ITO["grey"], linestyle=":", linewidth=0.8,
                   zorder=1, label="$P_\\mathrm{unif}$")
        ideal = style_for("ideal")
        ax.plot([0.0], [r["p_ideal"]], linestyle="none", marker="*", markersize=9,
                color=ideal.color, zorder=ideal.zorder, label="ideal")
        if r["zne_status"] == "resolved" and r.get("zne_reported_lo") is not None:
            value = float(r["zne_reported"])
            zne = style_for("zne_scalar")
            ax.errorbar(
                [0.0], [value],
                yerr=[[value - float(r["zne_reported_lo"])],
                      [float(r["zne_reported_hi"]) - value]],
                capsize=2, elinewidth=0.8, label="ZNE (reported)",
                **zne.line_kwargs(linestyle="none", markersize=6,
                                  markerfacecolor=zne.color),
            )
        ax.set_title(f"{r['key']}, {r['two_q_count']} two-qubit gates\n{r['zne_status']}",
                     fontsize="small")
        ax.set_xlabel("noise scale $\\lambda$")
        ax.set_xticks([0, *d["scales"]])
        ax.set_xlim(-0.3, max(d["scales"]) + 0.5)
    flat[0].set_ylabel("$P(\\mathrm{solution})$")
    flat[0].set_ylim(0.0, 1.05)
    figure_legend(fig, flat, ncol=4)
    return save_figure(fig, "fig_hw_folding", directory=out)


def fig_hw_zne(summary: dict[str, Any], out: Path) -> list[Path]:
    """
    Our scalar ZNE against Mitiq on the same hardware counts, per record.

    Each row is a record with signal, ordered by two-qubit gate count.  Our
    estimate is drawn filled where it is an extrapolation (resolved and
    identified, with its bootstrap interval) and open where the estimator
    declines and reports ``lambda = 1``.  Mitiq is drawn with its interval; a
    value beyond :data:`ZNE_AXIS_MAX` sits on the edge with its number.

    Returns
    -------
    list[Path]
        Files written; empty when no record has signal.
    """
    records = sorted(_with_signal(summary["records"]), key=lambda r: r["two_q_count"])
    if not records:
        return []
    fig, ax = figure(width=COLUMN_WIDTH_IN, aspect=0.95)
    y = np.arange(len(records))[::-1].astype(float)
    ax.axvspan(1.0, ZNE_AXIS_MAX, color=OKABE_ITO["grey"], alpha=0.12, zorder=0)
    ideal, hw = style_for("ideal"), style_for("hardware")
    ours, mitiq = style_for("zne_scalar"), style_for("mitiq")
    edge = ZNE_AXIS_MAX - 0.03
    labelled: set[str] = set()

    def once(name: str) -> str | None:
        if name in labelled:
            return None
        labelled.add(name)
        return name

    for row_y, r in zip(y, records, strict=True):
        ax.plot([r["p_ideal"]], [row_y], linestyle="none", marker="|", markersize=10,
                markeredgewidth=1.6, color=ideal.color, zorder=ideal.zorder,
                label=once("ideal"))
        ax.plot([r["hardware"]], [row_y + 0.18], linestyle="none", marker=hw.marker,
                markersize=4, color=hw.color, markerfacecolor=hw.color, zorder=hw.zorder,
                label=once("hardware, $\\lambda=1$"))
        resolved = r["zne_status"] == "resolved"
        value = float(r["zne_reported"])
        if resolved:
            ax.errorbar([value], [row_y], xerr=[[value - r["zne_reported_lo"]],
                                                [r["zne_reported_hi"] - value]],
                        capsize=2, elinewidth=0.8, label=once("ours, extrapolated"),
                        **ours.line_kwargs(linestyle="none", markersize=5,
                                           markerfacecolor=ours.color))
        else:
            ax.plot([value], [row_y], linestyle="none", markersize=5,
                    label=once("ours, declines (reports $\\lambda=1$)"),
                    **{k: v for k, v in ours.line_kwargs().items() if k != "linestyle"})
        m = r.get("mitiq")
        if isinstance(m, (int, float)) and math.isfinite(m):
            shown = min(float(m), edge)
            lo = r.get("mitiq_lo")
            hi = r.get("mitiq_hi")
            xerr = None
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                xerr = [[max(0.0, shown - max(float(lo), 0.0))],
                        [max(0.0, min(float(hi), edge) - shown)]]
            ax.errorbar([shown], [row_y - 0.18], xerr=xerr, capsize=2, elinewidth=0.6,
                        label=once("Mitiq"),
                        **mitiq.line_kwargs(linestyle="none", markersize=4))
            if isinstance(hi, (int, float)) and hi > edge:
                # The interval runs past the axis: an arrowhead, not a cap.
                ax.plot([edge], [row_y - 0.18], linestyle="none", marker=">",
                        markersize=4, color=mitiq.color, zorder=mitiq.zorder + 1)
            if m > edge:
                ax.annotate(f"{m:.2f}$\\rightarrow$", (edge, row_y - 0.18),
                            textcoords="offset points", xytext=(-2, 4), ha="right",
                            fontsize="xx-small", color=mitiq.color)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['key']} ({r['two_q_count']})" for r in records],
                       fontsize="xx-small")
    ax.set_xlim(-0.02, ZNE_AXIS_MAX)
    ax.set_xlabel("$P(\\mathrm{solution})$")
    ax.annotate("outside $[0,1]$", (1.0 + (ZNE_AXIS_MAX - 1.0) / 2, y.max() + 0.6),
                ha="center", fontsize="xx-small", color=OKABE_ITO["grey"],
                annotation_clip=False)
    legend_below(ax, ncol=2)
    return save_figure(fig, "fig_hw_zne", directory=out)


def build_hw(data_dir: Path, figures_dir: Path, grid: str) -> list[Path]:
    """
    Build the hardware figures.

    Raises
    ------
    FileNotFoundError
        Without ``hw_summary.json``.
    """
    summary = load_hw_summary(data_dir)
    if summary is None:
        raise FileNotFoundError(
            f"No hw_summary.json in {data_dir}. Run hw_aggregate.py first."
        )
    records = summary["records"]
    written: list[Path] = []
    written += fig_hw_example(summary, figures_dir)
    written += fig_hw_signal(summary, read_csv(data_dir / "sim_per_instance.csv"),
                             figures_dir, grid)
    written += fig_hw_folding(summary, figures_dir)
    written += fig_hw_zne(summary, figures_dir)
    written += fig_sim_sweep(load_sweeps(data_dir), figures_dir, hardware=records,
                             name="fig_hw_sweep")
    cells = read_csv(data_dir / "sim_cells.csv")
    if cells:
        written += fig_sim_dtv(cells, read_csv(data_dir / "sim_pathology.csv"), grid,
                               figures_dir, hardware=records, name="fig_hw_dtv")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_sim(data_dir: Path, figures_dir: Path, grid: str) -> list[Path]:
    """
    Build the five simulation figures.

    Parameters
    ----------
    data_dir:
        ``paper/data``.
    figures_dir:
        ``paper/figures``.
    grid:
        Which grid the main figures use.

    Returns
    -------
    list[Path]
        Every file written.
    """
    cells = read_csv(data_dir / "sim_cells.csv")
    rows = read_csv(data_dir / "sim_per_instance.csv")
    pathology = read_csv(data_dir / "sim_pathology.csv")
    models = read_csv(data_dir / "sim_models.csv")
    if not cells:
        raise FileNotFoundError(
            f"No sim_cells.csv in {data_dir}. Run aggregate.py first."
        )

    written: list[Path] = []
    written += fig_sim_psol(cells, grid, figures_dir)
    written += fig_sim_dtv(cells, pathology, grid, figures_dir)
    written += fig_sim_grid(cells, figures_dir)
    written += fig_sim_sweep(load_sweeps(data_dir), figures_dir)
    written += fig_sim_models(rows, models, MODEL_GRID, figures_dir)
    return written


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
    parser.add_argument("--set", dest="which", choices=("sim", "hw"), required=True)
    parser.add_argument("--grid", default=MAIN_GRID)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Build one figure set.

    Parameters
    ----------
    argv:
        Command line; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        ``0`` on success.
    """
    args = parse_args(argv)
    build = build_hw if args.which == "hw" else build_sim
    written = build(args.data_dir, args.figures_dir, args.grid)
    for path in written:
        print(f"wrote {path}")
    if not written:
        print("nothing written: the campaign has produced no tables yet")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
