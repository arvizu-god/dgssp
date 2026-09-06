"""
plotting.py

One matplotlib style and one executor colour/marker map for the whole paper.

Every figure imports from here, so "zne_pointwise" is the same green triangle
in Figure 2 as in Figure 7 and a reader never has to re-learn the legend. The
palette is Okabe--Ito, which is distinguishable under deuteranopia, protanopia
and tritanopia, and each series also carries a distinct marker and line style
so the figures survive greyscale printing.

Sizes target a two-column journal (EPJ Quantum Technology / QIP): text is set
at 8 pt inside a 3.35 in column, which is what the body font renders as after
the publisher scales the figure to column width. Do not resize figures in
LaTeX -- build them at final size here so the font sizes come out right.

Usage
-----
::

    from plotting import EXECUTORS, apply_style, figure, save_figure

    apply_style()
    fig, ax = figure()
    style = EXECUTORS["zne_pointwise"]
    ax.plot(x, y, label=style.label, **style.line_kwargs())
    save_figure(fig, "p_solution_vs_iterations")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from _common import FIGURES_DIR

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

#: Single-column width in inches for a two-column journal page.
COLUMN_WIDTH_IN = 3.35

#: Full text width in inches (a figure spanning both columns).
FULL_WIDTH_IN = 6.9

#: Default height:width ratio. 0.72 leaves room for a legend without squashing.
DEFAULT_ASPECT = 0.72


# ---------------------------------------------------------------------------
# Palette (Okabe--Ito)
# ---------------------------------------------------------------------------

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#7F7F7F",
}


@dataclass(frozen=True)
class SeriesStyle:
    """
    Fixed visual identity of one executor across every figure in the paper.

    Attributes
    ----------
    key:
        The executor name as it appears in ``Record.executor`` /
        ``Record.mitigation_mode``.
    label:
        Legend text.
    color:
        Hex colour from :data:`OKABE_ITO`.
    marker:
        Matplotlib marker code.
    linestyle:
        Matplotlib line style.
    zorder:
        Draw order; reference curves sit on top of noisy ones.
    """

    key: str
    label: str
    color: str
    marker: str
    linestyle: str
    zorder: int = 2

    def line_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """
        Keyword arguments for ``ax.plot``.

        Parameters
        ----------
        **overrides:
            Any keyword to override (e.g. ``alpha=0.5``).

        Returns
        -------
        dict
            Colour, marker, line style and z-order for this series.
        """
        kwargs: dict[str, Any] = {
            "color": self.color,
            "marker": self.marker,
            "linestyle": self.linestyle,
            "markerfacecolor": "none",
            "markeredgecolor": self.color,
            "zorder": self.zorder,
        }
        kwargs.update(overrides)
        return kwargs

    def bar_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """
        Keyword arguments for ``ax.bar`` / ``ax.errorbar`` fills.

        Parameters
        ----------
        **overrides:
            Any keyword to override.

        Returns
        -------
        dict
            Face and edge colours for this series.
        """
        kwargs: dict[str, Any] = {
            "color": self.color,
            "edgecolor": self.color,
            "zorder": self.zorder,
        }
        kwargs.update(overrides)
        return kwargs


#: The fixed executor -> style map. Extend it; never reassign a colour, or two
#: figures in the same paper will disagree about what green means.
EXECUTORS: dict[str, SeriesStyle] = {
    "ideal": SeriesStyle(
        key="ideal",
        label="Ideal",
        color=OKABE_ITO["black"],
        marker="o",
        linestyle="-",
        zorder=5,
    ),
    "noisy": SeriesStyle(
        key="noisy",
        label="Noisy (unmitigated)",
        color=OKABE_ITO["blue"],
        marker="s",
        linestyle="--",
        zorder=3,
    ),
    "zne_scalar": SeriesStyle(
        key="zne_scalar",
        label="ZNE (scalar)",
        color=OKABE_ITO["orange"],
        marker="^",
        linestyle="-",
        zorder=4,
    ),
    "zne_pointwise": SeriesStyle(
        key="zne_pointwise",
        label="ZNE (pointwise)",
        color=OKABE_ITO["bluish_green"],
        marker="v",
        linestyle="-",
        zorder=4,
    ),
    "mitiq": SeriesStyle(
        key="mitiq",
        label="Mitiq ZNE",
        color=OKABE_ITO["reddish_purple"],
        marker="D",
        linestyle=":",
        zorder=3,
    ),
    "hardware": SeriesStyle(
        key="hardware",
        label="Hardware",
        color=OKABE_ITO["vermillion"],
        marker="X",
        linestyle="-",
        zorder=6,
    ),
}


def style_for(key: str) -> SeriesStyle:
    """
    Look up an executor's style, failing loudly on an unknown name.

    Parameters
    ----------
    key:
        Executor or mitigation-mode name.

    Returns
    -------
    SeriesStyle
        The registered style.

    Raises
    ------
    KeyError
        If the name is not in :data:`EXECUTORS`.  This is deliberate: a typo
        should not silently produce an off-palette colour.
    """
    if key not in EXECUTORS:
        raise KeyError(
            f"Unknown executor '{key}'. Known: {sorted(EXECUTORS)}. "
            "Add it to plotting.EXECUTORS rather than picking a colour inline."
        )
    return EXECUTORS[key]


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

#: rcParams applied by :func:`apply_style`.
PAPER_RCPARAMS: dict[str, Any] = {
    # Fonts: a serif body to match the journal, at final print size.
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 9,
    "mathtext.fontset": "dejavuserif",
    # Lines and markers: thin enough for a 3.35 in column.
    "lines.linewidth": 1.2,
    "lines.markersize": 4.0,
    "lines.markeredgewidth": 1.0,
    # Axes.
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": cycler(color=[s.color for s in EXECUTORS.values()]),
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "grid.color": OKABE_ITO["grey"],
    # Ticks.
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    # Legend.
    "legend.frameon": False,
    "legend.handlelength": 2.0,
    "legend.columnspacing": 1.0,
    "legend.borderaxespad": 0.3,
    # Figure and output.
    "figure.dpi": 150,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.transparent": False,
    # Keep text as text in the PDF so the publisher can search and reflow it.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style() -> None:
    """
    Install the paper's rcParams globally.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If :data:`PAPER_RCPARAMS` contains a key matplotlib does not know.
    """
    # matplotlib >= 3.10 types RcParams keys as a Literal union of every known
    # rcParam, so a plain dict[str, Any] is rejected statically even when every
    # key is valid.  Keeping PAPER_RCPARAMS keyed by `str` is what makes it
    # readable and editable, so the mismatch is absorbed here rather than by
    # contorting the table.  Nothing is lost: RcParams.update validates through
    # __setitem__ and raises KeyError on an unknown key at import time, which
    # tests/test_paper_plotting.py exercises.
    mpl.rcParams.update(cast("Any", PAPER_RCPARAMS))


def figure(
    *,
    width: float = COLUMN_WIDTH_IN,
    aspect: float = DEFAULT_ASPECT,
    nrows: int = 1,
    ncols: int = 1,
    **subplot_kwargs: Any,
) -> tuple[Any, Any]:
    """
    Create a correctly sized figure with the paper style applied.

    Parameters
    ----------
    width:
        Figure width in inches; :data:`COLUMN_WIDTH_IN` or
        :data:`FULL_WIDTH_IN`.
    aspect:
        Height as a fraction of width, per subplot row.
    nrows, ncols:
        Subplot grid.
    **subplot_kwargs:
        Forwarded to ``plt.subplots``.

    Returns
    -------
    tuple
        ``(figure, axes)`` exactly as ``plt.subplots`` returns them.
    """
    apply_style()
    height = width * aspect * nrows
    return plt.subplots(nrows, ncols, figsize=(width, height), **subplot_kwargs)


def save_figure(
    fig: Any,
    name: str,
    *,
    directory: Path | str = FIGURES_DIR,
    formats: tuple[str, ...] = ("pdf", "png"),
    close: bool = True,
) -> list[Path]:
    """
    Write a figure to ``paper/figures`` in every requested format.

    The ``.pdf`` is the paper asset and is tracked in git; the ``.png`` is a
    preview for quick viewing and is gitignored.

    Parameters
    ----------
    fig:
        The matplotlib figure.
    name:
        Base filename, no extension.
    directory:
        Output directory; created if missing.
    formats:
        Extensions to write.
    close:
        Close the figure afterwards, so a script that makes many figures does
        not accumulate them in memory.

    Returns
    -------
    list[Path]
        The files written, in ``formats`` order.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for ext in formats:
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, format=ext)
        written.append(path)

    if close:
        plt.close(fig)
    return written


def legend_below(ax: Any, *, ncol: int = 3, **kwargs: Any) -> Any:
    """
    Put a legend under the axes, where it does not cover data.

    Parameters
    ----------
    ax:
        The axes to attach the legend to.
    ncol:
        Number of legend columns.
    **kwargs:
        Forwarded to ``ax.legend``.

    Returns
    -------
    matplotlib.legend.Legend
        The legend created.
    """
    defaults: dict[str, Any] = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, -0.18),
        "ncol": ncol,
    }
    defaults.update(kwargs)
    return ax.legend(**defaults)


__all__ = [
    "COLUMN_WIDTH_IN",
    "FULL_WIDTH_IN",
    "DEFAULT_ASPECT",
    "OKABE_ITO",
    "SeriesStyle",
    "EXECUTORS",
    "style_for",
    "PAPER_RCPARAMS",
    "apply_style",
    "figure",
    "save_figure",
    "legend_below",
]
