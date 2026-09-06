"""
Tests for ``paper/scripts/plotting.py``.

The style module is loaded by every figure script, so a bad rcParam key or a
duplicated colour should fail here rather than three weeks later while
regenerating figures for a referee.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "paper" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

plotting = pytest.importorskip("plotting")


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def test_every_rcparam_key_is_real():
    """``RcParams.update`` raises on an unknown key, so applying is the check."""
    plotting.apply_style()

    for key, value in plotting.PAPER_RCPARAMS.items():
        assert key in matplotlib.rcParams, f"{key} is not a matplotlib rcParam"
        assert matplotlib.rcParams[key] is not None or value is None


def test_figures_are_built_at_final_print_size():
    """Figures are sized here, not scaled in LaTeX, so font sizes come out right."""
    fig, _ = plotting.figure()
    try:
        width, height = fig.get_size_inches()
        assert width == pytest.approx(plotting.COLUMN_WIDTH_IN)
        assert height == pytest.approx(
            plotting.COLUMN_WIDTH_IN * plotting.DEFAULT_ASPECT
        )
    finally:
        matplotlib.pyplot.close(fig)


def test_save_figure_writes_pdf_and_png(tmp_path):
    """The .pdf is the paper asset; the .png is the gitignored preview."""
    fig, ax = plotting.figure()
    ax.plot([1, 2, 3], [0.9, 0.5, 0.2])

    written = plotting.save_figure(fig, "check", directory=tmp_path)

    assert [p.name for p in written] == ["check.pdf", "check.png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


# ---------------------------------------------------------------------------
# Executor identity
# ---------------------------------------------------------------------------


def test_all_six_executors_are_registered():
    """The legend vocabulary the paper commits to."""
    assert set(plotting.EXECUTORS) == {
        "ideal",
        "noisy",
        "zne_scalar",
        "zne_pointwise",
        "mitiq",
        "hardware",
    }


def test_colours_and_markers_are_unique():
    """Two series sharing a colour or a marker would be unreadable in print."""
    styles = list(plotting.EXECUTORS.values())
    assert len({s.color for s in styles}) == len(styles)
    assert len({s.marker for s in styles}) == len(styles)


def test_every_colour_comes_from_the_palette():
    """No off-palette colour sneaks in; the palette is the accessibility claim."""
    palette = set(plotting.OKABE_ITO.values())
    for style in plotting.EXECUTORS.values():
        assert style.color in palette


def test_keys_match_their_map_entry():
    """``EXECUTORS['noisy'].key == 'noisy'``, so a style can be passed around alone."""
    for key, style in plotting.EXECUTORS.items():
        assert style.key == key


def test_line_kwargs_are_plottable_and_overridable():
    """The kwargs go straight into ``ax.plot`` without further massaging."""
    style = plotting.style_for("zne_pointwise")
    kwargs = style.line_kwargs(alpha=0.5)

    assert kwargs["color"] == style.color
    assert kwargs["alpha"] == 0.5

    fig, ax = plotting.figure()
    try:
        ax.plot([1, 2], [0.1, 0.2], label=style.label, **kwargs)
    finally:
        matplotlib.pyplot.close(fig)


def test_unknown_executor_fails_loudly():
    """A typo must not silently produce an off-palette colour."""
    with pytest.raises(KeyError, match="Unknown executor"):
        plotting.style_for("znr_pointwise")
