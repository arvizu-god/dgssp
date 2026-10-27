"""
Tests for ``paper/scripts/build_diagrams.py``.

The diagrams are drawn by hand, so what can be protected is that every source
compiles and that a rebuild of an unchanged source gives the same bytes (the
sources omit the creation date and trailer ID).  Skipped without ``pdflatex``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "paper" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

build_diagrams = pytest.importorskip("build_diagrams")
needs_latex = pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no pdflatex")


def test_the_two_diagrams_are_there():
    """Fig. "circuit" and Fig. "pipeline"; asking for an unknown one is an error."""
    assert [p.stem for p in build_diagrams.sources()] == ["fig_circuit", "fig_pipeline"]
    with pytest.raises(FileNotFoundError, match="fig_nope"):
        build_diagrams.sources(only=["fig_nope"])


@needs_latex
def test_every_diagram_compiles_to_the_same_bytes_twice(tmp_path):
    """Same source, same PDF: no date or random ID inside."""
    for source in build_diagrams.sources():
        first = build_diagrams.compile_diagram(source, tmp_path / "a")[0]
        second = build_diagrams.compile_diagram(source, tmp_path / "b")[0]
        assert first.read_bytes() == second.read_bytes(), source.name


@needs_latex
def test_a_latex_error_is_reported_not_swallowed(tmp_path):
    """A broken source fails with the LaTeX error, and writes nothing."""
    broken = tmp_path / "fig_broken.tex"
    broken.write_text("\\documentclass{standalone}\\begin{document}\\nosuchmacro\\end{document}\n",
                      encoding="utf-8")
    with pytest.raises(RuntimeError, match="Undefined control sequence"):
        build_diagrams.compile_diagram(broken, tmp_path / "out")
    assert not (tmp_path / "out" / "fig_broken.pdf").exists()
