"""
build_diagrams.py

Compile the paper's TikZ diagrams into ``paper/figures``.

Every ``paper/tex/diagrams/<name>.tex`` is a standalone document drawn by
hand, not from data: Fig. "the Draper--Grover circuit" (``fig_circuit``) and
Fig. "experimental pipeline" (``fig_pipeline``).  Each is compiled with
``pdflatex`` in a scratch directory, and ``<name>.pdf`` (plus a ``.png``
preview when ``pdftoppm`` is installed) lands in ``paper/figures``.

The sources omit the PDF creation date and trailer ID, so the same source
gives the same bytes and a rebuild shows in git only when a diagram changed.

Run::

    python paper/scripts/build_diagrams.py
    python paper/scripts/build_diagrams.py --only fig_circuit
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _common import FIGURES_DIR, TEX_DIR

#: Where the diagram sources live.
DIAGRAMS_DIR = TEX_DIR / "diagrams"

#: Seconds one ``pdflatex`` run may take before it is treated as hung
#: (MiKTeX can stop to ask about a missing package).
COMPILE_TIMEOUT_S = 180

#: Resolution of the ``.png`` previews.
PREVIEW_DPI = 200


def sources(directory: Path = DIAGRAMS_DIR, only: list[str] | None = None) -> list[Path]:
    """The ``.tex`` diagram sources, sorted; ``only`` restricts them by stem."""
    found = sorted(directory.glob("*.tex"))
    if only:
        missing = sorted(set(only) - {p.stem for p in found})
        if missing:
            raise FileNotFoundError(f"no diagram source for {missing} in {directory}")
        found = [p for p in found if p.stem in only]
    return found


def compile_diagram(source: Path, out_dir: Path) -> list[Path]:
    """
    Compile one standalone source and copy its PDF (and preview) to ``out_dir``.

    Raises
    ------
    RuntimeError
        If ``pdflatex`` fails; the message carries the first LaTeX error.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={scratch}", source.name],
            cwd=source.parent, capture_output=True, text=True, errors="replace",
            timeout=COMPILE_TIMEOUT_S, check=False,
        )
        pdf = Path(scratch) / f"{source.stem}.pdf"
        if result.returncode != 0 or not pdf.exists():
            log = (Path(scratch) / f"{source.stem}.log")
            text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            error = next((line for line in text.splitlines() if line.startswith("!")),
                         result.stdout[-500:])
            raise RuntimeError(f"{source.name}: pdflatex failed: {error}")
        written = [out_dir / pdf.name]
        shutil.copyfile(pdf, written[0])
        if shutil.which("pdftoppm"):
            preview = out_dir / source.stem
            subprocess.run(["pdftoppm", "-r", str(PREVIEW_DPI), "-png", "-singlefile",
                            str(written[0]), str(preview)], check=True,
                           timeout=COMPILE_TIMEOUT_S)
            written.append(preview.with_suffix(".png"))
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--only", nargs="+", default=None, help="diagram stems to build")
    parser.add_argument("--source-dir", type=Path, default=DIAGRAMS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Build every diagram.

    Returns
    -------
    int
        ``0`` on success, ``1`` without ``pdflatex`` or on a LaTeX error.
    """
    args = parse_args(argv)
    if shutil.which("pdflatex") is None:
        print("pdflatex not found; install a TeX distribution (MiKTeX, TeX Live)")
        return 1
    try:
        for source in sources(args.source_dir, args.only):
            for path in compile_diagram(source, args.figures_dir):
                print(f"wrote {path}")
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
