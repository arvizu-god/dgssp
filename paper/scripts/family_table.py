"""
family_table.py

WP3: the appendix table describing the frozen instance family.

Prints one LaTeX ``tabular`` row per ``(family, n, M-class)`` cell: how many
instances the cell holds, the mean sum-register width, the mean ``r_opt`` and
the mean transpiled two-qubit depth.

Where each column comes from
----------------------------
``count``, ``n_sum`` and ``r_opt``
    ``paper/instances/family.json``, written by ``build_family.py``.  These are
    properties of the instance and its encoding, so every cell has them.

``2Q depth``
    ``paper/instances/hardware_subset.json``, written by
    ``select_hardware_subset.py``: the two-qubit depth of the circuit built at
    that instance's own ``r_opt`` and transpiled on the primary device's fake
    twin at optimization level 3.  This is device- and transpiler-dependent, so
    it belongs to the selection run and not to the family file, and the table
    names the backend it came from.  A cell whose instances were not part of
    that run prints ``--`` rather than a mean over nothing.

The signed encoding is broken out by ``g``, the number of negative items, so the
appendix can show directly what WP2 measured: at fixed ``n``, family S has the
same ``n_sum`` distribution as family P (``range_len`` is invariant under sign
flips) and a comparable two-qubit depth (the signed encoding costs single-qubit
phases only).

Usage
-----
::

    python paper/scripts/family_table.py
    python paper/scripts/family_table.py --out paper/tex/family_table.tex
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import INSTANCES_DIR  # noqa: E402

#: Order M-classes appear in, matching ``build_family.M_CLASSES``.
M_CLASS_ORDER: tuple[str, ...] = ("0", "1", "2", "3+")


def cell_key(record: dict[str, Any]) -> tuple[str, int, int, int]:
    """
    Grouping key for one instance: family, item count, negatives, M-class.

    Parameters
    ----------
    record:
        An instance record from ``family.json``.

    Returns
    -------
    tuple
        Sortable key; the M-class is replaced by its index in
        :data:`M_CLASS_ORDER` so ``"3+"`` sorts after ``"2"``.
    """
    return (
        record["family"],
        record["n_items"],
        record["n_negative"],
        M_CLASS_ORDER.index(record["m_class"]),
    )


def build_rows(
    family: dict[str, Any], depths: dict[str, int]
) -> list[dict[str, Any]]:
    """
    Aggregate the family into one row per cell.

    Parameters
    ----------
    family:
        The loaded ``family.json`` payload.
    depths:
        Mapping instance name -> transpiled two-qubit depth, from
        ``hardware_subset.json``.  Names absent from it are simply not averaged.

    Returns
    -------
    list[dict]
        One row per cell, in table order.
    """
    groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for record in family["instances"]:
        groups.setdefault(cell_key(record), []).append(record)

    rows: list[dict[str, Any]] = []
    for key in sorted(groups):
        members = groups[key]
        measured = [depths[r["name"]] for r in members if r["name"] in depths]
        rows.append(
            {
                "family": key[0],
                "n_items": key[1],
                "n_negative": key[2],
                "m_class": M_CLASS_ORDER[key[3]],
                "count": len(members),
                "mean_n_sum": statistics.fmean(r["n_sum"] for r in members),
                "mean_r_opt": statistics.fmean(r["r_opt"] for r in members),
                "n_transpiled": len(measured),
                "mean_two_q_depth": statistics.fmean(measured) if measured else None,
            }
        )

    # Cells the generator could not fill at all leave no instances behind, so
    # they would silently vanish from the table.  Reinstate them as zero-count
    # rows from the generator's own cell report: an empty cell is a result.
    present = {(r["family"], r["n_items"], r["n_negative"], r["m_class"]) for r in rows}
    for cell in family["cells"]:
        key = (cell["family"], cell["n_items"], cell["n_negative"], cell["m_class"])
        if key in present:
            continue
        rows.append(
            {
                "family": cell["family"],
                "n_items": cell["n_items"],
                "n_negative": cell["n_negative"],
                "m_class": cell["m_class"],
                "count": 0,
                "mean_n_sum": None,
                "mean_r_opt": None,
                "n_transpiled": 0,
                "mean_two_q_depth": None,
            }
        )

    rows.sort(key=lambda r: (r["family"], r["n_items"], r["n_negative"],
                             M_CLASS_ORDER.index(r["m_class"])))
    return rows


def render_latex(
    rows: list[dict[str, Any]], *, backend: str | None, family: dict[str, Any]
) -> str:
    """
    Render the rows as a LaTeX ``tabular``.

    Parameters
    ----------
    rows:
        Output of :func:`build_rows`.
    backend:
        Backend the two-qubit depths were measured on, for the caption.
    family:
        The family payload, for the seed and hash in the caption.

    Returns
    -------
    str
        A complete ``table`` environment, ready to ``\\input``.
    """
    def fmt(value: float | None, places: int) -> str:
        return "--" if value is None else f"{value:.{places}f}"

    lines = [
        r"% Generated by paper/scripts/family_table.py -- do not edit by hand.",
        r"% Requires \usepackage{booktabs} for \toprule, \midrule, \bottomrule.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{llrrrrrr}",
        r"    \toprule",
        r"    Family & $g$ & $n$ & $M$-class & count & $\overline{b}$ & "
        r"$\overline{r_{\mathrm{opt}}}$ & $\overline{D_{2Q}}$ \\",
        r"    \midrule",
    ]

    previous_family = None
    for row in rows:
        if previous_family is not None and row["family"] != previous_family:
            lines.append(r"    \midrule")
        previous_family = row["family"]
        m_class = row["m_class"].replace("+", r"$+$")
        depth = fmt(row["mean_two_q_depth"], 0)
        if row["mean_two_q_depth"] is not None and row["n_transpiled"] < row["count"]:
            depth += f"$^{{{row['n_transpiled']}}}$"
        lines.extend(
            [
                f"    {row['family']} & {row['n_negative']} & {row['n_items']} & "
                f"{m_class} & {row['count']} & {fmt(row['mean_n_sum'], 2)} & "
                f"{fmt(row['mean_r_opt'], 2)} & {depth} \\\\"
            ]
        )

    total = sum(r["count"] for r in rows)
    caption = (
        f"The frozen instance family: {total} instances, master seed "
        f"{family['master_seed']}, payload hash "
        f"\\texttt{{{str(family.get('payload_sha256'))[:12]}}}. "
        r"$g$ is the number of negative items, $b$ the sum-register width "
        r"including the guard bit, and $D_{2Q}$ the two-qubit depth after "
        r"transpilation at optimization level 3"
    )
    # Backend names carry underscores, and \texttt is not verbatim -- an
    # unescaped underscore is a subscript and breaks the build.
    safe_backend = backend.replace("_", r"\_") if backend else None
    caption += f" on \\texttt{{{safe_backend}}}" if safe_backend else ""
    caption += (
        r", with each circuit built at its own $r_{\mathrm{opt}}$. "
        r"A count of 0 marks a cell that exhaustive enumeration shows to be empty."
    )
    # Only explain the superscript when one is actually printed; a caption that
    # describes marks the table does not contain is noise a referee will query.
    if any(
        r["mean_two_q_depth"] is not None and r["n_transpiled"] < r["count"] for r in rows
    ):
        caption += (
            r" A superscript on $\overline{D_{2Q}}$ gives the number of instances "
            r"in the cell that were transpiled, where that is fewer than the cell count."
        )

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            f"  \\caption{{{caption}}}",
            r"  \label{tab:instance-family}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Command-line interface.

    Parameters
    ----------
    argv:
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description="Print the WP3 appendix table.")
    parser.add_argument(
        "--family", type=Path, default=INSTANCES_DIR / "family.json"
    )
    parser.add_argument(
        "--subset", type=Path, default=INSTANCES_DIR / "hardware_subset.json"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="write the LaTeX here as well as stdout"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Load the two files, aggregate and print.

    Parameters
    ----------
    argv:
        Argument list.

    Returns
    -------
    int
        0 on success, 1 when the family file is missing.
    """
    args = parse_args(argv)
    if not args.family.exists():
        print(
            f"{args.family} not found; run `python paper/scripts/build_family.py` first.",
            file=sys.stderr,
        )
        return 1

    family = json.loads(args.family.read_text(encoding="utf-8"))

    depths: dict[str, int] = {}
    backend: str | None = None
    if args.subset.exists():
        subset = json.loads(args.subset.read_text(encoding="utf-8"))
        backend = subset.get("backend")
        depths = {r["name"]: r["two_q_depth"] for r in subset.get("candidates", [])}
        stored = subset.get("family_payload_sha256")
        if stored and stored != family.get("payload_sha256"):
            print(
                "WARNING: hardware_subset.json was built from a different family "
                f"({stored} != {family.get('payload_sha256')}); depths may not match "
                "the instances in this table.",
                file=sys.stderr,
            )
    else:
        print(
            f"note: {args.subset.name} not found; the depth column will be empty.",
            file=sys.stderr,
        )

    rows = build_rows(family, depths)
    latex = render_latex(rows, backend=backend, family=family)
    print(latex)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(latex + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
