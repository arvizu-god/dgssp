"""
report_bounds_diff.py

Step 5, part 2: every derived number that moved when ``REPORT_BOUNDS`` became
``"clip"``, old -> new.  Reads the derived files as they are now and as they
were at ``--base`` (``git show <base>:<path>``); reads no record and runs no
fit.

Compared:

* ``paper/data/sim_per_instance.csv`` (key: name, grid), ``sim_cells.csv``
  (key: family, n, M-class, grid, series), ``sim_models.csv``,
  ``sim_pathology.csv``, ``hw_records.csv``, ``hw_records_2026-10-28.csv``
  (key: record), cell by cell;
* ``sim_summary.json`` (without ``cells``, which the CSV covers),
  ``hw_summary.json`` and ``hw_summary_2026-10-28.json``, leaf by leaf;
* ``paper/tex/*.tex`` and ``paper/notes/sim_results.md`` (generated), line by
  line;
* ``paper/figures/*.pdf``: which files changed, byte for byte;
* the resolved / identifiable counts, which must not move.

Ignored: ``provenance`` blocks and the generated-at footer (they move on every
run), and the ``bias_bias_*`` columns of ``sim_per_instance.csv``, which
the base wrote by mistake with hash-order-dependent values (removed in this
step; listed under ``removed_columns``).

Output: ``paper/data/report_bounds_diff.json`` (``save_record`` never
overwrites).

Run::

    python paper/scripts/report_bounds_diff.py --base 0e90e9e
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import DATA_DIR, REPO_ROOT, record_provenance, save_record  # noqa: E402

SCHEMA = "report_bounds_diff/1"

#: The commit before Step 5 (``REPORT_BOUNDS`` did not exist: "flag").
DEFAULT_BASE = "0e90e9e"

#: CSV files and the columns that key a row.
CSV_KEYS: dict[str, tuple[str, ...]] = {
    "paper/data/sim_per_instance.csv": ("name", "grid"),
    "paper/data/sim_cells.csv": ("family", "n_items", "m_class", "grid", "series"),
    "paper/data/sim_models.csv": ("grid", "method"),
    "paper/data/sim_pathology.csv": ("name", "grid"),
    "paper/data/hw_records.csv": ("key",),
    "paper/data/hw_records_2026-10-28.csv": ("key",),
}
JSON_FILES: tuple[str, ...] = (
    "paper/data/sim_summary.json",
    "paper/data/hw_summary.json",
    "paper/data/hw_summary_2026-10-28.json",
)
TEXT_GLOBS: tuple[str, ...] = ("paper/tex/*.tex", "paper/notes/sim_results.md")
FIGURE_GLOB = "paper/figures/*.pdf"

#: Written by the base by mistake: which ``bias_bias_*`` columns exist, and
#: their values, followed the hash order of a set.
DROPPED_PREFIX = "bias_bias_"

#: Two numbers are equal when they agree to this (CSV round trip).
REL_TOL = 1e-12


def _git_bytes(base: str, path: str) -> bytes | None:
    """``git show base:path``, or ``None`` when the file did not exist."""
    result = subprocess.run(
        ["git", "show", f"{base}:{path}"], cwd=REPO_ROOT, capture_output=True, check=False
    )
    return result.stdout if result.returncode == 0 else None


def _number(text: Any) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value


def _same(old: Any, new: Any) -> bool:
    a, b = _number(old), _number(new)
    if a is not None and b is not None:
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b or abs(a - b) <= REL_TOL * max(abs(a), abs(b))
    return old == new


def diff_csv(base: str, path: str, keys: Sequence[str]) -> dict[str, Any]:
    """Cell-by-cell changes of one CSV, keyed by ``keys``."""
    old_bytes = _git_bytes(base, path)
    new_path = REPO_ROOT / path
    if old_bytes is None or not new_path.exists():
        return {"missing": True}
    old = list(csv.DictReader(io.StringIO(old_bytes.decode("utf-8"))))
    new = list(csv.DictReader(new_path.open(encoding="utf-8")))

    def key(row: Mapping[str, str]) -> str:
        return " | ".join(row[k] for k in keys)

    old_by, new_by = {key(r): r for r in old}, {key(r): r for r in new}
    old_cols = set(old[0]) if old else set()
    new_cols = set(new[0]) if new else set()
    changes = []
    for k in sorted(set(old_by) & set(new_by)):
        for col in sorted(c for c in old_cols & new_cols if not c.startswith(DROPPED_PREFIX)):
            if not _same(old_by[k][col], new_by[k][col]):
                changes.append({"key": k, "column": col, "old": old_by[k][col],
                                "new": new_by[k][col]})
    return {
        "rows_old": len(old),
        "rows_new": len(new),
        "rows_only_old": sorted(set(old_by) - set(new_by)),
        "rows_only_new": sorted(set(new_by) - set(old_by)),
        "added_columns": sorted(new_cols - old_cols),
        "removed_columns": sorted(old_cols - new_cols),
        "n_changes": len(changes),
        "changed_columns": sorted({c["column"] for c in changes}),
        "changes": changes,
    }


def _leaves(node: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("provenance", "cells"):
                continue
            yield from _leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, node


def diff_json(base: str, path: str) -> dict[str, Any]:
    """Leaf-by-leaf changes of one JSON (``provenance`` and ``cells`` skipped)."""
    old_bytes = _git_bytes(base, path)
    new_path = REPO_ROOT / path
    if old_bytes is None or not new_path.exists():
        return {"missing": True}
    old = dict(_leaves(json.loads(old_bytes)))
    new = dict(_leaves(json.loads(new_path.read_text(encoding="utf-8"))))
    changes = [
        {"path": p, "old": old[p], "new": new[p]}
        for p in sorted(set(old) & set(new))
        if not _same(old[p], new[p])
    ]
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "n_changes": len(changes),
        "changes": changes,
    }


def _content_lines(text: str) -> list[str]:
    """Lines without the generated-at footer of ``sim_results.md``."""
    lines = text.splitlines()
    out = []
    skip_next = False
    for line in lines:
        if skip_next:
            skip_next = False
            continue
        if line.startswith("Generated ") and line.rstrip().endswith("from commit"):
            skip_next = True
            continue
        out.append(line)
    return out


def diff_text(base: str, path: Path) -> dict[str, Any] | None:
    """Changed lines of one text file, paired in order; ``None`` when unchanged."""
    rel = path.relative_to(REPO_ROOT).as_posix()
    old_bytes = _git_bytes(base, rel)
    old = _content_lines(old_bytes.decode("utf-8")) if old_bytes is not None else []
    new = _content_lines(path.read_text(encoding="utf-8"))
    if old == new:
        return None
    import difflib

    pairs = []
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            pairs.append({"old": old[i1:i2], "new": new[j1:j2]})
    return {"file": rel, "hunks": pairs}


def figures(base: str) -> dict[str, list[str]]:
    """PDFs whose bytes changed."""
    changed, same = [], []
    for path in sorted(REPO_ROOT.glob(FIGURE_GLOB)):
        rel = path.relative_to(REPO_ROOT).as_posix()
        (changed if _git_bytes(base, rel) != path.read_bytes() else same).append(rel)
    return {"changed": changed, "unchanged": same}


def flag_counts(base: str) -> dict[str, Any]:
    """Resolved / identifiable / reported-source counts, old and new, per grid and job."""

    def counts(rows: Sequence[Mapping[str, str]], group: str, keys: Sequence[str]) -> dict:
        out: dict[str, dict[str, int]] = {}
        for row in rows:
            g = row.get(group, "") if group else "all"
            block = out.setdefault(g, {k: 0 for k in keys})
            for k in keys:
                if row.get(k) in ("True", "extrapolation", "resolved"):
                    block[k] += 1
        return out

    result: dict[str, Any] = {}
    specs = {
        "paper/data/sim_per_instance.csv": ("grid", ("resolved", "identifiable",
                                                     "reported_source")),
        "paper/data/hw_records.csv": ("", ("resolved", "identifiable", "reported_source",
                                           "zne_status")),
        "paper/data/hw_records_2026-10-28.csv": ("", ("resolved", "identifiable",
                                                      "reported_source", "zne_status")),
    }
    for path, (group, keys) in specs.items():
        old_bytes = _git_bytes(base, path)
        old = list(csv.DictReader(io.StringIO(old_bytes.decode("utf-8")))) if old_bytes else []
        new = list(csv.DictReader((REPO_ROOT / path).open(encoding="utf-8")))
        o, n = counts(old, group, keys), counts(new, group, keys)
        result[path] = {"old": o, "new": n, "identical": o == n}
    result["all_identical"] = all(v["identical"] for v in result.values())
    return result


def main(argv: list[str] | None = None) -> int:
    """Diff every derived file against ``--base``; write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "report_bounds_diff.json")
    args = parser.parse_args(argv)

    texts = [
        d for pattern in TEXT_GLOBS for p in sorted(REPO_ROOT.glob(pattern))
        if (d := diff_text(args.base, p)) is not None
    ]
    payload = {
        "schema": SCHEMA,
        "base": args.base,
        "flag_counts": flag_counts(args.base),
        "csv": {path: diff_csv(args.base, path, keys) for path, keys in CSV_KEYS.items()},
        "json": {path: diff_json(args.base, path) for path in JSON_FILES},
        "text": texts,
        "figures": figures(args.base),
        "provenance": record_provenance(),
    }
    path = save_record(args.out, payload)
    print("flag counts identical:", payload["flag_counts"]["all_identical"])
    for name, d in payload["csv"].items():
        print(f"{name}: {d.get('n_changes')} cells in {d.get('changed_columns')}; "
              f"+cols {d.get('added_columns')} -cols {d.get('removed_columns')}")
    for name, d in payload["json"].items():
        print(f"{name}: {d.get('n_changes')} leaves; +{len(d.get('added', []))} "
              f"-{len(d.get('removed', []))}")
    for t in texts:
        print(f"{t['file']}: {len(t['hunks'])} hunks")
    print("figures changed:", payload["figures"]["changed"])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
