"""
verify_campaign.py

Is ``paper/data`` one complete campaign: every record present, all from one
commit and one environment, every count where it should be?  Run it after
the results come back from the cloud, before anything is aggregated.

Per stage of :data:`sim_campaign.CAMPAIGN`, against ``instances/family.json``
and ``instances/sweep.json``:

* every expected record is present, and nothing else is (no record of an
  instance the family does not hold, no ``_001`` duplicate from two runs
  writing at once, no ``.json.tmp`` left by a killed write, nothing
  unreadable);
* each record says what it is: its name, and the ``stage`` / ``grid`` /
  ``noise`` / ``seed_simulator`` in its campaign block, match the directory
  it sits in;
* ``schema_version == 2`` on every record (the sweep's payloads are not
  records and carry ``schema: dgssp/paper/sim-sweep/1`` instead);
* shot totals: every counts dictionary sums to the stage's configured shots
  (10 000 ideal and noisy, 8 192 per scale for ZNE), with one counts entry
  per nominal scale;
* the Mitiq baseline is present wherever it should be -- every ZNE record and
  every ZNE sweep row (``--no-mitiq`` for a run that left it out on purpose).

Across the whole dataset:

* one git commit, ``dirty == False`` everywhere, one set of pinned package
  versions and one Python version -- a dataset spanning two commits or two
  environments is not one dataset;
* those package versions are the ones in ``requirements-lock.txt``
  (``--no-lock-check`` to skip), and, with ``--commit``, the commit is the one
  the VM was launched at.

Prints a one-screen summary, with each stage's size on disk and mean record
size, and exits 1 on any failure.

Run::

    python paper/scripts/verify_campaign.py
    python paper/scripts/verify_campaign.py --commit 6de4bb3
    python paper/scripts/verify_campaign.py --data-dir C:/tmp/slice --limit 20 \\
        --stages ideal noisy zne-planned
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

import sim_campaign  # noqa: E402
from _common import DATA_DIR, INSTANCES_DIR, LOCK_FILE, pin_drift, read_pins  # noqa: E402
from run_campaign import LOG_NAME, PlannedStage, build_plan  # noqa: E402

#: Record schema the campaign writes, and the sweep payloads' tag.
RECORD_SCHEMA_VERSION = 2
SWEEP_SCHEMA = "dgssp/paper/sim-sweep/1"

#: Examples printed per failed check.
SHOWN = 4


def configured_shots(stage: str, executor: str | None = None) -> int:
    """
    Shots per circuit (per scale, for ZNE) a stage is configured with.

    Parameters
    ----------
    stage:
        ``ideal``, ``noisy``, ``zne`` or ``sweep``.
    executor:
        For the sweep: ``ideal``, ``noisy`` or ``zne``.

    Returns
    -------
    int
        From the constants in ``sim_campaign.py``.
    """
    kind = executor if stage == "sweep" else stage
    return {
        "ideal": sim_campaign.IDEAL_SHOTS,
        "noisy": sim_campaign.NOISY_SHOTS,
        "zne": sim_campaign.ZNE_SHOTS,
    }[str(kind)]


def _sum_problems(counts: dict[str, Any], shots: int, where: str) -> list[str]:
    """One line per counts dictionary that does not sum to ``shots``."""
    return [
        f"{where}counts[{key}] sums to {sum(values.values())}, not {shots}"
        for key, values in counts.items()
        if sum(values.values()) != shots
    ]


def check_record(
    payload: dict[str, Any],
    name: str,
    stage: PlannedStage,
    *,
    noise: str,
    include_mitiq: bool,
    sweep_iterations: dict[str, list[int]],
) -> list[str]:
    """
    Everything wrong with one record, as ``"check: detail"`` lines.

    Parameters
    ----------
    payload:
        The loaded JSON.
    name:
        Its file stem.
    stage:
        The stage whose directory it sits in.
    noise:
        The campaign's noise tag.
    include_mitiq:
        Whether the Mitiq baseline must be present.
    sweep_iterations:
        ``sweep.json``'s iteration list per sweep instance.

    Returns
    -------
    list[str]
        Empty when the record is what its directory says it is.
    """
    spec = stage.spec
    problems: list[str] = []
    campaign = (payload.get("metrics") or {}).get("campaign") or payload.get("campaign") or {}

    if campaign.get("seed_simulator") != sim_campaign.SEED_SIMULATOR:
        problems.append(f"config: seed_simulator {campaign.get('seed_simulator')}")

    if spec.stage == "sweep":
        if payload.get("schema") != SWEEP_SCHEMA:
            problems.append(f"schema: {payload.get('schema')!r}, not {SWEEP_SCHEMA!r}")
        executor = str(payload.get("executor"))
        instance = str((payload.get("instance") or {}).get("name"))
        if f"{instance}_{executor}" != name or campaign.get("sweep_executor") != executor:
            problems.append(f"identity: holds {instance}/{executor}")
            return problems
        rows = payload.get("rows") or []
        if [row.get("iterations") for row in rows] != sweep_iterations.get(instance):
            problems.append(f"identity: iterations {[r.get('iterations') for r in rows]}")
        shots = configured_shots("sweep", executor)
        for row in rows:
            where = f"r={row.get('iterations')} "
            if row.get("shots") != shots:
                problems.append(f"shots: {where}row says {row.get('shots')}, not {shots}")
            problems += [f"shots: {p}" for p in _sum_problems(row.get("counts") or {}, shots, where)]
            if executor == "zne" and include_mitiq and not row.get("mitiq"):
                problems.append(f"mitiq: {where}no Mitiq block")
        return problems

    if payload.get("schema_version") != RECORD_SCHEMA_VERSION:
        problems.append(f"schema: schema_version {payload.get('schema_version')!r}")
    if (payload.get("instance") or {}).get("name") != name:
        problems.append(f"identity: holds {(payload.get('instance') or {}).get('name')!r}")
    expected_noise = "none" if spec.stage == "ideal" else noise
    for key, want in (("stage", spec.stage), ("grid", spec.grid), ("noise", expected_noise)):
        if campaign.get(key) != want:
            problems.append(f"identity: campaign {key} {campaign.get(key)!r}, not {want!r}")

    shots = configured_shots(spec.stage)
    counts = payload.get("counts") or {}
    if payload.get("shots") != shots or campaign.get("shots") != shots:
        problems.append(f"shots: record says {payload.get('shots')}, not {shots}")
    if spec.stage == "zne":
        scales = campaign.get("nominal_scales") or []
        if len(counts) != len(scales):
            problems.append(f"shots: {len(counts)} counts entries for {len(scales)} scales")
        if include_mitiq and not (campaign.get("include_mitiq") and payload.get("mitiq")):
            problems.append("mitiq: no Mitiq block")
    elif list(counts) != ["1"]:
        problems.append(f"shots: counts keyed {list(counts)}, not ['1']")
    problems += [f"shots: {p}" for p in _sum_problems(counts, shots, "")]
    return problems


@dataclass
class StageReport:
    """What one stage directory holds, and what is wrong with it."""

    stage: PlannedStage
    present: int = 0
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    problems: dict[str, list[str]] = field(default_factory=dict)
    size_bytes: int = 0
    record_bytes: int = 0

    @property
    def bad(self) -> int:
        """Records with at least one failed content check."""
        return len({line.split(":", 1)[0] for lines in self.problems.values() for line in lines})


def scan_stage(
    stage: PlannedStage,
    *,
    noise: str,
    include_mitiq: bool,
    sweep_iterations: dict[str, list[int]],
    identities: list[dict[str, Any]],
) -> StageReport:
    """
    Read every file in one stage directory and check it.

    Parameters
    ----------
    stage:
        The stage.
    noise, include_mitiq, sweep_iterations:
        As for :func:`check_record`.
    identities:
        Appended to: one ``{"commit", "dirty", "packages", "python", "host",
        "timestamp"}`` per readable file, for the dataset-wide checks.

    Returns
    -------
    StageReport
        The directory's contents and problems.
    """
    report = StageReport(stage)
    directory = stage.directory
    expected = set(stage.expected)
    if not directory.is_dir():
        report.missing = sorted(expected)
        return report

    report.size_bytes = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    report.partial = sorted(p.name for p in directory.glob("*.tmp"))
    on_disk = {p.stem: p for p in sorted(directory.glob("*.json"))}
    report.missing = sorted(expected - set(on_disk))
    for stem in sorted(set(on_disk) - expected):
        base, _, suffix = stem.rpartition("_")
        if base in expected and len(suffix) == 3 and suffix.isdigit():
            report.duplicates.append(stem)
        else:
            report.extra.append(stem)

    for stem in sorted(expected & set(on_disk)):
        path = on_disk[stem]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            report.unreadable.append(f"{stem} ({type(exc).__name__})")
            continue
        report.present += 1
        report.record_bytes += path.stat().st_size
        provenance = payload.get("provenance") or {}
        identities.append(
            {
                "commit": (provenance.get("git") or {}).get("commit"),
                "dirty": (provenance.get("git") or {}).get("dirty"),
                "packages": json.dumps(provenance.get("packages"), sort_keys=True),
                "python": (provenance.get("python") or {}).get("version"),
                "host": provenance.get("host"),
                "timestamp": provenance.get("timestamp_utc"),
                "where": f"{directory.name}/{stem}",
            }
        )
        for line in check_record(
            payload,
            stem,
            stage,
            noise=noise,
            include_mitiq=include_mitiq,
            sweep_iterations=sweep_iterations,
        ):
            check, _, detail = line.partition(": ")
            report.problems.setdefault(check, []).append(f"{stem}: {detail}")
    return report


def dataset_problems(
    identities: list[dict[str, Any]],
    *,
    lock: Path | None,
    commit: str | None,
) -> tuple[list[str], list[str]]:
    """
    The one-commit, one-environment checks, over every record read.

    Parameters
    ----------
    identities:
        From :func:`scan_stage`.
    lock:
        ``requirements-lock.txt`` to compare the packages against, or
        ``None`` to skip.
    commit:
        The commit (or a prefix of it) the dataset must come from, if known.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(problems, description)``: failures, and the lines describing the
        dataset that is there.
    """
    problems: list[str] = []
    lines: list[str] = []
    total = len(identities)
    if not total:
        return ["dataset: no readable records"], lines

    def spread(key: str) -> Counter[Any]:
        return Counter(identity[key] for identity in identities)

    commits = spread("commit")
    if len(commits) > 1:
        problems.append(
            f"commit: {len(commits)} commits -- "
            + ", ".join(f"{str(c)[:7]} x{n}" for c, n in commits.most_common())
        )
    (top_commit, _), *_ = commits.most_common()
    if commit and not str(top_commit).startswith(commit):
        problems.append(f"commit: dataset is {str(top_commit)[:12]}, expected {commit}")
    lines.append(f"commit    {top_commit} ({commits[top_commit]}/{total} records)")

    dirty = [i["where"] for i in identities if i["dirty"] is not False]
    if dirty:
        problems.append(f"dirty: {len(dirty)} record(s) not from a clean tree, e.g. {dirty[:SHOWN]}")
    lines.append(f"dirty     {len(dirty)} record(s)")

    packages = spread("packages")
    if len(packages) > 1:
        problems.append(f"packages: {len(packages)} different package sets in one dataset")
    (top_packages, _), *_ = packages.most_common()
    versions = json.loads(top_packages) or {}
    lines.append(
        "packages  " + ", ".join(f"{k} {v}" for k, v in versions.items())
        + f" ({packages[top_packages]}/{total})"
    )
    if lock is not None:
        drift = pin_drift(read_pins(lock), versions)
        if drift:
            problems.append(f"packages: differ from {lock.name}: {drift[:SHOWN]}")
        else:
            lines.append(f"          = {lock.name}")

    pythons = spread("python")
    if len(pythons) > 1:
        problems.append(f"python: {dict(pythons)}")
    lines.append(f"python    {', '.join(f'{v} x{n}' for v, n in pythons.most_common())}")
    hosts = spread("host")
    lines.append(f"hosts     {', '.join(f'{h} x{n}' for h, n in hosts.most_common(3))}")
    stamps = sorted(str(i["timestamp"]) for i in identities if i["timestamp"])
    if stamps:
        lines.append(f"written   {stamps[0]} .. {stamps[-1]}")
    return problems, lines


def print_table(reports: Sequence[StageReport]) -> None:
    """The per-stage summary: counts, problems, size."""
    print(f"  {'stage':<12} {'directory':<22} {'expected':>8} {'present':>8} {'missing':>8} "
          f"{'extra':>6} {'dup':>4} {'bad':>4} {'size':>9} {'KB/rec':>7}")
    totals = Counter[str]()
    for r in reports:
        row = {
            "expected": len(r.stage.expected),
            "present": r.present,
            "missing": len(r.missing),
            "extra": len(r.extra) + len(r.partial) + len(r.unreadable),
            "dup": len(r.duplicates),
            "bad": r.bad,
            "size": r.size_bytes,
            "record_bytes": r.record_bytes,
        }
        totals.update(row)
        mean = r.record_bytes / r.present / 1e3 if r.present else 0.0
        print(f"  {r.stage.key:<12} {r.stage.directory.name:<22} {row['expected']:>8} "
              f"{row['present']:>8} {row['missing']:>8} {row['extra']:>6} {row['dup']:>4} "
              f"{row['bad']:>4} {row['size'] / 1e6:>7.1f}MB {mean:>7.1f}")
    mean = totals["record_bytes"] / totals["present"] / 1e3 if totals["present"] else 0.0
    print(f"  {'total':<12} {'':<22} {totals['expected']:>8} {totals['present']:>8} "
          f"{totals['missing']:>8} {totals['extra']:>6} {totals['dup']:>4} {totals['bad']:>4} "
          f"{totals['size'] / 1e6:>7.1f}MB {mean:>7.1f}")


def stage_failures(report: StageReport) -> list[str]:
    """Every failure in one stage, one line each, with a few examples."""
    name = report.stage.directory.name
    out = []
    for label, items in (
        ("missing", report.missing),
        ("not in the family", report.extra),
        ("duplicate (two runs wrote at once)", report.duplicates),
        ("partial write (.tmp)", report.partial),
        ("unreadable", report.unreadable),
    ):
        if items:
            out.append(f"{name}: {len(items)} {label}: {items[:SHOWN]}")
    for check, items in sorted(report.problems.items()):
        out.append(f"{name}: {len(items)} failed {check}: {items[:SHOWN]}")
    return out


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
    keys = [stage.key for stage in sim_campaign.CAMPAIGN]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--instances-dir", type=Path, default=INSTANCES_DIR)
    parser.add_argument("--noise", choices=("fake", "live"), default="fake")
    parser.add_argument("--stages", nargs="+", choices=keys, default=keys)
    parser.add_argument("--limit", type=int, default=None,
                        help="The --limit the campaign was run with (a slice).")
    parser.add_argument("--commit", default=None,
                        help="The commit (or a prefix) the dataset must come from.")
    parser.add_argument("--no-mitiq", action="store_true",
                        help="The run left the Mitiq baseline out on purpose.")
    parser.add_argument("--no-lock-check", action="store_true",
                        help="Do not compare the packages with requirements-lock.txt.")
    parser.add_argument("--lock", type=Path, default=LOCK_FILE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Verify a campaign's data directory.

    Returns
    -------
    int
        ``0`` when every check passes, else ``1``.
    """
    args = parse_args(argv)
    plan = build_plan(
        args.stages,
        data_dir=args.data_dir,
        noise=args.noise,
        limit=args.limit,
        instances_dir=args.instances_dir,
    )
    _family, _subset, sweep = sim_campaign.load_inputs(args.instances_dir)
    sweep_iterations = {row["name"]: [int(r) for r in row["iterations"]] for row in sweep["instances"]}

    print(f"verify_campaign: {args.data_dir}  noise={args.noise}  stages={len(plan)}  "
          f"limit={args.limit or 'none'}  mitiq={'off' if args.no_mitiq else 'required'}\n")
    identities: list[dict[str, Any]] = []
    reports = [
        scan_stage(
            stage,
            noise=args.noise,
            include_mitiq=not args.no_mitiq,
            sweep_iterations=sweep_iterations,
            identities=identities,
        )
        for stage in plan
    ]
    print_table(reports)

    problems, description = dataset_problems(
        identities,
        lock=None if args.no_lock_check else args.lock,
        commit=args.commit,
    )
    print("\ndataset:")
    for line in description:
        print(f"  {line}")

    log_path = args.data_dir / LOG_NAME
    if log_path.exists():
        runs = json.loads(log_path.read_text(encoding="utf-8")).get("runs", [])
        wall = sum(s.get("wall_s") or 0.0 for run in runs for s in run.get("stages", []))
        last = runs[-1] if runs else {}
        print(f"  log       {len(runs)} run(s), last {last.get('status')!r}, "
              f"{wall / 60:.1f} min of stage wall clock")

    failures = [line for report in reports for line in stage_failures(report)] + problems
    print()
    for line in failures:
        print(f"  FAIL  {line}")
    if failures:
        print(f"\nFAIL: {len(failures)} problem(s).")
        return 1
    print("PASS: one complete campaign, one commit, one environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
