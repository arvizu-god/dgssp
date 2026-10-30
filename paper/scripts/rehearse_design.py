"""
rehearse_design.py

Rehearse a design job end to end, offline, in a scratch folder: plan, dry run,
pre-register, then treat the dry run as the hardware -- collect it, build the
day's alpha cache, aggregate, score.  Nothing contacts the service and nothing
under ``paper/`` is written: every output goes to ``--root``.

The device is a saved calibration (``--snapshot``, e.g. the 2026-10-28 plan's),
rebuilt as the device it describes, so the design's pins (past jobs' circuits
on ibm_fez) resolve on it.  The past jobs' manifests, jobs files and usage
the plan reads are copied into ``--root`` first; their circuits and
calibrations are read where they are.

The "hardware" here is the noise model at alpha = 1 (the dry run), so the
verdicts it prints exercise the code, not the device: alpha_hat will "not
hold", and the odd grid may resolve.  The alpha cache of the day is built on
the plan calibration (a rehearsal has no execution calibration).

Run::

    python paper/scripts/rehearse_design.py --root $env:TEMP\\dgssp_job3_rehearsal
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

import hardware_batch as hb  # noqa: E402
import hw_aggregate  # noqa: E402
import noise_model_check  # noqa: E402
import preregister  # noqa: E402
import score_preregistration  # noqa: E402
from _common import HARDWARE_DIR  # noqa: E402

#: The plan date the rehearsal files carry (never a real job's).
REHEARSAL_DATE = "2099-11-01"


def stage_sources(design: dict, hardware_dir: Path, source_dir: Path) -> list[str]:
    """Copy what ``--plan --design`` reads from past jobs into the rehearsal's folder."""
    dates = {hb.parse_pin(r["source"])["date"] for r in design["records"]}
    dates |= set((design.get("estimate") or {}).get("jobs", []))
    copied = []
    for date in sorted(dates):
        for name in (f"manifest_{date}.json", f"jobs_{date}.json"):
            if (source_dir / name).exists():
                shutil.copy2(source_dir / name, hardware_dir / name)
                copied.append(name)
    shutil.copy2(source_dir / "usage.json", hardware_dir / "usage.json")
    return [*copied, "usage.json"]


def stage_hardware(paths: hb.Paths) -> Path:
    """The dry run's result as the day's "hardware": a jobs file and its raw result."""
    dry = json.loads(paths.jobs_dryrun.read_text(encoding="utf-8"))
    entry = dry["jobs"][0]
    raw = hb.resolve(entry["raw_result"])
    target = paths.hardware_out_dir / "raw" / f"result_{entry['job_id']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw, target)
    hb.write_json(paths.jobs, {**dry, "source": "rehearsal: the dry run stands in for hardware"})
    return target


def run(argv: list[str], label: str, fn) -> None:  # type: ignore[no-untyped-def]
    """One step, printed; a non-zero exit stops the rehearsal."""
    print(f"\n=== {label}: {' '.join(argv)}", flush=True)
    code = fn(argv)
    if code != 0:
        raise SystemExit(f"{label} exited {code}")


def main(argv: list[str] | None = None) -> int:
    """Plan, dry run, pre-register, collect, alpha cache, aggregate, score."""
    parser = argparse.ArgumentParser(description="Rehearse a design job offline.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=HARDWARE_DIR / "design_job3.json")
    parser.add_argument("--snapshot", default="paper/hardware/calibration_2026-10-28.json")
    parser.add_argument("--remaining-seconds", type=float, required=True,
                        help="QPU seconds left (the plan's cap is half of it).")
    parser.add_argument("--date", default=REHEARSAL_DATE)
    parser.add_argument("--max-threads", type=int, default=None)
    args = parser.parse_args(argv)

    if args.root.exists() and any(args.root.iterdir()):
        raise SystemExit(f"{args.root} is not empty; give a new folder.")
    hw, data, tex = args.root / "hw", args.root / "data", args.root / "tex"
    for folder in (hw, data, tex):
        folder.mkdir(parents=True, exist_ok=True)
    design = hb.load_design(args.design)
    print("staged:", ", ".join(stage_sources(design, hw, HARDWARE_DIR)))
    common = ["--hardware-dir", str(hw), "--data-dir", str(data)]
    threads = ["--max-threads", str(args.max_threads)] if args.max_threads else []
    paths = hb.Paths(args.date, hw, data)

    run(["--plan", "--design", str(args.design), "--offline-backend",
         f"{hb.SNAPSHOT_PREFIX}{args.snapshot}", "--remaining-seconds",
         repr(args.remaining_seconds), "--date", args.date, *common], "plan", hb.main)
    run(["--dry-run", "--date", args.date, *common, *threads], "dry run", hb.main)
    run(["--date", args.date, *common, *threads], "preregister", preregister.main)
    run(["--collect", "--jobs", str(paths.jobs_dryrun), "--out-dir",
         str(paths.hardware_out_dir), *common], "collect (dry run as hardware)", hb.main)
    stage_hardware(paths)
    prereg = json.loads(hb.preregistration_path(paths).read_text(encoding="utf-8"))
    run(["--fit-alpha", "--date", args.date, "--label",
         score_preregistration.DEFAULT_ALPHA_LABEL, "--calibration", str(paths.calibration),
         "--alphas", *[f"{a:g}" for a in prereg["alpha_r_grid"]], *common, *threads],
        "day alpha cache (plan calibration)", noise_model_check.main)
    run(["--date", args.date, *common, "--tex-dir", str(tex)], "aggregate", hw_aggregate.main)
    run(["--date", args.date, *common], "score", score_preregistration.main)
    print(f"\nrehearsal done in {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
