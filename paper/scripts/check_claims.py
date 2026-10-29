"""
check_claims.py

Recompute every quantitative claim of the WP7 review from the committed data,
and compare it with what the notes now say.

The claims are the ones ``paper/notes/hw_review_wp7.md`` lists in §1 (rows
1-10) and §2 (items A-K and F2), plus the n = 4 wording of
``status_after_wp6.md``.  For each one this script

1. recomputes the value from the data files (nothing is simulated, fitted or
   sent anywhere): ``paper/data/hw_summary.json`` (job of record),
   ``hw_summary_<SECOND_JOB>.json`` (second job), the simulation records
   through :func:`aggregate.build_rows` (the same rows ``sim_per_instance.csv``
   holds), ``paper/data/sim_sweep_fake/`` and the git log;
2. reads the value *as written* in the note, with a regular expression scoped
   to one section, and
3. calls the claim OK when the written text equals the recomputed value at the
   written precision, MISMATCH otherwise.

Some claims are that a phrase is *gone* ("upper bound", null #9, "committed
before"); those are OK when the phrase is absent from the section.

Writes ``paper/data/claims_check.json`` (one entry per claim, the SHA-256 of
every data file read, provenance) and prints one row per claim::

    claim id | file:section | value as written | recomputed | OK / MISMATCH

Claim ids are the review's: ``R<row>`` for §1, the item letter for §2
(``A.<record>`` per record of item A).  Exit status 1 if any claim mismatches.

Run::

    python paper/scripts/check_claims.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import DATA_DIR, REPO_ROOT, record_provenance  # noqa: E402
from aggregate import DEFAULT_GRIDS, build_rows, discover_noise, headline, with_signal  # noqa: E402
from dgssp.mitigation.zne import RESOLVABILITY_K  # noqa: E402
from hardware_batch import rel, sha256_file, write_json  # noqa: E402
from hw_aggregate import JOB_OF_RECORD  # noqa: E402

SCHEMA = "dgssp/paper/claims-check/1"

#: The second hardware job (gentle grid), by its folder date.
SECOND_JOB = "2026-10-28"

#: The commit that added the second job's data, whose time the notes quote.
DATA_COMMIT = "09b9c46"

#: The record that was at random on the second day (``hw_results.md`` §4).
FAILED_RECORD = "P-n3-M2-17"

#: "Kept more than 0.13 of their signal": the cut the cube-rule footnote uses.
KEPT_CUT = 0.13

#: A record's exponent ``ln(kept) / ln(S)`` is quoted only when its kept
#: signal is at least this many standard errors above zero.
EXPONENT_MIN_Z = 3.0

#: Shallow vs deep for the exponent ranges.  No job-1 record lies between
#: 233 and 350 two-qubit gates, so any cut in that gap gives the same groups.
DEPTH_SPLIT_2Q = 300

#: Sections of ``hw_results.md`` that hold claims (everything above the log).
BODY = "§1-§8"
LOG_HEADING = "## Corrections log"

MINUS = "\u2212"
NUM = rf"[{MINUS}+-]?\d+(?:\.\d+)?"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def fmt(x: float, digits: int = 3) -> str:
    """``x`` at ``digits`` decimals (the precision the note writes)."""
    return f"{x:.{digits}f}"


def signed(x: float, digits: int = 3) -> str:
    """``x`` at ``digits`` decimals with an explicit sign."""
    return f"{x:+.{digits}f}"


def normalise(text: str) -> str:
    """The note's typographic minus as ASCII, surrounding space dropped."""
    return text.replace(MINUS, "-").strip()


def hhmm(stamp: str) -> str:
    """``HH:MM`` of an ISO time, in UTC."""
    return utc(stamp).strftime("%H:%M")


def utc(stamp: str) -> datetime:
    """An ISO time (``Z`` or an offset) as an aware UTC datetime."""
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------


class Facts:
    """Every file the claims are recomputed from, loaded once."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.read: list[Path] = []
        self.job1 = self._json(data_dir / "hw_summary.json")
        self.job2 = self._json(data_dir / f"hw_summary_{SECOND_JOB}.json")

    def _json(self, path: Path) -> Any:
        self.read.append(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def job(self, n: int) -> Mapping[str, Any]:
        """Summary of job 1 (of record) or job 2."""
        return self.job1 if n == 1 else self.job2

    def records(self, n: int) -> dict[str, Mapping[str, Any]]:
        """``{key: record entry}`` of a job (drift repeats are not records)."""
        return {r["key"]: r for r in self.job(n)["records"]}

    def rec(self, n: int, key: str) -> Mapping[str, Any]:
        return self.records(n)[key]

    def surv(self, n: int, key: str) -> Mapping[str, Any]:
        return {r["key"]: r for r in self.job(n)["survival"]["records"]}[key]

    def pubs(self, n: int) -> list[Mapping[str, Any]]:
        return list(self.job(n)["model_vs_hardware"]["pubs"])

    def distinct_unfolded(self, n: int) -> list[Mapping[str, Any]]:
        """The ``lambda = 1`` PUB of every record with signal: one per circuit."""
        keys = self.records(n)
        return [p for p in self.pubs(n) if p["key"] in keys and p["scale"] == 1.0 and p["signal"]]

    def repeat_of(self, n: int) -> dict[str, str]:
        """``{drift key: the record it repeats}``."""
        return {d["key"]: d["repeat_of"] for d in self.job(n)["drift"]}

    @cached_property
    def sim_rows(self) -> list[dict[str, Any]]:
        """The per-instance simulation rows, built from the records as ``aggregate.py`` does."""
        return build_rows(self.data_dir, list(DEFAULT_GRIDS), discover_noise(self.data_dir))

    def sweep(self, instance: str, stage: str) -> dict[int, Mapping[str, Any]]:
        """``{iterations: row}`` of one ``sim_sweep_fake`` record."""
        doc = self._json(self.data_dir / "sim_sweep_fake" / f"{instance}_{stage}.json")
        return {int(r["iterations"]): r for r in doc["rows"]}

    @cached_property
    def data_commit_utc(self) -> datetime:
        out = subprocess.run(
            ["git", "show", "-s", "--format=%cI", DATA_COMMIT],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return utc(out.stdout.strip())


# ---------------------------------------------------------------------------
# Derived quantities (each used by one or more claims)
# ---------------------------------------------------------------------------


def sigma_p(rec: Mapping[str, Any], scale_index: int = 0) -> float:
    """Shot-noise standard error of ``P(solution)`` at one scale."""
    return float(rec["detail"]["hardware_sigma"][scale_index])


def kept_and_sigma(rec: Mapping[str, Any]) -> tuple[float, float]:
    """Signal kept at ``lambda = 1`` and its standard error."""
    span = abs(float(rec["p_ideal"]) - float(rec["p_uniform"]))
    return float(rec["kept_hardware"]), sigma_p(rec) / span


def exponent_and_se(facts: Facts, key: str) -> tuple[float, float]:
    """
    ``ln(kept) / ln(S)`` on job 1 and its delta-method standard error.

    ``S`` (the survival estimate at alpha = 1) has no sampling error, so the
    error is that of ``ln(kept)``: ``sigma_kept / kept``, divided by ``|ln S|``.
    """
    kept, sk = kept_and_sigma(facts.rec(1, key))
    s = facts.surv(1, key)
    return float(s["exponent"]), sk / (kept * abs(math.log(float(s["survival_alpha_1"]))))


def z_above_random(rec: Mapping[str, Any], scale_index: int = 0) -> float:
    """``(P - P_uniform) / sigma`` at one scale."""
    p = float(rec["detail"]["hardware"][scale_index])
    return (p - float(rec["p_uniform"])) / sigma_p(rec, scale_index)


def unresolved_with_signal(facts: Facts) -> list[tuple[str, Mapping[str, Any]]]:
    """
    ``(label, record)`` of every record with signal whose scalar fit is not resolved.

    Both jobs; a key unresolved on both days is labelled ``"<key> (job <n>)"``.
    """
    found = [(n, key, r) for n in (1, 2) for key, r in facts.records(n).items()
             if r["signal_direction"] != 0 and r["zne_status"] != "resolved"]
    keys = [key for _, key, _ in found]
    return [(f"{key} (job {n})" if keys.count(key) > 1 else key, r) for n, key, r in found]


def high_kept(facts: Facts) -> list[str]:
    """Job-1 records that kept more than :data:`KEPT_CUT` of their signal, in summary order."""
    return [k for k, r in facts.records(1).items()
            if r["kept_hardware"] is not None and r["kept_hardware"] > KEPT_CUT]


def next_below_cut(facts: Facts, count: int = 2) -> list[str]:
    """The ``count`` job-1 records with the most kept signal below :data:`KEPT_CUT`."""
    below = [(r["kept_hardware"], k) for k, r in facts.records(1).items()
             if r["kept_hardware"] is not None and 0 < r["kept_hardware"] <= KEPT_CUT]
    return [k for _, k in sorted(below, reverse=True)[:count]]


def exponent_groups(facts: Facts) -> tuple[list[str], list[str]]:
    """Job-1 records whose kept signal is >= :data:`EXPONENT_MIN_Z` sigma, shallow / deep."""
    keep = []
    for key, r in facts.records(1).items():
        if r["kept_hardware"] is None or facts.surv(1, key).get("exponent") is None:
            continue
        kept, sk = kept_and_sigma(r)
        if kept / sk >= EXPONENT_MIN_Z:
            keep.append(key)
    shallow = [k for k in keep if facts.rec(1, k)["two_q_count"] < DEPTH_SPLIT_2Q]
    deep = [k for k in keep if facts.rec(1, k)["two_q_count"] >= DEPTH_SPLIT_2Q]
    return shallow, deep


def exponent_range(facts: Facts, keys: Sequence[str]) -> list[str]:
    values = [facts.surv(1, k)["exponent"] for k in keys]
    return [fmt(min(values), 2), fmt(max(values), 2)]


def mitiq_delta(facts: Facts, n: int, key: str) -> str:
    r = facts.rec(n, key)
    return fmt(abs(float(r["mitiq"]) - float(r["zne_reported"])))


def alpha_hat_deviation(facts: Facts, key: str) -> float:
    """Job 2, ``lambda = 1``: hardware minus the model at alpha-hat."""
    (pub,) = [p for p in facts.distinct_unfolded(2) if p["key"] == key]
    return float(pub["dev_alpha"])


def rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values))


def threshold_interval(facts: Facts) -> tuple[str, int, int]:
    """
    Every 2Q threshold ``T`` ("identifiable iff 2Q < T") with the best agreement.

    Measured identifiable = two or more informative scales, as
    ``hw_aggregate.survival_block`` counts it.  Returns the interval of best
    thresholds as ``"(lo, hi]"`` and the agreement.
    """
    rows = facts.job1["survival"]["records"]
    counts = [int(r["two_q_count"]) for r in rows]
    scores = {}
    for t in range(min(counts), max(counts) + 2):
        scores[t] = sum(1 for r in rows
                        if (int(r["two_q_count"]) < t) == (r["n_informative_scales"] >= 2))
    best = max(scores.values())
    ts = sorted(t for t, s in scores.items() if s == best)
    if ts != list(range(ts[0], ts[-1] + 1)):
        raise ValueError(f"best thresholds are not one interval: {ts}")
    return f"({ts[0] - 1}, {ts[-1]}]", best, len(rows)


def recovery_median(rows: Sequence[dict[str, Any]], n_items: int, *, exact: bool) -> float:
    """
    Planned-grid median signal recovery at one ``n``, by :func:`aggregate.headline`.

    ``exact`` swaps the noiseless reference: the exact ``P_ideal`` instead of
    the ideal (sampled) run that ``aggregate.py`` divides by.
    """
    subset = [dict(r, ideal=r["p_ideal_exact"]) if exact else r
              for r in rows if r["n_items"] == n_items]
    return float(headline(subset, [], ["planned"])["per_grid"]["planned"]["signal_recovery_median"])


def strict_values(facts: Facts, *, reported: bool) -> list[float]:
    """
    Planned grid, instances with signal: the strict rule's value per instance.

    ``reported=False`` is the fit before ``REPORT_BOUNDS`` (the extrapolation
    as fitted where it is reported, else the lambda = 1 value);
    ``reported=True`` is the value the tables report.
    """
    planned = with_signal([r for r in facts.sim_rows if r["grid"] == "planned"])
    key = "zne_scalar_reported"
    return [
        r[key] if reported or r["reported_source"] != "extrapolation" else r["zne_scalar_value"]
        for r in planned if r[key] is not None
    ]


def strict_outside(facts: Facts, *, reported: bool = False) -> list[float]:
    """Planned grid, instances with signal: strict-rule values outside [0, 1]."""
    return [v for v in strict_values(facts, reported=reported) if not 0.0 <= v <= 1.0]


def strict_median_bias(facts: Facts) -> float:
    """Median of reported minus ideal (the ideal run), planned, with signal."""
    planned = with_signal([r for r in facts.sim_rows if r["grid"] == "planned"])
    return statistics.median(r["bias_zne_scalar_reported"] for r in planned
                             if r["bias_zne_scalar_reported"] is not None)


def clipped_interval(facts: Facts, job: int, key: str) -> list[str]:
    """A resolved hardware record's reported value and interval, at 3 decimals."""
    r = facts.rec(job, key)
    return [fmt(r["zne_reported"]), fmt(r["zne_reported_lo"]), fmt(r["zne_reported_hi"])]


def relaxed_outside(facts: Facts) -> list[float]:
    """The same for the relaxed rule: the extrapolation whenever it is identified."""
    planned = with_signal([r for r in facts.sim_rows if r["grid"] == "planned"])
    values = [r["zne_scalar_value"] if r["identifiable"] else r["zne_lambda1"] for r in planned]
    return [v for v in values if v is not None and not 0.0 <= v <= 1.0]


def tv_model_gap(facts: Facts, n: int, *, drop: str | None) -> tuple[float, float]:
    """Mean D_TV(hardware, model at alpha-hat) and its shot floor, optionally without one record."""
    repeats = facts.repeat_of(n)
    pubs = [p for p in facts.pubs(n) if p["tv_hardware_model_alpha"] is not None
            and (drop is None or drop not in (p["key"], repeats.get(p["key"])))]
    return (statistics.mean(p["tv_hardware_model_alpha"] for p in pubs),
            statistics.mean(p["tv_floor_alpha"] for p in pubs))


def day_to_day_z(facts: Facts, key: str) -> float:
    """``lambda = 1`` P(solution), one copy each day: (day 1 - day 2) / combined sigma."""
    r1, r2 = facts.rec(1, key), facts.rec(2, key)
    diff = float(r1["detail"]["hardware"][0]) - float(r2["detail"]["hardware"][0])
    return diff / math.hypot(sigma_p(r1), sigma_p(r2))


def false_informative(facts: Facts) -> tuple[int, float]:
    """Folded ``n >= 3`` points with signal on job 1, and how many pass a one-sided k-sigma test by chance."""
    from scipy.stats import norm

    records = facts.records(1)
    n = sum(1 for p in facts.pubs(1) if p["key"] in records and p["signal"]
            and p["scale"] != 1.0 and records[p["key"]]["n_items"] >= 3)
    return n, n * float(norm.sf(RESOLVABILITY_K))


# ---------------------------------------------------------------------------
# The claims
# ---------------------------------------------------------------------------

HW = "paper/notes/hw_results.md"
STATUS = "paper/notes/status_after_wp6.md"
REVIEW = "paper/notes/hw_review_wp7.md"
METRICS = "src/dgssp/metrics.py"
ZNE = "src/dgssp/mitigation/zne.py"
FAMILY = "paper/notes/instance_family.md"
RESOURCES = "paper/notes/resource_model.md"


@dataclass(frozen=True)
class Claim:
    """One quantitative claim: where it is written, and how to recompute it."""

    cid: str
    note: str
    section: str
    pattern: str
    compute: Callable[[Facts], Sequence[str]]
    source: str
    absent: bool = False


def _a(label: str, rec: Mapping[str, Any]) -> Claim:
    raw, clip = fmt(rec["tv_raw"]), fmt(rec["tv_pointwise_clip"])
    return Claim(
        f"A.{label.replace(' (job ', '@j').rstrip(')')}", HW, "§2",
        rf"(?<![\w-]){re.escape(label)} ({NUM}) → ({NUM})",
        lambda f: [raw, clip],
        "hw_summary*.json: records[].tv_raw, tv_pointwise_clip",
    )


def _h(key: str, source: str) -> Claim:
    return Claim(f"H.{key}", HW, "§1", rf"\| {re.escape(key)} \| \d+ \| ({NUM}) \|",
                 lambda f: [signed(alpha_hat_deviation(f, key))], source)


def claims(facts: Facts) -> list[Claim]:
    j1 = f"hw_summary.json ({JOB_OF_RECORD})"
    j2 = f"hw_summary_{SECOND_JOB}.json"
    sim = "sim records via aggregate.build_rows"
    out = [
        # --- §1 of the review -------------------------------------------------
        Claim("R1a", HW, "§2", r"\|Δ\| = (" + NUM + r") \(job 1, n = 2\)",
              lambda f: [mitiq_delta(f, 1, "P-n2-M1-11")],
              f"{j1}: records[P-n2-M1-11].mitiq - zne_reported"),
        Claim("R1b", HW, "§2",
              rf"({NUM}) \(n = 2\), ({NUM}) \(P-n3-M1-17\) and ({NUM}) \(P-n3-M1-17_r1\), job 2",
              lambda f: [mitiq_delta(f, 2, k) for k in ("P-n2-M1-11", "P-n3-M1-17", "P-n3-M1-17_r1")],
              f"{j2}: records[].mitiq - zne_reported"),
        Claim("R2", HW, "§2",
              rf"n = 3: raw ({NUM}) → clip ({NUM}), simplex ({NUM}) \((\d+)× and (\d+)×\)",
              lambda f: (lambda r: [fmt(r["tv_raw"]), fmt(r["tv_pointwise_clip"]),
                                    fmt(r["tv_pointwise_simplex"]),
                                    fmt(r["tv_pointwise_clip"] / r["tv_raw"], 0),
                                    fmt(r["tv_pointwise_simplex"] / r["tv_raw"], 0)])(
                  f.rec(1, "P-n3-M0-06")),
              f"{j1}: records[P-n3-M0-06].tv_raw, tv_pointwise_clip, tv_pointwise_simplex"),
        Claim("R3", HW, "§3", rf"largest gap ({NUM})",
              lambda f: [fmt(max(abs(f.surv(1, k)["survival_alpha_hat"] - f.rec(1, k)["kept_hardware"])
                                 for k in high_kept(f)))],
              f"{j1}: survival.records[].survival_alpha_hat vs records[].kept_hardware, kept > {KEPT_CUT}"),
        Claim("R3b", HW, "§3",
              rf"({NUM}) vs ({NUM}) \(n = 2\), ({NUM}) vs ({NUM}), ({NUM}) vs ({NUM}), ({NUM}) vs ({NUM}) \(n = 3\)",
              lambda f: [v for k in high_kept(f) for v in (fmt(f.surv(1, k)["survival_alpha_hat"]),
                                                          fmt(f.rec(1, k)["kept_hardware"]))],
              f"{j1}: survival.records[].survival_alpha_hat, records[].kept_hardware"),
        Claim("R4", HW, "§4", rf"({NUM})σ below day 1 \(one copy each day\)",
              lambda f: [fmt(day_to_day_z(f, FAILED_RECORD), 1)],
              f"{j1} and {j2}: records[{FAILED_RECORD}].detail.hardware[0], hardware_sigma[0]"),
        Claim("R5", HW, "§7", rf"median of ({NUM}), ({NUM}) and ({NUM}) at n = 2, 3 and 4",
              lambda f: [fmt(recovery_median(f.sim_rows, n, exact=False), 2) for n in (2, 3, 4)],
              f"{sim}: aggregate.headline signal_recovery_median, planned, per n"),
        Claim("R5x", HW, "§7", rf"n = 3 median is ({NUM}) with the exact P_ideal",
              lambda f: [fmt(recovery_median(f.sim_rows, 3, exact=True))],
              f"{sim}: as R5 with ideal := p_ideal_exact"),
        Claim("R5r", REVIEW, "§1", rf"\| 5 \|[^\n]*\*\*({NUM})\*\*",
              lambda f: [fmt(recovery_median(f.sim_rows, 3, exact=True))],
              f"{sim}: as R5 with ideal := p_ideal_exact (the review's value)"),
        Claim("R6", HW, "§7",
              rf"falls monotonically \(({NUM}), ({NUM}), ({NUM}), ({NUM}) for r = 1–4",
              lambda f: [fmt(f.sweep("P-n4-M1-10", "noisy")[r]["p_sol"]) for r in (1, 2, 3, 4)],
              "sim_sweep_fake/P-n4-M1-10_noisy.json: rows[iterations].p_sol"),
        Claim("R7a", HW, "top",
              r"`dbfeqr7gdj9s73bbgh6g`, ran (\S+) (\d\d:\d\d)–(\d\d:\d\d) UTC \(folder label `(\S+)`",
              lambda f: [utc(f.job1["config"]["running_utc"]).date().isoformat(),
                         hhmm(f.job1["config"]["running_utc"]),
                         hhmm(f.job1["config"]["finished_utc"]), f.job1["date"]],
              f"{j1}: config.running_utc, finished_utc, date"),
        Claim("R7b", HW, "top",
              r"`dbgkqq8te5oc738icnfg`, ran (\S+) (\d\d:\d\d)–(\d\d:\d\d) UTC \(folder label `(\S+)`",
              lambda f: [utc(f.job2["config"]["running_utc"]).date().isoformat(),
                         hhmm(f.job2["config"]["running_utc"]),
                         hhmm(f.job2["config"]["finished_utc"]), f.job2["date"]],
              f"{j2}: config.running_utc, finished_utc, date"),
        Claim("R8", HW, "§3", r"same instance \((\d+) 2Q on the odd grid, (\d+) 2Q on the gentle grid\)",
              lambda f: [str(f.rec(n, "P-n3-M1-17")["two_q_count"]) for n in (1, 2)],
              f"{j1}, {j2}: records[P-n3-M1-17].two_q_count"),
        Claim("R9a", HW, "§1", r"timestamped (\d+) min before the job started",
              lambda f: [fmt((utc(f.job2["config"]["running_utc"])
                              - utc(f.job2["model_alpha"]["made_utc"])).total_seconds() / 60, 0)],
              f"{j2}: config.running_utc - model_alpha.made_utc"),
        Claim("R9b", HW, "§1", r"prediction file (\d\d:\d\d:\d\d) UTC, job start (\d\d:\d\d:\d\d) UTC",
              lambda f: [utc(f.job2["model_alpha"]["made_utc"]).strftime("%H:%M:%S"),
                         utc(f.job2["config"]["running_utc"]).strftime("%H:%M:%S")],
              f"{j2}: model_alpha.made_utc, config.running_utc"),
        Claim("R9c", HW, "§1", rf"`{DATA_COMMIT}` is (\d\d:\d\d) UTC",
              lambda f: [f.data_commit_utc.strftime("%H:%M")],
              f"git show -s --format=%cI {DATA_COMMIT}"),
        Claim("R9d", HW, BODY, r"committed before", lambda f: [], "phrase removed", absent=True),
        Claim("R10a", HW, "§1",
              rf"({NUM}) on the (\d+) distinct unfolded circuits with signal of the first run, "
              rf"({NUM}) on the (\d+) of the second",
              lambda f: [x for n in (1, 2) for x in (
                  fmt(statistics.mean(p["dev_1"] for p in f.distinct_unfolded(n)), 2),
                  str(len(f.distinct_unfolded(n))))],
              f"{j1}, {j2}: model_vs_hardware.pubs[] at lambda = 1, records only, dev_1"),
        Claim("R10b", HW, "§1", r"(\d+) distinct circuits plus (\d+) drift copies",
              lambda f: [str(len(f.distinct_unfolded(1))),
                         str(sum(1 for p in f.pubs(1) if p["key"] in f.repeat_of(1)))],
              f"{j1}: model_vs_hardware.pubs[] at lambda = 1 with signal"),
        # --- §2 of the review -------------------------------------------------
        Claim("A", HW, "§2", r"lowered D_TV on (\d+) of (\d+) records and raised it on (\d+), up to ("
              + NUM + ")",
              lambda f: (lambda u: [str(sum(1 for _, r in u if r["tv_pointwise_clip"] < r["tv_raw"])),
                                    str(len(u)),
                                    str(sum(1 for _, r in u if r["tv_pointwise_clip"] > r["tv_raw"])),
                                    fmt(max(r["tv_pointwise_clip"] for _, r in u))])(
                  unresolved_with_signal(f)),
              "hw_summary*.json: records[] with signal, zne_status != resolved"),
        *[_a(label, r) for label, r in unresolved_with_signal(facts)],
        Claim("B", HW, "§2",
              r"strict rule's fit still lands outside \[0, 1\] on (\d+) of (\d+) planned-grid "
              rf"values, (\d+) of them above 1 \(max ({NUM})\)",
              lambda f: (lambda o: [str(len(o)),
                                    str(len(with_signal([r for r in f.sim_rows if r["grid"] == "planned"]))),
                                    str(sum(1 for v in o if v > 1)), fmt(max(o), 2)])(strict_outside(f)),
              f"{sim}: reported_source == extrapolation ? zne_scalar_value : zne_scalar_reported, "
              "planned, signal_direction != 0"),
        Claim("B3", HW, "§2", r"reported clipped to \[P_uniform, 1\]: (\d+) reported values outside",
              lambda f: [str(len(strict_outside(f, reported=True)))],
              f"{sim}: zne_scalar_reported (REPORT_BOUNDS = clip), planned, with signal"),
        Claim("B4", HW, "§7", rf"\| Strict rule \| ({NUM}) \|",
              lambda f: [fmt(strict_median_bias(f), 2)],
              f"{sim}: median bias_zne_scalar_reported, planned, with signal"),
        Claim("B5", HW, "§2",
              rf"\| P-n3-M1-17 \|[^\n]*\| ({NUM}) \[({NUM}), ({NUM})\], resolved \|",
              lambda f: clipped_interval(f, 2, "P-n3-M1-17"),
              f"{j2}: records[P-n3-M1-17].zne_reported, _lo, _hi"),
        Claim("B6", HW, "§6",
              rf"\| 13 \|[^\n]*\| ({NUM}) \[({NUM}), ({NUM})\] vs ({NUM}) \[({NUM}), ({NUM})\] \|",
              lambda f: [*clipped_interval(f, 1, "P-n2-M1-11"), *clipped_interval(f, 2, "P-n2-M1-11")],
              f"{j1} and {j2}: records[P-n2-M1-11].zne_reported, _lo, _hi"),
        Claim("B2", HW, "§7", rf"relaxed rule adds (\d+) more \(max ({NUM})\)",
              lambda f: [str(len(relaxed_outside(f)) - len(strict_outside(f))),
                         fmt(max(relaxed_outside(f)), 0)],
              f"{sim}: identifiable ? zne_scalar_value : zne_lambda1, planned, with signal"),
        Claim("C", HW, "§3", r"(\d+) of (\d+) at α = 1 and (\d+) of (\d+) at α̂",
              lambda f: [str(f.job2["survival"]["identifiable_agreement"][lab][k])
                         for lab in ("alpha_1", "alpha_hat") for k in ("agree", "of")],
              f"{j2}: survival.identifiable_agreement"),
        Claim("C2", HW, "§3", r"correctly predicted", lambda f: [], "clause removed", absent=True),
        Claim("D", HW, "§3", r"any 2Q threshold in (\(\d+, \d+\]) gives (\d+) of (\d+)",
              lambda f: (lambda t: [t[0], str(t[1]), str(t[2])])(threshold_interval(f)),
              f"{j1}: survival.records[].two_q_count, n_informative_scales"),
        Claim("D2", HW, "§3",
              r"pre-registered test, α = 1, failed: (\d+) of (\d+), limit placed at (\d+)–(\d+) 2Q "
              r"instead of (\d+)–(\d+)",
              lambda f: (lambda s: [str(s["identifiable_agreement"]["alpha_1"]["agree"]),
                                    str(s["identifiable_agreement"]["alpha_1"]["of"]),
                                    str(s["bands"]["alpha_1"]["lower"]), str(s["bands"]["alpha_1"]["upper"]),
                                    str(s["bands"]["measured"]["lower"]),
                                    str(s["bands"]["measured"]["upper"])])(f.job1["survival"]),
              f"{j1}: survival.identifiable_agreement.alpha_1, survival.bands"),
        Claim("D3", HW, "§3", r"places the odd-grid limit exactly", lambda f: [], "phrase removed",
              absent=True),
        Claim("E1", HW, BODY, r"upper bound", lambda f: [], "phrase removed", absent=True),
        Claim("E2", METRICS, "file", r"coherence-free upper bound", lambda f: [],
              "survival_estimate docstring", absent=True),
        Claim("E3", ZNE, "file", r"``survival`` is an \*upper bound\*", lambda f: [],
              "plan_scales docstring", absent=True),
        Claim("E4", FAMILY, "§2", r"upper\s+bound", lambda f: [],
              "survival model of the edge instance", absent=True),
        Claim("E5", RESOURCES, "file", r"upper\s+bound", lambda f: [],
              "survival table, cites the survival_estimate docstring", absent=True),
        Claim("F", HW, "§3", rf"exponents are ({NUM}), ({NUM}), ({NUM}) and ({NUM}) for the four records",
              lambda f: [fmt(f.surv(1, k)["exponent"], 2) for k in high_kept(f)],
              f"{j1}: survival.records[].exponent, kept > {KEPT_CUT}"),
        Claim("F.next", HW, "§3",
              rf"next two records \(kept ({NUM}) and ({NUM})\) give ({NUM}) and ({NUM})",
              lambda f: (lambda ks: [fmt(f.rec(1, k)["kept_hardware"]) for k in ks]
                         + [fmt(f.surv(1, k)["exponent"], 2) for k in ks])(next_below_cut(f)),
              f"{j1}: survival.records[].exponent, records[].kept_hardware"),
        Claim("F2.r2", HW, "§3",
              rf"P-n4-M1-10_r2 keeps ({NUM}) ± ({NUM}) \(({NUM})σ above random\) and gives an "
              rf"exponent of ({NUM}) ± ({NUM})",
              lambda f: (lambda ks, e: [fmt(ks[0]), fmt(ks[1]), fmt(ks[0] / ks[1], 0),
                                        fmt(e[0], 2), fmt(e[1], 2)])(
                  kept_and_sigma(f.rec(1, "P-n4-M1-10_r2")), exponent_and_se(f, "P-n4-M1-10_r2")),
              f"{j1}: records[P-n4-M1-10_r2].kept_hardware, hardware_sigma; survival exponent"),
        Claim("F2.r", HW, "§3",
              rf"P-n4-M1-10 at r_opt gives ({NUM}) ± ({NUM}), P-n4-M2-02 ({NUM}) ± ({NUM})",
              lambda f: [fmt(x, 2) for k in ("P-n4-M1-10", "P-n4-M2-02") for x in exponent_and_se(f, k)],
              f"{j1}: survival.records[].exponent; delta-method SE from hardware_sigma"),
        Claim("F2.range", HW, "§3",
              rf"up to (\d+) 2Q the exponent is ({NUM})–({NUM}); from (\d+) 2Q on it is ({NUM})–({NUM})",
              lambda f: (lambda g: [str(max(f.rec(1, k)["two_q_count"] for k in g[0])),
                                    *exponent_range(f, g[0]),
                                    str(min(f.rec(1, k)["two_q_count"] for k in g[1])),
                                    *exponent_range(f, g[1])])(exponent_groups(f)),
              f"{j1}: survival.records[].exponent, kept >= {EXPONENT_MIN_Z} sigma"),
        Claim("F2.res", HW, "§3",
              rf"P-n4-M1-10_r1 and _r2: ({NUM}) and ({NUM})",
              lambda f: [signed(p["dev_alpha"]) for k in ("P-n4-M1-10_r1", "P-n4-M1-10_r2")
                         for p in f.distinct_unfolded(1) if p["key"] == k],
              f"{j1}: model_vs_hardware.pubs[] at lambda = 1, dev_alpha"),
        Claim("G1", HW, "§1",
              rf"job 1 \(the fit day\) within ({NUM}) of the shot floor \(({NUM}) against ({NUM})\)",
              lambda f: (lambda g: [fmt(g[0] - g[1]), fmt(g[0]), fmt(g[1])])(tv_model_gap(f, 1, drop=None)),
              f"{j1}: model_vs_hardware.pubs[].tv_hardware_model_alpha, tv_floor_alpha"),
        Claim("G2", HW, "§1",
              rf"job 2 without {FAILED_RECORD}: ({NUM}) against a floor of ({NUM}) \(({NUM})×\)",
              lambda f: (lambda g: [fmt(g[0]), fmt(g[1]), fmt(g[0] / g[1], 1)])(
                  tv_model_gap(f, 2, drop=FAILED_RECORD)),
              f"{j2}: model_vs_hardware.pubs[] without {FAILED_RECORD} and its repeat"),
        *[_h(key, f"{j2}: model_vs_hardware.pubs[{key}, lambda = 1].dev_alpha")
          for key in facts.records(2)],
        Claim("H.rms", HW, "§1", rf"RMS ({NUM}) over the (\d+) records; ({NUM}) without",
              lambda f: (lambda d: [fmt(rms(list(d.values()))), str(len(d)),
                                    fmt(rms([v for k, v in d.items() if k != FAILED_RECORD]))])(
                  {k: alpha_hat_deviation(f, k) for k in f.records(2)}),
              f"{j2}: model_vs_hardware.pubs[] at lambda = 1, dev_alpha"),
        Claim("H.max", HW, "§1", rf"within ({NUM}) on every PUB",
              lambda f: [fmt(max(abs(p["dev_alpha"]) for p in f.pubs(2)
                                 if FAILED_RECORD not in (p["key"], f.repeat_of(2).get(p["key"], ""))))],
              f"{j2}: model_vs_hardware.pubs[].dev_alpha"),
        Claim("H.gone", HW, "§6", r"prediction unbiased", lambda f: [], "null #10 removed", absent=True),
        Claim("I", HW, "§6", r"Day 1 vs day 2", lambda f: [], "null #9 removed", absent=True),
        Claim("J", HW, "§2", rf"P-n3-M1-17 gives ({NUM}) against ({NUM}) \(z = ({NUM})\)",
              lambda f: (lambda r: [fmt(r["detail"]["hardware"][1]), fmt(r["p_uniform"]),
                                    fmt(z_above_random(r, 1), 1)])(f.rec(1, "P-n3-M1-17")),
              f"{j1}: records[P-n3-M1-17].detail.hardware[1]"),
        Claim("J2", HW, "§2",
              r"the odd grid resolves (\d+) % of n = 3 instances with signal, the planned grid (\d+) %",
              lambda f: (lambda g: [fmt(100 * g["odd"]["fraction_resolved_with_signal"], 0),
                                    fmt(100 * g["planned"]["fraction_resolved_with_signal"], 0)])(
                  headline(f.sim_rows, [], list(DEFAULT_GRIDS))["grid_comparison"]["3"]),
              f"{sim}: aggregate.headline grid_comparison.3"),
        Claim("K", HW, "§3",
              rf"λ = 5 point at z = ({NUM}) with χ²_red = ({NUM})",
              lambda f: (lambda r: [fmt(z_above_random(r, 2), 1), fmt(r["zne_chi2_red"], 1)])(
                  f.rec(1, "P-n4-M2-02")),
              f"{j1}: records[P-n4-M2-02].detail, zne_chi2_red"),
        Claim("K2", HW, "§3",
              rf"on the (\d+) folded n ≥ 3 points about ({NUM}) false",
              lambda f: (lambda x: [str(x[0]), fmt(x[1], 1)])(false_informative(f)),
              f"{j1}: model_vs_hardware.pubs[] folded, n >= 3, signal; norm.sf(RESOLVABILITY_K)"),
        Claim("K3", HW, "§3", rf"resolved with χ²_red = ({NUM})",
              lambda f: [fmt(f.sweep("P-n3-M1-17", "zne")[1]["zne"]["chi2_red"], 1)],
              "sim_sweep_fake/P-n3-M1-17_zne.json: rows[iterations=1].zne.chi2_red"),
        # --- status_after_wp6.md ------------------------------------------------
        Claim("S1", STATUS, "Results", r"indistinguishable\s+from\s+random", lambda f: [],
              "n = 4 wording replaced", absent=True),
        Claim("S2", STATUS, "Results",
              rf"detectable\s+but\s+useless\s+for\s+search\s+at\s+n\s+=\s+4\s+\(({NUM})σ,\s+({NUM})σ\s+and\s+"
              rf"({NUM})σ\s+above\s+random",
              lambda f: [fmt(z_above_random(f.rec(1, k)), 1)
                         for k in ("P-n4-M1-10", "P-n4-M2-02", "S-n4-g1-M2-00")],
              f"{j1}: records[].detail.hardware[0], hardware_sigma[0]"),
    ]
    return out


# ---------------------------------------------------------------------------
# Reading the notes
# ---------------------------------------------------------------------------


def section_text(text: str, section: str) -> str:
    """
    The part of a note one claim is read from.

    ``"file"`` is the whole file, ``"top"`` everything before the first
    ``## `` heading, :data:`BODY` everything before the corrections log,
    ``"§<n>"`` the ``## <n>.`` section, any other name the ``## <name>``
    section.
    """
    if section == "file":
        return text
    if section == "top":
        return text.split("\n## ", 1)[0]
    if section == BODY:
        return text.split(LOG_HEADING, 1)[0]
    head = rf"## {section[1:]}\." if section.startswith("§") else rf"## {re.escape(section)}\b"
    match = re.search(rf"^{head}.*?(?=^## |\Z)", text, flags=re.M | re.S)
    if match is None:
        raise KeyError(f"no section {section!r}")
    return match.group(0)


def check(claim: Claim, facts: Facts, texts: Mapping[str, str]) -> dict[str, Any]:
    """
    One row of the report.

    Every occurrence of the pattern in the section is read; occurrences that
    disagree with each other are written as ``a | b`` and so mismatch.
    """
    body = section_text(texts[claim.note], claim.section)
    matches = list(re.finditer(claim.pattern, body))
    if claim.absent:
        written = "absent" if not matches else matches[0].group(0)
        recomputed = "absent"
    else:
        found = list(dict.fromkeys(" / ".join(normalise(g) for g in m.groups()) for m in matches))
        written = " | ".join(found) if found else "(not found)"
        recomputed = " / ".join(claim.compute(facts))
    return {
        "id": claim.cid,
        "note": claim.note,
        "section": claim.section,
        "written": written,
        "recomputed": recomputed,
        "status": "OK" if written == recomputed else "MISMATCH",
        "source": claim.source,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "claims_check.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    facts = Facts(args.data_dir)
    todo = claims(facts)
    texts = {c.note: (REPO_ROOT / c.note).read_text(encoding="utf-8") for c in todo}
    rows = [check(c, facts, texts) for c in todo]

    width = max(len(r["written"]) for r in rows)
    for r in rows:
        where = f"{Path(r['note']).name}:{r['section']}"
        print(f"{r['id']:<16} | {where:<26} | {r['written']:<{width}} | "
              f"{r['recomputed']:<{width}} | {r['status']}")
    bad = sum(1 for r in rows if r["status"] != "OK")
    print(f"{len(rows)} claims, {len(rows) - bad} OK, {bad} MISMATCH")

    sim_dirs = sorted(p for p in args.data_dir.glob("sim_*") if p.is_dir())
    write_json(args.out, {
        "schema": SCHEMA,
        "jobs": {"job_of_record": JOB_OF_RECORD, "second_job": SECOND_JOB},
        "n_claims": len(rows),
        "n_mismatch": bad,
        "claims": rows,
        "inputs": {rel(p): sha256_file(p) for p in sorted(set(facts.read))},
        "input_dirs": [rel(p) for p in sim_dirs],
        "notes": {note: sha256_file(REPO_ROOT / note) for note in sorted(texts)},
        "provenance": record_provenance(),
    })
    print(f"wrote {rel(args.out)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
