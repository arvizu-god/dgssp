"""
Tests for a job planned from a design file (step 6a): ``hardware_batch.py
--plan --design``, ``preregister.py``, ``score_preregistration.py``, the
multi-grid fits of ``hw_aggregate.py`` and the submit gates of a design job.

Everything runs offline on a bundled fake device: a small source job is
planned first, and a small design pins its circuits, as the real design pins
jobs 1 and 2.  The service and ``git`` are replaced where ``--submit`` would
reach them; each submit gate has a test that fails if the gate is removed
(the submission would then go through and return 0).
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "paper" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("scipy")
pytest.importorskip("qiskit_ibm_runtime")
hb = pytest.importorskip("hardware_batch")
import hw_aggregate  # noqa: E402
import preregister  # noqa: E402
import score_preregistration  # noqa: E402

SOURCE_DATE = "2099-05-01"
DATE = "2099-05-02"
SHOTS = 256
HEAD = "0123456789abcdef0123456789abcdef01234567"


def mini_design(source_date: str) -> dict:
    """The real design's grids and criteria, on three records pinned to a fake job."""
    design = json.loads((REPO_ROOT / "paper" / "hardware" / "design_job3.json")
                        .read_text(encoding="utf-8"))
    design["device"] = "fake_marrakesh"
    design["shots"] = SHOTS
    design.pop("estimate")
    design["records"] = [
        {"key": "P-n2-M1-11_L1", "source": f"P-n2-M1-11:L1={source_date}",
         "grids": ["gentle", "rule"], "drift": True, "circuit_set": "fitted"},
        {"key": "P-n3-M1-17_L1", "source": f"P-n3-M1-17:L1={source_date}",
         "grids": ["odd", "gentle"], "drift": False, "circuit_set": "fitted"},
        {"key": "P-n3-M1-17_S", "source": f"P-n3-M1-17:S={source_date}",
         "grids": [], "drift": False, "circuit_set": "out_of_sample"},
    ]
    design["trim_order"] = [
        {"drop": "folds", "record": "P-n3-M1-17_L1", "removes": ["Q3"]},
        {"drop": "record", "record": "P-n2-M1-11_L1", "removes": ["Q4c"]},
    ]
    return design


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    """A source plan, a design pinned to it, its plan, dry run and pre-registration."""
    root = tmp_path_factory.mktemp("design")
    hw, data = root / "hw", root / "data"
    common = ["--hardware-dir", str(hw), "--data-dir", str(data)]
    source = hb.main(["--plan", "--offline-backend", "fake_marrakesh", "--only", "P-n2-M1-11",
                      "P-n3-M1-17", "--no-sweep", "--seed-max", "2", "--shots", str(SHOTS),
                      "--date", SOURCE_DATE, *common])
    design_path = root / "design.json"
    design_path.write_text(json.dumps(mini_design(SOURCE_DATE)), encoding="utf-8")
    plan = hb.main(["--plan", "--design", str(design_path), "--offline-backend",
                    "fake_marrakesh", "--remaining-seconds", "100000", "--date", DATE, *common])
    dry = hb.main(["--dry-run", "--date", DATE, *common])
    prereg = preregister.main(["--date", DATE, *common])
    paths = hb.Paths(DATE, hw, data)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    return {"root": root, "common": common, "paths": paths, "manifest": manifest,
            "design_path": design_path, "status": (source, plan, dry, prereg)}


def test_design_job_plans_dry_runs_and_preregisters(job):
    """Source plan, design plan, dry run and pre-registration all exit 0."""
    assert job["status"] == (0, 0, 0, 0)


def test_design_records_run_the_union_of_their_grids(job):
    """lambda = 1 shared; each record's scales are its grids' union; a no-grid record is single."""
    manifest = job["manifest"]
    rows = {r["key"]: r for r in manifest["records"]}
    p2 = rows["P-n2-M1-11_L1"]
    assert set(p2["grids"]) == {"gentle", "rule"}
    assert p2["nominal_scales"] == sorted({1.0, *p2["grids"]["gentle"], *p2["grids"]["rule"]})
    p3 = rows["P-n3-M1-17_L1"]
    assert p3["nominal_scales"] == [1.0, 1.25, 1.5, 3.0, 5.0]
    single = rows["P-n3-M1-17_S"]
    assert single["kind"] == "single" and single["nominal_scales"] == [1.0]
    assert rows["P-n2-M1-11_L1_drift"]["kind"] == "drift"
    assert manifest["totals"]["circuits"] == len(p2["nominal_scales"]) + 5 + 1 + 1


def test_rule_grid_is_planned_from_the_simulated_signal(job):
    """The rule grid is plan_scales_spacing on the simulated kept signal, beside optimal_step."""
    from dgssp.mitigation.zne import plan_scales_spacing
    from sim_campaign import PLAN_CANDIDATES, PLAN_K

    rule = job["manifest"]["design"]["rule_grids"]["P-n2-M1-11_L1"]
    again = plan_scales_spacing(rule["survival"], rule["p_ideal"] - rule["p_uniform"], SHOTS,
                                PLAN_CANDIDATES, PLAN_K, p_uniform=rule["p_uniform"],
                                n_points=rule["n_points"])
    assert rule["grid"] == again
    assert rule["step"] == pytest.approx(rule["grid"][1] - 1.0)
    assert rule["optimal_step"] > 0 and rule["alpha"] == pytest.approx(
        job["manifest"]["design"]["alphas"]["all"]["value"])


def test_pinned_circuits_keep_their_source_two_q_counts(job):
    """Every pinned lambda = 1 circuit equals its source job's: 2Q count and qubits."""
    check = job["manifest"]["design"]["two_q_check"]
    assert check and all(c["equal"] for c in check)
    assert job["manifest"]["design"]["two_q_equal"]


def test_trim_order_applies_when_over_the_cap(job, tmp_path):
    """Over the cap, the design's trim steps apply in order and are recorded."""
    hw = tmp_path / "hw"
    shutil.copytree(job["paths"].hardware_dir, hw)
    status = hb.main(["--plan", "--force", "--design", str(job["design_path"]),
                      "--offline-backend", "fake_marrakesh", "--remaining-seconds", "1",
                      "--date", "2099-05-09", "--hardware-dir", str(hw),
                      "--data-dir", str(tmp_path / "data")])
    manifest = json.loads((hw / "manifest_2099-05-09.json").read_text(encoding="utf-8"))
    assert status == 1  # still over the cap after every trim
    trimmed = manifest["design"]["trimmed"]
    assert [(t["drop"], t["record"]) for t in trimmed] == [
        ("folds", "P-n3-M1-17_L1"), ("record", "P-n2-M1-11_L1")]
    keys = {r["key"] for r in manifest["records"]}
    assert "P-n2-M1-11_L1" not in keys and "P-n2-M1-11_L1_drift" not in keys
    p3 = next(r for r in manifest["records"] if r["key"] == "P-n3-M1-17_L1")
    assert p3["nominal_scales"] == [1.0] and p3["grids"] == {}


def test_a_design_refuses_the_flags_it_replaces(job, capsys):
    """--design names records and drift itself; --pin-layout / --drift beside it are refused."""
    status = hb.main(["--plan", "--design", str(job["design_path"]), "--drift", "X",
                      "--offline-backend", "fake_marrakesh", "--date", "2099-05-10",
                      *job["common"]])
    assert status == 1 and "--drift" in capsys.readouterr().out


def test_preregistration_copies_the_criteria_and_hashes(job):
    """The pre-registration carries the design's criteria verbatim and the manifest's sha256."""
    paths = job["paths"]
    doc = json.loads(hb.preregistration_path(paths).read_text(encoding="utf-8"))
    design = json.loads(job["design_path"].read_text(encoding="utf-8"))
    assert doc["criteria"] == design["criteria"]
    assert doc["manifest_sha256"] == hb.sha256_file(paths.manifest)
    assert doc["report_bounds"] == "clip" and doc["resolved_rule"] == "v2"
    fits = doc["predictions"]["fits"]
    assert set(fits) == {"P-n2-M1-11_L1", "P-n3-M1-17_L1"}
    for grids in fits.values():
        for entry in grids.values():
            assert sum(entry["status_probability"].values()) == pytest.approx(1.0)
            if entry["se_pred"] is None:
                assert entry["se_note"].startswith("formula not applicable")
            else:
                assert entry["relative_se"] <= 0.2
            # each band end under the same cut as SE_pred
            for se, end in zip(entry["se_band"], ("band_lo", "band_hi"), strict=True):
                rel = entry[end]["relative_se"]
                assert (se is not None) == (rel is not None and rel <= 0.2)
    assert hb.preregistration_path(paths).with_suffix(".md").exists()
    # never replaced without --force
    assert preregister.main(["--date", DATE, *job["common"]]) == 1


def test_grid_fit_of_all_scales_reproduces_the_record(job):
    """A grid equal to every scale the record ran gives back its stored zne_fit."""
    record = json.loads((job["paths"].dryrun_dir / "P-n3-M1-17_L1.json")
                        .read_text(encoding="utf-8"))
    fit = hw_aggregate.grid_fit(record, record["zne_fit"]["nominal_scales"])
    stored = record["zne_fit"]
    assert fit["value"] == stored["value"]
    assert fit["ci"] == stored["ci"]
    assert fit["observed"] == stored["observed"]


@pytest.fixture(scope="module")
def scored(job):
    """The dry run collected as the job's records, then scored (no alpha cache: Q6 pending)."""
    paths = job["paths"]
    collect = hb.main(["--collect", "--jobs", str(paths.jobs_dryrun), "--out-dir",
                       str(paths.hardware_out_dir), *job["common"]])
    status = score_preregistration.main(["--date", DATE, *job["common"]])
    card = json.loads((paths.data_dir / f"scorecard_{DATE}.json").read_text(encoding="utf-8"))
    return {"status": (collect, status), "card": card}


def test_scorecard_answers_every_question(scored):
    """Every question has a verdict; Q6 waits for the day's alpha cache and says how to build it."""
    assert scored["status"] == (0, 0)
    questions = scored["card"]["questions"]
    assert list(questions) == [f"Q{i}" for i in range(1, 10)]
    assert questions["Q6"]["verdict"] == "pending"
    assert "--fit-alpha" in questions["Q6"]["how"]
    assert questions["Q3"]["verdict"] in ("confirmed", "not confirmed")
    # Q2 names every set, the primary one first
    q2 = questions["Q2"]
    assert q2["verdict"].startswith(f"{q2['primary_set']} {q2['primary_verdict']}")
    assert all(s in q2["verdict"] for s in q2["sets"])
    assert scored["card"]["deviations"] == []


def test_scorecard_reads_thresholds_from_the_preregistration(job, scored, tmp_path):
    """A threshold changed in the pre-registration changes the verdict: none is in the code."""
    paths = job["paths"]
    hw = tmp_path / "hw"
    shutil.copytree(paths.hardware_dir, hw)
    prereg_path = hw / f"preregistration_{DATE}.json"
    doc = json.loads(prereg_path.read_text(encoding="utf-8"))
    q1 = next(c for c in doc["criteria"] if c["id"] == "Q1")
    q1["threshold_abs"] = 10.0
    prereg_path.write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    assert score_preregistration.main(["--date", DATE, "--hardware-dir", str(hw), "--data-dir",
                                       str(paths.data_dir), "--out-dir", str(out)]) == 0
    card = json.loads((out / f"scorecard_{DATE}.json").read_text(encoding="utf-8"))
    assert all(s["verdict"] == "held" for s in card["questions"]["Q1"]["sets"].values()
               if s["n"])


def test_hand_added_deviations_survive_a_rescore(job, scored):
    """A deviation added to the scorecard by hand is kept when it is regenerated."""
    path = job["paths"].data_dir / f"scorecard_{DATE}.json"
    card = json.loads(path.read_text(encoding="utf-8"))
    card["deviations"] = [{"reason": "test", "diff": "-"}]
    path.write_text(json.dumps(card), encoding="utf-8")
    assert score_preregistration.main(["--date", DATE, *job["common"]]) == 0
    assert json.loads(path.read_text(encoding="utf-8"))["deviations"] == card["deviations"]


# ---------------------------------------------------------------------------
# Submit gates
# ---------------------------------------------------------------------------


class FakeGit:
    """``hardware_batch.git`` with every call answered, failures chosen per test."""

    def __init__(self) -> None:
        self.fail: dict[tuple[str, ...], tuple[int, str]] = {}
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *arguments: str) -> tuple[int, str]:
        self.calls.append(arguments)
        for prefix, answer in self.fail.items():
            if arguments[: len(prefix)] == prefix:
                return answer
        if arguments[:1] == ("rev-parse",):
            return 0, "blob"
        return 0, ""


def _checklist(path: Path, prereg_sha: str, commit: str = HEAD[:10]) -> None:
    """The template, every box ticked and filled, the pushed line naming ``commit``."""
    text = (REPO_ROOT / "paper" / "hardware" / "CHECKLIST_template.md").read_text(encoding="utf-8")
    body = text.split("## Before `--submit`", 1)[1].split("## After", 1)[0]
    lines = []
    for line in body.splitlines():
        if line.startswith("- [ ]"):
            line = line.replace("- [ ]", "- [x]").replace("<hash>", commit)
            line = line.replace("<date>", DATE).replace("<sha256>", prereg_sha)
            line = line.replace("<instance 1>", "a").replace("<instance 2>", "b")
            if line.startswith("- [x] Estimated QPU time"):
                line = ("- [x] Estimated QPU time <= 50% of remaining allowance: "
                        "estimate 1 s, remaining 100000 s")
            line = line.replace("____", "1")
        lines.append(line)
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture
def live_design(job, tmp_path, monkeypatch):
    """The design job's files relabelled live, a passing set of gates, the service mocked."""
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    hw, data = tmp_path / "hw", tmp_path / "data"
    shutil.copytree(job["paths"].hardware_dir, hw)
    shutil.copytree(job["paths"].data_dir, data)
    paths = hb.Paths(DATE, hw, data)
    for stale in (paths.jobs,):
        stale.unlink(missing_ok=True)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["backend"]["source"] = "live"
    manifest["estimate"]["within_budget"] = True
    manifest["estimate"]["estimated_seconds"] = 1.0
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    sha = hb.sha256_file(paths.manifest)
    for doc_path in (paths.jobs_dryrun, hb.preregistration_path(paths)):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        doc["manifest_sha256"] = sha
        doc_path.write_text(json.dumps(doc), encoding="utf-8")
    prereg_sha = hb.sha256_file(hb.preregistration_path(paths))
    _checklist(paths.checklist, prereg_sha)

    git = FakeGit()
    monkeypatch.setattr(hb, "git", git)
    monkeypatch.setattr(hb, "git_state", lambda **_: {"commit": HEAD, "dirty": False})
    device = FakeMarrakesh()
    monkeypatch.setattr(hb, "connect", lambda *a, **k: (None, device, "live"))
    monkeypatch.setattr(hb, "RUN_LOG", tmp_path / "runs" / "jobs.jsonl")
    monkeypatch.setattr(hb, "submit_sampler_job", lambda *a, **k: StubJob())
    # The plan's snapshot is of this same fake device: its calibration is unchanged.
    return {"common": ["--hardware-dir", str(hw), "--data-dir", str(data)], "paths": paths,
            "git": git, "prereg_sha": prereg_sha, "monkeypatch": monkeypatch}


class StubJob:
    """What --submit reads from a submitted job; nothing is executed."""

    usage_estimation = {"quantum_seconds": 1.0}

    def job_id(self) -> str:
        return "stub-job"


def _submit(live, *extra: str) -> int:
    return hb.main(["--submit", "--date", DATE, *live["common"], *extra])


def test_submit_passes_every_gate_when_all_hold(live_design):
    """With everything in place the design job is submitted and the gates' facts are recorded."""
    assert _submit(live_design) == 0
    jobs = json.loads(live_design["paths"].jobs.read_text(encoding="utf-8"))
    assert jobs["preregistration"]["sha256"] == live_design["prereg_sha"]
    assert jobs["calibration_changes_since_plan"] == []
    assert ("fetch", "origin") in live_design["git"].calls


def test_submit_refuses_without_a_preregistration(live_design, capsys):
    hb.preregistration_path(live_design["paths"]).unlink()
    assert _submit(live_design) == 1
    assert "does not exist; run preregister.py" in capsys.readouterr().out


def test_submit_refuses_a_preregistration_of_another_manifest(live_design, capsys):
    path = hb.preregistration_path(live_design["paths"])
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(doc), encoding="utf-8")
    _checklist(live_design["paths"].checklist, hb.sha256_file(path))
    assert _submit(live_design) == 1
    assert "made from another manifest" in capsys.readouterr().out


def test_submit_refuses_an_uncommitted_preregistration(live_design, capsys):
    live_design["git"].fail[("ls-files",)] = (1, "error: pathspec did not match")
    assert _submit(live_design) == 1
    assert "is not committed" in capsys.readouterr().out


def test_submit_refuses_a_preregistration_that_differs_from_head(live_design, capsys):
    live_design["git"].fail[("status",)] = (0, " M paper/hardware/preregistration.json")
    assert _submit(live_design) == 1
    assert "differs from HEAD" in capsys.readouterr().out


def test_submit_refuses_when_head_is_not_pushed(live_design, capsys):
    live_design["git"].fail[("merge-base", "--is-ancestor", "HEAD", "origin/main")] = (1, "")
    assert _submit(live_design) == 1
    assert "not on origin/main" in capsys.readouterr().out


def test_submit_refuses_a_pushed_line_with_another_sha256(live_design, capsys):
    _checklist(live_design["paths"].checklist, hashlib.sha256(b"other").hexdigest())
    assert _submit(live_design) == 1
    assert "is not the file's" in capsys.readouterr().out


def test_submit_refuses_a_pushed_line_without_its_commit(live_design, capsys):
    live_design["git"].fail[("merge-base", "--is-ancestor", HEAD[:10], "HEAD")] = (1, "")
    assert _submit(live_design) == 1
    out = capsys.readouterr().out
    assert "is not an ancestor of HEAD" in out


def test_submit_refuses_a_changed_calibration_unless_accepted(live_design, capsys):
    """Changed entries on the job's qubits refuse; --accept-recalibration lets it through."""
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    from dgssp.backends import scale_calibration_errors

    recalibrated, _ = scale_calibration_errors(FakeMarrakesh(), 1.1)
    live_design["monkeypatch"].setattr(hb, "connect",
                                       lambda *a, **k: (None, recalibrated, "live"))
    assert _submit(live_design) == 1
    assert "calibration of the job's qubits changed" in capsys.readouterr().out
    assert _submit(live_design, "--accept-recalibration") == 0
    jobs = json.loads(live_design["paths"].jobs.read_text(encoding="utf-8"))
    assert jobs["accepted_recalibration"] is True and jobs["calibration_changes_since_plan"]


def test_design_checklist_needs_the_pushed_line(live_design):
    """A design job's checklist without the pre-registration line is refused."""
    path = live_design["paths"].checklist
    text = "\n".join(x for x in path.read_text(encoding="utf-8").splitlines()
                     if "Pre-registration pushed" not in x)
    path.write_text(text, encoding="utf-8")
    problems = hb.check_checklist(path, head=HEAD, dirty=False, estimate_seconds=1.0,
                                  prereg={"file": "x", "sha256": live_design["prereg_sha"]},
                                  date=DATE)
    assert any("Pre-registration pushed" in p for p in problems)


def test_design_pytest_line_may_name_the_code_commit(live_design):
    """The pytest line may name an ancestor of HEAD when only this job's files changed since."""
    path = live_design["paths"].checklist
    prereg = {"file": "x", "sha256": live_design["prereg_sha"]}
    other = "fedcba9876"
    _checklist(path, live_design["prereg_sha"])
    text = path.read_text(encoding="utf-8").replace(f"commit {HEAD[:10]} (also", f"commit {other} (also")
    path.write_text(text, encoding="utf-8")
    live_design["git"].fail[("diff", "--name-only")] = (
        0, f"paper/hardware/manifest_{DATE}.json\npaper/data/hw_dryrun_{DATE}/a.json")
    assert hb.check_checklist(path, head=HEAD, dirty=False, estimate_seconds=1.0,
                              prereg=prereg, date=DATE) == []
    live_design["git"].fail[("diff", "--name-only")] = (0, "src/dgssp/mitigation/zne.py")
    problems = hb.check_checklist(path, head=HEAD, dirty=False, estimate_seconds=1.0,
                                  prereg=prereg, date=DATE)
    assert any("files other than this job's" in p for p in problems)


def test_non_design_manifests_keep_the_old_checklist_rule(tmp_path):
    """Without a pre-registration the pytest line must still name HEAD, and git is not asked."""
    path = tmp_path / "CHECKLIST.md"
    _checklist(path, "0" * 64, commit="deadbee")
    problems = hb.check_checklist(path, head=HEAD, dirty=False, estimate_seconds=1.0)
    assert any("does not name HEAD" in p for p in problems)


def test_the_real_design_loads_and_names_every_question():
    """paper/hardware/design_job3.json is valid and declares Q1-Q9 with numeric thresholds."""
    design = hb.load_design(REPO_ROOT / "paper" / "hardware" / "design_job3.json")
    ids = [c["id"] for c in design["criteria"]]
    assert ids == [f"Q{i}" for i in range(1, 10)]
    c = {x["id"]: x for x in design["criteria"]}
    assert c["Q1"]["threshold_abs"] == 0.05
    assert c["Q4"]["parts"]["Q4a"]["window"] == [0.75, 1.33]
    assert len(c["Q7"]["map"]) == 9
    n_pubs = sum(len({1.0, *(s for g in r["grids"] for s in (design["grids"][g] if isinstance(
        design["grids"][g], list) else [2.0, 3.0]))}) for r in design["records"])
    n_pubs += sum(1 for r in design["records"] if r["drift"])
    assert n_pubs == 24
    assert copy.deepcopy(design) == design


def _q6_case(day3_alphas):
    """Q6 scored on four circuits over two reference days, day-3 alpha_r given."""
    design = json.loads((REPO_ROOT / "paper" / "hardware" / "design_job3.json")
                        .read_text(encoding="utf-8"))
    c = next(x for x in design["criteria"] if x["id"] == "Q6")
    refs = {"a": (2.2, "2026-10-25"), "b": (3.0, "2026-10-25"), "c": (3.8, "2026-10-25"),
            "d": (2.6, "2026-10-28"), "e": (3.4, "2026-10-28")}
    pred = {"q6": {"references": {
        k: {"alpha_r": v, "alpha_r_se": 0.05, "day": day, "excluded_because": []}
        for k, (v, day) in refs.items()}}}
    day3 = [{"key": k, "alpha_r": a, "alpha_r_se": 0.05, "excluded_because": []}
            for k, a in zip(refs, day3_alphas, strict=True)]
    return score_preregistration.score_q6(c, pred, day3, None)


def test_q6_a_common_shift_is_not_irreproducibility():
    """Every circuit moved by the same amount: reproducible per circuit, shift reported."""
    out = _q6_case([2.5, 3.3, 4.1, 2.6, 3.4])
    assert out["verdict"] == "reproducible per circuit"
    assert out["shifts"]["2026-10-25"]["shift"] == pytest.approx(0.3)
    assert out["shifts"]["2026-10-28"]["shift"] == pytest.approx(0.0)
    assert out["dof_differences"] == 3


def test_q6_reshuffled_alphas_are_not_reproducible():
    """The same spread, but not by circuit: not reproducible."""
    out = _q6_case([3.8, 2.2, 3.0, 3.4, 2.6])
    assert out["verdict"] == "not reproducible"


def test_q6_without_scatter_on_day_three():
    """Circuits that agree on day 3 leave no per-circuit scatter to test."""
    out = _q6_case([3.0, 3.0, 3.0, 3.0, 3.0])
    assert out["verdict"] in ("no scatter to reproduce", "not reproducible")
    assert out["chi2_red_between"] == pytest.approx(0.0)


def test_q6_exclusions_name_their_day():
    """An exclusion says whether the reference or the day-3 alpha_r caused it."""
    design = json.loads((REPO_ROOT / "paper" / "hardware" / "design_job3.json")
                        .read_text(encoding="utf-8"))
    c = next(x for x in design["criteria"] if x["id"] == "Q6")
    edge = "alpha_r at the grid edge"
    pred = {"q6": {"references": {"a": {"alpha_r": 8.0, "alpha_r_se": 0.2,
                                        "day": "2026-10-28", "excluded_because": [edge]}}}}
    day3 = [{"key": "a", "alpha_r": 1.0, "alpha_r_se": 0.01, "excluded_because": [edge]}]
    row = score_preregistration.score_q6(c, pred, day3, None)["rows"][0]
    assert row["excluded_because"] == [f"reference: {edge}", f"day 3: {edge}"]
