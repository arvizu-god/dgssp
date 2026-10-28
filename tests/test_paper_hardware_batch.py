"""
Tests for ``paper/scripts/hardware_batch.py`` (WP6).

The hardware job runs once, so what can be pinned offline is pinned here: the
plan builds the circuits the manifest describes, the dry run and the collect
step produce identical records from the same counts, collect is idempotent, a
changed circuit file is caught, and ``--submit`` refuses everything it should
refuse without touching the network.
"""

from __future__ import annotations

import json
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

#: Two n = 2 instances (one informative, one M = N/2) and the n = 3 sweep
#: instance, whose r = 1 and r = 3 variants join the job.
ONLY = ["P-n2-M1-11", "P-n2-M2-03", "P-n3-M1-17"]
DATE = "2099-03-03"
SHOTS = 256


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory):
    """One offline plan + dry run + collect, shared by the tests below."""
    root = tmp_path_factory.mktemp("wp6")
    common = ["--hardware-dir", str(root / "hw"), "--data-dir", str(root / "data")]
    plan = hb.main(["--plan", "--offline-backend", "fake_marrakesh", "--only", *ONLY,
                    "--seed-max", "2", "--shots", str(SHOTS), "--date", DATE, *common])
    dry = hb.main(["--dry-run", "--date", DATE, *common])
    paths = hb.Paths(DATE, root / "hw", root / "data")
    collect = hb.main(["--collect", "--jobs", str(paths.jobs_dryrun), *common])
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    return {"root": root, "common": common, "paths": paths, "manifest": manifest,
            "status": (plan, dry, collect)}


def test_plan_dry_run_and_collect_succeed(rehearsal):
    """All three offline modes exit 0; collect reports the records identical."""
    assert rehearsal["status"] == (0, 0, 0)


def test_manifest_shape(rehearsal):
    """3 instances + 2 sweep variants at 3 scales, plus 2 drift repeats, last."""
    manifest = rehearsal["manifest"]
    rows = manifest["circuits"]
    assert manifest["totals"]["circuits"] == len(rows) == 3 * 3 + 2 * 3 + 2
    assert manifest["totals"]["shots"] == SHOTS * len(rows)
    assert [r["pub_index"] for r in rows] == list(range(len(rows)))
    assert {r["role"] for r in rows[-2:]} == {"drift_repeat"}
    assert all(r["shots"] == SHOTS for r in rows)
    assert {r["nominal_scale"] for r in rows if r["role"] != "drift_repeat"} == {1.0, 3.0, 5.0}
    assert manifest["backend"]["source"] == "fake"
    # The job is capped server-side at the budget line (offline: half of 600 s).
    assert manifest["sampler"]["options"]["max_execution_time"] == 300
    # Error suppression is gate twirling only; DD is off unless asked for.
    assert manifest["sampler"]["options"]["dynamical_decoupling"] == {"enable": False}
    assert manifest["sampler"]["options"]["twirling"]["enable_gates"] is True
    assert manifest["preflight"]["passed"]
    assert manifest["backend"]["snapshot_matches_live_target"]


def test_drift_instances_are_informative(rehearsal):
    """M = N/2 sits at P_uniform with or without noise, so it is never a drift probe."""
    drift = [r for r in rehearsal["manifest"]["records"] if r["kind"] == "drift"]
    assert len(drift) == 2
    assert "P-n2-M2-03" not in {r["instance"] for r in drift}


def test_drift_repeat_is_the_same_circuit(rehearsal):
    """The repeat PUB is a copy of the instance's scale-1 circuit."""
    _, _, circuits = hb.load_manifest(rehearsal["paths"].manifest)
    for row in rehearsal["manifest"]["records"]:
        if row["kind"] == "drift":
            assert circuits[row["pub_indices"][0]] == circuits[row["repeat_of_pub"]]


def test_sweep_variants_keep_the_layout(rehearsal):
    """Every r of the sweep instance starts on the same physical qubits."""
    layouts = {
        tuple(r["initial_layout"])
        for r in rehearsal["manifest"]["records"]
        if r["instance"] == "P-n3-M1-17"
    }
    iterations = {
        r["iterations"] for r in rehearsal["manifest"]["records"]
        if r["instance"] == "P-n3-M1-17"
    }
    assert len(layouts) == 1
    assert iterations == {1, 2, 3}


def test_estimate_follows_the_stated_formula(rehearsal):
    """(shots x per-shot + circuits x overhead) x 2, with the defaults recorded."""
    est = rehearsal["manifest"]["estimate"]
    expected = (est["total_shots"] * est["per_shot_time_s"]
                + est["n_circuits"] * est["per_circuit_overhead_s"]) * est["safety_factor"]
    assert est["estimated_seconds"] == pytest.approx(expected)
    assert est["safety_factor"] == 2.0
    assert est["floor_seconds"] is not None and est["floor_seconds"] < est["estimated_seconds"]


def test_dry_run_records(rehearsal):
    """Records carry the prediction label, all scales, every model and the source."""
    directory = rehearsal["paths"].dryrun_dir
    record = json.loads((directory / "P-n2-M1-11.json").read_text(encoding="utf-8"))
    assert record["executor"] == "noise_model_prediction"
    assert set(record["counts"]) == {"1", "3", "5"}
    assert sum(record["counts"]["1"].values()) == SHOTS
    assert set(record["metrics"]["zne_fits"]) == set(hb.ZNE_METHODS)
    assert set(record["pointwise"]) >= {"clip", "simplex"}
    assert record["metrics"]["hardware"]["source"] == "dryrun"
    assert record["metrics"]["hardware"]["ignored_options"]

    drift = json.loads((directory / "P-n2-M1-11_drift.json").read_text(encoding="utf-8"))
    assert set(drift["counts"]) == {"1"}
    assert "z" in drift["metrics"]["hardware"]["drift"]


def test_collect_is_idempotent(rehearsal):
    """A second collect writes nothing and still passes the comparison."""
    before = {p.name: p.read_bytes() for p in rehearsal["paths"].dryrun_collect_dir.glob("*.json")}
    status = hb.main(["--collect", "--jobs", str(rehearsal["paths"].jobs_dryrun),
                      *rehearsal["common"]])
    after = {p.name: p.read_bytes() for p in rehearsal["paths"].dryrun_collect_dir.glob("*.json")}
    assert status == 0
    assert before == after


def test_a_changed_circuit_file_is_caught(rehearsal, tmp_path):
    """The manifest pins the QPY file by hash."""
    paths = rehearsal["paths"]
    original = paths.circuits.read_bytes()
    try:
        paths.circuits.write_bytes(original + b"\0")
        with pytest.raises(SystemExit, match="changed since --plan"):
            hb.load_manifest(paths.manifest)
    finally:
        paths.circuits.write_bytes(original)


def test_submit_refuses_a_rehearsal_manifest(rehearsal, monkeypatch):
    """A fake-device manifest is never submitted, and no service is built."""
    monkeypatch.setattr(hb, "get_service", lambda **_: pytest.fail("service built"))
    status = hb.main(["--submit", "--date", DATE, *rehearsal["common"]])
    assert status == 1


def test_plan_refuses_to_overwrite(rehearsal):
    """A second plan for the same date needs --force."""
    status = hb.main(["--plan", "--offline-backend", "fake_marrakesh", "--only", *ONLY,
                      "--date", DATE, *rehearsal["common"]])
    assert status == 1


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

HEAD = "0123456789abcdef0123456789abcdef01234567"


def _filled_checklist() -> str:
    """The template with every blank filled and every box ticked."""
    text = (REPO_ROOT / "paper" / "hardware" / "CHECKLIST_template.md").read_text(encoding="utf-8")
    body = text.split("## Before `--submit`", 1)[1].split("## After", 1)[0]
    lines = []
    for line in body.splitlines():
        if line.startswith("- [ ]"):
            line = line.replace("- [ ]", "- [x]")
            line = line.replace("<hash>", HEAD[:10]).replace("<date>", DATE)
            line = line.replace("<instance 1>", "P-n2-M1-11").replace("<instance 2>", "P-n3-M2-17")
            if line.startswith("- [x] Estimated QPU time"):
                line = "- [x] Estimated QPU time <= 50% of remaining allowance: estimate 250 s, remaining 600 s"
            line = line.replace("____", "1")
        lines.append(line)
    return "\n".join(lines)


def test_checklist_template_passes_when_filled(tmp_path):
    """The shipped template, filled in, is accepted."""
    path = tmp_path / "CHECKLIST.md"
    path.write_text(_filled_checklist(), encoding="utf-8")
    assert hb.check_checklist(path, head=HEAD, dirty=False, estimate_seconds=240.0) == []


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda t: t.replace("- [x] Token", "- [ ] Token", 1), "not ticked"),
        (lambda t: t.replace("P-n3-M2-17", "<instance 2>", 1), "placeholder"),
        (lambda t: t.replace(HEAD[:10], "deadbee", 1), "does not name HEAD"),
        (lambda t: t.replace("estimate 250 s", "estimate 400 s", 1), "exceeds"),
        (lambda t: t.replace("estimate 250 s", "estimate 100 s", 1), "below the manifest"),
        (lambda t: "\n".join(x for x in t.splitlines() if "Job log" not in x), "check-boxes"),
    ],
)
def test_checklist_problems_are_reported(tmp_path, edit, message):
    """Each way of skipping a step is caught."""
    path = tmp_path / "CHECKLIST.md"
    path.write_text(edit(_filled_checklist()), encoding="utf-8")
    problems = hb.check_checklist(path, head=HEAD, dirty=False, estimate_seconds=240.0)
    assert any(message in p for p in problems), problems


def test_checklist_requires_a_clean_tree_and_the_file(tmp_path):
    """A dirty tree or a missing checklist refuses."""
    path = tmp_path / "CHECKLIST.md"
    assert hb.check_checklist(path, head=HEAD, dirty=False, estimate_seconds=1.0)
    path.write_text(_filled_checklist(), encoding="utf-8")
    problems = hb.check_checklist(path, head=HEAD, dirty=True, estimate_seconds=240.0)
    assert any("dirty" in p for p in problems)


def test_sampler_options_are_fixed_and_exact():
    """Twirled shots multiply back to the requested shots; odd shots are refused."""
    options = hb.sampler_options(4096, twirling=True, dd=True, date=DATE)
    twirl = options["twirling"]
    assert twirl["num_randomizations"] * twirl["shots_per_randomization"] == 4096
    assert twirl["enable_measure"] is False
    assert options["dynamical_decoupling"]["sequence_type"] == hb.DD_SEQUENCE
    with pytest.raises(ValueError, match="multiple"):
        hb.sampler_options(4100, twirling=True, dd=True, date=DATE)


def test_drift_metrics():
    """No change gives z = 0; a large change is significant."""
    same = hb.drift_metrics({"1": 50, "0": 50}, {"1": 50, "0": 50}, ["1"])
    assert same["z"] == 0 and same["tv_between"] == 0
    moved = hb.drift_metrics({"1": 900, "0": 100}, {"1": 500, "0": 500}, ["1"])
    assert moved["p_value"] < 1e-6 and moved["tv_between"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def _span(start_s: float, stop_s: float, shots_by_pub: dict[int, int]):
    """A synthetic execution span over whole PUBs."""
    from datetime import datetime, timedelta, timezone

    from qiskit_ibm_runtime.execution_span import DoubleSliceSpan

    t0 = datetime(2026, 10, 25, tzinfo=timezone.utc)
    return DoubleSliceSpan(
        t0 + timedelta(seconds=start_s),
        t0 + timedelta(seconds=stop_s),
        {i: ((n,), slice(0, 1), slice(0, n)) for i, n in shots_by_pub.items()},
    )


def _twirled_span(start_s: float, stop_s: float, pubs: list[int], shots: int = 4096):
    """A TwirledSliceSpanV2 like the one ibm_fez returned: 32 x 128 per PUB."""
    from datetime import datetime, timedelta, timezone

    from qiskit_ibm_runtime.execution_span import TwirledSliceSpanV2

    t0 = datetime(2026, 10, 26, tzinfo=timezone.utc)
    rand = 32
    return TwirledSliceSpanV2(
        t0 + timedelta(seconds=start_s),
        t0 + timedelta(seconds=stop_s),
        {i: ((rand, shots // rand), True, slice(0, rand), slice(0, shots // rand), shots)
         for i in pubs},
    )


#: The live plan's numbers (ibm_fez, 2026-10-25): 47 circuits, 65.9 s floor.
ESTIMATE = {"total_shots": 47 * 4096, "n_circuits": 47, "floor_seconds": 65.9,
            "budget_cap_seconds": 300.0}
#: The live probe's PUBs: shortest and longest circuit.
PUBS = [{"duration_s": 8.568e-06}, {"duration_s": 453.48e-06}]
REP = 250e-6
MODEL = 4096 * (8.568e-06 + REP) + 4096 * (453.48e-06 + REP)


def test_twirled_spans_are_read_without_the_broken_filter_by_pub():
    """qiskit-ibm-runtime 0.49.0's TwirledSliceSpanV2.filter_by_pub raises; masks do not."""
    span = _twirled_span(0, 6.235, [0, 1])
    with pytest.raises(IndexError):
        span.filter_by_pub(0)  # the library bug this code works around
    assert hb.span_shots_per_pub(span, 2) == {0: 4096, 1: 4096}


def test_measure_probe_on_the_live_probe_shape():
    """One span shared by both PUBs, 6.235 s: both readings projected, the larger kept."""
    spans = [_twirled_span(0, 6.235, [0, 1])]
    out = hb.measure_probe(PUBS, shots=4096, usage_seconds=8.0, spans=spans,
                           rep_delay=REP, manifest_estimate=ESTIMATE)
    assert out["spans_shared_by_pubs"] is True
    assert out["per_pub_seconds"] is None
    k = 6.235 / MODEL
    c = (6.235 - MODEL) / 2
    assert out["measured_over_model"] == pytest.approx(k)
    main = out["main_job"]
    assert main["in_span_if_per_shot_s"] == pytest.approx(k * 65.9)
    assert main["in_span_if_per_pub_s"] == pytest.approx(65.9 + 47 * c)
    fixed = 8.0 - 6.235
    in_span = max(k * 65.9, 65.9 + 47 * c)
    assert main["expected_conservative_s"] == pytest.approx(in_span + 47 * fixed / 2)
    assert main["expected_optimistic_s"] == pytest.approx(k * 65.9 + fixed)
    assert main["estimated_seconds"] == pytest.approx(2 * (in_span + 47 * fixed / 2))
    # The flags reproduce the estimate through --plan's own formula.
    rec = out["recommendation"]
    replanned = (ESTIMATE["total_shots"] * rec["per_shot_time_s"] + 47 * rec["overhead_s"]) * 2
    assert replanned == pytest.approx(main["estimated_seconds"])


def test_exclusive_spans_give_per_pub_times_and_overlaps_count_once():
    """Spans that each cover one PUB are attributed; overlapping windows are not double-counted."""
    spans = [_span(0, 1.0, {0: 4096}), _span(0.5, 4.0, {1: 4096})]
    assert hb.span_seconds_per_pub(spans, 2) == pytest.approx([1.0, 3.5])
    assert hb.spans_union_seconds(spans) == pytest.approx(4.0)
    shared = [_span(0, 2.0, {0: 1024, 1: 3072})]
    assert hb.span_seconds_per_pub(shared, 2) is None


def test_measure_probe_without_spans_or_final_usage():
    """No spans: the schedule model stands in and is flagged.  usage 0: no recommendation yet."""
    out = hb.measure_probe(PUBS, shots=4096, usage_seconds=5.0, spans=None,
                           rep_delay=REP, manifest_estimate=ESTIMATE)
    assert out["executed_source"].startswith("schedule model")
    assert out["measured_over_model"] == pytest.approx(1.0)
    pending = hb.measure_probe(PUBS, shots=4096, usage_seconds=0, spans=None,
                               rep_delay=REP, manifest_estimate=ESTIMATE)
    assert pending["recommendation"] is None


def test_probe_picks_the_shortest_and_longest_circuit(rehearsal):
    """The longest circuit is the one the auto per-shot estimate charges every shot."""
    rows = [r for r in rehearsal["manifest"]["circuits"] if r["role"] != "drift_repeat"]
    short, long_ = hb.probe_pubs(rehearsal["manifest"])
    assert short["duration_s"] == min(r["duration_s"] for r in rows)
    assert long_["duration_s"] == max(r["duration_s"] for r in rows)
    assert short["nominal_scale"] == 1.0


@pytest.fixture
def live_copy(rehearsal, tmp_path, monkeypatch):
    """The rehearsal's files, relabelled live, with the service replaced by the fake device."""
    import shutil

    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    hw, data = tmp_path / "hw", tmp_path / "data"
    shutil.copytree(rehearsal["paths"].hardware_dir, hw)
    shutil.copytree(rehearsal["paths"].data_dir, data)
    manifest_path = hw / f"manifest_{DATE}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["backend"]["source"] = "live"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    device = FakeMarrakesh()
    monkeypatch.setattr(hb, "connect", lambda *a, **k: (None, device, "live"))
    monkeypatch.setattr(hb, "RUN_LOG", tmp_path / "runs" / "jobs.jsonl")
    return ["--hardware-dir", str(hw), "--data-dir", str(data)], hw, tmp_path


def test_probe_submits_two_pubs_and_reports(live_copy, monkeypatch):
    """--probe sends 2 PUBs with the manifest's options and a 60 s cap; --probe-report measures."""
    common, hw, tmp = live_copy
    submitted = {}
    real_submit = hb.submit_sampler_job

    def spy(mode, circuits, **kwargs):
        job = real_submit(mode, circuits, **kwargs)
        submitted.update(job=job, n=len(circuits), options=kwargs["options"])
        return job

    monkeypatch.setattr(hb, "submit_sampler_job", spy)
    with pytest.warns(UserWarning, match="no effect in local testing mode"):
        assert hb.main(["--probe", "--date", DATE, *common]) == 0
    assert submitted["n"] == 2
    assert submitted["options"]["max_execution_time"] == hb.PROBE_CAP_SECONDS
    assert submitted["options"]["twirling"]["enable_gates"] is True
    assert (tmp / "runs" / "jobs.jsonl").exists()
    assert hb.main(["--probe", "--date", DATE, *common]) == 1  # never twice

    class Status:
        name = "DONE"

    class Job:
        def __init__(self, job):
            self._job = job

        def job_id(self):
            return self._job.job_id()

        def status(self):
            return Status()

        def result(self):
            return self._job.result()

        def usage(self):
            return 9.0

        usage_estimation = {"quantum_seconds": 8.0}

        def metrics(self):
            return {"usage": {"quantum_seconds": 9.0}}

    class Service:
        def job(self, job_id):
            return Job(submitted["job"])

    monkeypatch.setenv(hb.TOKEN_ENV, "test-token")
    monkeypatch.setattr(hb, "get_service", lambda **_: Service())
    assert hb.main(["--probe-report", "--date", DATE, *common]) == 0
    doc = json.loads((hw / f"probe_{DATE}.json").read_text(encoding="utf-8"))
    assert doc["measured"]["options_accepted"] is True
    assert doc["measured"]["twirling_applied"] is False  # local mode never twirls
    assert doc["measured"]["recommendation"]["overhead_s"] >= 0
    assert "probe" in (hw / "usage.md").read_text(encoding="utf-8")


def test_submit_refuses_a_stale_dry_run(live_copy, monkeypatch, capsys):
    """A dry run of an older manifest does not count."""
    common, hw, _ = live_copy
    jobs = hw / f"jobs_dryrun_{DATE}.json"
    doc = json.loads(jobs.read_text(encoding="utf-8"))
    doc["manifest_sha256"] = "0" * 64
    jobs.write_text(json.dumps(doc), encoding="utf-8")
    manifest_path = hw / f"manifest_{DATE}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["estimate"]["within_budget"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(hb, "connect", lambda *a, **k: pytest.fail("service reached"))
    assert hb.main(["--submit", "--date", DATE, *common]) == 1
    assert "older plan" in capsys.readouterr().out


def test_a_failed_plan_leaves_nothing_that_blocks_the_next(tmp_path, monkeypatch):
    """A crash while freezing removes this run's files; an orphan without a manifest is replaced."""
    common = ["--hardware-dir", str(tmp_path / "hw"), "--data-dir", str(tmp_path / "data")]
    args = ["--plan", "--offline-backend", "fake_marrakesh", "--only", "P-n2-M1-11",
            "--no-sweep", "--seed-max", "1", "--shots", str(SHOTS), "--date", DATE, *common]
    paths = hb.Paths(DATE, tmp_path / "hw", tmp_path / "data")

    def boom(*_a, **_k):
        raise RuntimeError("calibration fetch failed")

    monkeypatch.setattr(hb, "save_noise_model", boom)
    with pytest.raises(RuntimeError, match="calibration fetch failed"):
        hb.main(args)
    assert not paths.circuits.exists() and not paths.calibration.exists()
    monkeypatch.undo()

    paths.circuits.parent.mkdir(parents=True, exist_ok=True)
    paths.circuits.write_bytes(b"left over")
    assert hb.main(args) == 0
    assert paths.manifest.exists()
    assert hb.load_manifest(paths.manifest)[2]  # the orphan was replaced by real circuits


def test_hardware_records_carry_an_offline_mitiq_block(rehearsal):
    """Mitiq extrapolates the dgssp-folded counts; the block says so."""
    from dgssp.mitigation import mitiq_is_available

    if not mitiq_is_available():
        pytest.skip("Mitiq not installed")
    record = json.loads(
        (rehearsal["paths"].dryrun_dir / "P-n2-M1-11.json").read_text(encoding="utf-8")
    )
    mitiq = record["mitiq"]
    assert mitiq["fold_source"] == hb.MITIQ_FOLD_SOURCE
    assert mitiq["scales"] == [1.0, 3.0, 5.0]  # the dgssp realised scales, not Mitiq's
    assert mitiq["method"] == record["zne_fit"]["method"]
    assert record["metrics"]["mitiq_solution_probability"] == mitiq["value"]
    assert record["metrics"]["hardware"]["mitiq"] == hb.MITIQ_FOLD_SOURCE
    # Same counts, so Mitiq's observed points are dgssp's observed points.
    assert mitiq["values"] == pytest.approx(record["zne_fit"]["observed"])


def test_sweep_variants_can_be_restricted_to_some_iterations():
    """``--sweep-iterations 1``: the r = 1 variants only; r_opt is never a variant."""
    rows = [
        {"name": "A", "r_opt": 2, "iterations": [1, 2, 3]},
        {"name": "B", "r_opt": 3, "iterations": [1, 2, 3, 4]},
        {"name": "C", "r_opt": 1, "iterations": [1, 2]},
    ]
    kept = hb.select_sweep_iterations(rows, [1])
    assert [(r["name"], r["iterations"]) for r in kept] == [("A", [1]), ("B", [1])]
    assert rows[0]["iterations"] == [1, 2, 3]  # the input is not modified
    assert hb.select_sweep_iterations(rows, [2])[0]["iterations"] == [2]


PIN_DATE = "2099-03-04"


@pytest.fixture(scope="module")
def pinned(rehearsal, tmp_path_factory):
    """A plan whose n = 3 records are pinned to the rehearsal's layouts, then dry run + collect."""
    root = tmp_path_factory.mktemp("wp6_pinned")
    hw = root / "hw"
    hw.mkdir()
    source = rehearsal["paths"].manifest
    (hw / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    fresh = next(r for r in rehearsal["manifest"]["records"] if r["key"] == "P-n2-M1-11")
    explicit = ",".join(str(q) for q in fresh["initial_layout"]) + f"@{fresh['best_seed']}"
    common = ["--hardware-dir", str(hw), "--data-dir", str(root / "data")]
    plan = hb.main([
        "--plan", "--offline-backend", "fake_marrakesh", "--only", "P-n2-M1-11", "--no-sweep",
        "--pin-layout", f"P-n3-M1-17:L1={DATE}", f"P-n3-M1-17_r1:L1={DATE}",
        f"P-n2-M1-11:E={explicit}",
        "--drift", "P-n2-M1-11", "P-n3-M1-17_L1",
        "--scales", "1", "1.25", "1.5",
        "--seed-max", "2", "--shots", str(SHOTS), "--date", PIN_DATE, *common,
    ])
    dry = hb.main(["--dry-run", "--date", PIN_DATE, *common])
    paths = hb.Paths(PIN_DATE, hw, root / "data")
    collect = hb.main(["--collect", "--jobs", str(paths.jobs_dryrun), *common])
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    return {"paths": paths, "manifest": manifest, "status": (plan, dry, collect),
            "common": common, "source": rehearsal}


def test_a_pinned_plan_runs_end_to_end(pinned):
    """Plan, dry run and collect all exit 0; 4 records x 3 scales + 2 drift repeats."""
    assert pinned["status"] == (0, 0, 0)
    manifest = pinned["manifest"]
    assert manifest["totals"]["circuits"] == 4 * 3 + 2
    assert manifest["totals"]["by_role"]["pinned"] == 3 * 3
    assert manifest["settings"]["drift_keys"] == ["P-n2-M1-11", "P-n3-M1-17_L1"]


def test_a_pinned_record_is_the_source_records_circuit(pinned):
    """Same layout, seed and r as the source record, and the same scale-1 ISA circuit."""
    _, _, source_circuits = hb.load_manifest(pinned["source"]["paths"].manifest)
    _, _, circuits = hb.load_manifest(pinned["paths"].manifest)
    source = {r["key"]: r for r in pinned["source"]["manifest"]["records"]}
    records = {r["key"]: r for r in pinned["manifest"]["records"] if r["kind"] == "zne"}
    for key, origin in (("P-n3-M1-17_L1", "P-n3-M1-17"), ("P-n3-M1-17_r1_L1", "P-n3-M1-17_r1")):
        row, src = records[key], source[origin]
        assert row["role"] == "pinned" and row["seed_source"] == "pinned_layout"
        assert row["instance"] == "P-n3-M1-17"
        assert (row["initial_layout"], row["best_seed"], row["iterations"]) == (
            src["initial_layout"], src["best_seed"], src["iterations"])
        assert row["pinned_from"]["record_key"] == origin
        assert row["pinned_from"]["manifest"].endswith(f"manifest_{DATE}.json")
        new = circuits[row["pub_indices"][row["nominal_scales"].index(1.0)]]
        old = source_circuits[src["pub_indices"][src["nominal_scales"].index(1.0)]]
        assert new == old  # the circuit that job sent, not a re-transpilation
        assert row["pinned_from"]["pub_index"] == src["pub_indices"][0]


def test_an_explicit_layout_is_kept_in_initial_layout_order(pinned):
    """The typed layout and seed are the ones used, virtual qubit i on the i-th entry."""
    _, _, circuits = hb.load_manifest(pinned["paths"].manifest)
    records = {r["key"]: r for r in pinned["manifest"]["records"]}
    fresh, typed = records["P-n2-M1-11"], records["P-n2-M1-11_E"]
    assert typed["pinned_from"] == "explicit"
    assert typed["seed_source"] == "pinned_layout"
    assert (typed["initial_layout"], typed["best_seed"]) == (
        fresh["initial_layout"], fresh["best_seed"])
    circuit = circuits[typed["pub_indices"][0]]
    assert circuit.layout.initial_index_layout(filter_ancillas=True) == fresh["initial_layout"]


def test_named_drift_records_lead_and_are_repeated_last(pinned):
    """--drift puts the named records first and their scale-1 copies last."""
    manifest = pinned["manifest"]
    zne = [r for r in manifest["records"] if r["kind"] == "zne"]
    assert [r["key"] for r in zne[:2]] == ["P-n2-M1-11", "P-n3-M1-17_L1"]
    drift = [r for r in manifest["records"] if r["kind"] == "drift"]
    assert [r["key"] for r in drift] == ["P-n2-M1-11_drift", "P-n3-M1-17_L1_drift"]
    assert drift[-1]["pub_indices"] == [manifest["totals"]["circuits"] - 1]
    assert drift[1]["seed_source"] == "pinned_layout"


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("P-n3-M2-17:L1=2026-10-25",
         {"key": "P-n3-M2-17", "tag": "L1", "record_key": "P-n3-M2-17_L1",
          "date": "2026-10-25", "layout": None, "seed": None}),
        ("P-n4-M1-10_r1:L1=2026-10-25",
         {"key": "P-n4-M1-10_r1", "tag": "L1", "record_key": "P-n4-M1-10_r1_L1",
          "date": "2026-10-25", "layout": None, "seed": None}),
        ("P-n2-M1-11:X=98,93,92,90,91@23",
         {"key": "P-n2-M1-11", "tag": "X", "record_key": "P-n2-M1-11_X",
          "date": None, "layout": [98, 93, 92, 90, 91], "seed": 23}),
        ("P-n2-M1-11:X=98,93,92,90,91",
         {"key": "P-n2-M1-11", "tag": "X", "record_key": "P-n2-M1-11_X",
          "date": None, "layout": [98, 93, 92, 90, 91], "seed": hb.SEED_MIN}),
    ],
)
def test_pin_values_are_parsed(spec, expected):
    """Both forms of --pin-layout."""
    assert hb.parse_pin(spec) == expected


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("P-n3-M2-17=2026-10-25", "expected KEY:TAG"),
        ("P-n3-M2-17:L_1=2026-10-25", "expected KEY:TAG"),
        ("P-n3-M2-17:L1=yesterday", "neither a date nor a layout"),
        ("P-n3-M2-17:L1=1,2,2", "appears twice"),
    ],
)
def test_malformed_pin_values_are_refused(spec, message):
    """A tag is alphanumeric (it becomes part of a file name); a layout has no repeats."""
    with pytest.raises(ValueError, match=message):
        hb.parse_pin(spec)


def test_pins_are_refused_when_they_cannot_reproduce_the_source(rehearsal):
    """Another device, an unknown record, a missing manifest, or the same key twice."""
    hw = rehearsal["paths"].manifest.parent
    device = rehearsal["manifest"]["backend"]["name"]
    subset = json.loads((hb.INSTANCES_DIR / "hardware_subset.json").read_text(encoding="utf-8"))
    rows = {r["name"]: r for r in subset["selected"]}
    with pytest.raises(ValueError, match="not on ibm_fez"):
        hb.resolve_pins([f"P-n3-M1-17:L1={DATE}"], rows, hw, "ibm_fez")
    with pytest.raises(ValueError, match="no ZNE record"):
        hb.resolve_pins([f"P-n4-M1-10:L1={DATE}"], rows, hw, device)
    with pytest.raises(ValueError, match="does not exist"):
        hb.resolve_pins(["P-n3-M1-17:L1=2000-01-01"], rows, hw, device)
    with pytest.raises(ValueError, match="pinned twice"):
        hb.resolve_pins([f"P-n3-M1-17:L1={DATE}", "P-n3-M1-17:L1=1,2,3,4,5,6"], rows, hw, device)
    with pytest.raises(ValueError, match="in no instance file"):
        hb.resolve_pins(["P-n9-M1-99:L1=1,2"], rows, hw, device)


def test_a_drift_key_must_be_a_record_with_signal(tmp_path, capsys):
    """--drift on a missing record, or on M = N/2 (always at P_uniform), refuses the plan."""
    common = ["--hardware-dir", str(tmp_path / "hw"), "--data-dir", str(tmp_path / "data")]
    base = ["--plan", "--offline-backend", "fake_marrakesh", "--no-sweep", "--seed-max", "1",
            "--shots", str(SHOTS), "--date", DATE, *common]
    assert hb.main([*base, "--only", "P-n2-M1-11", "--drift", "P-n2-M1-10"]) == 1
    assert "no such record" in capsys.readouterr().out
    assert hb.main([*base, "--only", "P-n2-M2-03", "--drift", "P-n2-M2-03"]) == 1
    assert "cannot show drift" in capsys.readouterr().out
    assert not (tmp_path / "hw" / f"manifest_{DATE}.json").exists()
