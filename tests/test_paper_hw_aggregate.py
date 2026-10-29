"""
Tests for ``paper/scripts/hw_aggregate.py`` and the hardware figure set.

The decisions that would silently corrupt a hardware table: which status a
record's estimate gets, whether the fitted curve the figure draws is the fit
the record stores, which model value stands beside the hardware, and whether
the run configuration is read from the files rather than typed.  The job of
record (``2026-10-25``) is committed, so most checks run on it.
"""

from __future__ import annotations

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
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

hw_aggregate = pytest.importorskip("hw_aggregate")
make_figures = pytest.importorskip("make_figures")
plotting = pytest.importorskip("plotting")

DATE = "2026-10-25"
DATA = REPO_ROOT / "paper" / "data"
HARDWARE = REPO_ROOT / "paper" / "hardware"
HAVE_JOB = (HARDWARE / f"manifest_{DATE}.json").exists() and (DATA / f"hw_{DATE}").exists()
needs_job = pytest.mark.skipif(not HAVE_JOB, reason="the job of record is not checked out")
LATER = "2026-10-28"
LATER_PREDICTION = DATA / f"hw_predict_plan_x3.176_{LATER}"
HAVE_LATER = HAVE_JOB and (HARDWARE / f"manifest_{LATER}.json").exists() and (
    DATA / f"hw_{LATER}").exists() and LATER_PREDICTION.exists()
needs_later = pytest.mark.skipif(not HAVE_LATER, reason="the gentle-grid job is not checked out")


@pytest.fixture(scope="module")
def summary():
    """The summary of the job of record, built from the committed files."""
    return hw_aggregate.build_summary(hw_aggregate.HwInputs(DATE, HARDWARE, DATA))


# ---------------------------------------------------------------------------
# Small rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "status"),
    [
        ({"signal_direction": 1, "resolved": True, "identifiable": True,
          "reported_source": "extrapolation"}, "resolved"),
        ({"signal_direction": 1, "resolved": False, "identifiable": True,
          "reported_source": "min_scale"}, "identified, not resolved"),
        ({"signal_direction": 1, "resolved": False, "identifiable": False,
          "reported_source": "min_scale"}, "not identifiable"),
        ({"signal_direction": 0, "resolved": False, "identifiable": False,
          "reported_source": "min_scale"}, "no signal"),
        ({"resolved": None, "identifiable": None}, None),
    ],
)
def test_status_names_what_the_estimate_is(flags, status):
    """M = 0 / M = N/2 is "no signal" before anything else is asked."""
    assert hw_aggregate.status_of(flags) == status


def test_the_curve_is_only_drawn_for_the_fixed_asymptote_exponential():
    """Another model's parameters mean something else; no curve is better than a wrong one."""
    fit = {"method": "exponential", "params": [0.5, 1.0], "asymptote": 0.25}
    assert hw_aggregate.exponential_curve(fit, [0.0]) == [pytest.approx(0.75)]
    assert hw_aggregate.exponential_curve({**fit, "method": "linear"}, [0.0]) is None
    assert hw_aggregate.exponential_curve({**fit, "params": [1, 2, 3]}, [0.0]) is None
    assert hw_aggregate.exponential_curve(None, [0.0]) is None


def test_latex_specials_in_keys_are_escaped():
    """``P-n3-M1-17_r1`` would break the table as a subscript."""
    assert hw_aggregate._tex("P-n3-M1-17_r1") == "P-n3-M1-17\\_r1"
    assert hw_aggregate._num(None) == "---"
    assert hw_aggregate._num(float("nan")) == "---"


def test_times_are_written_readably():
    """ISO stamps become date, time and offset; the job's in UTC."""
    assert hw_aggregate._when("2026-10-25T18:24:25-06:00") == "2026-10-25 18:24:25 (UTC-06:00)"
    assert hw_aggregate._when("2026-10-26T06:05:15.018736Z", utc=True) == (
        "2026-10-26 06:05:15 UTC"
    )
    assert hw_aggregate._when(None) == "---"


def test_the_limit_band_is_read_off_the_statuses():
    """Last resolved to the next not identifiable; no-signal records say nothing."""
    def rec(key, gates, status):
        return {"key": key, "two_q_count": gates, "zne_status": status}

    records = [
        rec("a", 50, "resolved"), rec("b", 90, "identified, not resolved"),
        rec("c", 40, "no signal"), rec("d", 150, "not identifiable"),
        rec("e", 300, "identified, not resolved"), rec("f", 120, "not identifiable"),
        rec("g", 30, "not identifiable"),
    ]
    band = hw_aggregate.limit_band(records)
    assert (band["lower"], band["lower_key"]) == (50, "a")
    assert (band["upper"], band["upper_key"]) == (120, "f")
    # e succeeds above the band, g fails below it: listed, not moving the edges.
    assert band["exceptions"] == ["e", "g"]
    assert hw_aggregate.limit_band([rec("d", 150, "not identifiable")]) is None
    assert hw_aggregate.limit_band([rec("a", 50, "resolved")]) is None


def test_a_figure_legend_lists_each_series_once():
    """A series drawn in every panel appears once, under the whole figure."""
    fig, axes = plotting.figure(ncols=2)
    for ax in axes:
        ax.plot([0, 1], [0, 1], label="a")
    axes[1].plot([0, 1], [1, 0], label="b")
    legend = plotting.figure_legend(fig, axes)
    assert [t.get_text() for t in legend.get_texts()] == ["a", "b"]
    matplotlib.pyplot.close(fig)


# ---------------------------------------------------------------------------
# The job of record
# ---------------------------------------------------------------------------


@needs_job
def test_every_zne_record_is_tabulated_and_the_drift_repeats_are_not(summary):
    """15 ZNE records; the two drift repeats go to the drift check instead."""
    keys = [r["key"] for r in summary["records"]]
    assert len(keys) == 15
    assert not any(k.endswith("_drift") for k in keys)
    assert [d["key"] for d in summary["drift"]] == ["P-n2-M1-11_drift", "P-n3-M2-17_drift"]
    assert summary["drift"][0]["repeat_of"] == "P-n2-M1-11"


@needs_job
def test_statuses_follow_the_current_rule(summary):
    """One resolved record (n = 2); the M = 0 and M = N/2 records have no signal."""
    assert summary["resolved_rule"] == "v2"
    assert summary["statuses"] == {
        "resolved": 1,
        "identified, not resolved": 2,
        "not identifiable": 8,
        "no signal": 4,
    }
    by_key = {r["key"]: r for r in summary["records"]}
    best = by_key["P-n2-M1-11"]
    assert best["zne_status"] == "resolved"
    assert best["zne_reported"] == pytest.approx(0.953, abs=5e-4)
    assert by_key["P-n2-M2-03"]["zne_status"] == "no signal"
    # Declined estimates report the lambda = 1 observation, with no interval.
    declined = by_key["P-n3-M1-17"]
    assert declined["zne_reported"] == pytest.approx(declined["hardware"])
    assert declined["zne_reported_lo"] is None


@needs_job
def test_report_bounds_clip_moves_only_the_interval_past_one(summary):
    """Step 5: clip vs flag on the job of record -- same statuses, one interval end."""
    assert summary["report_bounds"] == "clip"
    flagged = hw_aggregate.build_summary(hw_aggregate.HwInputs(DATE, HARDWARE, DATA), "flag")
    assert flagged["report_bounds"] == "flag"
    assert flagged["statuses"] == summary["statuses"]
    clip = {r["key"]: r for r in summary["records"]}
    flag = {r["key"]: r for r in flagged["records"]}
    assert flag["P-n2-M1-11"]["zne_reported_hi"] > 1.0
    assert clip["P-n2-M1-11"]["zne_reported_hi"] == 1.0
    assert clip["P-n2-M1-11"]["ci_out_of_range"] and not clip["P-n2-M1-11"]["out_of_range"]
    for key, r in clip.items():
        for column in ("zne_status", "resolved", "identifiable", "reported_source", "zne_value"):
            assert r[column] == flag[key][column]
        if key != "P-n2-M1-11":
            assert r["zne_reported_hi"] == flag[key]["zne_reported_hi"]


@needs_job
def test_the_limit_band_of_the_job(summary):
    """ZNE resolves at 71 two-qubit gates and fails from 194; one record bucks it."""
    band = summary["limit_band"]
    assert (band["lower"], band["lower_key"]) == (71, "P-n2-M1-11")
    assert (band["upper"], band["upper_key"]) == (194, "P-n3-M2-17")
    assert band["exceptions"] == ["P-n4-M2-02"]


@needs_job
def test_the_drawn_curve_is_the_stored_fit(summary):
    """At the measured scales the curve must equal the record's own fitted values."""
    for r in summary["records"]:
        curve = r["detail"]["fit_curve"]
        if not curve["p"]:
            continue
        stored = json.loads((DATA / f"hw_{DATE}" / f"{r['key']}.json").read_text(
            encoding="utf-8"))["zne_fit"]["fitted"]
        at_scales = [curve["p"][curve["lambda"].index(s)] for s in r["detail"]["scales"]]
        assert at_scales == pytest.approx(stored, abs=1e-9), r["key"]


@needs_job
def test_the_alpha_model_lands_on_the_grid_where_the_grid_has_the_point():
    """PCHIP at a grid point returns that point's prediction exactly."""
    fit_doc = json.loads((DATA / f"hw_alpha_exec_{DATE}" / "fit.json").read_text(
        encoding="utf-8"))
    predictions = hw_aggregate.alpha_predictions(fit_doc, 3.0)
    row = fit_doc["per_pub"][0]
    assert predictions[(row["key"], float(row["scale"]))] == pytest.approx(
        float(row["predicted"]["3"]))
    assert hw_aggregate.alpha_predictions(None, 3.0) == {}


@needs_job
def test_signal_kept_is_defined_exactly_where_there_is_signal(summary):
    """Signal kept needs P_ideal != P_uniform; the model at alpha_hat sits near the hardware."""
    for r in summary["records"]:
        has_signal = r["signal_direction"] in (1, -1)
        assert (r["kept_hardware"] is not None) == has_signal, r["key"]
    best = next(r for r in summary["records"] if r["key"] == "P-n2-M1-11")
    assert best["kept_model_exec"] == pytest.approx(0.792, abs=1e-3)
    assert best["kept_hardware"] == pytest.approx(0.466, abs=1e-3)
    assert abs(best["kept_model_alpha"] - best["kept_hardware"]) < 0.05


@needs_job
def test_the_configuration_is_read_from_the_files(summary):
    """Nothing in Table "run configuration" is typed in."""
    c = summary["config"]
    assert c["backend"] == "ibm_fez"
    assert c["circuits"] == 47 and c["total_shots"] == 192512
    assert c["qpu_charged_s"] == 75
    assert c["seed_min"] == 0 and c["seed_max_inclusive"] == 31
    assert c["gate_twirling"] and not c["dynamical_decoupling"]
    assert c["calibration_exec"] == "2026-10-25T23:36:48-06:00"
    text = hw_aggregate.config_table(summary)
    assert "ibm\\_fez" in text and "ibm_fez" not in text.replace("ibm\\_fez", "")
    assert "charged 75 s" in text


@needs_job
def test_the_results_table_marks_declined_and_unphysical_values(summary):
    """Dagger: reported at lambda = 1; double dagger: Mitiq outside [0, 1]."""
    text = hw_aggregate.results_table(summary)
    assert "0.953 [0.890, 1.000] & res." in text
    assert "3.547$^\\ddagger$" in text
    assert "0.257$^\\dagger$ & n.i." in text
    assert "P-n3-M1-17\\_r1" in text


@needs_job
def test_main_and_the_hardware_figures_run_on_a_copy(tmp_path, capsys):
    """End to end: tables, then every hardware figure that has its inputs."""
    data = tmp_path / "data"
    for name in (f"hw_{DATE}", f"hw_dryrun_{DATE}", f"hw_predict_exec_{DATE}",
                 f"hw_alpha_exec_{DATE}"):
        shutil.copytree(DATA / name, data / name)
    tex = tmp_path / "tex"
    assert hw_aggregate.main(["--date", DATE, "--hardware-dir", str(HARDWARE),
                              "--data-dir", str(data), "--tex-dir", str(tex)]) == 0
    assert "alpha_hat = 3.176" in capsys.readouterr().out
    for path in (data / "hw_records.csv", data / "hw_summary.json",
                 tex / "hw_table_results.tex", tex / "hw_table_config.tex"):
        assert path.exists()

    figures = tmp_path / "figures"
    written = make_figures.build_hw(data, figures, make_figures.MAIN_GRID)
    names = {p.stem for p in written}
    # No simulation tables in the copy: the sweep and M = 0 overlays are skipped.
    assert {"fig_hw_example", "fig_hw_signal", "fig_hw_folding", "fig_hw_zne"} <= names


def test_the_hardware_figures_refuse_to_guess_without_a_summary(tmp_path):
    """No hw_summary.json is an instruction to run hw_aggregate.py, not an empty figure."""
    with pytest.raises(FileNotFoundError, match="hw_aggregate"):
        make_figures.build_hw(tmp_path, tmp_path, make_figures.MAIN_GRID)


def test_without_a_manifest_there_is_nothing_to_aggregate(tmp_path, capsys):
    """An empty hardware directory is a message and exit 1."""
    assert hw_aggregate.main(["--hardware-dir", str(tmp_path), "--data-dir", str(tmp_path),
                              "--tex-dir", str(tmp_path)]) == 1
    assert "nothing to aggregate" in capsys.readouterr().out


def test_a_prediction_is_called_earlier_only_when_its_timestamp_is():
    """"before the job ran" is read off the two timestamps, never assumed."""
    predicted = {"a": {"backend": "fake_x2", "calibration_timestamp": "2026-10-27T18:12:25-06:00",
                       "provenance": {"timestamp_utc": "2026-10-28T01:04:37Z"}}}
    where = Path("paper/data/hw_predict_x")
    assert "before the job ran" in hw_aggregate.prediction_note(
        predicted, where, "2026-10-28T01:18:35Z")
    assert "before the job ran" not in hw_aggregate.prediction_note(
        predicted, where, "2026-10-28T01:00:00Z")
    assert "before the job ran" not in hw_aggregate.prediction_note(predicted, where, None)


@needs_later
def test_the_grid_table_sets_the_gentle_grid_beside_the_odd_grid(tmp_path, capsys):
    """Same records, both grids, the prediction made before the job; only the one file."""
    tex = tmp_path / "tex"
    assert hw_aggregate.main(["--grids", LATER, "--grids-prediction", str(LATER_PREDICTION),
                              "--tex-dir", str(tex)]) == 0
    assert [p.name for p in tex.iterdir()] == [hw_aggregate.GRID_TABLE]
    assert "before the job ran" in capsys.readouterr().out
    text = (tex / hw_aggregate.GRID_TABLE).read_text(encoding="utf-8")
    rows = {line.split(" & ")[0]: line for line in text.splitlines() if " & " in line}
    assert "\\{1, 3, 5\\}" in text and "\\{1, 1.25, 1.5\\}" in text
    assert set(rows) - {"", "Record"} == {
        "P-n2-M1-11", "P-n3-M1-17", "P-n3-M2-17", "P-n3-M1-17\\_r1"}
    # Odd grid: not identifiable; gentle grid: resolved, interval holding the ideal.
    assert "n.i. & 235 & 0.15 & 0.821 [0.660, 1.000]" in rows["P-n3-M1-17"]
    assert "id. & 126 & 0.31 & 0.730 [0.633, 0.831] & 0.609 [0.508, 0.719]" in rows[
        "P-n3-M1-17\\_r1"]
    # The M = 2 record fell to random on the day; a negative kept is typeset as a number.
    assert "$-0.01$ & n.i. & 0.806 [0.613, 1.000]" in rows["P-n3-M2-17"]


def test_a_later_job_never_replaces_the_papers_tables(tmp_path):
    """The job of record is the default; any other date writes beside it."""
    assert hw_aggregate.output_suffix(hw_aggregate.JOB_OF_RECORD) == ""
    assert hw_aggregate.output_suffix("2026-10-28") == "_2026-10-28"
    for date in (hw_aggregate.JOB_OF_RECORD, "2026-10-28"):
        (tmp_path / f"manifest_{date}.json").write_text("{}", encoding="utf-8")
    assert hw_aggregate.default_date(tmp_path) == hw_aggregate.JOB_OF_RECORD
    (tmp_path / f"manifest_{hw_aggregate.JOB_OF_RECORD}.json").unlink()
    assert hw_aggregate.default_date(tmp_path) == "2026-10-28"


# ---------------------------------------------------------------------------
# WP7: model vs hardware, survival, drift distance, the later job's model
# ---------------------------------------------------------------------------


def test_the_drift_distance_is_judged_against_shot_noise():
    """Identical histograms are no drift; a moved peak is; the distance is drift_metrics'."""
    from hardware_batch import drift_metrics

    same = {"00": 1000, "01": 1000, "10": 1000, "11": 1096}
    moved = {"00": 1300, "01": 900, "10": 900, "11": 996}
    flat = hw_aggregate.tv_bootstrap(same, dict(same), B=500)
    assert flat["tv"] == 0.0 and flat["tv_p_value"] == pytest.approx(1.0)
    assert flat["tv_null_mean"] > 0.0  # two samples of one distribution are still apart
    shifted = hw_aggregate.tv_bootstrap(same, moved, B=500)
    assert shifted["tv"] == pytest.approx(drift_metrics(same, moved, ["11"])["tv_between"])
    assert shifted["tv_p_value"] < 0.01
    assert shifted["tv_ci"][0] <= shifted["tv"] <= shifted["tv_ci"][1]
    assert shifted == hw_aggregate.tv_bootstrap(same, moved, B=500)  # seeded


def test_counts_are_found_at_a_fractional_realised_scale():
    """The gentle grid keys its counts by the realised scale, not the nominal one."""
    record = {"counts": {"1": {"0": 1}, "1.2535211267605635": {"1": 2}, "3": {"0": 3}}}
    assert hw_aggregate._counts_at(record, 1.2535211267605635) == {"1": 2}
    assert hw_aggregate._counts_at(record, 3.0) == {"0": 3}
    assert hw_aggregate._counts_at(record, 1.25) is None
    assert hw_aggregate._counts_at(None, 1.0) is None


def test_deviation_stats_keep_the_sign_and_skip_what_has_no_signal():
    rows = [
        {"scale": 1.0, "signal": True, "d": -0.2},
        {"scale": 1.0, "signal": True, "d": -0.1},
        {"scale": 3.0, "signal": True, "d": 0.1},
        {"scale": 3.0, "signal": False, "d": 5.0},
        {"scale": 3.0, "signal": True, "d": None},
    ]
    stats = hw_aggregate.deviation_stats(rows, "d", signal_only=True)
    assert stats["all"]["n"] == 3
    assert stats["all"]["mean_signed"] == pytest.approx(-0.2 / 3)
    assert stats["all"]["mean_abs"] == pytest.approx(0.4 / 3)
    assert (stats["all"]["n_negative"], stats["all"]["n_positive"]) == (2, 1)
    assert stats["lambda=1"]["mean_signed"] == pytest.approx(-0.15)
    assert hw_aggregate.deviation_stats(rows, "d", signal_only=False)["lambda=3"]["n"] == 2


@needs_job
def test_the_interpolated_distribution_is_the_cached_one_on_a_grid_point():
    """Between grid points it is interpolated; on one it is that point's counts."""
    manifest = json.loads((HARDWARE / f"manifest_{DATE}.json").read_text(encoding="utf-8"))
    cache = DATA / f"hw_alpha_exec_{DATE}"
    dists = hw_aggregate.alpha_distributions(manifest, cache, 3.0)
    stored = json.loads((cache / "counts_alpha_3.json").read_text(encoding="utf-8"))["counts"]
    for c in manifest["circuits"]:
        counts = stored[int(c["pub_index"])]
        total = sum(counts.values())
        dist = dists[(c["key"], float(c["nominal_scale"]))]
        assert sum(dist.values()) == pytest.approx(1.0)
        for k, v in counts.items():
            assert dist[k] == pytest.approx(v / total, abs=1e-12)
    assert hw_aggregate.alpha_distributions(manifest, cache, None) == {}


@needs_job
def test_every_pub_is_compared_with_both_models(summary):
    """47 PUBs; at alpha = 1 the device is below the model on every lambda = 1 PUB."""
    mvh = summary["model_vs_hardware"]
    assert len(mvh["pubs"]) == 47
    assert mvh["model_alpha"]["source"] == "fit" and mvh["model_alpha"]["in_sample"]
    at_1, at_hat = mvh["p_solution"]["alpha_1"], mvh["p_solution"]["alpha_hat"]
    assert at_1["all"]["n"] == 35  # the PUBs the alpha fit used
    assert at_1["lambda=1"]["n_negative"] == at_1["lambda=1"]["n"]
    assert at_hat["all"]["mean_abs"] < at_1["all"]["mean_abs"] / 5
    # P(solution) at alpha-hat is exactly the fit's interpolation.
    by_key = {r["key"]: r for r in summary["records"]}
    for p in mvh["pubs"]:
        if p["scale"] == 1.0 and p["key"] in by_key and by_key[p["key"]]["model_alpha"]:
            assert p["model_alpha"] == pytest.approx(by_key[p["key"]]["model_alpha"])
    floor = mvh["tv_hardware_model"]["alpha_hat"]
    assert floor["mean_shot_floor"] < floor["mean"] < mvh["tv_hardware_model"]["alpha_1"]["mean"]


@needs_job
def test_the_survival_estimate_scaled_by_alpha_hat_places_the_upper_edge(summary):
    block = summary["survival"]
    assert block["calibration"].endswith(f"calibration_exec_{DATE}.json")
    m1, mhat = block["medians"]["alpha_1"], block["medians"]["alpha_hat"]
    assert mhat["two_qubit"] == pytest.approx(summary["alpha_hat"] * m1["two_qubit"], rel=1e-6)
    assert block["bands"]["measured"]["upper"] == block["bands"]["alpha_hat"]["upper"] == 194
    assert block["bands"]["alpha_1"]["upper"] > 194  # unscaled, it is optimistic
    for r in block["records"]:
        assert r["survival_alpha_hat"] < r["survival_alpha_1"] <= 1.0
        assert len(r["kept_predicted_alpha_1"]) == len(r["scales"])


@needs_later
def test_the_later_jobs_model_columns_are_filled_and_say_from_where():
    """The execution model once fetched, else the plan's; and the prediction at alpha."""
    later = hw_aggregate.build_summary(hw_aggregate.HwInputs(LATER, HARDWARE, DATA))
    fetched = (DATA / f"hw_predict_exec_{LATER}").exists()
    source = "exec" if fetched else "plan"
    assert later["model_source"] == source
    assert later["model_alpha"]["source"] == "prediction"
    assert later["model_alpha"]["alpha"] == pytest.approx(3.176)
    assert later["model_alpha"]["in_sample"] is False
    for r in later["records"]:
        assert r["model"] == r[f"model_{source}"] is not None
        assert r["model_alpha"] is not None
    assert len(later["model_vs_hardware"]["pubs"]) == 14  # fractional scales found
    text = hw_aggregate.results_table(later)
    assert ("in force" if fetched else "plan's calibration") in text
    # No calibration entry changed between plan and execution: same value either way.
    row = next(line for line in text.splitlines() if line.startswith("P-n2-M1-11 &"))
    assert " & 0.859 & " in row


def test_the_console_table_prints_the_model_the_tex_table_uses(capsys):
    """The printed model column is ``model`` (exec or plan), not ``model_exec`` alone."""
    record = {"key": "R", "two_q_count": 71, "p_ideal": 1.0, "model": 0.859,
              "model_exec": None, "hardware": 0.646, "kept_hardware": 0.528,
              "zne_reported": 0.981, "zne_status": "resolved"}
    hw_aggregate.print_records({"records": [record], "alpha_hat": None,
                                "limit_band": None})
    assert " 0.859 " in capsys.readouterr().out


def test_error_class_predictions_are_not_the_jobs_model(tmp_path):
    """A --scale-only prediction never competes with the every-error one."""
    date = "2026-01-01"
    for name in (f"hw_predict_plan_x3.176_{date}", f"hw_predict_plan_x4.05_2q_{date}",
                 f"hw_predict_plan_x32_readout_{date}", f"hw_predict_exec_{date}"):
        (tmp_path / name).mkdir()
    dirs = hw_aggregate.HwInputs(date, tmp_path, tmp_path).alpha_prediction_dirs()
    assert [d.name for d in dirs] == [f"hw_predict_plan_x3.176_{date}"]
