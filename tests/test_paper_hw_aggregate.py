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
    assert "0.953 [0.890, 1.026] & res." in text
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
