"""
Tests for the WP5 campaign scripts: ``sim_campaign``, ``aggregate``,
``make_figures``.

These cover the decisions that would silently corrupt a table rather than
crash: how a job is packed, which scale grid an instance gets, when the ideal
stage is allowed to stop the campaign, and which statistic a cell is
summarised by.
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
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

aggregate = pytest.importorskip("aggregate")
make_figures = pytest.importorskip("make_figures")
sim_campaign = pytest.importorskip("sim_campaign")


# ---------------------------------------------------------------------------
# sim_campaign: layout, resume, chunking
# ---------------------------------------------------------------------------


def test_stage_directories_name_what_produced_them(tmp_path):
    """The ideal stage carries no noise tag; it does not depend on a device."""
    assert sim_campaign.stage_dir(tmp_path, "ideal", "fake", None).name == "sim_ideal"
    assert sim_campaign.stage_dir(tmp_path, "noisy", "fake", None).name == "sim_noisy_fake"
    assert (
        sim_campaign.stage_dir(tmp_path, "zne", "live", "planned").name
        == "sim_zne_live_planned"
    )


def test_resume_skips_only_what_is_on_disk(tmp_path):
    """A finished instance is skipped; an unfinished one is not."""
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    todo, done, duplicated = sim_campaign.pending(tmp_path, ["a", "b", "c"])
    assert (todo, done, duplicated) == (["b", "c"], ["a"], [])


def test_a_duplicate_record_is_reported(tmp_path):
    """
    ``save_record`` appends ``_001`` rather than overwriting, so a suffixed
    file means two runs wrote into this directory at once.
    """
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "a_001.json").write_text("{}", encoding="utf-8")
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    todo, done, duplicated = sim_campaign.pending(tmp_path, ["a", "b"])
    assert todo == [] and sorted(done) == ["a", "b"]
    assert duplicated == ["a"]


class _FakeItem:
    """Just enough of a PreparedInstance for the packer."""

    def __init__(self, name: str, scales: int) -> None:
        self.name = name
        self._scales = scales

    def n_circuits(self, *, include_mitiq: bool = False) -> int:
        return self._scales + (self._scales - 1 if include_mitiq else 0)


def test_jobs_are_packed_by_circuits_not_instances():
    """The chunk limit counts circuits, which is what a job actually holds."""
    items = [_FakeItem(f"i{k}", 3) for k in range(5)]
    chunks = sim_campaign.chunk_by_circuits(items, 7, include_mitiq=False)
    assert [len(c) for c in chunks] == [2, 2, 1]
    assert all(
        sum(i.n_circuits() for i in chunk) <= 7 for chunk in chunks
    )

    # Mitiq adds one circuit per scale above 1, so fewer instances fit.
    with_mitiq = sim_campaign.chunk_by_circuits(items, 7, include_mitiq=True)
    assert [len(c) for c in with_mitiq] == [1, 1, 1, 1, 1]


def test_one_instance_is_never_split_across_jobs():
    """A fold set must stay in one job or its scales stop being comparable."""
    chunks = sim_campaign.chunk_by_circuits([_FakeItem("big", 9)], 4, include_mitiq=False)
    assert len(chunks) == 1 and len(chunks[0]) == 1


# ---------------------------------------------------------------------------
# sim_campaign: scale grids
# ---------------------------------------------------------------------------


def _row(**overrides):
    """A minimal family.json-like record."""
    base = {
        "name": "X-n3-M1-00",
        "family": "P",
        "n_items": 3,
        "M": 1,
        "solution_bitstrings": ["111"],
        "r_opt": 2,
        "items": [1, 1, 1],
        "target": 3,
        "m_class": "1",
    }
    base.update(overrides)
    return base


def test_fixed_grids_ignore_the_instance():
    """The negative control and the model-comparison grid are fixed by design."""
    assert sim_campaign.scales_for(_row(), None, "odd", 8192) == (
        list(sim_campaign.ODD_SCALES),
        "odd",
    )
    assert sim_campaign.scales_for(_row(), None, "odd7", 8192) == (
        list(sim_campaign.ODD7_SCALES),
        "odd7",
    )


def test_planned_grid_comes_from_the_recorded_survival():
    """A healthy instance is planned; the plan is a prefix starting at 1."""
    candidate = {"survival": 0.75, "p_ideal": 0.94, "p_uniform": 0.125}
    scales, source = sim_campaign.scales_for(_row(), candidate, "planned", 8192)
    assert source == "planned"
    assert scales[0] == 1.0
    assert len(scales) >= 2
    assert scales == sorted(scales)


def test_a_signal_free_instance_falls_back_to_the_pathology_grid():
    """
    ``M = 0`` has zero signal, so ``plan_scales`` returns one scale -- with
    which nothing can be extrapolated *or* diagnosed.  The floor exists so the
    pathology measurement still happens, and the record says so.
    """
    candidate = {"survival": 0.4, "p_ideal": 0.0, "p_uniform": 0.0}
    scales, source = sim_campaign.scales_for(_row(M=0), candidate, "planned", 8192)
    assert source == "planned_floor"
    assert scales == list(sim_campaign.PATHOLOGY_GRID)
    assert len(scales) >= 3


def test_planning_without_a_candidate_row_is_refused():
    """Silently substituting a default grid would corrupt the comparison."""
    with pytest.raises(ValueError, match="survival"):
        sim_campaign.scales_for(_row(), None, "planned", 8192)


def test_r_opt_is_checked_against_the_library():
    """A file whose r_opt drifted from the library stops the campaign."""
    sim_campaign.check_r_opt([_row()])
    with pytest.raises(ValueError, match="r_opt"):
        sim_campaign.check_r_opt([_row(r_opt=5)])


# ---------------------------------------------------------------------------
# sim_campaign: the ideal-stage acceptance gate
# ---------------------------------------------------------------------------


def test_the_threshold_is_corrected_for_how_often_it_is_applied():
    """One test at 3 sigma; 347 tests need 4.5 to be equally safe."""
    assert sim_campaign.corrected_sigma(3.0, 1) == 3.0
    assert 4.4 < sim_campaign.corrected_sigma(3.0, 347) < 4.6
    assert sim_campaign.corrected_sigma(4.0, 347) > sim_campaign.corrected_sigma(3.0, 347)


def _deviation(sigma: float, check: str = "recorded", name: str = "i"):
    return {
        "name": name,
        "check": check,
        "observed": 0.5,
        "expected": 0.5,
        "sigma": sigma,
        "nominal": 3.0 if check == "recorded" else 4.0,
    }


def test_a_handful_of_nominal_exceedances_does_not_stop_a_good_run():
    """Over 347 comparisons, chance alone produces about one 3-sigma miss."""
    deviations = [_deviation(1.0, name=f"i{k}") for k in range(345)]
    deviations += [_deviation(3.1, name="a"), _deviation(3.05, name="b")]
    passed, report = sim_campaign.judge_ideal(deviations, strict=False)
    assert passed
    assert any("2 over the nominal" in line for line in report)


def test_one_enormous_deviation_does_stop_the_run():
    """A real error shows up past the corrected threshold, and chance does not."""
    deviations = [_deviation(1.0, name=f"i{k}") for k in range(346)]
    deviations.append(_deviation(9.0, name="broken"))
    passed, report = sim_campaign.judge_ideal(deviations, strict=False)
    assert not passed
    assert any("that is not chance" in line for line in report)


def test_too_many_mild_exceedances_also_stop_the_run():
    """A systematic bias is many small misses, not one large one."""
    deviations = [_deviation(1.0, name=f"i{k}") for k in range(300)]
    deviations += [_deviation(3.2, name=f"x{k}") for k in range(47)]
    passed, report = sim_campaign.judge_ideal(deviations, strict=False)
    assert not passed
    assert any("systematic bias" in line for line in report)


def test_strict_mode_restores_the_literal_per_instance_gate():
    """The uncorrected rule is still available, and it does fire."""
    deviations = [_deviation(1.0, name=f"i{k}") for k in range(346)]
    deviations.append(_deviation(3.1, name="a"))
    assert sim_campaign.judge_ideal(deviations, strict=False)[0] is True
    assert sim_campaign.judge_ideal(deviations, strict=True)[0] is False


def test_a_certain_outcome_can_still_be_compared():
    """``p = 1`` (n = 2, M = 1) has zero binomial variance; the floor saves it."""
    assert sim_campaign._binomial_sigma(1.0, 10_000) > 0.0
    assert sim_campaign._binomial_sigma(0.5, 10_000, 20_000) > sim_campaign._binomial_sigma(
        0.5, 10_000
    ) * 0.9


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def test_the_median_is_used_where_the_mean_would_measure_the_rate_bound():
    """An unidentified exponential is unbounded; one outlier proves it."""
    values = [0.5, 0.52, 0.48, 3018.0]
    mean = aggregate.summarise(values, use_median=False)
    median = aggregate.summarise(values, use_median=True)
    assert mean["statistic"] == "mean" and mean["value"] > 700
    assert median["statistic"] == "median" and 0.4 < median["value"] < 0.6
    assert median["se"] is not None and median["se"] < 1.0


def test_pooled_m_classes_separate_feasible_from_infeasible():
    """``P(solution)`` is identically zero when M = 0, so it is pooled apart."""
    rows = [
        {"family": "P", "n_items": 3, "m_class": "0", "grid": "odd", "M": 0},
        {"family": "P", "n_items": 3, "m_class": "1", "grid": "odd", "M": 1},
        {"family": "P", "n_items": 3, "m_class": "2", "grid": "odd", "M": 2},
    ]
    assert len(aggregate._cell_members(rows, "P", 3, "all", "odd")) == 3
    assert len(aggregate._cell_members(rows, "P", 3, "feasible", "odd")) == 2
    assert len(aggregate._cell_members(rows, "P", 3, "0", "odd")) == 1


def test_model_comparison_counts_what_it_excluded():
    """A model accurate only where it is identified is more selective, not better."""
    rows = [
        {
            "grid": "odd7",
            "M": 1,
            "ideal": 1.0,
            "fit_linear": 0.5,
            "fit_linear_identifiable": True,
            "fit_quadratic": 0.9,
            "fit_quadratic_identifiable": False,
            "fit_richardson": float("nan"),
            "fit_richardson_identifiable": True,
            "fit_exponential": 0.99,
            "fit_exponential_identifiable": True,
        }
    ]
    out = {row["method"]: row for row in aggregate.build_model_comparison(rows, "odd7")}
    assert out["linear"]["n_used"] == 1
    assert out["linear"]["median_abs_error"] == pytest.approx(0.5)
    assert out["quadratic"]["n_used"] == 0
    assert out["quadratic"]["excluded_not_identifiable"] == 1
    assert out["richardson"]["excluded_not_finite"] == 1
    assert out["exponential"]["median_abs_error"] == pytest.approx(0.01)


def test_notes_placeholders_mark_what_has_not_been_run(tmp_path):
    """An un-run stage reads as ``??``, never as a measured zero."""
    template = tmp_path / "t.md"
    template.write_text(
        "res {{per_grid.odd.fraction_resolved}} miss {{per_grid.nope.x}}",
        encoding="utf-8",
    )
    filled = aggregate.fill_notes(
        template, {"per_grid": {"odd": {"fraction_resolved": 0.155}}}
    )
    assert "0.155" in filled
    assert "??" in filled


def test_aggregation_of_an_empty_directory_says_so(tmp_path, capsys):
    """Nothing to aggregate is a message, not a traceback."""
    assert aggregate.main(["--data-dir", str(tmp_path), "--tex-dir", str(tmp_path)]) == 1
    assert "Run sim_campaign.py first" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# make_figures
# ---------------------------------------------------------------------------


def _write_tables(data_dir: Path) -> None:
    """A two-cell table, enough to exercise every figure path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for n_items in (2, 3):
        for series in (*aggregate.PSOL_SERIES, *aggregate.TV_SERIES, *aggregate.BIAS_SERIES):
            for m_class in ("all", "feasible"):
                cells.append(
                    {
                        "family": "P",
                        "n_items": n_items,
                        "m_class": m_class,
                        "grid": "planned",
                        "series": series,
                        "n_instances": 4,
                        "statistic": "mean",
                        "value": 0.4,
                        "se": 0.02,
                        "n": 4,
                        "min": 0.3,
                        "max": 0.5,
                        "fraction_resolved": 0.25 if n_items == 3 else 0.9,
                        "fraction_identifiable": 0.5,
                    }
                )
    aggregate.write_csv(data_dir / "sim_cells.csv", cells)
    aggregate.write_csv(
        data_dir / "sim_pathology.csv",
        [
            {
                "name": "P-n2-M0-00",
                "tv_zne_lambda1": 0.02,
                "tv_pointwise_clip": 0.25,
                "tv_pointwise_simplex": 0.24,
            }
        ],
    )
    aggregate.write_csv(data_dir / "sim_per_instance.csv", [])


def test_every_figure_builds_from_the_tables_alone(tmp_path):
    """No backend, no circuit: the figures are a function of paper/data."""
    data_dir, figures_dir = tmp_path / "data", tmp_path / "figures"
    _write_tables(data_dir)
    written = make_figures.build_sim(data_dir, figures_dir, "planned")
    names = {path.stem for path in written}
    assert {"fig_sim_psol", "fig_sim_dtv", "fig_sim_grid"} <= names
    assert all(path.exists() for path in written)


def test_a_sweep_figure_appears_once_a_sweep_exists(tmp_path):
    """The sweep panel is driven by the archived sweep records."""
    data_dir, figures_dir = tmp_path / "data", tmp_path / "figures"
    _write_tables(data_dir)
    sweep_dir = data_dir / "sim_sweep_fake"
    sweep_dir.mkdir(parents=True)
    payload = {
        "instance": {"items": [1, 1, 1], "target": 3, "name": "P-n3-M1-17"},
        "M": 1,
        "p_uniform": 0.125,
        "rows": [
            {
                "iterations": r,
                "p_sol": 0.5,
                "p_sol_ci": [0.48, 0.52],
                "analytic": 0.6,
                "zne": None,
            }
            for r in (1, 2, 3)
        ],
    }
    (sweep_dir / "P-n3-M1-17_noisy.json").write_text(json.dumps(payload), encoding="utf-8")

    written = make_figures.build_sim(data_dir, figures_dir, "planned")
    assert "fig_sim_sweep" in {path.stem for path in written}


def test_figures_refuse_to_guess_when_nothing_was_aggregated(tmp_path):
    """A missing table is an error, not an empty plot."""
    with pytest.raises(FileNotFoundError, match="aggregate.py"):
        make_figures.build_sim(tmp_path, tmp_path / "figures", "planned")
