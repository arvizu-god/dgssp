"""
Reporting bounds of the scalar extrapolation (Step 5, part 1).

What is asserted, and why:

* ``value_bounds=None`` is the old estimator, bit for bit, and a box wider
  than any fitted amplitude changes nothing (to golden-section noise).
* ``"bounded"`` equals ``"clip"`` on the point: ``P(0) = a + b`` depends on
  ``b`` alone and ``chi2`` is quadratic in ``b``, so the constrained optimum
  is the unconstrained one when that is feasible and lies on the bound
  otherwise.  Only the rate is re-fitted.
* Under ``"bounded"`` every bootstrap replicate is the clip of the unbounded
  one on the same resample; the intervals differ only by percentile
  interpolation at the bound.
* The resolved / identifiable flags never depend on the option.
* ``"clip"`` never moves the point farther from a truth inside the range.
"""

from __future__ import annotations

import numpy as np
import pytest

from dgssp.mitigation.zne import (
    ZNESamplingConfig,
    _exp_fixed_many,
    fit_scalar_series,
    physical_bounds,
    report_bounded,
    report_clip,
    report_flag,
    zne_bootstrap,
)

P_UNIFORM = 0.25
SCALES = (1.0, 1.25, 1.5)
SHOTS = 4096


def _series(p0: float, rate: float) -> list[float]:
    """Exact depolarising curve with ``P(0) = p0``."""
    return [P_UNIFORM + (p0 - P_UNIFORM) * np.exp(-rate * lam) for lam in SCALES]


def _counts(values: list[float]) -> dict[float, dict[str, int]]:
    """One-bit counts with ``P("1") = value`` at each scale."""
    out = {}
    for lam, v in zip(SCALES, values, strict=True):
        k = int(round(v * SHOTS))
        out[lam] = {"1": k, "0": SHOTS - k}
    return out


def _observable(counts):
    return counts.get("1", 0) / sum(counts.values())


CFG = ZNESamplingConfig(method="exponential", asymptote="p_uniform", n_bootstrap=300)


def test_no_bounds_is_the_old_estimator_and_a_wide_box_changes_nothing():
    rng = np.random.default_rng(0)
    x = np.array(SCALES)
    Y = P_UNIFORM + 0.6 * np.exp(-np.outer(rng.uniform(0.3, 2.0, 50), x))
    Y += rng.normal(0.0, 0.004, Y.shape)
    W = np.full((1, x.size), 1.0 / 0.004**2)
    b0, c0 = _exp_fixed_many(x, Y, W, P_UNIFORM)
    b1, c1 = _exp_fixed_many(x, Y, W, P_UNIFORM, amp_bounds=None)
    assert np.array_equal(b0, b1) and np.array_equal(c0, c1)
    b2, _ = _exp_fixed_many(x, Y, W, P_UNIFORM, amp_bounds=(-1e6, 1e6))
    assert np.allclose(b2, b0, atol=1e-6)


@pytest.mark.parametrize("p0", [1.06, 0.97, 0.6])
def test_bounded_point_is_the_clipped_point(p0):
    values = _series(p0, 1.2)
    shots = [SHOTS] * len(SCALES)
    free = fit_scalar_series(SCALES, values, shots, CFG, p_uniform=P_UNIFORM)
    bounds = physical_bounds(P_UNIFORM, 1)
    bounded = fit_scalar_series(
        SCALES, values, shots, CFG, p_uniform=P_UNIFORM, value_bounds=bounds
    )
    assert bounded["value"] == pytest.approx(min(max(free["value"], bounds[0]), bounds[1]), abs=1e-7)
    assert bounded["value_at_bound"] is (free["value"] > 1.0)
    assert (bounded["value_se"] is None) is (free["value"] > 1.0)
    for key in ("resolved", "identifiable", "reported_source", "resolved_scales"):
        assert bounded[key] == free[key]


def test_bounded_bootstrap_replicates_are_clipped_replicates():
    counts = _counts(_series(1.02, 1.0))
    free = zne_bootstrap(counts, CFG, observable_fn=_observable, p_uniform=P_UNIFORM)
    rep = report_bounded(counts, _observable, CFG, p_uniform=P_UNIFORM)
    assert rep["value"] == pytest.approx(1.0) and rep["value_at_bound"]
    assert P_UNIFORM <= rep["ci"][0] <= rep["ci"][1] <= 1.0
    assert free["hi"] > 1.0
    assert rep["ci"][0] == pytest.approx(min(free["lo"], 1.0), abs=1e-6)
    assert rep["B"] == 300 and rep["seed"] == 0
    assert rep["resolved"] is free["fit"]["resolved"]


def test_physical_bounds():
    assert physical_bounds(0.25, 1) == (0.25, 1.0)
    assert physical_bounds(0.25, -1) == (0.0, 0.25)
    assert physical_bounds(0.25, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        physical_bounds(0.25, 2)


def test_flag_and_clip():
    bounds = (0.125, 1.0)
    flag = report_flag(1.3, [0.84, 2.31], bounds)
    assert flag["value"] == 1.3 and flag["ci"] == [0.84, 2.31]
    assert flag["out_of_range"] and flag["ci_out_of_range"]
    clip = report_clip(1.3, [0.84, 2.31], bounds)
    assert clip["value"] == 1.0 and clip["ci"] == [0.84, 1.0]
    assert clip["out_of_range"] and clip["ci_out_of_range"]
    inside = report_clip(0.8, [0.66, 0.95], bounds)
    assert inside["value"] == 0.8 and not inside["out_of_range"] and not inside["ci_out_of_range"]


@pytest.mark.parametrize("value", [-0.2, 0.05, 0.5, 1.0, 1.4, 7.0])
@pytest.mark.parametrize("truth", [0.125, 0.6, 1.0])
def test_clip_never_moves_away_from_a_truth_in_range(value, truth):
    clipped = report_clip(value, [value, value], (0.125, 1.0))["value"]
    assert abs(clipped - truth) <= abs(value - truth)


def test_value_bounds_needs_the_fixed_asymptote_exponential():
    linear = ZNESamplingConfig(method="linear")
    with pytest.raises(ValueError, match="fixed-asymptote"):
        fit_scalar_series(SCALES, _series(0.9, 1.0), [SHOTS] * 3, linear, value_bounds=(0.0, 1.0))
    free = ZNESamplingConfig(method="exponential", asymptote=None)
    with pytest.raises(ValueError, match="fixed asymptote"):
        report_bounded(_counts(_series(0.9, 1.0)), _observable, free, p_uniform=P_UNIFORM)


# ---------------------------------------------------------------------------
# Part 2: the option in force and the aggregation path
# ---------------------------------------------------------------------------


def test_report_extrapolation_dispatches_and_defaults_to_clip():
    from dgssp.mitigation.zne import REPORT_BOUNDS, report_extrapolation

    assert REPORT_BOUNDS == "clip"
    default = report_extrapolation(1.3, [0.84, 2.31], 0.125, 0.95)
    assert default["report_bounds"] == "clip"
    assert default["value"] == 1.0 and default["ci"] == [0.84, 1.0]
    flag = report_extrapolation(1.3, [0.84, 2.31], 0.125, 0.95, option="flag")
    assert flag["value"] == 1.3 and flag["out_of_range"]
    with pytest.raises(ValueError, match="Unknown report option"):
        report_extrapolation(1.3, [0.84, 2.31], 0.125, 0.95, option="trim")
    with pytest.raises(ValueError, match="needs the counts"):
        report_extrapolation(1.3, [0.84, 2.31], 0.125, 0.95, option="bounded")


def test_bounded_through_the_dispatcher_equals_clip_on_the_point():
    from dgssp.mitigation.zne import report_extrapolation

    counts = _counts(_series(1.02, 1.0))
    free = zne_bootstrap(counts, CFG, observable_fn=_observable, p_uniform=P_UNIFORM)
    rep = report_extrapolation(
        free["point"], [free["lo"], free["hi"]], P_UNIFORM, None, option="bounded",
        counts_per_scale=counts, observable_fn=_observable, cfg=CFG,
    )
    assert rep["report_bounds"] == "bounded" and rep["out_of_range"]
    assert rep["value"] == pytest.approx(1.0) and rep["ci"][1] <= 1.0


def _stored_block(p0: float) -> dict:
    """A record-like scalar block, as sim_campaign stores it."""
    values = _series(p0, 1.2)
    fit = fit_scalar_series(SCALES, values, [SHOTS] * 3, CFG, p_uniform=P_UNIFORM)
    boot = zne_bootstrap(_counts(values), CFG, observable_fn=_observable, p_uniform=P_UNIFORM)
    return {**fit, "applicable": True, "ci": [boot["lo"], boot["hi"]], "B": 300,
            "bootstrap_seed": 0}


@pytest.mark.parametrize("p0", [1.05, 0.8])
def test_aggregate_rederive_flags_applies_the_option_to_extrapolations_only(p0):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper" / "scripts"))
    aggregate = pytest.importorskip("aggregate")

    block = _stored_block(p0)
    flag = aggregate.rederive_flags(block, 1.0, report_bounds="flag")
    clip = aggregate.rederive_flags(block, 1.0)
    assert clip["report_bounds"] == "clip" and flag["report_bounds"] == "flag"
    for key in ("resolved", "identifiable", "reported_source", "n_informative_scales"):
        assert clip[key] == flag[key]
    assert clip["reported_source"] == "extrapolation"
    assert flag["reported_value"] == block["value"]
    assert clip["reported_value"] == min(block["value"], 1.0)
    assert clip["reported_ci"][1] == min(block["ci"][1], 1.0)
    assert clip["out_of_range"] is (block["value"] > 1.0)
    # An unresolved fit reports the lambda = 1 observation, never clipped.
    flat = {**block, "observed": [P_UNIFORM] * 3, "value_at_min_scale": P_UNIFORM - 0.01}
    declined = aggregate.rederive_flags(flat, 1.0)
    assert declined["reported_source"] == "min_scale"
    assert declined["reported_value"] == P_UNIFORM - 0.01
    assert declined["reported_ci"] is None and declined["out_of_range"] is None
    with pytest.raises(ValueError, match="needs the record"):
        aggregate.rederive_flags(block, 1.0, report_bounds="bounded")
