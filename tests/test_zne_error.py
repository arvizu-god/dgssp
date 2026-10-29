"""
Error propagation of the fixed-asymptote exponential (``zne_theory.md`` §2-§4).

What is asserted, and why these thresholds:

* ``exp_fixed_se`` is the estimator's own fit and SE: identical to
  :func:`fit_scalar_series` (the numbers stored in every record).
* Monte Carlo (2000 replicas, seed 0) on ``P(lam) = 1/8 + 0.82 exp(-c lam)``
  for c in {0.5, 1, 2}, grids {1, 1.25, 1.5}, {1, 1.5, 2}, {1, 3, 5}, shots
  4096 and 12 288 (the two hardware jobs): the delta-method SE is within 10 %
  of the Monte Carlo SE on every design whose predicted relative SE
  (SE / signal) is <= 0.2.  That covers 14 of the 18 designs.  The other four
  (c = 2 on {1, 1.25, 1.5} and {1, 1.5, 2} at 4096 shots, relative SE 0.27 to
  0.31; c = 2 on {1, 3, 5}, not identified) are where the linearisation
  breaks: there the formula is *below* the Monte Carlo SE, which is asserted
  as a direction, not a size.  The Monte Carlo SE itself has a relative error
  of about 1.6 % at 2000 replicas.
* The estimator's small-sample bias, from the same Monte Carlo, is recorded
  with ``record_property`` and not asserted.
* ``optimal_step``: ``c * step`` tends to ``OPTIMAL_STEP_LIMIT`` = 1.1089 (the
  root of ``u = 1 + exp(-2u)``), within 5 % at c = 4.  It does **not** tend
  to 1: the 1/c rule of thumb is 13 % short at c = 4 and 11 % short in the
  limit, which is pinned here so the note cannot drift back to it.
* Two scales: the fit is exactly determined and the delta-method SE equals
  the closed form ``(P(0) - a) sqrt((1+k)**2/s1**2 + k**2/s2**2)``,
  ``k = lam1 / (lam2 - lam1)``, to rounding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dgssp.metrics import proportion_sigma
from dgssp.mitigation.zne import (
    EXP_RATE_MAX,
    OPTIMAL_STEP_LIMIT,
    RATE_CAP_RTOL,
    ZNESamplingConfig,
    _linearised_value_se,
    at_rate_cap,
    exp_fixed_monte_carlo,
    exp_fixed_se,
    fit_scalar_series,
    optimal_step,
    plan_scales,
    plan_scales_spacing,
    pointwise_fit_rates,
    pointwise_raw,
    predicted_se,
    rate_cap_value,
    two_point_relative_error,
)

P_UNIFORM = 0.125
SIGNAL = 0.82
RATES = (0.5, 1.0, 2.0)
GRIDS = ((1.0, 1.25, 1.5), (1.0, 1.5, 2.0), (1.0, 3.0, 5.0))
SHOTS = (4096, 12_288)
#: Designs whose predicted relative SE is at most this are in the linear regime.
LINEAR_REGIME = 0.2
CANDIDATES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0)


def _exact_series(scales, rate, shots):
    x = np.asarray(scales, dtype=float)
    y = P_UNIFORM + SIGNAL * np.exp(-rate * x)
    return x.tolist(), y.tolist(), [shots] * x.size


# ---------------------------------------------------------------------------
# exp_fixed_se is the estimator's own SE
# ---------------------------------------------------------------------------


def test_exp_fixed_se_equals_fit_scalar_series():
    rng = np.random.default_rng(3)
    cfg = ZNESamplingConfig(method="exponential", asymptote="p_uniform")
    for rate in RATES:
        x, y, n = _exact_series((1.0, 1.5, 2.0), rate, 8192)
        noisy = (np.asarray(y) + rng.normal(0, 0.003, len(y))).tolist()
        ours = exp_fixed_se(x, noisy, n, P_UNIFORM)
        stored = fit_scalar_series(x, noisy, n, cfg, p_uniform=P_UNIFORM)
        assert ours["value"] == stored["value"]
        assert ours["se"] == stored["value_se"]
        # The vectorised closed form used by the Monte Carlo agrees.
        closed = _linearised_value_se(
            np.asarray(x),
            np.array([ours["amplitude"]]),
            np.array([ours["rate"]]),
            np.asarray(ours["sigma"]),
        )[0]
        assert closed == pytest.approx(ours["se"], rel=1e-9)
        assert ours["se_known_f"] == pytest.approx(
            ours["sigma"][0] * math.exp(ours["rate"] * x[0])
        )
        assert ours["learning_cost"] > 1.0


def test_predicted_se_is_exp_fixed_se_on_expected_data():
    for rate in RATES:
        for grid in GRIDS[:2]:
            x, y, n = _exact_series(grid, rate, 4096)
            fitted = exp_fixed_se(x, y, n, P_UNIFORM)
            assert predicted_se(grid, math.exp(-rate), SIGNAL, P_UNIFORM, 4096) == (
                pytest.approx(fitted["se"], rel=1e-6)
            )


def test_predicted_se_amplitude_cancels_and_no_signal_is_infinite():
    # b cancels in the linearised variance: at equal sigma the SE is the same.
    x, b, c = np.array([1.0, 1.5, 2.0]), np.array([0.3, -0.2, 0.8]), np.full(3, 0.9)
    se = _linearised_value_se(x, b, c, np.full(3, 0.004))
    assert se == pytest.approx(np.full(3, se[0]), rel=1e-12)
    assert predicted_se((1.0, 2.0), 0.0, SIGNAL, P_UNIFORM, 4096) == math.inf
    assert predicted_se((1.0, 2.0), 0.5, 0.0, P_UNIFORM, 4096) == math.inf


# ---------------------------------------------------------------------------
# The two-scale closed form of zne_theory.md section 2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scales", [(1.0, 1.6), (1.0, 3.0), (1.2, 2.5)])
@pytest.mark.parametrize("rate", RATES)
def test_two_scales_match_the_closed_form(scales, rate):
    x, y, n = _exact_series(scales, rate, 12_288)
    fit = exp_fixed_se(x, y, n, P_UNIFORM)
    d = np.asarray(fit["observed"]) - P_UNIFORM
    s = d / np.asarray(fit["sigma"])
    kappa = x[0] / (x[1] - x[0])
    excess = d[0] * (d[0] / d[1]) ** kappa
    closed = excess * math.sqrt((1 + kappa) ** 2 / s[0] ** 2 + kappa**2 / s[1] ** 2)
    # The rate is a golden-section optimum: the value is exact to ~1e-8.
    assert fit["value"] - P_UNIFORM == pytest.approx(excess, rel=1e-6)
    assert fit["se"] == pytest.approx(closed, rel=1e-6)


def test_two_point_relative_error_is_the_equal_sigma_closed_form():
    # sigma_1 = sigma_2 makes s2 = s1 exp(-c step): the section 3 factor.
    rate, step = 1.3, 0.7
    s1 = 50.0
    kappa = 1.0 / step
    rel = math.sqrt((1 + kappa) ** 2 / s1**2 + kappa**2 / (s1 * math.exp(-rate * step)) ** 2)
    assert two_point_relative_error(step, rate) / s1 == pytest.approx(rel, rel=1e-12)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shots", SHOTS)
@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("rate", RATES)
def test_delta_method_matches_monte_carlo(rate, grid, shots, record_property):
    mc = exp_fixed_monte_carlo(grid, math.exp(-rate), SIGNAL, P_UNIFORM, shots, n_replicas=2000)
    every = mc["all"]
    record_property("relative_se_predicted", mc["relative_se_predicted"])
    record_property("ratio_predicted", every["ratio_predicted"])
    record_property("ratio_formula", every["ratio_formula"])
    record_property("bias_mean", every["bias_mean"])
    record_property("bias_mean_se", every["bias_mean_se"])
    record_property("bias_median", every["bias_median"])
    if mc["relative_se_predicted"] <= LINEAR_REGIME:
        assert every["ratio_predicted"] == pytest.approx(1.0, abs=0.10)
        assert every["ratio_formula"] == pytest.approx(1.0, abs=0.10)
    else:
        assert every["ratio_predicted"] < 0.95
        assert every["ratio_formula"] < 0.95


def test_linear_regime_covers_fourteen_of_eighteen_designs():
    inside = [
        (rate, grid, shots)
        for rate in RATES
        for grid in GRIDS
        for shots in SHOTS
        if predicted_se(grid, math.exp(-rate), SIGNAL, P_UNIFORM, shots) / SIGNAL
        <= LINEAR_REGIME
    ]
    assert len(inside) == 14
    assert all(rate < 2.0 or shots == 12_288 for rate, _, shots in inside)


# ---------------------------------------------------------------------------
# Optimal spacing
# ---------------------------------------------------------------------------


def test_optimal_step_limit_is_the_root():
    u = OPTIMAL_STEP_LIMIT
    assert u == pytest.approx(1.0 + math.exp(-2.0 * u), abs=1e-14)
    assert u == pytest.approx(1.1089, abs=1e-4)


def test_optimal_step_tends_to_the_limit_not_to_one_over_c():
    at4 = optimal_step(4.0)
    assert at4["c_step"] / OPTIMAL_STEP_LIMIT - 1.0 < 0.05
    # The 1/c rule of thumb is more than 10 % short, at c = 4 and in the limit.
    assert at4["step"] / at4["inverse_rate"] - 1.0 > 0.10
    at64 = optimal_step(64.0)
    assert at64["c_step"] == pytest.approx(OPTIMAL_STEP_LIMIT, rel=2e-3)
    # c * step decreases monotonically toward the limit.
    c_steps = [optimal_step(c)["c_step"] for c in (0.5, 1.0, 2.0, 4.0, 16.0)]
    assert all(a > b for a, b in zip(c_steps, c_steps[1:], strict=False))


def test_optimal_step_is_the_minimum():
    for rate in (0.7, 1.1, 1.8):
        best = optimal_step(rate)
        for factor in (0.9, 0.99, 1.01, 1.1):
            assert two_point_relative_error(best["step"] * factor, rate) > best["relative_error"]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("survival", [0.6, 0.35, 0.16])
def test_plan_scales_spacing_beats_plan_scales_on_predicted_se(survival):
    shots = 8192
    prefix = plan_scales(survival, SIGNAL, shots, CANDIDATES, 3.0, p_uniform=P_UNIFORM)
    evenly = plan_scales(
        survival, SIGNAL, shots, CANDIDATES, 3.0, p_uniform=P_UNIFORM, n_points=3
    )
    spaced = plan_scales_spacing(
        survival, SIGNAL, shots, CANDIDATES, 3.0, p_uniform=P_UNIFORM, n_points=3
    )
    assert spaced[0] == 1.0 and len(spaced) == len(evenly)
    assert set(spaced) <= set(prefix)
    assert predicted_se(spaced, survival, SIGNAL, P_UNIFORM, shots) <= predicted_se(
        evenly, survival, SIGNAL, P_UNIFORM, shots
    )


def test_plan_scales_spacing_without_signal_or_with_worst_case_sigma():
    assert plan_scales_spacing(0.5, 0.0, 8192, CANDIDATES, 3.0, p_uniform=0.25) == [1.0]
    grid = plan_scales_spacing(0.5, 0.5, 8192, CANDIDATES, 3.0)
    assert grid[0] == 1.0 and len(grid) == 3
    whole = plan_scales_spacing(0.5, 0.5, 8192, CANDIDATES, 3.0, n_points=None)
    assert whole == plan_scales(0.5, 0.5, 8192, CANDIDATES, 3.0)
    with pytest.raises(ValueError):
        plan_scales_spacing(0.5, 0.5, 8192, CANDIDATES, 3.0, n_points=1)


# ---------------------------------------------------------------------------
# The rate cap
# ---------------------------------------------------------------------------


def test_unidentified_fit_returns_the_cap_value():
    # Only lam = 1 above the asymptote: chi2 is flat in c, the fit stops at the cap.
    x = [1.0, 3.0, 5.0]
    y = [0.30, 0.124, 0.1251]
    fit = exp_fixed_se(x, y, [4096] * 3, P_UNIFORM)
    assert fit["at_cap"] is True
    assert fit["rate"] >= EXP_RATE_MAX * (1 - RATE_CAP_RTOL)
    assert fit["value"] == pytest.approx(rate_cap_value(x, y, P_UNIFORM), rel=RATE_CAP_RTOL)
    assert fit["cap_value"] == pytest.approx(
        P_UNIFORM + (0.30 - P_UNIFORM) * math.exp(EXP_RATE_MAX), rel=1e-12
    )


def test_pointwise_fit_rates_mark_the_capped_strings():
    rng = np.random.default_rng(0)
    counts = {
        lam: dict(zip(["00", "01", "10", "11"], rng.multinomial(4096, [0.25] * 4).tolist(),
                      strict=True))
        for lam in (1.0, 3.0, 5.0)
    }
    cfg = ZNESamplingConfig(method="exponential", asymptote="p_uniform")
    rates = pointwise_fit_rates(counts, cfg)
    raw = pointwise_raw(counts, cfg)
    assert set(rates) == set(raw)
    for key, rate in rates.items():
        if at_rate_cap(rate):
            p1 = counts[1.0][key] / 4096
            assert raw[key] == pytest.approx(
                rate_cap_value([1.0], [p1], 0.25), rel=RATE_CAP_RTOL
            )
    with pytest.raises(ValueError):
        pointwise_fit_rates(counts, ZNESamplingConfig(method="linear"))


def test_pointwise_richardson_is_linear_and_sums_to_one():
    rng = np.random.default_rng(1)
    keys = [format(i, "03b") for i in range(8)]
    counts = {
        lam: dict(zip(keys, rng.multinomial(2048, rng.dirichlet(np.ones(8))).tolist(),
                      strict=True))
        for lam in (1.0, 1.7486338797814207, 3.0)
    }
    raw = pointwise_raw(counts, ZNESamplingConfig(method="richardson"))
    assert sum(raw.values()) == pytest.approx(1.0, abs=1e-12)


def test_proportion_sigma_is_the_weight_used():
    # Guard: exp_fixed_se's sigma is the library's binomial sigma, not a copy.
    fit = exp_fixed_se([1.0, 2.0], [0.4, 0.2], [1000, 1000], P_UNIFORM)
    assert fit["sigma"] == [proportion_sigma(0.4, 1000), proportion_sigma(0.2, 1000)]
