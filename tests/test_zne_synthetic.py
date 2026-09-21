"""
The synthetic depolarising study (WP4, item 5a): no circuits, seconds to run.

Counts are drawn binomially from

    P(lam) = S**lam * P_ideal + (1 - S**lam) * P_uniform

at the subset's survival range S in {0.75, 0.40, 0.17}, on *realised* scales
(the rational two-qubit ratios partial folding produces on the instance's own
gate count), for a fixed list of 40 seeds.

What is asserted, and why these thresholds:

* exponential with the known asymptote P_uniform: truth inside the 95%
  bootstrap interval in >= 35/40 seeds on the pre-registered grid.  A correct
  95% interval fails this about 1.4% of the time; >= 38/40 would fail a
  correct estimator about one time in three.
* linear and Richardson: biased under exponential decay *by construction*.
  The Monte-Carlo mean must match the analytic value of the estimator on the
  exact curve to within 3 SE -- the bias is a reported result, not a bug.
* resolvability: S = 0.17 at odd scales [1, 3, 5] is flagged unresolved; the
  plan_scales grid at the same shots is resolved.
"""

from __future__ import annotations

import numpy as np
import pytest

from dgssp.metrics import solution_probability
from dgssp.mitigation.zne import (
    ZNESamplingConfig,
    expected_extrapolation,
    extrapolate_scalar,
    plan_scales,
    realised_scale,
    synthetic_depolarising_counts,
    zne_bootstrap,
)

SEEDS = list(range(20261020, 20261060))
SHOTS = 4000

#: (S, P_ideal, P_uniform, N_2q) from paper/instances/hardware_subset.json:
#: P-n2-M1-11, P-n3-M1-17 and P-n4-M2-02 -- S rounded to the study's grid.
CASES = [
    (0.75, 1.000, 0.250, 76),
    (0.40, 0.942, 0.125, 241),
    (0.17, 0.945, 0.125, 465),
]

#: Candidate scales for the pre-registration (plan_scales), thinned to 3.
CANDIDATES = (1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5)


def _observable(counts):
    return solution_probability(counts, ["1"])


def _grid(nominal, n_2q):
    return [float(realised_scale(lam, n_2q)) for lam in nominal]


def _planned(s, p_ideal, p_u):
    return plan_scales(s, p_ideal - p_u, SHOTS, CANDIDATES, 3, p_uniform=p_u, n_points=3)


def _study(s, p_ideal, p_u, scales, cfg, B=500):
    points, covered, resolved = [], 0, 0
    for seed in SEEDS:
        counts = synthetic_depolarising_counts(
            s, scales, p_ideal, p_u, SHOTS, np.random.default_rng(seed)
        )
        boot = zne_bootstrap(
            counts, cfg, observable_fn=_observable, p_uniform=p_u, B=B, seed=seed
        )
        points.append(boot["point"])
        covered += boot["lo"] <= p_ideal <= boot["hi"]
        resolved += bool(boot["fit"]["resolved"])
    return np.array(points), covered, resolved


@pytest.mark.parametrize(("s", "p_ideal", "p_u", "n_2q"), CASES)
def test_exponential_with_known_asymptote_covers_the_truth(s, p_ideal, p_u, n_2q):
    """>= 35/40 seeds on the pre-registered grid."""
    scales = _grid(_planned(s, p_ideal, p_u), n_2q)
    cfg = ZNESamplingConfig(method="exponential", asymptote="p_uniform")
    _, covered, _ = _study(s, p_ideal, p_u, scales, cfg)
    assert covered >= 35, (s, scales, covered)


@pytest.mark.parametrize("method", ["linear", "richardson"])
@pytest.mark.parametrize("grid", ["odd", "planned"])
@pytest.mark.parametrize(("s", "p_ideal", "p_u", "n_2q"), CASES)
def test_polynomial_bias_matches_its_analytic_value(method, grid, s, p_ideal, p_u, n_2q):
    """Monte-Carlo mean within 3 SE of the estimator applied to the exact curve."""
    nominal = [1, 3, 5] if grid == "odd" else _planned(s, p_ideal, p_u)
    scales = _grid(nominal, n_2q)
    cfg = ZNESamplingConfig(method=method)
    points, _, _ = _study(s, p_ideal, p_u, scales, cfg, B=50)
    analytic = expected_extrapolation(cfg, scales, s, p_ideal, p_u, SHOTS)
    se = points.std(ddof=1) / np.sqrt(points.size)
    assert abs(points.mean() - analytic) <= 3 * se, (points.mean(), analytic, se)
    # And the bias is real: under exponential decay it is negative (toward P_uniform).
    assert analytic < p_ideal


def test_resolvability_flags_odd_scales_and_passes_the_planned_grid():
    """S = 0.17: [1, 3, 5] is unresolved; plan_scales' grid at the same shots is resolved."""
    s, p_ideal, p_u, n_2q = CASES[2]
    planned = _planned(s, p_ideal, p_u)
    assert planned == [1.0, 1.5, 2.0]

    cfg = ZNESamplingConfig(method="exponential", asymptote="p_uniform")
    _, _, resolved_odd = _study(s, p_ideal, p_u, _grid([1, 3, 5], n_2q), cfg, B=50)
    _, _, resolved_plan = _study(s, p_ideal, p_u, _grid(planned, n_2q), cfg, B=50)
    assert resolved_odd <= 2, resolved_odd
    assert resolved_plan >= 36, resolved_plan


def test_odd_scales_do_not_identify_the_exponential_at_n_equals_4():
    """The design, not the implementation, is what fails at S = 0.17.

    Odd-integer folding leaves one informative scale, so (amplitude, rate) are
    not jointly identified and the extrapolation is unbounded; the planned
    grid identifies every seed.  This is the diagnosis behind the divergent
    cell in paper/data/zne_synthetic.json.
    """
    s, p_ideal, p_u, n_2q = CASES[2]
    cfg = ZNESamplingConfig(method="exponential", asymptote="p_uniform")

    def identified(nominal):
        scales = _grid(nominal, n_2q)
        out = 0
        for seed in SEEDS:
            counts = synthetic_depolarising_counts(
                s, scales, p_ideal, p_u, SHOTS, np.random.default_rng(seed)
            )
            fit = extrapolate_scalar(counts, _observable, cfg, p_uniform=p_u)
            out += bool(fit["identifiable"])
        return out

    assert identified([1, 3, 5]) <= 12
    assert identified(_planned(s, p_ideal, p_u)) >= 38
