"""
Tests for the ZNE estimators on exact and synthetic data (no circuits).

Covers the scalar fits (weighted least squares, Richardson, exponential with a
known asymptote), the simplex projection, the scale planner and the pathology
diagnostic.  The Monte-Carlo bias / coverage / resolvability study is in
``test_zne_synthetic.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from dgssp.metrics import total_variation, uniform_distribution
from dgssp.mitigation.zne import (
    ZNESamplingConfig,
    clip_renormalise,
    depolarised_probability,
    extrapolate_distribution,
    extrapolate_scalar,
    fit_scalar_series,
    informative_scales_mask,
    is_resolved,
    pathology_diagnostic,
    plan_scales,
    project_to_simplex,
    resolved_scales_mask,
    richardson_weights,
    simplex_renormalise,
    zne_bootstrap,
)

# ---------------------------------------------------------------------------
# Scalar fits
# ---------------------------------------------------------------------------


def test_richardson_weights_sum_to_one_and_are_the_known_ones():
    """Nodes 1, 3, 5: L(0) = (15/8, -5/4, 3/8)."""
    w = richardson_weights([1, 3, 5])
    assert w == pytest.approx([15 / 8, -5 / 4, 3 / 8])
    assert w.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("scales", [[1, 3, 5], [1.0, 1.4989, 2.0], [1, 1.5, 2, 3]])
def test_richardson_is_exact_on_its_polynomial_class(scales):
    """Degree len-1 polynomials are recovered exactly; dof = 0, chi2_red None."""
    rng = np.random.default_rng(1)
    coeffs = rng.normal(size=len(scales))
    y = np.polyval(coeffs, scales)
    cfg = ZNESamplingConfig(method="richardson")
    fit = fit_scalar_series(scales, np.clip(y, 0, 1).tolist(), [1000] * len(scales), cfg)
    if np.all((y >= 0) & (y <= 1)):
        assert fit["value"] == pytest.approx(coeffs[-1], abs=1e-9)
    assert fit["dof"] == 0
    assert fit["chi2_red"] is None
    assert fit["value_se"] is not None


def test_weighted_linear_fit_recovers_an_exact_line_with_covariance():
    """y = 0.5 - 0.05 x at every scale -> intercept 0.5, residuals 0, chi2 0."""
    scales = [1.0, 2.0, 3.0, 4.0]
    y = [0.5 - 0.05 * s for s in scales]
    fit = fit_scalar_series(scales, y, [4000] * 4, ZNESamplingConfig(method="linear"))
    assert fit["value"] == pytest.approx(0.5)
    assert fit["params"] == pytest.approx([-0.05, 0.5])
    assert fit["dof"] == 2
    assert fit["chi2_red"] == pytest.approx(0.0, abs=1e-18)
    assert np.array(fit["cov"]).shape == (2, 2)
    assert fit["value_se"] == pytest.approx(np.sqrt(fit["cov"][1][1]))


def test_exponential_with_known_asymptote_is_exact_on_the_depolarising_curve():
    """P(lam) = S^lam P_ideal + (1 - S^lam) P_u is recovered with a = P_u."""
    s, p_ideal, p_u = 0.4, 0.942, 0.125
    scales = [1.0, 1.5, 2.0, 3.0]
    y = [depolarised_probability(s, lam, p_ideal, p_u) for lam in scales]
    cfg = ZNESamplingConfig(method="exponential", asymptote="p_uniform")
    fit = fit_scalar_series(scales, y, [4000] * 4, cfg, p_uniform=p_u)
    assert fit["value"] == pytest.approx(p_ideal, abs=1e-6)
    assert fit["params"][1] == pytest.approx(-np.log(s), abs=1e-5)
    assert fit["asymptote"] == p_u
    assert fit["dof"] == 2
    assert fit["rate_at_bound"] is False


def test_float_asymptote_and_p_uniform_agree():
    """asymptote=0.125 and asymptote='p_uniform' with p_uniform=0.125 are the same fit."""
    y = [0.5, 0.3, 0.2]
    a = fit_scalar_series(
        [1, 2, 3], y, [1000] * 3, ZNESamplingConfig(method="exponential", asymptote=0.125)
    )
    b = fit_scalar_series(
        [1, 2, 3],
        y,
        [1000] * 3,
        ZNESamplingConfig(method="exponential", asymptote="p_uniform"),
        p_uniform=0.125,
    )
    assert a["value"] == pytest.approx(b["value"])


def test_p_uniform_asymptote_without_p_uniform_is_an_error():
    """The per-instance asymptote cannot be guessed."""
    cfg = ZNESamplingConfig(method="exponential", asymptote="p_uniform")
    with pytest.raises(ValueError):
        fit_scalar_series([1, 2, 3], [0.5, 0.3, 0.2], [1000] * 3, cfg)


def test_all_null_data_gives_the_asymptote_and_is_unresolved():
    """An M = 0 instance: P_sol = 0 at every scale, asymptote 0 -> value 0, unresolved."""
    counts = {1.0: {"00": 500, "01": 500}, 2.0: {"00": 500, "01": 500}, 3.0: {"00": 1000}}
    cfg = ZNESamplingConfig(method="exponential", asymptote="p_uniform")
    fit = extrapolate_scalar(counts, lambda c: 0.0, cfg, p_uniform=0.0)
    assert fit["value"] == 0.0
    assert fit["resolved"] is False
    assert fit["reported_source"] == "min_scale"


def test_unknown_names_are_rejected_by_the_config():
    """Typos fail at construction, not deep inside a batch."""
    with pytest.raises(ValueError):
        ZNESamplingConfig(method="cubic")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ZNESamplingConfig(mode="both")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ZNESamplingConfig(renorm="softmax")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ZNESamplingConfig(asymptote="uniform")  # type: ignore[arg-type]


def test_resolvability_flag_semantics():
    """Every scale must clear 2 SE; an exact-null observation never passes."""
    assert is_resolved([0.5, 0.4], [4000, 4000], 0.125) is True
    assert is_resolved([0.5, 0.126], [4000, 4000], 0.125) is False
    assert is_resolved([0.0, 0.0], [4000, 4000], 0.0) is False
    assert is_resolved([0.5], [4000], None) is None


# ---------------------------------------------------------------------------
# Simplex projection (Duchi et al. 2008)
# ---------------------------------------------------------------------------


def _is_distribution(p: np.ndarray) -> bool:
    return bool(np.all(p >= 0) and np.isclose(p.sum(), 1.0))


def test_simplex_projection_is_a_distribution_for_random_inputs():
    """Random, all-negative and single-spike inputs all land on the simplex."""
    rng = np.random.default_rng(2008)
    for _ in range(500):
        v = rng.normal(scale=rng.uniform(0.01, 2.0), size=rng.integers(1, 33))
        assert _is_distribution(project_to_simplex(v))
    assert _is_distribution(project_to_simplex(-np.abs(rng.normal(size=16))))
    spike = np.zeros(16)
    spike[5] = 3.7
    out = project_to_simplex(spike)
    assert _is_distribution(out) and out[5] == pytest.approx(1.0)


def test_simplex_projection_is_idempotent_on_the_simplex():
    """A distribution projects to itself."""
    rng = np.random.default_rng(3)
    for _ in range(100):
        p = rng.dirichlet(np.ones(12))
        assert project_to_simplex(p) == pytest.approx(p, abs=1e-12)


def test_simplex_projection_is_never_farther_in_l2_than_clip_renormalise():
    """Provable: clip+renorm is a feasible point, the projection is the closest one."""
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(1000):
        v = rng.normal(loc=1 / 16, scale=0.05, size=16)
        clipped = np.clip(v, 0, None)
        if clipped.sum() == 0:
            continue
        clipped /= clipped.sum()
        projected = project_to_simplex(v)
        assert np.linalg.norm(projected - v) <= np.linalg.norm(clipped - v) + 1e-12
        checked += 1
    assert checked > 900


def test_row_wise_projection_matches_one_row_at_a_time():
    """The vectorised path used by the bootstrap agrees with the 1-D path."""
    rng = np.random.default_rng(5)
    rows = rng.normal(size=(50, 8))
    stacked = project_to_simplex(rows)
    for row, out in zip(rows, stacked, strict=True):
        assert out == pytest.approx(project_to_simplex(row))


def test_dict_renormalisers():
    """The bitstring-keyed wrappers keep keys and produce distributions."""
    raw = {"00": 0.7, "01": -0.2, "10": 0.4, "11": 0.1}
    for out in (clip_renormalise(raw), simplex_renormalise(raw)):
        assert set(out) == set(raw)
        assert sum(out.values()) == pytest.approx(1.0)
        assert min(out.values()) >= 0.0


# ---------------------------------------------------------------------------
# Scale planning
# ---------------------------------------------------------------------------


def test_plan_scales_is_a_prefix_starting_at_one():
    """Admissible scales are the prefix whose predicted signal clears k sigma."""
    grid = plan_scales(0.17, 0.82, 4000, [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 5], 3, p_uniform=0.125)
    assert grid[0] == 1.0
    assert grid == sorted(grid)
    assert max(grid) < 3  # at lam = 3 the signal is below one sigma
    for lam in grid:
        p = 0.125 + 0.17**lam * 0.82
        assert 0.17**lam * 0.82 >= 3 * np.sqrt(p * (1 - p) / 4000)


def test_plan_scales_without_signal_returns_only_one():
    """No signal (M = 0, null control): nothing to extrapolate."""
    assert plan_scales(0.6, 0.0, 4000, [1, 3, 5]) == [1.0]


def test_plan_scales_thinning_keeps_the_ends():
    """n_points picks evenly from the admissible prefix, keeping 1 and the last."""
    full = plan_scales(0.75, 0.75, 4000, np.arange(1, 7, 0.25), 3, p_uniform=0.25)
    thin = plan_scales(0.75, 0.75, 4000, np.arange(1, 7, 0.25), 3, p_uniform=0.25, n_points=3)
    assert len(thin) == 3 and thin[0] == full[0] and thin[-1] == full[-1]


# ---------------------------------------------------------------------------
# Pointwise estimator and the pathology diagnostic
# ---------------------------------------------------------------------------


def _uniform_counts_with_bias(rng, shots=4000, scales=(1.0, 2.0, 3.0)):
    """An M = 0 instance: ideal uniform, noise adds a relaxation bias toward 00."""
    out = {}
    uniform = np.full(4, 0.25)
    bias = np.array([1.0, 0.0, 0.0, 0.0])
    for lam in scales:
        w = 1 - 0.8**lam
        p = (1 - 0.3 * w) * uniform + 0.3 * w * bias
        draws = rng.multinomial(shots, p)
        out[lam] = {format(i, "02b"): int(c) for i, c in enumerate(draws)}
    return out


def test_pathology_diagnostic_fields_and_directions():
    """On a uniform reference, mitigation cannot beat raw by more than shot noise."""
    counts = _uniform_counts_with_bias(np.random.default_rng(11))
    cfg = ZNESamplingConfig(method="linear", mode="pointwise")
    diag = pathology_diagnostic(counts, cfg, uniform_distribution(2))
    assert diag["reference_is_uniform"] is True
    assert 0.0 <= diag["fraction_positive_slope"] <= 1.0
    assert diag["clip_mass_removed"] >= 0.0
    assert diag["simplex_mass_moved"] >= 0.0
    for key in ("tv_raw_min_scale", "tv_pointwise_clip", "tv_pointwise_simplex"):
        assert 0.0 <= diag[key] <= 1.0
    # The bias grows with lam, so "00" has a positive slope and the others negative.
    assert diag["fraction_positive_slope"] == pytest.approx(0.25)


def test_simplex_mode_matches_the_explicit_projection():
    """extrapolate_distribution(renorm='simplex') is the projection of the raw estimate."""
    counts = _uniform_counts_with_bias(np.random.default_rng(12))
    cfg = ZNESamplingConfig(method="linear", mode="pointwise", renorm="simplex")
    diag = pathology_diagnostic(counts, cfg, uniform_distribution(2))
    out = extrapolate_distribution(counts, cfg)
    assert out == pytest.approx(simplex_renormalise(diag["raw_distribution"]))


def test_pointwise_bootstrap_intervals_bracket_the_point_estimate():
    """Per-bitstring intervals, and the D_TV interval to the reference."""
    counts = _uniform_counts_with_bias(np.random.default_rng(13))
    cfg = ZNESamplingConfig(method="linear", mode="pointwise", renorm="clip", n_bootstrap=300)
    ref = uniform_distribution(2)
    out = zne_bootstrap(counts, cfg, reference=ref, solution_bitstrings=["11"])
    assert set(out["intervals"]) == set(ref)
    for key, (lo, hi) in out["intervals"].items():
        assert lo <= hi
        assert lo - 1e-9 <= out["distribution"][key] <= hi + 0.05
    assert out["tv"]["point"] == pytest.approx(total_variation(out["distribution"], ref))
    assert out["tv"]["lo"] <= out["tv"]["hi"]
    assert out["solution_probability"]["lo"] <= out["solution_probability"]["hi"]


def test_scalar_bootstrap_requires_an_observable():
    """Scalar mode needs to know what it is extrapolating."""
    with pytest.raises(ValueError):
        zne_bootstrap({1.0: {"0": 10}, 3.0: {"0": 10}}, ZNESamplingConfig(mode="scalar"))


# ---------------------------------------------------------------------------
# Identifiability: the exponential is unbounded when too few scales resolve
# ---------------------------------------------------------------------------

#: A series in the n = 4 / odd-folding regime: only lambda = 1 carries signal,
#: and the higher scales fluctuate *below* the asymptote.  Taken from seed
#: 20261023 of the synthetic study (S = 0.17, P_uniform = 0.125, 4000 shots).
_DEGENERATE = ([1.0, 3.0, 5.0], [0.2620, 0.1155, 0.1113], 0.125)


def test_informative_scales_mask_is_signed():
    """A point k sigma on the *wrong* side of the null is not information."""
    scales, y, null = _DEGENERATE
    two_sided = resolved_scales_mask(y, [4000] * 3, null)
    signed = informative_scales_mask(y, [4000] * 3, null)
    # lambda = 5 sits ~2.8 sigma below the asymptote: "distinguishable", but it
    # says nothing about a signal decaying from above.
    assert two_sided == [True, False, True]
    assert signed == [True, False, False]


def test_unidentified_exponential_is_flagged_and_never_reported():
    """One informative scale cannot identify (amplitude, rate)."""
    scales, y, null = _DEGENERATE
    cfg = ZNESamplingConfig(method="exponential", asymptote="p_uniform")
    fit = fit_scalar_series(scales, y, [4000] * 3, cfg, p_uniform=null)

    assert fit["n_informative_scales"] == 1
    assert fit["n_params"] == 2
    assert fit["identifiable"] is False
    assert fit["resolved"] is False
    assert fit["rate_at_bound"] is True
    # The raw value is preserved (diagnostic) but is not what gets reported.
    assert fit["value"] > 100
    assert fit["reported_value"] == pytest.approx(y[0])
    assert fit["reported_source"] == "min_scale"


def test_unidentified_exponential_is_unbounded_in_the_rate_cap():
    """chi2 is flat in the rate: the 'estimate' just tracks exp_rate_max."""
    scales, y, null = _DEGENERATE
    values = []
    for cap in (5.0, 10.0, 20.0):
        cfg = ZNESamplingConfig(method="exponential", asymptote="p_uniform", exp_rate_max=cap)
        fit = fit_scalar_series(scales, y, [4000] * 3, cfg, p_uniform=null)
        values.append(fit["value"])
        assert fit["identifiable"] is False
        assert fit["exp_rate_max"] == cap
    # Roughly exp(cap) growth: the infimum of chi2 is at rate -> infinity.
    assert values[0] < values[1] < values[2]
    assert values[2] > 1e5


def test_identified_exponential_ignores_the_rate_cap():
    """On a resolvable design the bound never binds, so it changes nothing."""
    s, p_ideal, p_u = 0.4, 0.942, 0.125
    scales = [1.0, 1.5, 2.0]
    y = [depolarised_probability(s, lam, p_ideal, p_u) for lam in scales]
    fits = [
        fit_scalar_series(
            scales,
            y,
            [4000] * 3,
            ZNESamplingConfig(method="exponential", asymptote="p_uniform", exp_rate_max=cap),
            p_uniform=p_u,
        )
        for cap in (5.0, 10.0, 20.0)
    ]
    assert all(f["identifiable"] is True and f["rate_at_bound"] is False for f in fits)
    # The bound changes the rate-search grid, so agreement is numerical, not bitwise.
    assert fits[0]["value"] == pytest.approx(fits[1]["value"], abs=1e-6)
    assert fits[1]["value"] == pytest.approx(fits[2]["value"], abs=1e-6)


def test_polynomial_fits_stay_bounded_on_the_degenerate_series():
    """Only the exponential is unbounded there; the linear fits are not."""
    scales, y, null = _DEGENERATE
    for method in ("linear", "richardson"):
        fit = fit_scalar_series(
            scales, y, [4000] * 3, ZNESamplingConfig(method=method), p_uniform=null
        )
        assert abs(fit["value"]) < 1.0
        assert fit["exp_rate_max"] is None
