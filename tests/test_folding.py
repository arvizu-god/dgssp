"""
Tests for ISA-level gate folding.

Folding must be noise-scaling only: on a noiseless simulator the folded
circuit has to reproduce the unfolded distribution exactly, since G G-dagger G
equals G.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from dgssp import (
    DGConfig,
    DGSSPSolver,
    SubsetSumInstance,
    build_ideal_aer_backend,
    counts_to_probs,
    fold_transpiled,
    sample_counts,
)
from dgssp.mitigation.zne import (
    ZNESamplingConfig,
    extrapolate_distribution,
    zne_fit_single_value,
)


@pytest.fixture
def transpiled():
    """A small D-G circuit transpiled for the ideal simulator."""
    instance = SubsetSumInstance(items=[1, 2, 3], target=5)
    qc = DGSSPSolver(DGConfig(iterations=1)).build_circuit(instance)
    backend = build_ideal_aer_backend(seed_simulator=99)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    return pm.run(qc), backend


@pytest.mark.parametrize("scale", [3, 5, 7])
def test_folding_preserves_the_ideal_distribution(transpiled, scale):
    """Folding is a no-op without noise."""
    tqc, backend = transpiled
    folded = fold_transpiled(tqc, scale)

    base = counts_to_probs(
        sample_counts(backend, tqc, shots=4096, seed_simulator=99, run_log=None)[0]
    )
    scaled = counts_to_probs(
        sample_counts(backend, folded, shots=4096, seed_simulator=99, run_log=None)[0]
    )

    for bit in set(base) | set(scaled):
        assert base.get(bit, 0.0) == pytest.approx(scaled.get(bit, 0.0), abs=0.05)


@pytest.mark.parametrize("scale", [3, 5])
def test_folding_grows_the_gate_count(transpiled, scale):
    """A scale factor of s multiplies the foldable gate count by roughly s."""
    tqc, _ = transpiled
    folded = fold_transpiled(tqc, scale)
    assert folded.size() > tqc.size()
    assert folded.num_qubits == tqc.num_qubits


def test_scale_one_is_identity(transpiled):
    """Scale 1 returns an equivalent circuit."""
    tqc, _ = transpiled
    assert fold_transpiled(tqc, 1).size() == tqc.size()


@pytest.mark.parametrize("bad", [0, 2, 4, -1, -3])
def test_even_and_nonpositive_scales_rejected(transpiled, bad):
    """Only odd positive scale factors are meaningful for G G-dagger folding."""
    tqc, _ = transpiled
    with pytest.raises(ValueError):
        fold_transpiled(tqc, bad)


def test_measurements_are_not_folded(transpiled):
    """Measurements are copied through exactly once."""
    tqc, _ = transpiled
    folded = fold_transpiled(tqc, 5)

    def n_measure(circuit):
        return sum(1 for i in circuit.data if i.operation.name == "measure")

    assert n_measure(folded) == n_measure(tqc)


def test_linear_extrapolation_recovers_the_intercept():
    """A perfectly linear series extrapolates back to its intercept."""
    zero, _, _, _ = zne_fit_single_value("linear", [1, 3, 5], [0.8, 0.6, 0.4])
    assert zero == pytest.approx(0.9, abs=1e-6)


def test_quadratic_extrapolation_recovers_the_intercept():
    """A perfectly quadratic series extrapolates back to its intercept."""
    scales = [1, 3, 5, 7]
    values = [2.0 * s**2 - 3.0 * s + 0.5 for s in scales]
    zero, _, _, _ = zne_fit_single_value("quadratic", scales, values)
    assert zero == pytest.approx(0.5, abs=1e-6)


def test_two_scale_linear_fit_is_exact_richardson():
    """
    With two scales the linear fit is the closed-form 2-point Richardson
    estimator ``(3*y1 - y3)/2``, not an approximation of it.
    """
    zero, _, _, _ = zne_fit_single_value("linear", [1, 3], [0.20, 0.14])
    assert zero == pytest.approx((3 * 0.20 - 0.14) / 2, abs=1e-12)


def test_exactly_determined_fits_emit_no_warning():
    """
    A fit with as many scales as parameters has no residual degrees of freedom,
    so ``curve_fit`` cannot estimate a covariance and warns about it. That
    covariance is never used, so the warning was pure noise on every ZNE run.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        zne_fit_single_value("linear", [1, 3], [0.20, 0.14])
        zne_fit_single_value("quadratic", [1, 3, 5], [0.20, 0.15, 0.10])
        zne_fit_single_value("exponential", [1, 3, 5], [0.20, 0.15, 0.13])


def test_polynomial_fits_match_nonlinear_least_squares():
    """
    The closed-form solution for the models that are linear in their parameters
    agrees with the nonlinear optimiser it replaced, so no recorded number
    moves because of the change.
    """
    from scipy.optimize import curve_fit

    from dgssp.mitigation.zne import _linear_model, _quadratic_model

    rng = np.random.default_rng(20260906)
    for _ in range(50):
        values = rng.random(5) * 0.5
        scales = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            linear_popt, _ = curve_fit(_linear_model, scales, values)
            quadratic_popt, _ = curve_fit(_quadratic_model, scales, values)

        assert zne_fit_single_value("linear", scales, values)[0] == pytest.approx(
            float(_linear_model(0.0, *linear_popt)), abs=1e-7
        )
        assert zne_fit_single_value("quadratic", scales, values)[0] == pytest.approx(
            float(_quadratic_model(0.0, *quadratic_popt)), abs=1e-7
        )


def test_unknown_fit_method_rejected():
    """Only the three documented models are accepted."""
    with pytest.raises(ValueError):
        zne_fit_single_value("cubic", [1, 3, 5], [0.8, 0.6, 0.4])


def test_too_few_points_for_model_rejected():
    """A quadratic fit needs at least three scales."""
    with pytest.raises(ValueError):
        zne_fit_single_value("quadratic", [1, 3], [0.8, 0.6])


def test_extrapolated_distribution_is_normalised():
    """Clipping and renormalisation produce a valid distribution."""
    counts_per_scale = {
        1: {"01": 700, "10": 300},
        3: {"01": 600, "10": 400},
        5: {"01": 500, "10": 500},
    }
    cfg = ZNESamplingConfig(scales=[1, 3, 5], method="linear")
    mitigated = extrapolate_distribution(counts_per_scale, cfg)

    assert sum(mitigated.values()) == pytest.approx(1.0)
    assert all(v >= 0.0 for v in mitigated.values())
    # Extrapolation should push "01" above its scale-1 value.
    assert mitigated["01"] > 0.7


def test_zero_shots_at_a_scale_is_an_error():
    """An empty counts dict cannot be normalised and must not pass silently."""
    cfg = ZNESamplingConfig(scales=[1, 3], method="linear")
    with pytest.raises(ValueError):
        extrapolate_distribution({1: {"0": 10}, 3: {}}, cfg)


# ---------------------------------------------------------------------------
# ISA rebase: folding inserts inverse gates that need not be native
# ---------------------------------------------------------------------------


@pytest.fixture
def noisy_pair():
    """A fake IBM device plus the Aer noise model derived from it."""
    pytest.importorskip("qiskit_ibm_runtime")
    try:
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    except Exception:  # pragma: no cover
        pytest.skip("fake provider data unavailable")
    device = FakeManilaV2()
    return device, AerSimulator.from_backend(device)


def test_folding_leaves_the_native_basis(noisy_pair):
    """Folding sx produces sxdg, which IBM devices cannot execute."""
    from dgssp import DGConfig, DGSSPSolver, SubsetSumInstance
    from dgssp.mitigation.zne import rebase_to_backend

    device, _ = noisy_pair
    qc = DGSSPSolver(DGConfig(iterations=1)).build_circuit(
        SubsetSumInstance(items=[1, 2], target=3)
    )
    pm = generate_preset_pass_manager(
        backend=device, optimization_level=1, seed_transpiler=7
    )
    tqc = pm.run(qc)

    # Barriers are directives rather than instructions, so they never appear
    # in a Target's operation names.
    basis = set(device.target.operation_names) | {"barrier"}

    def ops(circuit):
        return set(circuit.count_ops())

    assert ops(tqc) <= basis, "transpiled circuit should start native"

    folded = fold_transpiled(tqc, 3)
    assert not ops(folded) <= basis, "expected non-native inverses such as sxdg"

    rebased = rebase_to_backend(folded, device)
    assert ops(rebased) <= basis, "rebase failed to restore the native basis"


def test_rebased_folds_actually_execute(noisy_pair):
    """The end-to-end guard: an unrebased fold raises 'unknown instruction'."""
    from dgssp import (
        DGConfig,
        DGSSPSolver,
        SubsetSumInstance,
        ZNESamplingConfig,
        zne_mitigated_distribution,
    )

    device, noisy = noisy_pair
    qc = DGSSPSolver(DGConfig(iterations=1)).build_circuit(
        SubsetSumInstance(items=[1, 2], target=3)
    )
    pm = generate_preset_pass_manager(
        backend=device, optimization_level=1, seed_transpiler=7
    )
    tqc = pm.run(qc)

    cfg = ZNESamplingConfig(
        scales=[1, 3], shots_per_scale=512, method="linear", seed_simulator=1234
    )
    mitigated = zne_mitigated_distribution(qc, noisy, cfg, transpiled=tqc)

    assert mitigated
    assert sum(mitigated.values()) == pytest.approx(1.0)


def test_rebase_is_a_noop_without_a_target(transpiled):
    """A backend with no target leaves the circuit untouched."""
    from dgssp.mitigation.zne import rebase_to_backend

    class NoTarget:
        target = None

    tqc, _ = transpiled
    assert rebase_to_backend(tqc, NoTarget()) is tqc


def test_optimized_executor_end_to_end(noisy_pair):
    """The optimized path yields both unmitigated and mitigated probabilities."""
    from dgssp import BatchConfig, SubsetSumInstance, ZNESamplingConfig, run_batch

    _, noisy = noisy_pair
    result = run_batch(
        [SubsetSumInstance(items=[1, 2], target=3, name="opt")],
        BatchConfig(
            executor="optimized",
            backend=noisy,
            zne_config=ZNESamplingConfig(
                scales=[1, 3],
                shots_per_scale=512,
                method="linear",
                seed_min=0,
                seed_max=2,
                seed_simulator=1234,
            ),
            shots=512,
            seed_simulator=1234,
        ),
    ).results[0]

    assert 0.0 <= result.solution_probability <= 1.0
    assert result.mitigated_solution_probability is not None
    assert result.mitigated_distribution
    assert result.error_metrics is not None
    assert result.error_metrics.total_error > 0.0
    assert result.two_qubit_depth is not None
