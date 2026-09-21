"""
Tests for the two WP5 library additions on top of the ZNE path.

1. ``analyse_zne_item(..., methods=...)`` fits several extrapolations to the
   *same* counts, so model comparison costs no shots and cannot be confounded
   by re-sampling.
2. ``prepare_from_records`` turns the frozen instance files into a transpiled
   item, reusing the recorded ``best_seed`` -- the campaign scripts hold no
   register, iteration or layout logic of their own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dgssp.experiments import PreparedInstance, prepare_from_records
from dgssp.mitigation.zne import (
    ZNE_METHODS,
    ZNEItem,
    ZNESamplingConfig,
    analyse_zne_item,
    run_zne_batch,
    transpile_once_seeded,
)
from dgssp.solvers import DGConfig, DGSSPSolver, optimal_iterations
from dgssp.transpilation import two_qubit_count

REPO = Path(__file__).resolve().parents[1]
FAMILY_FILE = REPO / "paper" / "instances" / "family.json"
SUBSET_FILE = REPO / "paper" / "instances" / "hardware_subset.json"

NAME = "P-n2-M1-11"


@pytest.fixture(scope="module")
def rows():
    """The family record and the hardware-subset candidate row of one instance."""
    family = {
        r["name"]: r
        for r in json.loads(FAMILY_FILE.read_text(encoding="utf-8"))["instances"]
    }
    subset = json.loads(SUBSET_FILE.read_text(encoding="utf-8"))
    candidates = {r["name"]: r for r in subset["candidates"]}
    return family[NAME], candidates[NAME], subset["backend"]


@pytest.fixture(scope="module")
def device_pair():
    """fake_marrakesh and the Aer noise model derived from it."""
    pytest.importorskip("qiskit_ibm_runtime")
    try:
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    except Exception:  # pragma: no cover - provider data varies by release
        pytest.skip("FakeMarrakesh unavailable")
    device = FakeMarrakesh()
    return device, AerSimulator.from_backend(device)


# ---------------------------------------------------------------------------
# prepare_from_records
# ---------------------------------------------------------------------------


def test_prepared_reuses_the_recorded_seed_and_circuit(rows, device_pair):
    """The recorded best_seed reproduces the ranked circuit exactly."""
    family_row, candidate_row, subset_backend = rows
    device, _ = device_pair
    assert subset_backend == device.name

    cfg = ZNESamplingConfig(scales=[1, 1.5, 2], optimization_level=3)
    prepared = prepare_from_records(family_row, candidate_row, device, cfg)

    assert isinstance(prepared, PreparedInstance)
    assert prepared.best_seed == candidate_row["best_seed"]
    assert prepared.seed_source == "hardware_subset"
    assert prepared.iterations == family_row["r_opt"]
    assert prepared.solution_bitstrings == family_row["solution_bitstrings"]
    assert prepared.p_uniform == pytest.approx(
        family_row["M"] / 2 ** family_row["n_items"]
    )
    # The ranked size is reproduced, which is what makes the seed worth storing.
    assert two_qubit_count(prepared.transpiled) == candidate_row["two_q_count"]

    reference = DGSSPSolver(DGConfig(iterations=family_row["r_opt"])).build_circuit(
        prepared.instance
    )
    tqc, _ = transpile_once_seeded(
        reference,
        device,
        ZNESamplingConfig(
            seed_min=candidate_row["best_seed"],
            seed_max=candidate_row["best_seed"] + 1,
            optimization_level=3,
        ),
    )
    assert prepared.transpiled == tqc


def test_prepared_without_a_candidate_row_searches_the_layout(rows, device_pair):
    """On any other device the recorded seed means nothing and is not used."""
    family_row, _, _ = rows
    device, _ = device_pair
    cfg = ZNESamplingConfig(seed_min=0, seed_max=4, optimization_level=3)
    prepared = prepare_from_records(family_row, None, device, cfg)
    assert prepared.seed_source == "sabre_sweep"
    assert 0 <= prepared.best_seed < 4

    fixed = prepare_from_records(
        family_row,
        None,
        device,
        ZNESamplingConfig(seed_min=7, seed_max=8, optimization_level=3),
    )
    assert (fixed.seed_source, fixed.best_seed) == ("fixed_seed", 7)


def test_prepared_scales_always_include_the_unmitigated_point(rows, device_pair):
    """Scale 1 is the observation, not an option."""
    family_row, candidate_row, _ = rows
    device, _ = device_pair
    prepared = prepare_from_records(
        family_row,
        candidate_row,
        device,
        ZNESamplingConfig(optimization_level=3),
        scales=[1.5, 2.0],
    )
    assert prepared.scales[0] == 1.0
    assert prepared.n_circuits() == 3
    assert prepared.n_circuits(include_mitiq=True) == 5
    assert prepared.zne_item().scales == prepared.scales


def test_prepared_rejects_a_record_that_disagrees_with_the_library(rows, device_pair):
    """r_opt and the ground truth are checked against the library, not trusted."""
    family_row, candidate_row, _ = rows
    device, _ = device_pair
    cfg = ZNESamplingConfig(optimization_level=3)

    bad_r = dict(family_row)
    bad_r["r_opt"] = family_row["r_opt"] + 1
    with pytest.raises(ValueError, match="optimal_iterations"):
        prepare_from_records(bad_r, candidate_row, device, cfg)

    bad_solutions = dict(family_row)
    bad_solutions["solution_bitstrings"] = ["00"]
    with pytest.raises(ValueError, match="DP solver"):
        prepare_from_records(bad_solutions, candidate_row, device, cfg)

    assert family_row["r_opt"] == optimal_iterations(
        family_row["n_items"], max(1, family_row["M"])
    )


# ---------------------------------------------------------------------------
# analyse_zne_item(methods=...)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def item_run(rows, device_pair):
    """One instance's counts at three scales, on the Aer noise model."""
    family_row, candidate_row, _ = rows
    device, noisy = device_pair
    cfg = ZNESamplingConfig(
        scales=[1, 1.5, 2],
        shots_per_scale=2000,
        optimization_level=3,
        seed_simulator=1234,
        n_bootstrap=50,
    )
    prepared = prepare_from_records(family_row, candidate_row, device, cfg)
    batch = run_zne_batch(
        [ZNEItem(name=prepared.name, circuit=prepared.transpiled)],
        noisy,
        cfg,
        include_mitiq=False,
    )
    return batch.items[0], cfg, prepared


def test_every_method_is_fitted_from_the_same_counts(item_run):
    """Four models, one set of counts: the observations are shared exactly."""
    run, cfg, prepared = item_run
    analysis = analyse_zne_item(
        run,
        cfg,
        solution_bitstrings=prepared.solution_bitstrings,
        p_uniform=prepared.p_uniform,
        methods=ZNE_METHODS,
    )
    fits = analysis["scalars"]
    assert set(fits) == set(ZNE_METHODS)
    observed = fits["linear"]["observed"]
    for name, fit in fits.items():
        assert fit["method"] == name
        assert fit["observed"] == observed
        assert fit["scales"] == fits["linear"]["scales"]
        assert fit["shots"] == fits["linear"]["shots"]
        assert fit["reported_source"] in ("extrapolation", "min_scale")


def test_the_reported_fit_is_still_the_configured_one(item_run):
    """`zne_fit` does not move when more methods are fitted beside it."""
    run, cfg, prepared = item_run
    kwargs = {
        "solution_bitstrings": prepared.solution_bitstrings,
        "p_uniform": prepared.p_uniform,
    }
    one = analyse_zne_item(run, cfg, **kwargs)
    many = analyse_zne_item(run, cfg, methods=ZNE_METHODS, **kwargs)

    assert one["scalar"] == many["scalar"]
    assert one["scalars"] == {cfg.method: one["scalar"]}
    assert many["scalar"] == many["scalars"][cfg.method]
    assert one["pointwise"] == many["pointwise"]


def test_an_unknown_method_is_refused(item_run):
    """A typo must not silently reduce the model comparison."""
    run, cfg, prepared = item_run
    with pytest.raises(ValueError, match="Unknown method"):
        analyse_zne_item(
            run,
            cfg,
            solution_bitstrings=prepared.solution_bitstrings,
            p_uniform=prepared.p_uniform,
            methods=["linear", "cubic"],
        )
